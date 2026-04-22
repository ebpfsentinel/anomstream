#![allow(clippy::unwrap_used, clippy::panic)]
//! Bulk scoring on a warmed forest — compare serial
//! `for p in points { f.score(p) }` vs `f.score_many(&points)`.
//! The bulk path parallelises across points on top of rayon's
//! per-tree parallelism, which matters for backfill / replay
//! workloads where latency-per-point is amortised over the batch.
//!
//! Run with `cargo run --example bulk_scoring --release`.

use std::time::Instant;

use anomstream_rs::{ForestBuilder, RcfError};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), RcfError> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()?;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..2048 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        f.update(p)?;
    }

    let probes: Vec<[f64; 4]> = (0..4096)
        .map(|_| {
            [
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
            ]
        })
        .collect();

    // Warmup both paths to fill caches / rayon workers.
    let _ = f.score_many(&probes[..32])?;

    let t0 = Instant::now();
    let mut serial_acc = 0.0_f64;
    for p in &probes {
        serial_acc += f64::from(f.score(p)?);
    }
    let serial = t0.elapsed();

    let t0 = Instant::now();
    let bulk = f.score_many(&probes)?;
    let bulk_elapsed = t0.elapsed();
    let bulk_acc: f64 = bulk.iter().map(|s| f64::from(*s)).sum();

    println!("== 4096 probes, 100-tree forest ==");
    println!("  serial for-loop : {serial:?}  sum_score={serial_acc:.4}");
    println!("  score_many      : {bulk_elapsed:?}  sum_score={bulk_acc:.4}");
    let speedup = serial.as_secs_f64() / bulk_elapsed.as_secs_f64();
    println!("  speedup         : {speedup:.2}×");

    Ok(())
}
