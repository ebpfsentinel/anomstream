#![allow(clippy::unwrap_used, clippy::panic)]
//! Timestamp + retention: stream 1000 observations tagged with
//! unix-epoch-like `u64` timestamps, then prune every point older
//! than a cutoff. GDPR / NIS2 / forensic-retention pattern.
//!
//! Run with `cargo run --example retention`.

use anomstream_rs::{RcfError, ThresholdedForestBuilder};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), RcfError> {
    let mut detector = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(16)
        .min_threshold(0.1)
        .seed(2026)
        .build()?;

    // Simulated epoch-ms timestamps: one observation per "minute"
    // starting at 1_700_000_000_000 (Nov 2023).
    let base_ts: u64 = 1_700_000_000_000;
    let minute_ms: u64 = 60_000;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    for i in 0_u64..1024 {
        let ts = base_ts + i * minute_ms;
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        detector.process_at(p, ts)?;
    }

    let forest = detector.forest();
    println!("== after 1024 observations ==");
    println!("  tracked_timestamps = {}", forest.tracked_timestamps());
    println!("  oldest_timestamp   = {:?}", forest.oldest_timestamp());
    println!("  newest_timestamp   = {:?}", forest.newest_timestamp());

    // Retention cutoff: drop everything older than "now − 12 h".
    let cutoff = base_ts + 1024 * minute_ms - 12 * 60 * minute_ms;
    let removed = detector.delete_before(cutoff)?;
    println!();
    println!("== delete_before(now − 12h) ==");
    println!("  cutoff_ts      = {cutoff}");
    println!("  removed_count  = {removed}");

    let forest = detector.forest();
    println!();
    println!("== after retention ==");
    println!("  tracked_timestamps = {}", forest.tracked_timestamps());
    println!("  oldest_timestamp   = {:?}", forest.oldest_timestamp());
    println!("  newest_timestamp   = {:?}", forest.newest_timestamp());

    Ok(())
}
