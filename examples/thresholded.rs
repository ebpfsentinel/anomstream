#![allow(clippy::unwrap_used, clippy::panic)]
//! Streaming anomaly detection with an adaptive threshold.
//!
//! Runs a tight noisy baseline through the detector, then two plants
//! a far outlier. The adaptive threshold rises during the baseline
//! warmup, the outlier fires with a graded verdict. Run with
//! `cargo run --example thresholded`.

use anomstream_rs::{RcfError, ThresholdedForestBuilder};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), RcfError> {
    let mut detector = ThresholdedForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .z_factor(3.0)
        .min_observations(64)
        .min_threshold(0.1)
        .seed(42)
        .build()?;

    let mut rng = ChaCha8Rng::seed_from_u64(7);

    // Warm the detector on 512 noisy baseline points.
    let mut warmup_fires = 0_u32;
    for _ in 0..512 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        let v = detector.process(p)?;
        if v.is_anomaly() {
            warmup_fires += 1;
        }
    }

    println!(
        "after warmup: threshold = {:.4}, baseline fires = {warmup_fires}/512",
        detector.current_threshold()
    );
    println!(
        "score stats:  mean = {:.4}, stddev = {:.4}, observations = {}",
        detector.stats().mean(),
        detector.stats().stddev(),
        detector.stats().observations(),
    );

    // Plant a far outlier. Should fire with grade > 0.
    let outlier = detector.process([50.0, 50.0, 50.0, 50.0])?;
    println!();
    println!("outlier verdict:");
    println!("  ready      = {}", outlier.ready());
    println!("  is_anomaly = {}", outlier.is_anomaly());
    println!("  score      = {}", outlier.score());
    println!("  threshold  = {:.4}", outlier.threshold());
    println!("  grade      = {:.4}", outlier.grade());

    // `score_only` re-evaluates without training. Useful for
    // forensic replay.
    let replay = detector.score_only(&[50.0, 50.0, 50.0, 50.0])?;
    println!();
    println!("replay (score_only, no training side effect):");
    println!("  grade      = {:.4}", replay.grade());

    Ok(())
}
