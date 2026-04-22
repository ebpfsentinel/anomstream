#![allow(clippy::unwrap_used, clippy::panic)]
//! Warm reload: checkpoint the detector to disk, then resume across
//! restarts without paying the cold-start warmup twice.
//!
//! First invocation trains on a burst of noisy points and saves the
//! thresholded detector to `/tmp/anomstream-rs-warm-reload.bin`. Later
//! invocations reload that file and continue training, preserving
//! the EMA stats and reservoir built up on prior runs. An outlier at
//! the end of every run confirms the adaptive threshold is hot.
//!
//! ```sh
//! cargo run --example warm_reload   # first run: fresh + save
//! cargo run --example warm_reload   # later runs: resume + save
//! ```

use std::path::PathBuf;

use anomstream_rs::{RcfError, ThresholdedForest, ThresholdedForestBuilder};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn snapshot_path() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push("anomstream-rs-warm-reload.bin");
    p
}

fn build_fresh() -> Result<ThresholdedForest<4>, RcfError> {
    ThresholdedForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .z_factor(3.0)
        .min_observations(32)
        .min_threshold(0.1)
        .initial_accept_fraction(0.125)
        .seed(42)
        .build()
}

fn main() -> Result<(), RcfError> {
    let path = snapshot_path();
    let mut detector = match ThresholdedForest::<4>::from_path(&path) {
        Ok(loaded) => {
            println!("resumed snapshot {}", path.display());
            loaded
        }
        Err(e) => {
            println!("no usable snapshot ({e}) — starting fresh");
            build_fresh()?
        }
    };
    let obs_before = detector.stats().observations();
    println!("before training: observations = {obs_before}");

    let mut rng = ChaCha8Rng::seed_from_u64(
        obs_before.wrapping_add(1), // slightly different rolls each run
    );
    for _ in 0..256 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        detector.process(p)?;
    }
    let obs_after = detector.stats().observations();
    println!(
        "after training:  observations = {obs_after} (+{})",
        obs_after - obs_before
    );
    println!("current threshold = {:.4}", detector.current_threshold());

    let verdict = detector.process([50.0_f64, 50.0, 50.0, 50.0])?;
    println!();
    println!("outlier probe:");
    println!("  ready      = {}", verdict.ready());
    println!("  is_anomaly = {}", verdict.is_anomaly());
    println!("  score      = {}", verdict.score());
    println!("  threshold  = {:.4}", verdict.threshold());
    println!("  grade      = {:.4}", verdict.grade());

    detector.to_path(&path)?;
    println!();
    println!("checkpoint saved to {}", path.display());
    Ok(())
}
