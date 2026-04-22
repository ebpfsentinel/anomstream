#![allow(clippy::unwrap_used, clippy::panic)]
//! Two additions in one demo: explicit-point retraction via
//! `update_indexed` / `delete`, and per-dim `feature_scales` to
//! rebalance dims with wildly different dynamic ranges.
//!
//! Scenario: a 3-D feature vector where `rate ∈ [0, 100_000]`,
//! `ratio ∈ [0, 1]`, `entropy ∈ [0, 8]`. Without scales, `rate`
//! dominates every random cut. With `feature_scales = [1e-5, 1, 0.1]`,
//! every dim pulls its weight. Then we demonstrate `delete` removing a
//! freshly-inserted observation by its `point_idx`.
//!
//! Run with `cargo run --example delete_and_scales`.

use anomstream_rs::{RcfError, ThresholdedForestBuilder};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), RcfError> {
    let mut detector = ThresholdedForestBuilder::<3>::new()
        .num_trees(100)
        .sample_size(128)
        .min_observations(32)
        .min_threshold(0.1)
        .feature_scales([1e-5, 1.0, 0.1])
        .seed(2026)
        .build()?;

    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..512 {
        let rate: f64 = rng.random::<f64>() * 100_000.0;
        let ratio: f64 = rng.random::<f64>();
        let entropy: f64 = rng.random::<f64>() * 8.0;
        detector.process([rate, ratio, entropy])?;
    }

    println!("== scores after 512 baseline observations ==");
    let baseline = detector.score_only(&[50_000.0, 0.5, 4.0])?;
    println!("  baseline probe grade = {:.3}", baseline.grade());

    let (idx, verdict) = detector.process_indexed([50_000.0, 0.5, 4.0])?;
    println!();
    println!("== inserted with point_idx = {idx} ==");
    println!("  verdict is_anomaly = {}", verdict.is_anomaly());

    let removed = detector.delete(idx)?;
    println!();
    println!("== retraction via delete(idx) ==");
    println!("  removed_from_any_tree = {removed}");

    // Same probe should score roughly as before the insert —
    // retraction undid the contribution of that single point.
    let after = detector.score_only(&[50_000.0, 0.5, 4.0])?;
    println!("  post-delete probe grade = {:.3}", after.grade());

    Ok(())
}
