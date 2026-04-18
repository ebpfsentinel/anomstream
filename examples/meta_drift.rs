//! CUSUM meta-drift chained onto a thresholded forest's anomaly
//! score stream.
//!
//! Runs three phases:
//!
//! 1. Baseline — tight noisy cluster around the origin. No CUSUM fire.
//! 2. Drift  — distribution shifts wider. CUSUM fires Upward on the
//!    sustained score-mean climb, *before* individual points would
//!    trigger the TRCF 3σ gate.
//! 3. Recovery — we reset the CUSUM, feed baseline again, and show
//!    the detector is ready to catch the next shift cleanly.
//!
//! Run with `cargo run --example meta_drift`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{CusumConfig, DriftKind, MetaDriftDetector, RcfError, ThresholdedForestBuilder};

fn tight(rng: &mut ChaCha8Rng) -> [f64; 4] {
    [
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
    ]
}

fn wider(rng: &mut ChaCha8Rng) -> [f64; 4] {
    [
        rng.random::<f64>() * 0.5 + 0.2,
        rng.random::<f64>() * 0.5 + 0.2,
        rng.random::<f64>() * 0.5 + 0.2,
        rng.random::<f64>() * 0.5 + 0.2,
    ]
}

fn main() -> Result<(), RcfError> {
    let mut forest = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(2026)
        .build()?;
    // `threshold_h = 8.0` gives a conservative fire rate on a
    // noisy baseline; the sustained drift in phase 2 is still well
    // above it.
    let mut meta = MetaDriftDetector::new(CusumConfig {
        allowance_k: 0.5,
        threshold_h: 8.0,
        min_observations: 32,
        decay: 0.05,
    })?;
    let mut rng = ChaCha8Rng::seed_from_u64(7);

    // Warm the forest WITHOUT attaching CUSUM — a real agent would
    // either bootstrap from a TSDB or resume from a snapshot, so the
    // meta-drift detector never sees the cold-start spike train.
    for _ in 0..512 {
        forest.process(tight(&mut rng))?;
    }

    println!("=== phase 1: baseline (1024 windows) ===");
    let mut baseline_fires = 0_u32;
    for _ in 0..1024 {
        let verdict = forest.process(tight(&mut rng))?;
        let drift = meta.observe(f64::from(verdict.score()));
        if drift.drift.is_some() {
            baseline_fires += 1;
        }
    }
    println!(
        "  CUSUM fires       : {baseline_fires}\n  EMA mean / stddev : {:.4} / {:.4}",
        meta.stats().mean(),
        meta.stats().stddev(),
    );
    println!(
        "  s_high / s_low    : {:.4} / {:.4}",
        meta.s_high(),
        meta.s_low()
    );

    println!();
    println!("=== phase 2: distribution shift (up to 512 windows) ===");
    let mut fire_window: Option<usize> = None;
    for i in 0..512 {
        let verdict = forest.process(wider(&mut rng))?;
        let drift = meta.observe(f64::from(verdict.score()));
        if matches!(drift.drift, Some(DriftKind::Upward)) {
            fire_window = Some(i);
            println!("  CUSUM fired Upward after {n} shifted windows", n = i + 1,);
            println!(
                "  s_high = {:.4}, threshold = {:.4}",
                drift.s_high, drift.threshold
            );
            println!(
                "  score = {:.4}, reference μ = {:.4}, σ = {:.4}",
                f64::from(verdict.score()),
                drift.mean,
                drift.stddev,
            );
            break;
        }
    }
    if fire_window.is_none() {
        println!("  CUSUM did not fire in phase 2 (drift may be too subtle for this config)");
    }

    println!();
    println!("=== phase 3: reset + recovery (feed baseline again) ===");
    meta.reset();
    println!(
        "  after reset: s_high = {:.4}, s_low = {:.4}",
        meta.s_high(),
        meta.s_low()
    );
    let mut recovery_fires = 0_u32;
    for _ in 0..256 {
        let verdict = forest.process(tight(&mut rng))?;
        let drift = meta.observe(f64::from(verdict.score()));
        if drift.drift.is_some() {
            recovery_fires += 1;
        }
    }
    println!("  CUSUM fires during recovery: {recovery_fires}");

    Ok(())
}
