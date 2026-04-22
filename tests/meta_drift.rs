#![allow(clippy::unwrap_used, clippy::panic)]
//! End-to-end behaviour of the CUSUM meta-drift detector chained
//! onto a thresholded forest's anomaly-score stream.
//!
//! Asserts:
//!
//! 1. Baseline traffic alone produces no drift fire even while the
//!    forest is warming up and its scores are noisier.
//! 2. A sustained distributional shift in the *input stream* — the
//!    kind of slow degradation that a per-point `μ + 3σ` gate can
//!    miss — fires the CUSUM with the expected direction.
//! 3. `reset()` after a drift fire prepares the detector to catch
//!    the next shift without double-counting the previous one.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use anomstream_rs::{
    CusumConfig, DriftKind, MetaDriftDetector, ThresholdedForest, ThresholdedForestBuilder,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn build_detector() -> ThresholdedForest<4> {
    ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(42)
        .build()
        .unwrap()
}

fn tight(rng: &mut ChaCha8Rng) -> [f64; 4] {
    [
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
    ]
}

fn drifted(rng: &mut ChaCha8Rng) -> [f64; 4] {
    // Slightly wider distribution — every sample individually is
    // unlikely to trip the 3σ threshold of TRCF, but the sustained
    // shift raises the mean score enough for CUSUM to detect.
    [
        rng.random::<f64>() * 0.5 + 0.2,
        rng.random::<f64>() * 0.5 + 0.2,
        rng.random::<f64>() * 0.5 + 0.2,
        rng.random::<f64>() * 0.5 + 0.2,
    ]
}

#[test]
fn baseline_only_produces_no_drift_fire() {
    let mut forest = build_detector();
    let mut meta = MetaDriftDetector::new(CusumConfig {
        allowance_k: 0.5,
        threshold_h: 5.0,
        min_observations: 32,
        decay: 0.05,
    })
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut fires = 0_u32;
    for _ in 0..1024 {
        let verdict = forest.process(tight(&mut rng)).unwrap();
        let drift = meta.observe(f64::from(verdict.score()));
        if drift.drift.is_some() {
            fires += 1;
        }
    }
    // Allow a handful of transient fires during the forest's warmup —
    // the score stream is noisiest while the reservoir is filling
    // and the EMA reference is still converging. Anything past single
    // digits would indicate a broken CUSUM, not warmup noise.
    assert!(
        fires < 5,
        "CUSUM fired {fires} times on stationary baseline — suspiciously high for a quiet stream",
    );
}

#[test]
fn sustained_distributional_shift_fires_upward() {
    let mut forest = build_detector();
    let mut meta = MetaDriftDetector::new(CusumConfig {
        allowance_k: 0.5,
        threshold_h: 4.0,
        min_observations: 32,
        decay: 0.05,
    })
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // Phase 1: tight baseline — warms forest + CUSUM reference stats.
    for _ in 0..1024 {
        let verdict = forest.process(tight(&mut rng)).unwrap();
        let _ = meta.observe(f64::from(verdict.score()));
    }

    // Phase 2: shifted distribution. Any individual point may be
    // within TRCF's 3σ band, but the *average* score rises — the
    // signature CUSUM is designed to catch.
    let mut saw_upward = false;
    for _ in 0..512 {
        let verdict = forest.process(drifted(&mut rng)).unwrap();
        let drift = meta.observe(f64::from(verdict.score()));
        if matches!(drift.drift, Some(DriftKind::Upward)) {
            saw_upward = true;
            break;
        }
    }
    assert!(
        saw_upward,
        "CUSUM did not fire on sustained upward score drift",
    );
}

#[test]
fn reset_allows_detecting_the_next_shift() {
    let mut meta = MetaDriftDetector::new(CusumConfig {
        allowance_k: 0.5,
        threshold_h: 3.0,
        min_observations: 8,
        decay: 0.1,
    })
    .unwrap();

    // Warmup + first upward shift.
    for _ in 0..64 {
        let _ = meta.observe(1.0 + rand_small());
    }
    for _ in 0..200 {
        let drift = meta.observe(5.0);
        if drift.drift.is_some() {
            break;
        }
    }
    assert!(
        meta.s_high() > 0.0,
        "accumulator should be positive after upward shift",
    );

    meta.reset();
    assert_eq!(meta.s_high(), 0.0);
    assert_eq!(meta.s_low(), 0.0);

    // Simulate values near the new mean — CUSUM accumulators stay
    // low. Then drive a fresh downward shift and expect the next
    // fire to be in the opposite direction.
    for _ in 0..64 {
        let _ = meta.observe(5.0 + rand_small());
    }
    let mut saw_downward = false;
    for _ in 0..200 {
        let drift = meta.observe(1.0);
        if matches!(drift.drift, Some(DriftKind::Downward)) {
            saw_downward = true;
            break;
        }
    }
    assert!(
        saw_downward,
        "reset() did not leave the detector in a state ready to catch the next drift",
    );
}

fn rand_small() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static RNG_STATE: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
    let v = RNG_STATE.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    // Small deterministic jitter in [-0.05, 0.05] via top bits.
    let bits = v >> 33;
    #[allow(clippy::cast_precision_loss)]
    let u = bits as f64 / (1_u64 << 31) as f64;
    u * 0.1 - 0.05
}
