#![allow(clippy::unwrap_used, clippy::panic)]
//! End-to-end behaviour of the adaptive threshold layer.
//!
//! Asserts four properties that callers rely on:
//!
//! 1. Warmup: the detector never fires before `min_observations`.
//! 2. Steady state: a noisy baseline does not fire on its own noise
//!    once the threshold has adapted.
//! 3. Outlier detection: a point far outside the baseline fires with
//!    a high grade.
//! 4. Threshold drift: when the score distribution shifts upward, the
//!    adaptive threshold tracks it so the detector does not produce
//!    a sustained alarm.

#![allow(clippy::cast_precision_loss)] // Tests cast small bounded counters.

use anomstream_rs::ThresholdedForestBuilder;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn noisy_point(rng: &mut ChaCha8Rng) -> [f64; 4] {
    [
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
    ]
}

#[test]
fn warmup_never_fires_before_min_observations() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(64)
        .seed(1)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..40 {
        let v = d.process(noisy_point(&mut rng)).unwrap();
        assert!(!v.is_anomaly(), "fired during warmup: grade={}", v.grade());
        assert!(!v.ready());
    }
}

#[test]
fn noisy_baseline_does_not_fire_in_steady_state() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .z_factor(3.0)
        .min_observations(64)
        .min_threshold(0.5)
        .seed(2)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // Warm the detector on 512 noisy baseline points.
    for _ in 0..512 {
        d.process(noisy_point(&mut rng)).unwrap();
    }

    // Probe the next 200 baseline points. Very few should fire.
    let mut fires = 0_u32;
    for _ in 0..200 {
        let v = d.process(noisy_point(&mut rng)).unwrap();
        if v.is_anomaly() {
            fires += 1;
        }
    }
    // z=3 on a Gaussian-ish stream: far less than 5% false-positive.
    assert!(fires < 20, "baseline fired too often: {fires}/200");
}

#[test]
fn outlier_fires_with_non_zero_grade() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .z_factor(3.0)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(3)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    for _ in 0..512 {
        d.process(noisy_point(&mut rng)).unwrap();
    }
    let outlier = d.process([50.0, 50.0, 50.0, 50.0]).unwrap();
    assert!(outlier.ready());
    assert!(
        outlier.is_anomaly(),
        "outlier did not fire: score={} threshold={} grade={}",
        outlier.score(),
        outlier.threshold(),
        outlier.grade(),
    );
    assert!(outlier.grade() > 0.0);
    assert!(f64::from(outlier.score()) > outlier.threshold());
}

#[test]
fn score_only_does_not_train() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(16)
        .seed(4)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    for _ in 0..128 {
        d.process(noisy_point(&mut rng)).unwrap();
    }
    let obs_before = d.stats().observations();
    let updates_before = d.forest().updates_seen();

    for _ in 0..50 {
        let _ = d.score_only(&[10.0, 10.0, 10.0, 10.0]).unwrap();
    }

    assert_eq!(d.stats().observations(), obs_before);
    assert_eq!(d.forest().updates_seen(), updates_before);
}

#[test]
fn threshold_adapts_to_shifted_distribution() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .z_factor(3.0)
        .min_observations(64)
        .min_threshold(0.1)
        .score_decay(0.05)
        .seed(5)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);

    // Phase 1: tight cluster near origin.
    for _ in 0..512 {
        d.process(noisy_point(&mut rng)).unwrap();
    }
    let threshold_phase1 = d.current_threshold();

    // Phase 2: distribution shifts — now a wider cluster with more
    // variance. Some points will fire initially but the threshold
    // should climb to accommodate the new baseline.
    let mut fires_phase2 = 0_u32;
    for _ in 0..1024 {
        let p = [
            rng.random::<f64>() * 5.0,
            rng.random::<f64>() * 5.0,
            rng.random::<f64>() * 5.0,
            rng.random::<f64>() * 5.0,
        ];
        let v = d.process(p).unwrap();
        if v.is_anomaly() {
            fires_phase2 += 1;
        }
    }
    let threshold_phase2 = d.current_threshold();

    assert!(
        threshold_phase2 > threshold_phase1,
        "threshold did not rise: before={threshold_phase1} after={threshold_phase2}"
    );
    // After the drift has been absorbed, the tail of phase 2 should
    // fire much less often than the head. Not worth pinning a tight
    // number — just confirm we did not sustain an alarm for the full
    // run (which would indicate broken adaptation).
    assert!(
        fires_phase2 < 300,
        "detector never recovered from drift: {fires_phase2}/1024 fired"
    );
}
