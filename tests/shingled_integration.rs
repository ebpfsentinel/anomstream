#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_precision_loss)]
//! Integration coverage for `ShingledForest` — the contextual-
//! temporal anomaly fix for the NAB `rogue_agent_key_hold` = 0.145
//! / `SWaT` = 0.282 failure modes documented in
//! `docs/performance.md`. Isolation depth on a raw scalar stream
//! cannot catch dwell / drop / frequency-shift anomalies because
//! the scalar value itself stays in the baseline's range; the
//! shingle does because the subsequence sits far from every other
//! subsequence in the `D`-dim shingle space.

#![cfg(all(feature = "std", feature = "parallel"))]

use rcf_rs::ShingledForestBuilder;

#[test]
fn dwell_anomaly_shingle_scores_materially_above_baseline() {
    const SHINGLE: usize = 16;
    let mut forest = ShingledForestBuilder::<SHINGLE>::new()
        .num_trees(100)
        .sample_size(128)
        .seed(2026)
        .build()
        .unwrap();

    // Warm on a clean periodic baseline — only sine shingles land
    // in the reservoir.
    let mut t = 0.0_f64;
    for _ in 0..512 {
        let v = (t * 0.5).sin();
        forest.update_scalar(v).unwrap();
        t += 1.0;
    }

    // Construct the two contrastive shingles manually so the forest
    // stays frozen between scorings (avoids folding the dwell into
    // the reservoir). Exposed through the bare-forest accessor.
    let mut baseline_shingle = [0.0_f64; SHINGLE];
    for (i, slot) in baseline_shingle.iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let ti = t + i as f64;
        *slot = (ti * 0.5).sin();
    }
    let dwell_shingle = [0.95_f64; SHINGLE]; // `rogue_agent_key_hold` shape

    let baseline: f64 = forest.forest().score(&baseline_shingle).unwrap().into();
    let dwell: f64 = forest.forest().score(&dwell_shingle).unwrap().into();
    assert!(
        dwell > baseline * 1.5,
        "dwell score {dwell} did not exceed baseline {baseline} by 1.5×"
    );
}

#[test]
fn scalar_outlier_scores_above_baseline() {
    const SHINGLE: usize = 32;
    let mut forest = ShingledForestBuilder::<SHINGLE>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(7)
        .build()
        .unwrap();

    let mut t = 0.0_f64;
    for _ in 0..1_024 {
        let v = (t * 0.3).sin();
        forest.update_scalar(v).unwrap();
        t += 1.0;
    }

    let baseline: f64 = forest.score_scalar(0.1).unwrap().into();
    let outlier: f64 = forest.score_scalar(50.0).unwrap().into();
    assert!(
        outlier > baseline * 3.0,
        "outlier score {outlier} did not exceed baseline {baseline} by 3×"
    );
}

#[test]
fn stateless_codisp_scalar_survives_many_probes_without_drift() {
    const SHINGLE: usize = 8;
    let mut forest = ShingledForestBuilder::<SHINGLE>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(2026)
        .build()
        .unwrap();
    for i in 0..200 {
        let v = (f64::from(i) * 0.1).sin();
        forest.update_scalar(v).unwrap();
    }
    // Drift regression: the stateless codisp path must return the
    // exact same score after N repeats — no reservoir mutation.
    let first: f64 = forest.score_codisp_stateless_scalar(2.0).unwrap().into();
    for _ in 0..2_000 {
        let _ = forest.score_codisp_stateless_scalar(2.0).unwrap();
    }
    let last: f64 = forest.score_codisp_stateless_scalar(2.0).unwrap().into();
    assert!((first - last).abs() < 1.0e-12);
}

#[test]
fn attribution_on_spike_produces_non_zero_divector() {
    const SHINGLE: usize = 8;
    let mut forest = ShingledForestBuilder::<SHINGLE>::new()
        .num_trees(100)
        .sample_size(128)
        .seed(11)
        .build()
        .unwrap();
    // Warm on a varying signal so the bounding box is non-degenerate
    // (a zero-range box produces zero cut probabilities on every
    // dim and collapses attribution to zero).
    let mut t = 0.0_f64;
    for _ in 0..512 {
        let v = (t * 0.3).sin() * 0.1;
        forest.update_scalar(v).unwrap();
        t += 1.0;
    }
    let di = forest.attribution_scalar(100.0).unwrap();
    // Contract: attribution on a real outlier should produce a
    // non-zero DiVector (some lag dim picked up the divergence).
    // Which specific lag index wins is a random-cut implementation
    // detail and not pinned here.
    let total: f64 = (0..SHINGLE).map(|d| di.high()[d] + di.low()[d]).sum();
    assert!(total > 0.0, "attribution DiVector collapsed to zero");
}
