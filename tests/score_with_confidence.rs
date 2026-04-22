#![allow(clippy::unwrap_used, clippy::panic)]
//! Integration checks on `RandomCutForest::score_with_confidence`.

use anomstream_rs::{ForestBuilder, RcfError};

fn warm_forest() -> anomstream_rs::RandomCutForest<4> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(64)
        .seed(2026)
        .build()
        .unwrap();
    for i in 0..256_u32 {
        let v = f64::from(i) * 0.01;
        f.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    f
}

#[test]
fn ci_contains_point_estimate() {
    let f = warm_forest();
    let ci = f.score_with_confidence(&[5.0, 5.0, 5.0, 5.0]).unwrap();
    let (lo, hi) = ci.ci95();
    let mean = f64::from(ci.score);
    assert!(lo <= mean);
    assert!(hi >= mean);
}

#[test]
fn stderr_non_negative() {
    let f = warm_forest();
    let ci = f.score_with_confidence(&[0.1, 0.2, 0.3, 0.4]).unwrap();
    assert!(ci.stderr >= 0.0);
    assert!(ci.stddev >= 0.0);
    assert_eq!(ci.trees_evaluated, 100);
}

#[test]
fn mean_matches_bare_score_call_within_ulp() {
    // The bare `score` path aggregates via rayon par-fold (different
    // pair-sum order) while `score_with_confidence` sums
    // sequentially — 1 ULP drift is tolerated.
    let f = warm_forest();
    let plain = f64::from(f.score(&[0.5; 4]).unwrap());
    let ci = f.score_with_confidence(&[0.5; 4]).unwrap();
    let drift = (plain - f64::from(ci.score)).abs();
    assert!(drift <= plain.abs() * 1e-12, "drift = {drift}");
}

#[test]
fn nan_input_rejected() {
    let f = warm_forest();
    assert!(matches!(
        f.score_with_confidence(&[f64::NAN, 0.0, 0.0, 0.0])
            .unwrap_err(),
        RcfError::NaNValue
    ));
}

#[test]
fn wider_z_widens_interval() {
    let f = warm_forest();
    let ci = f.score_with_confidence(&[5.0, 5.0, 5.0, 5.0]).unwrap();
    let (lo95, hi95) = ci.ci95();
    let (lo99, hi99) = ci.ci(2.576);
    assert!(lo99 <= lo95);
    assert!(hi99 >= hi95);
}

#[test]
fn empty_forest_returns_error() {
    let f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(8)
        .seed(1)
        .build()
        .unwrap();
    assert!(matches!(
        f.score_with_confidence(&[0.0; 4]).unwrap_err(),
        RcfError::EmptyForest
    ));
}
