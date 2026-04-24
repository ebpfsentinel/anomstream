//! Smoke test — verify the `anomstream` facade re-exports from
//! all three workspace members are reachable under the
//! `core + triage + hotpath` feature set and that types compose
//! the way a real consumer would use them.
//!
//! Keeps the full all-features path green at the facade level so
//! breakages in member re-export surfaces surface here rather
//! than deep in a downstream consumer lockfile.

#![cfg(all(
    feature = "core",
    feature = "triage",
    feature = "hotpath",
    feature = "std",
    feature = "serde",
    feature = "postcard"
))]
#![allow(clippy::unwrap_used, clippy::panic)]

use anomstream::hot_path::{PrefixRateCap, UpdateSampler};
use anomstream::{
    AlertClusterer, AlertContext, AlertRecord, ClusterDecision, ForestBuilder, ForestSnapshot,
    MetricsSink, NoopSink, PerFeatureCusum, PerFeatureCusumConfig, PlattCalibrator, PlattFitConfig,
    RandomCutForest, Severity, SeverityBands,
};

/// Core re-exports resolve: build a forest, confirm `ForestSnapshot`
/// trait bounds survive through the facade.
#[test]
fn core_forest_roundtrip_via_facade() {
    let forest: RandomCutForest<4> = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(2026)
        .build()
        .unwrap();
    assert_eq!(forest.snapshot_num_trees(), 50);
    assert_eq!(forest.snapshot_dimension(), 4);
    assert_eq!(forest.snapshot_updates_seen(), 0);
}

/// Per-feature detector + severity classifier (core cross-cuts).
#[test]
fn core_per_feature_cusum_plus_severity_via_facade() {
    let mut det = PerFeatureCusum::<2>::new(PerFeatureCusumConfig {
        slack: 0.5,
        threshold: 5.0,
    });
    let _ = det.observe(&[100.0, 200.0]);
    for _ in 0..20 {
        let _ = det.observe(&[105.0, 200.0]);
    }
    let bands = SeverityBands::default();
    assert_eq!(bands.classify(0.5), Severity::Normal);
}

/// Triage types reachable through the facade root (glob import).
#[test]
fn triage_clusterer_plus_calibrator_plus_audit_via_facade() {
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(42)
        .build()
        .unwrap();
    for i in 0..32 {
        let v = f64::from(i) * 0.01;
        forest.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    let ctx = AlertContext::<String>::untenanted(1_000);
    let rec: AlertRecord<String, 4> = AlertRecord::from_forest(&forest, &[5.0; 4], &ctx).unwrap();
    let mut clusterer: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 60_000).unwrap();
    let decision = clusterer.observe(rec);
    assert!(matches!(decision, ClusterDecision::NewCluster(_)));

    let calibration = vec![(0.5, false), (4.0, true), (0.6, false), (3.9, true)];
    let cal = PlattCalibrator::fit(&calibration, PlattFitConfig::default()).unwrap();
    let p = cal.calibrate(3.5);
    assert!((0.0..=1.0).contains(&p));
}

/// Hot-path primitives reachable through `anomstream::hot_path::*`.
#[test]
fn hot_path_sampler_plus_rate_cap_via_facade() {
    let sampler = UpdateSampler::new(4);
    // Stride sampler: 1st call true (counter = 0 % 4 == 0), next 3 false.
    assert!(sampler.accept_stride());
    assert!(!sampler.accept_stride());

    let cap = PrefixRateCap::new(10, 1_000);
    assert!(cap.check_and_record(0x1234_5678_u64, 0));
}

/// `MetricsSink` cross-cut: `NoopSink` reachable at root,
/// implements the core trait alias.
#[test]
fn metrics_sink_reachable_via_facade() {
    let sink = NoopSink;
    let _: &dyn MetricsSink = &sink;
}
