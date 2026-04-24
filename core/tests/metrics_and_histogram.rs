#![allow(clippy::unwrap_used, clippy::panic)]
//! End-to-end tests for the observability surface:
//!
//! - [`anomstream_core::MetricsSink`] receives the documented events from
//!   `RandomCutForest`, `ThresholdedForest`, `TenantForestPool`,
//!   `MetaDriftDetector`.
//! - [`anomstream_core::ScoreHistogram`] bins a streamed score distribution
//!   into well-formed buckets.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use std::sync::Arc;

use anomstream_core::{
    CusumConfig, ForestBuilder, MetaDriftDetector, ScoreHistogram, TenantForestPool,
    ThresholdedForestBuilder,
    metrics::{TestSink, names},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn noisy(rng: &mut ChaCha8Rng) -> [f64; 2] {
    [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1]
}

#[test]
fn forest_sink_records_updates_scores_deletes() {
    let sink = Arc::new(TestSink::new());
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(1)
        .build()
        .unwrap()
        .with_metrics_sink(sink.clone());

    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..20 {
        f.update(noisy(&mut rng)).unwrap();
    }
    for _ in 0..5 {
        let _ = f.score(&[0.05, 0.05]).unwrap();
    }
    let idx = f.update_indexed([9.0, 9.0]).unwrap();
    assert!(f.delete(idx).unwrap());

    assert_eq!(sink.counter(names::UPDATES_TOTAL), 21);
    assert_eq!(sink.counter(names::DELETES_TOTAL), 1);
    assert_eq!(sink.histogram(names::SCORE_OBSERVATION).len(), 5);
    assert_eq!(sink.gauge(names::FOREST_TREES), Some(50.0));
}

#[test]
fn thresholded_sink_records_process_grade_threshold() {
    let sink = Arc::new(TestSink::new());
    let mut d = ThresholdedForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .min_observations(4)
        .min_threshold(0.1)
        .seed(2)
        .build()
        .unwrap()
        .with_metrics_sink(sink.clone());

    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for _ in 0..32 {
        d.process(noisy(&mut rng)).unwrap();
    }
    let _ = d.process([50.0, 50.0]).unwrap();

    let processed = sink.counter(names::PROCESS_TOTAL);
    assert_eq!(processed, 33);
    assert!(sink.counter(names::ANOMALIES_FIRED_TOTAL) >= 1);
    assert_eq!(sink.histogram(names::GRADE_OBSERVATION).len(), 33);
    assert!(
        sink.gauge(names::THRESHOLD_CURRENT).is_some(),
        "threshold gauge should have been set at least once",
    );
}

#[test]
fn pool_sink_records_evictions_and_resident_gauge() {
    let sink = Arc::new(TestSink::new());
    let mut pool: TenantForestPool<u32, 2> = TenantForestPool::new(2, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(3)
            .build()
    })
    .unwrap()
    .with_metrics_sink(sink.clone());

    let mut rng = ChaCha8Rng::seed_from_u64(3);
    pool.process(&1, noisy(&mut rng)).unwrap();
    pool.process(&2, noisy(&mut rng)).unwrap();
    pool.process(&3, noisy(&mut rng)).unwrap(); // Triggers LRU eviction
    assert_eq!(sink.counter(names::TENANT_EVICTIONS_TOTAL), 1);
    assert_eq!(sink.gauge(names::TENANTS_RESIDENT), Some(2.0));
}

#[test]
fn drift_sink_records_cusum_and_fires() {
    let sink = Arc::new(TestSink::new());
    let mut drift = MetaDriftDetector::new(CusumConfig {
        allowance_k: 0.5,
        threshold_h: 3.0,
        min_observations: 8,
        decay: 0.1,
    })
    .unwrap()
    .with_metrics_sink(sink.clone());

    for _ in 0..64 {
        let _ = drift.observe(1.0);
    }
    for _ in 0..200 {
        let v = drift.observe(5.0);
        if v.drift.is_some() {
            break;
        }
    }
    let s_high_obs = sink.histogram(names::DRIFT_S_HIGH);
    assert!(!s_high_obs.is_empty());
    assert!(sink.counter(names::DRIFT_FIRES_TOTAL) >= 1);
}

#[test]
fn score_histogram_bins_streamed_grades() {
    let mut h = ScoreHistogram::with_range(0.0, 1.0).unwrap();
    // Bimodal: 80% near 0.1 (normal), 20% near 0.9 (anomaly).
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    for _ in 0..800 {
        h.record(rng.random::<f64>() * 0.2);
    }
    for _ in 0..200 {
        h.record(0.8 + rng.random::<f64>() * 0.2);
    }
    assert_eq!(h.total(), 1000);
    assert_eq!(h.underflow(), 0);
    // The lower half should hold the 800 normal observations, the
    // upper half the 200 anomaly observations.
    let mid = h.bins().len() / 2;
    let lower: u64 = h.bins()[..mid].iter().sum();
    let upper: u64 = h.bins()[mid..].iter().sum();
    assert!(lower > upper);
    let p50 = h.percentile(0.5).unwrap();
    assert!(
        p50 < 0.3,
        "median of bimodal lower mode should be < 0.3, got {p50}"
    );
}

#[test]
fn histogram_merge_produces_union_distribution() {
    let mut a = ScoreHistogram::with_range(0.0, 1.0).unwrap();
    let mut b = ScoreHistogram::with_range(0.0, 1.0).unwrap();
    for _ in 0..100 {
        a.record(0.1);
    }
    for _ in 0..200 {
        b.record(0.9);
    }
    a.merge(&b).unwrap();
    assert_eq!(a.total(), 300);
}

#[test]
fn forest_sink_records_attribution_and_nan_rejection() {
    let sink = Arc::new(TestSink::new());
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(11)
        .build()
        .unwrap()
        .with_metrics_sink(sink.clone());

    let mut rng = ChaCha8Rng::seed_from_u64(11);
    for _ in 0..20 {
        f.update(noisy(&mut rng)).unwrap();
    }
    for _ in 0..3 {
        let _ = f.attribution(&[0.05, 0.05]).unwrap();
    }
    // NaN must go to REJECTED_NAN_TOTAL without panicking.
    assert!(f.score(&[f64::NAN, 0.0]).is_err());
    assert!(f.update([0.0, f64::INFINITY]).is_err());

    assert_eq!(sink.counter(names::ATTRIBUTION_TOTAL), 3);
    assert_eq!(sink.counter(names::REJECTED_NAN_TOTAL), 2);
}

#[test]
fn forest_sink_records_early_term_stops() {
    use anomstream_core::EarlyTermConfig;
    let sink = Arc::new(TestSink::new());
    let mut f = ForestBuilder::<2>::new()
        .num_trees(100)
        .sample_size(32)
        .seed(12)
        .build()
        .unwrap()
        .with_metrics_sink(sink.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    for _ in 0..80 {
        f.update(noisy(&mut rng)).unwrap();
    }
    // Loose threshold → short-circuits routinely.
    let loose = EarlyTermConfig {
        min_trees: 16,
        confidence_threshold: 0.5,
    };
    for _ in 0..10 {
        let _ = f.score_early_term(&[0.05, 0.05], loose).unwrap();
    }
    assert!(
        sink.counter(names::EARLY_TERM_STOPPED_TOTAL) >= 1,
        "loose early-term must have short-circuited at least once",
    );
    assert!(!sink.histogram(names::EARLY_TERM_TREES).is_empty());
}

#[test]
fn trcf_sink_exposes_ema_and_observation_gauges() {
    let sink = Arc::new(TestSink::new());
    let mut d = ThresholdedForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .min_observations(4)
        .seed(13)
        .build()
        .unwrap()
        .with_metrics_sink(sink.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(13);
    for _ in 0..20 {
        d.process(noisy(&mut rng)).unwrap();
    }
    assert!(sink.gauge(names::EMA_MEAN).is_some());
    assert!(sink.gauge(names::EMA_STDDEV).is_some());
    let obs = sink.gauge(names::OBSERVATIONS_SEEN).unwrap();
    assert!(obs >= 1.0, "expected observations gauge >= 1, got {obs}");
}

#[test]
fn pool_sink_records_idle_eviction_counter() {
    let sink = Arc::new(TestSink::new());
    let mut pool: TenantForestPool<u32, 2> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(16)
            .build()
    })
    .unwrap()
    .with_metrics_sink(sink.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(16);
    pool.process(&1, noisy(&mut rng)).unwrap();
    pool.process(&2, noisy(&mut rng)).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let evicted = pool.evict_idle(std::time::Duration::from_millis(10));
    assert_eq!(evicted.len(), 2);
    assert_eq!(sink.counter(names::TENANT_IDLE_EVICTIONS_TOTAL), 2);
    // Aggregate evictions counter also includes the two TTL fires.
    assert!(sink.counter(names::TENANT_EVICTIONS_TOTAL) >= 2);
}

#[test]
fn pool_sink_records_tenant_created_and_capacity() {
    let sink = Arc::new(TestSink::new());
    let mut pool: TenantForestPool<u32, 2> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(14)
            .build()
    })
    .unwrap()
    .with_metrics_sink(sink.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(14);
    pool.process(&10, noisy(&mut rng)).unwrap();
    pool.process(&11, noisy(&mut rng)).unwrap();
    pool.process(&10, noisy(&mut rng)).unwrap(); // repeat — no new tenant

    assert_eq!(sink.counter(names::TENANT_CREATED_TOTAL), 2);
    assert_eq!(sink.gauge(names::TENANT_CAPACITY), Some(4.0));
}

#[test]
fn bootstrap_sink_records_points_and_skips() {
    let sink = Arc::new(TestSink::new());
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(15)
        .build()
        .unwrap()
        .with_metrics_sink(sink.clone());
    let points = vec![
        [0.1, 0.2],
        [f64::NAN, 0.0], // skip
        [0.3, 0.4],
        [0.0, f64::INFINITY], // skip
        [0.5, 0.6],
    ];
    let report = f.bootstrap(points).unwrap();
    assert_eq!(report.points_ingested, 3);
    assert_eq!(report.points_skipped, 2);
    assert_eq!(sink.counter(names::BOOTSTRAP_POINTS_TOTAL), 3);
    assert_eq!(sink.counter(names::BOOTSTRAP_SKIPPED_TOTAL), 2);
}

#[test]
fn drift_sink_splits_upward_and_downward() {
    let sink = Arc::new(TestSink::new());
    let mut drift = MetaDriftDetector::new(CusumConfig {
        allowance_k: 0.5,
        threshold_h: 3.0,
        min_observations: 8,
        decay: 0.1,
    })
    .unwrap()
    .with_metrics_sink(sink.clone());

    // Warm the EMA, then drive score upward.
    for _ in 0..64 {
        let _ = drift.observe(1.0);
    }
    for _ in 0..200 {
        if drift.observe(5.0).drift.is_some() {
            break;
        }
    }
    // Reset accumulators, drive downward.
    drift.reset();
    for _ in 0..200 {
        if drift.observe(-5.0).drift.is_some() {
            break;
        }
    }
    assert!(sink.counter(names::DRIFT_UP_TOTAL) >= 1);
    assert!(sink.counter(names::DRIFT_DOWN_TOTAL) >= 1);
    let total = sink.counter(names::DRIFT_UP_TOTAL) + sink.counter(names::DRIFT_DOWN_TOTAL);
    assert_eq!(sink.counter(names::DRIFT_FIRES_TOTAL), total);
}
