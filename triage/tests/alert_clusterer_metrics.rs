//! Cross-crate integration — verify `AlertClusterer` (triage)
//! drives the `MetricsSink` contract (core) through the
//! `observe` / `prune` lifecycle.

#![cfg(all(feature = "postcard", feature = "serde_json"))]
#![allow(clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;

use anomstream_core::ForestBuilder;
use anomstream_core::metrics::{TestSink, names};
use anomstream_triage::audit::{AlertContext, AlertRecord};
use anomstream_triage::{AlertClusterer, ClusterDecision};

#[test]
fn alert_clusterer_sink_records_observe_new_joined() {
    let sink = Arc::new(TestSink::new());
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(17)
        .build()
        .unwrap();
    for i in 0..32_u32 {
        let v = f64::from(i) * 0.01;
        forest.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    let mut clusterer: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 60_000)
        .unwrap()
        .with_metrics_sink(sink.clone());
    let ctx = AlertContext::<String>::for_tenant("t1".into(), 1000);
    let r1 = AlertRecord::from_forest(&forest, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
    let r2 = AlertRecord::from_forest(&forest, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
    let d1 = clusterer.observe(r1);
    let d2 = clusterer.observe(r2);
    assert!(matches!(d1, ClusterDecision::NewCluster(_)));
    assert!(matches!(d2, ClusterDecision::Joined(_)));
    assert_eq!(sink.counter(names::ALERTS_OBSERVED_TOTAL), 2);
    assert_eq!(sink.counter(names::ALERT_CLUSTERS_NEW_TOTAL), 1);
    assert_eq!(sink.counter(names::ALERT_CLUSTERS_JOINED_TOTAL), 1);
    assert_eq!(sink.gauge(names::ALERT_CLUSTERS_ACTIVE), Some(1.0));
}
