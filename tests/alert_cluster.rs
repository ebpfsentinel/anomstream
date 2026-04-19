#![allow(clippy::unwrap_used, clippy::panic)]
//! End-to-end [`AlertClusterer`] behaviour on a live RCF stream.

#![cfg(all(feature = "serde", feature = "postcard", feature = "serde_json"))]

use rcf_rs::audit::{AlertContext, AlertRecord};
use rcf_rs::{AlertClusterer, ClusterDecision, ForestBuilder};

fn warm_forest() -> rcf_rs::RandomCutForest<4> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(64)
        .seed(101)
        .build()
        .unwrap();
    for i in 0..256_u32 {
        let v = f64::from(i) * 0.01;
        f.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    f
}

#[test]
fn identical_alerts_fold_into_one_cluster() {
    let f = warm_forest();
    let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 60_000).unwrap();
    let ctx = AlertContext::<String>::for_tenant("t1".into(), 1_000);
    for _ in 0..20 {
        let rec = AlertRecord::from_forest(&f, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
        c.observe(rec);
    }
    assert_eq!(c.len(), 1);
    assert_eq!(c.clusters()[0].count, 20);
}

#[test]
fn disparate_alerts_open_separate_clusters() {
    let f = warm_forest();
    let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.99, 60_000).unwrap();
    let ctx = AlertContext::<String>::for_tenant("t".into(), 1_000);
    // Attribution spikes on dim 0 vs dim 3 → different direction →
    // clusters split.
    let r0 = AlertRecord::from_forest(&f, &[100.0, 0.0, 0.0, 0.0], &ctx).unwrap();
    let r3 = AlertRecord::from_forest(&f, &[0.0, 0.0, 0.0, 100.0], &ctx).unwrap();
    c.observe(r0);
    c.observe(r3);
    assert_eq!(c.len(), 2);
}

#[test]
fn cluster_prune_drops_stale() {
    let f = warm_forest();
    let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 500).unwrap();
    let ctx = AlertContext::<String>::untenanted(1_000);
    let rec = AlertRecord::from_forest(&f, &[5.0; 4], &ctx).unwrap();
    c.observe(rec);
    c.prune_stale(10_000); // 10 s later, window 500 ms
    assert!(c.is_empty());
}

#[test]
fn observe_prunes_before_matching() {
    let f = warm_forest();
    let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 500).unwrap();
    let ctx_old = AlertContext::<String>::untenanted(1_000);
    let r_old = AlertRecord::from_forest(&f, &[5.0; 4], &ctx_old).unwrap();
    c.observe(r_old);

    let ctx_new = AlertContext::<String>::untenanted(5_000);
    let r_new = AlertRecord::from_forest(&f, &[5.0; 4], &ctx_new).unwrap();
    let d = c.observe(r_new);
    assert!(matches!(d, ClusterDecision::NewCluster(_)));
    assert_eq!(c.len(), 1);
}

#[test]
fn cluster_tracks_multiple_tenants() {
    let f = warm_forest();
    let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.5, 60_000).unwrap();
    for tenant in &["a", "b", "c"] {
        let ctx = AlertContext::<String>::for_tenant((*tenant).to_string(), 1_000);
        let rec = AlertRecord::from_forest(&f, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
        c.observe(rec);
    }
    assert_eq!(c.len(), 1);
    assert_eq!(c.clusters()[0].contributing_tenants.len(), 3);
}
