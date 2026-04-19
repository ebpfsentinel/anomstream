#![allow(clippy::unwrap_used, clippy::panic)]
//! SOC alert dedup — cluster near-duplicate anomalies so the
//! dashboard shows rolled-up incidents instead of raw event rows.
//!
//! Identical attribution profiles within the sliding window merge
//! into the same cluster; a disparate probe opens a new one.
//!
//! Run with `cargo run --example alert_clustering --features postcard,serde_json`.

use rcf_rs::audit::{AlertContext, AlertRecord};
use rcf_rs::{AlertClusterer, ClusterDecision, ForestBuilder, RcfError};

fn main() -> Result<(), RcfError> {
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(64)
        .seed(42)
        .build()?;
    for i in 0..256 {
        let v = f64::from(i) * 0.01;
        forest.update([v, v + 0.1, v + 0.2, v + 0.3])?;
    }

    let mut clusterer: AlertClusterer<String, 4> =
        AlertClusterer::new(0.95, 60_000).expect("valid config");

    // 50 near-identical anomalies from one tenant + 1 disparate
    // anomaly from another.
    let ctx_a = AlertContext::<String>::for_tenant("tenant-a".into(), 1_000);
    for i in 0..50 {
        let jitter = f64::from(i) * 0.001;
        let rec = AlertRecord::from_forest(&forest, &[10.0 + jitter, 10.0, 10.0, 10.0], &ctx_a)?;
        clusterer.observe(rec);
    }
    let ctx_b = AlertContext::<String>::for_tenant("tenant-b".into(), 2_000);
    let outlier = AlertRecord::from_forest(&forest, &[0.0, 0.0, 0.0, 50.0], &ctx_b)?;
    let decision = clusterer.observe(outlier);
    assert!(matches!(decision, ClusterDecision::NewCluster(_)));

    println!("active clusters: {}", clusterer.len());
    for (i, c) in clusterer.clusters().iter().enumerate() {
        println!(
            "  cluster {i}: count = {}, tenants = {:?}, max_score = {}",
            c.count,
            c.contributing_tenants,
            f64::from(c.max_score),
        );
    }
    Ok(())
}
