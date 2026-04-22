#![allow(clippy::unwrap_used, clippy::panic)]
//! Integration checks on `TenantForestPool::readiness_summary`.

use anomstream_rs::{RcfError, TenantForestPool, ThresholdedForestBuilder};

fn factory_2d() -> impl Fn() -> Result<anomstream_rs::ThresholdedForest<2>, RcfError> {
    || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .min_observations(8)
            .seed(11)
            .build()
    }
}

#[test]
fn empty_pool_is_vacuously_ready() {
    let p: TenantForestPool<&'static str, 2> = TenantForestPool::new(4, factory_2d()).unwrap();
    let s = p.readiness_summary();
    assert_eq!(s.resident, 0);
    assert_eq!(s.warming, 0);
    assert_eq!(s.ready, 0);
    assert!(s.is_fully_ready());
    assert!(!s.is_at_capacity());
    assert!(s.readiness_ratio().is_nan());
}

#[test]
fn mixed_warm_and_warming_tenants() {
    let mut p: TenantForestPool<&'static str, 2> = TenantForestPool::new(4, factory_2d()).unwrap();
    // Warm "a" past min_observations(8).
    for i in 0..20_u32 {
        let v = f64::from(i) * 0.01;
        p.process(&"a", [v, v]).unwrap();
    }
    // "b" still warming.
    p.process(&"b", [0.0, 0.0]).unwrap();
    p.process(&"b", [0.01, 0.01]).unwrap();

    let s = p.readiness_summary();
    assert_eq!(s.resident, 2);
    assert_eq!(s.ready, 1);
    assert_eq!(s.warming, 1);
    assert!(!s.is_fully_ready());
    assert!((s.readiness_ratio() - 0.5).abs() < 1e-9);
}

#[test]
fn lifetime_counters_track_create_and_evict() {
    let mut p: TenantForestPool<&'static str, 2> = TenantForestPool::new(2, factory_2d()).unwrap();
    p.process(&"a", [0.0, 0.0]).unwrap();
    p.process(&"b", [1.0, 1.0]).unwrap();
    p.process(&"c", [2.0, 2.0]).unwrap(); // evicts "a"
    let s = p.readiness_summary();
    assert_eq!(s.tenants_created_lifetime, 3);
    assert_eq!(s.tenants_evicted_lifetime, 1);
    assert_eq!(s.resident, 2);
    assert!(s.is_at_capacity());
}
