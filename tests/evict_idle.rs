#![allow(clippy::unwrap_used, clippy::panic)]
//! Integration checks on `TenantForestPool::evict_idle`.

use std::thread::sleep;
use std::time::Duration;

use rcf_rs::{RcfError, TenantForestPool, ThresholdedForestBuilder};

fn factory_2d() -> impl Fn() -> Result<rcf_rs::ThresholdedForest<2>, RcfError> {
    || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(7)
            .build()
    }
}

#[test]
fn ttl_evicts_stale_and_retains_fresh() {
    let mut p: TenantForestPool<&'static str, 2> = TenantForestPool::new(8, factory_2d()).unwrap();
    p.process(&"stale", [0.0, 0.0]).unwrap();
    p.process(&"fresh", [1.0, 1.0]).unwrap();
    sleep(Duration::from_millis(40));
    // Touch only `fresh` — stale has not been accessed in 40 ms.
    p.process(&"fresh", [0.5, 0.5]).unwrap();
    let evicted = p.evict_idle(Duration::from_millis(20));
    let keys: Vec<_> = evicted.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, vec!["stale"]);
    assert!(p.contains(&"fresh"));
    assert!(!p.contains(&"stale"));
}

#[test]
fn ttl_on_empty_pool_noop() {
    let mut p: TenantForestPool<&'static str, 2> = TenantForestPool::new(4, factory_2d()).unwrap();
    let evicted = p.evict_idle(Duration::from_millis(10));
    assert!(evicted.is_empty());
}

#[test]
fn zero_ttl_evicts_everything_but_just_touched() {
    let mut p: TenantForestPool<&'static str, 2> = TenantForestPool::new(4, factory_2d()).unwrap();
    p.process(&"a", [0.0, 0.0]).unwrap();
    p.process(&"b", [1.0, 1.0]).unwrap();
    sleep(Duration::from_millis(5));
    let evicted = p.evict_idle(Duration::from_millis(0));
    assert_eq!(evicted.len(), 2);
    assert!(p.is_empty());
}

#[test]
fn ttl_path_bumps_lifetime_evicted_counter() {
    let mut p: TenantForestPool<&'static str, 2> = TenantForestPool::new(8, factory_2d()).unwrap();
    p.process(&"t", [0.0, 0.0]).unwrap();
    sleep(Duration::from_millis(20));
    p.evict_idle(Duration::from_millis(5));
    let s = p.readiness_summary();
    assert_eq!(s.tenants_evicted_lifetime, 1);
}
