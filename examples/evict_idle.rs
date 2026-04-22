#![allow(clippy::unwrap_used, clippy::panic)]
//! `TTL` / idle-based tenant eviction.
//!
//! LRU alone keeps a tenant slot warm as long as the pool isn't
//! full, even when the tenant has been dormant for hours. For
//! `MSSP` / `SaaS` deployments with thousands of intermittent tenants,
//! an **idle** tenant should make room for the next tier. The
//! `evict_idle(ttl)` call sheds every tenant whose last access is
//! older than `ttl`.
//!
//! Run with `cargo run --example evict_idle`.

use std::thread::sleep;
use std::time::Duration;

use anomstream_rs::{RcfError, TenantForestPool, ThresholdedForestBuilder};

fn main() -> Result<(), RcfError> {
    let mut pool: TenantForestPool<&'static str, 2> = TenantForestPool::new(16, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(2026)
            .build()
    })?;

    // Two tenants push some traffic, then fall silent.
    for i in 0..10 {
        let v = f64::from(i) * 0.01;
        pool.process(&"alpha", [v, v])?;
        pool.process(&"beta", [v, v + 1.0])?;
    }

    println!("before idle: {} tenants resident", pool.len());

    // Simulate a 50 ms dormancy window.
    sleep(Duration::from_millis(50));

    // Touch `alpha` — only `beta` is idle past the TTL.
    pool.process(&"alpha", [0.1, 0.1])?;

    let evicted = pool.evict_idle(Duration::from_millis(25));
    println!(
        "evict_idle shed {} tenant(s); resident = {}",
        evicted.len(),
        pool.len()
    );
    for (k, _) in &evicted {
        println!("  evicted: {k}");
    }

    let summary = pool.readiness_summary();
    println!(
        "resident = {} / capacity = {}, evicted lifetime = {}",
        summary.resident, summary.capacity, summary.tenants_evicted_lifetime
    );
    Ok(())
}
