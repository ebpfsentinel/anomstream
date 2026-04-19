#![allow(clippy::unwrap_used, clippy::panic)]
//! Aggregate pool readiness — surface for `/healthz` / `/readyz`
//! health-check endpoints. `readiness_summary()` classifies every
//! resident tenant as warming or ready and reports lifetime
//! create/evict counters.
//!
//! Run with `cargo run --example readiness_summary`.

use rcf_rs::{RcfError, TenantForestPool, ThresholdedForestBuilder};

fn main() -> Result<(), RcfError> {
    let mut pool: TenantForestPool<String, 2> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .min_observations(8)
            .seed(7)
            .build()
    })?;

    // Warm tenant "a" past min_observations; leave "b" warming.
    for i in 0..20 {
        let v = f64::from(i) * 0.01;
        pool.process(&"alpha".into(), [v, v])?;
    }
    pool.process(&"beta".into(), [0.0, 0.0])?;
    pool.process(&"beta".into(), [0.01, 0.01])?;

    let s = pool.readiness_summary();
    println!("resident = {}", s.resident);
    println!("  ready   = {}", s.ready);
    println!("  warming = {}", s.warming);
    println!("capacity  = {}", s.capacity);
    println!("ratio     = {:.2}", s.readiness_ratio());
    println!("fully ready? {}", s.is_fully_ready());
    println!(
        "lifetime: created = {}, evicted = {}",
        s.tenants_created_lifetime, s.tenants_evicted_lifetime
    );
    Ok(())
}
