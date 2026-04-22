#![allow(clippy::unwrap_used, clippy::panic)]
//! Streaming quantile estimator. t-digest keeps sub-percent error
//! on tail percentiles (p99 / p99.9) where fixed-bin histograms
//! lose resolution.
//!
//! Run with `cargo run --example tdigest_quantiles`.

use anomstream_rs::TDigest;

fn main() {
    let mut digest = TDigest::new(200.0).expect("valid compression");

    // Skewed stream: 99 % small values + 1 % large outliers — the
    // classic SOC latency / anomaly-score shape.
    for i in 0..99_000 {
        digest.record(f64::from(i) * 1e-4);
    }
    for i in 0..1_000 {
        digest.record(100.0 + f64::from(i) * 0.01);
    }

    println!("total weight     = {}", digest.total_weight());
    println!("centroid count   = {}", digest.centroid_count());
    println!("min / max        = {:?} / {:?}", digest.min(), digest.max());
    for p in &[50.0, 90.0, 99.0, 99.9] {
        let v = digest.percentile(*p).expect("non-empty digest");
        println!("p{p:<5} = {v:.6}");
    }
}
