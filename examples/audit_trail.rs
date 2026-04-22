#![allow(clippy::unwrap_used, clippy::panic)]
//! Structured audit record for NIS2 / SOC2 compliance.
//!
//! Every "alert fired" event can be serialised to a durable log —
//! score, grade, attribution, forensic baseline, severity, tenant
//! key, timestamp — without the caller needing to reach into the
//! underlying detector's analytic primitives.
//!
//! Run with `cargo run --example audit_trail --features postcard,serde_json`.

use anomstream_rs::audit::{AlertContext, AlertRecord};
use anomstream_rs::{ForestBuilder, RcfError, SeverityBands};

fn main() -> Result<(), RcfError> {
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(64)
        .seed(2026)
        .build()?;
    // Warm: dense cluster around origin.
    for i in 0..256 {
        let v = f64::from(i) * 0.01;
        forest.update([v, v + 0.1, v + 0.2, v + 0.3])?;
    }

    // Outlier probe triggers audit record capture.
    let probe = [5.0, 5.0, 5.0, 5.0];
    let ctx = AlertContext::<String>::for_tenant("customer-42".into(), 1_700_000_000_000);
    let bands = SeverityBands::default();
    let record = AlertRecord::from_forest(&forest, &probe, &ctx)?.with_severity(&bands);

    println!(
        "alert: version={} tenant={:?} ts={}",
        record.version, record.tenant, record.timestamp_ms
    );
    println!("  score    = {}", f64::from(record.score));
    println!("  severity = {:?}", record.severity);
    println!("  argmax attr dim     = {:?}", record.attribution.argmax());
    println!(
        "  argmax |zscore| dim = {:?}",
        record.baseline.argmax_abs_zscore()
    );

    // Serialise to both JSON (self-describing) and postcard (compact).
    #[cfg(feature = "serde_json")]
    {
        let json = serde_json::to_string(&record).expect("JSON serialise");
        println!("json ({} bytes) — send to SIEM / WORM log", json.len());
    }
    #[cfg(feature = "postcard")]
    {
        let bytes = postcard::to_allocvec(&record).expect("postcard serialise");
        println!("postcard ({} bytes) — compact wire format", bytes.len());
    }
    Ok(())
}
