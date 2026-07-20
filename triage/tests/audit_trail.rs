#![allow(clippy::unwrap_used, clippy::panic, clippy::float_cmp)]
//! End-to-end [`AlertRecord`] wiring on bare forest + TRCF plus
//! serde roundtrip on both codecs.

#![cfg(all(feature = "serde", feature = "postcard", feature = "serde_json"))]

use anomstream_core::{ForestBuilder, SeverityBands, ThresholdedForestBuilder};
use anomstream_triage::audit::{ALERT_RECORD_VERSION, AlertContext, AlertRecord};

fn warm_forest() -> anomstream_core::RandomCutForest<4> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(2026)
        .build()
        .unwrap();
    for i in 0..64_u32 {
        let v = f64::from(i) * 0.01;
        f.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    f
}

#[test]
fn audit_record_from_forest_captures_all_analytics() {
    let f = warm_forest();
    let ctx = AlertContext::<String>::for_tenant("tenant-x".into(), 1_700_000_000_000);
    let rec = AlertRecord::from_forest(&f, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
    assert_eq!(rec.version, ALERT_RECORD_VERSION);
    assert_eq!(rec.tenant.as_deref(), Some("tenant-x"));
    assert!(f64::from(rec.score) > 0.0);
    assert_eq!(rec.attribution.dim(), 4);
    assert!(rec.baseline.live_points > 0);
    assert!(rec.grade.is_none());
}

#[test]
fn audit_record_from_trcf_emits_grade() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(16)
        .min_observations(4)
        .seed(2026)
        .build()
        .unwrap();
    for i in 0..32_u32 {
        let v = f64::from(i) * 0.01;
        d.process([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    let ctx = AlertContext::<String>::untenanted(42);
    let rec = AlertRecord::from_thresholded(&mut d, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
    assert!(rec.grade.is_some());
    assert!(rec.tenant.is_none());
}

#[test]
fn audit_record_with_severity_maps_band() {
    let f = warm_forest();
    let ctx = AlertContext::<String>::untenanted(0);
    let bands = SeverityBands::default();
    let rec = AlertRecord::from_forest(&f, &[10.0, 10.0, 10.0, 10.0], &ctx)
        .unwrap()
        .with_severity(&bands);
    assert!(rec.severity.is_some());
}

#[test]
fn audit_record_postcard_roundtrip() {
    let f = warm_forest();
    let ctx = AlertContext::<String>::for_tenant("t".into(), 17);
    let rec = AlertRecord::from_forest(&f, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
    let bytes = postcard::to_allocvec(&rec).unwrap();
    let back: AlertRecord<String, 4> = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(rec, back);
}

#[test]
fn audit_record_json_roundtrip_bit_exact_non_fp_fields() {
    let f = warm_forest();
    let ctx = AlertContext::<String>::for_tenant("t".into(), 17);
    let rec = AlertRecord::from_forest(&f, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
    let json = serde_json::to_string(&rec).unwrap();
    let back: AlertRecord<String, 4> = serde_json::from_str(&json).unwrap();
    // Non-fp identity must be exact. `point` is the caller-supplied
    // probe (exactly representable), so it is exact too.
    assert_eq!(rec.version, back.version);
    assert_eq!(rec.tenant, back.tenant);
    assert_eq!(rec.timestamp_ms, back.timestamp_ms);
    assert_eq!(rec.point, back.point);
    assert_eq!(rec.baseline.live_points, back.baseline.live_points);
    // `score` is a *derived* f64 (a mean over per-tree scores), so it
    // can land on a double whose shortest decimal form re-parses one
    // ULP away — e.g. serde_json writes `1.8295224971362654` and reads
    // back `1.8295224971362656`. Which double we land on legitimately
    // depends on the ensemble summation order (serial fold vs rayon
    // reduce tree), so pin the tolerance at one ULP rather than
    // asserting bit-equality the format cannot guarantee.
    let (lhs, rhs) = (f64::from(rec.score), f64::from(back.score));
    assert!(
        (lhs - rhs).abs() <= f64::EPSILON * lhs.abs().max(1.0),
        "score drifted more than 1 ULP across the JSON round-trip: {lhs} vs {rhs}"
    );
}
