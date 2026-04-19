#![allow(clippy::unwrap_used, clippy::panic)]
//! Accuracy / behaviour checks on [`TDigest`] against closed-form
//! reference distributions.

use rcf_rs::TDigest;

#[test]
fn uniform_stream_quantiles_within_one_percent() {
    let mut d = TDigest::new(200.0).unwrap();
    for i in 0..10_000_u32 {
        d.record(f64::from(i));
    }
    for (q, truth) in &[
        (0.5, 4999.5),
        (0.9, 8999.0),
        (0.99, 9899.0),
        (0.999, 9989.0),
    ] {
        let v = d.quantile(*q).unwrap();
        let err = (v - truth).abs() / 10_000.0;
        assert!(
            err < 0.01,
            "quantile({q}) = {v}, truth = {truth}, err = {err}"
        );
    }
}

#[test]
fn skewed_stream_tail_accurate() {
    // 90 % small values + 10 % large — typical anomaly-score shape.
    // p95 lands well inside the upper mode, p99 near its middle.
    let mut d = TDigest::new(200.0).unwrap();
    for i in 0..90_000_u32 {
        d.record(f64::from(i) * 1.0e-4);
    }
    for i in 0..10_000_u32 {
        d.record(100.0 + f64::from(i) * 0.01);
    }
    let p95 = d.quantile(0.95).unwrap();
    let p99 = d.quantile(0.99).unwrap();
    assert!(
        p95 > 100.0 && p95 < 150.0,
        "p95 should land inside the upper mode, got {p95}"
    );
    assert!(
        p99 > 100.0 && p99 < 200.0,
        "p99 should sit mid-upper-mode, got {p99}"
    );
}

#[test]
fn percentile_matches_quantile_over_100() {
    let mut d = TDigest::with_default_compression();
    for i in 0..1_000_u32 {
        d.record(f64::from(i));
    }
    let a = d.quantile(0.5).unwrap();
    let b = d.percentile(50.0).unwrap();
    assert!((a - b).abs() < 1e-9);
}

#[test]
fn centroid_count_bounded_post_flush() {
    let mut d = TDigest::new(100.0).unwrap();
    for i in 0..100_000_u32 {
        d.record(f64::from(i));
    }
    d.flush();
    assert!(d.centroid_count() <= 250);
}

#[test]
fn merge_preserves_extremes() {
    let mut a = TDigest::new(100.0).unwrap();
    let mut b = TDigest::new(100.0).unwrap();
    for i in 0..1_000_u32 {
        a.record(f64::from(i));
    }
    for i in 1_000..2_000_u32 {
        b.record(f64::from(i));
    }
    a.merge(&b).unwrap();
    assert_eq!(a.min(), Some(0.0));
    assert_eq!(a.max(), Some(1_999.0));
    let median = a.quantile(0.5).unwrap();
    assert!((median - 999.5).abs() < 50.0);
}

#[test]
fn merge_rejects_incompatible_compression() {
    let mut a = TDigest::new(100.0).unwrap();
    let b = TDigest::new(200.0).unwrap();
    assert!(a.merge(&b).is_err());
}

#[test]
fn empty_digest_quantile_is_none() {
    let mut d = TDigest::with_default_compression();
    assert!(d.quantile(0.5).is_none());
}

#[cfg(feature = "postcard")]
#[test]
fn postcard_roundtrip_preserves_quantiles() {
    let mut d = TDigest::new(200.0).unwrap();
    for i in 0..2_000_u32 {
        d.record(f64::from(i));
    }
    d.flush();
    let bytes = postcard::to_allocvec(&d).unwrap();
    let mut back: TDigest = postcard::from_bytes(&bytes).unwrap();
    let before = d.quantile(0.9).unwrap();
    let after = back.quantile(0.9).unwrap();
    assert!((before - after).abs() < 1e-9);
}
