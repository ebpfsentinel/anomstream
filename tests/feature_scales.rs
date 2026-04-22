#![allow(clippy::unwrap_used, clippy::panic)]
//! End-to-end behaviour of `feature_scales` pre-scaling.
//!
//! Asserts:
//!
//! 1. A unit scale vector (`[1.0; D]`) produces scores identical to
//!    no scales configured — scaling is neutral at the identity.
//! 2. A non-unit scale vector changes scoring behaviour
//!    deterministically.
//! 3. Builder validation rejects a wrong-length scale vector before
//!    the forest is instantiated.
//! 4. Builder validation rejects non-finite and non-positive scale
//!    components.
//! 5. `delete_by_value` matches the scaled-space representation —
//!    inserting and then retracting the same raw point works end to
//!    end.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use anomstream_rs::{ForestBuilder, RcfError};

#[test]
fn identity_scales_preserve_scoring() {
    let mut reference = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(2026)
        .build()
        .unwrap();
    let mut scaled = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(32)
        .feature_scales([1.0, 1.0])
        .seed(2026)
        .build()
        .unwrap();
    for i in 0_u32..128 {
        let v = f64::from(i) * 0.01;
        reference.update([v, v + 0.5]).unwrap();
        scaled.update([v, v + 0.5]).unwrap();
    }
    let probe = [3.0_f64, 3.5];
    let s_ref: f64 = reference.score(&probe).unwrap().into();
    let s_sc: f64 = scaled.score(&probe).unwrap().into();
    // The same seed + identity scales => identical sampling + scoring.
    assert_eq!(s_ref, s_sc);
}

#[test]
fn non_unit_scales_change_scoring() {
    let mut reference = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(2026)
        .build()
        .unwrap();
    let mut scaled = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(32)
        .feature_scales([10.0, 0.1])
        .seed(2026)
        .build()
        .unwrap();
    for i in 0_u32..256 {
        let v = f64::from(i) * 0.01;
        reference.update([v, v + 0.5]).unwrap();
        scaled.update([v, v + 0.5]).unwrap();
    }
    // Probe on an uneven dimension — with scale [10, 0.1] dim 0 is
    // exaggerated and dim 1 is compressed, so the anomaly profile
    // differs from the reference forest.
    let probe = [1.0_f64, 1.5];
    let s_ref: f64 = reference.score(&probe).unwrap().into();
    let s_sc: f64 = scaled.score(&probe).unwrap().into();
    assert_ne!(
        s_ref, s_sc,
        "non-identity scales should change scoring; ref={s_ref} scaled={s_sc}",
    );
}

#[test]
fn build_rejects_wrong_length_scales() {
    // Build a config manually with a mismatched feature_scales length
    // — the builder's dimension check fires before the forest is
    // instantiated.
    let b = ForestBuilder::<2>::new().seed(1);
    let mut cfg = b.config().clone();
    cfg.feature_scales = Some(vec![1.0, 1.0, 1.0]); // length 3 != D=2
    let err = anomstream_rs::RandomCutForest::<2>::from_config(cfg).unwrap_err();
    assert!(matches!(err, RcfError::DimensionMismatch { .. }));
}

#[test]
fn build_rejects_non_finite_scales() {
    let err = ForestBuilder::<2>::new()
        .feature_scales([f64::NAN, 1.0])
        .build()
        .unwrap_err();
    assert!(matches!(err, RcfError::InvalidConfig(_)));
    let err = ForestBuilder::<2>::new()
        .feature_scales([f64::INFINITY, 1.0])
        .build()
        .unwrap_err();
    assert!(matches!(err, RcfError::InvalidConfig(_)));
}

#[test]
fn build_rejects_non_positive_scales() {
    let err = ForestBuilder::<2>::new()
        .feature_scales([0.0, 1.0])
        .build()
        .unwrap_err();
    assert!(matches!(err, RcfError::InvalidConfig(_)));
    let err = ForestBuilder::<2>::new()
        .feature_scales([-1.0, 1.0])
        .build()
        .unwrap_err();
    assert!(matches!(err, RcfError::InvalidConfig(_)));
}

#[test]
fn clear_feature_scales_restores_unweighted_builder() {
    let b = ForestBuilder::<2>::new()
        .feature_scales([2.0, 3.0])
        .clear_feature_scales();
    assert!(b.config().feature_scales.is_none());
}

#[test]
fn delete_by_value_roundtrip_with_scales() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .feature_scales([10.0, 0.1])
        .seed(42)
        .build()
        .unwrap();
    let target = [0.5_f64, 0.25];
    for _ in 0..4 {
        f.update(target).unwrap();
    }
    let removed = f.delete_by_value(&target).unwrap();
    assert!(
        removed > 0,
        "delete_by_value should find scaled entries stored under the same scaling",
    );
}
