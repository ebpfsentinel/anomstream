#![allow(clippy::unwrap_used, clippy::panic)]
//! Behaviour of the warmup admission gate (`initial_accept_fraction`).
//!
//! Three properties the forest-level wiring must preserve:
//!
//! 1. The default build has the gate disabled (`fraction == 1.0`) so
//!    legacy behaviour is byte-for-byte unchanged.
//! 2. A fraction below `1.0` actually slows down the early reservoir
//!    fill at the per-tree level.
//! 3. The gate does not compromise the without-replacement invariant
//!    (no duplicated `point_idx` in any tree's reservoir).

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // Test-only bounded casts.

use std::collections::HashSet;

use anomstream_rs::ForestBuilder;

#[test]
fn default_forest_has_warmup_gate_disabled() {
    let f = ForestBuilder::<4>::new().seed(1).build().unwrap();
    assert!((f.config().initial_accept_fraction - 1.0).abs() < f64::EPSILON);
    for (_, sampler, _) in f.trees() {
        assert!((sampler.initial_accept_fraction() - 1.0).abs() < f64::EPSILON);
    }
}

#[test]
fn warmup_gate_slows_reservoir_fill() {
    // Two forests built with the same seed but different warmup
    // fractions. After the same number of updates, the gated forest
    // should hold strictly fewer samples (on average across trees)
    // than the un-gated one.
    let mut open = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .initial_accept_fraction(1.0)
        .seed(2026)
        .build()
        .unwrap();
    let mut gated = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .initial_accept_fraction(0.125)
        .seed(2026)
        .build()
        .unwrap();

    for i in 0_u32..6 {
        // Only feed enough updates to stay within the warmup window:
        // threshold = 0.125 * 64 = 8, so the first ~7 offers are
        // where gating dominates.
        let v = f64::from(i) * 0.01;
        let p = [v, v + 0.1, v + 0.2, v + 0.3];
        open.update(p).unwrap();
        gated.update(p).unwrap();
    }

    let open_fill: usize = open.trees().iter().map(|(_, s, _)| s.len()).sum();
    let gated_fill: usize = gated.trees().iter().map(|(_, s, _)| s.len()).sum();
    assert_eq!(
        open_fill,
        6 * open.num_trees(),
        "gate-off reservoir should admit every warmup offer",
    );
    assert!(
        gated_fill < open_fill,
        "gated reservoir should hold fewer samples during warmup: \
         open={open_fill} gated={gated_fill}",
    );
}

#[test]
fn warmup_gate_preserves_without_replacement_invariant() {
    // Stream well past capacity and check each tree's reservoir still
    // holds unique indices — the gate must not break the baseline
    // AWS conformance property.
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .initial_accept_fraction(0.125)
        .seed(7)
        .build()
        .unwrap();
    for i in 0_u32..1_000 {
        let v = f64::from(i);
        f.update([v, v, v, v]).unwrap();
    }
    for (_, sampler, _) in f.trees() {
        let idxs: Vec<usize> = sampler.iter_indices().collect();
        let set: HashSet<usize> = idxs.iter().copied().collect();
        assert_eq!(idxs.len(), set.len(), "duplicate index under warmup gate");
    }
}

#[test]
fn forest_builder_validates_initial_accept_fraction() {
    let err = ForestBuilder::<4>::new()
        .initial_accept_fraction(1.5)
        .build()
        .unwrap_err();
    assert!(matches!(err, anomstream_rs::RcfError::InvalidConfig(_)));

    let err = ForestBuilder::<4>::new()
        .initial_accept_fraction(0.0)
        .build()
        .unwrap_err();
    assert!(matches!(err, anomstream_rs::RcfError::InvalidConfig(_)));
}
