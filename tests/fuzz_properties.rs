#![allow(clippy::unwrap_used, clippy::panic)]
//! Property-based / fuzz-style tests.
//!
//! Exercises every public entry point against adversarial inputs
//! (NaN, ±inf, subnormals, extreme magnitudes) and cross-API
//! invariants (roundtrip, serial-vs-bulk equivalence, insert/delete
//! symmetry). Stays on stable via `proptest` — libFuzzer harnesses
//! would require nightly and are intentionally out of scope.
//!
//! Every property runs `cases = 64` by default, small enough to
//! keep `cargo test` under a few seconds but large enough to surface
//! regressions that a handful of fixed cases would miss.
//!
//! Only compiled with both `postcard` and `serde_json` features so
//! persistence roundtrip properties have both codecs available —
//! matches `persistence_roundtrip.rs`.

#![cfg(all(feature = "postcard", feature = "serde_json"))]
#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use anomstream_rs::{
    AnomalyScore, ForestBuilder, RandomCutForest, RcfError, TenantForestPool,
    ThresholdedForestBuilder,
};
use proptest::prelude::*;

const D: usize = 4;

/// Build a warmed 4-D forest with `updates` points drawn from the
/// input strategy. Determinism comes from `seed`, so a failing
/// proptest shrink is reproducible.
fn warm_forest(seed: u64, points: &[[f64; D]]) -> RandomCutForest<D> {
    let mut forest = ForestBuilder::<D>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(seed)
        .build()
        .expect("builder");
    for p in points {
        let _ = forest.update(*p);
    }
    forest
}

/// Bounded-magnitude finite point — avoids `±inf` from arithmetic on
/// 1e308 values inside the bounding-box math while still covering a
/// wide dynamic range.
fn finite_point() -> impl Strategy<Value = [f64; D]> {
    prop::array::uniform4(-1.0e6_f64..1.0e6_f64)
}

/// Adversarial f64 — full IEEE-754 domain including NaN, ±inf,
/// subnormals. Used to prove we never panic on hostile input.
fn any_point() -> impl Strategy<Value = [f64; D]> {
    prop::array::uniform4(any::<f64>())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// Postcard roundtrip: encoding then decoding a forest must
    /// preserve the score for an arbitrary probe, bit-exact.
    #[test]
    fn postcard_roundtrip_preserves_score(
        seed in 0_u64..1_000,
        pts in prop::collection::vec(finite_point(), 20..80),
        probe in finite_point(),
    ) {
        let forest = warm_forest(seed, &pts);
        let bytes = forest.to_bytes().expect("encode");
        let back = RandomCutForest::<D>::from_bytes(&bytes).expect("decode");
        let s1: f64 = forest.score(&probe).expect("score").into();
        let s2: f64 = back.score(&probe).expect("score").into();
        prop_assert_eq!(s1, s2);
    }

    /// JSON roundtrip: same invariant, different codec. JSON uses
    /// ryu shortest-roundtrip for `f64`; 1-ULP drift on intermediate
    /// scaled values is tolerated (postcard path is bit-exact).
    #[test]
    fn json_roundtrip_preserves_score(
        seed in 0_u64..1_000,
        pts in prop::collection::vec(finite_point(), 20..80),
        probe in finite_point(),
    ) {
        let forest = warm_forest(seed, &pts);
        let json = forest.to_json().expect("encode");
        let back = RandomCutForest::<D>::from_json(&json).expect("decode");
        let s1: f64 = forest.score(&probe).expect("score").into();
        let s2: f64 = back.score(&probe).expect("score").into();
        let tol = s1.abs().max(1.0) * 1.0e-12;
        prop_assert!(
            (s1 - s2).abs() <= tol,
            "score drift {} vs {} exceeds tolerance {}", s1, s2, tol,
        );
    }

    /// Non-finite inputs on every write/read path must return
    /// `NaNValue` cleanly — never panic, never poison internal state.
    /// A follow-up finite score must still succeed.
    #[test]
    fn non_finite_input_rejected_cleanly(
        seed in 0_u64..1_000,
        warm in prop::collection::vec(finite_point(), 10..40),
        adversarial in any_point(),
        sentinel in finite_point(),
    ) {
        let mut forest = warm_forest(seed, &warm);
        let any_non_finite = adversarial.iter().any(|v| !v.is_finite());

        let update_result = forest.update(adversarial);
        let score_result = forest.score(&adversarial);
        let forensic_result = forest.forensic_baseline(&adversarial);

        if any_non_finite {
            prop_assert!(matches!(update_result, Err(RcfError::NaNValue)));
            prop_assert!(matches!(score_result, Err(RcfError::NaNValue)));
            prop_assert!(matches!(forensic_result, Err(RcfError::NaNValue)));
        }

        // Sentinel must still work — rejected input must not have
        // corrupted forest state.
        let s: AnomalyScore = forest.score(&sentinel).expect("post-adversarial score");
        prop_assert!(f64::from(s).is_finite());
    }

    /// `score_many` must be observationally equivalent to a serial
    /// `score` loop, element-wise and bit-exact.
    #[test]
    fn score_many_equals_serial_score(
        seed in 0_u64..1_000,
        pts in prop::collection::vec(finite_point(), 30..80),
        probes in prop::collection::vec(finite_point(), 1..32),
    ) {
        let forest = warm_forest(seed, &pts);
        let bulk = forest.score_many(&probes).expect("bulk");
        prop_assert_eq!(bulk.len(), probes.len());
        for (i, p) in probes.iter().enumerate() {
            let s: f64 = forest.score(p).expect("serial").into();
            let b: f64 = bulk[i].into();
            prop_assert_eq!(s, b);
        }
    }

    /// `delete_by_value` retracts every bit-exact match of the
    /// target point. Property: after insert-N-copies + delete-by-
    /// value, the live count falls by *at least* N copies; the
    /// forest remains scorable on a different probe.
    #[test]
    fn delete_by_value_shrinks_live_count(
        seed in 0_u64..1_000,
        pts in prop::collection::vec(finite_point(), 20..60),
        target in finite_point(),
        copies in 1_usize..6,
        probe in finite_point(),
    ) {
        let mut forest = warm_forest(seed, &pts);
        for _ in 0..copies {
            forest.update(target).expect("target insert");
        }
        let before = forest.point_store().live_count();
        let removed = forest.delete_by_value(&target).expect("delete");
        let after = forest.point_store().live_count();
        prop_assert!(removed >= copies, "should remove at least {} copies, got {}", copies, removed);
        prop_assert_eq!(before - after, removed);
        // Forest still scorable.
        let s: f64 = forest.score(&probe).expect("post-delete score").into();
        prop_assert!(s.is_finite());
    }

    /// TRCF `process` chain on arbitrary finite points: every call
    /// must yield a finite grade in `[0, 1]`. Non-finite inputs must
    /// error cleanly.
    #[test]
    fn trcf_process_never_panics(
        seed in 0_u64..1_000,
        stream in prop::collection::vec(any_point(), 5..40),
    ) {
        let mut trcf = ThresholdedForestBuilder::<D>::new()
            .num_trees(50)
            .sample_size(32)
            .min_observations(4)
            .seed(seed)
            .build()
            .expect("builder");
        for p in &stream {
            let any_non_finite = p.iter().any(|v| !v.is_finite());
            match trcf.process(*p) {
                Ok(grade) => {
                    prop_assert!(grade.grade().is_finite());
                    prop_assert!((0.0..=1.0).contains(&grade.grade()));
                }
                Err(RcfError::NaNValue) => prop_assert!(any_non_finite),
                Err(e) => return Err(TestCaseError::fail(format!("unexpected error: {e:?}"))),
            }
        }
    }

    /// Cross-tenant scoring on an arbitrary pool + probe must never
    /// panic; the returned vec is sorted descending on grade.
    #[test]
    fn score_across_tenants_sorted_and_safe(
        tenant_count in 2_usize..8,
        warm_per_tenant in 16_usize..40,
        probe in finite_point(),
        seed in 0_u64..1_000,
    ) {
        let pool_seed = seed;
        let mut pool: TenantForestPool<u32, D> = TenantForestPool::new(tenant_count + 2, move || {
            ThresholdedForestBuilder::<D>::new()
                .num_trees(50)
                .sample_size(32)
                .min_observations(4)
                .seed(pool_seed)
                .build()
        })
        .expect("pool");
        for t in 0..u32::try_from(tenant_count).expect("tenant fits") {
            let offset = f64::from(t);
            for i in 0..warm_per_tenant {
                let v = (i as f64) * 0.01 + offset;
                pool.process(&t, [v, v, v, v]).expect("process");
            }
        }
        let ranked = pool.score_across_tenants(&probe).expect("cross-tenant");
        for [a, b] in ranked.array_windows::<2>() {
            prop_assert!(a.1.grade() >= b.1.grade());
        }
    }

    /// `forensic_baseline` reconstructs a per-dim baseline from live
    /// reservoir points. Property: the baseline is finite and its
    /// per-dim value lies within the observed `[min, max]` range of
    /// the training set, since Welford-mean of any finite sample is
    /// bounded by its extremes.
    #[test]
    fn forensic_baseline_within_observed_range(
        seed in 0_u64..1_000,
        pts in prop::collection::vec(finite_point(), 40..80),
        probe in finite_point(),
    ) {
        let forest = warm_forest(seed, &pts);
        let baseline = forest.forensic_baseline(&probe).expect("forensic");
        // Observed per-dim min/max across the raw training set.
        let mut lo = [f64::INFINITY; D];
        let mut hi = [f64::NEG_INFINITY; D];
        for p in &pts {
            for d in 0..D {
                if p[d] < lo[d] { lo[d] = p[d]; }
                if p[d] > hi[d] { hi[d] = p[d]; }
            }
        }
        // Welford mean of the live sample ⊆ [lo, hi] on every dim.
        // Tolerate tiny FP drift from scaled-space conversion.
        for d in 0..D {
            let b = baseline.expected[d];
            prop_assert!(b.is_finite());
            let slack = (hi[d] - lo[d]).abs().max(1.0) * 1.0e-9;
            prop_assert!(
                b >= lo[d] - slack && b <= hi[d] + slack,
                "dim {}: baseline {} outside [{}, {}]", d, b, lo[d], hi[d],
            );
        }
    }
}
