//! SAGE-style Shapley attribution estimator via permutation
//! sampling — Covert, Lundberg & Lee, *Understanding Global
//! Feature Contributions With Additive Importance Measures*,
//! `NeurIPS` 2020.
//!
//! Exact Shapley values require scoring the model on all `2^D`
//! feature subsets, intractable past `D ≥ 20`. The Monte-Carlo
//! estimator samples `K` random permutations and, for each
//! permutation, computes the marginal contribution of each dim
//! when it joins the coalition. Averaging across permutations
//! gives an unbiased estimate of the Shapley value per dim.
//!
//! # Why this and not the per-dim attribution `DiVector`?
//!
//! The shipped [`anomstream_core::AttributionVisitor`] returns **marginal**
//! per-dim contributions — `attribution[d] = ∂score/∂dim_d` along
//! the forest's random-cut decomposition. Marginals ignore feature
//! interactions; when two dims together signal an anomaly (packet
//! rate × entropy = beacon) but neither alone does, the
//! `DiVector` underweights both. Shapley values account for every
//! coalition's contribution and return a per-dim importance that
//! sums to the score above baseline.
//!
//! # Cost
//!
//! `K · D` forest scores per estimator invocation. Default
//! `K = 64` permutations at `D = 16` = 1 024 scores ≈ 40 ms on
//! reference hardware — batch / SOC triage range, not hot-path.

#![cfg(feature = "std")]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use anomstream_core::error::{RcfError, RcfResult};
use anomstream_core::forest::RandomCutForest;

/// Default number of Monte-Carlo permutations. Higher → tighter
/// estimate at linear cost.
pub const DEFAULT_PERMUTATIONS: usize = 64;

/// Upper bound on permutations. Guards against caller-controlled
/// compute bombs: each permutation costs `D + 1` forest scores,
/// so `1e9 × 16 = 1.6 × 10¹⁰` traversals per `explain()` call.
/// 65 536 keeps the worst-case at `~4 s` on a modern core for
/// `D = 16` — ample convergence for a Monte-Carlo estimator.
pub const MAX_PERMUTATIONS: usize = 65_536;

/// Default RNG seed — reproducible attributions across runs.
pub const DEFAULT_SEED: u64 = 2026;

/// Per-dim Shapley estimates produced by [`SageEstimator::explain`].
#[derive(Debug, Clone, PartialEq)]
pub struct SageExplanation<const D: usize> {
    /// Per-dim Shapley value — positive means the dim pushes the
    /// score **up** vs baseline, negative means it pulls **down**.
    pub shapley: [f64; D],
    /// Baseline score the estimator was built against (score on
    /// the baseline point, with no feature contribution).
    pub baseline_score: f64,
    /// Probe score (no masking applied) — should approximately
    /// equal `baseline_score + sum(shapley)` up to sampling noise.
    pub probe_score: f64,
    /// Permutations actually executed (equals `K` on success).
    pub permutations: usize,
}

impl<const D: usize> SageExplanation<D> {
    /// Dim with the largest absolute Shapley value — the dim the
    /// score most depends on. `None` when `D == 0` or every value
    /// is exactly zero.
    #[must_use]
    pub fn argmax_abs(&self) -> Option<usize> {
        if D == 0 {
            return None;
        }
        let mut best: usize = 0;
        let mut best_val = self.shapley[0].abs();
        for d in 1..D {
            let v = self.shapley[d].abs();
            if v > best_val {
                best = d;
                best_val = v;
            }
        }
        if best_val == 0.0 { None } else { Some(best) }
    }
}

/// Shapley attribution estimator anchored on a caller-supplied
/// baseline point. Stateless beyond the baseline + RNG seed —
/// cheap to instantiate per-probe.
#[derive(Debug, Clone)]
pub struct SageEstimator<const D: usize> {
    /// Baseline feature vector — typically the warm-phase per-dim
    /// mean, or a synthetic "null" point the caller chooses.
    baseline: [f64; D],
    /// Number of Monte-Carlo permutations to sample per probe.
    permutations: usize,
    /// RNG seed for reproducible permutation draws.
    seed: u64,
}

impl<const D: usize> SageEstimator<D> {
    /// Build an estimator anchored on `baseline`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on non-finite `baseline`
    /// components, `permutations == 0`, or
    /// `permutations > MAX_PERMUTATIONS`.
    pub fn new(baseline: [f64; D], permutations: usize, seed: u64) -> RcfResult<Self> {
        if !baseline.iter().all(|v| v.is_finite()) {
            return Err(RcfError::NaNValue);
        }
        if permutations == 0 || permutations > MAX_PERMUTATIONS {
            return Err(RcfError::InvalidConfig(
                format!(
                    "SageEstimator: permutations {permutations} out of (0, {MAX_PERMUTATIONS}]"
                )
                .into(),
            ));
        }
        Ok(Self {
            baseline,
            permutations,
            seed,
        })
    }

    /// Convenience — [`DEFAULT_PERMUTATIONS`] + [`DEFAULT_SEED`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn default_anchor(baseline: [f64; D]) -> RcfResult<Self> {
        Self::new(baseline, DEFAULT_PERMUTATIONS, DEFAULT_SEED)
    }

    /// Estimate the per-dim Shapley contribution for `probe`
    /// against the forest. Runs `permutations` forest scores per
    /// dim-inclusion event → `K · D` total scores per call.
    ///
    /// # Errors
    ///
    /// Propagates [`RandomCutForest::score`] failures.
    #[must_use = "detector output should be checked — dropping it silently usually indicates a logic bug"]
    pub fn explain(
        &self,
        forest: &RandomCutForest<D>,
        probe: &[f64; D],
    ) -> RcfResult<SageExplanation<D>> {
        if !probe.iter().all(|v| v.is_finite()) {
            return Err(RcfError::NaNValue);
        }

        let baseline_score = f64::from(forest.score(&self.baseline)?);
        let probe_score = f64::from(forest.score(probe)?);

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut shapley = [0.0_f64; D];

        for _ in 0..self.permutations {
            // Permute the dim indices.
            let mut order: [usize; D] = core::array::from_fn(|i| i);
            for i in (1..D).rev() {
                let j = rng.random_range(0..=i);
                order.swap(i, j);
            }

            // Walk the permutation: accumulate a feature vector
            // that starts at baseline and adds one probe dim at a
            // time. The marginal contribution of a dim is the
            // score delta when that dim joins the coalition.
            let mut state = self.baseline;
            let mut prev_score = baseline_score;
            for &d in &order {
                state[d] = probe[d];
                let new_score = f64::from(forest.score(&state)?);
                shapley[d] += new_score - prev_score;
                prev_score = new_score;
            }
        }

        // Average across permutations.
        #[allow(clippy::cast_precision_loss)]
        let k = self.permutations as f64;
        for s in &mut shapley {
            *s /= k;
        }

        Ok(SageExplanation {
            shapley,
            baseline_score,
            probe_score,
            permutations: self.permutations,
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;
    use anomstream_core::ForestBuilder;

    fn train_forest<const D: usize>() -> RandomCutForest<D> {
        let mut f = ForestBuilder::<D>::new()
            .num_trees(50)
            .sample_size(64)
            .seed(2026)
            .build()
            .unwrap();
        for i in 0..256 {
            let v = f64::from(i) * 0.01;
            let p = [v; D];
            f.update(p).unwrap();
        }
        f
    }

    #[test]
    fn new_rejects_invalid_params() {
        assert!(SageEstimator::<2>::new([f64::NAN, 0.0], 10, 0).is_err());
        assert!(SageEstimator::<2>::new([0.0, 0.0], 0, 0).is_err());
    }

    #[test]
    fn new_rejects_oversized_permutations() {
        assert!(SageEstimator::<2>::new([0.0, 0.0], MAX_PERMUTATIONS + 1, 0).is_err());
        assert!(SageEstimator::<2>::new([0.0, 0.0], usize::MAX, 0).is_err());
    }

    #[test]
    fn explain_rejects_non_finite_probe() {
        let f = train_forest::<2>();
        let est = SageEstimator::default_anchor([0.0, 0.0]).unwrap();
        assert!(est.explain(&f, &[f64::NAN, 0.0]).is_err());
    }

    #[test]
    fn shapley_sums_approximate_score_delta() {
        let f = train_forest::<4>();
        let baseline = [0.0_f64; 4];
        let est = SageEstimator::new(baseline, 128, 11).unwrap();
        let probe = [50.0, 0.0, -30.0, 0.0];
        let exp = est.explain(&f, &probe).unwrap();
        let sum: f64 = exp.shapley.iter().sum();
        let delta = exp.probe_score - exp.baseline_score;
        // Monte-Carlo estimator → sum ≈ delta up to sampling
        // noise. Tolerance is generous for K = 128.
        assert!(
            (sum - delta).abs() < 0.5 * delta.abs().max(0.5),
            "sum(shapley) = {sum}, delta = {delta}"
        );
    }

    #[test]
    fn argmax_identifies_dominant_dim() {
        let f = train_forest::<4>();
        let est = SageEstimator::new([0.0; 4], 128, 17).unwrap();
        // Only dim 2 deviates — Shapley should attribute most of
        // the score delta there.
        let probe = [0.0_f64, 0.0, 50.0, 0.0];
        let exp = est.explain(&f, &probe).unwrap();
        let argmax = exp.argmax_abs().expect("non-zero shapley");
        assert_eq!(argmax, 2);
    }

    #[test]
    fn zero_probe_against_baseline_yields_near_zero_shapley() {
        let f = train_forest::<3>();
        let est = SageEstimator::new([0.1; 3], 64, 5).unwrap();
        let exp = est.explain(&f, &[0.1; 3]).unwrap();
        let sum: f64 = exp.shapley.iter().sum();
        assert!(sum.abs() < 0.01, "sum shapley = {sum} on baseline probe");
    }
}
