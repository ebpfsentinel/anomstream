//! Runtime-dim wrapper for `RandomCutForest` — unblocks
//! heterogeneous multi-tenant deployments where every tenant
//! ships its own feature-vector width (MSSP pools, per-tenant
//! feature extractors).
//!
//! The bare `RandomCutForest<D>` is const-generic on `D` — every
//! distinct dim needs its own monomorphisation. A
//! `TenantForestPool<K, D>` then has a single `D` across every
//! tenant, forcing operators to whitelist dim values at compile
//! time. [`DynamicForest<MAX_D>`] sidesteps the constraint by
//! picking a maximum dim at compile time and zero-padding every
//! caller-supplied point that is shorter than `MAX_D`. The forest
//! internally scores the zero-padded vector; dims above the
//! caller's `active_dim` contribute no range and therefore no
//! attribution.
//!
//! # What this buys
//!
//! - One monomorphisation for every tenant whose dim `≤ MAX_D`.
//! - No API break for callers who already have `[f64; D]` —
//!   [`DynamicForest::update`] / [`DynamicForest::score`] take
//!   `&[f64]` of runtime length.
//!
//! # What this costs
//!
//! - Zero-padding adds a dimension with permanent range `0.0` per
//!   tenant. RCF's range-weighted cut sampling naturally skips
//!   these dims (zero-range = never cut on); there is no AUC
//!   impact, but reservoir memory is paid at `MAX_D` always.
//! - Callers mixing narrow and wide tenants in the same pool pay
//!   the `MAX_D` memory for every tenant; size `MAX_D` to the
//!   widest expected tenant.
//!
//! # Not a replacement for the const-generic path
//!
//! Hot-path callers with a fixed known `D` should keep using
//! [`crate::RandomCutForest<D>`] — the const-generic path is
//! faster (fewer runtime checks, better inlining) and idiomatic.
//! [`DynamicForest`] is the escape hatch for MSSP /
//! heterogeneous-tenant deployments where compile-time `D` is a
//! dealbreaker.

#![cfg(feature = "std")]

use crate::config::ForestBuilder;
use crate::domain::{AnomalyScore, DiVector};
use crate::error::{RcfError, RcfResult};
use crate::forest::RandomCutForest;

/// Runtime-dim facade over `RandomCutForest<MAX_D>`. `MAX_D` must
/// be at compile time ≥ the widest `active_dim` the caller will
/// ever feed.
#[derive(Debug)]
pub struct DynamicForest<const MAX_D: usize> {
    /// Wrapped const-generic forest.
    forest: RandomCutForest<MAX_D>,
    /// Caller-declared dim count — every incoming point must have
    /// exactly `active_dim` finite components; the remaining
    /// `MAX_D − active_dim` slots are zero-padded.
    active_dim: usize,
}

impl<const MAX_D: usize> DynamicForest<MAX_D> {
    /// Build from a prepared [`ForestBuilder<MAX_D>`].
    ///
    /// # Errors
    ///
    /// - [`RcfError::InvalidConfig`] when `active_dim == 0` or
    ///   `active_dim > MAX_D`.
    /// - Propagates [`ForestBuilder::build`] failures.
    pub fn new(builder: ForestBuilder<MAX_D>, active_dim: usize) -> RcfResult<Self> {
        if active_dim == 0 {
            return Err(RcfError::InvalidConfig(
                "DynamicForest: active_dim must be > 0".into(),
            ));
        }
        if active_dim > MAX_D {
            return Err(RcfError::InvalidConfig(
                format!("DynamicForest: active_dim {active_dim} exceeds MAX_D {MAX_D}").into(),
            ));
        }
        let forest = builder.build()?;
        Ok(Self { forest, active_dim })
    }

    /// Active dim of this facade — every input slice must have
    /// this length.
    #[must_use]
    pub fn active_dim(&self) -> usize {
        self.active_dim
    }

    /// Maximum dim the underlying const-generic forest supports.
    #[must_use]
    pub const fn max_dim(&self) -> usize {
        MAX_D
    }

    /// Read-only handle to the underlying const-generic forest —
    /// useful for inspecting metrics / persistence state.
    #[must_use]
    pub fn forest(&self) -> &RandomCutForest<MAX_D> {
        &self.forest
    }

    /// Score a runtime-sized `point`. Returns
    /// [`RcfError::DimensionMismatch`] when `point.len() != active_dim`.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DimensionMismatch`] on length mismatch.
    /// - [`RcfError::NaNValue`] on non-finite components.
    /// - Propagates [`RandomCutForest::score`] failures.
    pub fn score(&self, point: &[f64]) -> RcfResult<AnomalyScore> {
        let padded = self.pad(point)?;
        self.forest.score(&padded)
    }

    /// Fold a runtime-sized `point` into the forest.
    ///
    /// # Errors
    ///
    /// Same as [`Self::score`] plus [`RandomCutForest::update`]
    /// failures.
    pub fn update(&mut self, point: &[f64]) -> RcfResult<()> {
        let padded = self.pad(point)?;
        self.forest.update(padded)
    }

    /// Attribution for a runtime-sized `point`. Returns a
    /// [`DiVector`] of `active_dim` entries (the zero-padded tail
    /// is truncated from the output).
    ///
    /// # Errors
    ///
    /// Same as [`Self::score`] plus [`RandomCutForest::attribution`]
    /// failures.
    pub fn attribution(&self, point: &[f64]) -> RcfResult<DiVector> {
        let padded = self.pad(point)?;
        let di_full = self.forest.attribution(&padded)?;
        // Truncate to active_dim — callers care only about their
        // own feature-vector dims.
        let mut di = DiVector::zeros(self.active_dim);
        for d in 0..self.active_dim {
            let _ = di.add_high(d, di_full.high()[d]);
            let _ = di.add_low(d, di_full.low()[d]);
        }
        Ok(di)
    }

    /// Pad `point` to `[f64; MAX_D]`, validating length and
    /// finite-ness.
    fn pad(&self, point: &[f64]) -> RcfResult<[f64; MAX_D]> {
        if point.len() != self.active_dim {
            return Err(RcfError::DimensionMismatch {
                expected: self.active_dim,
                got: point.len(),
            });
        }
        if !point.iter().all(|v| v.is_finite()) {
            return Err(RcfError::NaNValue);
        }
        let mut padded = [0.0_f64; MAX_D];
        padded[..self.active_dim].copy_from_slice(point);
        Ok(padded)
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

    fn builder() -> ForestBuilder<16> {
        ForestBuilder::<16>::new()
            .num_trees(50)
            .sample_size(64)
            .seed(2026)
    }

    #[test]
    fn new_rejects_zero_active_dim() {
        let err = DynamicForest::<16>::new(builder(), 0).unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn new_rejects_active_dim_above_max() {
        let err = DynamicForest::<16>::new(builder(), 32).unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn update_then_score_preserves_dim_contract() {
        let mut f = DynamicForest::<16>::new(builder(), 4).unwrap();
        for i in 0..200 {
            let v = f64::from(i) * 0.01;
            f.update(&[v, v + 0.5, v * 2.0, v - 0.1]).unwrap();
        }
        let s = f.score(&[10.0, 10.0, 10.0, 10.0]).unwrap();
        let raw: f64 = s.into();
        assert!(raw.is_finite());
        assert!(raw > 0.0);
    }

    #[test]
    fn length_mismatch_rejected() {
        let mut f = DynamicForest::<16>::new(builder(), 4).unwrap();
        for _ in 0..50 {
            f.update(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        }
        assert!(matches!(
            f.score(&[0.1, 0.2, 0.3]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            f.score(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn non_finite_rejected() {
        let mut f = DynamicForest::<16>::new(builder(), 4).unwrap();
        assert!(matches!(
            f.update(&[f64::NAN, 0.0, 0.0, 0.0]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn attribution_truncated_to_active_dim() {
        let mut f = DynamicForest::<16>::new(builder(), 3).unwrap();
        for i in 0..200 {
            let v = f64::from(i) * 0.01;
            f.update(&[v, v + 0.5, v * 2.0]).unwrap();
        }
        let di = f.attribution(&[10.0, 10.0, 10.0]).unwrap();
        assert_eq!(di.dim(), 3);
    }
}
