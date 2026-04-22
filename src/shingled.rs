//! Internal shingling on top of [`crate::RandomCutForest`].
//!
//! Turns a **scalar stream** into a `D`-dim feature vector by keeping
//! the last `D` observations in a ring buffer. Each new scalar shifts
//! the window and emits a fresh `[f64; D]` to the forest. Isolation-
//! depth scoring on the shingled view captures **temporal
//! autocorrelation** that bare scalar scoring cannot — a dwell
//! anomaly at constant rate (NAB `rogue_agent_key_hold`) does not
//! expand the forest's bounding box on the raw scalar, but on the
//! shingled vector the anomalous subsequence sits far from the
//! baseline subsequences in the `D`-dim shingle space.
//!
//! Matches the shape of AWS Java's `RotateShingle` (random cut forest
//! with internal ring buffer). This module is the RCF-side fix for
//! the `rogue_agent_key_hold` = 0.145 / `SWaT` = 0.282 failures
//! documented in `docs/performance.md`.
//!
//! # Build
//!
//! ```ignore
//! use anomstream_rs::ShingledForestBuilder;
//!
//! let mut forest = ShingledForestBuilder::<32>::new()
//!     .num_trees(100)
//!     .sample_size(256)
//!     .seed(2026)
//!     .build()?;
//!
//! for sample in stream_of_scalars {
//!     if forest.update_scalar(sample)? {
//!         let score = forest.score_scalar(sample)?;
//!         if f64::from(score) > 1.5 { eprintln!("contextual anomaly"); }
//!     }
//! }
//! ```
//!
//! # Shingled embedding shape
//!
//! For shingle size `D`, the emitted vector is
//! `[v_{t-D+1}, …, v_{t-1}, v_t]` — oldest-first, newest-last. The
//! ring buffer pre-loads on the first `D - 1` scalars; `update_scalar`
//! returns `false` during warm-up and `true` once the forest received
//! its first sample.
//!
//! # When to z-score
//!
//! Scaling is the caller's job. For NDR feature dims with wildly
//! different magnitudes (packet-rate, entropy, port-count), z-score
//! each scalar against its warm-phase `(mean, stddev)` **before**
//! handing it to [`ShingledForest::update_scalar`] — RCF cuts are
//! range-weighted, un-normalised scalars let whichever dim carries
//! the biggest range dominate every cut.

#![cfg(feature = "std")]

use crate::domain::{AnomalyScore, DiVector};
use crate::error::{RcfError, RcfResult};
use crate::forest::RandomCutForest;
use crate::{ForestBuilder, RcfConfig};

/// Builder producing a [`ShingledForest`]. Delegates every RCF
/// hyperparameter to [`ForestBuilder`] — the only extra is the
/// compile-time shingle size which equals the forest
/// dimensionality `D`.
///
/// The const-generic `D` **is** the shingle size: one shingled
/// vector = last `D` scalars.
#[derive(Debug)]
pub struct ShingledForestBuilder<const D: usize> {
    /// Underlying bare-forest builder — full passthrough of every
    /// tuning knob.
    inner: ForestBuilder<D>,
}

impl<const D: usize> Default for ShingledForestBuilder<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> ShingledForestBuilder<D> {
    /// Start a fresh builder with the bare-forest defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: ForestBuilder::<D>::new(),
        }
    }

    /// Number of trees — forwarded to [`ForestBuilder::num_trees`].
    #[must_use]
    pub fn num_trees(mut self, trees: usize) -> Self {
        self.inner = self.inner.num_trees(trees);
        self
    }

    /// Sample size — forwarded to [`ForestBuilder::sample_size`].
    #[must_use]
    pub fn sample_size(mut self, sample: usize) -> Self {
        self.inner = self.inner.sample_size(sample);
        self
    }

    /// Master seed — forwarded to [`ForestBuilder::seed`].
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.inner = self.inner.seed(seed);
        self
    }

    /// Time-decay — forwarded to [`ForestBuilder::time_decay`].
    #[must_use]
    pub fn time_decay(mut self, decay: f64) -> Self {
        self.inner = self.inner.time_decay(decay);
        self
    }

    /// Fetch the resolved [`RcfConfig`] that [`Self::build`] would
    /// use — mirrors [`ForestBuilder::config`].
    #[must_use]
    pub fn config(&self) -> &RcfConfig {
        self.inner.config()
    }

    /// Build the shingled forest. Fails exactly when the underlying
    /// [`ForestBuilder::build`] fails.
    ///
    /// # Errors
    ///
    /// Propagates [`ForestBuilder::build`] errors.
    pub fn build(self) -> RcfResult<ShingledForest<D>> {
        let forest = self.inner.build()?;
        Ok(ShingledForest {
            forest,
            ring: [0.0_f64; D],
            filled: 0,
            cursor: 0,
            warmed: false,
        })
    }
}

/// `D`-dim shingled wrapper over [`RandomCutForest`] — scalar-stream
/// input, internal ring buffer of the last `D` samples.
///
/// The ring buffer is stored oldest-to-newest logically but laid out
/// as a **circular array** internally — constant-time update with no
/// allocation. [`ShingledForest::current_shingle`] exposes the
/// logical shingle in read-only form (oldest-first) for diagnostics.
pub struct ShingledForest<const D: usize> {
    /// Wrapped bare forest operating on shingled `[f64; D]` points.
    forest: RandomCutForest<D>,
    /// Circular storage for the last `D` scalars. `cursor` points
    /// to the slot that will be overwritten on the next update.
    ring: [f64; D],
    /// Scalars received since construction / last `reset` — saturates
    /// at `D`, used by [`Self::is_warmed`].
    filled: usize,
    /// Next write position in `ring`.
    cursor: usize,
    /// `true` once at least one full shingle has been submitted to
    /// the forest.
    warmed: bool,
}

impl<const D: usize> core::fmt::Debug for ShingledForest<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShingledForest")
            .field("shingle_size", &D)
            .field("filled", &self.filled)
            .field("warmed", &self.warmed)
            .finish_non_exhaustive()
    }
}

impl<const D: usize> ShingledForest<D> {
    /// Shingle size (equals the compile-time `D`).
    #[must_use]
    pub const fn shingle_size(&self) -> usize {
        D
    }

    /// Whether the ring buffer holds a full `D`-scalar window and
    /// the forest has received at least one shingle.
    #[must_use]
    pub const fn is_warmed(&self) -> bool {
        self.warmed
    }

    /// Immutable view of the underlying bare forest — use this to
    /// inspect tree state, read metrics, or route through the
    /// [`RandomCutForest::forensic_baseline`] / `attribution`
    /// helpers on the already-shingled last point.
    #[must_use]
    pub fn forest(&self) -> &RandomCutForest<D> {
        &self.forest
    }

    /// Mutable escape hatch — handy for bootstrap replay
    /// ([`RandomCutForest::bootstrap`]) when the caller has
    /// pre-shingled their warm-up corpus.
    pub fn forest_mut(&mut self) -> &mut RandomCutForest<D> {
        &mut self.forest
    }

    /// Snapshot the current shingle in logical order (oldest-first).
    /// Returns `None` while the ring is still partially empty.
    #[must_use]
    pub fn current_shingle(&self) -> Option<[f64; D]> {
        if self.filled < D {
            return None;
        }
        Some(self.materialise_shingle())
    }

    /// Fold `value` into the ring buffer; once the ring is full,
    /// forward the shingled window to the forest. Returns `true`
    /// when the shingle was submitted to the forest (i.e. the ring
    /// was full before the call).
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] on non-finite `value`.
    /// - Propagates [`RandomCutForest::update`] failures once the
    ///   shingle is submitted.
    pub fn update_scalar(&mut self, value: f64) -> RcfResult<bool> {
        if !value.is_finite() {
            return Err(RcfError::NaNValue);
        }
        // Submit the *previous* shingle before rotating the ring —
        // the new scalar becomes the newest entry of the shingle
        // seen by the forest next call.
        let submitted = if self.filled >= D {
            let shingle = self.materialise_shingle();
            self.forest.update(shingle)?;
            self.warmed = true;
            true
        } else {
            false
        };
        // Rotate the ring.
        self.ring[self.cursor] = value;
        self.cursor = (self.cursor + 1) % D;
        if self.filled < D {
            self.filled += 1;
        }
        Ok(submitted)
    }

    /// Score `value` against the frozen forest **without** folding
    /// it into the ring buffer. The query uses the current shingle
    /// with `value` appended as the newest slot — matches what a
    /// subsequent [`Self::update_scalar`] would submit.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] on non-finite `value`.
    /// - [`RcfError::EmptyForest`] before the ring buffer is full
    ///   or the forest has not yet received its first update.
    /// - Propagates [`RandomCutForest::score`] failures.
    pub fn score_scalar(&self, value: f64) -> RcfResult<AnomalyScore> {
        if !value.is_finite() {
            return Err(RcfError::NaNValue);
        }
        let shingle = self.shingle_with(value)?;
        self.forest.score(&shingle)
    }

    /// Attribution on the shingle formed by appending `value` to
    /// the current ring. Returns a `D`-dim [`DiVector`] where each
    /// dim is a **lag index** (0 = oldest, `D-1` = newest / `value`).
    ///
    /// # Errors
    ///
    /// Same as [`Self::score_scalar`].
    pub fn attribution_scalar(&self, value: f64) -> RcfResult<DiVector> {
        if !value.is_finite() {
            return Err(RcfError::NaNValue);
        }
        let shingle = self.shingle_with(value)?;
        self.forest.attribution(&shingle)
    }

    /// Stateless codisp on the shingle formed with `value` appended.
    /// Non-mutating — preserves the frozen-baseline contract across
    /// long streams. Prefer this over the mutating `score_codisp`
    /// path for shingled forensic replay.
    ///
    /// # Errors
    ///
    /// Same as [`Self::score_scalar`].
    pub fn score_codisp_stateless_scalar(&self, value: f64) -> RcfResult<AnomalyScore> {
        if !value.is_finite() {
            return Err(RcfError::NaNValue);
        }
        let shingle = self.shingle_with(value)?;
        self.forest.score_codisp_stateless(&shingle)
    }

    /// Drop the ring buffer and reset the warm-up flag; the
    /// underlying forest is **not** reset — callers who want a
    /// full state wipe should rebuild.
    pub fn reset_ring(&mut self) {
        self.ring = [0.0_f64; D];
        self.filled = 0;
        self.cursor = 0;
        self.warmed = false;
    }

    /// Logical oldest-first materialisation of the ring.
    fn materialise_shingle(&self) -> [f64; D] {
        let mut out = [0.0_f64; D];
        // `cursor` points at the slot that will be overwritten
        // next, which equals the position of the *oldest* entry
        // when the ring is full.
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.ring[(self.cursor + i) % D];
        }
        out
    }

    /// Build the shingle that would result from appending `value`
    /// to the current ring, without mutating the ring.
    fn shingle_with(&self, value: f64) -> RcfResult<[f64; D]> {
        if self.filled < D {
            return Err(RcfError::EmptyForest);
        }
        let mut out = [0.0_f64; D];
        // Drop the oldest entry (at `cursor`), shift the rest left
        // by one, append `value` as newest.
        for (i, slot) in out.iter_mut().enumerate().take(D - 1) {
            *slot = self.ring[(self.cursor + 1 + i) % D];
        }
        out[D - 1] = value;
        Ok(out)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_lossless,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    fn small() -> ShingledForest<4> {
        ShingledForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .seed(2026)
            .build()
            .unwrap()
    }

    #[test]
    fn warm_up_requires_d_scalars() {
        let mut f = small();
        for i in 0..3 {
            let submitted = f.update_scalar(i as f64).unwrap();
            assert!(!submitted, "shouldn't submit before ring is full");
            assert!(!f.is_warmed());
        }
        // 4th scalar fills the ring but the next update is when the
        // first *shingle* lands in the forest.
        let submitted = f.update_scalar(3.0).unwrap();
        assert!(!submitted);
        assert_eq!(f.current_shingle(), Some([0.0, 1.0, 2.0, 3.0]));
        // 5th scalar — now previous shingle [0,1,2,3] gets submitted.
        let submitted = f.update_scalar(4.0).unwrap();
        assert!(submitted);
        assert!(f.is_warmed());
        assert_eq!(f.current_shingle(), Some([1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn update_scalar_rejects_nan() {
        let mut f = small();
        assert!(matches!(
            f.update_scalar(f64::NAN).unwrap_err(),
            RcfError::NaNValue
        ));
        assert!(matches!(
            f.update_scalar(f64::INFINITY).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn score_before_warm_fails() {
        let f = small();
        assert!(matches!(
            f.score_scalar(1.0).unwrap_err(),
            RcfError::EmptyForest
        ));
    }

    #[test]
    fn score_after_warm_returns_non_negative() {
        let mut f = small();
        for i in 0..200 {
            let _ = f.update_scalar(i as f64 * 0.01).unwrap();
        }
        let s: f64 = f.score_scalar(10.0).unwrap().into();
        assert!(s.is_finite());
        assert!(s >= 0.0);
    }

    #[test]
    fn outlier_scalar_scores_higher_than_in_cluster() {
        let mut f = ShingledForestBuilder::<8>::new()
            .num_trees(100)
            .sample_size(128)
            .seed(7)
            .build()
            .unwrap();
        // Warm on a tight cluster.
        let mut tick = 0.0_f64;
        for _ in 0..1_000 {
            let _ = f.update_scalar((tick.sin() + 1.0) * 0.1).unwrap();
            tick += 0.1;
        }
        // In-cluster probe.
        let normal: f64 = f.score_scalar(0.10).unwrap().into();
        // Outlier probe.
        let outlier: f64 = f.score_scalar(100.0).unwrap().into();
        assert!(
            outlier > normal,
            "outlier {outlier} should exceed in-cluster {normal}"
        );
    }

    #[test]
    fn shingle_with_does_not_mutate_ring() {
        let mut f = small();
        for i in 0..5 {
            let _ = f.update_scalar(i as f64).unwrap();
        }
        let before = f.current_shingle().unwrap();
        let _ = f.score_scalar(99.0).unwrap();
        let after = f.current_shingle().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn reset_ring_clears_warm_state_but_preserves_forest() {
        let mut f = small();
        for i in 0..10 {
            let _ = f.update_scalar(i as f64).unwrap();
        }
        assert!(f.is_warmed());
        f.reset_ring();
        assert!(!f.is_warmed());
        assert_eq!(f.current_shingle(), None);
        // Forest still holds its leaves — a fresh shingle submission
        // after re-warming should score against the prior baseline.
        for i in 0..10 {
            let _ = f.update_scalar(i as f64).unwrap();
        }
        let s: f64 = f.score_scalar(100.0).unwrap().into();
        assert!(s.is_finite());
    }

    #[test]
    fn codisp_stateless_on_shingle_matches_bare_forest() {
        let mut f = small();
        for i in 0..50 {
            let _ = f.update_scalar(i as f64 * 0.01).unwrap();
        }
        let scalar_codisp: f64 = f.score_codisp_stateless_scalar(5.0).unwrap().into();
        let shingle = f.shingle_with(5.0).unwrap();
        let direct: f64 = f.forest().score_codisp_stateless(&shingle).unwrap().into();
        assert!((scalar_codisp - direct).abs() < 1.0e-12);
    }
}
