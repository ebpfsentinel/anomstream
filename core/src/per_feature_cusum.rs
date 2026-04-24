//! Per-feature two-sided CUSUM change-point detector.
//!
//! `D` parallel univariate CUSUMs track positive and negative
//! cumulative sums of the deviation from a reference mean.
//! Alerts when either side exceeds the threshold `h` — detects
//! *sustained* mean shifts that an `EWMA` adapts to and stops
//! reporting (e.g. slow-ramp `DDoS`, gradual leak).
//!
//! Complementary to [`crate::meta_drift::MetaDriftDetector`]
//! (scalar CUSUM on the RCF score stream): this module is
//! per-feature CUSUM on raw observations, so caller can answer
//! *which* feature drifted and in which direction.
//!
//! # CUSUM recurrence
//!
//! ```text
//! S+ ← max(0, S+ + (x − μ₀ − k))
//! S− ← max(0, S− − (x − μ₀ + k))
//! alert when S+ > h  (increase)  or  S− > h  (decrease)
//! ```
//!
//! `k` is the slack (allowable drift), `h` is the threshold,
//! `μ₀` is the reference mean (auto-learned on the first
//! observation unless overridden via [`PerFeatureCusum::set_reference`]).
//!
//! # References
//!
//! 1. E. S. Page, "Continuous Inspection Schemes",
//!    *Biometrika* 41, 1954.
//! 2. D. M. Hawkins & D. H. Olwell, *Cumulative Sum Charts and
//!    Charting for Quality Improvement*, Springer, 1998.

use alloc::vec::Vec;

/// Direction of a detected change-point drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DriftDirection {
    /// Sustained increase above the reference mean.
    Increase,
    /// Sustained decrease below the reference mean.
    Decrease,
}

/// One CUSUM alert — fired when a feature's positive or
/// negative cumulative sum exceeds the threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerFeatureCusumAlert {
    /// Feature index that tripped (0-based into the observation
    /// array).
    pub feature_index: usize,
    /// Which side of the two-sided chart fired.
    pub direction: DriftDirection,
    /// `max(S+, S−)` at the moment of the alert.
    pub magnitude: f64,
    /// Consecutive samples the drift has been building.
    pub duration_samples: u64,
}

/// One univariate two-sided CUSUM accumulator.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(try_from = "PerFeatureCusumAccumulatorShadow")
)]
pub struct PerFeatureCusumAccumulator {
    /// Positive cumulative sum (detects increases).
    pub s_pos: f64,
    /// Negative cumulative sum (detects decreases).
    pub s_neg: f64,
    /// Reference mean `μ₀` (auto-learned or caller-set).
    pub reference: f64,
    /// Whether `reference` has been populated.
    pub reference_set: bool,
    /// Consecutive samples the current drift has been
    /// accumulating.
    pub drift_samples: u64,
}

/// Over-the-wire [`PerFeatureCusumAccumulator`] layout.
/// Deserialization lands here first so [`TryFrom`] can reject
/// `NaN` / `±inf` poisoning of the cumulative sums or the
/// reference mean — fields that later feed the `update()`
/// recurrence and would propagate non-finite state indefinitely.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::missing_docs_in_private_items)]
struct PerFeatureCusumAccumulatorShadow {
    s_pos: f64,
    s_neg: f64,
    reference: f64,
    reference_set: bool,
    drift_samples: u64,
}

#[cfg(feature = "serde")]
impl TryFrom<PerFeatureCusumAccumulatorShadow> for PerFeatureCusumAccumulator {
    type Error = crate::error::RcfError;

    fn try_from(raw: PerFeatureCusumAccumulatorShadow) -> Result<Self, Self::Error> {
        if !raw.s_pos.is_finite() || !raw.s_neg.is_finite() || !raw.reference.is_finite() {
            return Err(crate::error::RcfError::InvalidConfig(alloc::format!(
                "PerFeatureCusumAccumulator: non-finite field (s_pos={}, s_neg={}, reference={})",
                raw.s_pos,
                raw.s_neg,
                raw.reference
            ).into()));
        }
        if raw.s_pos < 0.0 || raw.s_neg < 0.0 {
            return Err(crate::error::RcfError::InvalidConfig(alloc::format!(
                "PerFeatureCusumAccumulator: cumulative sums must be non-negative (s_pos={}, s_neg={})",
                raw.s_pos,
                raw.s_neg
            ).into()));
        }
        Ok(Self {
            s_pos: raw.s_pos,
            s_neg: raw.s_neg,
            reference: raw.reference,
            reference_set: raw.reference_set,
            drift_samples: raw.drift_samples,
        })
    }
}

impl PerFeatureCusumAccumulator {
    /// Fresh accumulator — zeroed, reference unset.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            s_pos: 0.0,
            s_neg: 0.0,
            reference: 0.0,
            reference_set: false,
            drift_samples: 0,
        }
    }

    /// Reset to the zero state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Current magnitude — `max(S+, S−)`. Used by the caller
    /// to report a per-feature score even when no alert fired.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.s_pos.max(self.s_neg)
    }

    /// Ingest `value` and return an alert when either side
    /// exceeds `threshold`. First call seeds `reference = value`
    /// and returns `None` unconditionally.
    pub fn update(
        &mut self,
        value: f64,
        slack: f64,
        threshold: f64,
        feature_index: usize,
    ) -> Option<PerFeatureCusumAlert> {
        if !self.reference_set {
            self.reference = value;
            self.reference_set = true;
            return None;
        }

        self.s_pos = (self.s_pos + (value - self.reference - slack)).max(0.0);
        self.s_neg = (self.s_neg - (value - self.reference + slack)).max(0.0);

        if self.s_pos > threshold || self.s_neg > threshold {
            self.drift_samples += 1;
            let (direction, magnitude) = if self.s_pos > self.s_neg {
                (DriftDirection::Increase, self.s_pos)
            } else {
                (DriftDirection::Decrease, self.s_neg)
            };
            Some(PerFeatureCusumAlert {
                feature_index,
                direction,
                magnitude,
                duration_samples: self.drift_samples,
            })
        } else {
            self.drift_samples = 0;
            None
        }
    }
}

impl Default for PerFeatureCusumAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Hyper-parameters for [`PerFeatureCusum`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "PerFeatureCusumConfigShadow"))]
pub struct PerFeatureCusumConfig {
    /// Slack `k` — allowable drift before accumulation starts.
    /// Typical `0.5·σ` of the reference signal.
    pub slack: f64,
    /// Threshold `h` — cumulative sum at which an alert fires.
    /// Typical `4·σ` of the reference signal.
    pub threshold: f64,
}

/// Over-the-wire [`PerFeatureCusumConfig`] layout.
/// Deserialization lands here first so [`TryFrom`] can enforce
/// finite, non-negative `slack` and strictly-positive `threshold`
/// — an attacker-supplied `threshold ≤ 0` would make every
/// observation emit an alert.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::missing_docs_in_private_items)]
struct PerFeatureCusumConfigShadow {
    slack: f64,
    threshold: f64,
}

#[cfg(feature = "serde")]
impl TryFrom<PerFeatureCusumConfigShadow> for PerFeatureCusumConfig {
    type Error = crate::error::RcfError;

    fn try_from(raw: PerFeatureCusumConfigShadow) -> Result<Self, Self::Error> {
        if !raw.slack.is_finite() || raw.slack < 0.0 {
            return Err(crate::error::RcfError::InvalidConfig(
                alloc::format!(
                    "PerFeatureCusumConfig: slack must be finite and ≥ 0, got {}",
                    raw.slack
                )
                .into(),
            ));
        }
        if !raw.threshold.is_finite() || raw.threshold <= 0.0 {
            return Err(crate::error::RcfError::InvalidConfig(
                alloc::format!(
                    "PerFeatureCusumConfig: threshold must be finite and > 0, got {}",
                    raw.threshold
                )
                .into(),
            ));
        }
        Ok(Self {
            slack: raw.slack,
            threshold: raw.threshold,
        })
    }
}

impl Default for PerFeatureCusumConfig {
    fn default() -> Self {
        Self {
            slack: 0.5,
            threshold: 5.0,
        }
    }
}

/// Result of one [`PerFeatureCusum::observe`] call.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerFeatureCusumResult<const D: usize> {
    /// `max(S+, S−)` per feature at the moment the observation
    /// returned — includes the current update.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    pub per_feature_magnitude: [f64; D],
    /// `max(per_feature_magnitude)` — single-number summary.
    pub max_magnitude: f64,
    /// Alerts fired this tick (one per feature that exceeded
    /// `threshold`).
    pub alerts: Vec<PerFeatureCusumAlert>,
}

/// `D` parallel two-sided CUSUMs sharing one `(slack, threshold)`
/// configuration.
///
/// # Examples
///
/// ```
/// use anomstream_core::{PerFeatureCusum, PerFeatureCusumConfig};
///
/// let mut det = PerFeatureCusum::<2>::new(PerFeatureCusumConfig {
///     slack: 0.5,
///     threshold: 5.0,
/// });
/// det.observe(&[100.0, 200.0]); // seeds references
/// for _ in 0..20 {
///     det.observe(&[105.0, 200.0]);
/// }
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerFeatureCusum<const D: usize> {
    /// Per-dimension accumulator state.
    #[cfg_attr(feature = "serde", serde(with = "serde_accumulators"))]
    accumulators: [PerFeatureCusumAccumulator; D],
    /// Active configuration.
    config: PerFeatureCusumConfig,
    /// Observations ingested so far.
    total_samples: u64,
}

impl<const D: usize> PerFeatureCusum<D> {
    /// Build an empty detector.
    #[must_use]
    pub const fn new(config: PerFeatureCusumConfig) -> Self {
        Self {
            accumulators: [PerFeatureCusumAccumulator::new(); D],
            config,
            total_samples: 0,
        }
    }

    /// Active configuration.
    #[must_use]
    pub const fn config(&self) -> &PerFeatureCusumConfig {
        &self.config
    }

    /// Observations ingested so far.
    #[must_use]
    pub const fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Per-dimension accumulator snapshot (read-only).
    #[must_use]
    pub const fn accumulators(&self) -> &[PerFeatureCusumAccumulator; D] {
        &self.accumulators
    }

    /// Count of features currently in an active drift
    /// (`drift_samples > 0`).
    #[must_use]
    pub fn active_drifts(&self) -> usize {
        self.accumulators
            .iter()
            .filter(|a| a.drift_samples > 0)
            .count()
    }

    /// Override the auto-learned reference mean per dimension.
    /// Useful when feeding a stable external baseline (e.g. an
    /// EWMA mean) rather than the first observation.
    pub fn set_reference(&mut self, means: &[f64; D]) {
        for (acc, &mean) in self.accumulators.iter_mut().zip(means.iter()) {
            acc.reference = mean;
            acc.reference_set = true;
        }
    }

    /// Ingest `input`, returning per-feature magnitudes and any
    /// alerts that fired. Always updates the accumulators.
    #[must_use = "detector output should be checked — dropping it silently usually indicates a logic bug"]
    pub fn observe(&mut self, input: &[f64; D]) -> PerFeatureCusumResult<D> {
        let mut per_feature_magnitude = [0.0_f64; D];
        let mut alerts: Vec<PerFeatureCusumAlert> = Vec::new();

        for (i, &value) in input.iter().enumerate() {
            let pre_magnitude = self.accumulators[i].magnitude();
            per_feature_magnitude[i] = pre_magnitude;

            if let Some(alert) =
                self.accumulators[i].update(value, self.config.slack, self.config.threshold, i)
            {
                per_feature_magnitude[i] = alert.magnitude;
                alerts.push(alert);
            }
        }

        self.total_samples += 1;
        let max_magnitude = per_feature_magnitude
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);

        PerFeatureCusumResult {
            per_feature_magnitude,
            max_magnitude,
            alerts,
        }
    }

    /// Zero every accumulator and the sample counter.
    pub fn reset(&mut self) {
        for acc in &mut self.accumulators {
            acc.reset();
        }
        self.total_samples = 0;
    }
}

#[cfg(feature = "serde")]
mod serde_accumulators {
    //! `serde` adapter for `[PerFeatureCusumAccumulator; D]` —
    //! derive macro does not cover arbitrary-`D` arrays.
    use super::PerFeatureCusumAccumulator;
    use alloc::vec::Vec;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Serialize `[PerFeatureCusumAccumulator; D]` as a length-prefixed slice.
    pub fn serialize<S: Serializer, const D: usize>(
        accs: &[PerFeatureCusumAccumulator; D],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        accs.as_slice().serialize(s)
    }

    /// Deserialize a length-prefixed slice back into `[PerFeatureCusumAccumulator; D]`.
    pub fn deserialize<'de, DSer: Deserializer<'de>, const D: usize>(
        d: DSer,
    ) -> Result<[PerFeatureCusumAccumulator; D], DSer::Error> {
        let v: Vec<PerFeatureCusumAccumulator> = Vec::deserialize(d)?;
        if v.len() != D {
            return Err(serde::de::Error::invalid_length(
                v.len(),
                &"expected D accumulators",
            ));
        }
        let mut out = [PerFeatureCusumAccumulator::new(); D];
        for (slot, acc) in out.iter_mut().zip(v) {
            *slot = acc;
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn first_observation_seeds_reference() {
        let mut det = PerFeatureCusum::<1>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        let out = det.observe(&[100.0]);
        assert!(out.alerts.is_empty());
        assert!(det.accumulators()[0].reference_set);
        assert_eq!(det.accumulators()[0].reference, 100.0);
    }

    #[test]
    fn no_alert_on_stable_signal() {
        let mut det = PerFeatureCusum::<1>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        for _ in 0..100 {
            let out = det.observe(&[100.0]);
            assert!(out.alerts.is_empty());
        }
    }

    #[test]
    fn detects_upward_ramp() {
        let mut det = PerFeatureCusum::<1>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        let _ = det.observe(&[100.0]);
        let mut alerted = false;
        for _ in 0..20 {
            let out = det.observe(&[105.0]);
            if let Some(alert) = out.alerts.first() {
                assert_eq!(alert.direction, DriftDirection::Increase);
                assert_eq!(alert.feature_index, 0);
                alerted = true;
                break;
            }
        }
        assert!(alerted);
    }

    #[test]
    fn detects_downward_ramp() {
        let mut det = PerFeatureCusum::<1>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        let _ = det.observe(&[100.0]);
        let mut alerted = false;
        for _ in 0..20 {
            let out = det.observe(&[95.0]);
            if let Some(alert) = out.alerts.first() {
                assert_eq!(alert.direction, DriftDirection::Decrease);
                alerted = true;
                break;
            }
        }
        assert!(alerted);
    }

    #[test]
    fn drift_samples_counter_grows_then_resets() {
        let mut det = PerFeatureCusum::<1>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        let _ = det.observe(&[100.0]);
        for _ in 0..20 {
            let _ = det.observe(&[105.0]);
        }
        assert!(det.accumulators()[0].drift_samples > 0);
        // Return to reference — S+ decays by `slack` per tick.
        // 20 steps of +4.5 each ≈ S+=90 at trip; 200 steps of
        // −0.5 brings it back below threshold (5).
        for _ in 0..250 {
            let _ = det.observe(&[100.0]);
        }
        assert_eq!(det.accumulators()[0].drift_samples, 0);
    }

    #[test]
    fn set_reference_overrides_auto_learn() {
        let mut det = PerFeatureCusum::<2>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        det.set_reference(&[50.0, 100.0]);
        assert!(det.accumulators()[0].reference_set);
        assert_eq!(det.accumulators()[0].reference, 50.0);

        // Feeding at the reference must not trigger alerts.
        for _ in 0..50 {
            let out = det.observe(&[50.0, 100.0]);
            assert!(out.alerts.is_empty());
        }
    }

    #[test]
    fn max_magnitude_picks_largest_feature() {
        let mut det = PerFeatureCusum::<3>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        let _ = det.observe(&[0.0, 0.0, 0.0]);
        for _ in 0..20 {
            let _ = det.observe(&[0.0, 10.0, 0.0]);
        }
        let out = det.observe(&[0.0, 10.0, 0.0]);
        assert_eq!(out.max_magnitude, out.per_feature_magnitude[1]);
        assert!(out.per_feature_magnitude[1] > out.per_feature_magnitude[0]);
        assert!(out.per_feature_magnitude[1] > out.per_feature_magnitude[2]);
    }

    #[test]
    fn reset_clears_state() {
        let mut det = PerFeatureCusum::<2>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        let _ = det.observe(&[100.0, 200.0]);
        for _ in 0..20 {
            let _ = det.observe(&[110.0, 220.0]);
        }
        assert!(det.active_drifts() > 0);
        det.reset();
        assert_eq!(det.total_samples(), 0);
        assert_eq!(det.active_drifts(), 0);
        for acc in det.accumulators() {
            assert!(!acc.reference_set);
            assert_eq!(acc.s_pos, 0.0);
            assert_eq!(acc.s_neg, 0.0);
        }
    }

    #[test]
    fn active_drifts_counts_per_feature() {
        let mut det = PerFeatureCusum::<2>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        let _ = det.observe(&[100.0, 100.0]);
        for _ in 0..20 {
            let _ = det.observe(&[110.0, 100.0]);
        }
        // Only feature 0 is drifting.
        assert_eq!(det.active_drifts(), 1);
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn postcard_roundtrip_preserves_state() {
        let mut det = PerFeatureCusum::<3>::new(PerFeatureCusumConfig {
            slack: 0.5,
            threshold: 5.0,
        });
        let _ = det.observe(&[100.0, 200.0, 300.0]);
        for _ in 0..10 {
            let _ = det.observe(&[105.0, 200.0, 300.0]);
        }
        let bytes = postcard::to_allocvec(&det).expect("serde ok");
        let back: PerFeatureCusum<3> = postcard::from_bytes(&bytes).expect("serde ok");
        assert_eq!(back.total_samples(), det.total_samples());
        assert_eq!(back.accumulators()[0].s_pos, det.accumulators()[0].s_pos);
        assert_eq!(
            back.accumulators()[0].reference,
            det.accumulators()[0].reference
        );
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn deserialize_rejects_nan_in_accumulator() {
        let bad = PerFeatureCusumAccumulatorShadow {
            s_pos: f64::NAN,
            s_neg: 0.0,
            reference: 0.0,
            reference_set: true,
            drift_samples: 0,
        };
        let bytes = postcard::to_allocvec(&bad).unwrap();
        let back: Result<PerFeatureCusumAccumulator, _> = postcard::from_bytes(&bytes);
        assert!(back.is_err());
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn deserialize_rejects_negative_cumsum() {
        let bad = PerFeatureCusumAccumulatorShadow {
            s_pos: -1.0,
            s_neg: 0.0,
            reference: 0.0,
            reference_set: true,
            drift_samples: 0,
        };
        let bytes = postcard::to_allocvec(&bad).unwrap();
        let back: Result<PerFeatureCusumAccumulator, _> = postcard::from_bytes(&bytes);
        assert!(back.is_err());
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn deserialize_rejects_non_positive_threshold() {
        let bad = PerFeatureCusumConfigShadow {
            slack: 0.5,
            threshold: 0.0,
        };
        let bytes = postcard::to_allocvec(&bad).unwrap();
        let back: Result<PerFeatureCusumConfig, _> = postcard::from_bytes(&bytes);
        assert!(back.is_err());
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn deserialize_rejects_nan_threshold() {
        let bad = PerFeatureCusumConfigShadow {
            slack: 0.5,
            threshold: f64::NAN,
        };
        let bytes = postcard::to_allocvec(&bad).unwrap();
        let back: Result<PerFeatureCusumConfig, _> = postcard::from_bytes(&bytes);
        assert!(back.is_err());
    }
}
