//! Per-feature min-max / z-score normalizer.
//!
//! Rescales a `D`-dimensional point into `[0, 1]` (`MinMax`) or
//! `(value − μ) / σ` (`ZScore`) given per-dimension
//! [`NormParams`] learned from a batch or loaded from a saved
//! baseline. The `None` strategy is an identity transform kept
//! so the normalizer can be inserted into a detection pipeline
//! even before fit data arrives.
//!
//! Policy-free — the lib stores params and applies the math;
//! caller decides when to refit, when to swap, and what the
//! source of the samples is.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// Per-dimension learned parameters.
///
/// `identity()` yields `{mean=0, std_dev=1, min=0, max=1}` —
/// applied under [`NormStrategy::ZScore`] or
/// [`NormStrategy::MinMax`] the transform reduces to the input
/// unchanged when the value lies inside `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NormParams {
    /// Mean for z-score normalisation.
    pub mean: f64,
    /// Standard deviation for z-score normalisation.
    pub std_dev: f64,
    /// Minimum observed value for min-max normalisation.
    pub min: f64,
    /// Maximum observed value for min-max normalisation.
    pub max: f64,
}

impl NormParams {
    /// Identity parameters — pass-through transform.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
            min: 0.0,
            max: 1.0,
        }
    }
}

impl Default for NormParams {
    fn default() -> Self {
        Self::identity()
    }
}

/// Per-dim transform strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NormStrategy {
    /// Rescale to `[0, 1]` using `min` / `max` bounds.
    MinMax,
    /// Centre and scale using `mean` / `std_dev`.
    ZScore,
    /// Pass-through.
    None,
}

/// Fixed-dim normalizer with per-feature parameters.
///
/// # Examples
///
/// ```
/// use anomstream_rs::{NormStrategy, Normalizer};
///
/// let samples = [[0.0, 0.0], [100.0, 1000.0]];
/// let n = Normalizer::<2>::fit(NormStrategy::MinMax, &samples);
/// let out = n.transform(&[50.0, 500.0]);
/// assert!((out[0] - 0.5).abs() < 1e-12);
/// assert!((out[1] - 0.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Normalizer<const D: usize> {
    /// Active transform strategy.
    pub strategy: NormStrategy,
    /// Per-dimension learned parameters.
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub params: [NormParams; D],
}

impl<const D: usize> Normalizer<D> {
    /// Build with identity parameters. Use when no fit data is
    /// yet available; transform is then a pass-through (or a
    /// clamp-to-`[0,1]` under `MinMax`).
    #[must_use]
    pub const fn identity(strategy: NormStrategy) -> Self {
        Self {
            strategy,
            params: [NormParams::identity(); D],
        }
    }

    /// Learn per-dim `(mean, std_dev, min, max)` from `samples`.
    /// Returns [`Self::identity`] when `samples` is empty.
    ///
    /// Uses the same two-pass formulation as the original
    /// enterprise baseline so numeric results are stable across
    /// the migration.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fit(strategy: NormStrategy, samples: &[[f64; D]]) -> Self {
        if samples.is_empty() {
            return Self::identity(strategy);
        }

        let n = samples.len() as f64;
        let mut params = [NormParams {
            mean: 0.0,
            std_dev: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }; D];

        for sample in samples {
            for (i, &value) in sample.iter().enumerate() {
                params[i].mean += value;
                if value < params[i].min {
                    params[i].min = value;
                }
                if value > params[i].max {
                    params[i].max = value;
                }
            }
        }

        for p in &mut params {
            p.mean /= n;
        }

        for sample in samples {
            for (i, &value) in sample.iter().enumerate() {
                let diff = value - params[i].mean;
                params[i].std_dev += diff * diff;
            }
        }

        for p in &mut params {
            p.std_dev = (p.std_dev / n).sqrt();
        }

        Self { strategy, params }
    }

    /// Apply the active strategy to `input`, returning a new
    /// `[f64; D]`. `MinMax` clamps to `[0, 1]`; `ZScore` divides
    /// by `std_dev` with zero-guard.
    #[must_use]
    pub fn transform(&self, input: &[f64; D]) -> [f64; D] {
        let mut out = [0.0_f64; D];
        for (i, &value) in input.iter().enumerate() {
            let p = &self.params[i];
            out[i] = match self.strategy {
                NormStrategy::MinMax => {
                    let range = p.max - p.min;
                    if range.abs() < f64::EPSILON {
                        0.5
                    } else {
                        ((value - p.min) / range).clamp(0.0, 1.0)
                    }
                }
                NormStrategy::ZScore => {
                    if p.std_dev.abs() < f64::EPSILON {
                        0.0
                    } else {
                        (value - p.mean) / p.std_dev
                    }
                }
                NormStrategy::None => value,
            };
        }
        out
    }
}

#[cfg(feature = "serde")]
mod serde_arrays {
    //! `serde` adapter for `[NormParams; D]` — default serde
    //! derive works only up to `[T; 32]` for arbitrary `T`, so
    //! pass through a length-prefixed slice.
    use super::NormParams;
    use alloc::vec::Vec;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Serialize `[NormParams; D]` as a length-prefixed slice.
    pub fn serialize<S: Serializer, const D: usize>(
        params: &[NormParams; D],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        params.as_slice().serialize(s)
    }

    /// Deserialize a length-prefixed slice back into `[NormParams; D]`.
    pub fn deserialize<'de, DSer: Deserializer<'de>, const D: usize>(
        d: DSer,
    ) -> Result<[NormParams; D], DSer::Error> {
        let v: Vec<NormParams> = Vec::deserialize(d)?;
        if v.len() != D {
            return Err(serde::de::Error::invalid_length(
                v.len(),
                &"expected D entries",
            ));
        }
        let mut out = [NormParams::identity(); D];
        for (slot, p) in out.iter_mut().zip(v) {
            *slot = p;
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn identity_passthrough_under_none() {
        let n = Normalizer::<3>::identity(NormStrategy::None);
        let out = n.transform(&[10.0, 5000.0, -2.5]);
        assert_eq!(out, [10.0, 5000.0, -2.5]);
    }

    #[test]
    fn minmax_rescales_mid_point() {
        let mut n = Normalizer::<1>::identity(NormStrategy::MinMax);
        n.params[0] = NormParams {
            min: 0.0,
            max: 100.0,
            mean: 0.0,
            std_dev: 1.0,
        };
        let out = n.transform(&[50.0]);
        assert!((out[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn minmax_clamps_above_max() {
        let mut n = Normalizer::<1>::identity(NormStrategy::MinMax);
        n.params[0] = NormParams {
            min: 0.0,
            max: 10.0,
            mean: 0.0,
            std_dev: 1.0,
        };
        let out = n.transform(&[20.0]);
        assert!((out[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn minmax_clamps_below_min() {
        let mut n = Normalizer::<1>::identity(NormStrategy::MinMax);
        n.params[0] = NormParams {
            min: 0.0,
            max: 10.0,
            mean: 0.0,
            std_dev: 1.0,
        };
        let out = n.transform(&[-5.0]);
        assert!((out[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn minmax_zero_range_returns_mid() {
        let mut n = Normalizer::<1>::identity(NormStrategy::MinMax);
        n.params[0] = NormParams {
            min: 5.0,
            max: 5.0,
            mean: 5.0,
            std_dev: 0.0,
        };
        let out = n.transform(&[5.0]);
        assert!((out[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn zscore_centers_and_scales() {
        let mut n = Normalizer::<1>::identity(NormStrategy::ZScore);
        n.params[0] = NormParams {
            mean: 50.0,
            std_dev: 10.0,
            min: 0.0,
            max: 100.0,
        };
        let out = n.transform(&[70.0]);
        assert!((out[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn zscore_zero_std_returns_zero() {
        let mut n = Normalizer::<1>::identity(NormStrategy::ZScore);
        n.params[0] = NormParams {
            mean: 5.0,
            std_dev: 0.0,
            min: 5.0,
            max: 5.0,
        };
        let out = n.transform(&[10.0]);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn fit_learns_mean_min_max() {
        let samples = [[10.0, 1000.0], [20.0, 2000.0], [30.0, 3000.0]];
        let n = Normalizer::<2>::fit(NormStrategy::MinMax, &samples);
        assert!((n.params[0].min - 10.0).abs() < 1e-12);
        assert!((n.params[0].max - 30.0).abs() < 1e-12);
        assert!((n.params[0].mean - 20.0).abs() < 1e-12);
        assert!((n.params[1].mean - 2000.0).abs() < 1e-12);
    }

    #[test]
    fn fit_then_transform_rescales_correctly() {
        let samples = [[0.0, 0.0], [100.0, 1000.0]];
        let n = Normalizer::<2>::fit(NormStrategy::MinMax, &samples);
        let out = n.transform(&[50.0, 500.0]);
        assert!((out[0] - 0.5).abs() < 1e-12);
        assert!((out[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn fit_empty_falls_back_to_identity() {
        let n: Normalizer<5> = Normalizer::fit(NormStrategy::ZScore, &[]);
        for p in &n.params {
            assert_eq!(p.mean, 0.0);
            assert_eq!(p.std_dev, 1.0);
        }
    }

    #[test]
    fn fit_single_sample_has_zero_std() {
        let samples = [[5.0_f64]];
        let n = Normalizer::<1>::fit(NormStrategy::ZScore, &samples);
        assert_eq!(n.params[0].mean, 5.0);
        assert_eq!(n.params[0].std_dev, 0.0);
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn postcard_roundtrip_preserves_transform() {
        let samples = [[0.0_f64, 0.0], [100.0, 1000.0]];
        let n = Normalizer::<2>::fit(NormStrategy::MinMax, &samples);
        let bytes = postcard::to_allocvec(&n).expect("serde ok");
        let back: Normalizer<2> = postcard::from_bytes(&bytes).expect("serde ok");
        let probe = [50.0, 500.0];
        assert_eq!(back.transform(&probe), n.transform(&probe));
    }
}
