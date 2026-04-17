//! Forest configuration and the [`ForestBuilder`] entry point.
//!
//! [`RcfConfig`] enforces the AWS `SageMaker` Random Cut Forest
//! hyperparameter bounds at validation time. Callers should construct
//! a forest through [`ForestBuilder`] rather than instantiating
//! [`RcfConfig`] directly so the builder picks AWS-conformant defaults.
//!
//! Per-point dimensionality is encoded at the type level as the
//! `D` const-generic on [`ForestBuilder`] / [`crate::RandomCutForest`]
//! so the bounding-box and per-tree node storage live on the stack
//! and the compiler can vectorise the hot tree-traversal loops.

use crate::error::{RcfError, RcfResult};
use crate::forest::random_cut_forest::RandomCutForest;

/// AWS lower bound for `feature_dim`.
pub const MIN_DIMENSION: usize = 1;
/// AWS upper bound for `feature_dim`.
pub const MAX_DIMENSION: usize = 10_000;
/// AWS lower bound for `num_trees`.
pub const MIN_NUM_TREES: usize = 50;
/// AWS upper bound for `num_trees`.
pub const MAX_NUM_TREES: usize = 1_000;
/// AWS default for `num_trees`.
pub const DEFAULT_NUM_TREES: usize = 100;
/// AWS lower bound for `num_samples_per_tree`.
pub const MIN_SAMPLE_SIZE: usize = 1;
/// AWS upper bound for `num_samples_per_tree`.
pub const MAX_SAMPLE_SIZE: usize = 2_048;
/// AWS default for `num_samples_per_tree`.
pub const DEFAULT_SAMPLE_SIZE: usize = 256;
/// Default time-decay (no decay — uniform reservoir sampling).
pub const DEFAULT_TIME_DECAY: f64 = 0.0;

/// Validated forest hyperparameters (dimension is encoded separately
/// at the type level).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RcfConfig {
    /// Number of trees in the forest (`num_trees`).
    pub num_trees: usize,
    /// Maximum reservoir size per tree (`num_samples_per_tree`).
    pub sample_size: usize,
    /// Time-decay factor applied to reservoir sampling weights.
    /// `0.0` = uniform sampling.
    pub time_decay: f64,
    /// Optional deterministic seed; `None` falls back to entropy.
    pub seed: Option<u64>,
    /// Optional dedicated rayon thread pool size for the `parallel`
    /// cargo feature. `None` means "use rayon's global pool"
    /// (configurable via the `RAYON_NUM_THREADS` env var). `Some(n)`
    /// builds a per-forest [`rayon::ThreadPool`] of `n` workers so
    /// callers can isolate this forest from the rest of the
    /// application's rayon workload. Ignored without `parallel`.
    pub num_threads: Option<usize>,
}

impl RcfConfig {
    /// Validate the configuration against the AWS hyperparameter
    /// bounds. The forest's compile-time dimension `D` is checked
    /// separately via [`Self::validate_dimension`] so non-const
    /// callers can apply the AWS bounds without instantiating a
    /// generic.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] with the offending
    /// parameter when any bound is violated.
    pub fn validate(&self) -> RcfResult<()> {
        if !(MIN_NUM_TREES..=MAX_NUM_TREES).contains(&self.num_trees) {
            return Err(RcfError::InvalidConfig(format!(
                "num_trees {} out of [{}, {}]",
                self.num_trees, MIN_NUM_TREES, MAX_NUM_TREES
            )));
        }
        if !(MIN_SAMPLE_SIZE..=MAX_SAMPLE_SIZE).contains(&self.sample_size) {
            return Err(RcfError::InvalidConfig(format!(
                "sample_size {} out of [{}, {}]",
                self.sample_size, MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE
            )));
        }
        if !self.time_decay.is_finite() || !(0.0..=1.0).contains(&self.time_decay) {
            return Err(RcfError::InvalidConfig(format!(
                "time_decay {} out of [0.0, 1.0]",
                self.time_decay
            )));
        }
        if let Some(n) = self.num_threads
            && n == 0
        {
            return Err(RcfError::InvalidConfig(
                "num_threads must be > 0 when set; use None to fall back to rayon's global pool"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Validate the compile-time dimension `D` against the AWS
    /// `feature_dim` bounds. Called by [`ForestBuilder::build`] so
    /// every user-facing entry point gates on the AWS limits.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `D` is outside
    /// `[MIN_DIMENSION, MAX_DIMENSION]`.
    pub fn validate_dimension(dimension: usize) -> RcfResult<()> {
        if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&dimension) {
            return Err(RcfError::InvalidConfig(format!(
                "dimension {dimension} out of [{MIN_DIMENSION}, {MAX_DIMENSION}]"
            )));
        }
        Ok(())
    }
}

/// Fluent builder for [`RandomCutForest`].
///
/// Defaults match AWS `SageMaker` (`num_trees = 100`,
/// `sample_size = 256`, `time_decay = 0.0`, RNG seeded from entropy).
///
/// `D` is the per-point dimensionality. Callers pin it at construction
/// via turbofish: `ForestBuilder::<4>::new()`.
///
/// # Examples
///
/// ```
/// use rcf_rs::ForestBuilder;
///
/// let mut forest = ForestBuilder::<4>::new()
///     .num_trees(50)
///     .sample_size(64)
///     .seed(42)
///     .build()
///     .expect("AWS-conformant config");
/// forest.update([0.0, 0.0, 0.0, 0.0]).expect("dim matches");
/// ```
#[derive(Debug, Clone)]
pub struct ForestBuilder<const D: usize> {
    /// Working configuration mutated by the fluent builder methods
    /// and validated when [`ForestBuilder::build`] runs.
    config: RcfConfig,
}

impl<const D: usize> Default for ForestBuilder<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> ForestBuilder<D> {
    /// Start a new builder for `D`-dimensional points.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RcfConfig {
                num_trees: DEFAULT_NUM_TREES,
                sample_size: DEFAULT_SAMPLE_SIZE,
                time_decay: DEFAULT_TIME_DECAY,
                seed: None,
                num_threads: None,
            },
        }
    }

    /// Override the number of trees.
    #[must_use]
    pub fn num_trees(mut self, n: usize) -> Self {
        self.config.num_trees = n;
        self
    }

    /// Override the per-tree reservoir size.
    #[must_use]
    pub fn sample_size(mut self, s: usize) -> Self {
        self.config.sample_size = s;
        self
    }

    /// Override the sampler time-decay factor.
    #[must_use]
    pub fn time_decay(mut self, d: f64) -> Self {
        self.config.time_decay = d;
        self
    }

    /// Pin the RNG seed for reproducible runs.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Build a dedicated rayon thread pool of size `n` for this
    /// forest's parallel score / attribution / update paths.
    /// Requires the `parallel` cargo feature. When unset (default)
    /// rayon's global pool is used.
    #[must_use]
    pub fn num_threads(mut self, n: usize) -> Self {
        self.config.num_threads = Some(n);
        self
    }

    /// Read-only access to the config under construction.
    #[must_use]
    pub fn config(&self) -> &RcfConfig {
        &self.config
    }

    /// Per-point dimensionality (compile-time `D`).
    #[must_use]
    pub const fn dimension(&self) -> usize {
        D
    }

    /// Validate the config and instantiate the forest.
    ///
    /// # Errors
    ///
    /// Forwards [`RcfConfig::validate`] errors and propagates any
    /// failure from the underlying [`RandomCutForest`] constructor.
    pub fn build(self) -> RcfResult<RandomCutForest<D>> {
        RcfConfig::validate_dimension(D)?;
        self.config.validate()?;
        RandomCutForest::<D>::from_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n: usize, s: usize, td: f64) -> RcfConfig {
        RcfConfig {
            num_trees: n,
            sample_size: s,
            time_decay: td,
            seed: None,
            num_threads: None,
        }
    }

    #[test]
    fn validate_default_passes() {
        let c = cfg(DEFAULT_NUM_TREES, DEFAULT_SAMPLE_SIZE, DEFAULT_TIME_DECAY);
        c.validate().unwrap();
    }

    #[test]
    fn validate_dimension_rejects_zero() {
        assert!(matches!(
            RcfConfig::validate_dimension(0).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn validate_dimension_rejects_above_max() {
        assert!(RcfConfig::validate_dimension(10_001).is_err());
    }

    #[test]
    fn validate_dimension_accepts_at_max() {
        RcfConfig::validate_dimension(10_000).unwrap();
    }

    #[test]
    fn validate_rejects_num_trees_below_min() {
        assert!(cfg(49, 256, 0.0).validate().is_err());
    }

    #[test]
    fn validate_accepts_num_trees_at_bounds() {
        cfg(50, 256, 0.0).validate().unwrap();
        cfg(1000, 256, 0.0).validate().unwrap();
    }

    #[test]
    fn validate_rejects_num_trees_above_max() {
        assert!(cfg(1001, 256, 0.0).validate().is_err());
    }

    #[test]
    fn validate_rejects_sample_size_zero() {
        assert!(cfg(100, 0, 0.0).validate().is_err());
    }

    #[test]
    fn validate_accepts_sample_size_at_bounds() {
        cfg(100, 1, 0.0).validate().unwrap();
        cfg(100, 2048, 0.0).validate().unwrap();
    }

    #[test]
    fn validate_rejects_sample_size_above_max() {
        assert!(cfg(100, 2049, 0.0).validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_time_decay() {
        assert!(cfg(100, 256, -0.01).validate().is_err());
    }

    #[test]
    fn validate_rejects_time_decay_above_one() {
        assert!(cfg(100, 256, 1.01).validate().is_err());
    }

    #[test]
    fn validate_rejects_non_finite_time_decay() {
        assert!(cfg(100, 256, f64::NAN).validate().is_err());
        assert!(cfg(100, 256, f64::INFINITY).validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_num_threads() {
        let mut c = cfg(100, 256, 0.0);
        c.num_threads = Some(0);
        assert!(matches!(
            c.validate().unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn validate_accepts_some_num_threads() {
        let mut c = cfg(100, 256, 0.0);
        c.num_threads = Some(4);
        c.validate().unwrap();
    }

    #[test]
    fn validate_accepts_default_num_threads_none() {
        let c = cfg(100, 256, 0.0);
        assert_eq!(c.num_threads, None);
        c.validate().unwrap();
    }

    #[test]
    fn builder_num_threads_sets_field() {
        let b = ForestBuilder::<4>::new().num_threads(8);
        assert_eq!(b.config().num_threads, Some(8));
    }

    #[test]
    fn builder_defaults_match_aws() {
        let b = ForestBuilder::<8>::new();
        assert_eq!(b.dimension(), 8);
        assert_eq!(b.config().num_trees, 100);
        assert_eq!(b.config().sample_size, 256);
        assert!(b.config().time_decay.abs() < f64::EPSILON);
        assert_eq!(b.config().seed, None);
    }

    #[test]
    fn builder_overrides_apply() {
        let b = ForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .time_decay(0.05)
            .seed(42);
        assert_eq!(b.config().num_trees, 50);
        assert_eq!(b.config().sample_size, 64);
        assert!((b.config().time_decay - 0.05).abs() < f64::EPSILON);
        assert_eq!(b.config().seed, Some(42));
    }

    #[test]
    fn builder_build_validates() {
        let err = ForestBuilder::<4>::new().num_trees(10).build().unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }
}
