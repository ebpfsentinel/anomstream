//! Forest aggregate root.
//!
//! - [`point_store::PointStore`] — a refcounted ring buffer that
//!   holds the canonical copy of every point currently referenced by
//!   any tree. Trees see it through the
//!   [`crate::tree::PointAccessor`] trait.
//! - [`random_cut_forest::RandomCutForest`] — orchestrates `N`
//!   `(RandomCutTree, ReservoirSampler)` pairs sharing the
//!   [`point_store::PointStore`].
//! - [`ForestSnapshot`] — minimal read-only health + capacity view,
//!   exposed so downstream crates (`anomstream-triage`, any external
//!   calibrator / SOC dashboard) can consume forest state without
//!   reaching into the reservoir-level internals.

pub mod point_store;
pub mod random_cut_forest;

pub use point_store::PointStore;
pub use random_cut_forest::RandomCutForest;

/// Read-only snapshot view of a forest's capacity + health.
///
/// Lives in `anomstream-core` so any consumer — including the
/// downstream `anomstream-triage` crate that hosts SAGE, Platt,
/// `AlertClusterer`, `FeedbackStore` — can introspect a forest
/// (`RandomCutForest` or `ThresholdedForest`) without needing
/// access to the reservoir internals (`point_store()`, `trees()`).
///
/// The contract is intentionally tiny; calibration / triage
/// pipelines typically need only sizing + progress information,
/// not tree-level data.
///
/// Implemented by:
/// - [`RandomCutForest<D>`]
/// - [`crate::thresholded::ThresholdedForest<D>`] (delegates to
///   its inner forest)
///
/// # Examples
///
/// ```
/// use anomstream_core::{ForestBuilder, ForestSnapshot};
///
/// let forest = ForestBuilder::<4>::new()
///     .num_trees(50)
///     .sample_size(64)
///     .seed(42)
///     .build()
///     .unwrap();
/// assert_eq!(forest.snapshot_num_trees(), 50);
/// assert_eq!(forest.snapshot_dimension(), 4);
/// assert_eq!(forest.snapshot_updates_seen(), 0);
/// ```
pub trait ForestSnapshot {
    /// Number of trees in the forest.
    fn snapshot_num_trees(&self) -> usize;
    /// Per-tree reservoir capacity.
    fn snapshot_sample_size(&self) -> usize;
    /// Per-point compile-time dimensionality.
    fn snapshot_dimension(&self) -> usize;
    /// Live points currently referenced by at least one tree.
    fn snapshot_live_points(&self) -> usize;
    /// Total `update` calls observed since construction.
    fn snapshot_updates_seen(&self) -> u64;
    /// Pessimistic upper bound on the forest's memory footprint in
    /// bytes (point store + tree arenas + samplers + RNGs).
    fn snapshot_memory_estimate(&self) -> usize;
}

impl<const D: usize> ForestSnapshot for RandomCutForest<D> {
    fn snapshot_num_trees(&self) -> usize {
        self.num_trees()
    }
    fn snapshot_sample_size(&self) -> usize {
        self.sample_size()
    }
    fn snapshot_dimension(&self) -> usize {
        self.dimension()
    }
    fn snapshot_live_points(&self) -> usize {
        self.point_store().live_count()
    }
    fn snapshot_updates_seen(&self) -> u64 {
        self.updates_seen()
    }
    fn snapshot_memory_estimate(&self) -> usize {
        self.memory_estimate()
    }
}
