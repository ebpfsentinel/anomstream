//! Alert clustering / dedup — SOC alert-fatigue reducer.
//!
//! SIEM dashboards drown under RCF streams that fire many
//! near-identical alerts (same attribution driver, same tenant,
//! same window). [`AlertClusterer`] folds the raw
//! [`crate::audit::AlertRecord`] stream into a small set of
//! [`AlertCluster`]s so the dashboard shows a cluster of "47 alerts
//! over 12 minutes, driven by dim 3" instead of 47 individual
//! rows.
//!
//! # Similarity metric
//!
//! Cosine similarity on the flattened attribution
//! [`DiVector`] (`high ⧺ low`). Two alerts with the same dominant
//! dimension and similar magnitude ratios cluster; unrelated
//! attributions stay apart. Threshold defaults to `0.95` — loosen
//! to merge more aggressively.
//!
//! # Sliding window
//!
//! Every observed alert carries a caller-supplied `timestamp_ms`
//! (populated by [`crate::audit::AlertContext`]). Clusters whose
//! most-recent alert falls outside `window_ms` are automatically
//! pruned on the next [`AlertClusterer::observe`] call (and can be
//! pruned explicitly via [`AlertClusterer::prune_stale`]).
//!
//! # Wiring
//!
//! ```
//! # #[cfg(feature = "serde")] {
//! use anomstream_core::ForestBuilder;
//! use anomstream_triage::{AlertClusterer, audit::{AlertRecord, AlertContext}};
//!
//! let mut forest = ForestBuilder::<4>::new()
//!     .num_trees(50).sample_size(16).seed(42).build().unwrap();
//! for i in 0..32 {
//!     let v = f64::from(i) * 0.01;
//!     forest.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
//! }
//!
//! let mut clusterer: AlertClusterer<String, 4> =
//!     AlertClusterer::new(0.95, 60_000).unwrap();
//! let ctx = AlertContext::<String>::untenanted(1_700_000_000_000);
//! let rec = AlertRecord::from_forest(&forest, &[5.0; 4], &ctx).unwrap();
//! let _ = clusterer.observe(rec);
//! assert_eq!(clusterer.len(), 1);
//! # }
//! ```

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::audit::AlertRecord;
use anomstream_core::domain::{AnomalyScore, DiVector};
use anomstream_core::error::{RcfError, RcfResult};

#[cfg(feature = "std")]
use std::sync::Arc;

/// Default active-cluster cap. At one alert per millisecond this
/// gives ~16 s of unique-attribution headroom before LRU eviction
/// kicks in — enough for human-SOC flow rates, capped short of
/// the memory-exhaustion regime a hostile high-cardinality
/// stream would otherwise reach.
pub const DEFAULT_MAX_CLUSTERS: usize = 16_384;

/// Running cluster of near-duplicate alerts.
///
/// Serializable under the `serde` feature so SIEM sinks can emit
/// cluster summaries alongside the raw records — typical SOC
/// dashboard shows cluster rollups and lets analysts drill into
/// the representative on demand.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AlertCluster<K = String, const D: usize = 4>
where
    K: Clone,
{
    /// Representative alert — the first observation that opened the
    /// cluster. Keeping the first (not the max) gives analysts a
    /// reproducible anchor; the `max_score` / `last_seen_ms` fields
    /// track the worst and most recent manifestations separately.
    pub representative: AlertRecord<K, D>,
    /// Number of alerts folded into this cluster (includes the
    /// representative itself — never zero on a live cluster).
    pub count: u64,
    /// Timestamp of the oldest alert in this cluster.
    pub first_seen_ms: u64,
    /// Timestamp of the most recent alert in this cluster.
    pub last_seen_ms: u64,
    /// Running max score across the cluster (worst-case severity
    /// signal for SOC triage).
    pub max_score: AnomalyScore,
    /// Distinct tenant identifiers that contributed. Bounded by
    /// cluster `count`; callers that truly do not care about
    /// per-tenant provenance can ignore this field.
    pub contributing_tenants: Vec<Option<K>>,
}

/// Decision returned by [`AlertClusterer::observe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ClusterDecision {
    /// Alert opened a brand-new cluster at the returned index.
    NewCluster(usize),
    /// Alert merged into the existing cluster at the returned index.
    Joined(usize),
}

/// Streaming alert clusterer. Maintains an in-memory set of active
/// [`AlertCluster`]s; each [`Self::observe`] call either extends one
/// or opens a new one. Not thread-safe — wrap in `Mutex` / shard
/// per tenant for concurrent ingest.
pub struct AlertClusterer<K = String, const D: usize = 4>
where
    K: Clone + PartialEq,
{
    /// Cosine similarity above which two alerts join the same
    /// cluster. Range `(0, 1]` — validated at construction.
    similarity_threshold: f64,
    /// Sliding-window width in milliseconds. Clusters whose
    /// `last_seen_ms` is older than `now − window_ms` are pruned.
    window_ms: u64,
    /// Hard cap on the active-cluster count. When [`Self::observe`]
    /// would grow the pool beyond this bound, the oldest cluster
    /// (smallest `last_seen_ms`) is evicted first — protects
    /// against adversarial high-cardinality attribution streams
    /// that would otherwise keep opening new clusters until OOM.
    max_clusters: usize,
    /// Active clusters, order not meaningful to callers.
    clusters: Vec<AlertCluster<K, D>>,
    /// Pool-level metrics sink — every observe/prune emits to it.
    #[cfg(feature = "std")]
    metrics: Arc<dyn anomstream_core::metrics::MetricsSink>,
}

impl<K, const D: usize> core::fmt::Debug for AlertClusterer<K, D>
where
    K: Clone + PartialEq + core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("AlertClusterer");
        s.field("similarity_threshold", &self.similarity_threshold)
            .field("window_ms", &self.window_ms)
            .field("max_clusters", &self.max_clusters)
            .field("clusters", &self.clusters.len());
        #[cfg(feature = "std")]
        s.field("metrics", &self.metrics);
        s.finish()
    }
}

impl<K, const D: usize> AlertClusterer<K, D>
where
    K: Clone + PartialEq,
{
    /// Build a fresh clusterer. `similarity_threshold` must be in
    /// `(0, 1]`; `window_ms` is the sliding-window width in
    /// milliseconds (use `u64::MAX` to disable time-based pruning).
    /// The active-cluster cap defaults to [`DEFAULT_MAX_CLUSTERS`]
    /// — override via [`Self::with_max_clusters`] when the caller
    /// has a tighter memory budget or explicitly wants to allow a
    /// larger working set.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on out-of-range
    /// `similarity_threshold`.
    pub fn new(similarity_threshold: f64, window_ms: u64) -> RcfResult<Self> {
        if !similarity_threshold.is_finite()
            || similarity_threshold <= 0.0
            || similarity_threshold > 1.0
        {
            return Err(RcfError::InvalidConfig(
                format!("similarity_threshold {similarity_threshold} out of (0, 1]").into(),
            ));
        }
        Ok(Self {
            similarity_threshold,
            window_ms,
            max_clusters: DEFAULT_MAX_CLUSTERS,
            clusters: Vec::new(),
            #[cfg(feature = "std")]
            metrics: anomstream_core::metrics::default_sink(),
        })
    }

    /// Override the active-cluster cap. When
    /// [`Self::observe`] would push the pool past `max`, the
    /// oldest cluster (smallest `last_seen_ms`) is evicted before
    /// the new cluster is opened — bounded memory under
    /// adversarial high-cardinality alert streams.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on `max == 0`.
    pub fn with_max_clusters(mut self, max: usize) -> RcfResult<Self> {
        if max == 0 {
            return Err(RcfError::InvalidConfig(
                "AlertClusterer: max_clusters must be > 0".into(),
            ));
        }
        self.max_clusters = max;
        // Trim eagerly when the new cap is below the live cluster
        // count so the invariant `clusters.len() ≤ max_clusters`
        // holds after the setter returns.
        while self.clusters.len() > self.max_clusters {
            self.evict_oldest_cluster();
        }
        Ok(self)
    }

    /// Active-cluster cap.
    #[must_use]
    pub fn max_clusters(&self) -> usize {
        self.max_clusters
    }

    /// Install a [`anomstream_core::MetricsSink`] — every `observe` / `prune`
    /// call emits counters / gauges into it.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_metrics_sink(
        mut self,
        sink: Arc<dyn anomstream_core::metrics::MetricsSink>,
    ) -> Self {
        self.emit_active_gauge_on(&sink);
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn anomstream_core::metrics::MetricsSink> {
        &self.metrics
    }

    /// Cosine-similarity threshold used by the clusterer.
    #[must_use]
    pub fn similarity_threshold(&self) -> f64 {
        self.similarity_threshold
    }

    /// Sliding-window width in milliseconds.
    #[must_use]
    pub fn window_ms(&self) -> u64 {
        self.window_ms
    }

    /// Number of active clusters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    /// Whether the clusterer is currently empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }

    /// Read-only slice of active clusters.
    #[must_use]
    pub fn clusters(&self) -> &[AlertCluster<K, D>] {
        &self.clusters
    }

    /// Fold `rec` into the clusterer. Prunes stale clusters first
    /// (anything older than `window_ms` vs `rec.timestamp_ms`), then
    /// joins the highest-similarity cluster above threshold, or
    /// opens a new one.
    #[must_use = "detector output should be checked — dropping it silently usually indicates a logic bug"]
    pub fn observe(&mut self, rec: AlertRecord<K, D>) -> ClusterDecision {
        #[cfg(feature = "std")]
        self.metrics
            .inc_counter(anomstream_core::metrics::names::ALERTS_OBSERVED_TOTAL, 1);
        self.prune_stale_internal(rec.timestamp_ms);

        // Find the best cluster to merge into.
        let mut best_idx: Option<usize> = None;
        let mut best_sim = self.similarity_threshold;
        for (i, cluster) in self.clusters.iter().enumerate() {
            let sim = cosine_similarity(&cluster.representative.attribution, &rec.attribution);
            if sim >= best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if let Some(i) = best_idx {
            let cluster = &mut self.clusters[i];
            cluster.count = cluster.count.saturating_add(1);
            cluster.last_seen_ms = cluster.last_seen_ms.max(rec.timestamp_ms);
            cluster.first_seen_ms = cluster.first_seen_ms.min(rec.timestamp_ms);
            if rec.score > cluster.max_score {
                cluster.max_score = rec.score;
            }
            if !cluster.contributing_tenants.contains(&rec.tenant) {
                cluster.contributing_tenants.push(rec.tenant.clone());
            }
            #[cfg(feature = "std")]
            self.metrics.inc_counter(
                anomstream_core::metrics::names::ALERT_CLUSTERS_JOINED_TOTAL,
                1,
            );
            ClusterDecision::Joined(i)
        } else {
            // Cap enforcement: evict the oldest cluster before
            // opening a fresh one once we hit `max_clusters`.
            // Protects against adversarial attribution streams
            // that would otherwise fill the clusterer unboundedly
            // within `window_ms`.
            while self.clusters.len() >= self.max_clusters {
                if !self.evict_oldest_cluster() {
                    break;
                }
            }
            let ts = rec.timestamp_ms;
            let score = rec.score;
            let tenant = rec.tenant.clone();
            self.clusters.push(AlertCluster {
                representative: rec,
                count: 1,
                first_seen_ms: ts,
                last_seen_ms: ts,
                max_score: score,
                contributing_tenants: vec![tenant],
            });
            let idx = self.clusters.len() - 1;
            #[cfg(feature = "std")]
            {
                self.metrics
                    .inc_counter(anomstream_core::metrics::names::ALERT_CLUSTERS_NEW_TOTAL, 1);
                self.emit_active_gauge();
            }
            ClusterDecision::NewCluster(idx)
        }
    }

    /// Remove the cluster with the smallest `last_seen_ms` (oldest
    /// activity). Returns `true` when a cluster was removed.
    /// Called by the cap-enforcement path in [`Self::observe`] and
    /// by [`Self::with_max_clusters`] when tightening the cap.
    fn evict_oldest_cluster(&mut self) -> bool {
        let Some((oldest_idx, _)) = self
            .clusters
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| c.last_seen_ms)
        else {
            return false;
        };
        self.clusters.swap_remove(oldest_idx);
        #[cfg(feature = "std")]
        {
            self.metrics.inc_counter(
                anomstream_core::metrics::names::ALERT_CLUSTERS_PRUNED_TOTAL,
                1,
            );
            self.emit_active_gauge();
        }
        true
    }

    /// Drop every cluster whose `last_seen_ms` is older than
    /// `now_ms − window_ms`. Call on a schedule when the alert
    /// stream is sparse; [`Self::observe`] prunes automatically on
    /// every call so a live stream rarely needs this.
    pub fn prune_stale(&mut self, now_ms: u64) {
        self.prune_stale_internal(now_ms);
    }

    /// Internal prune used by `observe` + public `prune_stale`.
    fn prune_stale_internal(&mut self, now_ms: u64) {
        if self.clusters.is_empty() || self.window_ms == u64::MAX {
            return;
        }
        let cutoff = now_ms.saturating_sub(self.window_ms);
        let before = self.clusters.len();
        self.clusters.retain(|c| c.last_seen_ms >= cutoff);
        #[cfg(feature = "std")]
        {
            let pruned = before.saturating_sub(self.clusters.len());
            if pruned > 0 {
                self.metrics.inc_counter(
                    anomstream_core::metrics::names::ALERT_CLUSTERS_PRUNED_TOTAL,
                    pruned as u64,
                );
                self.emit_active_gauge();
            }
        }
        #[cfg(not(feature = "std"))]
        let _ = before;
    }

    /// Clear every cluster.
    pub fn clear(&mut self) {
        self.clusters.clear();
        #[cfg(feature = "std")]
        self.emit_active_gauge();
    }

    /// Emit the resident-count gauge on the installed sink.
    #[cfg(feature = "std")]
    fn emit_active_gauge(&self) {
        self.emit_active_gauge_on(&self.metrics);
    }

    /// Emit the resident-count gauge on an explicit sink — used by
    /// `with_metrics_sink` to stamp the initial value immediately.
    #[cfg(feature = "std")]
    fn emit_active_gauge_on(&self, sink: &Arc<dyn anomstream_core::metrics::MetricsSink>) {
        #[allow(clippy::cast_precision_loss)]
        sink.set_gauge(
            anomstream_core::metrics::names::ALERT_CLUSTERS_ACTIVE,
            self.clusters.len() as f64,
        );
    }
}

/// Cosine similarity on the flattened `DiVector` (`high ⧺ low`).
/// Returns `0.0` on degenerate (zero-norm) inputs — a zero-norm
/// attribution carries no directional information so cannot be
/// similar to anything.
fn cosine_similarity(a: &DiVector, b: &DiVector) -> f64 {
    if a.dim() != b.dim() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (x, y) in a.high().iter().zip(b.high().iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    for (x, y) in a.low().iter().zip(b.low().iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    let denom = (na * nb).sqrt();
    if denom <= 0.0 { 0.0 } else { dot / denom }
}

#[cfg(all(test, feature = "serde", feature = "postcard", feature = "serde_json"))]
#[allow(clippy::unwrap_used, clippy::panic, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::audit::AlertContext;
    use anomstream_core::ForestBuilder;

    fn warm_forest(seed: u64) -> anomstream_core::RandomCutForest<4> {
        let mut f = ForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(seed)
            .build()
            .unwrap();
        for i in 0..32_u32 {
            let v = f64::from(i) * 0.01;
            f.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
        }
        f
    }

    fn rec(
        forest: &anomstream_core::RandomCutForest<4>,
        p: [f64; 4],
        ts: u64,
    ) -> AlertRecord<String, 4> {
        let ctx = AlertContext::<String>::for_tenant("t1".into(), ts);
        AlertRecord::from_forest(forest, &p, &ctx).unwrap()
    }

    #[test]
    fn new_rejects_bad_threshold() {
        assert!(AlertClusterer::<String, 4>::new(0.0, 60_000).is_err());
        assert!(AlertClusterer::<String, 4>::new(1.5, 60_000).is_err());
        assert!(AlertClusterer::<String, 4>::new(f64::NAN, 60_000).is_err());
    }

    #[test]
    fn first_alert_opens_cluster() {
        let f = warm_forest(1);
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 60_000).unwrap();
        let decision = c.observe(rec(&f, [5.0, 5.0, 5.0, 5.0], 1000));
        assert_eq!(decision, ClusterDecision::NewCluster(0));
        assert_eq!(c.len(), 1);
        assert_eq!(c.clusters()[0].count, 1);
    }

    #[test]
    fn identical_alerts_join_cluster() {
        let f = warm_forest(2);
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 60_000).unwrap();
        let _ = c.observe(rec(&f, [5.0, 5.0, 5.0, 5.0], 1000));
        let d2 = c.observe(rec(&f, [5.0, 5.0, 5.0, 5.0], 2000));
        assert_eq!(d2, ClusterDecision::Joined(0));
        assert_eq!(c.len(), 1);
        assert_eq!(c.clusters()[0].count, 2);
        assert_eq!(c.clusters()[0].first_seen_ms, 1000);
        assert_eq!(c.clusters()[0].last_seen_ms, 2000);
    }

    #[test]
    fn disparate_points_open_separate_clusters() {
        // Two probes with very different attribution profiles.
        let f = warm_forest(3);
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.99, 60_000).unwrap();
        // Extreme anomaly on dim 0 only.
        let r1 = rec(&f, [100.0, 0.15, 0.25, 0.35], 1000);
        // Extreme anomaly on dim 3 only.
        let r2 = rec(&f, [0.0, 0.15, 0.25, 100.0], 1100);
        let _ = c.observe(r1);
        let _ = c.observe(r2);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn prune_drops_stale_clusters() {
        let f = warm_forest(4);
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 500).unwrap();
        let _ = c.observe(rec(&f, [5.0, 5.0, 5.0, 5.0], 1000));
        assert_eq!(c.len(), 1);
        // 2000 ms later — prune window 500 ms
        c.prune_stale(2000);
        assert!(c.is_empty());
    }

    #[test]
    fn observe_prunes_stale_automatically() {
        let f = warm_forest(5);
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 500).unwrap();
        let _ = c.observe(rec(&f, [5.0, 5.0, 5.0, 5.0], 1000));
        let d = c.observe(rec(&f, [5.0, 5.0, 5.0, 5.0], 3000));
        // First cluster pruned, second opened fresh.
        assert_eq!(d, ClusterDecision::NewCluster(0));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn max_score_tracks_worst_in_cluster() {
        let f = warm_forest(6);
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.5, 60_000).unwrap();
        let r1 = rec(&f, [5.0, 5.0, 5.0, 5.0], 1000);
        let r2 = rec(&f, [50.0, 50.0, 50.0, 50.0], 2000);
        let s1 = r1.score;
        let s2 = r2.score;
        let _ = c.observe(r1);
        let _ = c.observe(r2);
        // Both should join same cluster (loose threshold).
        assert_eq!(c.len(), 1);
        assert_eq!(c.clusters()[0].max_score, if s2 > s1 { s2 } else { s1 });
    }

    #[test]
    fn window_max_disables_pruning() {
        let f = warm_forest(7);
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, u64::MAX).unwrap();
        let _ = c.observe(rec(&f, [5.0, 5.0, 5.0, 5.0], 1000));
        c.prune_stale(u64::MAX);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn postcard_roundtrip_on_cluster() {
        let f = warm_forest(8);
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 60_000).unwrap();
        let _ = c.observe(rec(&f, [5.0, 5.0, 5.0, 5.0], 1000));
        let bytes = postcard::to_allocvec(&c.clusters()[0]).unwrap();
        let back: AlertCluster<String, 4> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(&c.clusters()[0], &back);
    }

    #[test]
    fn clear_drops_everything() {
        let f = warm_forest(9);
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 60_000).unwrap();
        let _ = c.observe(rec(&f, [5.0, 5.0, 5.0, 5.0], 1000));
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn cosine_similarity_identity() {
        let a = DiVector::from_arrays(vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.0]).unwrap();
        let b = DiVector::from_arrays(vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.0]).unwrap();
        let s = cosine_similarity(&a, &b);
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = DiVector::from_arrays(vec![1.0, 0.0], vec![0.0, 0.0]).unwrap();
        let b = DiVector::from_arrays(vec![0.0, 1.0], vec![0.0, 0.0]).unwrap();
        let s = cosine_similarity(&a, &b);
        assert!(s.abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_zero_norm() {
        let a = DiVector::from_arrays(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let b = DiVector::from_arrays(vec![1.0, 1.0], vec![0.0, 0.0]).unwrap();
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn with_max_clusters_rejects_zero() {
        let c: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 60_000).unwrap();
        assert!(c.with_max_clusters(0).is_err());
    }

    /// Build an `AlertRecord` with an orthogonal synthetic
    /// `DiVector` so cosine similarity between distinct records is
    /// exactly zero — decouples the cap-enforcement test from the
    /// forest's attribution behaviour.
    fn orthogonal_rec(slot: usize, ts: u64) -> AlertRecord<String, 4> {
        let mut high = alloc::vec![0.0_f64; 4];
        let mut low = alloc::vec![0.0_f64; 4];
        let idx = slot % 4;
        if (slot / 4).is_multiple_of(2) {
            high[idx] = 1.0;
        } else {
            low[idx] = 1.0;
        }
        let attr = DiVector::from_arrays(high, low).unwrap();
        let baseline = anomstream_core::forensic::ForensicBaseline::<4> {
            observed: [0.0; 4],
            expected: [0.0; 4],
            stddev: [0.0; 4],
            delta: [0.0; 4],
            zscore: [0.0; 4],
            live_points: 0,
        };
        AlertRecord::new(
            None,
            ts,
            [0.0; 4],
            AnomalyScore::new(1.0).unwrap(),
            None,
            None,
            attr,
            baseline,
        )
    }

    #[test]
    fn observe_enforces_cluster_cap_via_lru_eviction() {
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.5, u64::MAX)
            .unwrap()
            .with_max_clusters(3)
            .unwrap();
        // 8 orthogonal attributions — each opens its own cluster.
        // After the 4th the cap of 3 must start evicting.
        for i in 0..8 {
            let _ = c.observe(orthogonal_rec(i, 1_000 + i as u64));
            assert!(
                c.len() <= 3,
                "cluster cap breached at iteration {i}: len = {}",
                c.len()
            );
        }
        assert_eq!(c.len(), 3);
        // Earliest timestamps must have been evicted by LRU.
        let live_first_seen: Vec<u64> = c.clusters().iter().map(|cl| cl.first_seen_ms).collect();
        for ts in [1000_u64, 1001, 1002, 1003] {
            assert!(
                !live_first_seen.contains(&ts),
                "ts {ts} should have been evicted; live = {live_first_seen:?}"
            );
        }
    }

    #[test]
    fn with_max_clusters_trims_live_pool() {
        let mut c: AlertClusterer<String, 4> = AlertClusterer::new(0.5, u64::MAX).unwrap();
        for i in 0..5 {
            let _ = c.observe(orthogonal_rec(i, 1_000 + i as u64));
        }
        assert!(c.len() >= 4);
        // Shrinking the cap below current len must trim eagerly.
        let c = c.with_max_clusters(2).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c.max_clusters(), 2);
    }
}
