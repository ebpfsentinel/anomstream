# Features

`anomstream-core` is a **streaming anomaly detection toolkit**.
Three detector families compose with a shared substrate of streaming
stats, explanation, triage, and hot-path primitives — Random Cut Forest
is one detector family among several, drawn in wherever isolation-depth
scoring makes sense.

## Crate layout

Most entries in this catalogue live in `anomstream-core`; a handful
live in sibling crates (`anomstream-triage`, `anomstream-hotpath`)
so their SemVer can evolve at their own cadence:

| Section | Owning crate |
|---|---|
| Multivariate anomaly detectors | `anomstream-core` |
| Per-feature univariate detectors | `anomstream-core` |
| Score-level drift & regime change | `anomstream-core` |
| Streaming stats & sketches | `anomstream-core` |
| Forest scoring operations | `anomstream-core` |
| Explanation & triage → `SeverityBands`, `DiVector`, `AttributionStability`, `ForensicBaseline`, `FeatureGroups` | `anomstream-core` |
| Explanation & triage → `SageEstimator`, `PlattCalibrator` | **`anomstream-triage`** |
| SOC & ops → `AlertClusterer`, `LshAlertClusterer`, `FeedbackStore`, `AuditRecord` | **`anomstream-triage`** |
| Training & retention | `anomstream-core` |
| Persistence | `anomstream-core` |
| Observability → `MetricsSink` + metric names table | `anomstream-core` |
| Hot-path integration (eBPF ingress) → `UpdateSampler`, `PrefixRateCap`, `channel` | **`anomstream-hotpath`** |
| Security & threat model | `anomstream-core` |
| Multi-tenancy | `anomstream-core` |
| Quality | `anomstream-core` |

Consumers using the [`anomstream`](../meta/) meta-crate see every
type re-exported under a single import path regardless of the owning
member crate. Direct-dependency consumers import from the owning
crate explicitly.

## Catalogue

This doc catalogues every public module grouped by the same
taxonomy as the README:

1. [Multivariate anomaly detectors](#multivariate-anomaly-detectors)
2. [Per-feature univariate detectors](#per-feature-univariate-detectors)
3. [Score-level drift & regime change](#score-level-drift--regime-change)
4. [Streaming stats & sketches](#streaming-stats--sketches)
5. [Forest scoring operations](#forest-scoring-operations)
6. [Explanation & triage](#explanation--triage)
7. [SOC & ops](#soc--ops)
8. [Training & retention](#training--retention)
9. [Persistence](#persistence)
10. [Observability](#observability)
11. [Hot-path integration (eBPF ingress)](#hot-path-integration-ebpf-ingress)
12. [Security & threat model](#security--threat-model)
13. [Multi-tenancy](#multi-tenancy)
14. [Quality](#quality)

Driven by eBPFsentinel Enterprise needs but kept detector-agnostic —
none of the modules are required to use any other.

## Multivariate anomaly detectors

Operate on the joint `[f64; D]` distribution. RCF is the headline
but the wrappers (shingled / runtime-dim / shadow swap / pool) let
the same forest ride many shapes of workload.

### `RandomCutForest<D>` — bare RCF

Const-generic per-point dimensionality, enforces AWS SageMaker
hyperparameter bounds, score averaged across trees, reservoir
sampling without replacement. Built via `ForestBuilder<D>`.

Hot-path API: `update`, `update_indexed`, `score`, `attribution`,
`delete`, `delete_by_value`.

Source: `src/forest/`, `src/config.rs`.

Examples: `examples/quickstart.rs` (minimal 30-line demo),
`examples/streaming.rs` (CSV stdin loop),
`examples/attribution_explain.rs` (top-dim breakdown),
`examples/with_mimalloc.rs` (allocator swap).

### `ThresholdedForest<D>` — adaptive threshold (TRCF)

Tracks an EMA of the anomaly-score stream and derives a
continuously updated `mean + z · stddev` threshold. Callers
receive an `AnomalyGrade` (`score` + `threshold` + `grade ∈ [0, 1]`
+ `is_anomaly` + `ready`) instead of comparing a raw score against
a magic constant. Inspired by AWS's `TRCF` in
`randomcutforest-parkservices`, kept light (no short/long-term
duality, no near-threshold heuristics).

Built via `ThresholdedForestBuilder<D>`. API: `process`,
`process_indexed`, `score_only`, `current_threshold`, `stats`,
`reset_stats`.

**Two threshold modes** (see `ThresholdMode`):

- `ZSigma { z_factor }` (default, back-compat): `threshold =
  max(min_threshold, mean + z_factor × stddev)` on the EMA stats.
  Good for Gaussian-like scores (lag-embedded streams with
  symmetric noise).
- `Quantile { p }`: `threshold = tdigest.quantile(p)` of the
  streaming score distribution. Robust to the right-skew of
  isolation-depth scores; calibrates directly on the caller's
  alert-rate budget (`p = 0.99` ≈ 1 % firing rate in steady
  state; `0.999` ≈ 0.1 %). Uses the shipped `TDigest` primitive
  — no extra deps. Enable via `.quantile_threshold(p)` on the
  builder.

Source: `src/thresholded/`.

Example: `examples/thresholded.rs`.

### `ShingledForest<D>` — scalar-stream temporal wrapper

`ShingledForest<D>` wraps a bare `RandomCutForest<D>` with a ring
buffer of the last `D` scalars. Each `update_scalar(x)` shifts the
window and emits a fresh `[f64; D]` to the forest, turning a
scalar stream into a `D`-dim feature vector that captures
temporal autocorrelation. This is the fix for the NAB
`rogue_agent_key_hold` = 0.145 / SWaT = 0.282 failures caused by
isolation depth blind-spotting contextual temporal anomalies
(dwell / drop / frequency shift): the scalar value stays in
baseline range but the shingled subsequence sits far from every
baseline subsequence in the `D`-dim shingle space.

API: `update_scalar` (mutates ring + forest once ring is full),
`score_scalar` (non-mutating query with `value` as newest slot),
`attribution_scalar` (per-lag-index DiVector),
`score_codisp_stateless_scalar` (drift-free codisp over the
shingle). `current_shingle`, `is_warmed`, `reset_ring` for
diagnostics + lifecycle.

Built via `ShingledForestBuilder<D>` — same knobs as
`ForestBuilder` (`num_trees`, `sample_size`, `seed`, `time_decay`);
the const-generic `D` **is** the shingle size.

Types: `ShingledForest<D>`, `ShingledForestBuilder<D>`.

Source: `src/shingled.rs`.

Example: `examples/shingled.rs` (periodic sine baseline + three
injected contextual anomalies — dwell / drop / frequency shift).

### `DynamicForest<MAX_D>` — runtime-dim variant

`DynamicForest<MAX_D>` wraps `RandomCutForest<MAX_D>` behind a
runtime-sized input API (`&[f64]` of caller-declared `active_dim
≤ MAX_D`). Zero-pads shorter inputs to `MAX_D` — RCF's range-
weighted cut sampling naturally skips zero-range dims so the
pad contributes nothing to scores or attributions. Targets MSSP
/ heterogeneous multi-tenant deployments where const-generic
`D` blocks a single monomorphisation across tenants.

Hot-path callers with a known-fixed dim keep using the bare
`RandomCutForest<D>` — it's faster (fewer runtime checks, better
inlining). `DynamicForest` is the escape hatch.

Types: `DynamicForest<MAX_D>`.

Source: `src/dynamic_forest.rs`.

### `DriftAwareForest<D>` — shadow swap on drift

Facade around a live `RandomCutForest<D>` with an optional
shadow. Swap policy via `DriftRecoveryConfig`: `min_primary_age`
guards against flap-loops, `shadow_warmup` controls when the
shadow becomes the new primary. Call `on_drift()` to spawn a
shadow (trigger lives outside — route from `AdwinDetector`,
`FeatureDriftDetector`, or `MetaDriftDetector`). `update` feeds
both primary and shadow; once the shadow hits `shadow_warmup` the
swap is atomic on the next `update` tick. `score` always reads
from the primary (stable baseline until the swap lands).

Types: `DriftAwareForest<D>`, `DriftRecoveryConfig`.

Source: `src/drift_aware.rs`.

Example: `examples/drift_recovery.rs` (2-regime synthetic stream
with ADWIN-triggered shadow spawn + atomic swap after warmup).

### `TenantForestPool<K, D>` — per-tenant isolation

One `ThresholdedForest` per tenant key, bounded LRU eviction,
lazy instantiation via a factory closure. A baseline shock on
tenant A does not affect tenant B's adaptive threshold. Std-only.

API: `process`, `score_only`, `attribution`, `peek`, `get`,
`get_mut`, `insert`, `remove`, `clear`, `iter`, `iter_mut`,
`tenants`, `evict_lru`, `evict_idle`.

`evict_lru` sheds on capacity pressure (replace oldest when full).
`evict_idle(ttl)` sheds on wall-clock staleness (evict every tenant
whose last access is older than `ttl`) — intended for SaaS / MSSP
with thousands of intermittent tenants where LRU alone would
preserve dormant entries. Orthogonal paths; both emit
`rcf_tenant_evictions_total`, TTL path also emits
`rcf_tenant_idle_evictions_total`.

`readiness_summary()` returns a `ReadinessSummary`
(`resident / warming / ready / capacity / tenants_{created,evicted}_lifetime`)
for `/healthz` / `/readyz` endpoints — zero-allocation `O(resident)`
scan, no metrics-sink plumbing required. Helpers `readiness_ratio()`,
`is_fully_ready()`, `is_at_capacity()` give one-line health gates.

Source: `src/pool/`.

Example: `examples/tenant_pool.rs`.

## Per-feature univariate detectors

One accumulator per dimension — answer *which* feature drifted.
Complementary to multivariate detectors: the forest catches joint
anomalies the per-feature detectors miss, the per-feature
detectors attribute drift the forest reduces to a single scalar.

### `PerFeatureEwma<D>` — parallel EWMA z-score

`D` parallel univariate EWMA accumulators track per-dim mean +
variance with geometric decay `α`. After a warmup budget,
`observe(&[f64; D])` reports per-feature z-scores and the max
across dims. Policy-free — returns raw z-scores; caller maps
`max_z` to alert severity via `SeverityBands` or a custom rule.

Scoring runs *before* the accumulator update so the current
observation is judged against the prior distribution (textbook
EWCD / EWMA-Z convention) — without this a large step folds into
the mean in the same tick and the alert is missed.

EWMA recurrence:

```text
mean_t     ← α·x_t + (1 − α)·mean_{t−1}
variance_t ← α·(x_t − mean_{t−1})² + (1 − α)·variance_{t−1}
```

First observation seeds `mean_0 = x_0`, `variance_0 = 0`.

API: `new(PerFeatureEwmaConfig { alpha, warmup_samples })` +
`observe(&[f64; D]) → Option<PerFeatureEwmaResult<D>>` +
`is_warmed_up()` + `reset()` + `accumulators()` read-only
snapshot.

Types: `PerFeatureEwma<D>`, `PerFeatureEwmaConfig`,
`PerFeatureEwmaResult<D>`, `EwmaAccumulator`.

Reference: J. S. Hunter, *The Exponentially Weighted Moving
Average*, JQT 18(4), 1986.

Source: `src/per_feature_ewma.rs`.

### `PerFeatureCusum<D>` — parallel two-sided CUSUM

`D` parallel univariate CUSUMs track positive and negative
cumulative sums of the deviation from a reference mean. Alerts
when either side exceeds the threshold `h` — detects **sustained**
mean shifts that an EWMA adapts to and stops reporting (slow-ramp
DDoS, gradual leak).

CUSUM recurrence (per dim):

```text
S+ ← max(0, S+ + (x − μ₀ − k))
S− ← max(0, S− − (x − μ₀ + k))
alert when S+ > h  (increase)  or  S− > h  (decrease)
```

`k` is the slack (allowable drift, typical `0.5·σ`), `h` is the
threshold (typical `4·σ`), `μ₀` is the reference mean (auto-
learned on the first observation unless overridden via
`set_reference(&[f64; D])`).

API: `observe(&[f64; D]) → PerFeatureCusumResult<D>` (always
returns — no warmup gate, just empty alerts until trip) +
`active_drifts()` (count of dims in drift) + `reset()`.

Orthogonal to `MetaDriftDetector` (scalar CUSUM on the score
stream): this module is per-feature CUSUM on **raw observations**
so the caller can answer *which feature drifted and in which
direction*. Use both — they serve different triage paths.

Types: `PerFeatureCusum<D>`, `PerFeatureCusumConfig`,
`PerFeatureCusumResult<D>`, `PerFeatureCusumAlert`,
`DriftDirection`, `PerFeatureCusumAccumulator`.

References: E. S. Page, *Continuous Inspection Schemes*,
Biometrika 41, 1954. D. M. Hawkins & D. H. Olwell, *Cumulative
Sum Charts and Charting for Quality Improvement*, Springer, 1998.

Source: `src/per_feature_cusum.rs`.

### `FeatureDriftDetector<D>` — PSI / KL drift

Pins a baseline per-dim histogram, folds live traffic into a mirror
histogram with identical bin edges, and reports Population
Stability Index (`Σ (Q − P) · ln(Q/P)`) and KL divergence
`D_KL(Q || P)` per feature. CUSUM on the score stream catches the
detector *re-centring*; PSI on the features catches the data
*itself drifting*. Industry thresholds wired into
`DriftLevel::{Stable, Watch, Alert}` (`< 0.10`, `0.10..0.25`,
`≥ 0.25`). `argmax_psi()` pins the offending dim;
`reset_production()` starts a fresh monitoring window without
invalidating the baseline.

Types: `FeatureDriftDetector`, `DriftLevel`, `PSI_WATCH_THRESHOLD`,
`PSI_ALERT_THRESHOLD`.

Source: `src/feature_drift.rs`.

## Score-level drift & regime change

Operate on a scalar anomaly-score stream (or any univariate
signal). Catch baseline re-centring, regime change, and
heavy-tail shifts that the detectors above miss once they've
adapted.

### `MetaDriftDetector` — two-sided CUSUM on score stream

Runs a two-sided CUSUM on the anomaly-score stream. Fires
`DriftKind::Upward` on a sustained climb (baseline drift) and
`DriftKind::Downward` on a sustained decline — strictly more
sensitive to small persistent shifts than a `μ + 3σ` gate.
Standalone type; plug any score-like stream in. `reset()` clears
the accumulators while keeping the EMA reference, `reset_stats()`
clears everything.

Types: `CusumConfig`, `DriftKind`, `DriftVerdict`,
`MetaDriftDetector`.

Source: `src/meta_drift.rs`.

Example: `examples/meta_drift.rs`.

### `AdwinDetector` — adaptive windowing

Streaming change-point detector (Bifet & Gavaldà, SIAM SDM 2007).
Bounded ring buffer of the last `N` observations, per-update
`O(N)` scan over every split point with a Hoeffding bound
`ε_cut`; flags drift and drops the older sub-window on fire.
Confidence `δ`, window cap `N`, and observed stream amplitude
`range` are caller-configured. Use on the score stream (or any
per-step scalar) for an adaptive drift trigger with automatic
window sizing — strictly more sensitive than a fixed-window mean
test. Drives `DriftAwareForest` shadow spawn when paired.

Types: `AdwinDetector`, `ADWIN_DEFAULT_DELTA`,
`ADWIN_DEFAULT_WINDOW`.

Source: `src/adwin.rs`.

### `PotDetector` — SPOT / DSPOT univariate peaks-over-threshold

Streaming Peaks-Over-Threshold per-dimension anomaly detector
(Siffer et al., KDD 2017). For each feature dim it tracks a
running quantile `u` via the shipped `TDigest`, fits a Generalised
Pareto Distribution (GPD) to the peak excesses `(x − u) | x > u`
by method-of-moments, and returns the tail survival probability as
a p-value in `(0, 1]`. SPOT mode freezes the quantile after warm
via `freeze_baseline`; DSPOT mode keeps the digest drifting for
non-stationary streams.

Composition: run one `PotDetector` per feature dim, collect
per-dim p-values, pipe through `fisher_combine`. Joint p below
`1e-3` → anomaly. Orthogonal to RCF — the SPOT bank catches
per-dim marginal drift that isolation depth misses on
heterogeneously-distributed multivariate features (architect
review targets TSB-AD-M AUC lift 0.583 → 0.80+).

Types: `PotDetector`, `SPOT_DEFAULT_QUANTILE`, `SPOT_DEFAULT_ALERT_P`.

Source: `src/univariate_spot.rs`.

Example: `examples/univariate_spot_bank.rs` (4-dim baseline +
single-dim and all-dim outlier probes, joint p-value reported).

### `fisher_combine` — Fisher p-value combination

`fisher_combine(&[p_values]) -> f64` combines K independent
p-values into a joint anomaly score via Fisher 1932:
`T = −2 Σ ln(p_i) ~ χ²(2K)` → survival returned. Uses the
closed-form `χ²` survival for even dof.

Types: `fisher_combine`, `chi_squared_survival_even`.

Source: `src/ensemble.rs`.

## Streaming stats & sketches

Bounded-memory summaries reused across detectors — updating in
`O(1)` or `O(log n)` per observation, with read-time queries cheap
enough to run on every detection tick. Policy-free primitives;
detectors consume them internally, callers consume them directly
for dashboards or custom logic.

### `OnlineStats` — Welford mean + variance

Numerically stable single-pass variance (Welford 1962 / Knuth
TAOCP vol. 2 §4.2.2) — updates `(mean, M2)` in place so the
returned sample variance is equivalent to a two-pass algorithm
without storing the stream.

Shared by the normalizer, per-feature CUSUM / EWMA detectors, and
any downstream consumer that needs a cheap streaming
`(mean, std_dev)` summary. `O(1)` per sample, read-time
`variance()` / `std_dev()` both `O(1)`. Layer warmup / decay /
per-feature arrays on top.

API: `new()` + `update(value: f64)` + `variance()` + `std_dev()`
+ Default.

Types: `OnlineStats`.

Reference: B. P. Welford, *Note on a Method for Calculating
Corrected Sums of Squares and Products*, Technometrics 4(3),
1962.

Source: `src/online_stats.rs`.

### `TDigest` — streaming quantile digest

`TDigest` is a streaming quantile estimator (Dunning 2019) with
sub-percent accuracy on tail quantiles (p99, p99.9) where
`ScoreHistogram`'s fixed bins lose resolution. `record(x)` is
`O(1)` amortised, `quantile(q)` is `O(δ)` where `δ` is the
compression parameter (default `100`). Merge via `merge(&other)`
for multi-shard aggregation; `postcard`-serialisable under the
`serde` feature for persistence.

Uses scale function 1 (`k_1(q) = (δ / 2π) · asin(2q − 1)`) — the
canonical choice for uniform error across the quantile range.

Types: `TDigest`, `Centroid`, `TDIGEST_DEFAULT_COMPRESSION`.

Source: `src/tdigest.rs`.

### `ScoreHistogram` — fixed-bin histogram

Standalone fixed-bin histogram for score / grade / CUSUM
streams. `record(value)`, `bins()`, `bin_edges()`, `total()`,
`underflow()` / `overflow()` / `non_finite()`, `merge(&other)`,
`percentile(p)`, `reset()`. Export to Grafana / Prometheus
without re-deriving per-minute aggregates in the monitoring
pipeline.

Types: `HistogramConfig`, `ScoreHistogram`.

Source: `src/histogram.rs`.

Example: `examples/observability.rs`.

### `CountMinSketch` — probabilistic frequency sketch

`d` pairwise-independent hash rows over `w` counters. Each
`increment(key, c)` updates `d` cells (one per row); each
`estimate(key)` returns the minimum across the `d` cells the key
hashed to, guaranteeing `estimate(x) ≤ true_count(x) + ε·N` with
probability `1 − δ`, where `ε = e/w` and `δ = (1/e)^d` (Cormode
& Muthukrishnan 2005).

Default (`w=2048`, `d=4`): `ε ≈ 1.33·10⁻³`, `δ ≈ 1.83·10⁻²`,
memory `~64 KB` — enough headroom for per-flow or per-source
heavy-hitter counting over a multi-million-key stream.

Gated behind `std` because the row hashes rely on
`std::hash::DefaultHasher` (SipHash 1-3). The rest of the crate's
`no_std + alloc` surface is unaffected.

API: `new(width, depth)` + `increment(&[u8], u64)` (saturating)
+ `estimate(&[u8]) → u64` + `total()` + `reset()` +
`memory_bytes()` + `width()` / `depth()` accessors.

Types: `CountMinSketch`.

Reference: G. Cormode, S. Muthukrishnan, *An Improved Data Stream
Summary: The Count-Min Sketch and its Applications*, Journal of
Algorithms 55(1), 2005.

Source: `src/count_min_sketch.rs`.

### `Normalizer<D>` — per-feature min-max / z-score

Rescales a `D`-dimensional point into `[0, 1]` (MinMax) or
`(value − μ) / σ` (ZScore) given per-dimension `NormParams`
learned from a batch or loaded from a saved baseline. The `None`
strategy is an identity transform kept so the normalizer can sit
in a detection pipeline before fit data arrives.

Policy-free — the lib stores params and applies the math; caller
decides when to refit, when to swap, and what the source of the
samples is.

API: `identity(strategy)` (const fn) +
`fit(strategy, &[[f64; D]])` (two-pass: mean/min/max pass then
variance pass) + `transform(&[f64; D]) → [f64; D]`. Fields
`pub strategy` + `pub params: [NormParams; D]` so callers can
hand-tune individual dims after a fit.

MinMax clamps to `[0, 1]` and guards zero-range (returns `0.5`);
ZScore guards zero-variance (returns `0.0`).

Types: `Normalizer<D>`, `NormParams`, `NormStrategy`.

Source: `src/normalize.rs`.

## Forest scoring operations

RCF-specific scoring entry points beyond the bare `score()`. All
layered on `RandomCutForest<D>` (most also on `ThresholdedForest`
and `TenantForestPool`).

### Bulk batch scoring

`score_many(&[[f64; D]])` / `score_many_early_term(&[…], cfg)` /
`attribution_many(&[…])`. Under `parallel`, rayon parallelises
across points on top of the per-tree parallelism — 5-6× speedup
on backfill / SOC replay (see `docs/performance.md`). First error
aborts the batch. Available on `RandomCutForest`,
`ThresholdedForest`, and `TenantForestPool` (per-tenant;
absent-tenant returns `None` on read paths).

`score_many_with(points, |i, score| …)` — callback variant that
skips the intermediate `Vec<AnomalyScore>` allocation. Serial by
design so the closure needs no `Send + Sync` bound; preferred for
hot ingest paths that stream results directly into a writer /
histogram / alert pipeline.

Example: `examples/bulk_scoring.rs`.

### Score with confidence interval

`RandomCutForest::score_with_confidence(&point)` returns a
`ScoreWithConfidence` packaging the mean score plus per-tree
dispersion (`stddev`, `stderr`, `trees_evaluated`). SOC threshold
tuning benefits from knowing how tightly the ensemble agrees: a
`2.1 ± 0.05` verdict is qualitatively different from `2.1 ± 0.8`.
Call `ci95()` for the 95 % Gaussian CI, `ci(z)` for any custom
factor (99 % → `2.576`, 90 % → `1.645`). Always walks every tree;
use `score_early_term` when tail latency matters instead.

Types: `ScoreWithConfidence`, `DEFAULT_CI_Z_FACTOR` (`1.96`).

Source: `src/score_ci.rs`.

### Probe-based codisp scoring

`RandomCutForest::score_codisp(&point)` inserts the probe, walks
leaf → root in every tree accumulating
`max(sibling.mass / subtree.mass)` per level, then deletes the
probe. Matches the rrcf / AWS Java `codisp` semantic — captures
contextual displacement better than pure isolation depth on
wide-window anomalies. ~25× slower than `score()` post the
rayon-per-tree parallel walk + delete refactor; mutates the
reservoir per probe (baseline drifts on long eval streams, see
stateless variant below). On NAB `realKnownCause` lifts
aggregate AUC from 0.719 (`score()`) to 0.776 — beats rrcf
(0.748) and AWS Java (0.757).

`score_codisp_many(&[points])` — batched variant with per-tree
shared-walk cache + rayon across trees. Pre-inserts every probe,
walks once per unique leaf, bulk-deletes. Saturates the
reservoir past batch ≥ sample_size → `EmptyForest`.

`score_codisp_stateless(&point)` / `score_codisp_stateless_many`
— **drift-free** codisp estimate via root → leaf descent along
stored cuts, `max(sibling_mass / subtree_mass)` per depth, zero
reservoir mutation. Takes `&self`, parallel across trees,
preserves the frozen baseline exactly. Aggregate AUC 0.763 on
NAB, 0.751 on TSB-AD-M — ~0.01-0.02 below the mutating variant
but 12× faster on NAB (1.09 s full corpus vs 12.6 s). Preferred
for long eval streams.

Source: `src/forest/random_cut_forest.rs`, `src/tree/random_cut_tree.rs`.

### Fused score + attribution

`RandomCutForest::score_and_attribution(&point)` returns
`(AnomalyScore, DiVector)` from a **single tree walk** instead
of two — ~40 % faster than calling `score` + `attribution`
back-to-back. Uses `ScoreAttributionVisitor` which folds both
accumulators in one pass; scalar accumulator + per-dim
attribution split share the same `blend × damp` computation.

Types: `ScoreAttributionVisitor`.

Source: `src/visitor/combined.rs`.

### Early-termination scoring

`score_early_term(&point, config)` walks trees sequentially and
breaks as soon as the running per-tree mean has converged —
`stderr / |mean|` below `EarlyTermConfig::confidence_threshold`
after `min_trees`. Cuts wall-clock latency on
baseline-dominated traffic. `EarlyTermScore` reports
`trees_evaluated` / `trees_available` / `stderr` /
`early_stopped`. Sequential by design — use plain `score` when
full parallelism is preferred.

Types: `EarlyTermConfig`, `EarlyTermScore`.

Source: `src/early_term.rs`.

Example: `examples/early_term.rs`.

### Trimmed-mean ensemble scoring

`RandomCutForest::score_trimmed(&point, trim_fraction)` sorts
per-tree scores, drops the top + bottom `trim_fraction` fraction,
averages the middle. Robust against single-tree poisoning: an
attacker who manages to move a minority of trees' scores to the
extreme tails sees their contribution trimmed from the ensemble
mean. Typical `trim_fraction` values: `0.10` (10 %/10 %) or
`0.25` (quartile trim). `trim_fraction = 0.0` matches `score()`.

### Cross-tenant what-if

`TenantForestPool::score_across_tenants(&point)` pipes the same
probe through every resident tenant's detector and returns a
`Vec<(K, AnomalyGrade)>` sorted by descending grade. Warming-up
tenants are dropped. Intended for MSSP / threat-intel lateral
scans. Read-only — no tenant creation, no state mutation. Under
`parallel`, rayon fans out one `score_only` per tenant
(`O(N)` cost, near-linear at ~6.7 ms for 512 tenants).

Example: `examples/cross_tenant.rs`.

## Explanation & triage

*Why* did this point score high, and *how confident* is the
answer? These primitives turn scalar scores into analyst-
actionable narratives.

### Group-score decomposition

Declare named semantic groups over dim indices once
(e.g. `rate`, `payload`, `cardinality`). Every
`group_scores(&point, &groups)` returns per-group contributions
plus `top_group()` driver + `coverage()` (fraction of the raw
attribution mass the groups cover, catches gaps/overlaps). More
actionable than a 14-entry `DiVector`. Available on all three
detector types.

Types: `FeatureGroup`, `FeatureGroups`, `FeatureGroupsBuilder`,
`GroupScores`.

Source: `src/group_score.rs`.

Example: `examples/group_scores.rs`.

### `AttributionStability` — inter-tree dispersion

`attribution_stability(&point)` exposes per-dim variance /
stddev across trees alongside the mean. `confidence(dim)` is
`1 / (1 + CV)`; `argmax_weighted()` picks the dim with highest
`mean × confidence` — demotes dims where the forest disagreed.
`argmax_mean()` = classic `DiVector::argmax` for comparison.

Types: `AttributionStability`.

Source: `src/attribution_stability.rs`.

Example: `examples/attribution_stability.rs`.

### `SageEstimator<D>` — SAGE Shapley attribution

`SageEstimator<D>` (in `anomstream_core::sage`) — Monte-Carlo
permutation-sampling Shapley estimator (Covert et al. NeurIPS
2020). Accounts for feature interactions the marginal per-dim
`DiVector` ignores: when two dims jointly signal anomaly but
neither alone does, Shapley distributes the score contribution
across both. Caller supplies a baseline point (warm-phase mean
or synthetic null); estimator samples `K` random permutations,
computes each dim's marginal contribution as it joins the
coalition, averages. Cost `O(K · D)` forest scores per probe
— batch / forensic replay, not hot-path.

Types: `SageEstimator<D>`, `SageExplanation<D>`,
`SAGE_DEFAULT_PERMUTATIONS`, `SAGE_DEFAULT_SEED`.

Source: `src/sage.rs`.

### `PlattCalibrator` — probability calibration

`PlattCalibrator::fit(&[(score, label)], config)` fits a 1-D
sigmoid (Platt 1999 / Lin-Lin-Weng 2007) that maps raw scores to
`P(anomaly | score) ∈ [0, 1]`. Stable Newton-Raphson with
backtracking line search, target smoothing keeps
label-homogeneous sets numerically safe. `calibrate(score)` /
`calibrate_many(&scores)` at inference. Serde roundtrippable.
Intended for audit-defensible alerting policies (SOC2 / NIS2)
where a raw score is meaningless in compliance paperwork.

Online update via `PlattCalibrator::update_online(score, label,
lr)` applies one SGD step on the logistic loss per observation
— refine the fit as SOC feedback accumulates without re-running
the batch Newton-Raphson solver.

Types: `PlattCalibrator`, `PlattFitConfig`.

Source: `src/calibrator.rs`.

Example: `examples/calibrator.rs`.

### `SeverityBands` / `Severity` — ordinal classification

`SeverityBands` + `Severity` enum classify raw anomaly scores
into ordinal labels (`Normal` / `Low` / `Medium` / `High` /
`Critical`). Defaults match eBPFsentinel Enterprise ml-detection
(`2.0 / 3.0 / 4.0 / 5.0`). `Severity: Ord` lets callers route
alerts via `sev >= Severity::High`. Methods on both
`AnomalyScore::severity(&bands)` (bare forest) and
`AnomalyGrade::severity(&bands)` (TRCF). `SeverityBands::new`
validates `0 ≤ low < medium < high < critical`;
`SeverityBands::classify(score)` maps non-finite → `Normal`
(NaN-safe).

Types: `Severity`, `SeverityBands`.

Source: `src/severity.rs`.

Example: `examples/severity.rs`.

### `ForensicBaseline<D>` — imputation-like post-hoc baseline

`forensic_baseline(&point)` answers *"what would this dim have
looked like if the point were normal?"*. Returns
`ForensicBaseline<D>` with per-dim `expected` / `stddev` /
`delta` / `zscore` / `live_points` against every live sample in
the forest's reservoirs, plus `argmax_abs_zscore()`. Baseline
returned in raw caller coordinates — the internal `feature_scales`
transform is inverted. Great for SOC triage: alert table can
display *observed* vs *normal expected* per dim.

Types: `ForensicBaseline`.

Source: `src/forensic.rs`.

Example: `examples/forensic.rs`.

## SOC & ops

Turn a stream of alerts into something a human analyst can triage
without drowning — dedup, label, audit, retain.

### `AlertClusterer<K, D>` — cosine similarity dedup

`AlertClusterer<K, D>` groups near-duplicate `AlertRecord`s in a
sliding window so SOC dashboards see one cluster summary per
incident instead of hundreds of individual rows. Similarity is
cosine on the flattened attribution `DiVector` (`high ⧺ low`) —
two alerts with the same dominant driver and magnitude profile
cluster; unrelated attributions stay apart. Window is pruned
automatically on every `observe` call (or explicitly via
`prune_stale(now_ms)`).

Returns `ClusterDecision::{NewCluster(idx), Joined(idx)}` so
upstream can gate SIEM-write on `NewCluster` only and keep the
hot path cheap. Emits `rcf_alerts_observed_total`,
`rcf_alert_clusters_new_total`, `rcf_alert_clusters_joined_total`,
`rcf_alert_clusters_pruned_total` counters and the
`rcf_alert_clusters_active` gauge.

Types: `AlertClusterer`, `AlertCluster`, `ClusterDecision`.

Source: `src/alert_cluster.rs`.

### `LshAlertClusterer` — LSH-based dedup

`LshAlertClusterer` (in `anomstream_core::lsh_cluster`) — quantises
every per-dim attribution value into a 4-bit symbol and uses the
concatenated hex string as the bucket key. `O(1)` lookup via
`HashMap<String, u64>`. Complement to the cosine-similarity
`AlertClusterer` — LSH scales to MSSP-volume alert streams where
pairwise cosine is too slow. Mirrors the TLSH spirit (Oliver et
al. 2013) without bigram-frequency overhead.

Types: `LshAlertClusterer`, `LshClusterDecision`.

Source: `src/lsh_cluster.rs`.

### `FeedbackStore<D>` — SOC-label-driven score adjustment

`FeedbackStore<D>` (in `anomstream_core::feedback`) — bounded
ledger of analyst-labelled points. Das et al., *Incorporating
Feedback into Tree-based Anomaly Detection*, `arXiv:1708.09441`.
API: `label(point, FeedbackLabel::Benign | Confirmed)`,
`adjust(probe, raw_score) -> adjusted`. The adjustment adds a
Gaussian-kernel-weighted sum of every stored label's sign to the
raw score — Benign labels pull nearby probes **down**, Confirmed
labels push nearby probes **up**. Non-mutating on the forest
side: the hot-path `score()` is untouched, adjustment is an
additive bias layer the caller applies post-score.

Lifetime defaults: `capacity = 512`, `sigma = 1.0`, `strength =
1.0`. FIFO eviction on capacity pressure. Clamps adjusted score
to `≥ 0`.

Types: `FeedbackStore<D>`, `FeedbackLabel`,
`FEEDBACK_DEFAULT_CAPACITY`, `FEEDBACK_DEFAULT_SIGMA`,
`FEEDBACK_DEFAULT_STRENGTH`.

Source: `src/feedback.rs`.

Example: `examples/feedback_adjust.rs` (baseline forest + benign
label on an outlier probe, adjusted score drops from 2.23 → 1.23
on the exact probe and from 2.35 → 1.36 on a nearby one).

### `AlertRecord<K, D>` — audit trail (NIS2 / SOC2)

`AlertRecord<K, D>` packages every analytic output (`score`,
`grade`, `attribution`, `baseline`, `severity`) plus provenance
(`tenant`, `timestamp_ms`, `point`) into one serialisable struct.
Build via `AlertRecord::from_forest` (bare RCF) or
`AlertRecord::from_thresholded` (TRCF); chain `with_severity` to
attach a band. Versioned through `ALERT_RECORD_VERSION` so
incompatible schema changes surface at decode time. Emit to a
SIEM / object-store / WORM log via any `serde` sink — postcard
for compact per-event bytes, JSON for self-describing records.

Types: `AlertRecord`, `AlertContext`, `ALERT_RECORD_VERSION`.

Source: `src/audit.rs`.

## Training & retention

### Cold-start bootstrap

`bootstrap(points)` replays historical points from an upstream
TSDB (Prometheus / Loki / parquet dump) before live traffic is
switched on — eliminates the per-restart warmup coverage hole.
Non-finite rows are skipped and tallied in the returned
`BootstrapReport`. Available on `RandomCutForest`,
`ThresholdedForest`, and `TenantForestPool` (per-tenant).

Types: `BootstrapReport`.

Source: `src/bootstrap.rs`.

Example: `examples/bootstrap.rs`.

### Point timestamps + retention

`update_at(point, ts)` / `update_indexed_at(point, ts)` /
`process_at(point, ts)` / `process_indexed_at(point, ts)` tag
each fresh observation with a caller-supplied `u64` (epoch-ms,
sequence number…). `delete_before(cutoff)` retracts every older
point. `point_timestamp(idx)`, `oldest_timestamp()`,
`newest_timestamp()`, `tracked_timestamps()` for diagnostics.
Reservoir eviction cleans the side-map automatically. Per-tenant
on the pool. Intended for GDPR / NIS2 data retention and
forensic replay of a specific period.

Example: `examples/retention.rs`.

### Explicit retraction

`update_indexed` returns the fresh `point_idx`; `delete(idx)` /
`delete_by_value(&point)` remove a previously-observed point.
`process_indexed` on TRCF returns `(usize, AnomalyGrade)`.
Intended for SOC-driven false-positive retractions without
waiting for natural reservoir eviction. Pool variants do **not**
auto-create tenants on retraction.

Example: `examples/delete_and_scales.rs`.

### Per-dim feature scales

`ForestBuilder::feature_scales([f64; D])` pre-scales every caller
point before it reaches the forest hot paths. Use when dims have
wildly different dynamic ranges (packet-rate, protocol ratios,
entropy, cardinality); pass `1 / stddev[d]` per dim for
unit-variance normalisation. Mean-centre upstream for full
z-score. For richer pre-processing pipelines, `Normalizer<D>`
(see [Streaming stats & sketches](#streaming-stats--sketches))
supports `MinMax` / `ZScore` / `None` strategies with a batch
learner.

### Cold-start warmup (reservoir)

`initial_accept_fraction` ramps the reservoir admission
probability over the first `initial_accept_fraction · sample_size`
offers so the very first points do not dominate the sample. `1.0`
default disables the gate; AWS `CompactSampler` uses `0.125`.

## Persistence

### Warm reload

Serde-gated `to_bytes` / `from_bytes` (postcard binary) and
`to_json` / `from_json` on both `RandomCutForest` and
`ThresholdedForest`. Std-gated `to_path` / `from_path` /
`to_json_path` / `from_json_path` apply an atomic write
discipline (`tmp + fsync + rename`) — a mid-save crash leaves
the previous snapshot intact. Versioned envelope
(`PERSISTENCE_VERSION` / `THRESHOLDED_PERSISTENCE_VERSION`);
incompatible versions reject up front before any third-party
deserialiser runs.

Source: `src/persistence.rs`.

Example: `examples/warm_reload.rs`.

## Observability

### `MetricsSink` trait

`with_metrics_sink(Arc<dyn MetricsSink>)` on every detector
streams counters / gauges / histogram observations. `NoopSink`
is the zero-cost default. `TestSink` records events in-memory
for tests. Callers plug their own Prometheus / StatsD / OTel
adapter against the trait.

Canonical metric names (`metrics::names::*`):

| Type | Name | Source |
|---|---|---|
| counter | `rcf_updates_total` | every `RandomCutForest::update` |
| counter | `rcf_deletes_total` | every `delete` that actually removed a point |
| counter | `rcf_attribution_total` | every successful `attribution` |
| counter | `rcf_rejected_nan_total` | per-call NaN/±inf rejection, data-quality signal |
| counter | `rcf_early_term_stopped_total` | `score_early_term` short-circuits |
| counter | `rcf_process_total` | every `ThresholdedForest::process` |
| counter | `rcf_anomalies_fired_total` | `process` whose verdict was `is_anomaly` |
| counter | `rcf_drift_fires_total` | aggregate CUSUM fire (up + down) |
| counter | `rcf_drift_up_total` | CUSUM upward drift fires |
| counter | `rcf_drift_down_total` | CUSUM downward drift fires |
| counter | `rcf_tenant_evictions_total` | pool eviction (LRU + TTL, aggregate) |
| counter | `rcf_tenant_idle_evictions_total` | TTL-driven eviction subset |
| counter | `rcf_tenant_created_total` | pool factory invocation (fresh tenant) |
| counter | `rcf_bootstrap_points_total` | bootstrap-ingested points |
| counter | `rcf_bootstrap_skipped_total` | bootstrap points skipped (non-finite) |
| counter | `rcf_alerts_observed_total` | every `AlertClusterer::observe` |
| counter | `rcf_alert_clusters_new_total` | new cluster opened |
| counter | `rcf_alert_clusters_joined_total` | alert merged into existing cluster |
| counter | `rcf_alert_clusters_pruned_total` | cluster dropped by window prune |
| gauge | `rcf_alert_clusters_active` | active clusters in `AlertClusterer` |
| counter | `rcf_feature_drift_observed_total` | every `FeatureDriftDetector::observe` |
| gauge | `rcf_feature_drift_max_psi` | max per-dim PSI on `psi()` call |
| gauge | `rcf_forest_trees` | tree count of a forest |
| gauge | `rcf_threshold_current` | TRCF adaptive threshold |
| gauge | `rcf_ema_mean` | TRCF score-stream EMA mean |
| gauge | `rcf_ema_stddev` | TRCF score-stream EMA stddev |
| gauge | `rcf_observations_seen` | TRCF EMA observation count (warmup progress) |
| gauge | `rcf_tenants_resident` | live tenants in pool |
| gauge | `rcf_tenant_capacity` | configured pool capacity |
| histogram | `rcf_score` | raw anomaly score per scored point |
| histogram | `rcf_grade` | graded verdict `[0, 1]` per processed point |
| histogram | `rcf_drift_s_high` | CUSUM upward accumulator |
| histogram | `rcf_drift_s_low` | CUSUM downward accumulator |
| histogram | `rcf_early_term_trees` | trees walked per `score_early_term` |
| counter | `rcf_hot_path_sampler_accepted_total` | `UpdateSampler::accept_*` admitted |
| counter | `rcf_hot_path_sampler_rejected_total` | `UpdateSampler::accept_*` rejected |
| counter | `rcf_hot_path_queue_enqueued_total` | `UpdateProducer::try_enqueue` landed |
| counter | `rcf_hot_path_queue_dropped_total` | `UpdateProducer::try_enqueue` dropped on full |
| counter | `rcf_hot_path_prefix_admitted_total` | `PrefixRateCap::check_and_record` admitted |
| counter | `rcf_hot_path_prefix_capped_total` | `PrefixRateCap::check_and_record` capped |
| counter | `rcf_drift_aware_swaps_total` | `DriftAwareForest` shadow → primary swap |
| counter | `rcf_drift_aware_on_drift_total` | `DriftAwareForest::on_drift` actually spawned a shadow |
| gauge | `rcf_drift_aware_shadow_active` | 1.0 while shadow warming, 0.0 otherwise |
| counter | `rcf_adwin_observed_total` | `AdwinDetector::update` folded a finite value |
| counter | `rcf_adwin_drift_fires_total` | `AdwinDetector::update` detected drift |
| counter | `rcf_lsh_alerts_observed_total` | `LshAlertClusterer::observe` call |
| counter | `rcf_lsh_clusters_new_total` | LSH alert opened a new bucket |
| counter | `rcf_lsh_clusters_joined_total` | LSH alert merged into existing bucket |
| gauge | `rcf_lsh_clusters_active` | distinct active LSH cluster hashes |
| counter | `rcf_feedback_labels_observed_total` | `FeedbackStore::label` accepted |
| counter | `rcf_feedback_labels_benign_total` | label verdict `Benign` |
| counter | `rcf_feedback_labels_confirmed_total` | label verdict `Confirmed` |
| counter | `rcf_spot_observations_total` | `PotDetector::record` folded a finite value |
| counter | `rcf_spot_peaks_total` | value above the SPOT/DSPOT threshold `u` |

Every detector exposing these ships a `.with_metrics_sink(Arc<dyn
MetricsSink>)` chain-style builder; default is `NoopSink`.

Metric names keep the `rcf_` prefix for continuity with existing
deployments that pre-date the toolkit-framing scope expansion.

Source: `src/metrics.rs`.

## Hot-path integration (eBPF ingress)

### `UpdateSampler` + `channel` MPSC split

`anomstream_core::hot_path::UpdateSampler` drops low-value updates
before any RCF work. Two decision modes:

- `accept_stride()` — monotonic counter, keeps `1 / keep_every_n`
  offers deterministically.
- `accept_hash(flow_hash)` — per-flow sampling. Preserves
  baseline shape per flow rather than slicing any single flow.

Build via `UpdateSampler::new(keep)` (unkeyed, deterministic
admission — back-compatible) or **`UpdateSampler::new_keyed(keep)`**
(128-bit secret from `getrandom`, murmur3 keyed mix applied
before the modulo). Keyed sampler defends against the
reservoir-poisoning spray (MITRE ATLAS `AML.T0020`) where an
attacker steers their flow hash into the admitted residue class.

`channel::<D>(capacity)` returns `(UpdateProducer<D>,
UpdateConsumer<D>)` — bounded MPSC on `std::sync::mpsc::sync_channel`.
Clone the producer per classifier thread; hand the consumer to a
dedicated updater thread. `try_enqueue` is non-blocking; on
queue-full it drops + increments `dropped_total` (ops signal
for "classifier outpaces updater"). The classifier scores
against the previous-generation forest snapshot while the
updater drains at its own cadence.

Types: `UpdateSampler`, `UpdateProducer<D>`, `UpdateConsumer<D>`.

Source: `src/hot_path.rs`.

### `PrefixRateCap` — per-prefix admission rate cap

`anomstream_core::hot_path::PrefixRateCap::new(cap_per_window, window_ms)`
bounds how many admissions a single source-prefix hash bucket
can push into the reservoir within a rolling window. Fixed
256-bucket counter sketch, lock-free `check_and_record`.
Second defence line (alongside the keyed `UpdateSampler`)
against reservoir-poisoning floods from a single compromised
source — documented in `docs/threat_model.md`.

Types: `PrefixRateCap`.

## Security & threat model

See [`docs/threat_model.md`](threat_model.md) for the full
adversarial threat model covering reservoir poisoning,
evasion via contextual shift, model extraction, and classifier-
side resource exhaustion — with the MITRE ATLAS technique IDs
and the defences shipped in-crate for each.

## Multi-tenancy

### Tenant similarity index

`TenantForestPool::similarity_matrix(min_obs)` /
`most_similar(&key, top_n, min_obs)` return pairwise tenant
similarity ∈ `(0, 1]` computed on each tenant's TRCF
score-stream EMA (`mean`, `stddev`). Similarity =
`exp(-sqrt(Δmean² + Δstddev²))`. Intended for SaaS deployments
with thousands of tenants — identify clusters for tiered
alerting or spot a tenant whose baseline drifts away from its
peer group. Tenants below `min_obs` samples are excluded.

`similarity_matrix` parallelises the pair enumeration under
`parallel`; `most_similar` uses a bounded `BinaryHeap<top_n>` for
`O(N · log top_n)` — microsecond-scale at 512 tenants (~9 µs).

Example: `examples/tenant_similarity.rs`.

## Quality

### AWS SageMaker conformance

`tests/aws_conformance.rs` asserts the hyperparameter bounds,
score-averaging semantics and reservoir behaviour against the
published AWS RCF spec. Regression-guards the wire-level
compatibility story whenever the crate's internals move.

### Detection-quality regression guards

`tests/detection_quality.rs` pins AUC floors on synthetic
corpora (separable clusters > 0.95, transition > 0.90).
`tests/nab.rs` (ignored by default, needs `RCF_NAB_PATH`) pins
the NAB `realKnownCause` weighted aggregate at 0.70 via the
lag=32 + zscore + smooth(0.02) pipeline.
`tests/tsb_ad_m.rs` (ignored, needs `RCF_TSB_AD_M_PATH`) exercises
the native multivariate path across 192 / 200 files spanning
D ∈ {2 .. 66}. Two test functions: `tsb_ad_m_aggregate_auc_above_floor`
runs the fast `score()` path, `tsb_ad_m_codisp_aggregate_auc_above_floor`
runs `score_codisp()` stride-subsampled to 50 k eval rows per file
for direct parity with the AWS Java / rrcf codisp semantic. Both
pin a 0.55 aggregate floor. See `docs/performance.md` for the
per-source breakdown.

### Property-based fuzz suite

`tests/fuzz_properties.rs` runs eight adversarial properties via
`proptest` on stable: postcard + JSON roundtrip preserve scores,
non-finite inputs (`NaN` / ±inf / subnormals) are rejected cleanly
without poisoning forest state, `score_many` is bit-exact with
the serial `score` loop, `delete_by_value` shrinks live count by
exactly the number of matches, TRCF `process` never panics on
arbitrary input, `score_across_tenants` output is sorted
descending, `forensic_baseline` stays within the observed
per-dim `[min, max]`. Ships a regression file on failure so CI
pins the offending input.

### Benchmark suites

Five bench harnesses across the workspace — all pin `mimalloc`
globally and measure wall-clock with criterion:

- `core/benches/forest_throughput.rs` — core ops (insert, score,
  attribution, codisp batched + loop) over the
  `(trees, samples, D)` matrix.
- `core/benches/extended.rs` — bulk scoring, early-term, forensic,
  tenant similarity / cross-tenant, stateless codisp, TRCF
  process, delete.
- `core/benches/modules.rs` — 20 groups: shingled, quantiles,
  drift detectors, per-feature EWMA/CUSUM, sketches, normalize,
  dynamic + drift-aware, plus closing-pass adds (group_scores,
  attribution_stability, score_ci, bootstrap, persistence).
- `triage/benches/modules.rs` — LSH clustering, Platt calibration,
  SAGE explanations, cosine alert clustering, feedback store.
- `hotpath/benches/modules.rs` — UpdateSampler, PrefixRateCap,
  bounded MPSC channel.

Run the whole matrix with `cargo bench --workspace`, or target a
single crate with `cargo bench -p anomstream-core --bench modules`.
See [`performance.md`](performance.md) for the current reference
numbers on `x86_64`.
