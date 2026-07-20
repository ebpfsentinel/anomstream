# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses calendar versioning (`YYYY.M.X`).

## [Unreleased]

### Performance

- Gate the per-tree rayon fan-out on estimated work (`num_trees × D ≥
  2048`) instead of on the `parallel` cargo feature alone. A single tree
  walk is a few hundred nanoseconds — under rayon's task-dispatch floor —
  so small and mid-sized ensembles were paying scheduling overhead that
  exceeded the work being parallelised. Below the threshold the ensemble
  walk now stays on the calling thread.

  Measured back-to-back with only the threshold flipped: `forest_update`
  is 1.7–4.9× faster and `forest_score` 1.35–2.9× faster at
  `(100, 256, 4)`, `(50, 128, 16)` and `(100, 256, 16)` — the last being
  the AWS-default shape. Ensembles at or above the threshold
  (`(200, 512, 16)`, `(100, 256, 64)`) take the same parallel path as
  before and are unchanged. Applies to `score`, `attribution`,
  `score_and_attribution`, both codisp aggregates, `update` and `delete`.

  Batch entry points (`score_many`, `attribution_many`,
  `score_codisp_stateless_many`) fan out across *points*, where each task
  is a whole ensemble walk; they are unaffected and remain parallel.

  Behaviour-only change — scores are unchanged beyond float summation
  order, which is not associative and was already unpinned across the
  serial / parallel arms.

### Fixed

- `audit_record_json_roundtrip_bit_exact_non_fp_fields` asserted bit-exact
  `f64` equality on the derived `score` field across a `serde_json`
  round-trip, which the format does not guarantee: some doubles serialise
  to a shortest decimal form that re-parses one ULP away (e.g.
  `1.8295224971362654` → `1.8295224971362656`). The test now holds `score`
  to a one-ULP tolerance, matching what its own comment already described,
  and keeps bit-exactness for the non-floating-point fields.

## [2026.4.1] - 2026-04-26

First public release of the **anomstream** workspace — `anomstream-core`
+ `anomstream-triage` + `anomstream-hotpath` + the `anomstream` meta
facade. The crate was previously developed as `rcf-rs` →
`anomstream-rs`; this release is the first published version under the
final name.

This entry catalogues the public surface shipped at `2026.4.1`. Subsequent
releases will only list deltas.

### Workspace

- Cargo workspace bootstrapped with four members (`core`, `triage`,
  `hotpath`, `meta`), Rust edition 2024, MSRV 1.95, Apache-2.0.
- Workspace-wide lints: `clippy::all = deny`, `clippy::pedantic = warn`,
  `unsafe_code = forbid`, `missing_docs = warn`, `unwrap_used = deny`,
  `panic = deny`, `todo = deny`, `unimplemented = deny`,
  `dbg_macro = deny`.
- Release profile: `overflow-checks = true` (kept on in release).
- Per-crate + workspace CI matrix; full bench matrix runnable via
  `cargo bench` across all members.
- Calendar-versioned release workflow (`v2026.4.1` style) with
  in-tree version stamping (Cargo.toml + README), full release gate
  (`fmt` + `clippy -D warnings` + workspace tests + `cargo audit` +
  `cargo deny check` + dependency-ordered `cargo publish --dry-run`
  for `core → triage → hotpath → anomstream`), and tag + GitHub
  Release publish job.
- Weekly + manual CI on top of push CI.

### Added — `anomstream-core`

Core math floor: detectors + streaming primitives + cross-cut contracts.
Conformant with the AWS SageMaker Random Cut Forest specification
(Guha et al., ICML 2016; Park et al., SDM 2004 reservoir sampling).

**Multivariate forest detectors**

- `RandomCutForest<D>` — bare RCF with reservoir sampling without
  replacement, range-weighted random cuts, ensemble averaged anomaly
  score, AWS hyperparameter bounds enforced at build time
  (`feature_dim ∈ [1, 10000]`, `num_trees ∈ [50, 1000]`,
  `num_samples_per_tree ∈ [1, 2048]`,
  `time_decay = 0.1 / sample_size` default tracking the AWS Java
  `CompactSampler`).
- `ForestBuilder<D>` + `RcfConfig` typed builder.
- `ThresholdedForest<D>` (TRCF) — adaptive EMA threshold layer with
  `ThresholdedConfig`, `ThresholdMode`, `EmaStats`, `AnomalyGrade`.
- `ShingledForest<D>` + `ShingledForestBuilder` — scalar-stream
  temporal wrapper.
- `DynamicForest<MAX_D>` — runtime-dim variant.
- `DriftAwareForest<D>` + `DriftRecoveryConfig` — shadow-swap on drift.
- `TenantForestPool<K, D>` + `ReadinessSummary` — bounded per-tenant
  pool with LRU eviction.
- `MatrixProfile` (STOMP) — batch time-series discord and motif
  detection with `MIN_WINDOW` and `MAX_WINDOW` caps.

**Per-feature univariate detectors**

- `PerFeatureEwma<D>` + `EwmaAccumulator` + `PerFeatureEwmaConfig`
  + `PerFeatureEwmaResult` — parallel univariate EWMA z-score.
- `PerFeatureCusum<D>` + `PerFeatureCusumAccumulator`
  + `PerFeatureCusumConfig` + `PerFeatureCusumResult`
  + `PerFeatureCusumAlert` + `DriftDirection` — parallel two-sided
  CUSUM change-point detector.
- `FeatureDriftDetector<D>` + `DriftLevel` — PSI / KL feature drift.

**Score-level drift & regime change**

- `MetaDriftDetector` + `CusumConfig` + `DriftKind` + `DriftVerdict`
  — two-sided CUSUM on the score stream.
- `AdwinDetector` + `ADWIN_DEFAULT_DELTA` + `ADWIN_DEFAULT_WINDOW`
  — adaptive windowing change detector (`std`-gated).
- `PotDetector` + `SPOT_DEFAULT_QUANTILE` + `SPOT_DEFAULT_ALERT_P`
  — SPOT / DSPOT univariate peaks-over-threshold (`std`-gated).
- `fisher_combine` + `chi_squared_survival_even` — Fisher p-value
  ensemble combination.

**Streaming stats & sketches**

- `OnlineStats` — Welford streaming mean + variance.
- `TDigest` + `Centroid` + `TDIGEST_DEFAULT_COMPRESSION` — streaming
  quantile digest.
- `ScoreHistogram` + `HistogramConfig` — fixed-bin histogram.
- `BloomFilter` + `BLOOM_DEFAULT_FPR` + `BLOOM_MAX_HASHES` —
  probabilistic set membership using Kirsch–Mitzenmacher double
  hashing (`std`-gated).
- `CountMinSketch` — probabilistic frequency sketch (`std`-gated).
- `HyperLogLog` + `HLL_DEFAULT_PRECISION` + `HLL_MIN_PRECISION` +
  `HLL_MAX_PRECISION` — distinct-count cardinality sketch.
- `SpaceSaving<K>` + `HeavyHitter` + `HeavyHitterEntry`
  + `SPACE_SAVING_DEFAULT_CAPACITY` — deterministic top-K
  heavy-hitter sketch (Metwally–Agrawal–Abbadi).
- `Normalizer<D>` + `NormParams` + `NormStrategy` — per-feature
  `MinMax` / `ZScore` / `None` transforms.
- Caller-supplied sizing capped on `CountMinSketch`, `BloomFilter`,
  and the SAGE estimator to prevent oversized allocations.

**Forest scoring operations**

- Bulk batch scoring.
- `ScoreWithConfidence` + `DEFAULT_CI_Z_FACTOR` — score with
  confidence interval (`score_ci`).
- Probe-based codisp scoring.
- Fused score + attribution path (single tree traversal).
- `EarlyTermConfig` + `EarlyTermScore` — early-termination scoring.
- Trimmed-mean ensemble scoring (with thread-local scratch buffer).
- Cross-tenant what-if scoring (via `pool`).

**Explanation primitives (in core)**

- `group_score` — `FeatureGroup` + `FeatureGroups`
  + `FeatureGroupsBuilder` + `GroupScores` named-group decomposition
  of attribution vectors.
- `AttributionStability` — inter-tree dispersion + confidence for
  attribution.
- `ForensicBaseline<D>` — imputation-like post-hoc baseline.
- `Severity` + `SeverityBands` — ordinal severity classification.

**Training & retention**

- `BootstrapReport` — cold-start replay from upstream TSDB.
- Point timestamps + retention.
- Explicit retraction.
- Per-dim feature scales.
- Cold-start warmup via reservoir.

**Persistence (`serde`-gated)**

- `persistence` module with snapshot + warm reload.
- `serde_util` shadow-deserialize helpers with `try_from` validation.
- Deserialize payload size capped at 256 MiB / 1 GiB with
  `from_*_with_max_size` escape hatches; trust model documented.

**Observability**

- `MetricsSink` trait — telemetry contract consumed by every detector.
- `NoopSink` default with `LazyLock<Arc<NoopSink>>` shared static.
- `RcfError` + `RcfResult` typed error surface (`InvalidConfig`
  shrunk to `Box<str>`).

**Domain types & visitors**

- `AnomalyScore`, `BoundingBox`, `Cut`, `DiVector`, `Point`.
- `RandomCutTree`, `NodeRef`, `NodeStore`, `NodeView`, `NodeViewMut`,
  `InternalData`, `LeafData`, `PointAccessor`.
- `ReservoirSampler` + `SamplerOp`.
- `Visitor` trait with `ScalarScoreVisitor`, `AttributionVisitor`,
  `ScoreAttributionVisitor` strategies.

**Quality (`std`-gated)**

- `vus_pr`, `range_auc_pr`, `vus_pr_with_buffer`,
  `VUS_PR_DEFAULT_MAX_BUFFER` — VUS-PR quality metric.
- `TsbAdMDataset` — TSB-AD-M CSV loader and runnable example.

### Added — `anomstream-triage`

SOC-opinionated triage layer. Six components consuming `core` outputs;
none depend on each other except within-crate.

- `PlattCalibrator` + `PlattFitConfig` — batch closed-form Platt
  scaling with online SGD, skew-detection fallback when the
  closed-form is ill-conditioned, Hessian singularity guard.
- `SageEstimator` + `SageExplanation`
  + `SAGE_DEFAULT_PERMUTATIONS` + `SAGE_DEFAULT_SEED` — SAGE Shapley
  attribution via permutation sampling, deterministic
  `explain_with_seed` variant (`std`-gated).
- `AlertClusterer` + `AlertCluster` + `ClusterDecision` — cosine
  similarity alert dedup over a sliding window; `u64` key choice
  documented.
- `LshAlertClusterer` + `LshClusterDecision` — LSH-based dedup for
  MSSP-volume streams, per-instance seed rotation (`std`-gated).
- `FeedbackStore` + `FeedbackLabel` + `FEEDBACK_DEFAULT_CAPACITY`
  + `FEEDBACK_DEFAULT_SIGMA` + `FEEDBACK_DEFAULT_STRENGTH` —
  bounded analyst-label ledger with Gaussian-kernel score adjustment
  (Das et al. 2017), kernel-weighted normalisation bounds bias by
  `strength`, `MAX_CAPACITY` cap, contributing-tenants list capped
  at 32 with FIFO eviction (`std`-gated).
- `AlertRecord` + `AlertContext` + `ALERT_RECORD_VERSION` —
  serialisable audit envelope packaging every analytic output for
  SIEM / WORM export (NIS2 / SOC2 grade).
- `AuditChain` + `AuditChainEntry` + `verify_audit_chain`
  + `AUDIT_CHAIN_GENESIS_PREV` + `AUDIT_CHAIN_MIN_KEY_LEN`
  + `AUDIT_CHAIN_TAG_LEN` — HMAC-SHA256 tamper-evident envelope
  around each `AlertRecord`; `deny_unknown_fields` enforced on
  shadow types (gated by `audit-integrity` + `postcard`).

### Added — `anomstream-hotpath`

eBPF-style ingress primitives for the userspace side of a packet
pipeline.

- `UpdateSampler` with default + `new_keyed_with_seeds` constructor
  for restricted-entropy environments.
- `UpdateProducer<const D: usize>` + `UpdateConsumer<const D: usize>`
  MPSC split.
- `update_channel` + `update_channel_with_sink` + `try_update_channel`
  + `try_update_channel_with_sink` channel constructors.
- `MAX_CHANNEL_CAPACITY = 1 << 20` and validated capacity at
  construction.
- `METRICS_BATCH_SIZE = 64` — batched `MetricsSink` dispatch.
- `PrefixRateCap` — per-prefix admission rate cap with cache-line
  padded buckets, `NonZero` typing on the public config,
  `compare_exchange_weak` rollover loop, wrapping-counter semantics
  documented.
- 8-thread contention bench.

### Added — `anomstream` (meta facade)

Umbrella crate giving consumers a single import path regardless of
which layers they pull in.

- Cargo features: `core`, `triage`, `hotpath`, `parallel`, `serde`,
  `postcard`, `audit-integrity`, `std`, `full`. Default feature set
  is `["core", "std"]` — `full` is the explicit opt-in for the
  whole stack + `serde_json`.
- `triage` and `hotpath` both imply `core`.
- `cfg`-gated `pub use` re-exports mirror each member crate's root
  surface verbatim (no globs — every committed item is enumerated
  so SemVer travels through `cargo publish`, not docstrings).
- `hot_path` submodule preserves the `anomstream::hot_path::*`
  spelling used by pre-workspace-split callers.
- `core_lib` / `triage_lib` / `hotpath_lib` re-exports of the member
  crate roots as escape hatches for deep module paths
  (`anomstream::core_lib::persistence::Snapshot`,
  `anomstream::triage_lib::audit::AlertRecordShadow`, …); these
  namespaces are SemVer-scoped to the owning member, not to the
  facade.
- `non_exhaustive` on public enums and pub-field structs;
  `must_use` on builder / query returns.

### Changed

- Crate renamed `rcf-rs` → `anomstream-rs` → `anomstream` (final
  published name).
- `rand` 0.9 → 0.10, `rand_chacha` 0.9 → 0.10, `getrandom` 0.3 → 0.4
  (`RngCore` → `Rng` basic trait rename; `Rng` → `RngExt` for
  `.random()` / `.random_range()` extension methods).
- `meta` (`anomstream`) default features shrunk from `full` to
  `core` + `std`.
- Public surface uses explicit `pub use` enumerations; glob
  re-exports removed for SemVer stability. `serde_json` added to
  the `full` feature.
- `Deserialize` on stateful public types now goes through a shadow
  struct + `try_from` validation gate.
- `RcfError::InvalidConfig` shrunk to `Box<str>` to keep the error
  type small.
- Hot channel renamed `channel` → `update_channel` to avoid
  `std::sync::mpsc` shadowing.

### Fixed

- Calibrator line-search NaN guards and subnormal-step guard;
  `update_online` gradient sign correction.
- Platt skew detection with SGD fallback when the closed-form solve
  is ill-conditioned; Hessian singularity guarded.
- `FeedbackStore::adjust` normalised to a kernel-weighted mean to
  bound bias by `strength`; `FeedbackStore` `MAX_CAPACITY` cap and
  relaxed kernel-sum floor.
- LSH seed rotated per instance.
- `MatrixProfile` `MAX_WINDOW` cap; bounded `Debug` on
  `CountMinSketch` and `RandomCutForest`.
- `PrefixRateCap` rollover tightened with a `compare_exchange_weak`
  loop; buckets cache-line-padded.
- `MetricsSink` dispatch batched every 64 ops; `update_channel`
  capacity validated at construction.
- 17 broken intra-doc links repaired; strict rustdoc enabled
  (`-D rustdoc::broken_intra_doc_links`).
- Stale `AuditRecord` / `conformance.md` doc links repaired.

### Performance

- Hot-path allocation scrub: `FeedbackStore` switched to `VecDeque`,
  `score_trimmed` uses a thread-local scratch buffer, LSH clusterer
  keys on a `u128` FNV-1a hash.
- `default_sink()` backed by a `LazyLock<Arc<NoopSink>>` shared
  static (zero allocation on hot path).
- `#[inline]` added to EWMA + scalar-score micro-functions.
- Hot sketch helpers inlined.
- New benches: `accumulator_update_micro`, `accumulator_z_score_micro`,
  8-thread `PrefixRateCap` contention bench, full workspace bench
  matrix.

### Security & supply chain

- `#![forbid(unsafe_code)]` across all four crates.
- HMAC-SHA256 audit chain (`anomstream-triage::AuditChain`) for
  tamper-evident analytic exports.
- Deserialize payload size caps (256 MiB / 1 GiB) with explicit
  `from_*_with_max_size` escape hatches; persistence trust model
  documented.
- Caller-supplied sketch sizing capped to prevent allocation DoS.
- `non_exhaustive` on public enums and pub-field structs prevents
  consumer match-exhaustiveness from breaking on additive change.
- Weekly `cargo audit` + `cargo deny check` in CI.
- Explicit `pub use` enumeration so SemVer guarantees travel
  through `cargo publish`, not through prose.

### Tests & quality

- 7 added bench groups (`group_scores`, `attribution_stability`, …).
- Default-features facade smoke test.
- Property-based fuzz suite (`proptest`).
- AWS SageMaker conformance matrix exercised by the test suite.
- Detection-quality regression guards on TSB-AD-M slices.

### Docs

- `README.md` reshaped: scope catalogue, feature matrix, three
  consumption patterns (default, full facade, fine-grained,
  member-direct).
- `docs/features.md` (~1700 lines) — full primitive catalogue.
- `docs/conformance_rcf.md` — AWS SageMaker conformance matrix +
  comparison against `krcf` and the AWS Java port.
- `docs/performance.md` — bench grids and methodology.
- `docs/threat_model.md` — security posture + persistence trust
  model.
- Per-crate + workspace CI matrix and publishing notes.

### CI

- Weekly + manual CI in addition to push CI.
- Dedicated `release.yml`: calendar versioning, version stamping,
  full release gate, dependency-ordered publish dry-run, tag +
  GitHub Release publish job.
- All GitHub Actions pinned to latest major versions.
- Strict rustdoc (`RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links"`).
- Zero-warning policy (`RUSTFLAGS="-D warnings"`,
  `cargo clippy -- -D warnings`).
