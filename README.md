# anomstream

A Rust toolkit for **streaming anomaly detection**: several detector families (multivariate, per-feature, score-level) and the primitives that turn them into a pipeline - streaming stats, sketches, calibration, alert clustering, SOC triage, hot-path ingress. Streaming, bounded memory, online update throughout.

The Random Cut Forest is a focused port of Guha et al. (ICML 2016) within the AWS SageMaker bounds, not a feature match of `randomcutforest-by-aws`. The toolkit powers the ML pipeline of the **eBPFsentinel Enterprise** agent.

**Out of scope**: protocol parsers, IP-centric trackers, L7 intelligence; ONNX / torch runtimes, supervised training; rule synthesis, policy engines; density estimation, forecasting, GLAD, near-neighbour lists, `impute()` (the RCF imputation idea survives only as the `forensic_baseline` triage helper).

## Catalogue

### Multivariate anomaly detectors

Operate on the joint `[f64; D]` distribution.

- `RandomCutForest<D>` - AWS-conformant aggregate root (Guha 2016)
- `ThresholdedForest<D>` - adaptive threshold wrapper (TRCF)
- `ShingledForest` - scalar-stream temporal wrapper over the forest
- `MatrixProfile` - STOMP exact batch time-series discord / motif (complements `ShingledForest`)
- `DynamicForest` - runtime-dim variant
- `DriftAwareForest` - shadow-swap recovery when a drift detector fires
- `TenantForestPool` - bounded per-tenant forest pool with LRU eviction

### Per-feature univariate detectors

One accumulator per dimension, finds _which_ feature drifted.

- `PerFeatureEwma<D>` - parallel univariate EWMA z-score detector
- `PerFeatureCusum<D>` - parallel two-sided CUSUM change-point detector
- `FeatureDriftDetector<D>` - PSI / KL distributional drift on raw features

### Score-level drift + regime change

Operate on a scalar anomaly-score stream.

- `MetaDriftDetector` - two-sided CUSUM on the score stream
- `AdwinDetector` - adaptive windowing (Bifet 2007)
- `PotDetector` - SPOT / DSPOT univariate Peaks-Over-Threshold (Siffer 2017)
- `fisher_combine` - combine `k` independent p-values into one test statistic

### Streaming stats + sketches

Bounded-memory summaries reused across detectors.

- `OnlineStats` - Welford streaming mean + variance
- `TDigest` - Dunning streaming quantile digest
- `ScoreHistogram` - fixed-bin score histogram
- `CountMinSketch` - probabilistic frequency sketch
- `HyperLogLog` - distinct-count sketch
- `SpaceSaving<K>` - deterministic top-K heavy hitters in `O(K)` memory
- `BloomFilter` - set membership, zero false negatives, tunable FPR
- `Normalizer<D>` - per-feature `MinMax` / `ZScore` / `None`, with a `fit` learner

### Explanation + triage

- `DiVector` + `FeatureGroups` - per-dim and per-group attribution
- `AttributionStability` - inter-tree dispersion + confidence
- `SageEstimator<D>` - SAGE Shapley attribution (Covert 2020)
- `PlattCalibrator` - batch + online-SGD probability calibration
- `SeverityBands` / `Severity` - ordinal severity classification

### SOC + ops

- `AlertClusterer` / `LshAlertClusterer` - cosine and LSH alert dedup (LSH seeded per instance)
- `FeedbackStore` - SOC-label-driven score adjustment, at most `MAX_CAPACITY` (65 536) labels
- `AlertRecord` / `AlertContext` - immutable alert envelope, unknown fields rejected
- `AuditChain` / `verify_audit_chain` - HMAC-SHA256-chained tamper-evident audit trail
- `ForensicBaseline` - post-hoc distance-to-sample summary

### Hot-path ingress

- `UpdateSampler` - stride or per-flow-hash sampler, optionally keyed with a per-instance secret (MITRE ATLAS `AML.T0020`)
- `PrefixRateCap` - per-prefix admission cap over 256 cache-padded buckets, optionally keyed
- `update_channel` / `try_update_channel` - bounded MPSC channel for the classifier / updater split
- `MetricsSink` - pluggable telemetry; hot-path dispatch batched every `METRICS_BATCH_SIZE` (64) calls, `flush_metrics()` on shutdown

### Evaluation

- `vus_pr` / `vus_pr_with_buffer` / `range_auc_pr` - Volume Under Surface PR (Paparrizos VLDB 2022), threshold-free length-aware quality metric
- `TsbAdMDataset` - CSV loader for the TSB-AD-M multivariate benchmark (Liu & Paparrizos NeurIPS 2024)

Per-module detail: [docs/features.md](docs/features.md). Threat model: [docs/threat_model.md](docs/threat_model.md).

## Crate layout

| Crate                            | Role                                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [`anomstream`](meta/)            | Facade: feature-gated re-exports of the three members                                                |
| [`anomstream-core`](core/)       | Detectors, streaming primitives, shared contracts (`MetricsSink`, `SeverityBands`, `ForestSnapshot`) |
| [`anomstream-triage`](triage/)   | SOC layer: Platt, SAGE, alert clustering, feedback store, alert record                               |
| [`anomstream-hotpath`](hotpath/) | Ingress: `UpdateSampler`, `PrefixRateCap`, `update_channel`                                          |

```toml
# Default: core detectors and primitives
[dependencies]
anomstream = "0.0.0-dev"

# Everything, or pick layers
anomstream = { version = "0.0.0-dev", features = ["full"] }
anomstream = { version = "0.0.0-dev", features = ["core", "triage", "serde"] }

# Member crates directly, for per-member SemVer
[dependencies]
anomstream-core   = { version = "0.0.0-dev", features = ["parallel", "serde"] }
anomstream-triage = { version = "0.0.0-dev" }
```

## Quickstart

### Multivariate: Random Cut Forest

```rust,ignore
use anomstream::ForestBuilder;

let mut forest = ForestBuilder::<4>::new()
    .num_trees(100)
    .sample_size(256)
    .seed(42)
    .build()?;

for point in stream_of_points {
    forest.update(point)?;
    let score = forest.score(&point)?;
    if f64::from(score) > 1.5 {
        eprintln!("anomaly: {score}");
    }
}
# Ok::<(), anomstream::RcfError>(())
```

### Per-feature: two-sided CUSUM change-point

```rust,ignore
use anomstream::{PerFeatureCusum, PerFeatureCusumConfig};

let mut det = PerFeatureCusum::<4>::new(PerFeatureCusumConfig {
    slack: 0.5,
    threshold: 5.0,
});

for point in stream_of_points {
    let result = det.observe(&point);
    for alert in &result.alerts {
        eprintln!(
            "drift on feature {} ({:?}) magnitude {:.2}",
            alert.feature_index, alert.direction, alert.magnitude
        );
    }
}
```

They compose: feed the forest score into `MetaDriftDetector` for regime change, run `PerFeatureCusum` alongside to name the drifting feature, wrap the forest in `ThresholdedForest` for adaptive alerting.

## Algorithms

- **Random Cut Forest** - Guha, Mishra, Roy, Schrijvers, _Robust Random Cut Forest Based Anomaly Detection on Streams_, ICML 2016. Reservoir sampling without replacement: Park, Ostrouchov, Samatova, Geist - SIAM SDM 2004.
- **EWMA** - Hunter, _The Exponentially Weighted Moving Average_, JQT 18(4), 1986.
- **CUSUM** - Page, _Continuous Inspection Schemes_, Biometrika 41, 1954. Two-sided variant: Hawkins & Olwell, 1998.
- **ADWIN** - Bifet, _Learning from Time-Changing Data with Adaptive Windowing_, SIAM SDM 2007.
- **SPOT / DSPOT** - Siffer et al., _Anomaly Detection in Streams with Extreme Value Theory_, KDD 2017.
- **t-digest** - Dunning, _Computing Extremely Accurate Quantiles using t-Digests_, 2019.
- **Count-Min Sketch** - Cormode & Muthukrishnan, JoA 55(1), 2005.
- **HyperLogLog** - Flajolet, Fusy, Gandouet, Meunier - AofA 2007. _HyperLogLog in Practice_: Heule, Nunkesser, Hall, EDBT 2013.
- **Space-Saving** - Metwally, Agrawal, El Abbadi, _Efficient Computation of Frequent and Top-k Elements in Data Streams_, ICDT 2005.
- **Bloom filter** - Bloom, _Space/Time Trade-offs in Hash Coding with Allowable Errors_, CACM 13(7), 1970. Double-hashing: Kirsch & Mitzenmacher, _Less Hashing, Same Performance_, ESA 2006.
- **Matrix Profile / STOMP** - Zhu, Zimmerman, Senobari, Yeh, Funning, Mueen, Brisk, Keogh, _Matrix Profile II: Exploiting a Novel Algorithm and GPUs…_, ICDM 2016. Original MP: Yeh et al., _Matrix Profile I_, ICDM 2016.
- **VUS-PR** - Paparrizos, Boniol, Palpanas, Tsay, Elmore, Franklin, _Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection_, VLDB 2022.
- **TSB-AD-M** - Liu, Paparrizos, _The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark_, NeurIPS 2024.
- **SAGE** - Covert, Lundberg, Lee, _Understanding Global Feature Contributions Through Additive Importance Measures_, NeurIPS 2020.
- **Welford variance** - Welford, Technometrics 4(3), 1962.

RCF hyperparameter bounds follow AWS SageMaker and are enforced at build time: [docs/conformance_rcf.md](docs/conformance_rcf.md).

## Features

| Cargo feature     | Default | Role                                                                                                                          |
| ----------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `core`            | ✅      | Re-export of `anomstream-core` (bare forest + primitives)                                                                     |
| `std`             | ✅      | Standard library support (unlocks the full module surface)                                                                    |
| `triage`          | ❌      | Re-export of `anomstream-triage` (Platt, SAGE, LSH, feedback, audit)                                                          |
| `hotpath`         | ❌      | Re-export of `anomstream-hotpath` (sampler, rate cap, MPSC channel)                                                           |
| `parallel`        | ❌      | Per-tree / batch parallelism via `rayon` (implies `std`)                                                                      |
| `serde`           | ❌      | State serialisation                                                                                                           |
| `postcard`        | ❌      | Compact binary persistence (implies `serde`)                                                                                  |
| `serde_json`      | ❌      | JSON persistence (implies `serde`)                                                                                            |
| `audit-integrity` | ❌      | HMAC-SHA256-chained tamper-evident `AuditChain` (pulls `hmac` + `sha2` + `subtle`; implies `triage + std + serde + postcard`) |
| `full`            | ❌      | Convenience alias for `core + triage + hotpath + std + parallel + serde + postcard + serde_json + audit-integrity`            |

The member crates ship `default = []`; enable `std`, `serde` and the rest explicitly when depending on them directly.

### Module availability table

| Module / type                                                                                                                         | Requires feature              |
| ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `RandomCutForest`, `ThresholdedForest`, `RcfConfig`, `ForestBuilder`                                                                  | always (core, no_std + alloc) |
| `OnlineStats`, `Normalizer`, `PerFeatureEwma`, `PerFeatureCusum`, `FeatureDriftDetector`, `MetaDriftDetector`                         | always (core, no_std + alloc) |
| `TDigest`, `ScoreHistogram`, `ForensicBaseline`, `SeverityBands`, `AttributionStability`, `BootstrapReport`                           | always (core, no_std + alloc) |
| `AdwinDetector`, `PotDetector`, `ensemble::fisher_combine`                                                                            | `std`                         |
| `CountMinSketch`, `HyperLogLog`, `SpaceSaving`, `BloomFilter`                                                                         | `std`                         |
| `ShingledForest`, `DynamicForest`, `DriftAwareForest`, `TenantForestPool`, `MatrixProfile`                                            | `std`                         |
| `TsbAdMDataset`, `vus_pr` / `range_auc_pr`                                                                                            | `std`                         |
| `AlertClusterer`, `AlertRecord`, `FeedbackStore`, `PlattCalibrator`, `SageEstimator`, `LshAlertClusterer`                             | `triage` (+ `std`)            |
| `AuditChain`, `AuditChainEntry`, `verify_audit_chain`, `AUDIT_CHAIN_*` consts                                                         | `audit-integrity`             |
| `UpdateSampler`, `PrefixRateCap`, `update_channel`, `try_update_channel`, `MetricsSink`, `MAX_CHANNEL_CAPACITY`, `METRICS_BATCH_SIZE` | `hotpath` (+ `std`)           |

### `no_std` + `alloc`

Everything marked "always" above runs under `#![no_std]` with `alloc`; transcendentals go through `libm`, hash maps fall back to `BTreeMap`.

```toml
[dependencies]
anomstream = { version = "…", default-features = false, features = ["core"] }
# Optional: serde persistence under no_std
anomstream = { version = "…", default-features = false, features = ["core", "serde"] }
```

CI checks `--no-default-features`, with and without `serde`.

## Performance

Bench matrix, reference figures and how to run them: [docs/performance.md](docs/performance.md).

## Quality evaluation (TSB-AD-M)

The dataset is not bundled: [thedatumorg/TSB-AD](https://github.com/thedatumorg/TSB-AD) (~1 GiB).

```bash
cargo run --release --example tsb_ad_m_eval -- /path/to/TSB-AD-M/MSL_1_001.csv
# MSL_1_001.csv  n=2000  dim=55  pos=123  VUS-PR=0.4312  elapsed=218ms
```

```bash
for f in /path/to/TSB-AD-M/*.csv; do
    cargo run --release --example tsb_ad_m_eval -- "$f"
done | tee vus_pr.log
```

`core/examples/tsb_ad_m_eval.rs` scores with `DynamicForest<128>` on a 50/50 calibration / scoring split; swap the detector there.

## License

[Apache-2.0](LICENSE). Contributions under the same licence.
