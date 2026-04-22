# anomstream-rs

Streaming anomaly detection toolkit for Rust — Random Cut Forest
(Guha et al. ICML 2016) plus a suite of companion primitives
(per-feature drift, calibration, clustering, sketches, SOC triage)
that compose into real detection pipelines.

Powers the ML detection pipeline of the **eBPFsentinel Enterprise**
NDR agent.

## Scope

The core forest stays a focused implementation of the 2016 paper,
not an attempt to match every feature of AWS's
`randomcutforest-by-aws` port. Beyond the forest, the crate ships
**companion primitives** reused across detectors — streaming stats,
drift detectors, normalisers, frequency sketches — promoted from
the enterprise layer so OSS consumers and downstream crates get
them for free.

Out of scope: protocol parsers, IP-centric trackers, ONNX / torch
runtimes, rule synthesis, L7 intelligence. Features well outside
the streaming-anomaly charter (density estimation, forecasting,
GLAD variant, near-neighbour list, …) are intentionally absent —
the imputation concept is repurposed as a SOC-triage
`forensic_baseline` helper rather than a feature-completion
`impute()` call.

### Catalogue

**Core forest layer**

- `RandomCutForest<D>` — AWS-conformant aggregate root
- `ThresholdedForest<D>` — adaptive threshold wrapper (TRCF)
- `DynamicForest` / `ShingledForest` / `DriftAwareForest` — runtime-dim, scalar-stream temporal, and shadow-swap recovery variants
- `TenantForestPool` — bounded per-tenant pool with LRU eviction

**Companion primitives**

- `OnlineStats` — Welford streaming mean + variance
- `CountMinSketch` — probabilistic frequency sketch (std-gated)
- `Normalizer<D>` — per-feature `MinMax` / `ZScore` / `None` transforms
- `PerFeatureEwma<D>` — parallel univariate EWMA z-score detector
- `PerFeatureCusum<D>` — parallel two-sided CUSUM change-point detector
- `SeverityBands` / `Severity` — ordinal classification (shared SOC vocabulary)

**Drift + scoring**

- `MetaDriftDetector` — CUSUM on the score stream
- `FeatureDriftDetector<D>` — PSI / KL drift on raw features
- `AdwinDetector` — adaptive windowing
- `PotDetector` — SPOT / DSPOT univariate Peaks-Over-Threshold
- `fisher_combine` — Fisher p-value combination
- `TDigest`, `ScoreHistogram` — streaming quantiles

**Explain + triage**

- `DiVector` + `FeatureGroups` — per-dim and per-group attribution
- `SageEstimator<D>` — SAGE Shapley attribution
- `PlattCalibrator` — batch + online-SGD probability calibration
- `AttributionStability` — inter-tree dispersion + confidence

**SOC + ops**

- `AlertClusterer` / `LshAlertClusterer` — cosine + LSH alert dedup
- `FeedbackStore` — SOC-label-driven score adjustment
- `AuditRecord` — immutable alert envelope
- `ForensicBaseline` — post-hoc distance-to-sample summary

**Hot-path ingress**

- `hot_path::UpdateSampler` / `PrefixRateCap` / `channel` — stride + hash + keyed sampler, 256-bucket atomic counter sketch, bounded MPSC channel for classifier/updater thread split
- `MetricsSink` — pluggable telemetry (`NoopSink` + your own impl)

See [docs/features.md](docs/features.md) for the full module
catalogue with per-feature rationale.

## Quickstart

```rust,ignore
use anomstream_rs::ForestBuilder;

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
# Ok::<(), anomstream_rs::RcfError>(())
```

## Algorithm

Guha, Mishra, Roy, Schrijvers — *Robust Random Cut Forest Based
Anomaly Detection on Streams*, ICML 2016.

Reservoir sampling without replacement: Park, Ostrouchov, Samatova,
Geist — SIAM SDM 2004.

AWS `SageMaker` hyperparameter bounds are enforced at build time
(`feature_dim`, `num_trees`, `num_samples_per_tree`, `time_decay`).
Details: [docs/conformance.md](docs/conformance.md).

## Features

| Cargo feature | Default | Role |
|---|---|---|
| `std` | ✅ | Standard library support |
| `parallel` | ✅ | Per-tree / batch parallelism via `rayon` (implies `std`) |
| `serde` | ✅ | State serialisation |
| `postcard` | ✅ | Compact binary persistence (implies `serde`) |
| `serde_json` | ❌ | JSON persistence (implies `serde`) |

### `no_std` + `alloc`

`default-features = false` drops the runtime layer (MPSC channel,
tenant pool, drift-aware shadow swap, ADWIN, LSH clustering, SAGE,
SPOT/DSPOT, feedback store, shingled forest, dynamic forest,
`CountMinSketch`) and leaves the core forest + trees + reservoir
sampler + thresholded layer + meta / feature drift detectors +
t-digest + alert clusterer + bootstrap + calibrator + forensic
baseline + audit record + severity bands + companion primitives
(`OnlineStats`, `Normalizer<D>`, `PerFeatureEwma<D>`,
`PerFeatureCusum<D>`) running under `#![no_std]` with `alloc`.
Transcendentals (`ln`, `sqrt`, `exp`, …) route through `num-traits`
+ `libm`; hashing-dependent code paths fall back to
`alloc::collections::BTreeMap`.

```toml
[dependencies]
anomstream-rs = { version = "…", default-features = false }
# Optional: serde persistence under no_std
anomstream-rs = { version = "…", default-features = false, features = ["serde"] }
```

The `no_std` configuration is gated in CI (`cargo check
--no-default-features` + `--features serde`).

## Performance

See [docs/performance.md](docs/performance.md) for the full
criterion bench matrix.

## License

[Apache-2.0](LICENSE). Contributions under the same licence.
