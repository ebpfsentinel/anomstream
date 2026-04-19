# AWS SageMaker Conformance

`rcf-rs` enforces the documented AWS `SageMaker` hyperparameter
bounds at build time. Beyond these invariants the library does not
aim for bit-exact parity with
[aws/random-cut-forest-by-aws](https://github.com/aws/random-cut-forest-by-aws) —
feature evolution is driven by eBPFsentinel Enterprise needs.

Regression test: `tests/aws_conformance.rs` pins every row below.

| AWS specification | `rcf-rs` mapping |
|---|---|
| `feature_dim ∈ [1, 10000]` | const-generic `D`, validated by `ForestBuilder::build` |
| `num_trees ∈ [50, 1000]`, default `100` | enforced by `ForestBuilder` |
| `num_samples_per_tree ∈ [1, 2048]`, default `256` | enforced by `ForestBuilder` |
| `time_decay = 0.1 / sample_size` | resolved by `ForestBuilder`; pass `.time_decay(0.0)` to disable |
| `initial_accept_fraction ∈ [0, 1]`, default `1.0` (disabled) | `ForestBuilder::initial_accept_fraction` — pass `0.125` to match AWS `CompactSampler` |
| Reservoir sampling without replacement | `sampler::ReservoirSampler` |
| Score = average across trees | `forest::RandomCutForest::score` |
| Anomaly threshold `≥ 3σ` from mean | `ThresholdedForest` (default `z_factor = 3.0`), else caller responsibility |

Extensions beyond the AWS signature:

- `ForestBuilder::feature_scales([f64; D])` — per-dim pre-scaling
  applied before any hot-path call. `1 / stddev[d]` gives
  unit-variance normalisation without a separate caller pass.
- `ThresholdedForest` — adaptive threshold on top of the bare
  forest, inspired by AWS's `TRCF` in `randomcutforest-parkservices`
  but kept light (no short/long duality, no near-threshold
  heuristics). Builder `z_factor`, `min_threshold`,
  `min_observations`, `score_decay`.
- `forensic_baseline(&point)` — repurposes the AWS `ImputeVisitor`
  concept as a per-dim *"what would this have looked like under
  the live baseline?"* SOC triage helper. Returns raw-space
  `expected / stddev / delta / zscore`.
- `score_early_term` — sequential early-termination scoring on
  converged per-tree means, cuts latency on easy points.

Deliberately absent from `rcf-rs` (out of scope for streaming
network anomaly detection):

- Density estimation (AWS `density()`)
- Forecasting (AWS `RCFCaster`)
- Near-neighbor list (AWS `near_neighbor_list()`)
- Internal shingling + rotation
- GLAD locally adaptive variant
- Label / Attribute generics (`AugmentedRCF`)

## `parallel` and dedicated thread pool

Enable `parallel` to run per-tree work on rayon workers. Pin a
dedicated pool via `ForestBuilder::num_threads` to isolate the
forest from the rest of the application's rayon workload:

```rust,ignore
let forest = ForestBuilder::<16>::new()
    .num_trees(100)
    .sample_size(256)
    .num_threads(4)
    .build()?;
```

`num_threads` is only honoured with `--features parallel`.
