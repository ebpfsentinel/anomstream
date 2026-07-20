# Performance

Benchmark reference for `anomstream`. Numbers come from Criterion
(`cargo bench`, wall-clock mean, `mimalloc` pinned); detection-quality
numbers come from the NAB / TSB-AD-M test corpora.

## Contents

1. [Methodology](#methodology) — harnesses, hardware, caveats
2. [Forest hot path](#forest-hot-path) — single-probe, batch, early-term, delete
3. [Forest design decisions](#forest-design-decisions) — explored / shipped optimisations
4. [Forest variants](#forest-variants) — dynamic-dim, drift-aware, shingled, thresholded
5. [Explainability](#explainability) — attribution, confidence, SAGE, persistence
6. [Companion primitives](#companion-primitives) — sketches, quantiles, per-feature, drift
7. [SOC / triage layer](#soc--triage-layer) — clustering, calibration, tenant pool
8. [Hot-path ingress](#hot-path-ingress) — sampler, rate cap, channel
9. [Matrix profile](#matrix-profile)
10. [Detection quality](#detection-quality) — NAB, TSB-AD-M, external baselines

---

## Methodology

Five Criterion harnesses across the workspace:

| Harness                             | Covers                                                                                                                                              |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core/benches/forest_throughput.rs` | insert, score, attribution, codisp (batched + loop)                                                                                                 |
| `core/benches/extended.rs`          | bulk, early-term, forensic, tenant, stateless codisp, thresholded, delete                                                                           |
| `core/benches/modules.rs`           | 23 groups: shingled, matrix profile, t-digest/histogram, drift, ADWIN, SPOT/DSPOT, Fisher, dynamic-dim, companion primitives, explain/triage extras |
| `triage/benches/modules.rs`         | LSH clustering, Platt calibration, SAGE, cosine clusterer, feedback store                                                                           |
| `hotpath/benches/modules.rs`        | sampler, prefix rate cap, bounded MPSC channel                                                                                                      |

```bash
cargo bench --workspace                                             # full
cargo bench -p anomstream-core --bench modules                      # single harness
cargo bench --workspace -- --sample-size 10 --measurement-time 2    # quick
cargo bench -p anomstream-core --bench modules -- per_feature_ewma/ # single group
```

Criterion HTML reports land in `target/criterion/`.

### Reference hardware

|           |                                                    |
| --------- | -------------------------------------------------- |
| CPU       | Intel Core i7-1370P (13th gen), 14C/20T, L3 24 MiB |
| Memory    | 32 GB DDR5                                         |
| Kernel    | Linux 6.17                                         |
| Allocator | mimalloc 0.1 (pinned in bench harness)             |
| Compiler  | rustc 1.95 stable                                  |

### Caveats

- **Cross-group variance** — `b.iter()` mutates a persistent forest and
  Criterion picks batch sizes per-op, so reservoir state drifts across
  groups. Trust ratios _within_ a group.
- **Cross-session variance** — numbers here are warm-CPU (`performance`
  governor). A cool / powersave session ran the single-probe hot path
  25-30 % faster. Ratios are stable; absolutes are not.
- **Parallel ceiling** — `score_many` plateaus at ~6× on 14 cores,
  memory-bandwidth-bound once the working set spills L3.
- **Fan-out threshold** — single-probe ops below `num_trees × D = 2048`
  run the ensemble serially even with the `parallel` feature on; see
  [Ensemble fan-out threshold](#ensemble-fan-out-threshold). Absolute
  single-probe numbers below predate that change.

---

## Forest hot path

`(trees=100, sample=256, D=16)` unless noted.

### Single-probe ops

`--sample-size 50 --warm-up-time 3 --measurement-time 8`:

| Workload                              | Time      | Throughput |
| ------------------------------------- | --------- | ---------- |
| `forest_update`                       | **34 µs** | ~29 k/s    |
| `forest_score`                        | **34 µs** | ~29 k/s    |
| `forest_attribution`                  | **45 µs** | ~22 k/s    |
| `forest_score_and_attribution`        | **47 µs** | ~21 k/s    |
| `forest_split_score_then_attribution` | **79 µs** | ~13 k/s    |

- Fused `score_and_attribution` is **~41 % faster** than `score` +
  `attribution` separately (one traversal, `47/79 ≈ 0.59`).
- Fused bbox SIMD kernel (`total_probability_of_cut`) saves one
  `min`/`max` pass per internal node.
- Typed-arena refactor (persistence v4) cut leaf-arena memory **−90 %**
  (~320 B → ~40 B / slot).

Across shapes:

| Config           | `forest_update` | `forest_score` | `forest_attribution` |
| ---------------- | --------------- | -------------- | -------------------- |
| `(50, 128, 16)`  | 29 µs           | 25 µs          | —                    |
| `(100, 256, 4)`  | 29 µs           | 30 µs          | 35 µs                |
| `(100, 256, 16)` | 34 µs           | 34 µs          | 45 µs                |
| `(100, 256, 64)` | 104 µs          | 42 µs          | 88 µs                |
| `(200, 512, 16)` | 55 µs           | 52 µs          | —                    |

### Ensemble fan-out threshold

Per-tree work is small — roughly `D × depth`, a few hundred nanoseconds
at the AWS-default shape. That is below rayon's task-dispatch floor, so
splitting a *single* ensemble walk across workers used to cost more than
it saved. The fan-out is now gated on estimated work (`num_trees × D`,
threshold `2048`, `PARALLEL_FANOUT_MIN_WORK` in
`forest/random_cut_forest.rs`): below it the walk stays on the calling
thread, at or above it rayon fans out as before.

Measured back-to-back on one machine, `parallel` feature on in both
columns, only the threshold flipped (`0` = always fan out, the old
behaviour):

| Config           | `trees × D` | `forest_update` | `forest_score` | Arm      |
| ---------------- | ----------- | --------------- | -------------- | -------- |
| `(100, 256, 4)`  | 400         | **2.5× faster** | **1.7×**       | serial   |
| `(50, 128, 16)`  | 800         | **4.9× faster** | **2.9×**       | serial   |
| `(100, 256, 16)` | 1600        | **1.7× faster** | **1.35×**      | serial   |
| `(200, 512, 16)` | 3200        | unchanged       | unchanged      | parallel |
| `(100, 256, 64)` | 6400        | unchanged       | unchanged      | parallel |

The two `parallel`-arm rows run identical code in both columns and serve
as the control group: they moved ≤ 8 %, which sets the noise floor for
the ratios above.

This affects only per-tree fan-out. Batch entry points (`score_many`,
`attribution_many`, `score_codisp_stateless_many`) parallelise across
*points* — each task is a whole ensemble walk, so the fan-out always
pays and is never gated.

Ratios above are trustworthy (same machine, back-to-back, control group
included). The **absolute** µs figures in the tables on this page predate
the gate and were taken in an earlier session; the serial-arm shapes are
now faster than what they show. Re-stamp them from a quiet machine before
quoting them as current.

### Batch scoring

Parallel `score_many` vs serial loop (`D=16`):

| Batch | `score_many` | Serial  | Speedup |
| ----- | ------------ | ------- | ------- |
| 64    | 360 µs       | 2.11 ms | 5.9×    |
| 512   | 3.73 ms      | 17.1 ms | 4.6×    |
| 4096  | 28.6 ms      | 137 ms  | 4.8×    |

Caps at ~6× rayon speedup — past L3 the per-probe working set thrashes
L1/L2 and workers contend on the LLC→DRAM channel.

**Codisp variants** — probe-based (`score_codisp_many`) pre-inserts
probes, shares the leaf→root walk, fans out across trees:

| Batch K | `score_codisp_many` | `score_codisp` loop | Speedup |
| ------- | ------------------- | ------------------- | ------- |
| 16      | 1.76 ms             | 2.39 ms             | 1.4×    |
| 64      | 6.59 ms             | 9.58 ms             | 1.5×    |

Gain caps at ~1.5× — insert/delete mutation still scales `K × trees`.
For frozen-baseline batches at any size prefer **`score_codisp_stateless_many`**
(no reservoir mutation, no `O(K)` saturation):

| Workload                              | Time    |
| ------------------------------------- | ------- |
| `score_codisp_stateless` single probe | 29 µs   |
| `score_codisp_stateless_many` k=16    | 108 µs  |
| `score_codisp_stateless_many` k=64    | 312 µs  |
| `score_codisp_stateless_many` k=256   | 1.07 ms |

Stateless is ~1.1× faster than non-mutating `score()` single-probe (skips
EMA/reservoir update) and **~21× faster than mutating batched codisp**
(312 µs vs 6.59 ms @ k=64) — which is why NAB eval dropped 12.6 s → 1.09 s
after the switch.

### Early termination

Single probe, `score_early_term`:

| Path                                    | Time    |
| --------------------------------------- | ------- |
| `score` (full ensemble)                 | 33 µs   |
| threshold=0.02 (tight)                  | 36 µs   |
| threshold=0.20 (loose, stops ~20 trees) | 4.99 µs |

Loose threshold → **6.6×** on baseline-dominated traffic; a tight
threshold rarely short-circuits and matches a full `score`.

> The `score` row was previously labelled "parallel ensemble". At this
> shape the ensemble walk now runs serially — see
> [Ensemble fan-out threshold](#ensemble-fan-out-threshold).

### Delete

| Workload                                   | Time   |
| ------------------------------------------ | ------ |
| `update_indexed + delete` `(100, 256, 16)` | 115 µs |

~3.4× an update — bbox recompute up the path + arena slot release.
Pair with `update_indexed` for probe workflows; otherwise rely on the
reservoir's cheaper amortised eviction.

---

## Forest design decisions

Optimisations explored against the ~6× memory-bandwidth plateau.

### Cache-aware probe reordering — reverted

A `score_many_locality_sorted` variant quantised leading dims into a
Morton-lite key and sorted batches before dispatch. At `k=1024, D=16`
(correlated cluster): plain 5.10 ms vs sorted 5.69 ms — the `O(N log N)`
sort + double-gather beat the locality gain on uniform batches. Reverted;
callers can re-order their own batches if their workload benefits.

### Packed cut (`u8` dim + `f32` value) — shipped, opt-in

Feature `packed-cut` (off by default) halves the per-internal-node cut
footprint (16 B → 8 B) for tighter L1 fit; caps `D ≤ 256` and quantises
the cut coordinate to `f32`.

- `f32` rounding can push a value sampled in `[lo, hi)` onto `hi` and
  leave the augmented point un-isolated — the sampler pins it back inside
  `[lo, hi)` (`bounding_box::narrow_into_range`).
- Points closer than an `f32` ULP on every dim (un-isolable at the stored
  resolution) collapse onto one leaf as duplicates.
- Snapshots are wire-incompatible with default builds — a version-prefix
  flag makes either build reject the other's payload.

**AUC regression** (real datasets, default `f64` vs packed `f32`,
identical seeds):

| Suite                                      | Cases     | mean \|Δ\| | max \|Δ\| | within ±0.001 |
| ------------------------------------------ | --------- | ---------- | --------- | ------------- |
| NAB `realKnownCause` (23 ablation configs) | 7 series  | **0.000**  | **0.000** | 23/23         |
| TSB-AD-M (`dim ≤ 128`)                     | 113 files | **0.0015** | 0.012     | 89/113 (79 %) |

Deltas are two-signed and the worst single-file swing (0.012 VUS-PR) sits
inside the metric's own run-to-run variance — packed-cut is
**AUC-neutral**. Enable with `--features packed-cut` (or
`anomstream/packed-cut` via the façade) when cut-arena L1 footprint
matters more than exact `f64` parity.

---

## Forest variants

### Dynamic-dim + drift-aware

| Workload                                      | Time  |
| --------------------------------------------- | ----- |
| `DynamicForest::update` active=8 / MAX_D=16   | 32 µs |
| `DriftAwareForest::update` no shadow          | 34 µs |
| `DriftAwareForest::update` with active shadow | 80 µs |

Dynamic zero-pad overhead is tiny (shallower active-8 trees recover it);
the no-shadow wrapper has zero always-on cost; an active shadow runs
primary + shadow sequentially (2.4×).

### Shingled

| Workload                              | Time  |
| ------------------------------------- | ----- |
| `update_scalar` + `score_scalar` D=16 | 71 µs |

≈ update + score + a free ring-buffer push; cost is the downstream forest
ops on the embedded vector.

### Thresholded (TRCF pipeline)

`ThresholdedForest::process` — update + score + EMA + verdict per point:

| Workload                               | Time  |
| -------------------------------------- | ----- |
| `thresholded_process` `(100, 256, 16)` | 73 µs |

≈ `update (34) + score (34)` + ~5 µs EMA/tdigest/threshold logic.

---

## Explainability

### Attribution, confidence, forensic

| Workload                                     | Time     |
| -------------------------------------------- | -------- |
| `forensic_baseline` `(100, 256, 4)`          | 13 µs    |
| `forensic_baseline` `(100, 256, 16)`         | 16 µs    |
| `forensic_baseline` `(100, 1024, 16)`        | 64 µs    |
| `FeatureGroups::group_scores` D=16, 3 groups | 39.0 µs  |
| `attribution_stability` D=16                 | 55.4 µs  |
| `score_with_confidence` D=16                 | 58.4 µs  |
| `bootstrap` 4096 pts, `(50, 128)`            | 187.8 ms |

- `forensic_baseline` ≈ `O(live_points × D)` Welford sweep — linear in
  `sample_size`, ~1.3× over D 4→16.
- `group_scores` ≈ attribution + O(D) post-reduce.
- `attribution_stability` ≈ 1.2× attribution; `score_with_confidence`
  ≈ 1.7× score (non-parallel — needs per-tree outputs in order).
- `bootstrap` ~22 k pts/s on the reduced forest, linear in point count.

### SAGE Shapley attribution

| Workload                                         | Time    |
| ------------------------------------------------ | ------- |
| `SageEstimator::explain` D=16, K=64, `(50, 128)` | 40.3 ms |

`K · D × forest_score` — SOC triage / forensic replay, not per-alert.

### Persistence roundtrip

| Workload                        | Time    |
| ------------------------------- | ------- |
| `to_bytes` `(100, 256, D=16)`   | 5.98 ms |
| `from_bytes` `(100, 256, D=16)` | 7.66 ms |

2.6 MB postcard payload; tree rehydration dominates deserialise.

---

## Companion primitives

`D=16`, no forest involvement. Promoted from the enterprise ML layer into
`anomstream-core` for OSS reuse.

### Per-feature detectors

| Workload                                             | Time                     | Throughput          |
| ---------------------------------------------------- | ------------------------ | ------------------- |
| `OnlineStats::update` (Welford, hot)                 | 5.6 ns                   | ~180 M/s            |
| `OnlineStats::update` (cold, 32-loop)                | 75 ns                    | ~14 M/s             |
| `OnlineStats::variance` / `std_dev` read             | 0.2 ns                   | ~5 G/s              |
| `Normalizer<16>::transform` None / ZScore / MinMax   | 3.5 / 8.0 / 15.3 ns      | ~286 / 125 / 65 M/s |
| `Normalizer<16>::fit` 1024 samples                   | 6.5 µs                   | per-batch           |
| `PerFeatureEwma<16>::observe` warmed / spike / cold  | 129 ns / 96 ns / 1.50 µs | ~7.8 / 10 M/s       |
| `PerFeatureCusum<16>::observe` stable / below / trip | 40 / 68 / 85 ns          | ~25 / 15 / 12 M/s   |

- `Normalizer::None` (3.5 ns) is the memcpy baseline; ZScore adds
  centre+scale, MinMax adds the range clamp.
- EWMA spike < warmed: the zero-variance branch returns `f64::MAX` and
  skips the sqrt.
- CUSUM stable < below-threshold: settled sums skip the `max(0, …)`.

### Sketches

| Workload                                            | Time          | Throughput       |
| --------------------------------------------------- | ------------- | ---------------- |
| `CountMinSketch::increment` / `estimate` w=2048 d=4 | 65 / 61 ns    | ~15 M/s          |
| `CountMinSketch::reset` w=2048 d=4                  | 808 ns        | 64 KiB zero-fill |
| `BloomFilter::insert_bytes` n=1k p=0.01             | 22.6 ns       | ~44 M/s          |
| `BloomFilter::insert_hash` n=100k p=0.01            | 16.8 ns       | ~60 M/s          |
| `BloomFilter::contains_bytes` n=100k p=0.01         | 28.8 ns       | ~35 M/s          |
| `BloomFilter::union` two n=10k filters              | 260 ns        | per-merge        |
| `HyperLogLog::add_bytes` / `add_hash` p=12          | 11.5 / 2.7 ns | ~87 / 370 M/s    |
| `HyperLogLog::estimate` after 100k, p=12            | 18.6 µs       | query-only       |
| `HyperLogLog::merge` two p=12                       | 1.64 µs       | per-merge        |
| `SpaceSaving::observe` hot / evict, K=128           | 9.7 / 310 ns  | ~103 / 3.2 M/s   |
| `SpaceSaving::top_k(10)` from 1024 keys             | 5.6 µs        | per-query        |

- CMS increment ≈ estimate (both hash twice over 4 rows).
- `_hash` Bloom/HLL paths skip the `SipHash` call; `contains` miss-path
  probes every hash, positive lookups short-circuit; `union` scales with
  `m` not `n`.
- SpaceSaving hot ≪ evict: evict pays an `O(K)` min-scan; tables stabilise
  under heavy-hitter load so the hot path dominates.

### Quantiles / histograms

| Workload                             | Time   | Throughput |
| ------------------------------------ | ------ | ---------- |
| `TDigest::record`                    | 42 ns  | ~24 M/s    |
| `TDigest::quantile(0.99)` after 100k | 57 ns  | query-only |
| `ScoreHistogram::record`             | 4.9 ns | ~205 M/s   |

TDigest amortises compaction into the 10×-compression buffer flush;
ScoreHistogram is a bin-index + increment.

### Drift detectors

| Workload                                   | Time    | Throughput |
| ------------------------------------------ | ------- | ---------- |
| `MetaDriftDetector::observe` (CUSUM)       | 8.0 ns  | ~125 M/s   |
| `FeatureDriftDetector::observe` D=16/10bin | 85 ns   | ~12 M/s    |
| `FeatureDriftDetector::psi()` D=16/10bin   | 1.17 µs | query-only |
| `AdwinDetector::update` cap=4096           | 26.3 µs | ~38 k/s    |

ADWIN's `O(N)` prefix-sum dominates at cap=4096 — **not** per-packet
material. Run it on the score stream (one update per alert) or use
`MetaDriftDetector` (8 ns) instead.

---

## SOC / triage layer

### Clustering, calibration, ensemble, audit

| Workload                                          | Time              | Throughput          |
| ------------------------------------------------- | ----------------- | ------------------- |
| `LshAlertClusterer::hash_divector` D=16           | 73 ns             | ~13.7 M/s           |
| `LshAlertClusterer::observe` D=16                 | 95 ns             | ~10.5 M/s           |
| `AlertClusterer::observe` D=16 window=32 (cosine) | 771 ns            | ~1.3 M/s            |
| `PotDetector::record` / `p_value` post-freeze     | 43 / 8.2 ns       | ~23 / 122 M/s       |
| `PlattCalibrator::fit` 2048 samples               | 1.75 ms           | offline             |
| `PlattCalibrator::calibrate` single score         | 25.6 ns           | ~39 M/s             |
| `ensemble::fisher_combine` k=8 / 32 / 128         | 43 / 161 / 642 ns | ~24 / 6.2 / 1.6 M/s |
| `FeedbackStore::label` capacity=256               | 571 ns            | ~1.8 M/s            |
| `FeedbackStore::adjust` 512 labels, D=16          | 8.6 µs            | ~116 k/s            |
| `AuditChain::append` D=4 (HMAC-SHA256 + postcard) | 426 ns            | ~2.3 M/s            |
| `verify_audit_chain` 256 entries D=4              | 121 µs            | ~470 ns/entry       |

- LSH `observe` (95 ns, zero-alloc) is ~8× faster than cosine
  `AlertClusterer` at window=32 — prefer LSH at MSSP volume (>10k/min).
- SPOT `p_value` ~5× faster than `record` (closed-form GPD survival on
  cached γ, σ); Platt `calibrate` is two floats + a σ.
- `fisher_combine` ~5 ns/p-value (Kahan sum + χ² tail).
- FeedbackStore `adjust` scales with stored labels (Gaussian-kernel sum).

### Tenant pool at scale

Each tenant `D=4` / `(50, 64)`, warmed 128 samples:

| N   | `similarity_matrix` | `score_across_tenants` | `most_similar_top5` |
| --- | ------------------- | ---------------------- | ------------------- |
| 32  | 37 µs               | 126 µs                 | 0.30 µs             |
| 128 | 99 µs               | 470 µs                 | 1.12 µs             |
| 512 | 586 µs              | 2.69 ms                | 5.01 µs             |

`N=32→512` (16×): `similarity_matrix` (O(N²) parallel) ~16×,
`score_across_tenants` (O(N)) 21×, `most_similar_top5` (O(N·log k)) 17×
— rayon hides the quadratic until core saturation.

---

## Hot-path ingress

Per-call overhead on the classifier hot path:

| Workload                                             | Time       | Throughput       |
| ---------------------------------------------------- | ---------- | ---------------- |
| `UpdateSampler::accept_stride` keep=8                | 28 ns      | ~36 M/s          |
| `UpdateSampler::accept_hash` unkeyed / keyed keep=8  | 14 / 15 ns | ~73 / 67 M/s     |
| `PrefixRateCap::check_and_record` 100/1s             | 11.6 ns    | ~86 M/s          |
| `PrefixRateCap::check_and_record` 8-thread contended | ~9 µs/op   | contention floor |
| `update_channel::try_enqueue` cap=4096 (+ drain)     | 487 ns     | ~2.1 M/s         |
| `metrics::default_sink()` shared-Arc clone           | 12 ns      | ~85 M/s          |

- `accept_hash` beats `accept_stride` (skips the counter atomic; admission
  is multiply + mod). Keyed adds ~1.2 ns murmur-mix.
- `PrefixRateCap` 11.6 ns reflects the Acquire-load short-circuit on the
  valid-window case + batched metrics (1 sink call / 64 ops, down from
  18 ns, −36 %); the CAS loop fires once per window roll, not per packet.
  Buckets are `#[repr(C, align(64))]` (16 KiB) to avoid false sharing.
- Channel throughput is `sync_channel`-lock-bound; 2.1 M/s per producer
  covers typical TC/XDP rates with producer fan-out.

---

## Matrix profile

`MatrixProfile::compute` — exact STOMP discord/motif, `O(n²)` time,
`O(n)` memory:

| Workload                                   | Time    |
| ------------------------------------------ | ------- |
| `compute` n=1024, window=32                | 3.1 ms  |
| `compute` n=2048, window=64                | 12.3 ms |
| `compute` n=4096, window=128               | 49.3 ms |
| `discord_topk(5)` cached n=2048, window=64 | 27.8 µs |

Scaling matches `O(n²)` (doubling n ≈ 4× cost); window affects only the
seed column. `discord_topk` is trivial once cached — reuse the profile.
Use as a forensic complement to `ShingledForest`: the online path flags a
region, STOMP gives the exact discord inside it.

---

## Detection quality

### NAB `realKnownCause`

Same embedding pipeline (32-lag → warm-phase z-score → EMA α=0.02), 15 %
warm, 100 trees × 256 sample. Three scoring APIs:

| File                                   | `score()` | `score_codisp()` | `…_stateless()` | rrcf  | AWS Java  |
| -------------------------------------- | --------- | ---------------- | --------------- | ----- | --------- |
| `ambient_temperature_system_failure`   | **0.813** | **0.813**        | 0.793           | 0.734 | 0.786     |
| `cpu_utilization_asg_misconfiguration` | 0.953     | **0.969**        | 0.963           | 0.849 | 0.906     |
| `ec2_request_latency_system_failure`   | **0.709** | 0.706            | 0.621           | 0.481 | 0.482     |
| `machine_temperature_system_failure`   | 0.578     | **0.817**        | 0.815           | 0.880 | 0.883     |
| `nyc_taxi`                             | **0.698** | 0.636            | 0.623           | 0.571 | 0.540     |
| `rogue_agent_key_hold`                 | 0.145     | 0.198            | 0.181           | 0.535 | **0.633** |
| `rogue_agent_key_updown`               | **0.633** | 0.579            | 0.563           | 0.657 | 0.542     |
| **weighted aggregate**                 | 0.719     | **0.776**        | 0.763           | 0.748 | 0.757     |

`score_codisp()` (0.776) leads both rrcf (0.748) and AWS Java (0.757);
stateless (0.763) preserves the frozen baseline and runs 12× faster.
`tests/nab.rs` pins the aggregate floor at 0.70.

**Ablation** (`examples/nab_ablation.rs`, same corpus):

| Config                                  | Aggregate AUC |
| --------------------------------------- | ------------- |
| baseline (lag=8, raw score)             | 0.615         |
| lag=32                                  | 0.665         |
| lag=32 + diff                           | 0.640         |
| lag=32 + zscore                         | 0.683         |
| lag=32 + zscore + smooth(0.05)          | 0.718         |
| **lag=32 + zscore + smooth(0.02)**      | **0.719**     |
| trcf-online D=32                        | 0.320         |
| probe-score D=8 (naive hack)            | 0.330         |
| **codisp D=32 + zscore + smooth(0.02)** | **0.776**     |

- Longer embedding (+0.050) and warm-phase z-score (+0.018) help; RCF cuts
  are range-weighted, so un-normalised dims let one channel dominate.
- EMA smoothing (α≈0.02, half-life ~35) cuts noise (+0.036).
- Differencing regresses — NAB's signal is in absolute values.
- TRCF-online collapses (0.72 → 0.32): the EMA threshold adapts up during
  multi-day windows. Frozen baseline is correct for NAB.
- The naive `update→score→delete` hack (0.330) ranks the just-inserted
  probe as seen; proper codisp walks leaf→root then deletes.

### TSB-AD-M (multivariate)

200 series across 16 source datasets, per-point labels, native
multivariate. Per-dim z-score on the `tr_<N>` split, frozen baseline,
EMA α=0.02, forest `(100, 256)`, seed 2026. Const-generic whitelist covers
192/200 files (96 %); eight D=248 files skipped. Point-wise ROC-AUC
weighted by positive count:

| Source dataset         | Files             | `score()` | `score_codisp()` | `…_stateless()` | AWS Java  |
| ---------------------- | ----------------- | --------- | ---------------- | --------------- | --------- |
| Genesis                | 1                 | 0.968     | **0.991**        | **0.994**       | 0.982     |
| SMAP                   | 27                | 0.803     | **0.823**        | 0.716           | 0.805     |
| SMD                    | 22                | 0.618     | **0.760**        | 0.752           | 0.806     |
| MSL                    | 16                | **0.705** | 0.746            | 0.599           | 0.762     |
| SVDB                   | 31                | 0.692     | 0.737            | **0.779**       | 0.757     |
| LTDB                   | 5                 | 0.601     | 0.755            | **0.758**       | 0.755     |
| Exathlon               | 27                | 0.491     | 0.894            | **0.996**       | 0.865     |
| MITDB                  | 13                | 0.597     | **0.678**        | 0.603           | 0.660     |
| PSM                    | 1                 | 0.608     | 0.595            | **0.613**       | 0.611     |
| CATSv2                 | 6                 | **0.580** | 0.547            | 0.496           | 0.547     |
| CreditCard             | 1                 | 0.589     | 0.679            | 0.658           | **0.693** |
| Daphnet                | 1                 | 0.309     | 0.885            | **0.926**       | 0.944     |
| GECCO                  | 1                 | 0.412     | 0.523            | **0.753**       | 0.594     |
| GHL                    | 25                | 0.454     | 0.461            | **0.570**       | 0.419     |
| OPPORTUNITY            | 8 (skipped D=248) | —         | —                | —               | 0.298     |
| SWaT                   | 2                 | 0.282     | **0.825**        | 0.715           | 0.825     |
| TAO                    | 13                | 0.451     | 0.453            | **0.487**       | 0.471     |
| **aggregate weighted** | **192 / 200**     | 0.583     | **0.768**        | 0.751           | 0.753     |

- `score()` — isolation depth, rayon-parallel, full scan; the hot-path
  API. Floor pinned at 0.55 (regression guard, not a quality claim).
- `score_codisp()` — probe-based, stride-capped to 50 k eval rows/file;
  leads aggregate **0.768** vs AWS Java 0.753.
- `score_codisp_stateless()` — no mutation, full eval stream, preserves
  the frozen baseline at scale; **0.751**, within noise of AWS Java.

Caveats: this is point-wise ROC-AUC; the official leaderboard ranks on
**VUS-PR** (now in-crate, `examples/tsb_ad_m_eval.rs`). RCF is classical —
transformer SOTA wins on heavy-physics datasets (SWaT, Daphnet, GECCO)
where the signature is higher-order cross-channel; RCF stays competitive
where per-dim statistical drift dominates (Genesis/SMAP/MSL/SVDB), closer
to eBPFsentinel's production feature mix.

### External baselines (synthetic)

10k points, `D=16`, 1 % outliers, 30 % warm / 70 % eval, frozen baseline,
5-seed variance (2026–2030), mean ± stddev (CV in parens):

| Impl                                        | Backend             | Updates/s            | Scores/s                | AUC       |
| ------------------------------------------- | ------------------- | -------------------- | ----------------------- | --------- |
| `anomstream-core` `score()` (1 seed)        | Rust, rayon         | **31 500**           | **197 900**             | 1.000     |
| `anomstream-core` `score_codisp()` (1 seed) | Rust, parallel walk | —                    | 8 150                   | 1.000     |
| `anomstream-core` `score()` (5-seed)        | Rust, rayon         | 17 500 ± 1 240 (7 %) | 125 900 ± 1 840 (1.5 %) | 1.000 ± 0 |
| `randomcutforest-java` 4.4.0                | JVM 26, cold        | 2 090 ± 134 (6 %)    | 8 870 ± 415 (5 %)       | 1.000 ± 0 |
| `rrcf` 0.4.4                                | Python + NumPy      | 73 ± 3 (4 %)         | 94 150 ± 4 840 (5 %)    | 0.992 ± 0 |
| `sklearn.IsolationForest`                   | NumPy + Cython      | batch-only           | 136 300 ± 2 450 (2 %)   | 1.000 ± 0 |

- **Updates**: ~8.4× faster than AWS Java, ~240× faster than rrcf, well
  outside the 5-7 % noise floor.
- **Scores (fast path)**: sklearn edges `score()` by 8 % (~3σ); rrcf
  trails ~25 %, AWS Java ~14×.
- **Scores (codisp)**: mutating per probe → ~8 k/s, ~25× slower than
  `score()`; matches AWS Java / rrcf semantics — SOC triage, not hot path.
- **AUC**: identical within precision (0.992 rrcf, 1.000 others).

Ratios are portable; absolute numbers vary with thermal state (an earlier
cool-CPU session hit ~32k/203k for `score()`).

### Reproduce

```bash
# Synthetic baseline sweep
scripts/synthetic/variance_sweep.sh /tmp/aws-rcf/randomcutforest-core-4.4.0.jar

# NAB
git clone --depth 1 https://github.com/numenta/NAB.git /opt/nab
RCF_NAB_PATH=/opt/nab cargo test --test nab --all-features -- --ignored --nocapture
python3 scripts/nab/bench_rrcf_nab.py --nab /opt/nab
java -cp "scripts/nab:/tmp/aws-rcf/randomcutforest-core-4.4.0.jar" RcfBenchNab /opt/nab

# TSB-AD-M
scripts/tsb_ad/fetch.sh /tmp/tsb-ad
RCF_TSB_AD_M_PATH=/tmp/tsb-ad/TSB-AD-M \
    cargo test --release --test tsb_ad_m --all-features -- --ignored --nocapture
python3 scripts/tsb_ad/bench_rrcf_tsb_ad.py \
    --dir /tmp/tsb-ad/TSB-AD-M --max-eval 1500 --workers "$(nproc)"
javac -cp /tmp/aws-rcf-central/randomcutforest-core-4.4.0.jar scripts/tsb_ad/RcfBenchTsbAd.java
java -cp scripts/tsb_ad:/tmp/aws-rcf-central/randomcutforest-core-4.4.0.jar \
    RcfBenchTsbAd /tmp/tsb-ad/TSB-AD-M 50000
```
