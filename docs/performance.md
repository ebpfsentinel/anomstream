# Performance

Criterion benches (`cargo bench`), wall-clock mean point estimate
on `x86_64` with `mimalloc` pinned globally. Two bench files:

- `benches/forest_throughput.rs` — core ops (insert, score,
  attribution) across the `(trees, samples, D)` matrix.
- `benches/extended.rs` — value-add APIs: bulk, early-term,
  forensic, tenant.

Quick run with smaller sample: `cargo bench -- --sample-size 10
--warm-up-time 1 --measurement-time 2`. Full run (default
criterion config): `cargo bench`.

## Core ops (`forest_throughput`)

| Workload | `(trees, samples, D)` | Time |
|---|---|---|
| `forest_update` | `(50, 128, 16)` | 35.91 µs |
| `forest_update` | `(100, 256, 4)` | 31.89 µs |
| `forest_update` | `(100, 256, 16)` | 47.98 µs |
| `forest_update` | `(100, 256, 64)` | 104.93 µs |
| `forest_update` | `(200, 512, 16)` | 84.91 µs |
| `forest_score` | `(50, 128, 16)` | 26.60 µs |
| `forest_score` | `(100, 256, 4)` | 37.08 µs |
| `forest_score` | `(100, 256, 16)` | 38.88 µs |
| `forest_score` | `(100, 256, 64)` | 46.62 µs |
| `forest_score` | `(200, 512, 16)` | 67.05 µs |
| `forest_attribution` | `(100, 256, 4)` | 72.21 µs |
| `forest_attribution` | `(100, 256, 16)` | 131.26 µs |
| `forest_attribution` | `(100, 256, 64)` | 150.39 µs |

At `(100, 256, 16)`: ~21k inserts/s and ~26k scores/s
single-thread-equivalent.

## Tuning sweep at `D = 16`

`forest_tuning_dim16` bench group:

| `(num_trees, sample_size)` | `update` | `score` |
|---|---|---|
| `(50, 64)` | 32.44 µs | 27.71 µs |
| `(50, 128)` | 35.98 µs | 27.97 µs |
| `(50, 256)` | 43.30 µs | 30.41 µs |
| `(100, 64)` | 36.85 µs | 35.13 µs |
| `(100, 128)` | 41.78 µs | 37.41 µs |
| `(100, 256)` | 50.75 µs | 37.61 µs |

## Bulk batch scoring

`bulk_scoring` bench group, `D=16`, forest `(100, 256)`, batches
of random probes:

| Batch size | `score_many` (par) | Serial for-loop | Speedup |
|---|---|---|---|
| 64 | 773 µs | 3.99 ms | 5.2× |
| 512 | 5.39 ms | 32.6 ms | 6.1× |
| 4096 | 40.2 ms | 257.6 ms | 6.4× |

Speedup grows with batch size as rayon amortises task-scheduling
overhead across more work.

## Early-termination scoring

`early_term` bench group, `D=16`, forest `(100, 256)`, single
probe:

| Path | Time |
|---|---|
| `score` (full parallel ensemble) | 59 µs |
| `score_early_term`, `threshold=0.02` (tight, rarely stops) | 79 µs |
| `score_early_term`, `threshold=0.20` (loose, stops ~20 trees) | 3.8 µs |

Tight threshold is slower than plain `score` because it walks
trees sequentially and rarely short-circuits — the parallel
ensemble wins when ambiguity forces a full traversal. Loose
threshold gives a **~15× speedup** on baseline-dominated traffic
where most points stop early.

## Forensic baseline

`forensic_baseline` bench group, `D` and `sample_size` swept:

| `(trees, samples, D)` | Time |
|---|---|
| `(100, 256, 4)` | 248 µs |
| `(100, 256, 16)` | 245 µs |
| `(100, 1024, 16)` | 1.05 ms |

Cost is dominated by the `O(live_points × D)` Welford sweep over
the union of tenant reservoirs. Quadrupling `sample_size` → ~4×
slower. Per-dim cost is marginal vs. the iteration overhead.

## Tenant pool at scale

`tenant_pool` bench group, each tenant `D=4` / `(50, 64)`, warmed
with 128 samples:

| N tenants | `similarity_matrix` | `score_across_tenants` | `most_similar_top5` |
|---|---|---|---|
| 32 | 3.4 µs | 1.52 ms | 1.37 µs |
| 128 | 153 µs | 6.61 ms | 5.19 µs |
| 512 | 2.65 ms | 34.5 ms | 24.1 µs |

Observations:
- `similarity_matrix` is O(N²) on EMA-stat pairs (confirmed by
  N=32→512 giving ~780× longer for 16× more tenants).
- `score_across_tenants` is O(N) — one `score_only` per tenant,
  linearly scaling (32→512 gives ~23× for 16× more tenants).
- `most_similar_top5` is O(N) scan + `O(N log N)` sort — still
  microsecond-scale up to 512 tenants.
