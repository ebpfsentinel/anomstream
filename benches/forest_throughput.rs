//! Criterion benchmarks for [`rcf_rs::RandomCutForest`] insert and
//! score throughput across a representative `(num_trees, sample_size,
//! dim)` matrix. Targets ≥ 100k inserts/sec and ≥ 50k scores/sec at
//! the AWS-default `(100, 256, 16)` configuration on a typical
//! `x86_64` dev box.
//!
//! Forests are now const-generic over `D`, so each `dim` is its own
//! monomorphisation. The matrix entries are expanded inline through
//! the [`bench_dim`] helper, parameterised by the target dimension.
//!
//! The bench process pins `mimalloc` as the global allocator —
//! reduces small-allocation overhead vs the system allocator, which
//! is the same trick callers can apply at their `main.rs` to free
//! a few percent on every `update`/`score` call.
//!
//! Run with `cargo bench`. Compile-only check via `cargo bench --no-run`.

#![allow(clippy::cast_precision_loss)]

use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use mimalloc::MiMalloc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rcf_rs::{ForestBuilder, RandomCutForest};
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Tuning matrix at the AWS-default `D = 16`: `(num_trees,
/// sample_size)` pairs spanning the AWS minima through the centre.
const TUNING_MATRIX: &[(usize, usize)] = &[
    (50, 64),
    (50, 128),
    (50, 256),
    (100, 64),
    (100, 128),
    (100, 256),
];

fn build_warm_forest<const D: usize>(
    num_trees: usize,
    sample_size: usize,
    seed: u64,
) -> RandomCutForest<D> {
    let mut forest = ForestBuilder::<D>::new()
        .num_trees(num_trees)
        .sample_size(sample_size)
        .seed(seed)
        .build()
        .expect("AWS-conformant config");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..(sample_size * 4) {
        let mut p = [0.0_f64; D];
        for slot in &mut p {
            *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
        }
        forest.update(p).expect("update succeeds");
    }
    forest
}

fn bench_update_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d");
    group.bench_function(&id, |b| {
        let mut forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        b.iter(|| {
            let mut p = [0.0_f64; D];
            for slot in &mut p {
                *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
            }
            forest.update(black_box(p)).expect("update succeeds");
        });
    });
}

fn bench_score_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d");
    group.bench_function(&id, |b| {
        let forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        b.iter(|| {
            let mut p = [0.0_f64; D];
            for slot in &mut p {
                *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
            }
            let s = forest.score(black_box(&p)).expect("score succeeds");
            black_box(s);
        });
    });
}

fn bench_attribution_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d");
    group.bench_function(&id, |b| {
        let forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        b.iter(|| {
            let mut p = [0.0_f64; D];
            for slot in &mut p {
                *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
            }
            let di = forest
                .attribution(black_box(&p))
                .expect("attribution succeeds");
            black_box(di);
        });
    });
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_update");
    bench_update_for::<16>(&mut group, 50, 128);
    bench_update_for::<4>(&mut group, 100, 256);
    bench_update_for::<16>(&mut group, 100, 256);
    bench_update_for::<64>(&mut group, 100, 256);
    bench_update_for::<16>(&mut group, 200, 512);
    group.finish();
}

fn bench_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_score");
    bench_score_for::<16>(&mut group, 50, 128);
    bench_score_for::<4>(&mut group, 100, 256);
    bench_score_for::<16>(&mut group, 100, 256);
    bench_score_for::<64>(&mut group, 100, 256);
    bench_score_for::<16>(&mut group, 200, 512);
    group.finish();
}

fn bench_attribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_attribution");
    bench_attribution_for::<4>(&mut group, 100, 256);
    bench_attribution_for::<16>(&mut group, 100, 256);
    bench_attribution_for::<64>(&mut group, 100, 256);
    group.finish();
}

/// Tuning sweep: at the AWS-default `D = 16`, vary
/// `(num_trees, sample_size)` so callers can pick a config matching
/// their precision / latency budget.
fn bench_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_tuning_dim16");
    for &(num_trees, sample_size) in TUNING_MATRIX {
        let update_id = format!("update_{num_trees}t_{sample_size}s");
        group.bench_function(&update_id, |b| {
            let mut forest = build_warm_forest::<16>(num_trees, sample_size, 2026);
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            b.iter(|| {
                let mut p = [0.0_f64; 16];
                for slot in &mut p {
                    *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
                }
                forest.update(black_box(p)).expect("update succeeds");
            });
        });
        let score_id = format!("score_{num_trees}t_{sample_size}s");
        group.bench_function(&score_id, |b| {
            let forest = build_warm_forest::<16>(num_trees, sample_size, 2026);
            let mut rng = ChaCha8Rng::seed_from_u64(11);
            b.iter(|| {
                let mut p = [0.0_f64; 16];
                for slot in &mut p {
                    *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
                }
                let s = forest.score(black_box(&p)).expect("score succeeds");
                black_box(s);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_score,
    bench_attribution,
    bench_tuning
);
criterion_main!(benches);
