//! Criterion bench suite for `anomstream-hotpath`: eBPF-style
//! ingress primitives (`UpdateSampler`, `PrefixRateCap`, bounded
//! MPSC `channel`). Run with
//! `cargo bench -p anomstream-hotpath --bench modules`.

#![allow(clippy::cast_precision_loss)]

use anomstream_hotpath::{PrefixRateCap, UpdateSampler, update_channel};
use core::num::{NonZeroU32, NonZeroU64};
use criterion::{Criterion, criterion_group, criterion_main};
use mimalloc::MiMalloc;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// `UpdateSampler` `accept_stride` + `accept_hash` + keyed
/// `accept_hash` (murmur-mix secret). Target: per-packet overhead
/// on the classifier hot path.
fn bench_hot_path_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_sampler");

    group.bench_function("accept_stride_keep_8", |b| {
        let s = UpdateSampler::new(8);
        b.iter(|| {
            let v = s.accept_stride();
            black_box(v);
        });
    });

    group.bench_function("accept_hash_keep_8", |b| {
        let s = UpdateSampler::new(8);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let h: u64 = rng.random();
            let v = s.accept_hash(black_box(h));
            black_box(v);
        });
    });

    group.bench_function("accept_hash_keyed_keep_8", |b| {
        let s = UpdateSampler::new_keyed(8).expect("getrandom");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let h: u64 = rng.random();
            let v = s.accept_hash(black_box(h));
            black_box(v);
        });
    });

    group.finish();
}

/// `PrefixRateCap::check_and_record` - 256-bucket atomic counter
/// sketch + window roll. Lock-free, `O(1)`.
fn bench_hot_path_prefix_cap(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_prefix_cap");

    group.bench_function("check_and_record_100cap_1s", |b| {
        let cap = PrefixRateCap::new(
            NonZeroU32::new(100).expect("non-zero"),
            NonZeroU64::new(1_000).expect("non-zero"),
        );
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let mut now_ms = 0_u64;
        b.iter(|| {
            let h: u64 = rng.random();
            now_ms = now_ms.wrapping_add(1);
            let v = cap.check_and_record(black_box(h), now_ms);
            black_box(v);
        });
    });

    // Quantify the gain from batched metrics emission. With
    // METRICS_BATCH_SIZE = 64 the sink dispatch lands once per 64
    // ops on the noop sink path; this bench drives the bare hot
    // path so any regression in the batching helper surfaces here.
    group.bench_function("check_and_record_batched_dispatch", |b| {
        let cap = PrefixRateCap::new(
            NonZeroU32::new(1_000_000).expect("non-zero"),
            NonZeroU64::new(60_000).expect("non-zero"),
        );
        let mut h: u64 = 0;
        b.iter(|| {
            h = h.wrapping_add(1);
            let v = cap.check_and_record(black_box(h), 0);
            black_box(v);
        });
    });

    // Multi-thread contention measurement: 8 threads hammer 8
    // distinct buckets concurrently. Without cache-line padding
    // the adjacent `AtomicU32` buckets share a 64-byte line,
    // forcing every cross-thread write to bounce the line through
    // MOESI/MESI - measurable as a >5× per-op slowdown on a
    // typical x86_64 box. Cache-padded buckets (current layout)
    // keep this bench near the single-thread cost.
    group.bench_function("check_and_record_contended_8threads", |b| {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;
        let cap = Arc::new(PrefixRateCap::new(
            NonZeroU32::new(u32::MAX).expect("non-zero"),
            NonZeroU64::new(60_000).expect("non-zero"),
        ));
        b.iter_custom(|iters| {
            let stop = Arc::new(AtomicBool::new(false));
            // 7 background threads keep the other cache lines hot -
            // each one targets a distinct bucket so every write
            // lands on its own cache line. Without padding, false
            // sharing across these adjacent atomics would dominate
            // the measurement.
            let bg: Vec<_> = (1..8_u64)
                .map(|t| {
                    let cap = Arc::clone(&cap);
                    let stop = Arc::clone(&stop);
                    thread::spawn(move || {
                        while !stop.load(Ordering::Relaxed) {
                            let _ = cap.check_and_record(t, 0);
                        }
                    })
                })
                .collect();
            // Foreground thread runs the timed loop on bucket 0.
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let v = cap.check_and_record(black_box(0), 0);
                black_box(v);
            }
            let elapsed = start.elapsed();
            stop.store(true, Ordering::Relaxed);
            for h in bg {
                h.join().expect("thread join");
            }
            elapsed
        });
    });

    group.finish();
}

/// Bounded MPSC `update_channel::try_enqueue` with background
/// drain thread keeping the queue non-full.
fn bench_hot_path_channel(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_channel");

    group.bench_function("try_enqueue_4096cap", |b| {
        let (producer, consumer) = update_channel::<16>(4096);
        let drain = std::thread::spawn(move || while consumer.recv().is_some() {});
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random();
            }
            let ok = producer.try_enqueue(black_box(p));
            black_box(ok);
        });
        drop(producer);
        let _ = drain.join();
    });

    group.finish();
}

/// `default_sink()` - cost of the shared-Arc clone path used by
/// every `UpdateSampler::new` / `update_channel` / detector
/// constructor. Previously a per-call `Arc::new(NoopSink)` heap
/// allocation; now a refcount bump on a lazily-initialised
/// process-wide static.
fn bench_default_sink(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_default_sink");
    group.bench_function("clone_shared_noop", |b| {
        b.iter(|| {
            let sink = anomstream_core::metrics::default_sink();
            black_box(sink);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_hot_path_sampler,
    bench_hot_path_prefix_cap,
    bench_hot_path_channel,
    bench_default_sink
);
criterion_main!(benches);
