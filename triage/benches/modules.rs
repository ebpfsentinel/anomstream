//! Criterion bench suite for the triage layer modules on top of
//! `anomstream-core`: LSH alert clustering, Platt calibration,
//! SAGE Shapley attribution.
//!
//! Run with `cargo bench -p anomstream-triage --bench modules`.

#![allow(clippy::cast_precision_loss)]

use anomstream_core::{AnomalyScore, DiVector, ForensicBaseline, ForestBuilder, RandomCutForest};
use anomstream_triage::alert_cluster::MAX_TENANTS_PER_CLUSTER;
use anomstream_triage::{
    AlertClusterer, AlertContext, AlertRecord, FeedbackLabel, FeedbackStore, LshAlertClusterer,
    PlattCalibrator, PlattFitConfig, SageEstimator,
};
use criterion::{Criterion, criterion_group, criterion_main};
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// `LshAlertClusterer::hash_divector` + `observe` — quantise +
/// bucket lookup on a `DiVector<16>`.
fn bench_lsh_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_cluster");

    let mut di = DiVector::zeros(16);
    di.add_high(3, 2.5).expect("add_high");
    di.add_low(7, 1.5).expect("add_low");
    let rec: AlertRecord<u32, 16> = AlertRecord::new(
        Some(1_u32),
        0,
        [0.0; 16],
        AnomalyScore::new(1.0).expect("score"),
        None,
        None,
        di.clone(),
        ForensicBaseline::<16> {
            observed: [0.0; 16],
            expected: [0.0; 16],
            stddev: [0.0; 16],
            delta: [0.0; 16],
            zscore: [0.0; 16],
            live_points: 0,
        },
    );

    group.bench_function("hash_divector_d16", |b| {
        let clusterer = LshAlertClusterer::default_lsh();
        b.iter(|| {
            let h = clusterer.hash_divector(black_box(&di));
            black_box(h);
        });
    });

    group.bench_function("observe_d16", |b| {
        let mut clusterer = LshAlertClusterer::default_lsh();
        b.iter(|| {
            let (h, d) = clusterer.observe(black_box(&rec));
            black_box((h, d));
        });
    });

    // Same shape as `hash_divector_d16` but on a `with_seed`
    // clusterer — confirms the per-instance seed mix adds zero
    // measurable overhead vs the default constructor (the seed
    // is one extra XOR pre-loop and one extra wrapping_mul +
    // XOR post-loop).
    group.bench_function("hash_divector_d16_seeded", |b| {
        let clusterer =
            LshAlertClusterer::with_seed(16, 8.0, 0xdead_beef_cafe_f00d_dead_beef_cafe_f00d)
                .expect("with_seed");
        b.iter(|| {
            let h = clusterer.hash_divector(black_box(&di));
            black_box(h);
        });
    });

    group.finish();
}

/// `PlattCalibrator::fit` (Newton-Raphson) + `calibrate` (σ).
fn bench_calibrator(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibrator");

    group.bench_function("fit_2048_samples", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let data: Vec<(f64, bool)> = (0..2048)
            .map(|_| {
                let s: f64 = rng.random::<f64>() * 5.0;
                (s, s > 3.0)
            })
            .collect();
        b.iter(|| {
            let cal =
                PlattCalibrator::fit(black_box(&data), PlattFitConfig::default()).expect("fit");
            black_box(cal);
        });
    });

    // Singular-Hessian path: every score collapses to the same
    // value so `det = 0`. Validates the bail-out short-circuits
    // promptly instead of NaN-spinning to `max_iters`.
    group.bench_function("fit_singular_hessian", |b| {
        let data: Vec<(f64, bool)> = (0..512_u32)
            .map(|i| (1.0_f64, i.is_multiple_of(2)))
            .collect();
        b.iter(|| {
            let cal =
                PlattCalibrator::fit(black_box(&data), PlattFitConfig::default()).expect("fit");
            black_box(cal);
        });
    });

    group.bench_function("calibrate_single", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let data: Vec<(f64, bool)> = (0..1024)
            .map(|_| {
                let s: f64 = rng.random::<f64>() * 5.0;
                (s, s > 3.0)
            })
            .collect();
        let cal = PlattCalibrator::fit(&data, PlattFitConfig::default()).expect("fit");
        let mut rng_live = ChaCha8Rng::seed_from_u64(7);
        b.iter(|| {
            let s: f64 = rng_live.random::<f64>() * 5.0;
            let p = cal.calibrate(black_box(s));
            black_box(p);
        });
    });

    group.finish();
}

/// `SageEstimator::explain` — permutation Shapley with default
/// 64 permutations. Expensive: `K · D` forest scores per call.
fn bench_sage(c: &mut Criterion) {
    let mut group = c.benchmark_group("sage");
    group.sample_size(20);

    group.bench_function("explain_d16_k64", |b| {
        let mut forest: RandomCutForest<16> = ForestBuilder::<16>::new()
            .num_trees(50)
            .sample_size(128)
            .seed(2026)
            .build()
            .expect("forest");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..512 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random();
            }
            forest.update(p).expect("update");
        }
        let sage: SageEstimator<16> = SageEstimator::default_anchor([0.0_f64; 16]).expect("sage");
        let probe: [f64; 16] = core::array::from_fn(|_| rng.random::<f64>());
        b.iter(|| {
            let e = sage.explain(&forest, black_box(&probe)).expect("explain");
            black_box(e);
        });
    });

    group.finish();
}

/// `AlertClusterer::observe` — cosine-similarity clustering.
/// Target: per-record cost of the `observe` path (similarity
/// scan + decision) under a typical 32-alert window.
fn bench_alert_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("alert_cluster");

    let mut forest: RandomCutForest<16> = ForestBuilder::<16>::new()
        .num_trees(50)
        .sample_size(128)
        .seed(2026)
        .build()
        .expect("forest");
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..512 {
        let mut p = [0.0_f64; 16];
        for slot in &mut p {
            *slot = rng.random::<f64>();
        }
        forest.update(p).expect("update");
    }

    // Pre-build a pool of 64 varying records so consecutive observes
    // mix between Joined (similar) and NewCluster (dissimilar) paths.
    let records: Vec<AlertRecord<u32, 16>> = (0..64_u32)
        .map(|i| {
            let mut p = [0.0_f64; 16];
            for (k, slot) in p.iter_mut().enumerate() {
                let k_u32 = u32::try_from(k).unwrap_or(0);
                *slot = rng.random::<f64>() + f64::from(i % 4) * f64::from(k_u32);
            }
            let ctx = AlertContext::<u32>::for_tenant(i, u64::from(i) * 1_000);
            AlertRecord::from_forest(&forest, &p, &ctx).expect("rec")
        })
        .collect();

    group.bench_function("observe_d16_window_32", |b| {
        let mut clusterer: AlertClusterer<u32, 16> =
            AlertClusterer::new(0.95, 60_000).expect("clusterer");
        // Warm the window so observe exercises similarity scan.
        for r in records.iter().take(16) {
            let _ = clusterer.observe(r.clone());
        }
        let mut idx = 16;
        b.iter(|| {
            let r = records[idx % records.len()].clone();
            idx += 1;
            let d = clusterer.observe(black_box(r));
            black_box(d);
        });
    });

    // High-cardinality tenant churn against a single cluster:
    // shared attribution + rotating tenant key floods the
    // contributing_tenants rolodex. Validates that capping at
    // MAX_TENANTS_PER_CLUSTER keeps per-observe cost bounded.
    let churn_count: u32 = u32::try_from(MAX_TENANTS_PER_CLUSTER).expect("cap fits u32") * 4;
    let shared_attr_records: Vec<AlertRecord<u32, 16>> = (0..churn_count)
        .map(|tenant| {
            let mut high = vec![0.0_f64; 16];
            high[0] = 1.0;
            let attr = DiVector::from_arrays(high, vec![0.0; 16]).expect("divector");
            AlertRecord::new(
                Some(tenant),
                u64::from(tenant),
                [0.0; 16],
                AnomalyScore::new(1.0).expect("score"),
                None,
                None,
                attr,
                ForensicBaseline::<16> {
                    observed: [0.0; 16],
                    expected: [0.0; 16],
                    stddev: [0.0; 16],
                    delta: [0.0; 16],
                    zscore: [0.0; 16],
                    live_points: 0,
                },
            )
        })
        .collect();

    group.bench_function("observe_d16_tenant_churn_capped", |b| {
        let mut clusterer: AlertClusterer<u32, 16> =
            AlertClusterer::new(0.5, u64::MAX).expect("clusterer");
        let mut idx = 0;
        b.iter(|| {
            let r = shared_attr_records[idx % shared_attr_records.len()].clone();
            idx += 1;
            let d = clusterer.observe(black_box(r));
            black_box(d);
        });
    });

    group.finish();
}

/// `FeedbackStore::label` + `adjust` — SOC label ingestion and
/// Gaussian-kernel score adjustment. Two variants: hot-path
/// `adjust` cost (every probe), warm-up `label` cost (analyst
/// cadence, much rarer).
fn bench_feedback(c: &mut Criterion) {
    let mut group = c.benchmark_group("feedback");
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    // Warm store with 256 benign + 256 confirmed labels spread
    // across the unit cube so `adjust` walks a full kernel sum.
    let labelled: Vec<([f64; 16], FeedbackLabel)> = (0..512)
        .map(|i| {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random::<f64>();
            }
            let label = if i % 2 == 0 {
                FeedbackLabel::Benign
            } else {
                FeedbackLabel::Confirmed
            };
            (p, label)
        })
        .collect();

    group.bench_function("label_hot_capacity_256", |b| {
        let mut store: FeedbackStore<16> = FeedbackStore::new(256, 1.0, 1.0).expect("fb");
        let mut i = 0_usize;
        b.iter(|| {
            let (p, lab) = labelled[i % labelled.len()];
            i += 1;
            let _ = store.label(black_box(p), black_box(lab));
        });
    });

    group.bench_function("adjust_hot_512_labels", |b| {
        let mut store: FeedbackStore<16> = FeedbackStore::new(1024, 1.0, 1.0).expect("fb");
        for (p, lab) in &labelled {
            let _ = store.label(*p, *lab);
        }
        let probe: [f64; 16] = core::array::from_fn(|_| rng.random::<f64>());
        b.iter(|| {
            let adjusted = store.adjust(black_box(&probe), black_box(1.5_f64));
            black_box(adjusted);
        });
    });

    group.finish();
}

/// `AuditChain::append` + `verify_chain` cost on a 256-entry
/// chain — characterises HMAC-SHA256 + postcard-encode overhead
/// per emission and the linear walk on verification.
#[cfg(all(feature = "audit-integrity", feature = "postcard"))]
fn bench_audit_chain(c: &mut Criterion) {
    use anomstream_triage::audit::{AlertContext, AlertRecord};
    use anomstream_triage::audit_chain::{AuditChain, GENESIS_PREV, verify_chain};

    let mut group = c.benchmark_group("audit_chain");
    group.sample_size(20);

    let mut forest: RandomCutForest<4> = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(2026)
        .build()
        .expect("forest");
    for i in 0..32_u32 {
        let v = f64::from(i) * 0.01;
        forest
            .update([v, v + 0.1, v + 0.2, v + 0.3])
            .expect("update");
    }
    let key = [0x42u8; 32];

    group.bench_function("append_d4", |b| {
        let mut chain: AuditChain<String, 4> = AuditChain::new(&key).expect("chain");
        let ctx = AlertContext::<String>::for_tenant("t1".into(), 0);
        let rec = AlertRecord::from_forest(&forest, &[5.0; 4], &ctx).expect("rec");
        b.iter(|| {
            let entry = chain.append(black_box(rec.clone())).expect("append");
            black_box(entry);
        });
    });

    group.bench_function("verify_chain_256_entries", |b| {
        let mut chain: AuditChain<String, 4> = AuditChain::new(&key).expect("chain");
        let entries: Vec<_> = (0..256_u64)
            .map(|ts| {
                let ctx = AlertContext::<String>::for_tenant("t1".into(), ts);
                let rec = AlertRecord::from_forest(&forest, &[5.0; 4], &ctx).expect("rec");
                chain.append(rec).expect("append")
            })
            .collect();
        b.iter(|| {
            verify_chain(black_box(&entries), &key, &GENESIS_PREV).expect("verify");
        });
    });

    group.finish();
}

#[cfg(all(feature = "audit-integrity", feature = "postcard"))]
criterion_group!(
    benches,
    bench_lsh_cluster,
    bench_calibrator,
    bench_sage,
    bench_alert_cluster,
    bench_feedback,
    bench_audit_chain
);
#[cfg(not(all(feature = "audit-integrity", feature = "postcard")))]
criterion_group!(
    benches,
    bench_lsh_cluster,
    bench_calibrator,
    bench_sage,
    bench_alert_cluster,
    bench_feedback
);
criterion_main!(benches);
