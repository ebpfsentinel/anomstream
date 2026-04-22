#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::bool_to_int_with_if
)]
//! Detection-quality integration tests — measure whether the
//! forest actually distinguishes anomalies from normal points on
//! synthetic streams with known ground truth.
//!
//! Speed alone (criterion benches) says nothing about whether the
//! scorer catches the anomalies. These tests report
//!
//! - **`AUC`** — area under the `ROC` curve. Perfect ranking = 1.0,
//!   random = 0.5. RCF on trivial separable data should sit very
//!   close to 1.0.
//! - **Precision / recall at a budget** — top-K alerts ranked by
//!   score; how many true anomalies fall in the top-K?
//! - **Score separation ratio** — `mean(anomaly_scores) /
//!   mean(normal_scores)`. A value close to `1` means the scorer
//!   cannot tell the classes apart; values > 2 indicate strong
//!   separation.
//!
//! These are **not** a substitute for the Numenta NAB / Yahoo S5
//! benchmarks (tracked under future work in `docs/performance.md`)
//! — but they validate the core quality claim on controlled data
//! and regression-guard it so future refactors cannot silently
//! break detection accuracy.

use anomstream_rs::ForestBuilder;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Compute the area under the ROC curve via trapezoidal rule on
/// the (fpr, tpr) pairs induced by every possible threshold.
/// `labels[i] = 1` for anomaly, `0` for normal.
fn auc(scores: &[f64], labels: &[u8]) -> f64 {
    assert_eq!(scores.len(), labels.len());
    let mut pairs: Vec<(f64, u8)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    // Descending by score — highest score first.
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    let total_pos: u64 = labels.iter().map(|&l| u64::from(l)).sum();
    let total_neg: u64 = labels.len() as u64 - total_pos;
    if total_pos == 0 || total_neg == 0 {
        return 0.5;
    }

    let mut auc = 0.0_f64;
    let mut tp = 0_u64;
    let mut fp = 0_u64;
    let mut prev_tpr = 0.0_f64;
    let mut prev_fpr = 0.0_f64;
    for (_, label) in &pairs {
        if *label == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let tpr = tp as f64 / total_pos as f64;
        #[allow(clippy::cast_precision_loss)]
        let fpr = fp as f64 / total_neg as f64;
        // Trapezoidal area from (prev_fpr, prev_tpr) → (fpr, tpr).
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    auc
}

/// Precision / recall when the top-`k` highest-scoring points are
/// alerted. Returns `(precision, recall)`.
fn precision_recall_at_k(scores: &[f64], labels: &[u8], k: usize) -> (f64, f64) {
    let mut pairs: Vec<(f64, u8)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    let top_k = pairs.iter().take(k);
    let true_in_top: u64 = top_k.clone().filter(|(_, l)| *l == 1).count() as u64;
    let total_pos: u64 = labels.iter().map(|&l| u64::from(l)).sum();
    #[allow(clippy::cast_precision_loss)]
    let precision = if k == 0 {
        0.0
    } else {
        true_in_top as f64 / k as f64
    };
    #[allow(clippy::cast_precision_loss)]
    let recall = if total_pos == 0 {
        0.0
    } else {
        true_in_top as f64 / total_pos as f64
    };
    (precision, recall)
}

/// Mean score separation: `mean(anomaly_scores) / mean(normal_scores)`.
/// `NaN` when either class is empty.
fn score_separation(scores: &[f64], labels: &[u8]) -> f64 {
    let mut anom_sum = 0.0_f64;
    let mut anom_n = 0_u64;
    let mut norm_sum = 0.0_f64;
    let mut norm_n = 0_u64;
    for (&s, &l) in scores.iter().zip(labels.iter()) {
        if l == 1 {
            anom_sum += s;
            anom_n += 1;
        } else {
            norm_sum += s;
            norm_n += 1;
        }
    }
    if anom_n == 0 || norm_n == 0 {
        return f64::NAN;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        (anom_sum / anom_n as f64) / (norm_sum / norm_n as f64)
    }
}

#[test]
fn auc_perfect_ranking_is_1() {
    // All anomalies above all normals.
    let scores = vec![1.0, 0.9, 0.8, 0.1, 0.2, 0.3];
    let labels = vec![1_u8, 1, 1, 0, 0, 0];
    assert!((auc(&scores, &labels) - 1.0).abs() < 1.0e-9);
}

#[test]
fn auc_random_ranking_near_half() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut scores = Vec::with_capacity(1000);
    let mut labels = Vec::with_capacity(1000);
    for _ in 0..1000 {
        scores.push(rng.random::<f64>());
        labels.push(if rng.random::<f64>() < 0.1 { 1_u8 } else { 0 });
    }
    let a = auc(&scores, &labels);
    assert!(
        (a - 0.5).abs() < 0.1,
        "expected near 0.5 for random, got {a}"
    );
}

/// Tight cluster + far outliers — the textbook case. RCF should
/// rank every outlier above every in-cluster point.
///
/// Two-phase protocol: train the forest on a **clean baseline**
/// window, then score a mixed evaluation set. This matches a
/// realistic agent deployment where the first minutes of traffic
/// are assumed benign (baselining) and anomaly detection kicks in
/// once the forest is warm.
#[test]
fn cluster_plus_outliers_separable() {
    let mut forest = ForestBuilder::<2>::new()
        .num_trees(100)
        .sample_size(64)
        .seed(2026)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Phase 1 — warm the forest on 500 normal points only.
    for _ in 0..500 {
        let p = [rng.random::<f64>() * 0.2, rng.random::<f64>() * 0.2];
        forest.update(p).unwrap();
    }

    // Phase 2 — evaluation set: 200 normal + 25 anomalies.
    let mut points: Vec<[f64; 2]> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    for _ in 0..200 {
        points.push([rng.random::<f64>() * 0.2, rng.random::<f64>() * 0.2]);
        labels.push(0);
    }
    for _ in 0..25 {
        points.push([
            8.0 + rng.random::<f64>() * 1.0,
            8.0 + rng.random::<f64>() * 1.0,
        ]);
        labels.push(1);
    }

    let scores: Vec<f64> = points
        .iter()
        .map(|p| f64::from(forest.score(p).unwrap()))
        .collect();

    let a = auc(&scores, &labels);
    let sep = score_separation(&scores, &labels);
    let (p, r) = precision_recall_at_k(&scores, &labels, 25);
    assert!(a > 0.95, "AUC = {a}, expected > 0.95");
    assert!(sep > 1.5, "separation = {sep}, expected > 1.5");
    assert!(p > 0.6, "precision@25 = {p}, expected > 0.6");
    assert!(r > 0.6, "recall@25 = {r}, expected > 0.6");
}

/// Transition anomalies — forest trained on a tight cluster,
/// then queried with a mix of in-cluster and transition points
/// (halfway between baseline and a hypothetical new mode).
/// Transition points should rank above in-cluster ones.
#[test]
fn transition_points_score_above_baseline() {
    let mut forest = ForestBuilder::<2>::new()
        .num_trees(100)
        .sample_size(64)
        .seed(17)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    // Phase 1 — warm on 800 baseline points around (0, 0).
    for _ in 0..800 {
        forest
            .update([rng.random::<f64>() * 0.2, rng.random::<f64>() * 0.2])
            .unwrap();
    }
    // Phase 2 — evaluation: 200 baseline + 50 transition points
    // (far from the trained centroid).
    let mut points: Vec<[f64; 2]> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    for _ in 0..200 {
        points.push([rng.random::<f64>() * 0.2, rng.random::<f64>() * 0.2]);
        labels.push(0);
    }
    for _ in 0..50 {
        points.push([
            2.0 + rng.random::<f64>() * 0.2,
            2.0 + rng.random::<f64>() * 0.2,
        ]);
        labels.push(1);
    }
    let scores: Vec<f64> = points
        .iter()
        .map(|p| f64::from(forest.score(p).unwrap()))
        .collect();
    let a = auc(&scores, &labels);
    assert!(a > 0.90, "transition AUC = {a}, expected > 0.90");
}

/// Streaming score — bare forest's `score` call precedes `update`,
/// so every point gets a prediction **before** it contaminates the
/// forest's baseline. Confirms the online-scoring protocol matches
/// eBPFsentinel Enterprise's per-packet deployment.
#[test]
fn online_score_then_update_preserves_separation() {
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(64)
        .seed(7)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut scores_normal = Vec::with_capacity(500);
    let mut scores_anom = Vec::with_capacity(50);
    // Pre-warm.
    for _ in 0..200 {
        let p: [f64; 4] = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        forest.update(p).unwrap();
    }
    // Online evaluation phase — interleave normal + anomaly.
    for i in 0..500 {
        let p: [f64; 4] = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        scores_normal.push(f64::from(forest.score(&p).unwrap()));
        forest.update(p).unwrap();
        if i % 10 == 0 {
            let anom: [f64; 4] = [5.0, 5.0, 5.0, 5.0];
            scores_anom.push(f64::from(forest.score(&anom).unwrap()));
            // Do NOT update on anomaly — realistic: true anomalies
            // should not re-baseline the forest on first sight.
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let mean_normal = scores_normal.iter().sum::<f64>() / scores_normal.len() as f64;
    #[allow(clippy::cast_precision_loss)]
    let mean_anom = scores_anom.iter().sum::<f64>() / scores_anom.len() as f64;
    let sep = mean_anom / mean_normal.max(1.0e-9);
    assert!(
        sep > 2.0,
        "online separation = {sep} (normal {mean_normal}, anom {mean_anom})"
    );
}

#[test]
fn precision_recall_at_k_extremes() {
    // All anomalies at the top → precision@k == recall@k == 1 when
    // k equals number of anomalies.
    let scores = vec![1.0, 0.9, 0.8, 0.1, 0.2];
    let labels = vec![1_u8, 1, 1, 0, 0];
    let (p, r) = precision_recall_at_k(&scores, &labels, 3);
    assert!((p - 1.0).abs() < 1.0e-9);
    assert!((r - 1.0).abs() < 1.0e-9);
}
