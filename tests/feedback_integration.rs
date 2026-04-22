#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_precision_loss)]
//! Integration coverage for SOC feedback ingestion —
//! `FeedbackStore` composes with `RandomCutForest::score` without
//! mutating the forest. Verifies the label/adjust contract on
//! realistic forest-score streams.

#![cfg(feature = "std")]

use anomstream_rs::{FeedbackLabel, FeedbackStore, ForestBuilder};

fn small_forest() -> anomstream_rs::RandomCutForest<2> {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(2026)
        .build()
        .unwrap();
    for i in 0..500 {
        let v = f64::from(i) * 0.01;
        f.update([v, v + 0.5]).unwrap();
    }
    f
}

#[test]
fn benign_label_pulls_score_down_on_exact_probe() {
    let forest = small_forest();
    let probe = [5.0_f64, -3.0];
    let raw: f64 = forest.score(&probe).unwrap().into();
    assert!(raw > 0.0, "baseline outlier must score above zero");

    let mut store = FeedbackStore::<2>::new(8, 2.0, 1.0).unwrap();
    store.label(probe, FeedbackLabel::Benign).unwrap();
    let adjusted = store.adjust(&probe, raw);
    assert!(
        adjusted < raw,
        "Benign label did not pull score down: raw {raw} -> adjusted {adjusted}"
    );
    assert!(adjusted >= 0.0, "adjusted score went negative");
}

#[test]
fn confirmed_label_pushes_score_up_on_nearby_probe() {
    let forest = small_forest();
    let known_bad = [-4.0_f64, 5.0];
    let raw: f64 = forest.score(&known_bad).unwrap().into();

    let mut store = FeedbackStore::<2>::new(8, 2.0, 1.0).unwrap();
    store.label(known_bad, FeedbackLabel::Confirmed).unwrap();
    // Probe near the confirmed label — adjusted score lifts by
    // the kernel weighted Confirmed sign.
    let nearby = [-3.95_f64, 5.05];
    let raw_nb: f64 = forest.score(&nearby).unwrap().into();
    let adj_nb = store.adjust(&nearby, raw_nb);
    assert!(
        adj_nb > raw_nb,
        "Confirmed label did not lift nearby probe: raw {raw_nb} -> adjusted {adj_nb}"
    );
    let _ = raw;
}

#[test]
fn feedback_does_not_mutate_forest() {
    let forest = small_forest();
    let probe = [3.0_f64, -2.0];
    let before: f64 = forest.score(&probe).unwrap().into();

    let mut store = FeedbackStore::<2>::new(8, 1.0, 1.0).unwrap();
    for _ in 0..20 {
        store.label(probe, FeedbackLabel::Benign).unwrap();
    }
    let _ = store.adjust(&probe, before);
    let after: f64 = forest.score(&probe).unwrap().into();
    assert!(
        (before - after).abs() < 1.0e-12,
        "forest score drifted after feedback ops — forest must stay immutable"
    );
}

#[test]
fn capacity_cap_enforced_under_label_pressure() {
    let mut store = FeedbackStore::<2>::new(4, 1.0, 1.0).unwrap();
    for i in 0..20 {
        store
            .label([f64::from(i), f64::from(i)], FeedbackLabel::Benign)
            .unwrap();
    }
    assert_eq!(store.len(), 4);
    // Remaining entries must be the last 4 labels inserted.
    let kept: Vec<f64> = store.entries().map(|(p, _)| p[0]).collect();
    assert_eq!(kept, vec![16.0, 17.0, 18.0, 19.0]);
}

#[test]
fn mixed_labels_net_behaviour_is_linear() {
    let mut store = FeedbackStore::<2>::new(10, 1.0, 1.0).unwrap();
    store.label([0.0, 0.0], FeedbackLabel::Benign).unwrap();
    store.label([0.0, 0.0], FeedbackLabel::Confirmed).unwrap();
    // Co-located Benign + Confirmed cancel → adjusted == raw.
    let probe = [0.0, 0.0];
    let adjusted = store.adjust(&probe, 0.5);
    assert!((adjusted - 0.5).abs() < 1.0e-9);
}
