//! End-to-end integration tests for the RCF.6 visitors against a
//! real [`RandomCutTree`].
//!
//! Story RCF.6 acceptance criteria #5–#7:
//! - uniform 2D dataset → low scores
//! - clustered + outlier → outlier scores ≥ 2× cluster mean
//! - 16-dim with anomaly only on dim 5 → attribution argmax = 5

#![allow(clippy::cast_precision_loss)] // Tests use small bounded counts.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rcf_rs::{
    AttributionVisitor, RandomCutTree, ScalarScoreVisitor, tree::PointAccessor,
};

const SAMPLE_SIZE: u32 = 256;

fn build_tree(dim: usize, points: &[Vec<f64>], seed: u64) -> RandomCutTree {
    let mut tree = RandomCutTree::new(SAMPLE_SIZE, dim).expect("tree builds");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for (idx, p) in points.iter().enumerate() {
        tree.add(idx, p, points, &mut rng).expect("add succeeds");
    }
    tree
}

fn root_mass(tree: &RandomCutTree) -> u64 {
    tree.root()
        .map_or(0, |root| tree.store().node(root).expect("root live").mass())
}

fn score(tree: &RandomCutTree, point: &[f64]) -> f64 {
    let visitor = ScalarScoreVisitor::new(root_mass(tree));
    let s = tree.traverse(point, visitor).expect("traverse succeeds");
    f64::from(s)
}

fn attribute(tree: &RandomCutTree, point: &[f64]) -> rcf_rs::DiVector {
    let visitor =
        AttributionVisitor::new(point.to_vec(), root_mass(tree)).expect("visitor builds");
    tree.traverse(point, visitor).expect("traverse succeeds")
}

/// AC #5: uniform dataset produces low scores almost everywhere.
#[test]
fn uniform_dataset_yields_low_scores() {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut points: Vec<Vec<f64>> = Vec::with_capacity(200);
    for _ in 0..200 {
        let p = vec![
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng),
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng),
        ];
        points.push(p);
    }
    let tree = build_tree(2, &points, 7);

    // Score every member of the dataset itself. The vast majority
    // should sit well below the AWS "≥3σ" anomaly threshold.
    let mut scores: Vec<f64> = points.iter().map(|p| score(&tree, p)).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = scores[(scores.len() * 95) / 100];
    assert!(
        p95 < 1.5,
        "95th-percentile score on uniform data = {p95} (expected < 1.5)"
    );
}

/// AC #6: clustered dataset — a far outlier *not* in the tree
/// scores significantly higher than cluster members. RCF anomaly
/// scoring is meaningful for points the tree has not seen; scoring
/// an inserted outlier walks back to its own leaf and produces
/// near-zero isolation probability.
#[test]
fn outlier_scores_above_cluster_mean() {
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let mut points: Vec<Vec<f64>> = Vec::with_capacity(200);
    // Tight cluster around the origin (uniform [−0.05, 0.05]).
    for _ in 0..200 {
        let p = vec![
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1 - 0.05,
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1 - 0.05,
        ];
        points.push(p);
    }
    let tree = build_tree(2, &points, 11);

    // Score every cluster member (each member IS in the tree, so this
    // gives the baseline self-similarity score).
    let cluster_scores: Vec<f64> = points.iter().map(|p| score(&tree, p)).collect();
    let cluster_mean = cluster_scores.iter().sum::<f64>() / cluster_scores.len() as f64;

    // Score a far outlier that was NEVER inserted — this is the
    // realistic anomaly-detection use case.
    let outlier = vec![10.0, 10.0];
    let outlier_score = score(&tree, &outlier);

    assert!(
        outlier_score >= 2.0 * cluster_mean,
        "outlier score {outlier_score} not ≥ 2× cluster mean {cluster_mean}"
    );
}

/// AC #7: 16-dim dataset with anomaly only on dim 5 — attribution
/// argmax should pick dim 5.
#[test]
fn single_dim_anomaly_attribution_argmax() {
    const DIM: usize = 16;
    const ANOM_DIM: usize = 5;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut points: Vec<Vec<f64>> = Vec::with_capacity(200);
    for _ in 0..200 {
        let p: Vec<f64> = (0..DIM)
            .map(|_| <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng))
            .collect();
        points.push(p);
    }
    let tree = build_tree(DIM, &points, 17);

    // Query: a point that matches the cluster on every dim except
    // dim ANOM_DIM, where it sits far outside.
    let mut anomaly = vec![0.5; DIM];
    anomaly[ANOM_DIM] = 100.0;

    let di = attribute(&tree, &anomaly);
    assert_eq!(
        di.argmax(),
        Some(ANOM_DIM),
        "attribution argmax = {:?}, expected {ANOM_DIM}; full DiVector = high={:?} low={:?}",
        di.argmax(),
        di.high(),
        di.low(),
    );
}

/// Sanity: the [`PointAccessor`] impl on `Vec<Vec<f64>>` is what we
/// actually use throughout these tests — make sure the public API
/// surface lines up.
#[test]
fn point_accessor_impl_is_visible() {
    let v: Vec<Vec<f64>> = vec![vec![1.0, 2.0]];
    let slice: &[f64] = v.point(0).expect("point present");
    assert_eq!(slice, &[1.0, 2.0]);
    assert!(v.point(99).is_none());
}
