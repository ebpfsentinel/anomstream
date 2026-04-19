#![allow(clippy::unwrap_used, clippy::panic)]
//! End-to-end integration tests for the RCF visitors against a
//! real [`RandomCutTree`].
//!
//! - uniform 2D dataset → low scores
//! - clustered + outlier → outlier scores ≥ 2× cluster mean
//! - 16-dim with anomaly only on dim 5 → attribution argmax = 5

#![allow(clippy::cast_precision_loss, clippy::float_cmp)] // Tests use small bounded counts and exact-equality probes.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rcf_rs::{AttributionVisitor, RandomCutTree, ScalarScoreVisitor, tree::PointAccessor};

const SAMPLE_SIZE: u32 = 256;

fn build_tree<const D: usize>(points: &[[f64; D]], seed: u64) -> RandomCutTree<D> {
    let mut tree = RandomCutTree::<D>::new(SAMPLE_SIZE).expect("tree builds");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for (idx, p) in points.iter().enumerate() {
        tree.add(idx, p, points, &mut rng).expect("add succeeds");
    }
    tree
}

fn root_mass<const D: usize>(tree: &RandomCutTree<D>) -> u64 {
    tree.root()
        .map_or(0, |root| tree.store().view(root).expect("root live").mass())
}

fn score<const D: usize>(tree: &RandomCutTree<D>, point: &[f64; D]) -> f64 {
    let visitor = ScalarScoreVisitor::new(root_mass(tree));
    let s = tree.traverse(point, visitor).expect("traverse succeeds");
    f64::from(s)
}

fn attribute<const D: usize>(tree: &RandomCutTree<D>, point: &[f64; D]) -> rcf_rs::DiVector {
    let visitor = AttributionVisitor::new(point, root_mass(tree)).expect("visitor builds");
    tree.traverse(point, visitor).expect("traverse succeeds")
}

/// Uniform dataset produces low scores almost everywhere.
#[test]
fn uniform_dataset_yields_low_scores() {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut points: Vec<[f64; 2]> = Vec::with_capacity(200);
    for _ in 0..200 {
        let p = [
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng),
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng),
        ];
        points.push(p);
    }
    let tree = build_tree::<2>(&points, 7);

    let mut scores: Vec<f64> = points.iter().map(|p| score(&tree, p)).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = scores[(scores.len() * 95) / 100];
    assert!(
        p95 < 1.5,
        "95th-percentile score on uniform data = {p95} (expected < 1.5)"
    );
}

/// Clustered dataset — a far outlier *not* in the tree scores
/// significantly higher than cluster members.
#[test]
fn outlier_scores_above_cluster_mean() {
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let mut points: Vec<[f64; 2]> = Vec::with_capacity(200);
    for _ in 0..200 {
        let p = [
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1 - 0.05,
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1 - 0.05,
        ];
        points.push(p);
    }
    let tree = build_tree::<2>(&points, 11);

    let cluster_scores: Vec<f64> = points.iter().map(|p| score(&tree, p)).collect();
    let cluster_mean = cluster_scores.iter().sum::<f64>() / cluster_scores.len() as f64;

    let outlier: [f64; 2] = [10.0, 10.0];
    let outlier_score = score(&tree, &outlier);

    assert!(
        outlier_score >= 2.0 * cluster_mean,
        "outlier score {outlier_score} not ≥ 2× cluster mean {cluster_mean}"
    );
}

/// 16-dim dataset with anomaly only on dim 5 — attribution argmax
/// should pick dim 5.
#[test]
fn single_dim_anomaly_attribution_argmax() {
    const DIM: usize = 16;
    const ANOM_DIM: usize = 5;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut points: Vec<[f64; DIM]> = Vec::with_capacity(200);
    for _ in 0..200 {
        let mut p = [0.0_f64; DIM];
        for slot in &mut p {
            *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
        }
        points.push(p);
    }
    let tree = build_tree::<DIM>(&points, 17);

    let mut anomaly = [0.5_f64; DIM];
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

/// Sanity: the [`PointAccessor`] impl on `Vec<[f64; D]>` is what we
/// actually use throughout these tests — make sure the public API
/// surface lines up.
#[test]
fn point_accessor_impl_is_visible() {
    let v: Vec<[f64; 2]> = vec![[1.0, 2.0]];
    let arr: &[f64; 2] = <Vec<[f64; 2]> as PointAccessor<2>>::point(&v, 0).expect("point present");
    assert_eq!(arr, &[1.0, 2.0]);
    assert!(<Vec<[f64; 2]> as PointAccessor<2>>::point(&v, 99).is_none());
}
