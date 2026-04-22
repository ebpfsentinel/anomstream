#![allow(clippy::unwrap_used, clippy::panic)]
//! AWS `SageMaker` Random Cut Forest conformance suite.
//!
//! Asserts every documented invariant of the AWS `SageMaker` RCF
//! reference (<https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html>):
//!
//! - `feature_dim ∈ [1, 10000]` — both ends checked
//! - `num_trees ∈ [50, 1000]`, default `100`
//! - `num_samples_per_tree ∈ [1, 2048]`, default `256`
//! - Reservoir sampling **without replacement** (no duplicate
//!   `point_idx` in any tree's reservoir)
//! - Score = average across trees (computed manually for a 2-tree
//!   forest and compared bit-exactly)
//! - Anomaly score is monotonic in tree depth — a far outlier scores
//!   strictly higher than a tight cluster member

#![allow(clippy::cast_precision_loss, clippy::float_cmp)] // exact-equality probes + small bounded counters.

use std::collections::HashSet;

use anomstream_rs::config::{
    DEFAULT_NUM_TREES, DEFAULT_SAMPLE_SIZE, MAX_DIMENSION, MAX_NUM_TREES, MAX_SAMPLE_SIZE,
    MIN_DIMENSION, MIN_NUM_TREES, MIN_SAMPLE_SIZE,
};
use anomstream_rs::{ForestBuilder, RcfConfig, ScalarScoreVisitor};

#[test]
fn aws_feature_dim_lower_bound_zero_rejected() {
    // `D = 0` rejected by `validate_dimension` at builder time.
    assert!(RcfConfig::validate_dimension(0).is_err());
    assert_eq!(MIN_DIMENSION, 1);
}

#[test]
fn aws_feature_dim_lower_bound_one_accepted() {
    let f = ForestBuilder::<1>::new().seed(1).build();
    assert!(f.is_ok());
}

#[test]
fn aws_feature_dim_upper_bound_above_max_rejected() {
    assert!(RcfConfig::validate_dimension(MAX_DIMENSION + 1).is_err());
    assert_eq!(MAX_DIMENSION, 10_000);
}

#[test]
fn aws_feature_dim_upper_bound_at_max_accepted() {
    // Construct via `RcfConfig::validate_dimension` directly — the
    // `[f64; 10_000]` monomorphisation would blow up the compiled
    // bench binary if we instantiated the forest at the limit.
    RcfConfig::validate_dimension(MAX_DIMENSION).expect("dimension at MAX_DIMENSION must validate");
}

#[test]
fn aws_num_trees_lower_bound_below_min_rejected() {
    let err = ForestBuilder::<4>::new()
        .num_trees(MIN_NUM_TREES - 1)
        .build()
        .unwrap_err();
    assert!(matches!(err, anomstream_rs::RcfError::InvalidConfig(_)));
    assert_eq!(MIN_NUM_TREES, 50);
}

#[test]
fn aws_num_trees_lower_bound_at_min_accepted() {
    ForestBuilder::<4>::new()
        .num_trees(MIN_NUM_TREES)
        .seed(1)
        .build()
        .expect("MIN_NUM_TREES must build");
}

#[test]
fn aws_num_trees_upper_bound_at_max_accepted() {
    ForestBuilder::<4>::new()
        .num_trees(MAX_NUM_TREES)
        .seed(1)
        .build()
        .expect("MAX_NUM_TREES must build");
    assert_eq!(MAX_NUM_TREES, 1_000);
}

#[test]
fn aws_num_trees_upper_bound_above_max_rejected() {
    let err = ForestBuilder::<4>::new()
        .num_trees(MAX_NUM_TREES + 1)
        .build()
        .unwrap_err();
    assert!(matches!(err, anomstream_rs::RcfError::InvalidConfig(_)));
}

#[test]
fn aws_num_samples_per_tree_lower_bound_zero_rejected() {
    let err = ForestBuilder::<4>::new()
        .sample_size(0)
        .build()
        .unwrap_err();
    assert!(matches!(err, anomstream_rs::RcfError::InvalidConfig(_)));
}

#[test]
fn aws_num_samples_per_tree_lower_bound_one_accepted() {
    ForestBuilder::<4>::new()
        .sample_size(MIN_SAMPLE_SIZE)
        .seed(1)
        .build()
        .expect("MIN_SAMPLE_SIZE must build");
    assert_eq!(MIN_SAMPLE_SIZE, 1);
}

#[test]
fn aws_num_samples_per_tree_upper_bound_at_max_accepted() {
    ForestBuilder::<4>::new()
        .sample_size(MAX_SAMPLE_SIZE)
        .seed(1)
        .build()
        .expect("MAX_SAMPLE_SIZE must build");
    assert_eq!(MAX_SAMPLE_SIZE, 2_048);
}

#[test]
fn aws_num_samples_per_tree_upper_bound_above_max_rejected() {
    let err = ForestBuilder::<4>::new()
        .sample_size(MAX_SAMPLE_SIZE + 1)
        .build()
        .unwrap_err();
    assert!(matches!(err, anomstream_rs::RcfError::InvalidConfig(_)));
}

#[test]
fn aws_default_hyperparameters() {
    let f = ForestBuilder::<4>::new().seed(1).build().unwrap();
    assert_eq!(f.num_trees(), DEFAULT_NUM_TREES);
    assert_eq!(f.sample_size(), DEFAULT_SAMPLE_SIZE);
    assert_eq!(DEFAULT_NUM_TREES, 100);
    assert_eq!(DEFAULT_SAMPLE_SIZE, 256);
}

#[test]
fn aws_reservoir_without_replacement() {
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(128)
        .seed(42)
        .build()
        .unwrap();
    for i in 0_u32..2_000 {
        let v = f64::from(i);
        forest.update([v, v, v, v]).unwrap();
    }
    for (_, sampler, _) in forest.trees() {
        let indices: Vec<usize> = sampler.iter_indices().collect();
        let unique: HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(
            indices.len(),
            unique.len(),
            "reservoir sampler kept duplicate point_idx — violates without-replacement invariant"
        );
    }
}

#[test]
fn aws_score_equals_average_across_trees() {
    // Build a 2-tree forest, train it on a tight cluster, then
    // recompute the score manually as the mean of per-tree
    // ScalarScoreVisitor outputs and verify bit-exact match against
    // `RandomCutForest::score`.
    let mut forest = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(7)
        .build()
        .unwrap();
    for i in 0_u32..200 {
        let v = f64::from(i) * 0.01;
        forest.update([v, v + 0.5]).unwrap();
    }
    let probe: [f64; 2] = [10.0, 10.0];
    let forest_score: f64 = forest.score(&probe).unwrap().into();

    // Manual aggregation: per-tree visitor on each live tree.
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for (tree, _, _) in forest.trees() {
        let Some(root) = tree.root() else {
            continue;
        };
        let mass = tree.store().view(root).unwrap().mass();
        let visitor = ScalarScoreVisitor::new(mass);
        let s: f64 = tree.traverse(&probe, visitor).unwrap().into();
        total += s;
        count += 1;
    }
    let manual_mean = (total / count as f64).max(0.0);
    assert_eq!(
        forest_score, manual_mean,
        "RandomCutForest::score must equal the mean of per-tree visitor outputs"
    );
}

#[test]
fn aws_outlier_strictly_above_cluster_member() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut forest = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(11)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    for _ in 0..400 {
        let p = [
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1,
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1,
        ];
        forest.update(p).unwrap();
    }
    let cluster: f64 = forest.score(&[0.05, 0.05]).unwrap().into();
    let outlier: f64 = forest.score(&[10.0, 10.0]).unwrap().into();
    assert!(
        outlier > cluster,
        "outlier {outlier} not strictly > cluster {cluster} — score must reflect tree-depth monotonicity",
    );
}
