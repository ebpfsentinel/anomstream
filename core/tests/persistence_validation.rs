//! A snapshot is admitted only when its structure is one the library
//! could have produced: each corruption below would otherwise panic
//! or loop forever on the first score after loading.
#![allow(clippy::unwrap_used, clippy::panic)]
#![cfg(all(feature = "postcard", feature = "serde_json"))]

use anomstream_core::{ForestBuilder, RandomCutForest, RcfError};
use proptest::prelude::*;
use serde_json::Value;

/// Three points, one tree shape repeated over fifty trees: tree 0 is
/// root internal 1 -> (internal 0 -> leaves 0, 1), leaf 2.
fn snapshot() -> Value {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(4)
        .seed(1)
        .build()
        .unwrap();
    for i in 0..3 {
        f.update([f64::from(i), 0.0]).unwrap();
    }
    serde_json::from_str(&f.to_json().unwrap()).unwrap()
}

fn tree0(v: &mut Value) -> &mut Value {
    &mut v["forest"]["trees"][0][0]
}

fn assert_refused(v: &Value, needle: &str) {
    match RandomCutForest::<2>::from_json(&v.to_string()) {
        Err(RcfError::DeserializationFailed(msg)) => {
            assert!(msg.contains(needle), "expected `{needle}` in `{msg}`");
        }
        other => panic!("expected a refused snapshot, got {other:?}"),
    }
}

#[test]
fn untouched_snapshot_loads() {
    RandomCutForest::<2>::from_json(&snapshot().to_string()).unwrap();
}

#[test]
fn node_that_is_its_own_child_is_refused() {
    let mut v = snapshot();
    tree0(&mut v)["store"]["internals"][1]["left"] = 1.into();
    assert_refused(&v, "parent");
}

#[test]
fn child_shared_by_two_parents_is_refused() {
    let mut v = snapshot();
    let t = tree0(&mut v);
    t["store"]["internals"][1]["right"] = 2_147_483_648_u64.into();
    t["store"]["leaves"][0]["parent"] = 1.into();
    assert_refused(&v, "RandomCutTree");
}

#[test]
fn child_past_capacity_is_refused() {
    let mut v = snapshot();
    tree0(&mut v)["store"]["internals"][1]["right"] = 2_147_483_700_u64.into();
    assert!(RandomCutForest::<2>::from_json(&v.to_string()).is_err());
}

#[test]
fn arena_shorter_than_capacity_is_refused() {
    let mut v = snapshot();
    tree0(&mut v)["store"]["leaves"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert_refused(&v, "arenas");
}

#[test]
fn free_list_naming_a_live_slot_is_refused() {
    let mut v = snapshot();
    tree0(&mut v)["store"]["leaf_free"] = serde_json::json!([3, 0]);
    assert_refused(&v, "live slot");
}

#[test]
fn free_list_longer_than_capacity_is_refused() {
    let mut v = snapshot();
    tree0(&mut v)["store"]["internal_free"] = serde_json::json!([3, 2, 3, 2, 3]);
    assert_refused(&v, "twice");
}

#[test]
fn inconsistent_internal_mass_is_refused() {
    let mut v = snapshot();
    tree0(&mut v)["store"]["internals"][1]["mass"] = 7.into();
    assert_refused(&v, "children mass");
}

#[test]
fn cut_on_a_missing_dimension_is_refused() {
    let mut v = snapshot();
    tree0(&mut v)["store"]["internals"][0]["cut"]["dim"] = 9.into();
    assert_refused(&v, "cut");
}

#[test]
fn reverse_index_disagreeing_with_leaves_is_refused() {
    let mut v = snapshot();
    tree0(&mut v)["leaf_index"][1] = 2_147_483_648_u64.into();
    assert_refused(&v, "RandomCutTree");
}

#[test]
fn wrong_tree_count_is_refused() {
    let mut v = snapshot();
    v["forest"]["trees"].as_array_mut().unwrap().pop();
    assert_refused(&v, "num_trees");
}

#[test]
fn refcount_disagreeing_with_samplers_is_refused() {
    let mut v = snapshot();
    v["forest"]["point_store"]["ref_counts"][1] = 3.into();
    assert_refused(&v, "refcount");
}

#[test]
fn point_store_free_list_naming_a_live_point_is_refused() {
    let mut v = snapshot();
    v["forest"]["point_store"]["free_list"] = serde_json::json!([0]);
    assert_refused(&v, "PointStore");
}

#[test]
fn sampler_larger_than_its_tree_is_refused() {
    let mut v = snapshot();
    v["forest"]["trees"][0][1]["capacity"] = 2.into();
    assert_refused(&v, "ReservoirSampler");
}

#[test]
fn timestamp_on_a_dead_point_is_refused() {
    let mut v = snapshot();
    v["forest"]["timestamps"] = serde_json::json!({ "9": 1 });
    assert_refused(&v, "timestamp");
}

#[test]
fn truncated_postcard_never_panics() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(8)
        .seed(3)
        .build()
        .unwrap();
    for i in 0..20 {
        f.update([f64::from(i % 5), f64::from(i)]).unwrap();
    }
    let bytes = f.to_bytes().unwrap();
    for cut in (0..bytes.len()).step_by(97) {
        assert!(RandomCutForest::<2>::from_bytes(&bytes[..cut]).is_err());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Whatever sequence of updates and deletes built a forest,
    /// including duplicates and freed slots handed out again, its
    /// snapshot passes the validation and scores identically.
    #[test]
    fn any_reachable_forest_roundtrips(
        ops in prop::collection::vec((0_u8..4, 0_u8..6, any::<bool>()), 1..160),
        seed in any::<u64>(),
    ) {
        let mut f = ForestBuilder::<2>::new().num_trees(50).sample_size(8).seed(seed).build().unwrap();
        let mut live = Vec::new();
        for (kind, x, dup) in ops {
            if kind == 0 && !live.is_empty() {
                let idx = live.remove(usize::from(x) % live.len());
                f.delete(idx).unwrap();
            } else {
                let y = if dup { 0.0 } else { f64::from(kind) };
                live.push(f.update_indexed([f64::from(x), y]).unwrap());
            }
        }
        let back = RandomCutForest::<2>::from_bytes(&f.to_bytes().unwrap()).unwrap();
        let probe = [2.5, 1.0];
        prop_assert_eq!(
            f64::from(f.score(&probe).unwrap()).to_bits(),
            f64::from(back.score(&probe).unwrap()).to_bits()
        );
    }
}
