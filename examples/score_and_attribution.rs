#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_precision_loss)]
//! Demo of `RandomCutForest::score_and_attribution` — single-walk
//! API that returns `(AnomalyScore, DiVector)` from one tree
//! traversal. Cheaper than calling `score` + `attribution`
//! back-to-back when the caller needs both (alert pipelines that
//! emit the score for threshold gating AND the per-dim breakdown
//! for SOC triage).
//!
//! Runs a sanity cross-check: numerical parity vs split calls +
//! wall-clock A/B on 2 k probes.

use std::time::Instant;

use anomstream_rs::{ForestBuilder, RcfError};

const DIM: usize = 16;
const ANOM_DIM: usize = 5;

fn main() -> Result<(), RcfError> {
    let mut forest = ForestBuilder::<DIM>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()?;

    let mut rng = simple_lcg(0x00C0_FFEE);
    for _ in 0..1_024 {
        let mut p = [0.0_f64; DIM];
        for slot in &mut p {
            *slot = rng();
        }
        forest.update(p)?;
    }

    let mut query = [0.5_f64; DIM];
    query[ANOM_DIM] = 50.0;

    // Parity check: fused walk must match split calls up to fp.
    let s_split: f64 = forest.score(&query)?.into();
    let di_split = forest.attribution(&query)?;
    let (s_fused, di_fused) = forest.score_and_attribution(&query)?;
    let s_fused_f: f64 = s_fused.into();
    assert!(
        (s_split - s_fused_f).abs() < 1.0e-9,
        "score mismatch: split={s_split} fused={s_fused_f}"
    );
    for d in 0..DIM {
        let h = (di_split.high()[d] - di_fused.high()[d]).abs();
        let l = (di_split.low()[d] - di_fused.low()[d]).abs();
        assert!(h < 1.0e-9, "dim {d} high mismatch");
        assert!(l < 1.0e-9, "dim {d} low mismatch");
    }
    println!("parity          : OK (split == fused within 1e-9)");

    // A/B timing on a small batch.
    let n_probes = 2_048_usize;
    let probes: Vec<[f64; DIM]> = (0..n_probes)
        .map(|_| {
            let mut p = [0.0_f64; DIM];
            for slot in &mut p {
                *slot = rng();
            }
            p
        })
        .collect();

    let t_split = Instant::now();
    for p in &probes {
        let _s = forest.score(p)?;
        let _d = forest.attribution(p)?;
    }
    let split_ns = t_split.elapsed().as_nanos();

    let t_fused = Instant::now();
    for p in &probes {
        let _both = forest.score_and_attribution(p)?;
    }
    let fused_ns = t_fused.elapsed().as_nanos();

    let speedup = split_ns as f64 / fused_ns as f64;
    println!(
        "split (score+attr) per probe = {:.2} µs",
        (split_ns as f64) / (n_probes as f64) / 1_000.0
    );
    println!(
        "fused score_and_attribution  = {:.2} µs",
        (fused_ns as f64) / (n_probes as f64) / 1_000.0
    );
    println!("speedup                      = {speedup:.2}×");

    // Top-3 driver dims from the fused DiVector.
    let mut totals: Vec<(usize, f64)> = (0..DIM)
        .map(|d| (d, di_fused.high()[d] + di_fused.low()[d]))
        .collect();
    totals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\nscore           = {s_fused}");
    println!("argmax dim      = {:?}", di_fused.argmax());
    println!("top-3 dims (dim, contribution):");
    for (d, v) in totals.iter().take(3) {
        println!("  dim {d:2} → {v:.6}");
    }

    Ok(())
}

/// Tiny linear-congruential RNG so the example has zero non-anomstream-rs
/// dependencies — produces uniform `f64` in `[0, 1)`.
fn simple_lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let frac = u32::try_from(state >> 32).unwrap_or(u32::MAX) >> 11;
        f64::from(frac) / f64::from(1_u32 << 21)
    }
}
