#![allow(clippy::unwrap_used, clippy::panic)]
//! Confidence interval on the anomaly score. The mean by itself
//! does not tell the SOC whether the ensemble agrees; the per-tree
//! stderr does. This example prints the 95 % CI for a tight-cluster
//! probe and an outlier — the outlier CI is almost always wider.
//!
//! Run with `cargo run --example score_confidence`.

use anomstream_rs::{ForestBuilder, RcfError};

fn main() -> Result<(), RcfError> {
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(64)
        .seed(2026)
        .build()?;
    for i in 0..256 {
        let v = f64::from(i) * 0.01;
        forest.update([v, v + 0.1, v + 0.2, v + 0.3])?;
    }

    let normal = forest.score_with_confidence(&[0.05, 0.15, 0.25, 0.35])?;
    let outlier = forest.score_with_confidence(&[50.0, 50.0, 50.0, 50.0])?;

    println!("normal point:");
    print_row(&normal);
    println!("outlier:");
    print_row(&outlier);
    Ok(())
}

fn print_row(s: &anomstream_rs::ScoreWithConfidence) {
    let (lo, hi) = s.ci95();
    println!(
        "  score = {:.3}  stderr = {:.3}  rel = {:.2}%  95% CI = [{:.3}, {:.3}]",
        f64::from(s.score),
        s.stderr,
        s.relative_stderr() * 100.0,
        lo,
        hi,
    );
}
