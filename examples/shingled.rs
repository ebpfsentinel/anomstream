#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_precision_loss)]
//! Demo of `ShingledForest`: scalar-stream anomaly detection with
//! internal ring-buffer shingling. Scores a periodic sine baseline
//! plus three injected contextual anomaly windows (dwell, drop,
//! frequency shift) and prints the anomaly scores.
//!
//! Run: `cargo run --release --example shingled`

use anomstream_rs::{RcfError, ShingledForestBuilder};

const SHINGLE: usize = 32;
const WARM: usize = 1_024;

fn main() -> Result<(), RcfError> {
    let mut forest = ShingledForestBuilder::<SHINGLE>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()?;

    // Warm phase — a clean periodic baseline (period 16 samples).
    let mut t = 0.0_f64;
    for _ in 0..WARM {
        let v = (t * 0.4).sin();
        forest.update_scalar(v)?;
        t += 1.0;
    }

    // Eval phase — same baseline with three injected anomalies at
    // known positions. For each position we score the value that
    // *would* land there next, so the report shows the contextual
    // score without mutating the forest.
    let mut eval_scores = Vec::new();
    let mut anomaly_scores: Vec<(&'static str, f64)> = Vec::new();
    let mut inject_dwell = false;
    let mut inject_drop = false;
    let mut inject_freq = false;

    for step in 0..200 {
        let base = (t * 0.4).sin();
        let v = if (100..116).contains(&step) {
            inject_dwell = true;
            0.9 // stuck-high dwell for 16 samples
        } else if (140..148).contains(&step) {
            inject_drop = true;
            -1.5 // negative spike
        } else if step >= 170 {
            inject_freq = true;
            (t * 1.6).sin() // 4× frequency
        } else {
            base
        };

        let score = forest.score_scalar(v)?;
        eval_scores.push(f64::from(score));
        if step == 107 && inject_dwell {
            anomaly_scores.push(("dwell @ step 107", eval_scores[step]));
        }
        if step == 142 && inject_drop {
            anomaly_scores.push(("drop  @ step 142", eval_scores[step]));
        }
        if step == 180 && inject_freq {
            anomaly_scores.push(("freq  @ step 180", eval_scores[step]));
        }
        forest.update_scalar(v)?;
        t += 1.0;
    }

    // Baseline = mean of scores outside injected windows.
    let baseline: f64 = (0..100).map(|i| eval_scores[i]).sum::<f64>() / 100.0;

    println!("shingle_size    = {SHINGLE}");
    println!("warm_samples    = {WARM}");
    println!("baseline score  = {baseline:.3} (mean of first 100 scored eval samples)");
    println!("\ninjected anomalies:");
    for (label, s) in &anomaly_scores {
        println!(
            "  {label}   score = {s:.3}   ({:.1}× baseline)",
            s / baseline
        );
    }

    Ok(())
}
