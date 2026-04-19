#![allow(clippy::unwrap_used, clippy::panic)]
//! Input-feature drift detection (PSI / KL). CUSUM on the score
//! stream catches the RCF re-centring around a new baseline; PSI
//! on raw features catches the *data itself* drifting. This demo
//! freezes a baseline, then pushes a shifted distribution and
//! reports per-dim PSI + drift level.
//!
//! Run with `cargo run --example feature_drift`.

use rcf_rs::RcfError;
use rcf_rs::feature_drift::{DriftLevel, FeatureDriftDetector};

fn main() -> Result<(), RcfError> {
    let mut detector: FeatureDriftDetector<3> = FeatureDriftDetector::new(10)?;

    // Baseline window — 2_000 points roughly uniform in [0, 1).
    for i in 0..2_000 {
        let v = (f64::from(i) % 10.0) * 0.1;
        detector.observe(&[v, v, v])?;
    }
    detector.freeze_baseline()?;
    println!(
        "baseline frozen: {} edges",
        detector.bin_edges().map_or(0, |e| e.len())
    );

    // Production: dim 1 shifts to the high tail (protocol mix
    // change, whatever). Other dims stay stable.
    for i in 0..2_000 {
        let v = (f64::from(i) % 10.0) * 0.1;
        detector.observe(&[v, 0.95, v])?;
    }

    let psi = detector.psi()?;
    println!("per-dim PSI:");
    for (d, p) in psi.iter().enumerate() {
        println!(
            "  dim {d}: psi = {p:.3}  level = {:?}",
            DriftLevel::classify(*p)
        );
    }
    let argmax = detector.argmax_psi()?;
    println!("offending dim: {argmax:?}");
    let kl = detector.kl_divergence()?;
    println!("per-dim KL(Q||P): {kl:?}");
    Ok(())
}
