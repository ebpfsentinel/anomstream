#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_precision_loss)]
//! Integration test for the SPOT univariate bank + Fisher p-value
//! combination — Siffer KDD 2017. Fits one detector per dim on a
//! synthetic multivariate baseline, freezes, then asserts that a
//! heavy outlier's joint p-value drops below the alert threshold
//! while baseline probes stay near `p = 1`.

#![cfg(feature = "std")]

use anomstream_rs::{PotDetector, fisher_combine};

const DIM: usize = 4;
const ALERT_P: f64 = 1.0e-3;

fn sample(rng: &mut impl FnMut() -> f64) -> [f64; DIM] {
    [rng(), rng() * 2.0, rng() * 0.5, rng() * 3.0]
}

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

fn build_bank(seed: u64) -> Vec<PotDetector> {
    let mut bank: Vec<PotDetector> = (0..DIM).map(|_| PotDetector::default_spot()).collect();
    let mut rng = simple_lcg(seed);
    // Warm.
    for _ in 0..2_000 {
        let values = sample(&mut rng);
        for (d, v) in values.iter().enumerate() {
            bank[d].record(*v);
        }
    }
    for d in &mut bank {
        d.freeze_baseline().unwrap();
    }
    // Feed peaks.
    for _ in 0..2_000 {
        let values = sample(&mut rng);
        for (d, v) in values.iter().enumerate() {
            bank[d].record(*v);
        }
    }
    bank
}

#[test]
fn baseline_probe_yields_high_joint_p_value() {
    let bank = build_bank(0xC0DE_CAFE);
    let mut rng = simple_lcg(0x1234_5678);
    let probe = sample(&mut rng);
    let p_values: Vec<f64> = bank
        .iter()
        .zip(probe.iter())
        .map(|(d, &v)| d.p_value(v))
        .collect();
    let joint = fisher_combine(&p_values);
    assert!(joint > 0.1, "baseline joint p {joint} too low");
    assert!(joint <= 1.0);
}

#[test]
fn all_dim_heavy_outlier_fires_fisher_combined() {
    let bank = build_bank(0xC0DE_BEEF);
    let probe = [50.0_f64, 50.0, 50.0, 50.0];
    let p_values: Vec<f64> = bank
        .iter()
        .zip(probe.iter())
        .map(|(d, &v)| d.p_value(v))
        .collect();
    let joint = fisher_combine(&p_values);
    assert!(
        joint < ALERT_P,
        "all-dim outlier joint p {joint} did not cross alert {ALERT_P:e}"
    );
}

#[test]
fn single_dim_outlier_still_lifts_joint_signal() {
    let bank = build_bank(0xFEED_FACE);
    let mut rng = simple_lcg(0xAA55);
    let normal = sample(&mut rng);
    // One-dim outlier: dim 2 well above its [0, 0.5] range.
    let probe = [normal[0], normal[1], 50.0, normal[3]];
    let p_values: Vec<f64> = bank
        .iter()
        .zip(probe.iter())
        .map(|(d, &v)| d.p_value(v))
        .collect();
    let joint = fisher_combine(&p_values);
    // Single-dim outlier should drag the joint below 0.01 even
    // when the other dims are baseline — this is the point of
    // combining: one very small p dominates Fisher's statistic.
    assert!(joint < 0.01, "dim-2 outlier alone gave joint p {joint}");
}
