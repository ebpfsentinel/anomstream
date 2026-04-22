#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_precision_loss)]
//! Demo of the univariate SPOT detector bank + Fisher's method —
//! Siffer KDD 2017. Fits one [`PotDetector`] per feature dim on a
//! warm-phase corpus, freezes the baseline, then scores eval
//! points through the bank and combines the per-dim p-values into
//! a joint anomaly signal via [`fisher_combine`].
//!
//! Complements the RCF scorer for heterogeneously-distributed
//! multivariate features where isolation depth alone leaves AUC
//! on the table.
//!
//! Run: `cargo run --release --example univariate_spot_bank`

use anomstream_rs::{PotDetector, fisher_combine};

const DIM: usize = 4;

fn main() {
    // Build one detector per feature dim.
    let mut bank: Vec<PotDetector> = (0..DIM).map(|_| PotDetector::default_spot()).collect();

    // Warm phase — each dim draws from its own distribution.
    let mut rng = simple_lcg(0x_C0DE_CAFE);
    for _ in 0..2_000 {
        let values = sample(&mut rng);
        for (d, v) in values.iter().enumerate() {
            bank[d].record(*v);
        }
    }
    for d in &mut bank {
        d.freeze_baseline().unwrap();
    }
    // Continue feeding baseline points past freeze so peaks
    // accumulate against the frozen `u` — enough for the GPD MoM
    // fit to kick in (MIN_PEAKS_FOR_FIT = 16 peaks → with q=0.98
    // need ≈ 800 post-freeze samples per dim on average).
    for _ in 0..2_000 {
        let values = sample(&mut rng);
        for (d, v) in values.iter().enumerate() {
            bank[d].record(*v);
        }
    }
    println!(
        "bank ready: per-dim peaks = {:?}, total seen = {:?}",
        bank.iter().map(PotDetector::peak_count).collect::<Vec<_>>(),
        bank.iter().map(PotDetector::total_seen).collect::<Vec<_>>()
    );

    // Score three contrastive probes.
    let normal = sample(&mut rng);
    let dim2_outlier = [normal[0], normal[1], 50.0, normal[3]];
    let all_dim_outlier = [20.0, 20.0, 20.0, 20.0];

    for (label, probe) in [
        ("normal        ", &normal),
        ("dim-2 outlier ", &dim2_outlier),
        ("all-dim spike ", &all_dim_outlier),
    ] {
        let p_values: Vec<f64> = bank
            .iter()
            .zip(probe.iter())
            .map(|(d, &v)| d.p_value(v))
            .collect();
        let joint = fisher_combine(&p_values);
        println!(
            "{label}  per-dim p = {p_values:?}  joint p = {joint:.2e}  anomaly = {}",
            joint < 1.0e-3
        );
    }
}

fn sample(rng: &mut impl FnMut() -> f64) -> [f64; DIM] {
    [rng() * 1.0, rng() * 2.0, rng() * 0.5, rng() * 3.0]
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
