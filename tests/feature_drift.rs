#![allow(clippy::unwrap_used, clippy::panic)]
//! Integration-level checks on [`FeatureDriftDetector`] — empirical
//! PSI / KL behaviour against known distributions.

use anomstream_rs::feature_drift::{
    DriftLevel, FeatureDriftDetector, PSI_ALERT_THRESHOLD, PSI_WATCH_THRESHOLD,
};

#[test]
fn stable_stream_psi_stays_below_watch() {
    let mut d: FeatureDriftDetector<2> = FeatureDriftDetector::new(10).unwrap();
    for i in 0..2_000_u32 {
        let v = (f64::from(i % 10)) * 0.1;
        d.observe(&[v, v + 0.1]).unwrap();
    }
    d.freeze_baseline().unwrap();
    // Replay same distribution.
    for i in 0..2_000_u32 {
        let v = (f64::from(i % 10)) * 0.1;
        d.observe(&[v, v + 0.1]).unwrap();
    }
    for p in d.psi().unwrap() {
        assert!(p < PSI_WATCH_THRESHOLD, "expected stable, got {p}");
    }
}

#[test]
fn shifted_stream_reaches_alert() {
    let mut d: FeatureDriftDetector<1> = FeatureDriftDetector::new(10).unwrap();
    for i in 0..5_000_u32 {
        let v = (f64::from(i % 10)) * 0.1;
        d.observe(&[v]).unwrap();
    }
    d.freeze_baseline().unwrap();
    for _ in 0..5_000 {
        d.observe(&[0.95]).unwrap();
    }
    let psi = d.psi().unwrap();
    assert!(psi[0] > PSI_ALERT_THRESHOLD);
    assert_eq!(DriftLevel::classify(psi[0]), DriftLevel::Alert);
}

#[test]
fn argmax_psi_pins_drifting_dim() {
    let mut d: FeatureDriftDetector<3> = FeatureDriftDetector::new(10).unwrap();
    for i in 0..2_000_u32 {
        let v = (f64::from(i % 10)) * 0.1;
        d.observe(&[v, v, v]).unwrap();
    }
    d.freeze_baseline().unwrap();
    // Push dim 2 only.
    for i in 0..2_000_u32 {
        let v = (f64::from(i % 10)) * 0.1;
        d.observe(&[v, v, 0.95]).unwrap();
    }
    assert_eq!(d.argmax_psi().unwrap(), Some(2));
}

#[test]
fn reset_production_wipes_window_without_baseline() {
    let mut d: FeatureDriftDetector<1> = FeatureDriftDetector::new(10).unwrap();
    for i in 0..1_000_u32 {
        d.observe(&[f64::from(i) * 0.001]).unwrap();
    }
    d.freeze_baseline().unwrap();
    for i in 0..1_000_u32 {
        d.observe(&[f64::from(i) * 0.001]).unwrap();
    }
    d.reset_production();
    assert!(d.is_baseline_frozen());
    let psi = d.psi().unwrap();
    assert!(psi[0].is_finite());
}

#[test]
fn kl_nonnegative_on_any_stream() {
    let mut d: FeatureDriftDetector<1> = FeatureDriftDetector::new(10).unwrap();
    for i in 0..2_000_u32 {
        let v = (f64::from(i % 10)) * 0.1;
        d.observe(&[v]).unwrap();
    }
    d.freeze_baseline().unwrap();
    for _ in 0..2_000 {
        d.observe(&[0.5]).unwrap();
    }
    let kl = d.kl_divergence().unwrap();
    assert!(kl[0] >= 0.0);
}

#[test]
fn nan_input_rejected() {
    let mut d: FeatureDriftDetector<2> = FeatureDriftDetector::new(10).unwrap();
    assert!(d.observe(&[f64::NAN, 0.0]).is_err());
}
