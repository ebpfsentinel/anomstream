//! Fisher's method — combine K independent p-values into a joint
//! anomaly score.
//!
//! Fisher 1932. Under the null `H_0` that every individual test is
//! non-anomalous and p-values are independent and Uniform(0, 1),
//! the statistic
//!
//! ```text
//! T = -2 · Σ_{i=1..K} ln(p_i)
//! ```
//!
//! follows a chi-squared distribution with `2K` degrees of freedom.
//! The joint p-value is the upper-tail survival `Pr(χ²(2K) > T)`.
//!
//! Combined with [`crate::univariate_spot::PotDetector`] (one per
//! feature dim) this gives a **univariate detector bank** whose
//! joint signal is tighter than any single dim in isolation — the
//! SPOT KDD 2017 prescription for multivariate anomaly detection
//! on heterogeneously-distributed features.
//!
//! # Example
//!
//! ```ignore
//! use anomstream_rs::ensemble::fisher_combine;
//! use anomstream_rs::univariate_spot::PotDetector;
//!
//! let mut bank = vec![PotDetector::default_spot(); 4];
//! for detector in &mut bank { /* record baseline */ }
//! for detector in &mut bank { detector.freeze_baseline()?; }
//!
//! let feature_values = [0.9, 1.5, 0.1, 3.0];
//! let p_values: Vec<f64> = bank.iter()
//!     .zip(feature_values.iter())
//!     .map(|(d, &v)| d.p_value(v))
//!     .collect();
//! let joint_p = fisher_combine(&p_values);
//! if joint_p < 1.0e-3 { /* anomaly */ }
//! # Ok::<(), anomstream_rs::RcfError>(())
//! ```

#![cfg(feature = "std")]

/// Combine K independent p-values via Fisher's method. Returns the
/// joint p-value — probability that the combined test statistic
/// `T = -2 · Σ ln(p_i)` would exceed its observed value under the
/// null `H_0` that every component is non-anomalous.
///
/// Returns `1.0` when `p_values` is empty (no evidence → no
/// anomaly). Non-finite or out-of-range (`p ≤ 0` or `p > 1`)
/// p-values are clamped to `[EPSILON, 1]` so the combination stays
/// well-defined.
///
/// # Complexity
///
/// `O(K)` — tight loop over `p_values`, no allocations. Uses the
/// closed-form chi-squared-with-even-dof survival
/// `Q(K, T/2) = e^{-T/2} · Σ_{i=0..K-1} (T/2)^i / i!` since
/// `2K` degrees of freedom are always even.
#[must_use]
pub fn fisher_combine(p_values: &[f64]) -> f64 {
    // Statistic T = -2 · Σ ln(p). Non-finite slots are skipped
    // entirely — both from the T sum and from the degrees of
    // freedom count — so a NaN does not inflate `K` and depress
    // the joint p artificially.
    let mut t = 0.0_f64;
    let mut k = 0_usize;
    for &p in p_values {
        if !p.is_finite() {
            continue;
        }
        let clamped = p.clamp(f64::EPSILON, 1.0);
        t += -2.0 * clamped.ln();
        k += 1;
    }
    if k == 0 {
        return 1.0;
    }
    chi_squared_survival_even(k, t)
}

/// Upper-tail survival of a `χ²(2k)` distribution — the closed-form
/// `Q(k, x/2)` for integer `k` and `x ≥ 0`.
///
/// `Q(k, y) = e^{−y} · Σ_{i=0..k-1} y^i / i!` for integer `k`.
/// Numerically stable for `k ≤ ~160` under f64 — beyond that the
/// `y^i / i!` terms start to lose precision; anomstream-rs's detector
/// banks top out well below that bound (typical `k ≤ 64` features).
#[must_use]
pub fn chi_squared_survival_even(k: usize, t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    if k == 0 {
        return 0.0;
    }
    let y = 0.5 * t;
    // Kahan-compensated running sum — the `y^i / i!` terms mix
    // very different magnitudes when `k ≥ 20`.
    let mut term = 1.0_f64; // i = 0: y^0 / 0! = 1
    let mut sum = 1.0_f64;
    let mut compensation = 0.0_f64;
    for i in 1..k {
        #[allow(clippy::cast_precision_loss)]
        let divisor = i as f64;
        term *= y / divisor;
        // Kahan step.
        let adjusted = term - compensation;
        let temp = sum + adjusted;
        compensation = (temp - sum) - adjusted;
        sum = temp;
    }
    let q = (-y).exp() * sum;
    q.clamp(0.0, 1.0)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_one() {
        assert_eq!(fisher_combine(&[]), 1.0);
    }

    #[test]
    fn single_p_value_survives_roundtrip_at_boundaries() {
        // k=1 → chi-squared with 2 dof at T = -2·ln(p) has
        // survival Q(1, T/2) = e^{-T/2} = p. So fisher_combine of
        // a single p-value returns that p-value itself.
        let p = 0.3_f64;
        let out = fisher_combine(&[p]);
        assert!((out - p).abs() < 1.0e-9, "single-p = {out} expected {p}");
    }

    #[test]
    fn all_ones_yield_p_equal_to_one() {
        let out = fisher_combine(&[1.0; 8]);
        assert_eq!(out, 1.0);
    }

    #[test]
    fn small_p_values_combine_to_smaller_joint() {
        let out = fisher_combine(&[0.001, 0.001, 0.001]);
        assert!(out < 0.001);
    }

    #[test]
    fn combining_mixed_evidence_stays_bounded() {
        // One very small p alongside several `1.0`s must yield a
        // joint p smaller than 1 but not absurdly tiny.
        let out = fisher_combine(&[0.0001, 1.0, 1.0, 1.0]);
        assert!(out > 0.0);
        assert!(out < 1.0);
    }

    #[test]
    fn non_finite_p_clamps_to_unity() {
        let out = fisher_combine(&[f64::NAN, 0.1, 0.1]);
        // The NaN slot maps to 1.0, so the combination equals the
        // two-element fisher of [0.1, 0.1].
        let ref_out = fisher_combine(&[0.1, 0.1]);
        assert!((out - ref_out).abs() < 1.0e-12);
    }

    #[test]
    fn chi_squared_survival_zero_argument_is_one() {
        assert_eq!(chi_squared_survival_even(1, 0.0), 1.0);
        assert_eq!(chi_squared_survival_even(8, 0.0), 1.0);
    }

    #[test]
    fn chi_squared_survival_one_dof_closed_form() {
        // Q(1, y) = e^{-y}, so the survival of χ²(2) at T is
        // e^{-T/2}. Spot-check.
        let t: f64 = 3.0;
        let expected = (-0.5 * t).exp();
        let out = chi_squared_survival_even(1, t);
        assert!((out - expected).abs() < 1.0e-12);
    }
}
