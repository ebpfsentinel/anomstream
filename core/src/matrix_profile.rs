//! Matrix profile — batch time-series discord / motif detector
//! (STOMP, Zhu et al. 2016).
//!
//! For a univariate series `T` of length `n` and a window length
//! `m`, the matrix profile `P[i]` is the z-normalised Euclidean
//! distance from the subsequence `T[i..i+m]` to its nearest
//! *non-trivial* neighbour in `T` (i.e. skipping a small exclusion
//! zone around `i`). `P` localises anomalies two ways:
//!
//! - **Discord** — `argmax P[i]`: subsequence least similar to
//!   anything else in the series. Analogue of a point-wise outlier,
//!   but at the shape level. Ideal for "this one window looks
//!   unlike anything we've seen before" detection.
//! - **Motif** — `argmin P[i]`: most-repeated shape. Useful for
//!   carving out the dominant beaconing or periodic pattern before
//!   feeding residuals to another detector.
//!
//! Complements [`crate::ShingledForest`]: the shingled forest is an
//! online, approximate, tree-based detector; the matrix profile is
//! an exact, batch, distance-based detector. Run the forest on the
//! hot stream, run the matrix profile on a captured window when
//! forensic-grade exactness matters.
//!
//! STOMP computes the profile in `O(n²)` time with `O(n)` memory,
//! using the diagonal recurrence
//! `QT[i, j] = QT[i-1, j-1] + T[i+m-1]·T[j+m-1] - T[i-1]·T[j-1]`
//! over pre-computed per-subsequence means / standard deviations.
//!
//! # References
//!
//! 1. Y. Zhu, Z. Zimmerman, N. Senobari, C. Yeh, G. Funning,
//!    A. Mueen, P. Brisk, E. Keogh, "Matrix Profile II: Exploiting
//!    a Novel Algorithm and GPUs to Break the One Hundred Million
//!    Barrier for Time Series Motifs and Joins", ICDM 2016.
//! 2. C. Yeh, Y. Zhu, L. Ulanova, N. Begum, Y. Ding, H. A. Dau,
//!    D. F. Silva, A. Mueen, E. Keogh, "Matrix Profile I: All
//!    Pairs Similarity Joins for Time Series", ICDM 2016.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::error::{RcfError, RcfResult};

/// Minimum window length. STOMP requires at least `m = 4` for the
/// z-normalisation to be meaningful.
pub const MIN_WINDOW: usize = 4;

/// Computed matrix profile for a fixed `(series, window)` pair.
///
/// The profile array is always in 1-to-1 correspondence with the
/// `n − m + 1` candidate subsequences; `profile[i]` is the nearest
/// non-trivial-neighbour distance for subsequence `T[i..i+m]`, and
/// `index[i]` is that neighbour's starting offset.
///
/// # Examples
///
/// ```
/// use anomstream_core::MatrixProfile;
///
/// // Synthetic: smooth cosine with one injected spike near i=48.
/// let mut series: Vec<f64> = (0..128)
///     .map(|i| (f64::from(i as i32) * 0.3).cos())
///     .collect();
/// for v in &mut series[48..56] {
///     *v += 5.0;
/// }
/// let mp = MatrixProfile::compute(&series, 8, None).expect("mp");
/// let (pos, _score) = mp.discord();
/// assert!((40..=56).contains(&pos));
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatrixProfile {
    /// Per-subsequence nearest-neighbour distance.
    profile: Vec<f64>,
    /// Per-subsequence nearest-neighbour index.
    index: Vec<usize>,
    /// Window length used to compute this profile.
    window: usize,
    /// Exclusion-zone half-width used during the join.
    exclusion_zone: usize,
}

impl MatrixProfile {
    /// Run STOMP over `series` with subsequence length `window`.
    ///
    /// `exclusion_zone` is the half-width of the trivial-match
    /// band around each query (`|i − j| < exclusion_zone` is
    /// skipped). Pass `None` for the conventional `ceil(window / 4)`
    /// default (Keogh / Mueen matrix-profile tutorials).
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `window < MIN_WINDOW`,
    /// when the series is too short (`series.len() < 2 · window`),
    /// when `series` contains a non-finite value, or when
    /// `exclusion_zone` would leave zero valid neighbours.
    #[must_use = "detector output should be checked — dropping it silently usually indicates a logic bug"]
    pub fn compute(
        series: &[f64],
        window: usize,
        exclusion_zone: Option<usize>,
    ) -> RcfResult<Self> {
        if window < MIN_WINDOW {
            return Err(RcfError::InvalidConfig(alloc::format!(
                "MatrixProfile: window {window} < MIN_WINDOW {MIN_WINDOW}"
            )));
        }
        let n = series.len();
        if n < window * 2 {
            return Err(RcfError::InvalidConfig(alloc::format!(
                "MatrixProfile: series len {n} must be ≥ 2·window ({})",
                window * 2
            )));
        }
        if series.iter().any(|v| !v.is_finite()) {
            return Err(RcfError::InvalidConfig(alloc::string::ToString::to_string(
                "MatrixProfile: series contains non-finite values",
            )));
        }
        let subseq_n = n - window + 1;
        let exclusion_zone = exclusion_zone.unwrap_or_else(|| window.div_ceil(4));
        if exclusion_zone >= subseq_n {
            return Err(RcfError::InvalidConfig(alloc::format!(
                "MatrixProfile: exclusion_zone {exclusion_zone} ≥ subseq count {subseq_n}"
            )));
        }

        let (means, stds) = sliding_stats(series, window);
        // First column of the QT matrix — sliding dot products of
        // `series` against the prefix `series[0..window]`.
        let qt_first = sliding_dot_product(series, &series[0..window]);
        let mut qt = qt_first.clone();

        let mut profile = vec![f64::INFINITY; subseq_n];
        let mut index = vec![0_usize; subseq_n];

        update_row(
            &mut profile,
            &mut index,
            &qt,
            0,
            window,
            &means,
            &stds,
            exclusion_zone,
        );

        for j in 1..subseq_n {
            // Diagonal recurrence — must iterate top-down in
            // reverse so `qt[i]` reads the previous-iteration
            // `qt[i-1]` before it is overwritten.
            for i in (1..subseq_n).rev() {
                qt[i] = qt[i - 1] + series[i + window - 1] * series[j + window - 1]
                    - series[i - 1] * series[j - 1];
            }
            qt[0] = qt_first[j];
            update_row(
                &mut profile,
                &mut index,
                &qt,
                j,
                window,
                &means,
                &stds,
                exclusion_zone,
            );
        }

        Ok(Self {
            profile,
            index,
            window,
            exclusion_zone,
        })
    }

    /// Window length used when computing this profile.
    #[must_use]
    pub fn window(&self) -> usize {
        self.window
    }

    /// Exclusion-zone half-width used when computing this profile.
    #[must_use]
    pub fn exclusion_zone(&self) -> usize {
        self.exclusion_zone
    }

    /// Number of subsequences (`n − m + 1`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.profile.len()
    }

    /// `true` when the profile holds zero subsequences — never
    /// returned by [`Self::compute`], provided as a total accessor.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profile.is_empty()
    }

    /// Per-subsequence nearest-neighbour distance vector.
    #[must_use]
    pub fn profile(&self) -> &[f64] {
        &self.profile
    }

    /// Per-subsequence nearest-neighbour index vector.
    #[must_use]
    pub fn profile_index(&self) -> &[usize] {
        &self.index
    }

    /// Discord — subsequence whose nearest neighbour is farthest.
    /// Returns `(start_index, distance)`.
    #[must_use = "detector output should be checked — dropping it silently usually indicates a logic bug"]
    pub fn discord(&self) -> (usize, f64) {
        self.profile
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or((0, f64::NAN), |(i, d)| (i, *d))
    }

    /// Top-`k` discords ranked by descending distance. `k` is
    /// clamped to [`Self::len`]. Uses a greedy suppression pass
    /// that skips any candidate within the exclusion zone of an
    /// already-emitted discord — prevents the top-`k` from
    /// clustering inside a single anomalous region.
    #[must_use = "detector output should be checked — dropping it silently usually indicates a logic bug"]
    pub fn discord_topk(&self, k: usize) -> Vec<(usize, f64)> {
        let mut candidates: Vec<(usize, f64)> = self.profile.iter().copied().enumerate().collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut out: Vec<(usize, f64)> = Vec::with_capacity(k.min(candidates.len()));
        for (pos, dist) in candidates {
            if out.len() >= k {
                break;
            }
            if !dist.is_finite() {
                continue;
            }
            if out
                .iter()
                .any(|(p, _)| p.abs_diff(pos) < self.exclusion_zone)
            {
                continue;
            }
            out.push((pos, dist));
        }
        out
    }

    /// Motif — subsequence whose nearest neighbour is closest.
    /// Returns `(start_index, distance)`.
    #[must_use = "detector output should be checked — dropping it silently usually indicates a logic bug"]
    pub fn motif(&self) -> (usize, f64) {
        self.profile
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map_or((0, f64::NAN), |(i, d)| (i, *d))
    }
}

/// Sliding mean and standard deviation of every length-`window`
/// window of `series`. Output has length `n − window + 1`.
#[allow(clippy::cast_precision_loss)]
fn sliding_stats(series: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = series.len();
    let subseq_n = n - window + 1;
    let w = window as f64;

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &v in &series[0..window] {
        sum += v;
        sum_sq += v * v;
    }

    let mut means = vec![0.0_f64; subseq_n];
    let mut stds = vec![0.0_f64; subseq_n];
    means[0] = sum / w;
    let var0 = (sum_sq / w - means[0] * means[0]).max(0.0);
    stds[0] = var0.sqrt();

    for i in 1..subseq_n {
        let drop = series[i - 1];
        let add = series[i + window - 1];
        sum += add - drop;
        sum_sq += add * add - drop * drop;
        let mean = sum / w;
        let var = (sum_sq / w - mean * mean).max(0.0);
        means[i] = mean;
        stds[i] = var.sqrt();
    }
    (means, stds)
}

/// Sliding dot product of `series` against a fixed `query` of
/// length `m`. Naïve `O(n · m)` — used once to seed the first
/// column of `QT`; subsequent columns ride the diagonal recurrence.
fn sliding_dot_product(series: &[f64], query: &[f64]) -> Vec<f64> {
    let m = query.len();
    let subseq_n = series.len() - m + 1;
    let mut out = vec![0.0_f64; subseq_n];
    for i in 0..subseq_n {
        let mut acc = 0.0_f64;
        for k in 0..m {
            acc += series[i + k] * query[k];
        }
        out[i] = acc;
    }
    out
}

/// Fold one row of the current `QT` column into the running
/// profile. `j` is the query index; `qt[i]` is the dot product of
/// `series[i..i+m]` with `series[j..j+m]`.
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
fn update_row(
    profile: &mut [f64],
    index: &mut [usize],
    qt: &[f64],
    j: usize,
    window: usize,
    means: &[f64],
    stds: &[f64],
    exclusion_zone: usize,
) {
    let m = window as f64;
    let sigma_j = stds[j];
    for i in 0..qt.len() {
        if i.abs_diff(j) < exclusion_zone {
            continue;
        }
        let sigma_i = stds[i];
        // Flat (constant) subsequence → distance undefined. Skip
        // rather than propagate `NaN`.
        if sigma_i == 0.0 || sigma_j == 0.0 {
            continue;
        }
        let numer = qt[i] - m * means[i] * means[j];
        let denom = m * sigma_i * sigma_j;
        // Clamp to the `[−1, 1]` Pearson range. QT noise in flat
        // regions can push the ratio slightly out of band.
        let corr = (numer / denom).clamp(-1.0, 1.0);
        let dist_sq = (2.0 * m * (1.0 - corr)).max(0.0);
        let dist = dist_sq.sqrt();
        if dist < profile[i] {
            profile[i] = dist;
            index[i] = j;
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    fn cosine_series(n: usize, freq: f64) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * freq).cos()).collect()
    }

    #[test]
    fn compute_rejects_tiny_window() {
        let data = cosine_series(128, 0.3);
        assert!(MatrixProfile::compute(&data, 3, None).is_err());
    }

    #[test]
    fn compute_rejects_short_series() {
        let data = cosine_series(10, 0.3);
        assert!(MatrixProfile::compute(&data, 8, None).is_err());
    }

    #[test]
    fn compute_rejects_non_finite_input() {
        let mut data = cosine_series(128, 0.3);
        data[42] = f64::NAN;
        assert!(MatrixProfile::compute(&data, 8, None).is_err());
    }

    #[test]
    fn profile_length_equals_subsequence_count() {
        let data = cosine_series(128, 0.3);
        let mp = MatrixProfile::compute(&data, 16, None).unwrap();
        assert_eq!(mp.len(), 128 - 16 + 1);
        assert_eq!(mp.profile().len(), mp.len());
        assert_eq!(mp.profile_index().len(), mp.len());
    }

    #[test]
    fn discord_finds_injected_anomaly() {
        let mut data = cosine_series(256, 0.25);
        // Inject a shape anomaly — large triangular pulse.
        for (k, v) in data.iter_mut().enumerate().skip(120).take(16) {
            *v += (k - 120) as f64 * 0.8;
        }
        let mp = MatrixProfile::compute(&data, 16, None).unwrap();
        let (pos, dist) = mp.discord();
        assert!(
            (100..=140).contains(&pos),
            "discord at unexpected position {pos}"
        );
        assert!(dist.is_finite() && dist > 0.0);
    }

    #[test]
    fn motif_finds_repeated_shape() {
        // Pure cosine → every window is a near-copy of some other
        // window → motif distance is tiny.
        let data = cosine_series(256, 0.2);
        let mp = MatrixProfile::compute(&data, 16, None).unwrap();
        let (_, d) = mp.motif();
        assert!(d < 0.5, "motif dist {d} unexpectedly large");
    }

    #[test]
    fn exclusion_zone_respected() {
        let data = cosine_series(128, 0.3);
        let mp = MatrixProfile::compute(&data, 16, Some(8)).unwrap();
        for (i, &j) in mp.profile_index().iter().enumerate() {
            // Skip entries whose profile is infinite (can happen if
            // every candidate is flat).
            if mp.profile()[i].is_finite() {
                assert!(
                    i.abs_diff(j) >= 8,
                    "neighbour inside exclusion zone: i={i} j={j}"
                );
            }
        }
    }

    #[test]
    fn discord_topk_suppresses_within_exclusion_zone() {
        let mut data = cosine_series(512, 0.25);
        for (k, v) in data.iter_mut().enumerate().skip(128).take(16) {
            *v += (k - 128) as f64 * 0.8;
        }
        for (k, v) in data.iter_mut().enumerate().skip(320).take(16) {
            *v -= (k - 320) as f64 * 0.8;
        }
        let mp = MatrixProfile::compute(&data, 16, None).unwrap();
        let top = mp.discord_topk(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].0.abs_diff(top[1].0) >= mp.exclusion_zone());
    }

    #[test]
    fn accessors_mirror_inputs() {
        let data = cosine_series(128, 0.3);
        let mp = MatrixProfile::compute(&data, 16, Some(6)).unwrap();
        assert_eq!(mp.window(), 16);
        assert_eq!(mp.exclusion_zone(), 6);
        assert!(!mp.is_empty());
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn postcard_roundtrip_preserves_profile() {
        let data = cosine_series(128, 0.3);
        let mp = MatrixProfile::compute(&data, 16, None).unwrap();
        let bytes = postcard::to_allocvec(&mp).expect("serde ok");
        let back: MatrixProfile = postcard::from_bytes(&bytes).expect("serde ok");
        assert_eq!(mp.profile(), back.profile());
        assert_eq!(mp.profile_index(), back.profile_index());
        assert_eq!(mp.window(), back.window());
        assert_eq!(mp.exclusion_zone(), back.exclusion_zone());
    }
}
