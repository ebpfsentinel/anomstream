//! Count-Min Sketch — probabilistic frequency estimation in
//! constant memory.
//!
//! `d` pairwise-independent hash rows over `w` counters. Each
//! `increment(key, c)` updates `d` cells (one per row); each
//! `estimate(key)` returns the minimum across the `d` cells the
//! key hashed to, guaranteeing `estimate(x) ≤ true_count(x) +
//! ε·N` with probability `1 − δ`, where `ε = e/w` and `δ =
//! (1/e)^d` (Cormode & Muthukrishnan 2005).
//!
//! Default (`w=2048`, `d=4`): `ε ≈ 1.33·10⁻³`, `δ ≈ 1.83·10⁻²`,
//! memory `~64 KB` — enough headroom for per-flow or per-source
//! heavy-hitter counting over a multi-million-key stream.
//!
//! Gated behind `std` because the row hashes rely on
//! [`core::hash::Hasher`] via [`std::hash::DefaultHasher`]
//! (`SipHash` 1-3). The rest of the crate's `no_std+alloc` surface is
//! unaffected.
//!
//! # References
//!
//! 1. G. Cormode, S. Muthukrishnan, "An Improved Data Stream
//!    Summary: The Count-Min Sketch and its Applications",
//!    *Journal of Algorithms* 55(1), 2005.

use alloc::vec;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};
use std::hash::DefaultHasher;

/// Probabilistic frequency counter with additive-error bound.
///
/// # Examples
///
/// ```
/// use rcf_rs::CountMinSketch;
///
/// let mut cms = CountMinSketch::new(2048, 4);
/// cms.increment(b"10.0.0.1", 100);
/// cms.increment(b"10.0.0.1", 50);
/// assert!(cms.estimate(b"10.0.0.1") >= 150);
/// ```
pub struct CountMinSketch {
    /// `depth × width` counter matrix; row `r` tracks hits under
    /// hash seed `r`.
    table: Vec<Vec<u64>>,
    /// Per-row deterministic seeds mixed into
    /// [`DefaultHasher`] so the rows are pairwise-independent.
    seeds: Vec<(u64, u64)>,
    /// Columns per row.
    width: usize,
    /// Rows — one pairwise-independent hash per row.
    depth: usize,
    /// Sum of every `count` ever passed to `increment`.
    total: u64,
}

impl CountMinSketch {
    /// Build a sketch with `width` columns and `depth` rows.
    ///
    /// Seeds are derived deterministically from the FNV and
    /// Knuth multiplicative hash constants so two instances with
    /// the same `(width, depth)` produce identical estimates for
    /// identical input streams.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(width: usize, depth: usize) -> Self {
        let seeds: Vec<(u64, u64)> = (0..depth)
            .map(|i| {
                let idx = i as u64 + 1;
                let a = 0x517c_c1b7_2722_0a95_u64.wrapping_mul(idx);
                let b = 0x6c62_272e_07bb_0142_u64.wrapping_mul(idx);
                (a, b)
            })
            .collect();

        Self {
            table: vec![vec![0_u64; width]; depth],
            seeds,
            width,
            depth,
            total: 0,
        }
    }

    /// Number of rows in the sketch.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of columns per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Increment `key`'s counters by `count`. Saturates at
    /// [`u64::MAX`] to keep adversarial streams from wrapping.
    pub fn increment(&mut self, key: &[u8], count: u64) {
        self.total = self.total.saturating_add(count);
        for row in 0..self.depth {
            let col = self.hash_to_col(key, row);
            self.table[row][col] = self.table[row][col].saturating_add(count);
        }
    }

    /// Upper-bound estimate of `key`'s true count. Always `≥ true
    /// count`; may overestimate by up to `ε·N` with probability
    /// `1 − δ`.
    #[must_use]
    pub fn estimate(&self, key: &[u8]) -> u64 {
        (0..self.depth)
            .map(|row| {
                let col = self.hash_to_col(key, row);
                self.table[row][col]
            })
            .min()
            .unwrap_or(0)
    }

    /// Sum of every `count` ever passed to [`Self::increment`].
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Reset every counter to zero. Allocation is preserved.
    pub fn reset(&mut self) {
        for row in &mut self.table {
            for cell in row.iter_mut() {
                *cell = 0;
            }
        }
        self.total = 0;
    }

    /// Counter-matrix footprint in bytes (excludes seeds / struct
    /// overhead).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.width * self.depth * core::mem::size_of::<u64>()
    }

    /// Map `(key, row)` to a column in `[0, width)` via a
    /// per-row-seeded [`DefaultHasher`].
    #[allow(clippy::cast_possible_truncation)]
    fn hash_to_col(&self, key: &[u8], row: usize) -> usize {
        let (a, b) = self.seeds[row];
        let mut hasher = DefaultHasher::new();
        a.hash(&mut hasher);
        key.hash(&mut hasher);
        b.hash(&mut hasher);
        (hasher.finish() as usize) % self.width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_increment_and_estimate() {
        let mut cms = CountMinSketch::new(2048, 4);
        cms.increment(b"192.168.1.1", 100);
        cms.increment(b"192.168.1.1", 50);
        cms.increment(b"10.0.0.1", 30);

        assert!(cms.estimate(b"192.168.1.1") >= 150);
        assert!(cms.estimate(b"10.0.0.1") >= 30);
        assert_eq!(cms.total(), 180);
    }

    #[test]
    fn reset_clears_all() {
        let mut cms = CountMinSketch::new(256, 3);
        cms.increment(b"key", 1000);
        assert!(cms.estimate(b"key") >= 1000);

        cms.reset();
        assert_eq!(cms.estimate(b"key"), 0);
        assert_eq!(cms.total(), 0);
    }

    #[test]
    fn accuracy_bounds_with_many_keys() {
        let mut cms = CountMinSketch::new(2048, 4);
        let n = 100_000_u64;

        for i in 0..n {
            let key = i.to_le_bytes();
            cms.increment(&key, 1);
        }

        let heavy = b"heavy_hitter";
        cms.increment(heavy, 1000);

        let estimate = cms.estimate(heavy);
        assert!(estimate >= 1000, "estimate {estimate} < true count 1000");
        assert!(
            estimate <= 1000 + 200,
            "estimate {estimate} too far from true count 1000 (> 200 error)"
        );
    }

    #[test]
    fn memory_footprint() {
        let cms = CountMinSketch::new(2048, 4);
        assert_eq!(cms.memory_bytes(), 2048 * 4 * 8); // 64 KB
    }

    #[test]
    fn different_keys_different_estimates() {
        let mut cms = CountMinSketch::new(1024, 4);
        cms.increment(b"alpha", 500);
        cms.increment(b"beta", 100);

        assert!(cms.estimate(b"alpha") >= 500);
        assert!(cms.estimate(b"beta") >= 100);
        assert!(cms.estimate(b"gamma") < 50);
    }

    #[test]
    fn saturates_at_u64_max() {
        let mut cms = CountMinSketch::new(64, 2);
        cms.increment(b"k", u64::MAX - 10);
        cms.increment(b"k", 100);
        assert_eq!(cms.estimate(b"k"), u64::MAX);
        assert_eq!(cms.total(), u64::MAX);
    }

    #[test]
    fn dim_accessors_match_constructor() {
        let cms = CountMinSketch::new(512, 5);
        assert_eq!(cms.width(), 512);
        assert_eq!(cms.depth(), 5);
    }
}
