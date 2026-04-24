//! `HyperLogLog` — probabilistic distinct-count (cardinality)
//! estimator.
//!
//! `m = 2^p` register bank; each `add(x)` hashes `x` into 64 bits,
//! uses the top `p` bits as a register index and counts the
//! leading zeros of the remaining `64 − p` bits. Per-register
//! state is `max(zeros + 1)` across every element routed there.
//! Cardinality is recovered by a harmonic mean of `2^(-register)`
//! with an `α_m` bias correction, plus small-range linear
//! counting when many registers are still empty.
//!
//! Memory: `m` bytes (one `u8` per register). Typical configs:
//!
//! | `p` | `m` | memory | standard error |
//! |---|---|---|---|
//! | 10 | 1 024  | 1 KiB   | 3.25 %  |
//! | 12 | 4 096  | 4 KiB   | 1.625 % |
//! | 14 | 16 384 | 16 KiB  | 0.81 %  |
//! | 16 | 65 536 | 64 KiB  | 0.40 %  |
//!
//! Gated behind the `std` feature because the hash path relies on
//! [`std::hash::DefaultHasher`] (`SipHash` 1-3).
//!
//! # References
//!
//! 1. P. Flajolet, É. Fusy, O. Gandouet, F. Meunier,
//!    "`HyperLogLog`: the analysis of a near-optimal cardinality
//!    estimation algorithm", `AofA` 2007.
//! 2. S. Heule, M. Nunkesser, A. Hall, "`HyperLogLog` in Practice:
//!    Algorithmic Engineering of a State of the Art Cardinality
//!    Estimation Algorithm", EDBT 2013.

use alloc::vec;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};
use std::hash::DefaultHasher;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::error::{RcfError, RcfResult};

/// Minimum precision bit count — 16 registers.
pub const MIN_PRECISION: u8 = 4;
/// Maximum precision bit count — 65 536 registers.
pub const MAX_PRECISION: u8 = 16;
/// Default precision `p = 12` — 4 096 registers, ≈ 1.625 % std
/// error, ~4 KiB memory.
pub const DEFAULT_PRECISION: u8 = 12;

/// Probabilistic distinct-count sketch.
///
/// # Examples
///
/// ```
/// use anomstream_core::HyperLogLog;
///
/// let mut hll = HyperLogLog::with_default_precision();
/// for ip in 0..10_000_u32 {
///     hll.add(&ip.to_le_bytes());
/// }
/// let est = hll.estimate();
/// let err = (est as i64 - 10_000).unsigned_abs() as f64 / 10_000.0;
/// assert!(err < 0.05); // 3× the theoretical 1.625 % — conservative
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "HyperLogLogShadow"))]
pub struct HyperLogLog {
    /// Precision — register count is `2^precision`.
    precision: u8,
    /// Per-register max-leading-zero count. `len == 2^precision`.
    registers: Vec<u8>,
    /// Total values offered to [`Self::add`] — ops signal, not
    /// used by the estimator.
    total_added: u64,
}

/// Over-the-wire [`HyperLogLog`] layout — mirrors the public type
/// field-for-field. Deserialization lands here first so
/// [`TryFrom`] can re-run the constructor's invariant checks
/// (`precision ∈ [MIN, MAX]`, register bank length `== 2^precision`)
/// before a live sketch is handed out.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::missing_docs_in_private_items)]
struct HyperLogLogShadow {
    precision: u8,
    registers: Vec<u8>,
    total_added: u64,
}

#[cfg(feature = "serde")]
impl TryFrom<HyperLogLogShadow> for HyperLogLog {
    type Error = RcfError;

    fn try_from(raw: HyperLogLogShadow) -> Result<Self, Self::Error> {
        if !(MIN_PRECISION..=MAX_PRECISION).contains(&raw.precision) {
            return Err(RcfError::InvalidConfig(
                alloc::format!(
                    "HyperLogLog: precision {} out of [{MIN_PRECISION}, {MAX_PRECISION}]",
                    raw.precision
                )
                .into(),
            ));
        }
        let expected = 1_usize << raw.precision;
        if raw.registers.len() != expected {
            return Err(RcfError::InvalidConfig(
                alloc::format!(
                    "HyperLogLog: register bank length {} != expected {expected} for precision {}",
                    raw.registers.len(),
                    raw.precision
                )
                .into(),
            ));
        }
        Ok(Self {
            precision: raw.precision,
            registers: raw.registers,
            total_added: raw.total_added,
        })
    }
}

impl HyperLogLog {
    /// Build a sketch with caller-chosen precision.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `precision` is
    /// outside `[MIN_PRECISION, MAX_PRECISION]`.
    pub fn new(precision: u8) -> RcfResult<Self> {
        if !(MIN_PRECISION..=MAX_PRECISION).contains(&precision) {
            return Err(RcfError::InvalidConfig(
                alloc::format!(
                    "HyperLogLog: precision {precision} out of [{MIN_PRECISION}, {MAX_PRECISION}]"
                )
                .into(),
            ));
        }
        let m = 1_usize << precision;
        Ok(Self {
            precision,
            registers: vec![0_u8; m],
            total_added: 0,
        })
    }

    /// Default sketch — `p = 12`, 4 096 registers, ≈ 1.625 % std
    /// error, ~4 KiB memory.
    ///
    /// # Panics
    ///
    /// Never in practice — [`DEFAULT_PRECISION`] is a compile-time
    /// constant validated against the `[MIN, MAX]` range.
    #[must_use]
    pub fn with_default_precision() -> Self {
        Self::new(DEFAULT_PRECISION).expect("DEFAULT_PRECISION is in range")
    }

    /// Register count `m = 2^p`.
    #[must_use]
    pub fn register_count(&self) -> usize {
        1_usize << self.precision
    }

    /// Precision bit count.
    #[must_use]
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// Total values passed through [`Self::add`].
    #[must_use]
    pub fn total_added(&self) -> u64 {
        self.total_added
    }

    /// Memory footprint in bytes (register bank only).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.registers.len()
    }

    /// Ingest a `Hash`-able value. Pre-hashes through
    /// [`DefaultHasher`] (`SipHash`) so per-key distribution is
    /// uniform irrespective of the user type's own `Hash` impl
    /// quality.
    pub fn add<T: Hash + ?Sized>(&mut self, value: &T) {
        let mut h = DefaultHasher::new();
        value.hash(&mut h);
        self.add_hash(h.finish());
    }

    /// Ingest a raw byte key. Cheaper when the caller already has
    /// a fixed-size fingerprint (e.g. `[u8; 16]` IP, flow-hash
    /// tuple) — skips the generic `Hash` dispatch.
    pub fn add_bytes(&mut self, key: &[u8]) {
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        self.add_hash(h.finish());
    }

    /// Ingest a caller-supplied 64-bit hash. Escape hatch for
    /// callers with a stronger hasher (e.g. xxhash, siphash with
    /// a keyed seed) — the sketch's accuracy depends on
    /// `hash % 2^p` being uniform.
    #[allow(clippy::cast_possible_truncation)]
    pub fn add_hash(&mut self, hash: u64) {
        self.total_added = self.total_added.saturating_add(1);
        let p = self.precision;
        // `hash >> (64 - p)` yields a value in `[0, 2^p)`; the
        // `as usize` cast is infallible on 32-bit+ targets since
        // `p ≤ 16` bounds the result to ≤ 65 535.
        let idx = (hash >> (64 - p)) as usize;
        // Retained bits after the index — shifted so `leading_zeros`
        // counts within the full 64-bit lane. Result fits `u8`
        // because `leading_zeros` is in `[0, 64]` and `p ≥ 4`.
        let tail = hash << p;
        let rho = if tail == 0 {
            64 - p + 1
        } else {
            (tail.leading_zeros() as u8) + 1
        };
        let slot = &mut self.registers[idx];
        if rho > *slot {
            *slot = rho;
        }
    }

    /// Cardinality estimate — number of distinct values ingested.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn estimate(&self) -> u64 {
        let m = self.register_count();
        let m_f = m as f64;

        // Harmonic mean of 2^(-register).
        let mut sum = 0.0_f64;
        let mut zeros: usize = 0;
        for &r in &self.registers {
            if r == 0 {
                zeros += 1;
            }
            sum += 2.0_f64.powi(-(i32::from(r)));
        }

        let alpha = alpha_m(m);
        let raw = alpha * m_f * m_f / sum;

        // Small-range correction — switch to linear counting
        // when many registers still hold the initial zero.
        if raw <= 2.5 * m_f && zeros > 0 {
            let v = zeros as f64;
            return (m_f * (m_f / v).ln()).round().max(0.0) as u64;
        }

        // No large-range correction needed for 64-bit hashes —
        // the usual `2^32` ceiling only matters for 32-bit output.
        raw.round().max(0.0) as u64
    }

    /// Fold `other` into `self` by taking a per-register maximum.
    /// Two sketches must share the same precision — HLL merge is
    /// the whole reason the sketch is decomposable across shards
    /// / time windows.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when the two sketches
    /// disagree on `precision`.
    pub fn merge(&mut self, other: &Self) -> RcfResult<()> {
        if self.precision != other.precision {
            return Err(RcfError::InvalidConfig(
                alloc::format!(
                    "HyperLogLog::merge: precision mismatch ({} vs {})",
                    self.precision,
                    other.precision
                )
                .into(),
            ));
        }
        for (slot, other_r) in self.registers.iter_mut().zip(other.registers.iter()) {
            if *other_r > *slot {
                *slot = *other_r;
            }
        }
        self.total_added = self.total_added.saturating_add(other.total_added);
        Ok(())
    }

    /// Zero every register. Allocation is preserved.
    pub fn reset(&mut self) {
        self.registers.iter_mut().for_each(|r| *r = 0);
        self.total_added = 0;
    }
}

/// Bias correction coefficient `α_m` (Flajolet 2007, Figure 3).
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::items_after_statements,
    clippy::manual_range_contains
)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_out_of_range_precision() {
        assert!(HyperLogLog::new(3).is_err());
        assert!(HyperLogLog::new(17).is_err());
        assert!(HyperLogLog::new(MIN_PRECISION).is_ok());
        assert!(HyperLogLog::new(MAX_PRECISION).is_ok());
    }

    #[test]
    fn empty_sketch_estimates_zero() {
        let hll = HyperLogLog::with_default_precision();
        assert_eq!(hll.estimate(), 0);
    }

    #[test]
    fn exact_cardinality_at_tiny_scale() {
        // Linear-counting regime: small cardinality should land
        // very close to the truth (no variance floor yet).
        let mut hll = HyperLogLog::with_default_precision();
        for i in 0..10_u32 {
            hll.add(&i.to_le_bytes());
        }
        let est = hll.estimate();
        assert!(est >= 9 && est <= 11, "est {est}");
    }

    #[test]
    fn estimates_within_5pct_at_10k_distinct() {
        let mut hll = HyperLogLog::with_default_precision();
        for i in 0..10_000_u32 {
            hll.add(&i.to_le_bytes());
        }
        let est = hll.estimate();
        let err = (est as i64 - 10_000).unsigned_abs() as f64 / 10_000.0;
        assert!(err < 0.05, "err {err:.4}, est {est}");
    }

    #[test]
    fn estimates_within_3pct_at_100k_distinct_p14() {
        let mut hll = HyperLogLog::new(14).expect("p=14 in range");
        for i in 0..100_000_u32 {
            hll.add(&i.to_le_bytes());
        }
        let est = hll.estimate();
        let err = (est as i64 - 100_000).unsigned_abs() as f64 / 100_000.0;
        assert!(err < 0.03, "err {err:.4}, est {est}");
    }

    #[test]
    fn duplicate_inserts_do_not_inflate_estimate() {
        let mut hll = HyperLogLog::with_default_precision();
        for _ in 0..1_000 {
            for i in 0..100_u32 {
                hll.add(&i.to_le_bytes());
            }
        }
        let est = hll.estimate();
        // Ingested 100k times but only 100 distinct values.
        assert!(est >= 90 && est <= 110, "est {est}");
        assert_eq!(hll.total_added(), 100_000);
    }

    #[test]
    fn merge_agrees_with_single_sketch() {
        let mut a = HyperLogLog::with_default_precision();
        let mut b = HyperLogLog::with_default_precision();
        let mut full = HyperLogLog::with_default_precision();
        for i in 0..5_000_u32 {
            a.add(&i.to_le_bytes());
            full.add(&i.to_le_bytes());
        }
        for i in 5_000..10_000_u32 {
            b.add(&i.to_le_bytes());
            full.add(&i.to_le_bytes());
        }
        a.merge(&b).expect("same precision");
        // Merged estimate should hit within a register-noise
        // window of the single-sketch ground truth.
        let merged_est = a.estimate();
        let full_est = full.estimate();
        let delta = (merged_est as i64 - full_est as i64).unsigned_abs();
        assert!(
            delta < 200,
            "delta {delta}, merged {merged_est}, full {full_est}"
        );
    }

    #[test]
    fn merge_rejects_precision_mismatch() {
        let mut a = HyperLogLog::new(10).unwrap();
        let b = HyperLogLog::new(12).unwrap();
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn reset_clears_all_registers_and_counter() {
        let mut hll = HyperLogLog::with_default_precision();
        for i in 0..1_000_u32 {
            hll.add(&i.to_le_bytes());
        }
        assert!(hll.estimate() > 0);
        hll.reset();
        assert_eq!(hll.estimate(), 0);
        assert_eq!(hll.total_added(), 0);
    }

    #[test]
    fn add_hash_path_matches_add_bytes_path() {
        let mut a = HyperLogLog::with_default_precision();
        let mut b = HyperLogLog::with_default_precision();
        for i in 0..1_000_u32 {
            let bytes = i.to_le_bytes();
            a.add_bytes(&bytes);
            // Mirror the same hash DefaultHasher produces.
            use core::hash::Hash;
            use std::hash::DefaultHasher;
            let mut h = DefaultHasher::new();
            bytes.hash(&mut h);
            b.add_hash(h.finish());
        }
        // Bit-exact agreement: same hash → same register update.
        assert_eq!(a.estimate(), b.estimate());
    }

    #[test]
    fn memory_bytes_matches_register_count() {
        let hll = HyperLogLog::new(12).unwrap();
        assert_eq!(hll.memory_bytes(), 4096);
        let hll16 = HyperLogLog::new(16).unwrap();
        assert_eq!(hll16.memory_bytes(), 65_536);
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn postcard_roundtrip_preserves_estimate() {
        let mut hll = HyperLogLog::with_default_precision();
        for i in 0..5_000_u32 {
            hll.add(&i.to_le_bytes());
        }
        let before = hll.estimate();
        let bytes = postcard::to_allocvec(&hll).expect("serde ok");
        let back: HyperLogLog = postcard::from_bytes(&bytes).expect("serde ok");
        assert_eq!(back.estimate(), before);
        assert_eq!(back.total_added(), hll.total_added());
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn deserialize_rejects_out_of_range_precision() {
        let bad = HyperLogLogShadow {
            precision: MAX_PRECISION + 1,
            registers: alloc::vec![0_u8; 1 << MAX_PRECISION],
            total_added: 0,
        };
        let bytes = postcard::to_allocvec(&bad).unwrap();
        let back: Result<HyperLogLog, _> = postcard::from_bytes(&bytes);
        assert!(back.is_err());
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn deserialize_rejects_register_length_mismatch() {
        let bad = HyperLogLogShadow {
            precision: DEFAULT_PRECISION,
            registers: alloc::vec![0_u8; 10], // should be 4096.
            total_added: 0,
        };
        let bytes = postcard::to_allocvec(&bad).unwrap();
        let back: Result<HyperLogLog, _> = postcard::from_bytes(&bytes);
        assert!(back.is_err());
    }
}
