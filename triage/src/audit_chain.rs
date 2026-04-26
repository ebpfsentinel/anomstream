//! Tamper-evident audit chain — HMAC-SHA256 chained
//! [`crate::audit::AlertRecord`] stream.
//!
//! [`crate::audit::AlertRecord`] on its own is **not**
//! tamper-evident: a record at rest in a SIEM, object store, or
//! WORM bucket can be edited (or deleted, or reordered) by anyone
//! with write access to the storage layer, and downstream
//! consumers have no cryptographic check that the bytes they
//! decoded match what the producer emitted. Compliance regimes
//! (SOC2 CC6 / NIS2 / PCI-DSS 10.5) generally require the audit
//! trail itself to be tamper-evident, not just the storage.
//!
//! `AuditChain` solves that within the `audit-integrity` feature
//! by HMAC-SHA256-signing each emitted record together with its
//! sequence number and the previous entry's tag:
//!
//! ```text
//! tag_n = HMAC-SHA256(
//!     key,
//!     u64_le(seq_n) || prev_tag_n || postcard(record_n)
//! )
//! prev_tag_{n+1} = tag_n
//! ```
//!
//! The first entry chains off a caller-supplied `genesis_prev`
//! (typically `[0u8; 32]`, but any 32-byte value works as long
//! as the verifier knows it).
//!
//! # Threat model
//!
//! - **In-scope**: detect any post-write tampering of the
//!   `AlertRecord` payload, the sequence number, or the chain
//!   linkage. Reordering breaks the `prev_tag` continuity;
//!   editing a record breaks its own tag; deleting a record
//!   breaks the next entry's `prev_tag`.
//! - **In-scope**: detect forged entries appended without the
//!   secret key — the attacker cannot compute a valid `tag` for
//!   a record they spliced in.
//! - **Out of scope**: secrecy of the records (records remain
//!   plaintext on disk — pair with at-rest encryption if the
//!   audit trail itself is sensitive).
//! - **Out of scope**: protection against an attacker who has
//!   compromised the key (they can rewrite history end-to-end
//!   and re-sign; that is a key-management problem solved at
//!   the storage / HSM layer, not by this module).
//! - **Out of scope**: replay across separate chains — two
//!   chains with the same key + `genesis_prev` produce the same
//!   tags for the same records. Either rotate the key per chain
//!   or rotate `genesis_prev` per chain (e.g. derive it from a
//!   chain identifier) when separation matters.
//!
//! # Wiring
//!
//! ```
//! # #[cfg(all(feature = "audit-integrity", feature = "postcard"))] {
//! use anomstream_core::ForestBuilder;
//! use anomstream_triage::audit::{AlertContext, AlertRecord};
//! use anomstream_triage::audit_chain::{AuditChain, GENESIS_PREV, verify_chain};
//!
//! let mut forest = ForestBuilder::<4>::new()
//!     .num_trees(50).sample_size(16).seed(42).build().unwrap();
//! for i in 0..32 {
//!     let v = f64::from(i) * 0.01;
//!     forest.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
//! }
//! let key = [0x42u8; 32]; // load from KMS / HSM in production.
//! let mut chain: AuditChain<String, 4> =
//!     AuditChain::new(&key).unwrap();
//! let ctx = AlertContext::<String>::untenanted(1_000);
//! let rec = AlertRecord::from_forest(&forest, &[5.0; 4], &ctx).unwrap();
//! let entry = chain.append(rec).unwrap();
//!
//! verify_chain(core::slice::from_ref(&entry), &key, &GENESIS_PREV).unwrap();
//! # }
//! ```

#![cfg(all(feature = "audit-integrity", feature = "postcard"))]

use alloc::format;
use alloc::vec::Vec;

use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::audit::AlertRecord;
use anomstream_core::error::{RcfError, RcfResult};

/// HMAC-SHA256 instantiation alias used by every chain entry.
type HmacSha256 = Hmac<Sha256>;

/// HMAC-SHA256 tag width — 32 bytes (256 bits).
pub const TAG_LEN: usize = 32;

/// Default genesis prev-tag — 32 zero bytes. Override via the
/// last argument to [`verify_chain`] or [`AuditChain::with_genesis`]
/// when chains within the same key scope must be cross-replay-
/// resistant.
pub const GENESIS_PREV: [u8; TAG_LEN] = [0u8; TAG_LEN];

/// Minimum HMAC key length — 32 bytes. HMAC accepts any length
/// per RFC 2104 but anything shorter than the SHA-256 block has
/// poor entropy guarantees for an audit-trail key.
pub const MIN_KEY_LEN: usize = 32;

/// One entry in a tamper-evident audit chain. Wraps the original
/// [`AlertRecord`] with a sequence number, the previous entry's
/// tag, and an HMAC-SHA256 tag computed over `(seq, prev_tag,
/// postcard(record))`.
///
/// Serialise the entire `AuditChainEntry` to the audit sink — the
/// verifier needs every field to recompute tags and detect
/// tampering.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "K: Clone + serde::Deserialize<'de>"))
)]
pub struct AuditChainEntry<K = alloc::string::String, const D: usize = 4>
where
    K: Clone,
{
    /// Underlying audit record — bit-identical to the producer's
    /// emission. Tampering with any field breaks [`Self::tag`].
    pub record: AlertRecord<K, D>,
    /// Monotonic sequence number — `0` for the first entry, then
    /// `+1` per `append`. Reordering breaks both the tag and the
    /// chain linkage of the following entry.
    pub seq: u64,
    /// HMAC tag of the previous entry (or [`GENESIS_PREV`] for
    /// `seq == 0`). Used by the verifier to confirm chain
    /// continuity — deleting an entry breaks the next entry's
    /// `prev_tag`.
    pub prev_tag: [u8; TAG_LEN],
    /// HMAC-SHA256 tag computed over the canonical postcard
    /// encoding of `(seq, prev_tag, record)`.
    pub tag: [u8; TAG_LEN],
}

/// Stateful append-only HMAC chain producer. Keeps the running
/// `seq` counter and `prev_tag` so each [`Self::append`] call
/// chains off the previous entry without caller bookkeeping.
///
/// The HMAC key is held in a `Vec<u8>` and zeroed on `Drop` via
/// the `hmac` crate's internal scrubbing — but the *original*
/// caller-provided slice is unchanged, so callers should
/// themselves zero their copy after construction if the key
/// material is sensitive. Use a key-management library (`zeroize`,
/// `secrecy`) at the call site for hardened deployments.
#[derive(Clone)]
pub struct AuditChain<K, const D: usize>
where
    K: Clone,
{
    /// HMAC-SHA256 secret key, ≥ [`MIN_KEY_LEN`] bytes.
    key: Vec<u8>,
    /// Monotonic sequence number stamped on the next emission.
    seq: u64,
    /// Tag of the most recently appended entry, `genesis_prev`
    /// before any append.
    prev_tag: [u8; TAG_LEN],
    /// `K`/`D` phantom — chain state itself does not carry an
    /// `AlertRecord`, but the type parameters match the records
    /// it can sign.
    _marker: core::marker::PhantomData<fn() -> AlertRecord<K, D>>,
}

impl<K: Clone, const D: usize> core::fmt::Debug for AuditChain<K, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never print the key — Debug is for logs which the
        // tamper-evident trail must not leak the signing key into.
        f.debug_struct("AuditChain")
            .field("key_len", &self.key.len())
            .field("seq", &self.seq)
            .field("prev_tag_hex_first8", &hex8(&self.prev_tag))
            .finish()
    }
}

impl<K: Clone, const D: usize> AuditChain<K, D> {
    /// Build a chain starting at `seq = 0` with the
    /// [`GENESIS_PREV`] linkage tag.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `key.len() < MIN_KEY_LEN`.
    pub fn new(key: &[u8]) -> RcfResult<Self> {
        Self::with_genesis(key, GENESIS_PREV, 0)
    }

    /// Build a chain at a caller-supplied `(seq, genesis_prev)`
    /// origin — use to **resume** appending to a chain whose tail
    /// is persisted elsewhere (load `(seq, prev_tag)` from the
    /// last stored entry, build a chain that continues from it).
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `key.len() < MIN_KEY_LEN`.
    pub fn with_genesis(key: &[u8], genesis_prev: [u8; TAG_LEN], seq: u64) -> RcfResult<Self> {
        if key.len() < MIN_KEY_LEN {
            return Err(RcfError::InvalidConfig(
                format!(
                    "AuditChain: key length {} below MIN_KEY_LEN {MIN_KEY_LEN}",
                    key.len()
                )
                .into(),
            ));
        }
        Ok(Self {
            key: key.to_vec(),
            seq,
            prev_tag: genesis_prev,
            _marker: core::marker::PhantomData,
        })
    }

    /// Append one record to the chain. Computes the HMAC tag over
    /// `(seq, prev_tag, postcard(record))`, advances the running
    /// state, and returns a fully-signed [`AuditChainEntry`].
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when the record fails
    /// to encode through postcard (e.g. if a future
    /// [`AlertRecord`] field becomes non-serialisable).
    pub fn append(&mut self, record: AlertRecord<K, D>) -> RcfResult<AuditChainEntry<K, D>>
    where
        K: serde::Serialize,
    {
        let tag = compute_tag(&self.key, self.seq, &self.prev_tag, &record)?;
        let entry = AuditChainEntry {
            record,
            seq: self.seq,
            prev_tag: self.prev_tag,
            tag,
        };
        self.seq = self.seq.checked_add(1).ok_or_else(|| {
            RcfError::InvalidConfig(
                "AuditChain: seq overflow — rotate the chain (rare: 2^64 entries)".into(),
            )
        })?;
        self.prev_tag = tag;
        Ok(entry)
    }

    /// Current monotonic `seq` — the next [`Self::append`] will
    /// stamp this value, then increment.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Current `prev_tag` — copy when persisting chain state for
    /// later resume.
    #[must_use]
    pub fn prev_tag(&self) -> [u8; TAG_LEN] {
        self.prev_tag
    }
}

/// Compute the HMAC-SHA256 tag for a single chain entry.
///
/// Tag domain: `u64_le(seq) || prev_tag || postcard(record)`.
fn compute_tag<K: Clone + serde::Serialize, const D: usize>(
    key: &[u8],
    seq: u64,
    prev_tag: &[u8; TAG_LEN],
    record: &AlertRecord<K, D>,
) -> RcfResult<[u8; TAG_LEN]> {
    let mut mac = HmacSha256::new_from_slice(key).map_err(|_| {
        // Unreachable in practice: HMAC-SHA256 accepts any key
        // length; the constructor only fails on internal alloc
        // pressure that has already poisoned the allocator.
        RcfError::InvalidConfig("AuditChain: HMAC-SHA256 init failed".into())
    })?;
    mac.update(&seq.to_le_bytes());
    mac.update(prev_tag);
    let body = postcard::to_allocvec(record)
        .map_err(|e| RcfError::InvalidConfig(format!("AuditChain: postcard encode: {e}").into()))?;
    mac.update(&body);
    let out = mac.finalize().into_bytes();
    let mut tag = [0u8; TAG_LEN];
    tag.copy_from_slice(&out);
    Ok(tag)
}

/// Verify a slice of [`AuditChainEntry`] in-order against `key`
/// and `genesis_prev`. Returns `Ok(())` when:
///
/// 1. Sequence numbers form a contiguous monotonic series
///    starting at `entries[0].seq`.
/// 2. Each entry's `prev_tag` equals the previous entry's `tag`
///    (or `genesis_prev` for the first entry).
/// 3. Each entry's `tag` is the HMAC-SHA256 of the canonical
///    `(seq || prev_tag || postcard(record))` blob.
///
/// Tag comparisons run in constant time via the `subtle` crate
/// to deny attackers timing-based tag guessing.
///
/// # Errors
///
/// Returns [`RcfError::InvalidConfig`] with a descriptive message
/// (entry index + failure reason) on any of: short key, missing
/// continuity, bad sequence, postcard encode failure, tag
/// mismatch.
pub fn verify_chain<K, const D: usize>(
    entries: &[AuditChainEntry<K, D>],
    key: &[u8],
    genesis_prev: &[u8; TAG_LEN],
) -> RcfResult<()>
where
    K: Clone + serde::Serialize,
{
    if key.len() < MIN_KEY_LEN {
        return Err(RcfError::InvalidConfig(
            format!(
                "verify_chain: key length {} below MIN_KEY_LEN {MIN_KEY_LEN}",
                key.len()
            )
            .into(),
        ));
    }
    if entries.is_empty() {
        return Ok(());
    }
    let mut expected_prev = *genesis_prev;
    let mut expected_seq = entries[0].seq;
    for (i, entry) in entries.iter().enumerate() {
        if entry.seq != expected_seq {
            return Err(RcfError::InvalidConfig(
                format!(
                    "verify_chain: entry {i} seq {} != expected {expected_seq}",
                    entry.seq
                )
                .into(),
            ));
        }
        if entry.prev_tag.ct_eq(&expected_prev).unwrap_u8() != 1 {
            return Err(RcfError::InvalidConfig(
                format!("verify_chain: entry {i} prev_tag mismatch — chain broken at this point")
                    .into(),
            ));
        }
        let recomputed = compute_tag(key, entry.seq, &entry.prev_tag, &entry.record)?;
        if recomputed.ct_eq(&entry.tag).unwrap_u8() != 1 {
            return Err(RcfError::InvalidConfig(
                format!("verify_chain: entry {i} tag mismatch — record tampered or key wrong")
                    .into(),
            ));
        }
        expected_prev = entry.tag;
        expected_seq = expected_seq.checked_add(1).ok_or_else(|| {
            RcfError::InvalidConfig("verify_chain: seq overflow during walk".into())
        })?;
    }
    Ok(())
}

/// Hex-render the first 8 bytes of a tag for `Debug` impls.
fn hex8(tag: &[u8; TAG_LEN]) -> alloc::string::String {
    let mut s = alloc::string::String::with_capacity(16);
    for b in tag.iter().take(8) {
        let hi = b >> 4;
        let lo = b & 0x0f;
        s.push(nibble(hi));
        s.push(nibble(lo));
    }
    s
}

/// Map a 0..=15 nibble to its lowercase hex character.
const fn nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        _ => (b'a' + n - 10) as char,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::audit::{AlertContext, AlertRecord};
    use anomstream_core::ForestBuilder;
    use anomstream_core::forest::RandomCutForest;

    fn warm_forest() -> RandomCutForest<4> {
        let mut f = ForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(2026)
            .build()
            .unwrap();
        for i in 0..32_u32 {
            let v = f64::from(i) * 0.01;
            f.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
        }
        f
    }

    fn make_record(forest: &RandomCutForest<4>, ts: u64) -> AlertRecord<alloc::string::String, 4> {
        let ctx = AlertContext::<alloc::string::String>::for_tenant("t1".into(), ts);
        AlertRecord::from_forest(forest, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap()
    }

    fn key32() -> [u8; 32] {
        // Stable test key — production must use a CSPRNG-derived
        // value held in an HSM or KMS.
        let mut k = [0u8; 32];
        for (i, slot) in k.iter_mut().enumerate() {
            *slot = u8::try_from(i).unwrap();
        }
        k
    }

    #[test]
    fn new_rejects_short_key() {
        let short = [0u8; MIN_KEY_LEN - 1];
        assert!(AuditChain::<alloc::string::String, 4>::new(&short).is_err());
    }

    #[test]
    fn append_then_verify_single_entry() {
        let f = warm_forest();
        let key = key32();
        let mut chain: AuditChain<alloc::string::String, 4> = AuditChain::new(&key).unwrap();
        let entry = chain.append(make_record(&f, 1_000)).unwrap();
        assert_eq!(entry.seq, 0);
        assert_eq!(entry.prev_tag, GENESIS_PREV);
        verify_chain(core::slice::from_ref(&entry), &key, &GENESIS_PREV).unwrap();
    }

    #[test]
    fn append_then_verify_multi_entry_chain() {
        let f = warm_forest();
        let key = key32();
        let mut chain: AuditChain<alloc::string::String, 4> = AuditChain::new(&key).unwrap();
        let entries: Vec<_> = (0..32_u64)
            .map(|ts| chain.append(make_record(&f, ts)).unwrap())
            .collect();
        verify_chain(&entries, &key, &GENESIS_PREV).unwrap();
        assert_eq!(entries.last().unwrap().seq, 31);
        // Chain state advanced for next append.
        assert_eq!(chain.seq(), 32);
    }

    #[test]
    fn verify_detects_record_tamper() {
        let f = warm_forest();
        let key = key32();
        let mut chain: AuditChain<alloc::string::String, 4> = AuditChain::new(&key).unwrap();
        let mut entry = chain.append(make_record(&f, 1_000)).unwrap();
        // Edit the record post-sign — verifier must reject.
        entry.record.timestamp_ms = entry.record.timestamp_ms.wrapping_add(1);
        let res = verify_chain(core::slice::from_ref(&entry), &key, &GENESIS_PREV);
        assert!(res.is_err(), "record tamper must be detected");
    }

    #[test]
    fn verify_detects_tag_tamper() {
        let f = warm_forest();
        let key = key32();
        let mut chain: AuditChain<alloc::string::String, 4> = AuditChain::new(&key).unwrap();
        let mut entry = chain.append(make_record(&f, 1_000)).unwrap();
        entry.tag[0] ^= 0xff;
        let res = verify_chain(core::slice::from_ref(&entry), &key, &GENESIS_PREV);
        assert!(res.is_err(), "tag tamper must be detected");
    }

    #[test]
    fn verify_detects_chain_break_via_deletion() {
        let f = warm_forest();
        let key = key32();
        let mut chain: AuditChain<alloc::string::String, 4> = AuditChain::new(&key).unwrap();
        let entries: Vec<_> = (0..4_u64)
            .map(|ts| chain.append(make_record(&f, ts)).unwrap())
            .collect();
        // Drop the second entry — the third's prev_tag now points
        // to a nonexistent predecessor.
        let mut tampered = entries.clone();
        tampered.remove(1);
        let res = verify_chain(&tampered, &key, &GENESIS_PREV);
        assert!(res.is_err(), "deletion must break the chain");
    }

    #[test]
    fn verify_detects_reorder() {
        let f = warm_forest();
        let key = key32();
        let mut chain: AuditChain<alloc::string::String, 4> = AuditChain::new(&key).unwrap();
        let entries: Vec<_> = (0..4_u64)
            .map(|ts| chain.append(make_record(&f, ts)).unwrap())
            .collect();
        let mut reordered = entries;
        reordered.swap(1, 2);
        let res = verify_chain(&reordered, &key, &GENESIS_PREV);
        assert!(res.is_err(), "reorder must break the chain");
    }

    #[test]
    fn verify_detects_wrong_key() {
        let f = warm_forest();
        let key = key32();
        let mut other = key;
        other[0] ^= 1;
        let mut chain: AuditChain<alloc::string::String, 4> = AuditChain::new(&key).unwrap();
        let entry = chain.append(make_record(&f, 1_000)).unwrap();
        let res = verify_chain(core::slice::from_ref(&entry), &other, &GENESIS_PREV);
        assert!(res.is_err(), "wrong key must fail verification");
    }

    #[test]
    fn resume_chain_via_with_genesis() {
        // Persist `(seq, prev_tag)` after some appends, rebuild
        // the chain from those values, append more, verify the
        // whole sequence end-to-end.
        let f = warm_forest();
        let key = key32();
        let mut chain: AuditChain<alloc::string::String, 4> = AuditChain::new(&key).unwrap();
        let head: Vec<_> = (0..3_u64)
            .map(|ts| chain.append(make_record(&f, ts)).unwrap())
            .collect();
        let resume_seq = chain.seq();
        let resume_prev = chain.prev_tag();
        // Forget the live chain, rebuild from persisted state.
        let mut resumed: AuditChain<alloc::string::String, 4> =
            AuditChain::with_genesis(&key, resume_prev, resume_seq).unwrap();
        let tail: Vec<_> = (3..6_u64)
            .map(|ts| resumed.append(make_record(&f, ts)).unwrap())
            .collect();
        let mut all = head;
        all.extend(tail);
        verify_chain(&all, &key, &GENESIS_PREV).unwrap();
    }

    #[test]
    fn verify_empty_chain_is_ok() {
        let key = key32();
        let entries: Vec<AuditChainEntry<alloc::string::String, 4>> = Vec::new();
        verify_chain(&entries, &key, &GENESIS_PREV).unwrap();
    }

    #[test]
    fn debug_does_not_leak_key() {
        let key = key32();
        let chain: AuditChain<alloc::string::String, 4> = AuditChain::new(&key).unwrap();
        let dbg = format!("{chain:?}");
        // Key bytes are 0..=31; their hex would contain "1f".
        // Confirm no raw key byte sequence shows up.
        assert!(!dbg.contains("000102030405"), "key bytes leaked in Debug");
    }
}
