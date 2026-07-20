//! Hot-path integration primitives for `anomstream-core` callers that run on
//! latency-critical ingress paths (eBPF TC action, XDP, per-packet
//! anomaly classifier).
//!
//! The bare [`anomstream_core::RandomCutForest::update`] costs tens of
//! microseconds at the AWS-default `(100 trees, 256 samples, D = 16)`
//! config — an order of magnitude above the few-µs budget a TC action
//! has per packet at 10 Gbps+. (See `docs/performance.md` for current
//! figures; treat any absolute quoted here as indicative, since the
//! ensemble cost scales with `num_trees × D`.) Making `update` faster
//! hits diminishing returns; the architectural answer is to
//! **decouple** the score-on-path from the update-off-path and shed
//! low-value updates before they queue.
//!
//! This module ships two orthogonal building blocks:
//!
//! - [`UpdateSampler`] — stride or per-flow-hash decision
//!   "does this packet contribute to the reservoir?" — discards the
//!   rest before any RCF work. Per-flow sampling keeps the baseline
//!   shape (every flow has *some* representation) while cutting the
//!   update rate by the sampling ratio.
//! - [`update_channel`] — bounded MPSC channel carrying `[f64; D]` points
//!   from classifier threads to a **single dedicated updater
//!   thread**. The classifier thread `try_enqueue`s non-blockingly
//!   and scores against the *previous* forest snapshot; the updater
//!   thread drains the queue into `RandomCutForest::update` at its
//!   own cadence. Dropped points (queue full) are tallied for ops.
//!
//! Coarser `time_decay` is the third dial, but that's configured
//! directly on [`anomstream_core::ForestBuilder::time_decay`]; see
//! `docs/performance.md` for the trade-off.
//!
//! # Counter semantics
//!
//! All lifetime counters ([`UpdateSampler::accepted_total`],
//! `rejected_total`, `UpdateProducer::enqueued`, `dropped_total`,
//! [`PrefixRateCap::admitted_total`], `capped_total`) are plain
//! `AtomicU64::fetch_add(1, Relaxed)`. Atomic `fetch_add` is
//! wrapping by definition — `profile.release.overflow-checks` does
//! not apply to atomic operations, so no panic can occur at
//! wrap-around. At `10 Gpps` sustained load a `u64` wraps in
//! roughly 58 years, well past any realistic deployment lifetime.
//! Export-and-reset cadence is therefore only needed when an
//! operator wants fine-grained delta metrics; no correctness
//! pressure exists to call `*_total_reset` at any particular
//! frequency.
//!
//! # Example: classifier + updater split
//!
//! ```ignore
//! use anomstream_core::{ForestBuilder, hot_path};
//! use std::thread;
//!
//! let mut forest = ForestBuilder::<16>::new()
//!     .num_trees(100)
//!     .sample_size(256)
//!     .seed(42)
//!     .build()?;
//!
//! let (producer, mut consumer) = hot_path::update_channel::<16>(4096);
//! let sampler = hot_path::UpdateSampler::new(8); // 1/8 per-flow
//!
//! // Updater thread.
//! thread::spawn(move || loop {
//!     let (ingested, errors) = consumer.try_drain(|p| forest.update(p));
//!     if ingested == 0 && errors == 0 {
//!         std::thread::sleep(std::time::Duration::from_millis(1));
//!     }
//! });
//!
//! // Classifier thread (hot path — runs per packet).
//! fn on_packet(
//!     features: [f64; 16],
//!     flow_hash: u64,
//!     sampler: &hot_path::UpdateSampler,
//!     producer: &hot_path::UpdateProducer<16>,
//! ) {
//!     if sampler.accept_hash(flow_hash) {
//!         let _ = producer.try_enqueue(features);
//!     }
//! }
//! ```

#![cfg(feature = "std")]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::panic))]

use core::num::{NonZeroU32, NonZeroU64};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use anomstream_core::error::{RcfError, RcfResult};
use anomstream_core::metrics::{MetricsSink, default_sink, names};

/// Hard ceiling on the [`update_channel`] capacity. The bounded
/// MPSC channel allocates `capacity * size_of::<[f64; D]>()` bytes
/// up front; at `D = 16` and `MAX_CHANNEL_CAPACITY = 1_048_576`
/// the worst-case per-channel footprint is 128 MiB — well above
/// any realistic ingress rate × drain-cadence product. The cap
/// exists to defeat caller-controlled OOM at construction; raise
/// it deliberately if a deployment really needs more in-flight
/// queue depth.
pub const MAX_CHANNEL_CAPACITY: usize = 1 << 20;

/// Number of hot-path operations between [`MetricsSink`] flushes.
/// Per-call vtable dispatch on `Arc<dyn MetricsSink>` measurable
/// at line rate (≈13 ns × 12.5 Mpps ≈ 160 ms / s of clock burned
/// on dispatch alone). Buffering increments locally and flushing
/// every `METRICS_BATCH_SIZE` ops cuts that overhead by the same
/// factor while keeping the in-process [`AtomicU64`] counters
/// (`accepted_total`, `enqueued_total`, …) bit-exact every call.
/// Sink lag at process-exit is bounded by `METRICS_BATCH_SIZE`
/// minus one — call [`UpdateSampler::flush_metrics`] /
/// [`UpdateProducer::flush_metrics`] / [`PrefixRateCap::flush_metrics`]
/// before shutdown to drain the residue.
pub const METRICS_BATCH_SIZE: u64 = 64;

/// Increment `counter` by 1 and flush `METRICS_BATCH_SIZE` units
/// to `sink` every `METRICS_BATCH_SIZE` calls. Returns the
/// post-increment counter value. `last_flushed` advances in
/// lockstep with the sink emission so a subsequent
/// [`flush_batched`] only drains the residue (last `< BATCH`
/// increments) without double-counting.
#[inline]
fn record_batched(
    counter: &AtomicU64,
    last_flushed: &AtomicU64,
    sink: &Arc<dyn MetricsSink>,
    metric: &'static str,
) -> u64 {
    let next = counter.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    if next.is_multiple_of(METRICS_BATCH_SIZE) {
        sink.inc_counter(metric, METRICS_BATCH_SIZE);
        last_flushed.fetch_add(METRICS_BATCH_SIZE, Ordering::Relaxed);
    }
    next
}

/// Manually flush whatever residue (`counter - last_flushed`) has
/// accumulated since the last batched emission. Idempotent on
/// repeated calls — the second call emits nothing. Use to drain
/// the trailing 0..[`METRICS_BATCH_SIZE`] increments at process
/// shutdown or before exporting a metrics snapshot.
#[inline]
fn flush_batched(
    counter: &AtomicU64,
    last_flushed: &AtomicU64,
    sink: &Arc<dyn MetricsSink>,
    metric: &'static str,
) {
    let now = counter.load(Ordering::Relaxed);
    let prev = last_flushed.swap(now, Ordering::Relaxed);
    let delta = now.wrapping_sub(prev);
    if delta > 0 {
        sink.inc_counter(metric, delta);
    }
}

/// Stride-based or per-flow-hash update sampler.
///
/// Accepts `1 / keep_every_n` of the offered updates. The sampler
/// itself never touches the forest — callers invoke `accept_*`
/// before calling [`anomstream_core::RandomCutForest::update`].
///
/// `keep_every_n = 0` and `keep_every_n = 1` both disable sampling
/// (every offer is accepted). Values `>= 2` gate proportionally.
#[derive(Debug)]
#[non_exhaustive]
pub struct UpdateSampler {
    /// Divisor: keep `1 / keep_every_n` offered updates.
    keep_every_n: u32,
    /// Monotonic stride counter for [`Self::accept_stride`].
    counter: AtomicU64,
    /// Running total of accepted offers — observability signal.
    accepted: AtomicU64,
    /// Running total of rejected offers.
    rejected: AtomicU64,
    /// Sink-side cumulative emitted count for `accepted` — paired
    /// with `accepted` to drain the residue on
    /// [`Self::flush_metrics`] without double-counting.
    accepted_flushed: AtomicU64,
    /// Sink-side cumulative emitted count for `rejected`.
    rejected_flushed: AtomicU64,
    /// Per-sampler secret multipliers used by [`Self::accept_hash`].
    /// When non-zero the sampler runs a keyed remix of the caller-
    /// supplied `flow_hash` before the modulo decision — makes the
    /// admission boundary unpredictable to an attacker who can
    /// observe or influence their own `flow_hash` value but cannot
    /// learn the sampler secret. Zero-init means "no remix", matches
    /// the historical deterministic behaviour of [`Self::new`].
    mix_k1: u64,
    /// Second secret — XOR'd at the end of the mix to avoid the
    /// multiply by `mix_k1` alone (a structure the attacker could
    /// invert given enough observations).
    mix_k2: u64,
    /// Observability sink — emitted every [`METRICS_BATCH_SIZE`]
    /// hot-path calls (in-process atomic counters stay bit-exact
    /// every call). Defaults to [`anomstream_core::NoopSink`].
    metrics: Arc<dyn MetricsSink>,
}

impl UpdateSampler {
    /// Build a sampler keeping 1 offer out of every `keep_every_n`.
    /// `0` and `1` disable sampling (every offer accepted).
    ///
    /// **`accept_hash` admission is deterministic without a
    /// per-sampler secret** — an attacker who can probe the
    /// admission decision on a known `flow_hash` can spray
    /// 5-tuples whose hash lands on the admitted residue class
    /// and poison the reservoir. For internet-facing ingress,
    /// prefer [`Self::new_keyed`] which seeds the secret from
    /// `getrandom` at construction.
    #[must_use]
    pub fn new(keep_every_n: u32) -> Self {
        Self {
            keep_every_n,
            counter: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            accepted_flushed: AtomicU64::new(0),
            rejected_flushed: AtomicU64::new(0),
            mix_k1: 0,
            mix_k2: 0,
            metrics: default_sink(),
        }
    }

    /// Keyed variant — same ratio semantics as [`Self::new`] but
    /// with a per-sampler secret mix applied to every
    /// [`Self::accept_hash`] input. Defeats the deterministic-
    /// admission poisoning vector (MITRE ATLAS `AML.T0020`): the
    /// attacker can't steer their own flow hash into the admitted
    /// residue class without knowing the per-sampler mix keys,
    /// which are seeded from the OS CSPRNG at construction and
    /// never leave the process.
    ///
    /// # Errors
    ///
    /// Propagates [`getrandom::Error`] when the OS entropy source
    /// is unavailable (embedded / chroot without `/dev/urandom`).
    ///
    /// # Panics
    ///
    /// Never in practice — the two `try_into().expect(...)` calls
    /// unwrap a compile-time known 8-byte slice taken from a
    /// 16-byte buffer.
    pub fn new_keyed(keep_every_n: u32) -> Result<Self, getrandom::Error> {
        let mut buf = [0_u8; 16];
        getrandom::fill(&mut buf)?;
        let mix_k1 = u64::from_le_bytes(buf[0..8].try_into().expect("16 bytes"));
        let mix_k2 = u64::from_le_bytes(buf[8..16].try_into().expect("16 bytes"));
        Ok(Self::new_keyed_with_seeds(keep_every_n, mix_k1, mix_k2))
    }

    /// Caller-supplied-seed variant of [`Self::new_keyed`] — for
    /// restricted environments where `getrandom` is unavailable
    /// (early-boot embedded, chroot without `/dev/urandom`,
    /// `wasm32-unknown-unknown` without a JS host) **or** for
    /// reproducible / snapshot-replayable test fixtures.
    ///
    /// `k1` is forced odd via `| 1` so it remains a valid
    /// multiplicative-bijection modulus inside [`Self::keyed_mix`];
    /// `k2` is XOR'd at the end of the mix unchanged. Passing
    /// `(0, 0)` would degrade `mix_k1` to `1` and `mix_k2` to `0` —
    /// still keyed (the murmur finaliser still runs) but with a
    /// publicly-known seed, so the per-sampler-secret defence
    /// against `AML.T0020` poisoning sprays goes away. **Do not**
    /// use the all-zero seed in production; in that case prefer
    /// [`Self::new`] which advertises its deterministic admission
    /// behaviour explicitly.
    ///
    /// # Production seed sourcing
    ///
    /// When `getrandom` is unavailable, derive `k1` / `k2` from a
    /// local entropy source the deployment trusts: a
    /// configuration-file secret backed by a KMS, an HSM-derived
    /// session key, or boot-time RDRAND. Anything is acceptable as
    /// long as the attacker who can probe the admission decision
    /// cannot recover the seed.
    #[must_use]
    pub fn new_keyed_with_seeds(keep_every_n: u32, k1: u64, k2: u64) -> Self {
        // `| 1` forces odd + non-zero so `mix_k1` is always a
        // valid multiplicative-bijection modulus. A caller-passed
        // even `k1` (or `0`) silently rounds up to the next odd
        // value rather than degenerating the keyed mix.
        let mix_k1 = k1 | 1;
        Self {
            keep_every_n,
            counter: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            accepted_flushed: AtomicU64::new(0),
            rejected_flushed: AtomicU64::new(0),
            mix_k1,
            mix_k2: k2,
            metrics: default_sink(),
        }
    }

    /// Install a metrics sink — every `accept_*` call emits an
    /// accepted/rejected counter through it. Chain-style builder.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink.
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    /// Whether this sampler was built with a keyed mix.
    #[must_use]
    pub fn is_keyed(&self) -> bool {
        self.mix_k1 != 0
    }

    /// Configured ratio denominator.
    #[must_use]
    pub fn keep_every_n(&self) -> u32 {
        self.keep_every_n
    }

    /// Stride-based decision — every `keep_every_n`-th offer lands.
    /// Call order-dependent: counter increments on every invocation.
    /// Cheap (one atomic fetch-add) but not flow-aware.
    pub fn accept_stride(&self) -> bool {
        if self.keep_every_n <= 1 {
            record_batched(
                &self.accepted,
                &self.accepted_flushed,
                &self.metrics,
                names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL,
            );
            return true;
        }
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let keep = u64::from(self.keep_every_n);
        let ok = n.is_multiple_of(keep);
        if ok {
            record_batched(
                &self.accepted,
                &self.accepted_flushed,
                &self.metrics,
                names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL,
            );
        } else {
            record_batched(
                &self.rejected,
                &self.rejected_flushed,
                &self.metrics,
                names::HOT_PATH_SAMPLER_REJECTED_TOTAL,
            );
        }
        ok
    }

    /// Per-flow decision — flows with
    /// `keyed_mix(flow_hash) % keep_every_n == 0` are admitted,
    /// every other flow rejected in full. Deterministic across the
    /// sampler's lifetime **for that sampler**: the same flow
    /// always samples the same way, so the baseline keeps
    /// representative coverage of every sampled flow rather than
    /// slicing any single flow.
    ///
    /// The caller supplies `flow_hash` — typically a 64-bit mix of
    /// 5-tuple bytes (`SipHash` / `FxHash` / custom). Quality of
    /// sampling only matters modulo `keep_every_n`.
    ///
    /// When the sampler was built via [`Self::new_keyed`] a
    /// per-sampler secret mix (murmur-style finaliser keyed on
    /// `mix_k1` / `mix_k2`) is applied **before** the modulo so the
    /// admission residue class is unpredictable without the secret
    /// (defends against `AML.T0020` reservoir-poisoning sprays).
    pub fn accept_hash(&self, flow_hash: u64) -> bool {
        if self.keep_every_n <= 1 {
            record_batched(
                &self.accepted,
                &self.accepted_flushed,
                &self.metrics,
                names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL,
            );
            return true;
        }
        let mixed = self.keyed_mix(flow_hash);
        let ok = mixed.is_multiple_of(u64::from(self.keep_every_n));
        if ok {
            record_batched(
                &self.accepted,
                &self.accepted_flushed,
                &self.metrics,
                names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL,
            );
        } else {
            record_batched(
                &self.rejected,
                &self.rejected_flushed,
                &self.metrics,
                names::HOT_PATH_SAMPLER_REJECTED_TOTAL,
            );
        }
        ok
    }

    /// Murmur3-64 finaliser keyed by the sampler secret. Returns
    /// `h` unchanged when the sampler was built unkeyed (via
    /// [`Self::new`]) so the legacy `accept_hash` behaviour is
    /// preserved bit-for-bit.
    #[inline]
    fn keyed_mix(&self, h: u64) -> u64 {
        if self.mix_k1 == 0 {
            return h;
        }
        let mut x = h.wrapping_add(self.mix_k1);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        x ^= x >> 33;
        x ^ self.mix_k2
    }

    /// Running total of accepted offers since construction.
    /// Bit-exact even when the [`MetricsSink`] view lags by up to
    /// [`METRICS_BATCH_SIZE`] increments.
    #[must_use]
    pub fn accepted_total(&self) -> u64 {
        self.accepted.load(Ordering::Relaxed)
    }

    /// Running total of rejected offers since construction.
    /// Bit-exact independent of sink-side batching.
    #[must_use]
    pub fn rejected_total(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }

    /// Drain the residue (≤ [`METRICS_BATCH_SIZE`] − 1 increments
    /// per counter) that the batched fast paths have not yet
    /// emitted to the [`MetricsSink`]. Idempotent — call before
    /// process shutdown or before exporting a metrics snapshot
    /// the operator wants matched against [`Self::accepted_total`]
    /// / [`Self::rejected_total`].
    pub fn flush_metrics(&self) {
        flush_batched(
            &self.accepted,
            &self.accepted_flushed,
            &self.metrics,
            names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL,
        );
        flush_batched(
            &self.rejected,
            &self.rejected_flushed,
            &self.metrics,
            names::HOT_PATH_SAMPLER_REJECTED_TOTAL,
        );
    }
}

/// Producer end of the hot-path update queue. Multi-producer: clone
/// this handle per classifier thread. `try_enqueue` is non-blocking
/// — on queue full it returns `false` and increments `dropped_total`.
#[derive(Debug)]
#[non_exhaustive]
pub struct UpdateProducer<const D: usize> {
    /// Underlying bounded MPSC sender.
    tx: SyncSender<[f64; D]>,
    /// Capacity the channel was built with — surfaced for gauges.
    capacity: usize,
    /// Lifetime enqueued count.
    enqueued: Arc<AtomicU64>,
    /// Lifetime dropped-on-full count.
    dropped: Arc<AtomicU64>,
    /// Sink-side cumulative emitted count for `enqueued`.
    enqueued_flushed: Arc<AtomicU64>,
    /// Sink-side cumulative emitted count for `dropped`.
    dropped_flushed: Arc<AtomicU64>,
    /// Observability sink — shared with every clone of this
    /// producer so every classifier thread emits to the same
    /// endpoint. Emitted every [`METRICS_BATCH_SIZE`] hot-path
    /// calls (in-process atomic counters stay bit-exact every
    /// call).
    metrics: Arc<dyn MetricsSink>,
}

impl<const D: usize> Clone for UpdateProducer<D> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            capacity: self.capacity,
            enqueued: Arc::clone(&self.enqueued),
            dropped: Arc::clone(&self.dropped),
            enqueued_flushed: Arc::clone(&self.enqueued_flushed),
            dropped_flushed: Arc::clone(&self.dropped_flushed),
            metrics: Arc::clone(&self.metrics),
        }
    }
}

impl<const D: usize> UpdateProducer<D> {
    /// Non-blocking enqueue. Returns `true` when the point landed in
    /// the queue, `false` when the queue was full (point dropped,
    /// `dropped_total` incremented).
    #[must_use]
    pub fn try_enqueue(&self, point: [f64; D]) -> bool {
        if self.tx.try_send(point).is_ok() {
            record_batched(
                &self.enqueued,
                &self.enqueued_flushed,
                &self.metrics,
                names::HOT_PATH_QUEUE_ENQUEUED_TOTAL,
            );
            true
        } else {
            record_batched(
                &self.dropped,
                &self.dropped_flushed,
                &self.metrics,
                names::HOT_PATH_QUEUE_DROPPED_TOTAL,
            );
            false
        }
    }

    /// Drain the residue (≤ [`METRICS_BATCH_SIZE`] − 1 increments)
    /// that the batched [`Self::try_enqueue`] path has not yet
    /// emitted to the [`MetricsSink`]. Idempotent — call before
    /// shutdown / metrics export.
    pub fn flush_metrics(&self) {
        flush_batched(
            &self.enqueued,
            &self.enqueued_flushed,
            &self.metrics,
            names::HOT_PATH_QUEUE_ENQUEUED_TOTAL,
        );
        flush_batched(
            &self.dropped,
            &self.dropped_flushed,
            &self.metrics,
            names::HOT_PATH_QUEUE_DROPPED_TOTAL,
        );
    }

    /// Read-only handle to the installed sink.
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    /// Channel capacity as configured at [`update_channel`].
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Lifetime count of successfully enqueued points.
    #[must_use]
    pub fn enqueued_total(&self) -> u64 {
        self.enqueued.load(Ordering::Relaxed)
    }

    /// Lifetime count of points dropped because the queue was full.
    /// Non-zero indicates classifier thread is producing faster than
    /// the updater thread is draining — raise alert / widen the
    /// channel / raise the sampler ratio.
    #[must_use]
    pub fn dropped_total(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// Consumer end of the hot-path update queue. Single-consumer: hand
/// this handle to the dedicated updater thread.
#[derive(Debug)]
#[non_exhaustive]
pub struct UpdateConsumer<const D: usize> {
    /// Underlying MPSC receiver.
    rx: Receiver<[f64; D]>,
}

impl<const D: usize> UpdateConsumer<D> {
    /// Drain every point currently queued into `sink`. Returns
    /// `(ingested, errors)` — number of successful sink calls and
    /// number of errored sink calls (typically forest reservoir
    /// errors). The method returns as soon as the queue is empty so
    /// the updater thread can back off or gauge throughput.
    pub fn try_drain<F, E>(&self, mut sink: F) -> (usize, usize)
    where
        F: FnMut([f64; D]) -> Result<(), E>,
    {
        let mut ingested = 0;
        let mut errors = 0;
        while let Ok(p) = self.rx.try_recv() {
            if sink(p).is_ok() {
                ingested += 1;
            } else {
                errors += 1;
            }
        }
        (ingested, errors)
    }

    /// Blocking-take of the next point. Returns `None` when every
    /// [`UpdateProducer`] has been dropped (clean shutdown).
    #[must_use]
    pub fn recv(&self) -> Option<[f64; D]> {
        self.rx.recv().ok()
    }
}

/// Build a bounded MPSC hot-path update channel with the requested
/// capacity. Returns `(producer, consumer)` — clone the producer per
/// classifier thread; hand the consumer to the dedicated updater
/// thread.
///
/// `capacity` is the in-flight queue depth; sizing it ~1 second's
/// worth of expected update rate keeps the updater thread busy
/// without unbounded backpressure under micro-bursts. Drop events
/// (observable via [`UpdateProducer::dropped_total`]) are the ops
/// signal the channel needs widening.
///
/// # Panics
///
/// Panics when `capacity == 0` (would silently drop every offer)
/// or when `capacity > MAX_CHANNEL_CAPACITY` (caller-controlled
/// OOM at construction). Use [`try_update_channel`] for the
/// non-panicking variant that surfaces the same condition as a
/// recoverable [`RcfError`].
#[must_use]
#[allow(clippy::panic, clippy::missing_panics_doc)]
pub fn update_channel<const D: usize>(capacity: usize) -> (UpdateProducer<D>, UpdateConsumer<D>) {
    // Documented panic — callers wanting a `Result` use
    // `try_update_channel` instead.
    try_update_channel(capacity).unwrap_or_else(|e| panic!("update_channel: {e}"))
}

/// Same as [`update_channel`] but with a caller-supplied metrics
/// sink. Every clone of the returned producer shares the same
/// sink.
///
/// # Panics
///
/// Same conditions as [`update_channel`].
#[must_use]
#[allow(clippy::panic, clippy::missing_panics_doc)]
pub fn update_channel_with_sink<const D: usize>(
    capacity: usize,
    sink: Arc<dyn MetricsSink>,
) -> (UpdateProducer<D>, UpdateConsumer<D>) {
    try_update_channel_with_sink(capacity, sink)
        .unwrap_or_else(|e| panic!("update_channel_with_sink: {e}"))
}

/// Fallible variant of [`update_channel`] — returns
/// [`RcfError::InvalidConfig`] instead of panicking when
/// `capacity` is `0` or above [`MAX_CHANNEL_CAPACITY`].
///
/// # Errors
///
/// Returns [`RcfError::InvalidConfig`] when `capacity == 0` or
/// `capacity > MAX_CHANNEL_CAPACITY`.
pub fn try_update_channel<const D: usize>(
    capacity: usize,
) -> RcfResult<(UpdateProducer<D>, UpdateConsumer<D>)> {
    try_update_channel_with_sink(capacity, default_sink())
}

/// Fallible variant of [`update_channel_with_sink`].
///
/// # Errors
///
/// Same as [`try_update_channel`].
pub fn try_update_channel_with_sink<const D: usize>(
    capacity: usize,
    sink: Arc<dyn MetricsSink>,
) -> RcfResult<(UpdateProducer<D>, UpdateConsumer<D>)> {
    if capacity == 0 {
        return Err(RcfError::InvalidConfig(
            "update_channel: capacity must be > 0 (zero capacity drops every offer)".into(),
        ));
    }
    if capacity > MAX_CHANNEL_CAPACITY {
        return Err(RcfError::InvalidConfig(
            format!(
                "update_channel: capacity {capacity} exceeds MAX_CHANNEL_CAPACITY {MAX_CHANNEL_CAPACITY} (caller-controlled OOM guard)"
            )
            .into(),
        ));
    }
    let (tx, rx) = sync_channel::<[f64; D]>(capacity);
    let enqueued = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicU64::new(0));
    let enqueued_flushed = Arc::new(AtomicU64::new(0));
    let dropped_flushed = Arc::new(AtomicU64::new(0));
    Ok((
        UpdateProducer {
            tx,
            capacity,
            enqueued,
            dropped,
            enqueued_flushed,
            dropped_flushed,
            metrics: sink,
        },
        UpdateConsumer { rx },
    ))
}

/// Cache-line-padded `AtomicU32` — each bucket lives on its own
/// 64-byte cache line so concurrent `fetch_add`s on different
/// buckets do not bounce a shared line through the coherence
/// protocol. Footprint trade: 256 buckets × 64 B = 16 KiB
/// (vs 1 KiB unpadded). Worth it on the hot path where false
/// sharing measurably tanks multi-core throughput; the bench
/// `hot_path_prefix_cap/check_and_record_contended_8threads`
/// quantifies the gain.
#[repr(C, align(64))]
#[derive(Debug)]
struct PaddedBucket {
    /// Live counter. Surrounded by `align(64)` so adjacent
    /// buckets land on distinct cache lines.
    inner: AtomicU32,
}

/// Fixed-bucket per-prefix rate cap — bounds how many admissions a
/// single source prefix can push into the reservoir within a
/// rolling time window. Defends against the reservoir-poisoning
/// spray where an attacker floods the ingress from one IP prefix
/// hoping a fraction land in the reservoir via
/// [`UpdateSampler::accept_hash`].
///
/// Implementation: 256 atomic `u32` buckets indexed by
/// `prefix_hash & 0xff`. Collisions are soft — the cap bounds
/// across the *bucket*, not the exact prefix. This trades a small
/// amount of cross-prefix interference for O(1) lock-free
/// check-and-record with bounded memory.
///
/// # Soft over-admission window
///
/// [`Self::check_and_record`] is **lock-free**, not strictly
/// transactional: the bucket increment (`fetch_add`) and the
/// post-increment compare against [`Self::cap_per_window`] are
/// distinct atomics, so two concurrent admissions can both
/// increment past the cap before either notices and rolls back.
/// Each thread that lands on a bucket already at cap immediately
/// `fetch_sub`s its own increment to avoid permanent leakage,
/// but the brief over-admission window stays observable on
/// multi-core load. The empirical bound under stress (see
/// `prefix_rate_cap_concurrent_rollover_holds_hard_cap`) is
/// roughly `4 × cap_per_window` per bucket per window — design
/// the cap with that slack baked in (e.g. set the operator-facing
/// "max 25 admits / window" by passing `cap_per_window = 6`).
///
/// # Example
///
/// ```ignore
/// use core::num::{NonZeroU32, NonZeroU64};
/// use anomstream_hotpath::PrefixRateCap;
///
/// let cap = PrefixRateCap::new(
///     NonZeroU32::new(100).unwrap(),
///     NonZeroU64::new(1_000).unwrap(),
/// );
/// let now_ms = /* wall-clock */;
/// if cap.check_and_record(flow_prefix_hash, now_ms) {
///     forest.update(point)?;
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct PrefixRateCap {
    /// Per-bucket admit counter, cache-line padded to defeat
    /// false sharing. 256 buckets × 64 B = 16 KiB total.
    buckets: [PaddedBucket; Self::BUCKETS],
    /// Epoch-millisecond timestamp at which the current window
    /// opened. The next `check_and_record` past `+ window_ms`
    /// atomically resets the buckets.
    window_start_ms: AtomicU64,
    /// Window length. Cap counts reset every `window_ms`.
    window_ms: u64,
    /// Maximum admits per bucket per window. `0` means cap
    /// disabled — every call admits (set by [`Self::disabled`]).
    cap_per_window: u32,
    /// Lifetime count of admits that hit the cap and were rejected.
    capped_total: AtomicU64,
    /// Lifetime count of admits passed through.
    admitted_total: AtomicU64,
    /// Sink-side cumulative emitted count for `admitted`.
    admitted_flushed: AtomicU64,
    /// Sink-side cumulative emitted count for `capped`.
    capped_flushed: AtomicU64,
    /// Observability sink — emitted every [`METRICS_BATCH_SIZE`]
    /// hot-path calls (in-process atomic counters stay bit-exact
    /// every call).
    metrics: Arc<dyn MetricsSink>,
}

impl PrefixRateCap {
    /// Number of buckets. Compile-time constant; 256 buckets ×
    /// 64 B (cache-padded `AtomicU32`) = 16 KiB on 64-bit.
    pub const BUCKETS: usize = 256;

    /// Build a rate cap. Both arguments are typed `NonZero` so
    /// the previous footgun pair (`window_ms == 0` panics,
    /// `cap_per_window == 0` silently disabled the cap) is
    /// impossible to express. Use [`Self::disabled`] when the
    /// caller wants the always-admit mode explicitly.
    #[must_use]
    pub fn new(cap_per_window: NonZeroU32, window_ms: NonZeroU64) -> Self {
        Self::build(cap_per_window.get(), window_ms.get())
    }

    /// Always-admit mode — every [`Self::check_and_record`] call
    /// returns `true` and increments [`Self::admitted_total`].
    /// `window_ms` still has to be non-zero (kept on the type for
    /// future-proofing — a re-enable path could repurpose it).
    #[must_use]
    pub fn disabled(window_ms: NonZeroU64) -> Self {
        Self::build(0, window_ms.get())
    }

    /// Shared constructor — bypassed by [`Self::new`] /
    /// [`Self::disabled`] which guarantee the typed invariants.
    fn build(cap_per_window: u32, window_ms: u64) -> Self {
        // Cannot use `[PaddedBucket { ... }; 256]` because the
        // inner AtomicU32 is !Copy. Build via closure.
        let buckets: [PaddedBucket; Self::BUCKETS] = core::array::from_fn(|_| PaddedBucket {
            inner: AtomicU32::new(0),
        });
        Self {
            buckets,
            window_start_ms: AtomicU64::new(0),
            window_ms,
            cap_per_window,
            capped_total: AtomicU64::new(0),
            admitted_total: AtomicU64::new(0),
            admitted_flushed: AtomicU64::new(0),
            capped_flushed: AtomicU64::new(0),
            metrics: default_sink(),
        }
    }

    /// Install a metrics sink — every `check_and_record` call emits
    /// an admitted/capped counter through it.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink.
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    /// Record an admission attempt and return `true` when the
    /// caller is allowed to proceed. Thread-safe, lock-free.
    ///
    /// Under concurrent load the window-rollover path runs a
    /// `compare_exchange_weak` loop until the timestamp either
    /// already sits inside the live window or this thread wins
    /// the reset. `Release` / `Acquire` ordering makes the bucket
    /// zero-fill happen-before any subsequent bucket `fetch_add`.
    ///
    /// **Soft over-admission window** — see the type-level docs.
    /// The `fetch_add` + cap-comparison sequence is two distinct
    /// atomics, so concurrent admissions can briefly exceed
    /// [`Self::cap_per_window`] before each thread that loses
    /// the race rolls its own increment back. Empirical bound is
    /// ≈ `4 × cap_per_window` per bucket per window under
    /// stress. Operators sizing the cap for a hard ceiling should
    /// configure `cap_per_window = ceiling / 4` (or shard the
    /// admission across multiple [`PrefixRateCap`] instances).
    pub fn check_and_record(&self, prefix_hash: u64, now_ms: u64) -> bool {
        if self.cap_per_window == 0 {
            record_batched(
                &self.admitted_total,
                &self.admitted_flushed,
                &self.metrics,
                names::HOT_PATH_PREFIX_ADMITTED_TOTAL,
            );
            return true;
        }
        // Atomically roll the window if needed. Loop until the
        // observed `start` is either still valid or we successfully
        // install `now_ms` via CAS.
        loop {
            let start = self.window_start_ms.load(Ordering::Acquire);
            if start != 0 && now_ms.saturating_sub(start) < self.window_ms {
                break;
            }
            // Peer-move on `Err` — just loop; `Ok` wins the reset.
            // Bounded by thread count; no livelock.
            if self
                .window_start_ms
                .compare_exchange_weak(start, now_ms, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Winner — zero-fill the bucket bank. The
                // `Release` side of the successful CAS makes this
                // store visible to every peer whose subsequent
                // `fetch_add` acquires the updated
                // `window_start_ms`.
                for bucket in &self.buckets {
                    bucket.inner.store(0, Ordering::Relaxed);
                }
                break;
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = ((prefix_hash & 0xff) as usize) & (Self::BUCKETS - 1);
        let prior = self.buckets[idx].inner.fetch_add(1, Ordering::Relaxed);
        if prior < self.cap_per_window {
            record_batched(
                &self.admitted_total,
                &self.admitted_flushed,
                &self.metrics,
                names::HOT_PATH_PREFIX_ADMITTED_TOTAL,
            );
            true
        } else {
            // Already over — roll back the increment so a late
            // window roll doesn't accumulate forever on this
            // bucket.
            self.buckets[idx].inner.fetch_sub(1, Ordering::Relaxed);
            record_batched(
                &self.capped_total,
                &self.capped_flushed,
                &self.metrics,
                names::HOT_PATH_PREFIX_CAPPED_TOTAL,
            );
            false
        }
    }

    /// Drain the batched-metrics residue (≤ [`METRICS_BATCH_SIZE`] − 1
    /// increments per counter). Idempotent — call before shutdown
    /// or before exporting a metrics snapshot. In-process
    /// [`Self::admitted_total`] / [`Self::capped_total`] stay
    /// bit-exact independent of this call.
    pub fn flush_metrics(&self) {
        flush_batched(
            &self.admitted_total,
            &self.admitted_flushed,
            &self.metrics,
            names::HOT_PATH_PREFIX_ADMITTED_TOTAL,
        );
        flush_batched(
            &self.capped_total,
            &self.capped_flushed,
            &self.metrics,
            names::HOT_PATH_PREFIX_CAPPED_TOTAL,
        );
    }

    /// Lifetime admits that passed the cap.
    #[must_use]
    pub fn admitted_total(&self) -> u64 {
        self.admitted_total.load(Ordering::Relaxed)
    }

    /// Lifetime admits rejected because the bucket was at cap.
    #[must_use]
    pub fn capped_total(&self) -> u64 {
        self.capped_total.load(Ordering::Relaxed)
    }

    /// Window length in milliseconds.
    #[must_use]
    pub fn window_ms(&self) -> u64 {
        self.window_ms
    }

    /// Cap per bucket per window.
    #[must_use]
    pub fn cap_per_window(&self) -> u32 {
        self.cap_per_window
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn sampler_disabled_accepts_every_offer() {
        let s = UpdateSampler::new(0);
        for _ in 0..10 {
            assert!(s.accept_stride());
        }
        assert_eq!(s.accepted_total(), 10);
        assert_eq!(s.rejected_total(), 0);
    }

    #[test]
    fn sampler_one_accepts_every_offer() {
        let s = UpdateSampler::new(1);
        for _ in 0..10 {
            assert!(s.accept_stride());
            assert!(s.accept_hash(0xdead_beef_cafe_babe));
        }
    }

    #[test]
    fn sampler_stride_keeps_one_in_n() {
        let s = UpdateSampler::new(4);
        let mut accepted = 0_usize;
        for _ in 0..100 {
            if s.accept_stride() {
                accepted += 1;
            }
        }
        assert_eq!(accepted, 25);
        assert_eq!(s.accepted_total(), 25);
        assert_eq!(s.rejected_total(), 75);
    }

    #[test]
    fn sampler_hash_deterministic_per_flow() {
        let s = UpdateSampler::new(4);
        // Same hash always decides the same way.
        let h = 0xdead_beef_cafe_babe_u64;
        let d1 = s.accept_hash(h);
        let d2 = s.accept_hash(h);
        assert_eq!(d1, d2);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn channel_zero_capacity_panics() {
        let _ = update_channel::<2>(0);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_CHANNEL_CAPACITY")]
    fn channel_oversize_capacity_panics() {
        let _ = update_channel::<2>(MAX_CHANNEL_CAPACITY + 1);
    }

    #[test]
    fn try_update_channel_rejects_invalid_capacity() {
        assert!(try_update_channel::<2>(0).is_err());
        assert!(try_update_channel::<2>(MAX_CHANNEL_CAPACITY + 1).is_err());
        assert!(try_update_channel::<2>(MAX_CHANNEL_CAPACITY).is_ok());
        assert!(try_update_channel::<2>(1).is_ok());
    }

    #[test]
    fn channel_try_enqueue_drops_on_full() {
        let (p, _c) = update_channel::<2>(2);
        assert!(p.try_enqueue([1.0, 2.0]));
        assert!(p.try_enqueue([3.0, 4.0]));
        assert!(!p.try_enqueue([5.0, 6.0]));
        assert_eq!(p.enqueued_total(), 2);
        assert_eq!(p.dropped_total(), 1);
    }

    #[test]
    fn channel_try_drain_empties_queue() {
        let (p, c) = update_channel::<2>(8);
        let _ = p.try_enqueue([1.0, 2.0]);
        let _ = p.try_enqueue([3.0, 4.0]);
        let mut sink: Vec<[f64; 2]> = Vec::new();
        let (ing, err) = c.try_drain::<_, ()>(|pt| {
            sink.push(pt);
            Ok(())
        });
        assert_eq!(ing, 2);
        assert_eq!(err, 0);
        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn channel_producer_is_clone_multi_producer() {
        let (p1, c) = update_channel::<2>(8);
        let p2 = p1.clone();
        let _ = p1.try_enqueue([1.0, 2.0]);
        let _ = p2.try_enqueue([3.0, 4.0]);
        // Counters are shared across clones.
        assert_eq!(p1.enqueued_total(), 2);
        assert_eq!(p2.enqueued_total(), 2);
        let (ing, _) = c.try_drain::<_, ()>(|_| Ok(()));
        assert_eq!(ing, 2);
    }

    #[test]
    fn channel_try_drain_counts_errors() {
        let (p, c) = update_channel::<2>(8);
        let _ = p.try_enqueue([1.0, 2.0]);
        let _ = p.try_enqueue([3.0, 4.0]);
        let (ing, err) =
            c.try_drain::<_, &'static str>(|pt| if pt[0] > 2.0 { Err("nope") } else { Ok(()) });
        assert_eq!(ing, 1);
        assert_eq!(err, 1);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn end_to_end_hot_path_wires_sampler_producer_consumer() {
        use anomstream_core::ForestBuilder;

        let mut forest = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(64)
            .seed(2026)
            .build()
            .unwrap();

        let sampler = UpdateSampler::new(3);
        let (producer, consumer) = update_channel::<2>(16);

        // Simulated classifier hot-path.
        for i in 0..9_u64 {
            if sampler.accept_hash(i) {
                let _ = producer.try_enqueue([i as f64, (i * 2) as f64]);
            }
        }
        drop(producer);

        // Updater thread.
        let (ing, err) = consumer.try_drain(|p| forest.update(p));
        assert!(err == 0);
        // 0, 3, 6 accepted by accept_hash(keep_every_n=3): 3 points.
        assert_eq!(ing, 3);
    }

    /// Counts every `inc_counter` call by metric name. Lets the
    /// batching tests assert the sink saw exactly `actual /
    /// METRICS_BATCH_SIZE` flushes, not one per hot-path call.
    #[derive(Debug, Default)]
    struct CountingSink {
        calls: std::sync::Mutex<std::collections::HashMap<String, u64>>,
        units: std::sync::Mutex<std::collections::HashMap<String, u64>>,
    }

    impl MetricsSink for CountingSink {
        fn inc_counter(&self, name: &str, by: u64) {
            *self
                .calls
                .lock()
                .unwrap()
                .entry(name.to_owned())
                .or_default() += 1;
            *self
                .units
                .lock()
                .unwrap()
                .entry(name.to_owned())
                .or_default() += by;
        }
        fn set_gauge(&self, _: &str, _: f64) {}
        fn observe_histogram(&self, _: &str, _: f64) {}
    }

    #[test]
    fn sampler_metrics_emit_in_batches() {
        let sink = Arc::new(CountingSink::default());
        let s = UpdateSampler::new(0).with_metrics_sink(Arc::clone(&sink) as Arc<dyn MetricsSink>);
        // 200 ops, batch = 64 → expect 3 batched emissions
        // (at counts 64, 128, 192). Residue 8 stays in
        // accepted_total but not in sink until flush_metrics.
        for _ in 0..200 {
            assert!(s.accept_stride());
        }
        let calls = sink.calls.lock().unwrap();
        let units = sink.units.lock().unwrap();
        assert_eq!(
            *calls.get(names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL).unwrap(),
            3,
            "expected 3 batched emissions, got {calls:?}"
        );
        assert_eq!(
            *units.get(names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL).unwrap(),
            192
        );
        assert_eq!(s.accepted_total(), 200, "in-process counter must be exact");
        drop(calls);
        drop(units);
        // Flush drains the residue.
        s.flush_metrics();
        let units = sink.units.lock().unwrap();
        assert_eq!(
            *units.get(names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL).unwrap(),
            200,
            "flush_metrics must drain the residue"
        );
    }

    #[test]
    fn producer_metrics_emit_in_batches_and_flush_drains() {
        let sink = Arc::new(CountingSink::default());
        let (p, _c) =
            update_channel_with_sink::<2>(1024, Arc::clone(&sink) as Arc<dyn MetricsSink>);
        for _ in 0..130_u32 {
            assert!(p.try_enqueue([0.0, 0.0]));
        }
        let units_pre = sink
            .units
            .lock()
            .unwrap()
            .get(names::HOT_PATH_QUEUE_ENQUEUED_TOTAL)
            .copied()
            .unwrap_or(0);
        // 130 ops → 2 batched emissions (64, 128) → 128 units.
        assert_eq!(units_pre, 128);
        assert_eq!(p.enqueued_total(), 130);
        p.flush_metrics();
        let units_post = sink
            .units
            .lock()
            .unwrap()
            .get(names::HOT_PATH_QUEUE_ENQUEUED_TOTAL)
            .copied()
            .unwrap_or(0);
        assert_eq!(units_post, 130);
        // Idempotent flush.
        p.flush_metrics();
        let units_again = sink
            .units
            .lock()
            .unwrap()
            .get(names::HOT_PATH_QUEUE_ENQUEUED_TOTAL)
            .copied()
            .unwrap_or(0);
        assert_eq!(units_again, 130);
    }

    #[test]
    fn prefix_cap_metrics_emit_in_batches() {
        let sink = Arc::new(CountingSink::default());
        let cap = PrefixRateCap::new(nz_u32(1_000), nz_u64(60_000))
            .with_metrics_sink(Arc::clone(&sink) as Arc<dyn MetricsSink>);
        for i in 0..200_u64 {
            assert!(cap.check_and_record(i, 0));
        }
        let units = sink
            .units
            .lock()
            .unwrap()
            .get(names::HOT_PATH_PREFIX_ADMITTED_TOTAL)
            .copied()
            .unwrap_or(0);
        assert_eq!(units, 192, "200 ops, batch 64 → 192 units pre-flush");
        cap.flush_metrics();
        let units_post = sink
            .units
            .lock()
            .unwrap()
            .get(names::HOT_PATH_PREFIX_ADMITTED_TOTAL)
            .copied()
            .unwrap_or(0);
        assert_eq!(units_post, 200);
    }

    #[test]
    fn keyed_sampler_admission_differs_from_unkeyed() {
        // The unkeyed sampler admits h == 0 mod 4.
        let unkeyed = UpdateSampler::new(4);
        assert!(unkeyed.accept_hash(0));
        assert!(unkeyed.accept_hash(4));
        assert!(!unkeyed.accept_hash(1));
        assert!(!unkeyed.accept_hash(2));

        // A keyed sampler shifts the residue class: the same
        // hashes land on a different admission set. Use fixed seeds
        // so the assertion is deterministic — a randomly-keyed
        // sampler has a ~1/2^N chance of reproducing the unkeyed
        // decisions on N hashes, which would flake CI.
        let keyed = UpdateSampler::new_keyed_with_seeds(4, 0xdead_beef, 0xcafe_f00d);
        assert!(keyed.is_keyed());
        let diff_decision = (0..32_u64)
            .filter(|h| unkeyed.accept_hash(*h) != keyed.accept_hash(*h))
            .count();
        // The mix is a random oracle, not a uniform shuffle; the
        // point of the assertion is only that the keyed sampler is
        // *not* the unkeyed one. These seeds diverge on 14 of 32
        // hashes.
        assert!(
            diff_decision > 0,
            "keyed sampler accepted every hash exactly like unkeyed — mix ineffective"
        );

        // Smoke-check that the getrandom-seeded constructor builds a
        // keyed sampler too (entropy path, not admission semantics).
        assert!(
            UpdateSampler::new_keyed(4)
                .expect("getrandom works")
                .is_keyed()
        );
    }

    #[test]
    fn new_keyed_with_seeds_reproducible() {
        // Same explicit seeds → same admission decisions across
        // independently-built samplers. Required for snapshot /
        // replay test fixtures and for restricted environments
        // that cannot call getrandom.
        let s1 = UpdateSampler::new_keyed_with_seeds(4, 0xdead_beef, 0xcafe_f00d);
        let s2 = UpdateSampler::new_keyed_with_seeds(4, 0xdead_beef, 0xcafe_f00d);
        for h in 0..32_u64 {
            assert_eq!(s1.accept_hash(h), s2.accept_hash(h));
        }
        assert!(s1.is_keyed());
    }

    #[test]
    fn new_keyed_with_seeds_different_seeds_diverge() {
        // Distinct seed pairs must produce distinct admission
        // residue classes — the whole point of per-sampler
        // rotation against AML.T0020.
        let s_a = UpdateSampler::new_keyed_with_seeds(4, 0x1111, 0x2222);
        let s_b = UpdateSampler::new_keyed_with_seeds(4, 0x3333, 0x4444);
        let same = (0..32_u64)
            .filter(|h| s_a.accept_hash(*h) == s_b.accept_hash(*h))
            .count();
        // 32 hashes, 50% match probability per hash under random
        // mixes → P(all match) = 2⁻³² ≈ 2e-10. Test that flakes
        // would be observable across the universe's lifetime.
        assert!(
            same < 32,
            "two distinct seed pairs produced identical admission for all 32 probes"
        );
    }

    #[test]
    fn new_keyed_with_seeds_forces_odd_k1() {
        // Even k1 (including zero) must round up to odd so the
        // murmur mix stays valid. Two samplers built with
        // `(k1=0, k2=X)` and `(k1=1, k2=X)` therefore agree on
        // every hash because both effectively use `mix_k1 = 1`.
        let s_zero = UpdateSampler::new_keyed_with_seeds(4, 0, 0);
        let s_one = UpdateSampler::new_keyed_with_seeds(4, 1, 0);
        for h in 0..16_u64 {
            assert_eq!(s_zero.accept_hash(h), s_one.accept_hash(h));
        }
    }

    #[test]
    fn keyed_sampler_is_deterministic_within_sampler() {
        // Same flow hash must decide the same way every call on
        // the same sampler — baseline coverage per flow.
        let s = UpdateSampler::new_keyed(4).unwrap();
        let h = 0xdead_beef_cafe_babe_u64;
        let d1 = s.accept_hash(h);
        let d2 = s.accept_hash(h);
        let d3 = s.accept_hash(h);
        assert_eq!(d1, d2);
        assert_eq!(d2, d3);
    }

    fn nz_u32(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("non-zero")
    }

    fn nz_u64(n: u64) -> NonZeroU64 {
        NonZeroU64::new(n).expect("non-zero")
    }

    #[test]
    fn prefix_rate_cap_allows_up_to_cap() {
        let cap = PrefixRateCap::new(nz_u32(3), nz_u64(1_000));
        let prefix = 0x1234_5678_u64;
        let now = 1_000_u64;
        assert!(cap.check_and_record(prefix, now));
        assert!(cap.check_and_record(prefix, now));
        assert!(cap.check_and_record(prefix, now));
        // 4th call hits cap → rejected.
        assert!(!cap.check_and_record(prefix, now));
        assert_eq!(cap.admitted_total(), 3);
        assert_eq!(cap.capped_total(), 1);
    }

    #[test]
    fn prefix_rate_cap_disabled_admits_all() {
        let cap = PrefixRateCap::disabled(nz_u64(1_000));
        for i in 0..100_u64 {
            assert!(cap.check_and_record(i, 0));
        }
        assert_eq!(cap.admitted_total(), 100);
        assert_eq!(cap.capped_total(), 0);
    }

    #[test]
    fn prefix_rate_cap_resets_on_window_roll() {
        let cap = PrefixRateCap::new(nz_u32(2), nz_u64(1_000));
        let prefix = 0xabcd_u64;
        // Fill the cap inside one window.
        assert!(cap.check_and_record(prefix, 100));
        assert!(cap.check_and_record(prefix, 200));
        assert!(!cap.check_and_record(prefix, 300));
        // Advance past the window — next call resets and admits.
        assert!(cap.check_and_record(prefix, 1_500));
        assert_eq!(cap.admitted_total(), 3);
    }

    #[test]
    fn prefix_cap_buckets_are_cache_line_padded() {
        // Compile-time invariant: each PaddedBucket is exactly
        // one 64-byte cache line. If this regresses (e.g. a new
        // field added without re-aligning), the bench
        // `check_and_record_contended_8threads` will surface the
        // false-sharing penalty before the bug ships.
        assert_eq!(core::mem::size_of::<PaddedBucket>(), 64);
        assert_eq!(core::mem::align_of::<PaddedBucket>(), 64);
        assert_eq!(
            core::mem::size_of::<[PaddedBucket; PrefixRateCap::BUCKETS]>(),
            16 * 1024,
            "bucket bank must be 16 KiB (256 × 64 B cache-line-padded)"
        );
    }

    #[test]
    fn prefix_cap_distinct_buckets_concurrent_throughput() {
        // 8 threads hammer 8 distinct buckets so every write
        // lands on a different cache line; with cache-line padding
        // the test exercises the lock-free path with zero false
        // sharing. The point is correctness (admission counters
        // line up) not throughput — the bench
        // `check_and_record_contended_8threads` measures the
        // perf side.
        use std::sync::Arc;
        use std::thread;
        let cap = Arc::new(PrefixRateCap::new(nz_u32(1_000_000), nz_u64(60_000)));
        let n_threads = 8_u64;
        let n_per_thread = 50_000_u64;
        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let cap = Arc::clone(&cap);
                thread::spawn(move || {
                    // Each thread targets one distinct bucket via
                    // a stride that lands on (t & 0xff).
                    let prefix = u64::from(u8::try_from(t).unwrap());
                    for _ in 0..n_per_thread {
                        assert!(cap.check_and_record(prefix, 0));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(cap.admitted_total(), n_threads * n_per_thread);
        assert_eq!(cap.capped_total(), 0);
    }

    #[test]
    fn prefix_rate_cap_concurrent_rollover_holds_hard_cap() {
        // Every thread hammers the same prefix across many window
        // rollovers; after all joins, total admits per window
        // must equal `thread_count * windows_crossed` worth of
        // caps at most. The CAS-loop rollover path must not admit
        // more than `cap_per_window` per prefix-per-window even
        // under concurrent reset races.
        use std::sync::Arc;
        use std::thread;
        let cap = Arc::new(PrefixRateCap::new(nz_u32(4), nz_u64(100)));
        let threads: Vec<_> = (0..8_u64)
            .map(|t| {
                let cap = Arc::clone(&cap);
                thread::spawn(move || {
                    for step in 0..2_000_u64 {
                        // `now_ms` advances 1 ms per step; every
                        // 100 steps the window rolls.
                        let now = step + t;
                        let _ = cap.check_and_record(0xdead_beef, now);
                    }
                })
            })
            .collect();
        for h in threads {
            h.join().expect("thread join");
        }
        // Admitted count grows linearly with windows crossed, not
        // with thread count. The exact bound is fuzzy (rollover
        // races shift window boundaries) but admitted per window
        // must stay close to `cap_per_window = 4`. Over ~2000 ms
        // with a 100 ms window that is at most ~20 windows × 4 =
        // 80 admits. Allow 4× slack for timing jitter.
        let admitted = cap.admitted_total();
        assert!(
            admitted <= 320,
            "admit count {admitted} exceeds generous bound — race leaks"
        );
    }
}
