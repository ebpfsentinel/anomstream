# anomstream-rs threat model

Scope: adversarial inputs on the ingress path where `anomstream-rs`
consumes untrusted feature vectors (eBPF NDR agent, MSSP tenant
pool, public-facing API). Out of scope: host compromise of the
process running `anomstream-rs`, side-channel attacks on Ed25519 license
verification, upstream supply-chain compromises of rustc /
dependencies.

Referenced: MITRE ATLAS (Adversarial Threat Landscape for AI
Systems) tactic / technique IDs.

## T1 — Reservoir poisoning (`AML.T0020`)

### Attack

Attacker emits traffic whose feature vectors land in the
reservoir's retained sample. Once accepted as "baseline", the
attacker's traffic shape becomes the detector's normal, and
subsequent (real) anomalous traffic of the same shape is no
longer flagged. Two vectors:

1. **Deterministic admission spray** — when `UpdateSampler` uses
   the unkeyed `accept_hash(flow_hash)` path, admission is
   `flow_hash % keep_every_n == 0`. An attacker who can probe
   the admission decision (observe whether a given flow was
   reflected in the baseline via score drift) can spray 5-tuples
   whose caller-computed hash lands on the admitted residue class.
2. **Single-source flood** — even keyed admission admits a fixed
   rate per flow; a flood from a compromised source IP that
   churns through many 5-tuples saturates the reservoir with
   points all from the attacker's network.

### Defences shipped

- **Keyed sampler**: `UpdateSampler::new_keyed(keep_every_n)`
  seeds 128 bits of per-sampler secret from `getrandom` at
  construction, applies a murmur3-style keyed finaliser to every
  `accept_hash` input before the modulo. Attacker cannot steer
  their flow hash into the admitted residue class without learning
  the sampler secret; the secret never leaves the process.
- **Per-prefix rate cap**: `PrefixRateCap::new(cap, window_ms)`
  bounds how many admissions a single `/24`-prefix hash bucket
  can push within a rolling window. Fixed 256-bucket sketch;
  O(1) check-and-record; lock-free. Collisions are soft (by
  design — trades a little cross-prefix interference for constant
  memory).
- **Trimmed-mean score aggregator**:
  `RandomCutForest::score_trimmed(&point, trim_fraction)` sorts
  per-tree scores, drops the top and bottom `trim_fraction`
  fraction, averages the middle. An attacker who manages to
  poison a minority of trees sees their contribution trimmed
  from the ensemble mean. Typical value: `0.10` (10 %/10 %).

### Defences NOT shipped

- No host-level rate limit on ingress (caller concern).
- No cryptographic integrity on the feature vector itself
  (caller must sanitise upstream).
- No per-tenant separation of reservoir secrets across
  `TenantForestPool` entries — all tenants share the keyed
  sampler's secret. Rotate on tenant provisioning if higher
  isolation is needed.

## T2 — Evasion via contextual shift (`AML.T0043`)

### Attack

Attacker gradually shifts a target feature's distribution so the
detector's baseline drifts in step. When the payload arrives, it
sits inside the (drifted) baseline and is not flagged.

### Defences shipped

- `MetaDriftDetector` (CUSUM on the score stream) fires
  `DriftKind::Upward` / `DriftKind::Downward` on sustained
  baseline drift. Observability only — caller decides action.
- `FeatureDriftDetector` (PSI + KL per feature) fires
  `DriftLevel::Alert` when the production distribution diverges
  from the frozen baseline by `≥ 0.25` PSI. Pin the offending
  dim via `argmax_psi()`.
- `score_codisp_stateless` preserves the frozen baseline across
  long eval streams; the mutating `score_codisp` path drifts by
  design — the non-mutating variant is the one to use when the
  caller needs a contextual-displacement score on a trusted
  baseline.

### Defences NOT shipped

- Automatic drift *recovery* (shadow-forest swap on alert). The
  ADWIN-based swap is P1 on the roadmap.

## T3 — Model extraction (`AML.T0024`)

### Attack

Attacker queries the detector through the score-exposing API
(via exposed decisions or via a side channel), reconstructs the
isolation-depth boundary, and uses it offline to design
undetectable payloads.

### Defences shipped

- No public score API out of the process boundary. The crate
  provides in-process `score()` only; exposing it externally is
  the caller's architectural decision.
- `score` returns a clamped `AnomalyScore` newtype — its
  internal representation is not exposed beyond `f64` accessor,
  making pre-clamp score leakage unlikely under `rustc` opt.

### Defences NOT shipped

- No differentially-private score perturbation
  (DP-SGD-equivalent for isolation forests is an open research
  problem; not in scope).

## T4 — Classifier-side resource exhaustion

### Attack

Attacker emits traffic at a rate high enough to overflow the
in-process MPSC `hot_path::channel` between the classifier and
the updater thread, starving legitimate updates.

### Defences shipped

- `UpdateProducer::try_enqueue` is non-blocking — on full queue
  it increments `dropped_total` and returns `false`. The
  classifier stays hot-path-safe; the cost is visible via the
  counter so ops can alert on `dropped_total > 0`.
- `UpdateSampler` drops low-value updates before the queue. A
  1/N stride or per-flow gate is free (no allocations, no
  syscalls).

### Defences NOT shipped

- No back-pressure signal from the updater to the sampler. If
  the updater falls behind, the queue drops silently until ops
  reacts to the `dropped_total` gauge.

## What is explicitly NOT in the threat model

- Kernel-side eBPF verifier compromise (kernel concern).
- Compromise of the upstream CTI feeds that drive threat
  intelligence (out of scope — `anomstream-rs` does not consume CTI
  directly).
- Attacks on the persistence format (`to_bytes` / `from_bytes`)
  — the crate enforces versioned envelopes with upfront
  rejection of incompatible versions, but a compromised
  serialised state file trivially compromises the loaded
  detector. Callers must treat forest snapshots as
  integrity-sensitive (sign + verify out-of-band).
