//! `anomstream-triage` — SOC-opinionated triage layer on top of
//! `anomstream-core`.
//!
//! Six higher-level components that turn a raw anomaly-score
//! stream into something analysts can act on:
//!
//! - [`calibrator`] — Platt probability calibration (batch +
//!   online SGD)
//! - [`sage`] — SAGE Shapley attribution via permutation sampling
//! - [`alert_cluster`] — cosine-similarity alert dedup over a
//!   sliding window
//! - [`lsh_cluster`] — LSH-based alert dedup for MSSP-volume
//!   streams
//! - [`feedback`] — bounded ledger of analyst labels +
//!   Gaussian-kernel score adjustment (Das et al. 2017)
//! - [`audit`] — serialisable [`audit::AlertRecord`] envelope
//!   packaging every analytic output for SIEM / WORM export
//!
//! All six consume core output types (`DiVector`, `AnomalyScore`,
//! `AnomalyGrade`, `Severity`, `MetricsSink`, `RandomCutForest`,
//! `ForensicBaseline`) via `anomstream-core`; none depend on
//! each other except within-crate (`alert_cluster` and
//! `lsh_cluster` consume [`audit::AlertRecord`]).
//!
//! # Scope
//!
//! Policy-opinionated layer — not every consumer wants SOC
//! vocabulary (cluster dedup, audit records, feedback-weighted
//! scores). Consumers who only need detectors + primitives
//! should depend on `anomstream-core` alone.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::panic))]

extern crate alloc;

pub mod alert_cluster;
pub mod audit;
pub mod calibrator;
#[cfg(feature = "std")]
pub mod feedback;
#[cfg(feature = "std")]
pub mod lsh_cluster;
#[cfg(feature = "std")]
pub mod sage;

pub use alert_cluster::{AlertCluster, AlertClusterer, ClusterDecision};
pub use audit::{ALERT_RECORD_VERSION, AlertContext, AlertRecord};
pub use calibrator::{PlattCalibrator, PlattFitConfig};
#[cfg(feature = "std")]
pub use feedback::{
    DEFAULT_CAPACITY as FEEDBACK_DEFAULT_CAPACITY, DEFAULT_KERNEL_SIGMA as FEEDBACK_DEFAULT_SIGMA,
    DEFAULT_STRENGTH as FEEDBACK_DEFAULT_STRENGTH, FeedbackLabel, FeedbackStore,
};
#[cfg(feature = "std")]
pub use lsh_cluster::{LshAlertClusterer, LshClusterDecision};
#[cfg(feature = "std")]
pub use sage::{
    DEFAULT_PERMUTATIONS as SAGE_DEFAULT_PERMUTATIONS, DEFAULT_SEED as SAGE_DEFAULT_SEED,
    SageEstimator, SageExplanation,
};
