//! `anomstream` — streaming anomaly detection toolkit (facade).
//!
//! Umbrella crate that re-exports the three workspace members
//! under feature gates so consumers get a single import path
//! regardless of which layers they pull in:
//!
//! ```toml
//! [dependencies]
//! anomstream = { version = "0.2", default-features = false, features = ["core", "triage"] }
//! ```
//!
//! Feature matrix:
//!
//! | Feature | Pulls in | Purpose |
//! |---|---|---|
//! | `core` | `anomstream-core` | Detectors + streaming primitives + `MetricsSink` + `SeverityBands` |
//! | `triage` | `anomstream-triage` | Platt, SAGE, alert clustering, feedback, audit records |
//! | `hotpath` | `anomstream-hotpath` | eBPF-style ingress `UpdateSampler` / `PrefixRateCap` / `update_channel` |
//!
//! `triage` and `hotpath` both depend on `core`; enabling them
//! implies `core`. Default feature set enables all three plus
//! `std` / `parallel` / `serde` / `postcard` passthroughs.
//!
//! # Consumer DX
//!
//! Core + triage are re-exported as glob imports under the crate
//! root so consumers write one `use anomstream::*;` regardless of
//! which layer they depend on:
//!
//! ```ignore
//! use anomstream::{ForestBuilder, PerFeatureCusum, PerFeatureCusumConfig};
//! #[cfg(feature = "triage")]
//! use anomstream::{AlertClusterer, PlattCalibrator};
//! #[cfg(feature = "hotpath")]
//! use anomstream::hot_path::{PrefixRateCap, UpdateSampler};
//! ```
//!
//! Hot-path primitives live under a [`hot_path`] submodule to
//! preserve the `anomstream::hot_path::*` import path used by
//! pre-workspace-split callers and to flag the opinionated nature
//! of that layer.
//!
//! Sibling member namespaces also stay accessible verbatim via
//! [`core_lib`], [`triage_lib`], [`hotpath_lib`] when a consumer
//! needs to spell a module path that the glob cannot disambiguate.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
// Every member-crate doc link is either full-path
// (`[crate::Foo]`) or uses an in-scope identifier, so rustdoc
// resolves them cleanly under the facade namespace. The old
// blanket allow for `rustdoc::broken_intra_doc_links` is no
// longer needed and CI (`RUSTDOCFLAGS="-D
// rustdoc::broken_intra_doc_links"`) fails on new breakage.

// -- Glob re-exports (flat DX) --

#[cfg(feature = "core")]
#[doc(inline)]
pub use anomstream_core::*;

#[cfg(feature = "triage")]
#[doc(inline)]
pub use anomstream_triage::*;

/// eBPF-style ingress primitives (`UpdateSampler`, `PrefixRateCap`,
/// `channel`) — re-exported as a submodule so the path
/// `anomstream::hot_path::UpdateSampler` matches the pre-split
/// `anomstream_core::hot_path::UpdateSampler` spelling.
#[cfg(feature = "hotpath")]
pub mod hot_path {
    #[doc(inline)]
    pub use anomstream_hotpath::*;
}

// -- Named namespaces (escape hatch for module-path reaches) --

/// Direct access to the `anomstream-core` crate namespace for
/// consumers that need the full module tree (`anomstream::core_lib::forest::*`,
/// `anomstream::core_lib::serde_util::*`, …).
#[cfg(feature = "core")]
#[doc(inline)]
pub use anomstream_core as core_lib;

/// Direct access to the `anomstream-triage` crate namespace.
#[cfg(feature = "triage")]
#[doc(inline)]
pub use anomstream_triage as triage_lib;

/// Direct access to the `anomstream-hotpath` crate namespace.
#[cfg(feature = "hotpath")]
#[doc(inline)]
pub use anomstream_hotpath as hotpath_lib;
