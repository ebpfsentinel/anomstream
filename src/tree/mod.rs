//! Tree algorithm primitives.
//!
//! - [`node::InternalData`] — raw internal-node record
//! - [`node::LeafData`] — raw leaf-node record
//! - [`node::NodeView`] / [`node::NodeViewMut`] — zero-copy enums
//!   over references into the arenas
//! - [`node::NodeRef`] — `u32` packed reference (high bit
//!   discriminates internal from leaf, low bits hold the slot index)
//! - [`node_store::NodeStore`] — split-typed backing store, one
//!   arena per node kind (`InternalData` vs `LeafData`) so leaves
//!   don't pay the internal-variant worst case and fit more entries
//!   per cache line
//!
//! [`random_cut_tree::RandomCutTree`] sits on top of
//! [`node_store::NodeStore`] and provides `add` / `delete` /
//! `traverse` for the actual cut tree.

pub mod node;
pub mod node_store;
pub mod random_cut_tree;

pub use node::{InternalData, LeafData, NodeRef, NodeView, NodeViewMut};
pub use node_store::NodeStore;
pub use random_cut_tree::{PointAccessor, RandomCutTree};
