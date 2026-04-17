//! Demonstrates pinning [`mimalloc`] as the global allocator —
//! a one-line change in the caller's `main.rs` that frees a few
//! per cent on every `update`/`score` (most visible on
//! `attribution`, where the per-tree allocations dominate).
//!
//! Run with `cargo run --example with_mimalloc --features parallel`.
//! Add `mimalloc = "0.1"` to your `Cargo.toml` to use it in your
//! own binary.

use mimalloc::MiMalloc;
use rcf_rs::{ForestBuilder, RcfError};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> Result<(), RcfError> {
    // Same shape as `examples/quickstart.rs` — only the global
    // allocator differs.
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(42)
        .build()?;

    for i in 0..200 {
        let v = f64::from(i) * 0.001;
        forest.update([v, v + 0.5, v - 0.5, v * 2.0])?;
    }

    let outlier = forest.score(&[100.0, 100.0, 100.0, 100.0])?;
    let attribution = forest.attribution(&[100.0, 100.0, 100.0, 100.0])?;

    println!("global allocator = mimalloc");
    println!("outlier score    = {outlier}");
    println!("attribution argmax dim = {:?}", attribution.argmax());
    Ok(())
}
