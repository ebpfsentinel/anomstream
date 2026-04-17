//! Read CSV rows from stdin, score each one through the forest, and
//! print `score,is_anomaly`. The forest is pre-trained from the
//! first `WARMUP` rows of the stream so the score reflects deviation
//! from the historical distribution. Anomaly threshold defaults to
//! `1.5` (caller may override via the first CLI argument).
//!
//! Dimensionality is fixed at compile time via the [`DIM`] constant
//! (default `4`) — every input row must contain exactly `DIM`
//! comma-separated floats. Edit the constant and rebuild to ingest
//! a different feature width; the const-generic [`ForestBuilder`]
//! pins it at the type level so there is no runtime dim parameter.
//!
//! Run with `cargo run --example streaming -- 1.8 < data.csv`.

use std::env;
use std::io::{self, BufRead, Write};

use rcf_rs::{ForestBuilder, RcfError};

const WARMUP: usize = 200;
const DEFAULT_THRESHOLD: f64 = 1.5;
/// Per-point dimensionality this binary expects. Adjust to match
/// your feature width and rebuild.
const DIM: usize = 4;

fn parse_row(line: &str) -> Option<[f64; DIM]> {
    let parts: Result<Vec<f64>, _> = line
        .trim()
        .split(',')
        .map(str::trim)
        .map(str::parse::<f64>)
        .collect();
    let parts = parts.ok()?;
    parts.try_into().ok()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let threshold = env::args()
        .nth(1)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_THRESHOLD);

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    // Pull the first row.
    let first_line = loop {
        match lines.next() {
            Some(Ok(line)) if !line.trim().is_empty() => break line,
            Some(Ok(_)) => {}
            Some(Err(e)) => return Err(e.into()),
            None => return Ok(()),
        }
    };
    let first_point = parse_row(&first_line)
        .ok_or_else(|| format!("first row must be {DIM} numeric comma-separated floats"))?;

    let mut forest = ForestBuilder::<DIM>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(2026)
        .build()?;

    // Warmup phase: feed but don't score.
    forest.update(first_point)?;
    let mut warmed = 1;
    while warmed < WARMUP {
        match lines.next() {
            Some(Ok(line)) if !line.trim().is_empty() => {
                let p = parse_row(&line)
                    .ok_or_else(|| format!("malformed CSV row (expected {DIM} floats)"))?;
                forest.update(p)?;
                warmed += 1;
            }
            Some(Ok(_)) => {}
            Some(Err(e)) => return Err(e.into()),
            None => break,
        }
    }

    writeln!(out, "score,is_anomaly")?;

    // Streaming phase: score then absorb.
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let p =
            parse_row(&line).ok_or_else(|| format!("malformed CSV row (expected {DIM} floats)"))?;
        let score: f64 = match forest.score(&p) {
            Ok(s) => s.into(),
            Err(RcfError::EmptyForest) => 0.0,
            Err(e) => return Err(e.into()),
        };
        let flag = u8::from(score >= threshold);
        writeln!(out, "{score:.6},{flag}")?;
        forest.update(p)?;
    }

    Ok(())
}
