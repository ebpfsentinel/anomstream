//! A dash in this tree is the plain hyphen.
//!
//! The rule had held in prose and nowhere in a pipeline, which is why it drifted:
//! two repositories of fourteen checked it, and those two were the only two that
//! measured zero. This is the check for this one, over everything git tracks
//! rather than over the Rust sources, because the copy a reader sees is as often
//! a workflow, a chart or a line of documentation.
//!
//! The four code points are written as escapes on purpose. A guard naming the
//! character it forbids is a guard that fails on itself first, and the fix for
//! that failure is to weaken the guard.
//!
//! There is no exemption list, and the one case that would need one is dormant:
//! the copies of the design charter in this project carry no typographic dash
//! because the charter itself does not. The day it gains one, the exemption goes
//! here with the path and the reason, because a copy that corrects its source is
//! a copy that drifts.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Dashes that are not the plain hyphen, with the name to print when one is found.
const FORBIDDEN_DASHES: &[(char, &str)] = &[
    ('\u{2012}', "figure dash"),
    ('\u{2013}', "en dash"),
    ('\u{2014}', "em dash"),
    ('\u{2015}', "horizontal bar"),
];

/// Files a scan may not read as text, by extension.
const BINARY_EXTENSIONS: &[&str] = &[
    "onnx", "png", "jpg", "jpeg", "gif", "ico", "pdf", "woff", "woff2", "ttf", "otf", "mp4",
    "webm", "wav", "zip", "gz", "tar", "bin", "mmdb",
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .expect("this crate sits one level below the workspace root")
        .to_path_buf()
}

/// Every file git tracks, which is exactly the set that reaches a reader.
fn tracked_files(root: &Path) -> Vec<PathBuf> {
    let out = Command::new("git")
        .arg("ls-files")
        .arg("-z")
        .current_dir(root)
        .output()
        .expect("git ls-files runs at the workspace root");
    assert!(
        out.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("git prints paths as UTF-8")
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(|p| root.join(p))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_none_or(|e| !BINARY_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        })
        .collect()
}

#[test]
fn tracked_files_carry_no_typographic_dash() {
    let root = workspace_root();
    let files = tracked_files(&root);
    assert!(
        !files.is_empty(),
        "the scan found no tracked file at {}, so it was asserting nothing",
        root.display()
    );

    let mut offences: Vec<String> = Vec::new();
    for path in &files {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        for (number, line) in text.lines().enumerate() {
            for (dash, name) in FORBIDDEN_DASHES {
                if line.contains(*dash) {
                    offences.push(format!(
                        "{}:{}: {name}: {}",
                        path.strip_prefix(&root).unwrap_or(path).display(),
                        number + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        offences.is_empty(),
        "a dash in this tree is the plain hyphen, in code, comments, workflows, \
         documentation and user-facing copy alike:\n{}",
        offences.join("\n")
    );
}
