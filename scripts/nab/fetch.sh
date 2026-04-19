#!/usr/bin/env bash
# Shallow-clone the Numenta Anomaly Benchmark into a caller-chosen
# directory. NAB is Apache-2.0 and ~50 MB; not vendored into the
# crate because (a) binary bloat and (b) tests are gated `#[ignore]`
# anyway.
#
# Usage:   ./fetch.sh [/opt/nab]
#
# After fetch, point the ignored integration test at it:
#   RCF_NAB_PATH=/opt/nab cargo test --test nab -- --ignored

set -euo pipefail

TARGET="${1:-/tmp/NAB}"

if [[ -d "$TARGET/data/realKnownCause" ]]; then
    echo "NAB already present at $TARGET — nothing to do."
    exit 0
fi

mkdir -p "$(dirname "$TARGET")"
git clone --depth 1 https://github.com/numenta/NAB.git "$TARGET"
echo
echo "NAB fetched to $TARGET"
echo "Run:  RCF_NAB_PATH=$TARGET cargo test --test nab -- --ignored"
