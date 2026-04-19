#!/usr/bin/env bash
# Fetch and extract the TSB-AD multivariate track (~515 MB zip,
# ~1.6 GB extracted). Official download URL per
# https://github.com/thedatumorg/TSB-AD.
#
# Usage:
#   scripts/tsb_ad/fetch.sh [DEST_DIR]
#
# After extraction, export the path before running the test:
#
#   export RCF_TSB_AD_M_PATH="<DEST_DIR>/TSB-AD-M"
#   cargo test --test tsb_ad_m --all-features -- --ignored --nocapture
set -euo pipefail

DEST="${1:-/tmp/tsb-ad}"
URL="https://www.thedatum.org/datasets/TSB-AD-M.zip"

mkdir -p "$DEST"
cd "$DEST"

if [[ ! -f TSB-AD-M.zip ]]; then
    echo "Downloading $URL -> $DEST/TSB-AD-M.zip"
    curl -fSL --max-time 600 -o TSB-AD-M.zip "$URL"
else
    echo "Reusing cached $DEST/TSB-AD-M.zip"
fi

if [[ ! -d TSB-AD-M ]]; then
    echo "Extracting -> $DEST/TSB-AD-M/"
    unzip -q TSB-AD-M.zip
else
    echo "Reusing cached $DEST/TSB-AD-M/"
fi

echo
echo "Next:"
echo "  export RCF_TSB_AD_M_PATH=\"$DEST/TSB-AD-M\""
echo "  cargo test --test tsb_ad_m --all-features -- --ignored --nocapture"
