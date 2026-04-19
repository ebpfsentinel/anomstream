# NAB — detection-quality benchmark on real corpora

The Numenta Anomaly Benchmark is the canonical public data set
for streaming anomaly detection.  `scripts/nab/fetch.sh` clones
the upstream repo (Apache 2.0); `tests/nab.rs` is an `#[ignore]`
integration test that runs rcf-rs against the `realKnownCause`
subset and reports per-file + aggregate AUC.

## Running

```bash
./scripts/nab/fetch.sh /opt/nab
RCF_NAB_PATH=/opt/nab \
    cargo test --test nab --all-features -- --ignored --nocapture
```

The test expects the standard NAB layout:

```
$RCF_NAB_PATH/
  data/realKnownCause/*.csv
  labels/combined_windows.json
```

## Scoring protocol

- **Feature engineering**: 4-lag temporal embedding (`[v_{t-3},
  v_{t-2}, v_{t-1}, v_t]`) → `D = 4`. RCF on raw scalars loses
  most of its value; lag features give the tree cuts meaningful
  axes.
- **Two-phase**: warm on the first 15 % of each series (assumed
  mostly clean — NAB anomalies are concentrated mid-stream), then
  stream-score the rest.
- **Labels**: timestamp comparison against `combined_windows.json`
  `[start, end]` pairs. A row is labelled anomalous iff its
  timestamp falls inside *any* window.
- **AUC**: trapezoidal rule on the ROC curve; per-file + weighted
  aggregate (by number of anomalous rows).

## Expected thresholds

NAB is a hard benchmark. Published state-of-the-art per-file AUC
sits in the 0.65–0.90 range depending on the series. We pin the
aggregate floor at `0.60` as a regression guard — not a quality
claim. Substantially better numbers require time-aware detectors
(HTM, sequence models) beyond the RCF scope.

## Not covered

- **Yahoo S5**: requires registration with Yahoo and forbids
  redistribution; out of scope for a public open-source crate.
- **Wikipedia pageviews**: not a labeled anomaly corpus — public
  time series without ground truth.
