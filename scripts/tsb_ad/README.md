# TSB-AD-M detection-quality protocol

Modern complement to the NAB regression test. TSB-AD-M (TheDatumOrg,
2024) ships **200 multivariate time series** with per-point binary
labels across 16 source datasets (MSL, SMAP, SMD, MITDB, SVDB, PSM,
GHL, Exathlon, OPPORTUNITY, CATSv2, LTDB, …). Unlike NAB's
windowed labels, every row is independently labelled, and series
carry native multivariate structure (no lag embedding).

## Fetch

```bash
scripts/tsb_ad/fetch.sh /tmp/tsb-ad
export RCF_TSB_AD_M_PATH=/tmp/tsb-ad/TSB-AD-M
```

Zip is ~515 MB; extracted corpus ~1.6 GB.

## Run

```bash
cargo test --test tsb_ad_m --all-features -- --ignored --nocapture
```

Expect ~8–15 min on the reference hardware (13th-gen i7, 14C / 20T).

## Pipeline

Per file:
1. Parse filename — `NNN_DATASET_id_K_Category_tr_<train>_1st_<first>.csv`.
   `tr_<N>` is the upstream train-split boundary.
2. Per-dim z-score using the train-split mean / stddev — the
   datasets mix heterogeneous scales (voltage, heart-rate,
   acceleration, resource counters) and RCF's cut sampling is
   range-weighted.
3. Frozen-baseline forest: warm on the train split, never call
   `update` on eval rows — same paradigm as NAB.
4. EMA-smooth the raw score stream (α = 0.02) before AUC.
5. Point-wise trapezoidal ROC-AUC, aggregated weighted by positive
   count — matches the NAB protocol for comparability.

## Coverage

| `D` | Files |
|---|---|
| 2 | 48 |
| 3 | 14 |
| 7–9 | 4 |
| 12 | 3 |
| 16–19 | 46 |
| 25 | 28 |
| 29–31 | 9 |
| 38 | 22 |
| 51–55 | 17 |
| 66 | 1 |
| 248 | 8 *(skipped)* |

The 17-value const-generic whitelist covers **192 / 200** files
(96 %). Eight `D = 248` files are skipped — monomorphising a
248-dim forest only for 4 % of the corpus inflates compile time
without matching any plausible eBPFsentinel feature vector (native
prod dims are typically ≤ 64).

## Metric caveat

This test uses plain point-wise ROC-AUC. The official TSB-AD
leaderboard ranks on **VUS-PR** (volume under surface, PR variant —
Paparrizos et al. 2022) which integrates range-based precision /
recall over a sliding window and is more robust to lag / label
noise. For an apples-to-apples TSB-AD submission, emit the raw
score streams and evaluate with the
[TheDatumOrg/VUS](https://github.com/TheDatumOrg/VUS) Python
package offline.
