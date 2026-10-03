#!/usr/bin/env python3
"""Benchmark `rrcf` on the shared CSV dataset.

Reports:

- `updates_per_s` - forest-insert throughput
- `scores_per_s` - scoring throughput (`codisp` output)
- `auc` - ROC-AUC on the `label` column vs codisp score

Not benchmark-grade - best-effort single-process wall-clock timing.
Use `time.perf_counter_ns()` so the resolution is well below the
per-op cost.

Usage:
    pip install --user rrcf numpy
    python3 bench_rrcf_synthetic.py --input data.csv --trees 100 --sample 256
"""

import argparse
import csv
import time

import numpy as np
import rrcf


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            labels.append(int(row[0]))
            rows.append([float(x) for x in row[1:]])
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int8)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    total_pos = labels_sorted.sum()
    total_neg = len(labels_sorted) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    tp = 0
    fp = 0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auc_val = 0.0
    for y in labels_sorted:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc_val += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr = tpr
        prev_fpr = fpr
    return auc_val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--sample", type=int, default=256)
    args = parser.parse_args()

    X, labels = load_csv(args.input)
    n, d = X.shape
    print(f"points={n} dim={d} trees={args.trees} sample={args.sample}")

    # Warm reservoir with first `trees * sample` points.
    warm = args.trees * args.sample
    forest = [rrcf.RCTree() for _ in range(args.trees)]

    start = time.perf_counter_ns()
    for idx in range(min(warm, n)):
        for t in forest:
            if len(t.leaves) >= args.sample:
                # pop oldest to respect reservoir size
                t.forget_point(idx - args.sample)
            t.insert_point(X[idx], index=idx)
    insert_ns = time.perf_counter_ns() - start
    print(f"  warm inserts = {warm}, total {insert_ns / 1e6:.2f} ms")

    # Score the rest (post-warm).
    start = time.perf_counter_ns()
    scores = np.zeros(n)
    for idx in range(n):
        codisp = 0.0
        for t in forest:
            if idx in t.leaves:
                codisp += t.codisp(idx)
        scores[idx] = codisp / len(forest)
    score_ns = time.perf_counter_ns() - start
    print(f"  scores       = {n}, total {score_ns / 1e6:.2f} ms")
    print(f"  per-op insert = {insert_ns / warm:.0f} ns")
    print(f"  per-op score  = {score_ns / n:.0f} ns")
    print(f"  updates_per_s = {warm * 1e9 / insert_ns:.0f}")
    print(f"  scores_per_s  = {n * 1e9 / score_ns:.0f}")
    print(f"  auc           = {auc(scores, labels):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
