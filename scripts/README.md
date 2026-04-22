# Scripts — external bench harnesses

Three bench corpora; each lives under its own directory with a
uniform file layout:

| Dir | Corpus | Pipeline |
|---|---|---|
| [`synthetic/`](synthetic/README.md) | 10 k Gaussian points + 1 % outliers (`gen_points.py`) | update / score / AUC — primary throughput comparison |
| [`nab/`](nab/README.md) | Numenta Anomaly Benchmark `realKnownCause` | 8-lag temporal embedding, frozen baseline, weighted-AUC |
| [`tsb_ad/`](tsb_ad/README.md) | TSB-AD multivariate (TheDatumOrg, 2024) | native multivariate, per-dim z-score, frozen baseline |

Each directory ships a uniform set of files, suffixed by the
bench name so the filenames stay unique repo-wide:

| Dir | rrcf runner | AWS Java driver |
|---|---|---|
| `synthetic/` | `bench_rrcf_synthetic.py` | `RcfBenchSynthetic.java` |
| `nab/` | `bench_rrcf_nab.py` | `RcfBenchNab.java` |
| `tsb_ad/` | `bench_rrcf_tsb_ad.py` | `RcfBenchTsbAd.java` |

Auxiliary helpers as needed: `gen_points.py` +
`bench_sklearn_synthetic.py` + `variance_sweep.sh` in
`synthetic/`, `fetch.sh` in `tsb_ad/`.

## AWS `randomcutforest-java` prerequisites (shared)

Every Java driver needs `randomcutforest-core-4.4.0`. Grab the
prebuilt jar from Maven Central — building from source is
**not supported** on JDK 21+ (the upstream pom pins Lombok
1.18.30 which does not handle the modern JDK module layout;
bumping to 1.18.38 does not fix it).

```bash
mkdir -p /tmp/aws-rcf
curl -sLo /tmp/aws-rcf/randomcutforest-core-4.4.0.jar \
    https://repo1.maven.org/maven2/software/amazon/randomcutforest/randomcutforest-core/4.4.0/randomcutforest-core-4.4.0.jar
# SHA-256:
#   2e851c82add6d4bcdd13e5cd85fdd091b8a28185fe104775761e8ff6606fd51b
```

OpenJDK 21 or later (tested on 26) — only `javac` + `java` are
needed.

```bash
sudo apt install -y openjdk-26-jdk
```

Each `RcfBench<Bench>.java` compiles + runs against that jar:

```bash
JAR=/tmp/aws-rcf/randomcutforest-core-4.4.0.jar
# e.g. NAB
javac -cp "$JAR" scripts/nab/RcfBenchNab.java
java -cp "scripts/nab:$JAR" RcfBenchNab /opt/nab
```

## Notes

- Numbers are **cold JVM** on the Java driver — no JMH warmup.
  A proper JVM micro-benchmark would warm JIT for 5–10 s before
  measuring. The cold numbers here represent a realistic
  process-startup cost for a shell-invoked job, which is the
  fair comparison against a native Rust binary.
- AWS Java's `getAnomalyScore` uses a probability-of-separation
  visitor (codisp-like), directly comparable to anomstream-rs's
  `RandomCutForest::score_codisp()` — not to the isolation-depth
  `score()` fast path. See `docs/performance.md` for the
  apples-to-apples split.
