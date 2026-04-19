# AWS `randomcutforest-java` comparison outline

Not a runnable script — too many moving parts (Maven, JDK, JMH).
Sketch for someone willing to wire it up manually.

## Environment

```bash
sudo apt install -y openjdk-26-jdk maven
git clone https://github.com/aws/random-cut-forest-by-aws.git
cd random-cut-forest-by-aws
mvn -pl core install -DskipTests
```

## Minimal driver

Create a `RcfBench.java` that reads `data.csv` (same format as
`gen_points.py`), builds an `RandomCutForest` with the same
`(num_trees, sample_size, dim)` as rcf-rs, and times the same
update + score loop:

```java
import com.amazon.randomcutforest.RandomCutForest;

public class RcfBench {
    public static void main(String[] args) throws Exception {
        int D = 16;
        int trees = 100;
        int sample = 256;
        RandomCutForest forest = RandomCutForest.builder()
            .dimensions(D)
            .numberOfTrees(trees)
            .sampleSize(sample)
            .randomSeed(2026L)
            .build();

        // Load CSV (label, d0 .. dD-1) into a List<double[]>.
        // Measure with System.nanoTime() around update + getAnomalyScore
        // loops the same way the Python / Rust drivers do.
    }
}
```

Run with:

```bash
mvn -pl core exec:java -Dexec.mainClass=RcfBench -Dexec.args="data.csv 100 256"
```

## Expected comparison points

- **Updates/s**: AWS Java ~5–15× slower than rcf-rs on wallclock
  (JIT vs native + allocation profile). Not a fair comparison
  for cold-start — JVM warmup dominates short runs.
- **Scores/s**: closer, JVM vectorises score aggregation well.
- **AUC**: should match rcf-rs within floating-point noise on
  separable data.

## Caveats

- JMH is the right tool for JVM micro-benchmarks. `System.nanoTime()`
  around a bare loop does not handle JIT warmup, GC pauses, or
  thermal drift. Treat numbers produced by the sketch above as
  *indicative only*.
- AWS's library also ships a `ThresholdedRandomCutForest` wrapper
  analogous to rcf-rs's `ThresholdedForest`; compare the TRCF
  layer separately (`grade` emission cost + `stats()` bookkeeping).
