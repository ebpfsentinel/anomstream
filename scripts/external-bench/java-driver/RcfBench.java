// Minimal harness for `aws/random-cut-forest-by-aws` (v4.4.0)
// to sit in the same comparison matrix as rcf-rs / rrcf /
// sklearn. Reads the same CSV shape as gen_points.py.
//
// Usage:
//   javac -cp /path/to/randomcutforest-core-4.4.0.jar RcfBench.java
//   java -cp .:/path/to/randomcutforest-core-4.4.0.jar RcfBench data.csv 100 256

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import com.amazon.randomcutforest.RandomCutForest;

public class RcfBench {

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: RcfBench <csv> <num_trees> <sample_size>");
            System.exit(2);
        }
        String csv = args[0];
        int numTrees = Integer.parseInt(args[1]);
        int sample = Integer.parseInt(args[2]);

        // Parse CSV into (points, labels).
        List<double[]> rowsD = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        int dim = -1;
        try (BufferedReader br = new BufferedReader(new FileReader(csv))) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                if (first) {
                    first = false;
                    continue;
                }
                String[] parts = line.split(",");
                if (dim < 0) dim = parts.length - 1;
                labels.add(Integer.parseInt(parts[0]));
                double[] p = new double[dim];
                for (int i = 0; i < dim; i++) {
                    p[i] = Double.parseDouble(parts[i + 1]);
                }
                rowsD.add(p);
            }
        }
        int n = rowsD.size();
        int split = n * 3 / 10;
        System.out.printf("points=%d dim=%d trees=%d sample=%d warm=%d%n",
            n, dim, numTrees, sample, split);

        RandomCutForest forest = RandomCutForest.builder()
            .dimensions(dim)
            .numberOfTrees(numTrees)
            .sampleSize(sample)
            .randomSeed(2026L)
            .build();

        // Warm phase.
        long t0 = System.nanoTime();
        for (int i = 0; i < split; i++) {
            forest.update(rowsD.get(i));
        }
        long insertNs = System.nanoTime() - t0;

        // Eval phase.
        double[] scores = new double[n - split];
        int[] evalLabels = new int[n - split];
        t0 = System.nanoTime();
        for (int i = split; i < n; i++) {
            scores[i - split] = forest.getAnomalyScore(rowsD.get(i));
            evalLabels[i - split] = labels.get(i);
        }
        long scoreNs = System.nanoTime() - t0;

        double insertPerS = split * 1.0e9 / insertNs;
        double scorePerS = (n - split) * 1.0e9 / scoreNs;
        double auc = auc(scores, evalLabels);

        System.out.printf("  inserts        = %d, total %.2f ms%n",
            split, insertNs / 1.0e6);
        System.out.printf("  scores         = %d, total %.2f ms%n",
            n - split, scoreNs / 1.0e6);
        System.out.printf("  per-op insert  = %.0f ns%n",
            (double) insertNs / split);
        System.out.printf("  per-op score   = %.0f ns%n",
            (double) scoreNs / (n - split));
        System.out.printf("  updates_per_s  = %.0f%n", insertPerS);
        System.out.printf("  scores_per_s   = %.0f%n", scorePerS);
        System.out.printf("  auc            = %.3f%n", auc);
    }

    static double auc(double[] scores, int[] labels) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(scores[b], scores[a]));
        long totalPos = 0;
        for (int l : labels) if (l == 1) totalPos++;
        long totalNeg = labels.length - totalPos;
        if (totalPos == 0 || totalNeg == 0) return 0.5;
        double aucVal = 0.0;
        long tp = 0, fp = 0;
        double prevTpr = 0, prevFpr = 0;
        for (int idx : order) {
            if (labels[idx] == 1) tp++;
            else fp++;
            double tpr = (double) tp / totalPos;
            double fpr = (double) fp / totalNeg;
            aucVal += (fpr - prevFpr) * (tpr + prevTpr) / 2.0;
            prevTpr = tpr;
            prevFpr = fpr;
        }
        return aucVal;
    }
}
