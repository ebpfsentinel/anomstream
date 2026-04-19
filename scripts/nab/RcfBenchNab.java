// AWS randomcutforest-java on the NAB realKnownCause subset.
// Same protocol as tests/nab.rs / bench_rrcf_nab.py: 8-lag
// temporal embedding, 15 % warm fraction, frozen baseline.
//
// Usage:
//   javac -cp /path/to/randomcutforest-core-4.4.0.jar RcfBenchNab.java
//   java -cp .:/path/to/...jar RcfBenchNab /opt/nab

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

import com.amazon.randomcutforest.RandomCutForest;

public class RcfBenchNab {
    static final int LAG = 8;
    static final double WARM_FRAC = 0.15;
    static final int TREES = 100;
    static final int SAMPLE = 256;

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: RcfBenchNab <nab-root>");
            System.exit(2);
        }
        File nabRoot = new File(args[0]);
        File dataDir = new File(nabRoot, "data/realKnownCause");
        File windowsFile = new File(nabRoot, "labels/combined_windows.json");
        String windowsText = readAll(windowsFile);
        TreeMap<String, List<String[]>> windows = parseWindows(windowsText);

        File[] csvs = dataDir.listFiles((d, n) -> n.endsWith(".csv"));
        if (csvs == null) {
            System.err.println("no CSVs under " + dataDir);
            System.exit(3);
        }
        Arrays.sort(csvs);

        double weightedSum = 0.0;
        long totalPos = 0;
        for (File csv : csvs) {
            String key = "realKnownCause/" + csv.getName();
            List<String[]> w = windows.getOrDefault(key, new ArrayList<>());
            double[] result = scoreFile(csv, w);
            if (result == null) continue;
            double auc = result[0];
            long pos = (long) result[1];
            System.out.printf("  %.3f  pos=%-6d  %s%n", auc, pos, csv.getName());
            weightedSum += auc * pos;
            totalPos += pos;
        }
        if (totalPos > 0) {
            System.out.printf("aggregate weighted AUC: %.3f%n", weightedSum / totalPos);
        }
    }

    // Returns [auc, positive_count] or null when too few rows.
    static double[] scoreFile(File csv, List<String[]> windows) throws Exception {
        List<String> timestamps = new ArrayList<>();
        List<Double> values = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csv))) {
            String line = br.readLine(); // header
            while ((line = br.readLine()) != null) {
                int comma = line.indexOf(',');
                if (comma < 0) continue;
                String ts = line.substring(0, comma);
                if (ts.length() > 19) ts = ts.substring(0, 19);
                timestamps.add(ts);
                values.add(Double.parseDouble(line.substring(comma + 1)));
            }
        }
        if (values.size() < 2 * LAG) return null;

        int embedLen = values.size() - LAG + 1;
        double[][] emb = new double[embedLen][LAG];
        List<String> embTs = new ArrayList<>(embedLen);
        for (int i = 0; i < embedLen; i++) {
            for (int k = 0; k < LAG; k++) emb[i][k] = values.get(i + k);
            embTs.add(timestamps.get(i + LAG - 1));
        }

        int[] labels = new int[embedLen];
        for (int i = 0; i < embedLen; i++) {
            String t = embTs.get(i);
            for (String[] w : windows) {
                if (t.compareTo(w[0]) >= 0 && t.compareTo(w[1]) <= 0) {
                    labels[i] = 1;
                    break;
                }
            }
        }

        int warmEnd = (int) (embedLen * WARM_FRAC);
        RandomCutForest forest = RandomCutForest.builder()
            .dimensions(LAG)
            .numberOfTrees(TREES)
            .sampleSize(SAMPLE)
            .randomSeed(2026L)
            .build();

        // Warm.
        for (int i = 0; i < warmEnd; i++) forest.update(emb[i]);

        // Score (frozen baseline).
        double[] scores = new double[embedLen - warmEnd];
        int[] evalLabels = new int[embedLen - warmEnd];
        long pos = 0;
        for (int i = warmEnd; i < embedLen; i++) {
            scores[i - warmEnd] = forest.getAnomalyScore(emb[i]);
            evalLabels[i - warmEnd] = labels[i];
            if (labels[i] == 1) pos++;
        }
        double auc = auc(scores, evalLabels);
        return new double[]{auc, pos};
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

    // Minimal JSON parser sufficient for NAB's combined_windows.json shape.
    // Handles `{ "path": [[ts, ts], ...], ... }` only.
    static TreeMap<String, List<String[]>> parseWindows(String text) {
        TreeMap<String, List<String[]>> out = new TreeMap<>();
        int i = 0;
        int n = text.length();
        while (i < n) {
            while (i < n && text.charAt(i) != '"') i++;
            if (i >= n) break;
            int keyStart = ++i;
            while (i < n && text.charAt(i) != '"') i++;
            String key = text.substring(keyStart, i++);
            while (i < n && text.charAt(i) != ':') i++;
            i++; // skip ':'
            while (i < n && text.charAt(i) != '[') i++;
            int depth = 0;
            int arrStart = i;
            while (i < n) {
                char c = text.charAt(i);
                if (c == '[') depth++;
                else if (c == ']') {
                    depth--;
                    if (depth == 0) { i++; break; }
                }
                i++;
            }
            List<String[]> windows = parseArrayOfPairs(text.substring(arrStart, i));
            out.put(key, windows);
        }
        return out;
    }

    static List<String[]> parseArrayOfPairs(String arr) {
        List<String[]> out = new ArrayList<>();
        int i = 0;
        int n = arr.length();
        while (i < n) {
            while (i < n && arr.charAt(i) != '[') i++;
            if (i >= n) break;
            if (i + 1 < n && arr.charAt(i + 1) == '[') { i++; continue; }
            // Inner [ at position i. Check we actually have content.
            i++; // skip [
            if (i >= n || arr.charAt(i) == ']') { continue; }
            String[] pair = new String[2];
            boolean ok = true;
            for (int p = 0; p < 2; p++) {
                while (i < n && arr.charAt(i) != '"') i++;
                if (i >= n) { ok = false; break; }
                int s = ++i;
                while (i < n && arr.charAt(i) != '"') i++;
                if (i >= n) { ok = false; break; }
                int end = Math.min(i, s + 19);
                pair[p] = arr.substring(s, end);
                i++;
            }
            if (!ok) break;
            while (i < n && arr.charAt(i) != ']') i++;
            i++;
            out.add(pair);
        }
        return out;
    }

    static String readAll(File f) throws Exception {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line).append('\n');
            }
        }
        return sb.toString();
    }
}
