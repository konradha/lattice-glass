#!/usr/bin/env python3
"""Reweight aggregated PT energy histograms to target betas."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def load_hist(path):
    hist = defaultdict(lambda: defaultdict(int))
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            hist[float(row["beta"])][int(row["bin"])] += int(row["count"])
    return hist


def reweight(source, beta_source, beta_target, width):
    logs = []
    for b, n in source.items():
        if n <= 0:
            continue
        e = (b + 0.5) * width
        logs.append((b, math.log(n) - (beta_target - beta_source) * e))
    if not logs:
        return {}, None
    mx = max(x[1] for x in logs)
    weights = [(b, math.exp(l - mx)) for b, l in logs]
    z = sum(w for _, w in weights)
    probs = {b: w / z for b, w in weights}
    mean_e = sum(((b + 0.5) * width) * p for b, p in probs.items())
    return probs, mean_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="combined_hist.csv")
    ap.add_argument("--width", type=float, default=64.0)
    ap.add_argument("--targets", default="0,0.389,0.778,1.167,1.556,1.944,2.333,2.722,3.111,3.5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    hist = load_hist(args.hist)
    targets = [float(x) for x in args.targets.split(",")]
    sources = sorted(hist)
    out = Path(args.out) if args.out else Path(args.hist).with_name("reweighted_summary.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_beta", "source_beta", "mean_energy", "logZ_shift"])
        for t in targets:
            src = min(sources, key=lambda s: abs(s - t))
            probs, mean_e = reweight(hist[src], src, t, args.width)
            if mean_e is None:
                w.writerow([t, src, "", ""])
                continue
            w.writerow([t, src, f"{mean_e:.6f}", ""])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
