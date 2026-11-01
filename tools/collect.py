#!/usr/bin/env python3
"""Aggregate PT histograms and count configs from a ladder run root."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.run_root)
    combined = defaultdict(lambda: defaultdict(int))
    ladders = 0
    done = 0
    for ladder in sorted(root.glob("ladder_*")):
        hist = ladder / "hist.csv"
        if not hist.exists():
            continue
        ladders += 1
        if (ladder / "done.txt").exists() and "complete=1" in (ladder / "done.txt").read_text():
            done += 1
        with hist.open() as f:
            for row in csv.DictReader(f):
                beta = row["beta"]
                b = int(row["bin"])
                combined[beta][b] += int(row["count"])
    out = Path(args.out) if args.out else root / "combined_hist.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "bin", "count"])
        for beta in sorted(combined, key=float):
            for b in sorted(combined[beta]):
                w.writerow([beta, b, combined[beta][b]])
    print(f"ladders_with_hist={ladders} complete={done} out={out}")


if __name__ == "__main__":
    main()
