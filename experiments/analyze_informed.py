#!/usr/bin/env python3
"""Aggregate the informed-vs-blind occupancy-swap scan into a per-(L, beta)
median table with tau-resolution fractions.

Reads experiments/informed_results.csv (produced by run_informed_sweep.py) and
prints, per lattice size and temperature, the median occupancy-move acceptance
ratio, the median per-sweep and per-second bond/energy speedups, and what
fraction of seeds had a Madras-Sokal-resolved bond autocorrelation. Only the
resolved per-sweep numbers should be trusted; the table marks them.

Usage:
    python experiments/analyze_informed.py [--csv path]
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def frac(xs):
    xs = [x for x in xs if x is not None]
    return (sum(1 for x in xs if x >= 0.5) / len(xs)) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "informed_results.csv"))
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"missing {args.csv}")
        return 1

    groups = defaultdict(list)
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            groups[(int(r["L"]), float(r["beta"]))].append(r)

    hdr = (f"{'L':>3} {'beta':>5} {'T':>6} {'n':>2} "
           f"{'blind_acc':>10} {'inf_acc':>9} {'accx':>7} "
           f"{'bond/swp':>9} {'bond/s':>7} {'enr/s':>7} "
           f"{'res b/i':>8} {'inf<full':>8}")
    print(hdr)
    print("-" * len(hdr))
    for (L, beta) in sorted(groups):
        rows = groups[(L, beta)]
        T = round(1.0 / beta, 3)
        n = len(rows)
        blind_acc = med([fnum(r["blind_acc"]) for r in rows])
        inf_acc = med([fnum(r["informed_acc"]) for r in rows])
        accx = med([fnum(r["acc_ratio"]) for r in rows])
        bond_swp = med([fnum(r["speedup_bond_per_sweep"]) for r in rows])
        bond_s = med([fnum(r["speedup_bond_ess_s"]) for r in rows])
        enr_s = med([fnum(r["speedup_energy_ess_s"]) for r in rows])
        res_b = frac([fnum(r["blind_bond_res"]) for r in rows])
        res_i = frac([fnum(r["informed_bond_res"]) for r in rows])
        inf_full = med([fnum(r["speedup_inf_vs_full_bond_ess_s"]) for r in rows])

        def fmt(x, p=2):
            return f"{x:.{p}f}" if x is not None else "  -  "

        print(f"{L:>3} {beta:>5.2f} {T:>6.3f} {n:>2} "
              f"{fmt(blind_acc,4):>10} {fmt(inf_acc,4):>9} {fmt(accx,1):>7} "
              f"{fmt(bond_swp,2):>9} {fmt(bond_s,2):>7} {fmt(enr_s,2):>7} "
              f"{fmt(res_b,1)}/{fmt(res_i,1):>3} {fmt(inf_full,2):>8}")

    print("\nLegend: accx = informed/blind acceptance; bond/swp = per-sweep bond "
          "tau speedup (informed accelerates occupancy decorrelation by this "
          "factor); bond/s, enr/s = ESS-per-second speedup vs blind; res b/i = "
          "fraction of seeds with resolved bond tau (blind/informed) -- trust "
          "per-sweep numbers only where ~1.0; inf<full = informed bond ESS/s as "
          "a fraction of the trusted full nonlocal swap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
