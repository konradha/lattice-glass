#!/usr/bin/env python3
"""Parallel, resumable temperature scan for the informed (locally-balanced) vs
blind nonlocal occupancy swap.

For each (L, beta, seed) it runs experiments/informed_swap_efficiency, parses
occupancy-move acceptance, integrated autocorrelation times (with Madras-Sokal
resolved flags) and ESS-per-second for the energy and pure-occupancy bond
observables, and appends one row to a CSV. The production budget per beta is
scaled so tau resolves where feasible; deep-T points may stay unresolved and are
flagged honestly rather than trusted. Safe to re-run: points already present in
the CSV are skipped.

Usage:
    python experiments/run_informed_sweep.py [--workers N] [--out path] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "informed_swap_efficiency")

# Production budget per beta (sweeps, sample cadence). Higher beta = larger tau,
# so more sweeps; sample_every keeps the sample count ~15k (the IAT estimator is
# O(n^2)). Resolution needs production > ~50 * tau_sweeps, flagged per row.
PER_BETA = {
    1.0: {"production": 60000, "sample_every": 4},
    1.5: {"production": 80000, "sample_every": 5},
    2.0: {"production": 120000, "sample_every": 8},
    2.5: {"production": 180000, "sample_every": 12},
    3.0: {"production": 280000, "sample_every": 18},
    3.5: {"production": 400000, "sample_every": 26},
}

# Resolve-or-bust budgets for the hard regime, where blind tau is huge and only
# very long runs satisfy production > ~50 * tau_sweeps for the blind arm.
PER_BETA_HARD = {
    2.0: {"production": 120000, "sample_every": 8},
    2.5: {"production": 700000, "sample_every": 32},
    3.0: {"production": 1200000, "sample_every": 56},
}

FIELDS = [
    "L", "beta", "T", "seed",
    "blind_acc", "informed_acc", "acc_ratio",
    "fullswap_bond_tau", "fullswap_bond_res",
    "blind_bond_tau", "blind_bond_res",
    "informed_bond_tau", "informed_bond_res",
    "blind_energy_tau", "blind_energy_res",
    "informed_energy_tau", "informed_energy_res",
    "speedup_bond_per_sweep", "speedup_energy_per_sweep",
    "speedup_bond_ess_s", "speedup_energy_ess_s",
    "speedup_inf_vs_full_bond_ess_s",
    "fullswap_seconds", "blind_seconds", "informed_seconds",
]


def parse_output(text: str) -> dict:
    d = {}
    for line in text.splitlines():
        t = line.split()
        if len(t) < 2:
            continue
        key = t[0]
        try:
            d[key] = float(t[1])
        except ValueError:
            continue
        if len(t) >= 4 and t[2] == "resolved":
            d[key + ".resolved"] = int(float(t[3]))
    return d


def run_point(L: int, beta: float, seed: int, production: int,
              sample_every: int) -> dict:
    warmup = max(5000, production // 15)
    cmd = [
        BINARY,
        "--L", str(L),
        "--beta", str(beta),
        "--warmup", str(warmup),
        "--production", str(production),
        "--sample-every", str(sample_every),
        "--seed", str(seed),
    ]
    started = time.time()
    out = subprocess.check_output(cmd, text=True)
    elapsed = time.time() - started
    d = parse_output(out)
    blind_acc = d.get("blind.occ_acceptance")
    informed_acc = d.get("informed.occ_acceptance")
    acc_ratio = (informed_acc / blind_acc) if (blind_acc and blind_acc > 0) else None
    row = {
        "L": L, "beta": beta, "T": round(1.0 / beta, 4), "seed": seed,
        "blind_acc": blind_acc,
        "informed_acc": informed_acc,
        "acc_ratio": acc_ratio,
        "fullswap_bond_tau": d.get("fullswap.bond_tau"),
        "fullswap_bond_res": d.get("fullswap.bond_tau.resolved"),
        "blind_bond_tau": d.get("blind.bond_tau"),
        "blind_bond_res": d.get("blind.bond_tau.resolved"),
        "informed_bond_tau": d.get("informed.bond_tau"),
        "informed_bond_res": d.get("informed.bond_tau.resolved"),
        "blind_energy_tau": d.get("blind.energy_tau"),
        "blind_energy_res": d.get("blind.energy_tau.resolved"),
        "informed_energy_tau": d.get("informed.energy_tau"),
        "informed_energy_res": d.get("informed.energy_tau.resolved"),
        "speedup_bond_per_sweep": d.get("speedup.informed_vs_blind.bond_per_sweep"),
        "speedup_energy_per_sweep": d.get("speedup.informed_vs_blind.energy_per_sweep"),
        "speedup_bond_ess_s": d.get("speedup.informed_vs_blind.bond_ess_per_sec"),
        "speedup_energy_ess_s": d.get("speedup.informed_vs_blind.energy_ess_per_sec"),
        "speedup_inf_vs_full_bond_ess_s":
            d.get("speedup.informed_vs_fullswap.bond_ess_per_sec"),
        "fullswap_seconds": d.get("fullswap.seconds"),
        "blind_seconds": d.get("blind.seconds"),
        "informed_seconds": d.get("informed.seconds"),
        "_wall": round(elapsed, 1),
    }
    return row


def load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                done.add((int(r["L"]), float(r["beta"]), int(r["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "informed_results.csv"))
    ap.add_argument("--quick", action="store_true", help="tiny smoke grid")
    ap.add_argument("--hard", action="store_true",
                    help="resolve-or-bust hard-regime grid (L=8 beta 2.5,3.0)")
    args = ap.parse_args()

    if not os.path.exists(BINARY):
        print(f"missing binary {BINARY}; run "
              f"'make experiments/informed_swap_efficiency'", file=sys.stderr)
        return 1

    jobs = []
    if args.quick:
        c = PER_BETA[2.0]
        jobs = [(8, 2.0, 1, c["production"], c["sample_every"])]
    elif args.hard:
        # Resolve the hard window at L=8 (decisive) plus an L=10 size check.
        for beta in (2.5, 3.0):
            for seed in (1, 2):
                c = PER_BETA_HARD[beta]
                jobs.append((8, beta, seed, c["production"], c["sample_every"]))
        for seed in (1, 2):
            c = PER_BETA_HARD[2.0]
            jobs.append((10, 2.0, seed, c["production"], c["sample_every"]))
    else:
        # L=8 full scan, 4 seeds; L=10 confirmatory at the hard end, 2 seeds.
        for beta in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            for seed in (1, 2, 3, 4):
                c = PER_BETA[beta]
                jobs.append((8, beta, seed, c["production"], c["sample_every"]))
        for beta in (2.0, 2.5, 3.0):
            for seed in (1, 2):
                c = PER_BETA[beta]
                jobs.append((10, beta, seed, c["production"], c["sample_every"]))

    done = load_done(args.out)
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"informed scan: {len(jobs)} points, {len(done)} done, "
          f"{len(todo)} to run, {args.workers} workers", flush=True)

    write_header = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()
        f.flush()

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_point, L, beta, seed, prod, every): (L, beta, seed)
            for (L, beta, seed, prod, every) in todo
        }
        for fut in as_completed(futures):
            L, beta, seed = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL L={L} beta={beta} seed={seed}: {exc}", flush=True)
                continue
            wall = row.pop("_wall", None)
            writer.writerow(row)
            f.flush()
            completed += 1
            print(
                f"[{completed}/{len(todo)}] L={L} beta={beta} seed={seed} "
                f"acc b/i={row['blind_acc']:.2e}/{row['informed_acc']:.2e} "
                f"(x{row['acc_ratio']:.1f}) "
                f"bond/sweep={row['speedup_bond_per_sweep']:.2f} "
                f"bond/s={row['speedup_bond_ess_s']:.2f} "
                f"res b/i={row['blind_bond_res']}/{row['informed_bond_res']} "
                f"wall={wall}s",
                flush=True,
            )

    f.close()
    print("informed scan done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
