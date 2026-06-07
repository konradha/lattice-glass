#!/usr/bin/env python3
"""Parallel, resumable efficiency campaign for the reference-coupled balanced
cluster move.

For each (L, beta, seed) it runs experiments/fp_efficiency with epsilon(T)
calibrated to put the disagreement fraction below the 3D site-percolation
threshold, parses the integrated autocorrelation times and ESS-per-second for
base vs base+cluster, and appends one row to a CSV. Safe to re-run: points
already present in the CSV are skipped.

Usage:
    python experiments/run_campaign.py [--workers N] [--out path] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "fp_efficiency")

# Production budget per lattice size (sweeps, warmup, sample cadence).
PER_L = {
    10: {"production": 300000, "warmup": 15000, "sample_every": 4},
    12: {"production": 200000, "warmup": 12000, "sample_every": 4},
    16: {"production": 100000, "warmup": 10000, "sample_every": 4},
}

FIELDS = [
    "L", "beta", "seed", "epsilon", "disagreement", "subperc",
    "base_qref_tau", "cluster_qref_tau",
    "base_qref_resolved", "cluster_qref_resolved",
    "base_qref_ess_s", "cluster_qref_ess_s", "speedup_qref",
    "base_energy_tau", "cluster_energy_tau", "speedup_energy",
    "base_energy_resolved", "cluster_energy_resolved",
    "cluster_accept", "cluster_mean_size",
    "base_seconds", "cluster_seconds",
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


def run_point(L: int, beta: float, seed: int, target_d: float) -> dict:
    cfg = PER_L[L]
    cmd = [
        BINARY,
        "--L", str(L),
        "--beta", str(beta),
        "--target-d", str(target_d),
        "--ref-equil-sweeps", "8000",
        "--calib-equil-sweeps", "5000",
        "--warmup-sweeps", str(cfg["warmup"]),
        "--production-sweeps", str(cfg["production"]),
        "--sample-every", str(cfg["sample_every"]),
        "--cluster-every", "1",
        "--cluster-max-size", "32",
        "--kappa", "0.5",
        "--seed", str(seed),
    ]
    started = time.time()
    out = subprocess.check_output(cmd, text=True)
    elapsed = time.time() - started
    d = parse_output(out)
    row = {
        "L": L, "beta": beta, "seed": seed,
        "epsilon": d.get("fp.epsilon"),
        "disagreement": d.get("fp.calibrated_disagreement_fraction"),
        "subperc": int(d.get("fp.subpercolating", 0)),
        "base_qref_tau": d.get("base.q_ref_tau"),
        "cluster_qref_tau": d.get("cluster.q_ref_tau"),
        "base_qref_resolved": d.get("base.q_ref_tau.resolved"),
        "cluster_qref_resolved": d.get("cluster.q_ref_tau.resolved"),
        "base_qref_ess_s": d.get("base.q_ref_ess_per_sec"),
        "cluster_qref_ess_s": d.get("cluster.q_ref_ess_per_sec"),
        "speedup_qref": d.get("speedup.q_ref_ess_per_sec"),
        "base_energy_tau": d.get("base.energy_tau"),
        "cluster_energy_tau": d.get("cluster.energy_tau"),
        "speedup_energy": d.get("speedup.energy_ess_per_sec"),
        "base_energy_resolved": d.get("base.energy_tau.resolved"),
        "cluster_energy_resolved": d.get("cluster.energy_tau.resolved"),
        "cluster_accept": d.get("cluster.cluster_accept_per_attempt"),
        "cluster_mean_size": d.get("cluster.cluster_mean_accepted_size"),
        "base_seconds": d.get("base.seconds"),
        "cluster_seconds": d.get("cluster.seconds"),
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
    ap.add_argument("--out", default=os.path.join(HERE, "campaign_results.csv"))
    ap.add_argument("--target-d", type=float, default=0.12)
    ap.add_argument("--quick", action="store_true",
                    help="tiny grid for a smoke test")
    args = ap.parse_args()

    if not os.path.exists(BINARY):
        print(f"missing binary {BINARY}; run 'make experiments/fp_efficiency'",
              file=sys.stderr)
        return 1

    points = []
    if args.quick:
        points = [(10, 3.0, 1)]
    else:
        for L, seeds in ((10, [1, 2, 3, 4]), (12, [1, 2, 3, 4]), (16, [1, 2])):
            for beta in (2.0, 2.5, 3.0, 3.5):
                for seed in seeds:
                    points.append((L, beta, seed))

    done = load_done(args.out)
    todo = [p for p in points if p not in done]
    print(f"campaign: {len(points)} points, {len(done)} done, "
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
            pool.submit(run_point, L, beta, seed, args.target_d): (L, beta, seed)
            for (L, beta, seed) in todo
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
                f"eps={row['epsilon']} D={row['disagreement']} "
                f"speedup_qref={row['speedup_qref']} "
                f"(qref resolved b/c={row['base_qref_resolved']}/"
                f"{row['cluster_qref_resolved']}) wall={wall}s",
                flush=True,
            )

    f.close()
    print("campaign done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
