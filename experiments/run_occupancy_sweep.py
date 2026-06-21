#!/usr/bin/env python3
"""Parallel, resumable benchmark of occupancy samplers.

For each (beta, stride, seed) it runs experiments/occupancy_efficiency, which
compares, at matched wall-clock and on the same NH model:
    swap            : non-local random-pair swap (baseline)
    swapliftedswap  : swap + skew-DB lifted nonlocal swap (stride)
    swaplifted      : swap + lifted local vacancy chain
parsing integrated-autocorrelation times, ESS-per-second speedups, acceptance,
and the Rao-Blackwell measurement gain. Resumable via the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "occupancy_efficiency")

L = 10
PRODUCTION = 80000
WARMUP = 15000
SAMPLE_EVERY = 20

FIELDS = [
    "L", "beta", "stride", "seed",
    "swap_tau", "swap_res",
    "liftedswap_tau", "liftedswap_res",
    "lifted_tau", "lifted_res",
    "liftedswap_speedup", "lifted_speedup",
    "liftedswap_accept", "lifted_accept",
    "rb_gain", "swap_seconds", "liftedswap_seconds",
]


def parse_output(text: str) -> dict:
    d = {}
    for line in text.splitlines():
        t = line.split()
        if len(t) < 2:
            continue
        try:
            d[t[0]] = float(t[1])
        except ValueError:
            continue
        if len(t) >= 4 and t[2] == "resolved":
            d[t[0] + ".resolved"] = int(float(t[3]))
    return d


def run_point(beta: float, stride: int, seed: int) -> dict:
    cmd = [
        BINARY, "--L", str(L), "--beta", str(beta), "--stride", str(stride),
        "--warmup", str(WARMUP), "--production", str(PRODUCTION),
        "--sample-every", str(SAMPLE_EVERY), "--seed", str(seed),
    ]
    t0 = time.time()
    out = subprocess.check_output(cmd, text=True)
    d = parse_output(out)
    return {
        "L": L, "beta": beta, "stride": stride, "seed": seed,
        "swap_tau": d.get("swap.plain_energy_tau"),
        "swap_res": d.get("swap.plain_energy_tau.resolved"),
        "liftedswap_tau": d.get("swapliftedswap.plain_energy_tau"),
        "liftedswap_res": d.get("swapliftedswap.plain_energy_tau.resolved"),
        "lifted_tau": d.get("swaplifted.plain_energy_tau"),
        "lifted_res": d.get("swaplifted.plain_energy_tau.resolved"),
        "liftedswap_speedup": d.get("speedup.liftedswap_plain_energy_ess_per_sec"),
        "lifted_speedup": d.get("speedup.lifted_plain_energy_ess_per_sec"),
        "liftedswap_accept": d.get("swapliftedswap.chain_accept_per_attempt"),
        "lifted_accept": d.get("swaplifted.chain_accept_per_attempt"),
        "rb_gain": d.get("rb_measurement_gain.ess_per_sec"),
        "swap_seconds": d.get("swap.seconds"),
        "liftedswap_seconds": d.get("swapliftedswap.seconds"),
        "_wall": round(time.time() - t0, 1),
    }


def load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                done.add((float(r["beta"]), int(r["stride"]), int(r["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "occupancy_results.csv"))
    args = ap.parse_args()

    points = [
        (beta, stride, seed)
        for beta in (2.0, 2.5, 3.0)
        for stride in (4, 8)
        for seed in (1, 2, 3, 4)
    ]
    done = load_done(args.out)
    todo = [p for p in points if p not in done]
    print(f"occupancy sweep: {len(points)} points, {len(done)} done, "
          f"{len(todo)} to run, {args.workers} workers", flush=True)

    write_header = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()
        f.flush()

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_point, *p): p for p in todo}
        for fut in as_completed(futures):
            beta, stride, seed = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL beta={beta} stride={stride} seed={seed}: {exc}",
                      flush=True)
                continue
            wall = row.pop("_wall", None)
            writer.writerow(row)
            f.flush()
            completed += 1
            print(
                f"[{completed}/{len(todo)}] beta={beta} stride={stride} "
                f"seed={seed} liftedswap_speedup={row['liftedswap_speedup']} "
                f"(res s/ls={row['swap_res']}/{row['liftedswap_res']}) "
                f"accept={row['liftedswap_accept']} wall={wall}s",
                flush=True,
            )

    f.close()
    print("occupancy sweep done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
