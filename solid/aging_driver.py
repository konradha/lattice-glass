#!/usr/bin/env python3
"""Orchestrate solid/driver for temperature ladders and aggregate NPZ output.

Backend contract: this script calls the C++/metal-cpp executable

    solid/driver L beta rho rho1 rho2 NUM_SWEEPS NUM_CORES CPU|GPU [mode] [equil]

where mode is both|nonlocal|local and equil is the number of equilibration sweeps
before the production snapshot schedule. The C++ driver must support the
equilibration argument and writes run.npz in its working directory.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

Float = np.float64


# ---------------------------------------------------------------------------
# Provided tau / VFT helpers
# ---------------------------------------------------------------------------
def _gap(beta: ArrayLike, beta0: float) -> NDArray[Float]:
    b = np.asarray(beta, dtype=Float)
    if not np.all(np.isfinite(b)):
        raise ValueError("beta must be finite")
    if np.any(b >= beta0):
        raise ValueError(f"beta must be < beta0={beta0}")
    return beta0 - b


def _positive(name: str, value: ArrayLike) -> NDArray[Float]:
    v = np.asarray(value, dtype=Float)
    if np.any(v <= 0.0) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be finite and > 0")
    return v


def log_tau_local(beta, log_tau0, D, beta0):
    gap = _gap(beta, beta0)
    return log_tau0 + D * (beta0 - gap) / gap


def tau_local(beta, log_tau0, D, beta0):
    return np.exp(log_tau_local(beta, log_tau0, D, beta0))


def xi(beta, xi0, beta0, nu=0.68):
    gap = _gap(beta, beta0)
    _positive("xi0", xi0)
    _positive("nu", nu)
    return xi0 * gap ** (-nu)


def phi(x):
    return -np.expm1(-np.asarray(x, dtype=Float))


def log_tau_nonlocal(beta, L, log_tau0, D, beta0, xi0, nu=0.68):
    gap = _gap(beta, beta0)
    l = _positive("L", L)
    _positive("xi0", xi0)
    _positive("nu", nu)
    x = l * gap ** nu / xi0
    return log_tau0 + D * (beta0 - gap) / gap * phi(x)


def tau_nonlocal(beta, L, log_tau0, D, beta0, xi0, nu=0.68):
    return np.exp(log_tau_nonlocal(beta, L, log_tau0, D, beta0, xi0, nu))


def log_speedup(beta, L, local, nonlocal_, nu=0.68):
    return log_tau_local(beta, *local) - log_tau_nonlocal(beta, L, *nonlocal_, nu=nu)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_csv(s: str) -> list[float]:
    return [float(x) for x in s.replace(",", " ").split()]


def parse_tau_local(s: str) -> tuple[float, float, float]:
    v = parse_csv(s)
    if len(v) != 3:
        raise argparse.ArgumentTypeError("need log_tau0,D,beta0")
    return tuple(v)  # type: ignore


def parse_tau_nonlocal(s: str) -> tuple[float, float, float, float, float]:
    v = parse_csv(s)
    if len(v) != 5:
        raise argparse.ArgumentTypeError("need log_tau0,D,beta0,xi0,nu")
    return tuple(v)  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--betas", type=parse_csv, required=True, help="space/comma separated beta values")
    p.add_argument("--L", type=int, default=21)
    p.add_argument("--rho", type=float, default=0.75)
    p.add_argument("--rho1", type=float, default=0.30)
    p.add_argument("--rho2", type=float, default=0.45)
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--backend", choices=["CPU", "GPU"], default="GPU")
    p.add_argument("--mode", choices=["both", "nonlocal", "local"], default="nonlocal")
    p.add_argument("--sweeps", default="auto", help="production sweeps per chain, or auto")
    p.add_argument("--equilibration", default="auto", help="equilibration sweeps per chain, or auto")
    p.add_argument("--eq-factor", type=float, default=20.0, help="auto equilibration = factor*tau")
    p.add_argument("--prod-factor", type=float, default=50.0, help="auto production = factor*tau")
    p.add_argument("--tau-local", type=parse_tau_local, default=None, metavar="log_tau0,D,beta0")
    p.add_argument("--tau-nonlocal", type=parse_tau_nonlocal, default=None,
                   metavar="log_tau0,D,beta0,xi0,nu")
    p.add_argument("--proposal-ns", type=float, default=None,
                   help="optional measured ns/proposal for a wall-time estimate")
    p.add_argument("--output", type=Path, default=Path("aging_data.npz"))
    p.add_argument("--workdir", type=Path, default=Path("solid_runs"))
    p.add_argument("--driver", type=Path, default=Path(__file__).with_name("driver"))
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p


def _as_values(v, name, n):
    if isinstance(v, (int, float)):
        return [int(v)] * n
    text = str(v).strip()
    if text.lower() == "auto":
        return [None] * n
    vals = [int(x) for x in text.replace(",", " ").split()]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"{name} needs 1 or {n} values, got {len(vals)}")
    return vals


def choose_tau(beta: float, L: int, mode: str,
               local_params, nonlocal_params) -> float:
    vals = []
    if mode in ("both", "local"):
        if local_params is None:
            raise ValueError("local tau parameters are required for local/both auto estimates")
        vals.append(float(tau_local(beta, *local_params)))
    if mode in ("both", "nonlocal"):
        if nonlocal_params is None:
            raise ValueError("nonlocal tau parameters are required for nonlocal/both auto estimates")
        vals.append(float(tau_nonlocal(beta, L, *nonlocal_params)))
    return max(vals) if vals else 1.0


def plan_runs(args) -> list[dict]:
    plan = []
    eqs = _as_values(args.equilibration, "equilibration", len(args.betas))
    sws = _as_values(args.sweeps, "sweeps", len(args.betas))
    for beta, eq, sw in zip(args.betas, eqs, sws):
        tau = float("nan")
        if eq is None or sw is None:
            tau = choose_tau(beta, args.L, args.mode, args.tau_local, args.tau_nonlocal)
        if eq is None:
            eq = max(1, int(math.ceil(args.eq_factor * tau)))
        if sw is None:
            sw = max(1, int(math.ceil(args.prod_factor * tau)))
        attempts_per_sweep = args.L ** 3 * (2 if args.mode == "both" else 1)
        est_s = None
        if args.proposal_ns is not None:
            est_s = (eq + sw) * attempts_per_sweep * args.proposal_ns * 1e-9
        plan.append({"beta": beta, "tau": tau, "equilibration": eq,
                     "sweeps": sw, "estimated_seconds": est_s})
    return plan


def run_one(args, item: dict, root: Path) -> dict:
    beta = item["beta"]
    tag = f"beta_{beta:g}"
    wd = root / tag
    out_npz = wd / "run.npz"
    wd.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and out_npz.exists():
        return {**item, "npz": out_npz, "wall": 0.0, "status": "skipped"}
    cmd = [
        str(args.driver), str(args.L), f"{beta:.17g}", f"{args.rho:.17g}",
        f"{args.rho1:.17g}", f"{args.rho2:.17g}", str(item["sweeps"]),
        str(args.chains), args.backend, args.mode, str(item["equilibration"]),
    ]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=wd, text=True, capture_output=True)
    wall = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{tag} failed ({p.returncode})\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")
    if not out_npz.exists():
        raise RuntimeError(f"{tag}: driver produced no {out_npz}")
    return {**item, "npz": out_npz, "wall": wall, "status": "done",
            "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def aggregate(results: Sequence[dict], args) -> None:
    out = {}
    out["betas"] = np.asarray([r["beta"] for r in results], dtype=np.float64)
    out["equilibration_sweeps"] = np.asarray([r["equilibration"] for r in results], dtype=np.uint64)
    out["production_sweeps"] = np.asarray([r["sweeps"] for r in results], dtype=np.uint64)
    out["estimated_seconds"] = np.asarray(
        [r["estimated_seconds"] if r["estimated_seconds"] is not None else np.nan for r in results],
        dtype=np.float64)
    out["wall_seconds"] = np.asarray([r["wall"] for r in results], dtype=np.float64)
    for i, r in enumerate(results):
        z = np.load(r["npz"])
        for key in ("initial", "equilibrated", "final", "snapshots",
                    "snapshot_sweeps", "energy", "accept_local",
                    "accept_nonlocal", "scalars", "meta"):
            if key in z.files:
                out[f"beta_{i}_{key}"] = z[key]
    np.savez_compressed(args.output, **out)

    csv = args.output.with_suffix(".csv")
    with csv.open("w") as f:
        f.write("beta,equilibration_sweeps,production_sweeps,tau,estimated_seconds,wall_seconds,status\n")
        for r in results:
            tau = float(r["tau"])
            est = "" if r["estimated_seconds"] is None else format(float(r["estimated_seconds"]), ".6f")
            f.write(f"{float(r['beta']):.17g},{int(r['equilibration'])},{int(r['sweeps'])},"
                    f"{tau:.17g},{est},"
                    f"{float(r['wall']):.6f},{r['status']}\n")
    manifest = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "results": [{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
                    for r in results],
        "output": str(args.output),
        "csv": str(csv),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.driver = args.driver.resolve()
    if not (0.0 <= args.rho <= 1.0 and 0.0 <= args.rho1 <= 1.0 and 0.0 <= args.rho2 <= 1.0):
        raise SystemExit("rho, rho1, rho2 must be in [0,1]")
    if abs(args.rho1 + args.rho2 - args.rho) > 1e-8:
        print(f"WARNING: rho1+rho2={args.rho1 + args.rho2:g} != rho={args.rho:g}", file=sys.stderr)
    if not args.driver.exists():
        raise SystemExit(f"driver not found: {args.driver} (run make -C solid)")
    plan = plan_runs(args)
    print("beta\teq_sweeps\tprod_sweeps\ttau\test_seconds")
    for r in plan:
        est = "" if r["estimated_seconds"] is None else format(r["estimated_seconds"], ".3f")
        print(f"{r['beta']:.17g}\t{r['equilibration']}\t{r['sweeps']}\t"
              f"{r['tau']:.6g}\t{est}")
    if args.estimate_only:
        return 0
    args.workdir.mkdir(parents=True, exist_ok=True)
    results = []
    max_par = max(1, args.max_parallel)
    if args.backend == "GPU":
        max_par = 1  # one Metal process at a time; chains are inside each process
    with cf.ThreadPoolExecutor(max_workers=max_par) as ex:
        futures = [ex.submit(run_one, args, item, args.workdir) for item in plan]
        for fut in cf.as_completed(futures):
            r = fut.result()
            print(f"finished beta={r['beta']:.17g} wall={r['wall']:.1f}s status={r['status']}")
            results.append(r)
    results.sort(key=lambda r: r["beta"])
    aggregate(results, args)
    print(f"wrote {args.output} and {args.output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
