#!/usr/bin/env python3
"""Run the CPU/GPU x nonlocal/local fold for ten betas.

For each beta: 8 CPU chains and 8 M3-GPU chains run concurrently.
Modes run in sequence: nonlocal for all betas, then local for all betas.
Each backend gets one aggregate .npz and paper-style Fig. 1 output.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DRIVER = (ROOT / "solid" / "driver").resolve()
BETAS = [0.01, 0.286667, 0.563333, 0.84, 1.11667, 1.39333, 1.67, 1.94667, 2.22333, 2.5]
EQ = {
 "nonlocal": [1000,1000,1000,1000,1000,1000,2767,10773,41563,160000],
 "local":    [100,100,100,170,493,1591,6923,36324,259751,2000001],
}
SW = {
 "nonlocal": [1000,1000,1000,1000,1000,1000,2767,10773,41563,160000],
 "local":    [200,200,200,339,985,3181,13846,72648,519502,4000001],
}
CHAINS = 8

def run_one(mode: str, beta: float, idx: int, backend: str, root: Path) -> dict:
    tag = f"{backend.lower()}_{mode}_b{idx:02d}"
    wd = root / tag
    wd.mkdir(parents=True, exist_ok=True)
    cmd = [str(DRIVER), "21", f"{beta:.17g}", "0.75", "0.30", "0.45",
           str(SW[mode][idx]), str(CHAINS), backend, mode, str(EQ[mode][idx])]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=wd, text=True, capture_output=True)
    wall = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{tag} failed:\n{p.stdout}\n{p.stderr}")
    z = np.load(wd / "run.npz", allow_pickle=False)
    return {"beta": beta, "mode": mode, "backend": backend,
            "wall": wall, "npz": wd / "run.npz", "z": z,
            "equilibration": EQ[mode][idx], "sweeps": SW[mode][idx],
            "stdout": p.stdout, "stderr": p.stderr}

def aggregate(mode: str, backend: str, results: list[dict], outdir: Path) -> Path:
    out = {"betas": np.asarray([r["beta"] for r in results], np.float64),
           "equilibration_sweeps": np.asarray([r["equilibration"] for r in results], np.uint64),
           "production_sweeps": np.asarray([r["sweeps"] for r in results], np.uint64),
           "wall_seconds": np.asarray([r["wall"] for r in results], np.float64)}
    for i, r in enumerate(results):
        for key in ("initial", "equilibrated", "final", "snapshots",
                    "snapshot_sweeps", "energy", "accept_local",
                    "accept_nonlocal", "scalars", "meta", "seed",
                    "history_every", "history_sweeps", "start_sweep"):
            if key in r["z"].files:
                out[f"beta_{i}_{key}"] = r["z"][key]
    path = outdir / f"{backend.lower()}_{mode}.npz"
    np.savez_compressed(path, **out)
    return path

def main() -> int:
    outdir = ROOT / "campaigns"
    outdir.mkdir(parents=True, exist_ok=True)
    log = open(outdir / "fold_timing.jsonl", "w")
    for mode in ("nonlocal", "local"):
        results = {"CPU": [], "GPU": []}
        for idx, beta in enumerate(BETAS):
            print(f"=== {mode} beta[{idx}]={beta:.6g} CPU+GPU concurrent ===", flush=True)
            # CPU and GPU processes per beta, 8 chains each, concurrently.
            procs = {}
            for backend in ("CPU", "GPU"):
                tag = f"{backend.lower()}_{mode}_b{idx:02d}"
                wd = outdir / tag
                wd.mkdir(parents=True, exist_ok=True)
                cmd = [str(DRIVER), "21", f"{beta:.17g}", "0.75", "0.30", "0.45",
                       str(SW[mode][idx]), str(CHAINS), backend, mode, str(EQ[mode][idx])]
                procs[backend] = (subprocess.Popen(cmd, cwd=wd, text=True,
                                                  stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE),
                                  wd, time.time())
            for backend, (p, wd, t0) in procs.items():
                stdout, stderr = p.communicate()
                wall = time.time() - t0
                if p.returncode != 0:
                    raise RuntimeError(f"{backend} {mode} beta={beta} failed:\n{stdout}\n{stderr}")
                z = np.load(wd / "run.npz", allow_pickle=False)
                results[backend].append({"beta": beta, "mode": mode, "backend": backend,
                                         "wall": wall, "npz": wd / "run.npz", "z": z,
                                         "equilibration": EQ[mode][idx],
                                         "sweeps": SW[mode][idx]})
                rec = {"mode": mode, "backend": backend, "beta": beta,
                       "equilibration": EQ[mode][idx], "sweeps": SW[mode][idx],
                       "wall": wall, "stdout": stdout.strip(), "stderr": stderr.strip()}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"  {backend} done beta={beta:.6g} wall={wall:.1f}s", flush=True)
        for backend in ("CPU", "GPU"):
            path = aggregate(mode, backend, results[backend], outdir)
            print(f"wrote {path}", flush=True)
            # Plot after each backend/mode. Fig2 is skipped because the driver
            # stores configurations, not particle trajectories.
            subprocess.run([sys.executable, str(ROOT / "solid" / "plot_paper.py"),
                            "--input", str(path),
                            "--output-dir", str(ROOT / "plots" / f"{backend.lower()}_{mode}"),
                            "--skip-fig2"], check=False)
    log.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
