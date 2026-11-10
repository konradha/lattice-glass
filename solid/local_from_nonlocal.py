#!/usr/bin/env python3
"""Seed local CPU campaigns from the final configurations of the nonlocal run."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DRIVER = (ROOT / "solid" / "driver").resolve()
SRC = ROOT / "campaigns_cpu" / "nonlocal_energy.npz"
OUT = ROOT / "campaigns_cpu" / "local_from_nonlocal.npz"
WORK = ROOT / "campaigns_cpu" / "local_from_nonlocal"
PLOTS = ROOT / "plots_cpu" / "local_from_nonlocal"
BETAS = [0.01,0.242222,0.474444,0.706667,0.938889,1.17111,1.40333,1.63556,1.86778,2.1]
# 20 * local tau, interpolated from the measured table; no local equilibration.
SWEEPS = [200,200,200,210,482,1240,3319,11265,45285,209128]
CHAINS = 8

def aggregate(results):
    out = {"betas": np.asarray([r["beta"] for r in results], np.float64),
           "production_sweeps": np.asarray([r["sweeps"] for r in results], np.uint64),
           "wall_seconds": np.asarray([r["wall"] for r in results], np.float64)}
    for i, r in enumerate(results):
        z = np.load(r["npz"], allow_pickle=False)
        for key in ("initial","equilibrated","final","snapshots","snapshot_sweeps",
                    "energy","accept_local","accept_nonlocal","scalars","meta",
                    "seed","history_every","history_sweeps"):
            if key in z.files:
                out[f"beta_{i}_{key}"] = z[key]
    np.savez_compressed(OUT, **out)

def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    if not DRIVER.exists():
        raise SystemExit(f"missing {DRIVER}; run make -C solid")
    z = np.load(SRC, allow_pickle=False)
    WORK.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (beta, sw) in enumerate(zip(BETAS, SWEEPS)):
        wd = WORK / f"b{i:02d}"
        wd.mkdir(parents=True, exist_ok=True)
        final = np.ascontiguousarray(z[f"beta_{i}_final"], dtype=np.uint8)
        if final.shape[0] != CHAINS:
            raise SystemExit(f"beta {beta}: expected {CHAINS} chains, got {final.shape}")
        seed = wd / "state.raw"
        seed.write_bytes(final.tobytes())
        cmd = [str(DRIVER), "21", f"{beta:.17g}", "0.75", "0.30", "0.45",
               str(sw), str(CHAINS), "CPU", "local", "0", str(seed)]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=wd, text=True, capture_output=True)
        wall = time.time() - t0
        if p.returncode != 0:
            raise SystemExit(f"beta {beta} failed:\n{p.stdout}\n{p.stderr}")
        print(f"local beta={beta:.6g} sweeps={sw} wall={wall:.1f}s", flush=True)
        results.append({"beta": beta, "sweeps": sw, "wall": wall,
                        "npz": wd / "run.npz"})
    aggregate(results)
    (OUT.with_suffix(".json")).write_text(json.dumps(
        [{"beta": r["beta"], "sweeps": r["sweeps"], "wall": r["wall"]} for r in results], indent=2))
    print("wrote", OUT)
    subprocess.run([sys.executable, str(ROOT/"solid"/"plot_paper.py"),
                    "--input", str(OUT), "--output-dir", str(PLOTS),
                    "--skip-fig2"], check=False)

if __name__ == "__main__":
    main()
