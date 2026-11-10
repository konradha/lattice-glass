#!/usr/bin/env python3
"""CPU/GPU bitwise correctness check for the solid driver.

Each case runs CPU first. The GPU run must match every saved field.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(ROOT, "driver")
KEYS = [
    "initial", "equilibrated", "final", "snapshot_sweeps", "snapshots",
    "energy", "accept_local", "accept_nonlocal", "scalars", "meta",
]


def run_case(L, sweeps, cores, mode, equil, beta=2.0):
    args = [DRIVER, str(L), str(beta), "0.75", "0.30", "0.45",
            str(sweeps), str(cores), "CPU", mode, str(equil)]
    work = tempfile.mkdtemp(prefix="solid-check-")
    try:
        subprocess.run(args, cwd=work, check=True, stdout=subprocess.DEVNULL)
        cpu_path = os.path.join(work, "cpu.npz")
        shutil.copy(os.path.join(work, "run.npz"), cpu_path)
        args[8] = "GPU"
        subprocess.run(args, cwd=work, check=True, stdout=subprocess.DEVNULL)
        gpu_path = os.path.join(work, "gpu.npz")
        shutil.copy(os.path.join(work, "run.npz"), gpu_path)
        with np.load(cpu_path) as cpu, np.load(gpu_path) as gpu:
            bad = [k for k in KEYS if not np.array_equal(cpu[k], gpu[k])]
        return bad
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", action="store_true",
                        help="include the 1000-sweep L21 cases")
    args = parser.parse_args()
    cases = [
        (4, 17, 2, "both", 0),
        (4, 17, 2, "both", 5),
        (21, 17, 2, "nonlocal", 0),
        (21, 17, 2, "local", 0),
        (21, 17, 8, "both", 5),
    ]
    if args.long:
        cases += [
            (21, 1000, 8, "nonlocal", 20),
            (21, 1000, 8, "local", 20),
        ]
    failed = False
    for L, sweeps, cores, mode, equil in cases:
        bad = run_case(L, sweeps, cores, mode, equil)
        status = "OK" if not bad else "FAIL: " + ",".join(bad)
        print(f"L={L} sweeps={sweeps} chains={cores} mode={mode} equil={equil}: {status}")
        failed = failed or bool(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
