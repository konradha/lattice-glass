#!/usr/bin/env python3
"""Bitwise CPU/Metal equivalence gate for the solid sampler.

The two backends must reproduce every saved field exactly (initial,
equilibrated and final states, snapshots, energy and acceptance histories,
scalars, metadata, seed, decimation fields). Use this gate for implementation
changes: compiler flags, memory layouts, fast paths, precision-preserving
refactors. It assumes the CPU and Metal paths realize the identical move
sequence from the same RNG stream, which is what the baseline does.

For algorithmic update-order changes (colored/checkerboard schedules) use
statistical.py instead; see README.md for the change-class table. Because the
Metal local phase at L=21 is the 21-colour edge schedule, --schedule colored
(the default) compares only schedule-identical paths: L<21 local/both and
nonlocal at any L. --schedule reference restores the pre-coloured case set.

Driver resolution, first hit wins:
  CPU   --cpu-driver, $SOLID_CPU_DRIVER, $SOLID_DRIVER,
        <solid-opt2>/cpu/driver, <solid-opt2>/metal/driver, ./driver
  Metal --metal-driver, $SOLID_METAL_DRIVER, $SOLID_DRIVER,
        <solid-opt2>/metal/driver, ./driver

The CPU run is executed by the CPU binary and the GPU run by the Metal binary.
A binary built without Metal silently falls back to the CPU; that case is
detected and rejected so the gate cannot pass by comparing CPU with itself.

Exit codes: 0 all cases equal, 1 field mismatch, 2 setup/driver error.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

GATE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLID_DIR = os.path.dirname(GATE_DIR)

KEYS = [
    "initial", "equilibrated", "final", "snapshot_sweeps", "snapshots",
    "energy", "accept_local", "accept_nonlocal", "scalars", "meta",
    "seed", "history_every", "history_sweeps",
]

BETA = "2.0"
RHO = ("0.75", "0.30", "0.45")


class GateError(RuntimeError):
    pass


def _executable(path, label):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise GateError(f"{label}: no such file: {path}")
    if not os.access(path, os.X_OK):
        raise GateError(f"{label}: not executable: {path}")
    return path


def _driver_candidates():
    return {
        "CPU": [
            os.path.join(SOLID_DIR, "cpu", "driver"),
            os.path.join(SOLID_DIR, "metal", "driver"),
            os.path.join(GATE_DIR, "driver"),
            os.path.join(os.getcwd(), "driver"),
        ],
        "GPU": [
            os.path.join(SOLID_DIR, "metal", "driver"),
            os.path.join(GATE_DIR, "driver"),
            os.path.join(os.getcwd(), "driver"),
        ],
    }


def find_driver(backend, explicit, candidates):
    env_names = (["SOLID_CPU_DRIVER"] if backend == "CPU"
                 else ["SOLID_METAL_DRIVER"])
    env_names.append("SOLID_DRIVER")
    label = f"{backend} driver"
    if explicit:
        return _executable(explicit, label)
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return _executable(value, f"{label} ($%s)" % name)
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return os.path.abspath(path)
    tried = "\n  ".join(candidates)
    build = "make -C %s" % os.path.join(SOLID_DIR, backend.lower())
    raise GateError(
        f"no {label} found; pass --{'cpu' if backend == 'CPU' else 'metal'}-driver "
        f"or set ${'SOLID_CPU_DRIVER' if backend == 'CPU' else 'SOLID_METAL_DRIVER'}.\n"
        f"tried:\n  {tried}\nbuild with: {build}")


def _usage(driver, backend):
    return [driver, "4", BETA, RHO[0], RHO[1], RHO[2], "1", "1", backend,
            "both", "0"]


def probe(driver, backend, timeout):
    """Return (ok, message). Rejects CPU fallback instead of passing silently."""
    with tempfile.TemporaryDirectory(prefix="solid-probe-") as work:
        try:
            proc = subprocess.run(
                _usage(driver, backend), cwd=work, capture_output=True,
                text=True, timeout=timeout)
        except OSError as exc:
            return False, f"cannot execute: {exc}"
        except subprocess.TimeoutExpired:
            return False, "probe timed out"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, (tail[-1] if tail else f"exit {proc.returncode}")
        if backend == "GPU" and "using CPU" in proc.stderr:
            return False, ("binary silently falls back to CPU; rebuild the "
                           "Metal backend with: make -C %s"
                           % os.path.join(SOLID_DIR, "metal"))
        if not os.path.exists(os.path.join(work, "run.npz")):
            return False, "probe produced no run.npz"
    return True, ""


def _compare(cpu, gpu):
    bad, notes = [], []
    for key in KEYS:
        if key not in cpu.files or key not in gpu.files:
            missing = "cpu" if key not in cpu.files else "metal"
            bad.append(f"{key}(missing:{missing})")
            continue
        a, b = cpu[key], gpu[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append(f"{key}(layout)")
            notes.append(f"{key}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
            continue
        if np.array_equal(a, b):
            continue
        bad.append(key)
        if a.size and a.dtype.kind in "fiu":
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            notes.append(f"{key}: {np.count_nonzero(a != b)}/{a.size} differ, "
                         f"max|delta|={delta.max():.6g}")
    return bad, notes


def run_case(cpu_driver, metal_driver, case, seed, timeout):
    L, sweeps, cores, mode, equil = case
    base = [str(L), BETA, RHO[0], RHO[1], RHO[2], str(sweeps), str(cores),
            None, mode, str(equil), "--seed", str(seed)]
    work = tempfile.mkdtemp(prefix="solid-check-")
    saved = {}
    try:
        for backend, driver in (("CPU", cpu_driver), ("GPU", metal_driver)):
            args = list(base)
            args[7] = backend
            try:
                proc = subprocess.run([driver] + args, cwd=work,
                                      capture_output=True, text=True,
                                      timeout=timeout)
            except subprocess.TimeoutExpired:
                raise GateError(f"L={L} sweeps={sweeps} {backend}: timed out "
                                f"after {timeout}s")
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                raise GateError(f"L={L} sweeps={sweeps} {backend}: exit "
                                f"{proc.returncode}: "
                                f"{tail[-1] if tail else 'no output'}")
            if backend == "GPU" and "using CPU" in proc.stderr:
                raise GateError(f"L={L} sweeps={sweeps} {backend}: Metal "
                                f"driver fell back to CPU: "
                                f"{proc.stderr.strip().splitlines()[-1]}")
            path = os.path.join(work, f"{backend}.npz")
            os.replace(os.path.join(work, "run.npz"), path)
            saved[backend] = path
        with np.load(saved["CPU"]) as cpu, np.load(saved["GPU"]) as gpu:
            return _compare(cpu, gpu)
    finally:
        for path in saved.values():
            try:
                os.unlink(path)
            except OSError:
                pass
        shutil.rmtree(work, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Bitwise CPU/Metal equivalence gate.")
    parser.add_argument("--long", action="store_true",
                        help="include the 1000-sweep long cases")
    parser.add_argument("--schedule", choices=("colored", "reference"),
                        default="colored",
                        help="reference: every path uses the reference local "
                             "scan, so the original five cases are bitwise; "
                             "colored: Metal L21 local uses the 21-colour "
                             "schedule, so only schedule-identical paths "
                             "(L<21 local/both, nonlocal) are compared")
    parser.add_argument("--cpu-driver", help="CPU driver binary")
    parser.add_argument("--metal-driver", help="Metal driver binary")
    parser.add_argument("--seed", type=int, default=1,
                        help="RNG seed shared by both backends (default 1)")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="per-run timeout in seconds (default 1800)")
    parser.add_argument("--skip-metal", action="store_true",
                        help="exit 0 with SKIP when Metal is unavailable")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    candidates = _driver_candidates()
    try:
        cpu_driver = find_driver("CPU", args.cpu_driver, candidates["CPU"])
    except GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        metal_driver = find_driver("GPU", args.metal_driver, candidates["GPU"])
    except GateError as exc:
        if args.skip_metal:
            print(f"SKIP: Metal backend unavailable: {exc}")
            return 0
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"CPU driver:   {cpu_driver}")
    print(f"Metal driver: {metal_driver}")
    for driver, backend in ((cpu_driver, "CPU"), (metal_driver, "GPU")):
        ok, why = probe(driver, backend, args.timeout)
        if not ok:
            if backend == "GPU" and args.skip_metal:
                print(f"SKIP: Metal backend unavailable: {why}")
                return 0
            print(f"error: {backend} probe failed: {why}", file=sys.stderr)
            return 2

    if args.schedule == "reference":
        # Both backends use the reference random-site/random-direction local
        # scan, so every case is bitwise comparable.
        cases = [
            (4, 17, 2, "both", 0),
            (4, 17, 2, "both", 5),
            (21, 17, 2, "nonlocal", 0),
            (21, 17, 2, "local", 0),
            (21, 17, 8, "both", 5),
        ]
        long_cases = [
            (21, 1000, 8, "nonlocal", 20),
            (21, 1000, 8, "local", 20),
        ]
    else:
        # The Metal local phase at L=21 is the 21-colour edge schedule: a
        # different trajectory from the CPU reference scan. Only schedule-
        # identical paths are bitwise comparable; statistical.py is the
        # criterion for the coloured path.
        cases = [
            (4, 17, 2, "both", 0),
            (4, 17, 2, "both", 5),
            (4, 17, 2, "local", 0),
            (21, 17, 2, "nonlocal", 0),
            (21, 17, 8, "nonlocal", 5),
        ]
        long_cases = [
            (21, 1000, 8, "nonlocal", 20),
            (4, 1000, 8, "both", 20),
        ]
    if args.long:
        cases += long_cases

    print(f"schedule: {args.schedule}" + (
        " (L=21 local phase differs by design; statistical.py is the "
        "criterion there)" if args.schedule == "colored" else
        " (requires both backends to use the reference local scan; a "
        "coloured Metal build will mismatch L=21 local/both by design)"))

    failed = False
    for case in cases:
        L, sweeps, cores, mode, equil = case
        label = (f"L={L} sweeps={sweeps} chains={cores} mode={mode} "
                 f"equil={equil}")
        try:
            bad, notes = run_case(cpu_driver, metal_driver, case, args.seed,
                                  args.timeout)
        except GateError as exc:
            print(f"{label}: ERROR: {exc}")
            return 2
        print(f"{label}: {'OK' if not bad else 'FAIL: ' + ','.join(bad)}")
        if (bad and args.schedule == "reference" and L == 21
                and mode in ("local", "both")):
            print("    note: a coloured Metal build mismatches L=21 local/both "
                  "by design; rerun with --schedule colored and use "
                  "statistical.py for that path")
        for note in notes:
            print(f"    {note}")
        failed = failed or bool(bad)
    print("exact gate: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
