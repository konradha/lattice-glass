#!/usr/bin/env python3
"""Build, run, inspect, or launch the preserved solid backends."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
U64 = (1 << 64) - 1
BACKENDS = ("cpu", "metal", "cuda")
MODES = ("nonlocal", "local", "both", "local-colored")
PRESETS = ("auto", "depth", "throughput", "ampere")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def uint64(value):
    text = str(value)
    if not text.isascii() or not text.isdecimal() or not 0 <= int(text) <= U64:
        raise argparse.ArgumentTypeError("expected a decimal uint64")
    return int(text)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def resolved_preset(preset="depth", gpu=None):
    require(preset in PRESETS, "unknown CUDA preset")
    if preset == "auto":
        return "ampere" if gpu and "A100" in gpu.upper() else "depth"
    return preset


def driver_path(backend, mode="nonlocal", preset="depth", gpu=None):
    require(backend in BACKENDS and mode in MODES, "unknown backend or move")
    require(mode != "local-colored" or backend == "metal", "local-colored is Metal-only and is NOT physical dynamics")
    if backend == "cpu":
        return ROOT / "cpu" / "driver"
    if backend == "metal":
        return ROOT / "metal" / ("driver_persistent" if mode in ("nonlocal", "local-colored") else "driver_physical")
    return ROOT / "cuda" / ("driver-" + resolved_preset(preset, gpu))


def build_command(backend, mode="nonlocal", preset="depth", gpu=None):
    target = driver_path(backend, mode, preset, gpu)
    return ["make", "-C", str(target.parent), target.name]


def validate_run(params, backend):
    L, chains = int(params["L"]), int(params["chains"])
    require(4 <= L <= 21, "packaged shared-memory backends support 4 <= L <= 21")
    require(1 <= chains <= (1 << 31) - 1, "chains must be a positive int32")
    mode = params.get("mode", "nonlocal")
    require(mode in MODES, "unknown move mode")
    require(mode != "local-colored" or (backend == "metal" and L == 21), "local-colored requires Metal and L=21; it is not physical local dynamics")
    try:
        beta = Decimal(str(params["beta"]))
        rho, rho1, rho2 = (Decimal(str(params.get(k, v))) for k, v in (("rho", .75), ("rho1", .30), ("rho2", .45)))
    except InvalidOperation as exc:
        raise ValueError("invalid beta or composition") from exc
    require(beta.is_finite() and beta > 0, "beta must be finite and positive (the inherited beta=0 EFMA edge is not qualified)")
    require(all(x.is_finite() and 0 <= x <= 1 for x in (rho, rho1, rho2)), "densities must be finite in [0,1]")
    require(abs(rho - rho1 - rho2) <= Decimal("1e-8"), "rho must equal rho1+rho2")
    N = L ** 3
    n1, n2 = (int(x * N + Decimal("0.5")) for x in (rho1, rho2))
    require(0 < n1 + n2 <= N, "invalid rounded particle counts")
    for key, default in (("seed", 1), ("start_sweep", 0), ("sweeps", 0), ("equil_sweeps", 0), ("history_every", 1), ("max_history_samples", 100000)):
        value = params.get(key, default)
        require(type(value) is int and 0 <= value <= U64, f"{key} must be uint64")
    require(params.get("history_every", 1) > 0 and params.get("max_history_samples", 100000) > 0, "history bounds must be positive")
    next_sweep = params.get("start_sweep", 0) + params.get("equil_sweeps", 0) + params.get("sweeps", 0)
    factor = 3 if mode == "local-colored" else 2
    require(next_sweep <= U64 // (factor * N), "absolute proposal counter would wrap")
    if backend == "metal" and mode in ("local", "both"):
        require(next_sweep <= 1 << 32, "preserved physical Metal kernel supports next_sweep <= 2^32")
    return params


def native_command(params, backend, native_driver=None):
    validate_run(params, backend)
    mode = params.get("mode", "nonlocal")
    binary = Path(native_driver) if native_driver else driver_path(backend, mode, params.get("preset", "depth"))
    token = {"cpu": "CPU", "metal": "GPU", "cuda": "CUDA"}[backend]
    command = [str(binary.resolve()), str(params["L"]), str(params["beta"]),
               str(params.get("rho", .75)), str(params.get("rho1", .30)), str(params.get("rho2", .45)),
               str(params.get("sweeps", 0)), str(params["chains"]), token, mode,
               str(params.get("equil_sweeps", 0))]
    if params.get("state"):
        command.append(str(Path(params["state"]).resolve()))
    for key, default in (("seed", 1), ("start_sweep", 0), ("history_every", 1), ("max_history_samples", 100000)):
        command.extend(["--" + key.replace("_", "-"), str(params.get(key, default))])
    if backend == "cuda":
        for key, default in (("window", 256), ("threads", 256), ("ppt", 1), ("profile", 0)):
            command.extend(["--cuda-" + key, str(params.get(key, default))])
    return command


def build(backend, mode="nonlocal", preset="depth", dry_run=False):
    command = build_command(backend, mode, preset)
    if dry_run:
        return {"command": command, "binary": str(driver_path(backend, mode, preset))}
    require(backend != "metal" or sys.platform == "darwin", "Metal requires macOS; no silent CPU fallback")
    subprocess.run(command, check=True)
    return {"binary": str(driver_path(backend, mode, preset))}


def run_native(params, backend, output_dir, native_driver=None, timeout=None, build_if_missing=True):
    validate_run(params, backend)
    require(backend != "metal" or sys.platform == "darwin", "Metal requires macOS; no silent CPU fallback")
    command = native_command(params, backend, native_driver)
    binary = Path(command[0])
    if not binary.is_file() and native_driver is None and build_if_missing:
        build(backend, params.get("mode", "nonlocal"), params.get("preset", "depth"))
    require(binary.is_file() and os.access(binary, os.X_OK), f"native driver is not executable: {binary}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=output, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(f"native driver exited {result.returncode}\n{result.stdout}\n{result.stderr}")
    require((output / "run.npz").is_file(), "native driver produced no run.npz")
    record = {"schema": "solid-final.run.v1", "backend": backend, "mode": params.get("mode", "nonlocal"),
              "physical_local": params.get("mode") != "local-colored", "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in params.items()},
              "native_driver_sha256": sha256(binary), "argv": command, "wall_including_process_and_npz_s": elapsed,
              "stdout": result.stdout, "stderr": result.stderr}
    atomic_json(output / "run.json", record)
    return record


def inspect_path(path):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if path.suffix == ".json":
        return json.loads(path.read_text())
    import numpy as np
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        return {"path": str(path), "shape": list(data.shape), "dtype": str(data.dtype), "bytes": int(data.nbytes)}
    with np.load(path, allow_pickle=False) as data:
        result = {"path": str(path), "arrays": {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in data.files}}
        for key in ("meta", "scalars", "seed", "start_sweep", "next_sweep", "history_every"):
            if key in data:
                result[key] = data[key].tolist()
        for key in ("beta_text", "source_hash"):
            if key in data:
                result[key] = data[key].tobytes().decode("ascii")
        if "final" in data:
            state = data["final"]
            result["final_sha256"] = hashlib.sha256(state.tobytes()).hexdigest()
            result["final_total_energy"] = state_energy(state).tolist()
        return result


def state_energy(state):
    import numpy as np
    state = np.asarray(state, dtype=np.uint8)
    counts = np.array([i.bit_count() for i in range(64)], dtype=np.int16)[state >> 2]
    types = state & 3
    target = np.where(types == 1, 3, 5)
    return np.sum(np.where(types == 0, 0, (counts - target) ** 2), axis=-1, dtype=np.int64)


def info():
    return {"schema": "solid-final.info.v1", "platform": sys.platform,
            "cpu_default": "measured Apple opt2" if sys.platform == "darwin" else "validated cached K/type-fast",
            "cuda_presets": {"depth": "C07 L40S depth", "throughput": "C02 L40S whole-card", "ampere": "C08 A100", "auto": "A100 -> ampere; otherwise depth"},
            "metal_modes": {"nonlocal": "persistent exact prefix cut", "local": "physical random-edge exact cut", "local-colored": "opt-in nonphysical 21-colour schedule"},
            "selection": {backend: json.loads((ROOT / backend / "selection.json").read_text()) if (ROOT / backend / "selection.json").is_file() else None for backend in BACKENDS},
            "benchmarks": str(ROOT / "benchmarks"), "rerun_during_packaging": False}


def add_run_options(parser):
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--mode", choices=MODES, default="nonlocal")
    parser.add_argument("--preset", choices=PRESETS, default="depth")
    parser.add_argument("--L", type=int, default=21)
    parser.add_argument("--beta", required=True, help="positive decimal text; not rounded through float")
    parser.add_argument("--rho", default="0.75")
    parser.add_argument("--rho1", default="0.30")
    parser.add_argument("--rho2", default="0.45")
    parser.add_argument("--sweeps", type=uint64, required=True)
    parser.add_argument("--equil-sweeps", type=uint64, default=0)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=uint64, default=1)
    parser.add_argument("--start-sweep", type=uint64, default=0)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--history-every", type=uint64, default=1)
    parser.add_argument("--max-history-samples", type=uint64, default=100000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-driver", type=Path)
    parser.add_argument("--dry-run", action="store_true")


def legacy(argv):
    parser = argparse.ArgumentParser(description="Preserved native positional ABI, dispatched to the selected backend")
    for name in ("L", "beta", "rho", "rho1", "rho2", "sweeps", "chains", "backend"):
        parser.add_argument(name)
    parser.add_argument("mode", nargs="?", default="both", choices=MODES)
    parser.add_argument("equil_sweeps", nargs="?", default="0")
    parser.add_argument("state", nargs="?")
    parser.add_argument("--seed", type=uint64, default=1)
    parser.add_argument("--start-sweep", type=uint64, default=0)
    parser.add_argument("--history-every", type=uint64, default=1)
    parser.add_argument("--max-history-samples", type=uint64, default=100000)
    parser.add_argument("--cuda-window", dest="window", type=int, default=256)
    parser.add_argument("--cuda-threads", dest="threads", type=int, default=256)
    parser.add_argument("--cuda-ppt", dest="ppt", type=int, default=1)
    parser.add_argument("--cuda-profile", dest="profile", type=int, default=0)
    parser.add_argument("--preset", choices=PRESETS, default="depth")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    backend = {"CPU": "cpu", "GPU": "metal", "CUDA": "cuda"}.get(args.backend.upper())
    require(backend is not None, "backend token must be CPU, GPU, or CUDA")
    params = vars(args).copy()
    for key in ("L", "chains", "sweeps", "equil_sweeps"):
        params[key] = int(params[key])
    validate_run(params, backend)
    if args.dry_run:
        print(json.dumps({"argv": native_command(params, backend)}, indent=2))
    else:
        record = run_native(params, backend, Path.cwd())
        print(record["stdout"], end="")
        print(record["stderr"], end="", file=sys.stderr)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "campaign":
        from campaign import main as campaign_main
        return campaign_main(argv[1:])
    if argv and argv[0] == "modal":
        modal = shutil.which("modal")
        require(modal is not None, "install the optional Modal SDK: pip install modal")
        return subprocess.call([modal, "run", str(ROOT / "modal_app.py"), *argv[1:]])
    if argv and argv[0].isascii() and argv[0].isdecimal():
        return legacy(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("info", help="inspect sources, selections, and recorded evidence; no device allocation")
    bp = sub.add_parser("build", help="compile a selected native driver (never trains or benchmarks)")
    bp.add_argument("--backend", choices=BACKENDS, required=True)
    bp.add_argument("--mode", choices=MODES, default="nonlocal")
    bp.add_argument("--preset", choices=PRESETS, default="depth")
    bp.add_argument("--dry-run", action="store_true")
    rp = sub.add_parser("run", help="run one batch, build on first use")
    add_run_options(rp)
    ip = sub.add_parser("inspect", help="inspect an NPZ or campaign manifest locally")
    ip.add_argument("path", type=Path)
    sub.add_parser("campaign", help="stream/resume a beta campaign; see campaign --help")
    sub.add_parser("modal", help="launch a bounded Modal campaign; requires --budget-usd")
    args = parser.parse_args(argv)
    if args.action == "info":
        result = info()
    elif args.action == "inspect":
        result = inspect_path(args.path)
    elif args.action == "build":
        result = build(args.backend, args.mode, args.preset, args.dry_run)
    else:
        params = vars(args).copy()
        validate_run(params, args.backend)
        if args.dry_run:
            result = {"build": build_command(args.backend, args.mode, args.preset),
                      "argv": native_command(params, args.backend, args.native_driver), "cwd": str(args.output_dir.resolve())}
        else:
            result = run_native(params, args.backend, args.output_dir, args.native_driver)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"solid-final: {exc}", file=sys.stderr)
        raise SystemExit(1)
