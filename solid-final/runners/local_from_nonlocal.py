#!/usr/bin/env python3
"""Run physical-local replicas from saved nonlocal configurations, without redrawing states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driver import atomic_json, native_command, require, run_native, uint64


def sources(path):
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text())
        require(manifest["schema"] == "solid-final.configurations.v1", "expected exported configurations")
        for group in manifest["groups"]:
            yield group["beta"], np.load(path / group["directory"] / "configurations.npy", mmap_mode="r"), group
    else:
        with np.load(path, allow_pickle=False) as data:
            betas = data["betas"] if "betas" in data else [data["scalars"][0]]
            for index, beta in enumerate(betas):
                prefix = f"beta_{index}_" if "betas" in data else ""
                scalars, meta = data[prefix + "scalars"], data[prefix + "meta"]
                text_key = prefix + "beta_text"
                text = data[text_key].tobytes().decode("ascii") if text_key in data else str(float(beta))
                yield text, data[prefix + "final"], {"L": int(meta[0]), "rho": scalars[1], "rho1": scalars[2], "rho2": scalars[3]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="native/aging NPZ, or export_configs.py output directory")
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--preset", choices=("depth", "throughput", "ampere"), default="depth")
    parser.add_argument("--configs", type=int, default=32)
    parser.add_argument("--replicas", type=int, default=1, help="independent physical-local noise streams per identical initial configuration")
    parser.add_argument("--sweeps", type=uint64, required=True)
    parser.add_argument("--seed", type=uint64, default=1)
    parser.add_argument("--history-every", type=uint64, default=1)
    parser.add_argument("--max-history-samples", type=uint64, default=100000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    require(args.configs > 0 and args.replicas > 0 and args.sweeps > 0, "configs, replicas, sweeps must be positive")
    runs = []
    for index, (beta, state, model) in enumerate(sources(args.input.resolve())):
        require(state.dtype == np.uint8 and state.ndim == 2 and state.shape[1] == model["L"] ** 3, "invalid source state")
        require(state.shape[0] >= args.configs, "source contains fewer configurations than --configs")
        repeated = np.repeat(state[:args.configs], args.replicas, axis=0)
        seed = int.from_bytes(hashlib.sha256(f"solid-final.local-replicas.v1:{args.seed}:{beta}".encode()).digest()[:8], "big")
        directory = args.output_dir.resolve() / f"beta{index:03d}"
        params = {**model, "beta": beta, "mode": "local", "preset": args.preset, "chains": len(repeated),
                  "sweeps": args.sweeps, "equil_sweeps": 0, "start_sweep": 0, "seed": seed,
                  "history_every": args.history_every, "max_history_samples": args.max_history_samples,
                  "state": directory / "initial.raw"}
        record = {"beta": beta, "configs": args.configs, "replicas": args.replicas, "seed": str(seed),
                  "input_sha256": hashlib.sha256(repeated.tobytes()).hexdigest(), "output_dir": str(directory),
                  "argv": native_command(params, args.backend),
                  "row_mapping": "initial_configuration=row//replicas; replica=row%replicas; native RNG chain=row"}
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=False)
            params["state"].write_bytes(repeated.tobytes())
            record["run"] = run_native(params, args.backend, directory)
            np.savez(directory / "replicas.npz", configuration_index=np.repeat(np.arange(args.configs), args.replicas),
                     replica_index=np.tile(np.arange(args.replicas), args.configs))
            atomic_json(directory / "replicas.json", record)
        runs.append(record)
    require(runs, "input contains no configurations")
    result = {"schema": "solid-final.local-replicas.v1", "backend": args.backend, "physical_dynamics": True,
              "compute_started": not args.dry_run, "runs": runs}
    if not args.dry_run:
        atomic_json(args.output_dir / "manifest.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"solid-final local replicas: {exc}", file=sys.stderr)
        raise SystemExit(1)
