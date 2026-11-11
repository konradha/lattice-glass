#!/usr/bin/env python3
"""Export complete configuration ensembles to memory-mappable NPY files."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign import asset, targets
from driver import atomic_json, require, state_energy


def export(manifest_path, output_dir, mode="nonlocal", stage="equilibrated", beta=None):
    import numpy as np
    manifest_path = Path(manifest_path).resolve()
    source = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == "solid-final.campaign.manifest.v1", "expected a solid-final campaign manifest")
    output = Path(output_dir).resolve()
    require(not output.exists() or not any(output.iterdir()), "export output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    config, groups = manifest["config"], []
    require(mode in config["modes"], "requested mode is absent from campaign")
    for index, text in enumerate(config["betas"]):
        if beta is not None and Decimal(text) != Decimal(str(beta)):
            continue
        jobs = sorted((j for j in manifest["jobs"] if j["beta"] == text and j["mode"] == mode), key=lambda j: j["chain_start"])
        chains = targets(config, mode, index)[0]
        require(sum(j["chains"] for j in jobs) == chains, "ensemble batch coverage mismatch")
        require(all(stage in j for j in jobs), f"beta={text}: not every batch has reached {stage}")
        directory = output / f"beta{index:03d}-{mode}"
        directory.mkdir()
        states = np.lib.format.open_memmap(directory / "configurations.npy", mode="w+", dtype=np.uint8, shape=(chains, config["L"] ** 3))
        energies = np.lib.format.open_memmap(directory / "energies.npy", mode="w+", dtype=np.int64, shape=(chains,))
        seeds = np.lib.format.open_memmap(directory / "batch_seed.npy", mode="w+", dtype=np.uint64, shape=(chains,))
        rows = np.lib.format.open_memmap(directory / "chain_within_batch.npy", mode="w+", dtype=np.uint32, shape=(chains,))
        cursor = 0
        for job in jobs:
            require(job["chain_start"] == cursor, "noncontiguous ensemble batches")
            descriptor = job[stage]
            path = asset(source, descriptor["file"], descriptor["sha256"])
            require(path.stat().st_size == job["chains"] * config["L"] ** 3, "raw state size mismatch")
            state = np.memmap(path, mode="r", dtype=np.uint8, shape=(job["chains"], config["L"] ** 3))
            stop = cursor + job["chains"]
            states[cursor:stop] = state
            energies[cursor:stop] = state_energy(state)
            seeds[cursor:stop] = int(job["seed"])
            rows[cursor:stop] = np.arange(job["chains"], dtype=np.uint32)
            cursor = stop
        for array in (states, energies, seeds, rows):
            array.flush()
        groups.append({"beta": text, "directory": directory.name, "chains": chains, "mode": mode, "stage": stage,
                       "L": config["L"], "rho": config["rho"], "rho1": config["rho1"], "rho2": config["rho2"]})
    require(groups, "no beta matched")
    result = {"schema": "solid-final.configurations.v1", "source_manifest": str(manifest_path), "groups": groups,
              "config": config, "native_sha256": manifest["native_sha256"],
              "note": "Equilibrated denotes the configured burn-in boundary, not a measured relaxation guarantee. Exact continuation uses the original batch ordering and seed; row number alone is not the RNG chain key."}
    atomic_json(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("local", "nonlocal", "local-colored"), default="nonlocal")
    parser.add_argument("--stage", choices=("initial", "equilibrated", "final"), default="equilibrated")
    parser.add_argument("--beta")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.manifest, args.output_dir, args.mode, args.stage, args.beta), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"solid-final export: {exc}", file=sys.stderr)
        raise SystemExit(1)
