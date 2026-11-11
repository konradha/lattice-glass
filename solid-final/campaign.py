#!/usr/bin/env python3
"""Stream independent fixed-beta batches; checkpoint/resume without sweep-sized RAM."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

from driver import (BACKENDS, ROOT, U64, atomic_json, build, driver_path,
                    native_command, require, sha256, state_energy, uint64, validate_run)

SCHEMA = "solid-final.campaign.v1"
MANIFEST_SCHEMA = "solid-final.campaign.manifest.v1"


def positive(value, name):
    require(type(value) is int and 1 <= value <= U64, f"{name} must be a positive uint64")
    return value


def normalize_config(config):
    c = dict(config)
    require(c.get("schema", SCHEMA) == SCHEMA, "unknown campaign schema")
    c["schema"] = SCHEMA
    c.setdefault("L", 21)
    c["betas"] = [str(beta) for beta in c["betas"]]
    require(c["betas"] and len(set(c["betas"])) == len(c["betas"]), "betas must be nonempty and distinct")
    c.setdefault("modes", ["nonlocal", "local"])
    require(c["modes"] and len(set(c["modes"])) == len(c["modes"]) and all(m in ("nonlocal", "local", "local-colored") for m in c["modes"]), "invalid campaign modes")
    for key, default in (("chains", 16384), ("sweeps", 100000), ("equil_sweeps", 100000), ("batch_chains", 0), ("chunk_sweeps", 4096), ("history_every", 1), ("max_history_samples", 100000)):
        c.setdefault(key, default)
        require(type(c[key]) is int and 0 <= c[key] <= U64, f"invalid {key}")
    for key in ("chains", "sweeps", "chunk_sweeps", "history_every", "max_history_samples"):
        positive(c[key], key)
    require(c["chunk_sweeps"] <= 4096, "chunk_sweeps must be <=4096 for bounded checkpoint latency")
    c["seed"] = str(uint64(c.get("seed", "1")))
    c.setdefault("preset", "depth")
    for key, default in (("rho", "0.75"), ("rho1", "0.30"), ("rho2", "0.45")):
        c[key] = str(c.get(key, default))
    c.setdefault("mode_overrides", {})
    require(isinstance(c["mode_overrides"], dict) and all(m in c["modes"] for m in c["mode_overrides"]), "invalid mode_overrides")
    for mode in c["modes"]:
        override = c["mode_overrides"].get(mode, {})
        require(isinstance(override, dict) and set(override) <= {"chains", "sweeps", "equil_sweeps"}, "unknown mode override")
        for index, beta in enumerate(c["betas"]):
            chains, sweeps, equil = targets(c, mode, index)
            require(chains > 0 and sweeps > 0 and equil >= 0, "invalid mode schedule")
            # Backend-specific colored constraints are checked at launch.
            validate_run({**c, "beta": beta, "mode": "local" if mode == "local-colored" else mode,
                          "chains": chains, "sweeps": sweeps, "equil_sweeps": equil, "seed": int(c["seed"])}, "cpu")
    return c


def targets(config, mode, beta_index):
    override = config.get("mode_overrides", {}).get(mode, {})
    result = []
    for key in ("chains", "sweeps", "equil_sweeps"):
        value = override.get(key, config[key])
        if isinstance(value, list):
            require(len(value) == len(config["betas"]), f"{mode}.{key} must match beta count")
            value = value[beta_index]
        require(type(value) is int and 0 <= value <= U64, f"invalid {mode}.{key}")
        result.append(value)
    return tuple(result)


def cuda_sms():
    library = ctypes.CDLL("libcuda.so.1")
    for call in (lambda: library.cuInit(0),):
        require(call() == 0, "CUDA initialization failed")
    device, sms = ctypes.c_int(), ctypes.c_int()
    require(library.cuDeviceGet(ctypes.byref(device), 0) == 0, "CUDA device query failed")
    require(library.cuDeviceGetAttribute(ctypes.byref(sms), 16, device) == 0 and sms.value > 0, "CUDA SM count unavailable")
    return sms.value


def resolve_batch(config, backend, query_device=True):
    if config["batch_chains"]:
        return config["batch_chains"]
    if backend == "cpu":
        return min(64, os.cpu_count() or 1)
    if backend == "metal":
        return 256
    return cuda_sms() if query_device else None


def make_jobs(config):
    jobs, seeds = [], set()
    batch_chains = positive(config["batch_chains"], "resolved batch_chains")
    for index, beta in enumerate(config["betas"]):
        for mode in config["modes"]:
            chains, sweeps, equil = targets(config, mode, index)
            for batch, first in enumerate(range(0, chains, batch_chains)):
                # Same domain-separated construction as the proven opt3 producer.
                identity = json.dumps([config["seed"], beta, mode, first], separators=(",", ":")).encode("ascii")
                seed = int.from_bytes(hashlib.sha256(b"solid-opt3.fixed-production.seed.v1\0" + identity).digest()[:8], "big")
                require(seed not in seeds, "derived batch seed collision")
                seeds.add(seed)
                jobs.append({"id": f"beta{index:03d}-{mode}-batch{batch:06d}", "beta": beta, "mode": mode,
                             "seed": str(seed), "chain_start": first, "chains": min(batch_chains, chains - first),
                             "equil_target": equil, "production_target": sweeps, "equil_done": 0, "production_done": 0,
                             "next_sweep": 0, "chunks_done": 0, "checkpoint": None, "last_chunk": None})
    return jobs


def plan(config, backend):
    config = normalize_config(config)
    batch = resolve_batch(config, backend, query_device=False)
    groups = []
    for index, beta in enumerate(config["betas"]):
        for mode in config["modes"]:
            chains, sweeps, equil = targets(config, mode, index)
            groups.append({"beta": beta, "mode": mode, "chains": chains, "equil_sweeps": equil, "sweeps": sweeps,
                           "batches": (chains + batch - 1) // batch if batch else "runtime SM count",
                           "proposals": chains * (sweeps + equil) * config["L"] ** 3})
    result = {"config": config, "backend": backend, "resolved_batch_chains": batch, "groups": groups, "compute_started": False}
    if batch:
        config["batch_chains"] = batch
        jobs = make_jobs(config)
        result["jobs"] = len(jobs)
        result["first_job"] = jobs[0]
    return result


def read_native(path, params, previous_hash=None):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    chains, N = params["chains"], params["L"] ** 3
    for name in ("initial", "equilibrated", "final"):
        require(data[name].dtype == np.uint8 and data[name].shape == (chains, N), f"invalid native {name}")
    require(int(data["seed"][0]) == params["seed"], "native seed mismatch")
    require(int(data["start_sweep"][0]) == params["start_sweep"], "native start cursor mismatch")
    require(int(data["next_sweep"][0]) == params["start_sweep"] + params["equil_sweeps"] + params["sweeps"], "native next cursor mismatch")
    if previous_hash:
        require(hashlib.sha256(data["initial"].tobytes()).hexdigest() == previous_hash, "native initial state differs from committed checkpoint")
    require(int(data["meta"][0]) == params["L"] and int(data["meta"][1]) == chains, "native lattice/chain metadata mismatch")
    require(int(data["meta"][2]) == params["sweeps"] and int(data["meta"][4]) == params["equil_sweeps"], "native phase length mismatch")
    if "energy_int" not in data:
        data["energy_int"] = np.rint(data["energy"]).astype(np.uint32)
    for mode in ("local", "nonlocal"):
        key = f"accept_{mode}_count"
        if key not in data:
            data[key] = np.rint(data[f"accept_{mode}"] * N).astype(np.uint32)
    return data


def immutable_bytes(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def asset(output, relative, expected_hash=None):
    path = (output / relative).resolve()
    require(path.is_relative_to(output.resolve()), "checkpoint escapes campaign directory")
    require(path.is_file() and (expected_hash is None or sha256(path) == expected_hash), f"checkpoint missing or changed: {relative}")
    return path


def complete(job):
    return job["equil_done"] == job["equil_target"] and job["production_done"] == job["production_target"]


def chunk_count(config, job, limit):
    phase = "equil" if job["equil_done"] < job["equil_target"] else "production"
    done = job[phase + "_done"]
    remaining = job[phase + "_target"] - done
    count = min(remaining, limit)
    every = max(config["history_every"], (job["production_target"] + config["max_history_samples"] - 1) // config["max_history_samples"])
    if phase == "production":
        room = every - done % every
        if done % every:
            count = min(count, room)
        elif every <= count:
            count = count // every * every
        if done:
            count = min(count, (1 << done.bit_length()) - done)
    return phase, count, every


def save_observations(path, data, job, count, every):
    import numpy as np
    done, target = job["production_done"], job["production_target"]
    times = np.asarray(data["history_sweeps"], dtype=np.uint64) + np.uint64(done)
    keep = (times % every == 0) | (times == target)
    snap_times = np.asarray(data["snapshot_sweeps"], dtype=np.uint64) + np.uint64(done)
    snapshots = data["snapshots"]
    end = done + count
    # The old physical Metal host does not always snapshot the final partial segment.
    if ((end & (end - 1)) == 0 or end == target) and (not len(snap_times) or int(snap_times[-1]) != end):
        snap_times = np.append(snap_times, np.uint64(end))
        snapshots = np.concatenate((snapshots, data["final"][None]), axis=0)
    selected = np.asarray([int(t) > 0 and (int(t) & (int(t) - 1) == 0 or int(t) == target) for t in snap_times], dtype=bool)
    if not np.any(keep) and not np.any(selected):
        return None
    arrays = {"history_sweeps": times[keep], "history_sweeps_absolute": times[keep] + job["equil_target"],
              "snapshot_sweeps": snap_times[selected], "snapshots": snapshots[selected]}
    for key in ("energy", "energy_int", "accept_local", "accept_nonlocal", "accept_local_count", "accept_nonlocal_count"):
        arrays[key] = data[key][:, keep]
    for key in ("seed", "scalars"):
        arrays[key] = data[key]
    # Excluded checkpoint state avoids accumulating one full lattice per chunk.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": path.name, "sha256": sha256(path), "samples": int(np.count_nonzero(keep)), "snapshots": int(np.count_nonzero(selected))}


def execute_campaign(config, backend, native_driver, output_dir, max_wall_seconds=None, resume=False, commit_hook=None):
    require(backend in BACKENDS, "unknown backend")
    config = normalize_config(config)
    if max_wall_seconds is not None:
        require(math.isfinite(max_wall_seconds) and max_wall_seconds > 0, "wall limit must be finite and positive")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max_wall_seconds if max_wall_seconds is not None else math.inf
    with (output / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if resume:
            manifest = json.loads(manifest_path.read_text())
            require(manifest["schema"] == MANIFEST_SCHEMA, "unknown resume manifest")
            if not config["batch_chains"]:
                config["batch_chains"] = manifest["config"]["batch_chains"]
        else:
            require(not manifest_path.exists(), "campaign exists; pass --resume")
            manifest = None
        config["batch_chains"] = resolve_batch(config, backend)
        jobs = make_jobs(config)
        binaries = {mode: Path(native_driver).resolve() if native_driver else driver_path(backend, mode, config["preset"]) for mode in config["modes"]}
        for mode, binary in binaries.items():
            if not binary.exists() and native_driver is None:
                build(backend, mode, config["preset"])
            require(binary.is_file() and os.access(binary, os.X_OK), f"driver not executable: {binary}")
        hashes = {mode: sha256(binary) for mode, binary in binaries.items()}
        if resume:
            require(manifest["config"] == config and manifest["backend"] == backend and manifest["native_sha256"] == hashes,
                    "resume settings/backend/native binary differ; refusing trajectory reset")
            expected = [(j["id"], j["seed"], j["chains"]) for j in jobs]
            require(expected == [(j["id"], j["seed"], j["chains"]) for j in manifest["jobs"]], "resume batch identity mismatch")
            jobs = manifest["jobs"]
            for job in jobs:
                if job["checkpoint"]:
                    asset(output, job["checkpoint"]["file"], job["checkpoint"]["sha256"])
        else:
            manifest = {"schema": MANIFEST_SCHEMA, "config": config, "backend": backend, "native_sha256": hashes,
                        "jobs": jobs, "cursor": 0, "physical_local": "local-colored" not in config["modes"],
                        "history_note": "Observation shards use production-relative global timestamps; absolute timestamps include equilibration. Only latest checkpoint is retained; initial/equilibrated/final configurations persist."}

        def publish():
            atomic_json(manifest_path, manifest)
            if commit_hook:
                commit_hook()

        def finish(reason):
            result = {"reason": reason, "complete_jobs": sum(complete(j) for j in jobs), "jobs": len(jobs),
                      "completed_chain_sweeps": sum(j["chains"] * (j["equil_done"] + j["production_done"]) for j in jobs),
                      "manifest": str(manifest_path)}
            manifest["progress"] = result
            publish()
            return result

        publish()
        while any(not complete(job) for job in jobs):
            if deadline - time.monotonic() <= 5:
                return finish("deadline")
            index = manifest["cursor"] % len(jobs)
            job = jobs[index]
            manifest["cursor"] = (index + 1) % len(jobs)
            if complete(job):
                continue
            limit = config["chunk_sweeps"]
            if max_wall_seconds is not None:
                last = job["last_chunk"]
                if last and last["seconds"] > 0:
                    limit = min(limit, max(1, int(.5 * (deadline - time.monotonic() - 5) * last["sweeps"] / last["seconds"])))
                else:
                    limit = min(limit, 64)
            phase, count, every = chunk_count(config, job, limit)
            jobdir = output / job["id"]
            jobdir.mkdir(exist_ok=True)
            previous = job["checkpoint"]
            state = asset(output, previous["file"], previous["sha256"]) if previous else None
            params = {**config, "beta": job["beta"], "chains": job["chains"], "mode": job["mode"],
                      "seed": int(job["seed"]), "start_sweep": job["next_sweep"], "state": state,
                      "sweeps": count if phase == "production" else 0,
                      "equil_sweeps": count if phase == "equil" else 0,
                      "history_every": every, "max_history_samples": max(1, count)}
            command = native_command(params, backend, binaries[job["mode"]])
            started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="solid-final-") as temporary:
                timeout = None if math.isinf(deadline) else max(.001, deadline - started - 5)
                try:
                    result = subprocess.run(command, cwd=temporary, capture_output=True, text=True, timeout=timeout)
                except subprocess.TimeoutExpired:
                    return finish("native_timeout_rolled_back_to_last_checkpoint")
                require(result.returncode == 0, f"native failed ({result.returncode})\n{result.stdout}\n{result.stderr}")
                data = read_native(Path(temporary) / "run.npz", params, previous["sha256"] if previous else None)
            elapsed = time.monotonic() - started
            raw = data["final"].tobytes()
            final_hash = hashlib.sha256(raw).hexdigest()
            tag = f"{job['next_sweep'] + count:020d}-{uuid.uuid4().hex}"
            checkpoint = jobdir / ("checkpoint-" + tag + ".raw")
            immutable_bytes(checkpoint, raw)
            if job["chunks_done"] == 0:
                initial = jobdir / ("initial-" + tag + ".raw")
                immutable_bytes(initial, data["initial"].tobytes())
                job["initial"] = {"file": str(initial.relative_to(output)), "sha256": sha256(initial)}
            observation = None
            if phase == "production":
                observation = save_observations(jobdir / ("observations-" + tag + ".npz"), data, job, count, every)
            job[phase + "_done"] += count
            job["next_sweep"] += count
            job["chunks_done"] += 1
            job["checkpoint"] = {"file": str(checkpoint.relative_to(output)), "sha256": final_hash, "bytes": len(raw)}
            job["last_total_energy"] = state_energy(data["final"]).tolist()
            job["last_chunk"] = {"phase": phase, "sweeps": count, "seconds": elapsed, "observations": observation}
            if job["equil_done"] == job["equil_target"] and "equilibrated" not in job:
                equilibrated = jobdir / ("equilibrated-" + tag + ".raw")
                immutable_bytes(equilibrated, data["equilibrated"].tobytes() if phase == "production" else raw)
                job["equilibrated"] = {"file": str(equilibrated.relative_to(output)), "sha256": sha256(equilibrated)}
            if complete(job):
                # The retained final checkpoint is the production configuration.
                job["final"] = dict(job["checkpoint"])
            event = {"job": job["id"], "phase": phase, "sweeps": count, "next_sweep": job["next_sweep"],
                     "seed": job["seed"], "seconds": elapsed, "state_sha256": final_hash,
                     "observations": observation, "argv": command, "stdout": result.stdout, "stderr": result.stderr}
            with (output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            manifest.pop("progress", None)
            publish()
            # Delete only our superseded checkpoint, after the new manifest is durable.
            if previous:
                asset(output, previous["file"], previous["sha256"]).unlink()
            print(json.dumps({"job": job["id"], "equil_done": job["equil_done"], "production_done": job["production_done"], "next_sweep": job["next_sweep"]}), flush=True)
        return finish("complete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--native-driver", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--betas", default="0.01,2.0,2.5")
    parser.add_argument("--modes", default="nonlocal,local")
    parser.add_argument("--L", type=int, default=21)
    parser.add_argument("--chains", type=int, default=16384)
    parser.add_argument("--sweeps", type=int, default=100000)
    parser.add_argument("--equil-sweeps", type=int, default=100000)
    parser.add_argument("--seed", type=uint64, default=1)
    parser.add_argument("--batch-chains", type=int, default=0)
    parser.add_argument("--chunk-sweeps", type=int, default=4096)
    parser.add_argument("--history-every", type=int, default=1)
    parser.add_argument("--max-history-samples", type=int, default=100000)
    parser.add_argument("--preset", choices=("auto", "depth", "throughput", "ampere"), default="depth")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.config:
        config = json.loads(args.config.read_text())
    else:
        config = {key: getattr(args, key) for key in ("L", "chains", "sweeps", "equil_sweeps", "seed", "batch_chains", "chunk_sweeps", "history_every", "max_history_samples", "preset")}
        config.update(betas=args.betas.split(","), modes=args.modes.split(","))
    if args.dry_run:
        result = plan(config, args.backend)
    else:
        require(args.output_dir is not None, "--output-dir is required")
        result = execute_campaign(config, args.backend, args.native_driver, args.output_dir, args.max_wall_seconds, args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError, argparse.ArgumentTypeError) as exc:
        print(f"solid-final campaign: {exc}", file=sys.stderr)
        raise SystemExit(1)
