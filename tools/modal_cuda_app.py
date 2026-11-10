"""CUDA only. Prepare fixtures locally before `modal run tools/modal_cuda_app.py`.

Examples (explicit positive budget is required; reservations are never refunded):
  modal run tools/modal_cuda_app.py --budget-usd 1.0 --action verify --gpu L4
  modal run tools/modal_cuda_app.py --budget-usd 1.0 --action bench --gpu L40S --seconds 1
  modal run tools/modal_cuda_app.py --budget-usd 1.0 --action campaign --run tiny --betas 3.2 \
      --chains 2 --batches 2 --sweeps 2 --equil 1 --chunk-sweeps 1 --mode both
Repeat the identical campaign command to resume. `batches` is chains per batch;
0 resolves to the GPU SM count. `both` means separate local/nonlocal ladders,
never the mixed driver mode. Download with `modal volume get lattice-cuda-runs
RUN ./download`. Each chunk contains initial/final states and its manifest has
an exact CPU replay command. Run it from the downloaded chunk's directory with
SOLID_CPU_DRIVER set to the absolute path of a CPU-built solid/driver.
The fixed toolchain reserves $0.05 once; native-source builds reserve $0.02
per digest and persist on lattice-cuda-builds. Python/fixture changes do not
change that digest or rebuild the toolchain image.
"""

import argparse
import contextlib
import ctypes
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "lattice-cuda-runs"
GPU_LIMITS = {"L4": (360, 0.10, 0.000222), "L40S": (240, 0.18, 0.000542)}
WINDOWS = (1, 32, 64, 128, 256)
CUDA_BASE = "nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43"
TOOLCHAIN_ID = "cuda1281-ubuntu2404-amd64-" + CUDA_BASE.split("@sha256:")[1] + "-boost183-gxx-numpy226-v1"
BUILD_INPUTS = ("Makefile", "driver.cpp", "cuda_check.cpp", "cuda_backend.cu",
                "cuda_backend.hpp", "lattice.hpp", "sweeps_cpu.hpp", "efma.hpp", "kernels.metal")
SOURCE_DIR = Path("/opt/lattice/solid")
FIXTURE_PATH = "/opt/lattice/cuda_fixtures.bin"


def _source_digest(directory):
    digest = hashlib.sha256()
    for name in BUILD_INPUTS:
        data = (directory / name).read_bytes()
        digest.update(name.encode() + b"\0" + len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()


def _build_path(digest):
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("Invalid compiled source digest")
    return Path("/builds") / TOOLCHAIN_ID / digest


def _binary(config, name):
    return str(_build_path(config["build_digest"]) / name)


def _bounded_call(function, argument, execution_seconds):
    call = function.spawn(argument)
    try:
        # Bound total queue + execution time. Call graphs are best-effort and
        # cannot reliably distinguish a queued input from a running kernel.
        return call.get(timeout=execution_seconds + 60)
    except BaseException:
        call.cancel(terminate_containers=True)
        raise


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _reserve(path, budget, kind, dollars, once=False):
    with _lock(str(path) + ".lock"):
        ledger = json.loads(Path(path).read_text()) if Path(path).exists() else {"reservations": []}
        spent = sum(row["reserved_usd"] for row in ledger["reservations"])
        if spent > budget + 1e-10:
            raise RuntimeError(f"Budget refused: existing reservations ${spent:.3f} > ${budget:.3f}; {path}")
        if once:
            previous = next((row for row in ledger["reservations"] if row["kind"] == kind), None)
            if previous is not None:
                return previous
        if spent + dollars > budget + 1e-10:
            raise RuntimeError(f"Budget refused: reserved ${spent:.3f} + ${dollars:.3f} > ${budget:.3f}; {path}")
        row = {"id": uuid.uuid4().hex, "kind": kind, "reserved_usd": dollars,
               "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        ledger["reservations"].append(row)
        _json_write(path, ledger)
        print(json.dumps({"cost_reservation": row, "total_reserved_usd": spent + dollars}), flush=True)
        return row


# Modal builds images before calling the local entrypoint. Reserve the build
# before defining the image, not after deployment has already incurred its cost.
_build_reservation = None
if modal.is_local():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--ledger", default=str(ROOT / ".modal_cuda_budget.json"))
    early, _ = parser.parse_known_args()
    if not math.isfinite(early.budget_usd) or early.budget_usd <= 0:
        raise ValueError("budget-usd must be finite and positive")
    _build_reservation = _reserve(Path(early.ledger).expanduser().resolve(), early.budget_usd,
                                  "toolchain-image:" + TOOLCHAIN_ID, 0.05, once=True)

app = modal.App("lattice-clean-solid-cuda")
image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.11")
    .apt_install("build-essential", "libboost1.83-dev")
    .pip_install("numpy==2.2.6")
    .add_local_dir(str(ROOT / "solid"), str(SOURCE_DIR), copy=False,
                   ignore=["driver", "cuda_check", "cuda_check_prepare", "cuda_fixtures.bin",
                           "*.o", "*.npz", "*.py", "*.md", "*.sh", "__pycache__", ".gitignore"])
    .add_local_file(str(ROOT / "solid/cuda_fixtures.bin"), FIXTURE_PATH, copy=False)
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
build_volume = modal.Volume.from_name("lattice-cuda-builds", create_if_missing=True)


def _device():
    library = ctypes.CDLL("libcuda.so.1")
    def checked(code):
        if code:
            raise RuntimeError(f"CUDA driver API failed: {code}")
    checked(library.cuInit(0))
    device = ctypes.c_int()
    checked(library.cuDeviceGet(ctypes.byref(device), 0))
    sms = ctypes.c_int()
    checked(library.cuDeviceGetAttribute(ctypes.byref(sms), 16, device))
    return sms.value


def _command(argv, rows, deadline, cwd=None):
    remaining = deadline - time.monotonic()
    if remaining < 2:
        raise TimeoutError("Call work deadline reached")
    started = time.monotonic()
    try:
        process = subprocess.run(list(map(str, argv)), cwd=cwd, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=remaining - 1)
        row = {"argv": list(map(str, argv)), "seconds": time.monotonic() - started,
               "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        row = {"argv": list(map(str, argv)), "seconds": time.monotonic() - started,
               "exit_code": "timeout", "stdout": text(exc.stdout), "stderr": text(exc.stderr)}
    rows.append(row)
    print(json.dumps({"command_result": row}), flush=True)
    if row["exit_code"] != 0:
        raise RuntimeError(f"Command failed ({row['exit_code']}): {shlex.join(row['argv'])}")
    return row


@app.function(image=image, cpu=2, memory=4096, timeout=120, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/builds": build_volume})
def compile_backend(source_digest):
    """Compile once per fixed native-source digest, without allocating a GPU."""
    build_volume.reload()
    destination = _build_path(source_digest)
    marker = destination / "build.json"
    if marker.is_file():
        result = json.loads(marker.read_text())
        if result["source_digest"] != source_digest or result["toolchain"] != TOOLCHAIN_ID:
            raise RuntimeError("Compiled cache marker mismatch")
        if not all((destination / name).is_file() for name in ("driver", "cuda_check")):
            raise RuntimeError("Compiled cache is missing an executable")
        return {**result, "cached": True}
    if _source_digest(SOURCE_DIR) != source_digest:
        raise RuntimeError("Mounted native sources changed after the local digest was computed")
    rows = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for name in BUILD_INPUTS:
            shutil.copyfile(SOURCE_DIR / name, scratch / name)
        _command(["make", "-C", str(scratch), "HAVE_CUDA=1", "HAVE_METAL=0",
                  "CXX=g++", "BOOST_INC=/usr/include", "driver", "cuda_check"],
                 rows, started + 105)
        result = {"source_digest": source_digest, "toolchain": TOOLCHAIN_ID,
                  "rows": rows, "seconds": time.monotonic() - started}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".compile-") as staging:
            staging = Path(staging)
            for name in ("driver", "cuda_check"):
                shutil.copy2(scratch / name, staging / name)
            _json_write(staging / "build.json", result)
            os.replace(staging, destination)
        build_volume.commit()
    return {**result, "cached": False}


def _clocks(rows, deadline):
    return _command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.current.memory,clocks.max.sm,power.limit",
                     "--format=csv,noheader"], rows, deadline)["stdout"].strip()


def _verify(config, rows, deadline):
    checker = _binary(config, "cuda_check")
    fixture = FIXTURE_PATH
    if not Path(fixture).is_file():
        raise RuntimeError("Missing CPU-generated solid/cuda_fixtures.bin; prepare locally before deployment")
    selected = ",".join(map(str, config["windows"]))
    # Finish every correctness gate for both independent ladders before timing.
    for stage in ("smoke", "single", "multi", "saturated"):
        for move in ("local", "nonlocal"):
            _command([checker, "--check", fixture, "--stage", stage, "--windows", selected,
                      "--move", move], rows, deadline)
    for tool in ("memcheck", "racecheck", "synccheck"):
        for move in ("local", "nonlocal"):
            _command(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                      checker, "--check", fixture, "--stage", "smoke", "--windows", selected,
                      "--move", move], rows, deadline)


def _benchmark(config, sms, rows, deadline):
    # The single-chain ladder precedes every occupancy ladder, separately by move.
    for chains in (1, sms, 2 * sms, 4 * sms, 8 * sms):
        for move in ("local", "nonlocal"):
            for window in config["windows"]:
                if window == 1:
                    continue  # Serial correctness baseline, not a throughput candidate.
                _command([_binary(config, "cuda_check"), "--bench", "--move", move,
                          "--L", config["L"], "--beta", config["beta"], "--chains", chains,
                          "--window", window, "--seconds", config["seconds"]], rows, deadline)


def _final_gate(config, rows, deadline):
    benchmarks = []
    for row in rows:
        for line in row["stdout"].splitlines():
            if line.lstrip().startswith("{"):
                value = json.loads(line)
                if value.get("kind") == "benchmark":
                    benchmarks.append(value)
    selected = {}
    for move in ("local", "nonlocal"):
        ladder = [value for value in benchmarks if value["move"] == move]
        if ladder:
            # Preserve both the latency and whole-card throughput winner.
            windows = sorted({max(ladder, key=lambda value: value[metric])["window"]
                              for metric in ("per_chain_attempts_s", "aggregate_attempts_s")})
        else:
            windows = config["windows"]
        selected[move] = windows
        _command([_binary(config, "cuda_check"), "--check",
                  FIXTURE_PATH, "--stage", "final",
                  "--windows", ",".join(map(str, windows)), "--move", move], rows, deadline)
    return selected


def _campaign(config, sms, rows, deadline):
    import numpy as np

    volume.reload()
    directory = Path("/runs") / config["run"]
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    settings = {key: value for key, value in config.items()
                if key not in ("action", "windows", "seconds", "beta", "max_chunks")}
    settings["batches"] = settings["batches"] or sms
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if config["batches"] == 0:
            settings["batches"] = manifest["settings"]["batches"]
        if manifest["settings"] != settings:
            raise ValueError("Campaign settings differ from durable manifest; use a new --run")
        for job in manifest["jobs"]:
            for chunk in job["chunks"]:
                if not (directory / chunk["file"]).is_file():
                    raise FileNotFoundError(f"Durable chunk missing: {chunk['file']}")
    else:
        jobs = []
        moves = ("local", "nonlocal") if settings["mode"] == "both" else (settings["mode"],)
        for beta_index, beta in enumerate(settings["betas"]):
            for move in moves:
                for batch in range(math.ceil(settings["chains"] / settings["batches"])):
                    # An odd stride permutes uint64 counters: no seed collisions
                    # between the independent beta/move/batch jobs.
                    seed = (settings["seed"] + len(jobs) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
                    jobs.append({"id": f"beta{beta_index:03d}-{move}-batch{batch:06d}", "beta": beta,
                                 "move": move, "seed": seed,
                                 "chains": min(settings["batches"], settings["chains"] - batch * settings["batches"]),
                                 "equil_done": 0, "production_done": 0, "chunks": []})
        manifest = {"version": 1, "backend": "CUDA", "settings": settings, "jobs": jobs,
                    "history_note": "history_sweeps and snapshot_sweeps are chunk-relative; add production_start for production chunks"}
        _json_write(manifest_path, manifest)
        volume.commit()
    written = 0
    for job in manifest["jobs"]:
        while job["equil_done"] < settings["equil"] or job["production_done"] < settings["sweeps"]:
            if written >= config["max_chunks"] or deadline - time.monotonic() < 15:
                return _campaign_progress(manifest, written)
            phase = "equil" if job["equil_done"] < settings["equil"] else "production"
            key = "equil_done" if phase == "equil" else "production_done"
            target = settings["equil"] if phase == "equil" else settings["sweeps"]
            count = min(settings["chunk_sweeps"], target - job[key])
            start = job["equil_done"] + job["production_done"]
            name = f"{job['id']}-{phase}-{start:016d}-{start + count:016d}.npz"
            with tempfile.TemporaryDirectory() as scratch:
                scratch = Path(scratch)
                state = []
                if job["chunks"]:
                    previous = directory / job["chunks"][-1]["file"]
                    with np.load(previous, allow_pickle=False) as data:
                        raw = np.asarray(data["final"], dtype=np.uint8)
                        if raw.shape != (job["chains"], settings["L"] ** 3):
                            raise ValueError(f"Invalid checkpoint shape: {previous}")
                        (scratch / "state.raw").write_bytes(raw.tobytes())
                    state = [str(scratch / "state.raw")]
                argv = [_binary(config, "driver"), str(settings["L"]), job["beta"],
                        str(settings["rho"]), str(settings["rho1"]), str(settings["rho2"]),
                        str(count), str(job["chains"]), "CUDA", job["move"], "0", *state,
                        "--seed", str(job["seed"]), "--start-sweep", str(start),
                        "--cuda-window", str(settings["window"]),
                        "--history-every", str(settings["history_every"]),
                        "--max-history-samples", str(settings["max_history_samples"])]
                result = _command(argv, rows, deadline, cwd=scratch)
                with np.load(scratch / "run.npz", allow_pickle=False) as data:
                    if data["final"].shape != (job["chains"], settings["L"] ** 3):
                        raise ValueError("Driver produced an invalid final-state shape")
                destination = directory / name
                temporary = directory / (name + ".tmp")
                shutil.copyfile(scratch / "run.npz", temporary)
                os.replace(temporary, destination)
                # Replay uses the stored initial state even for the first chunk,
                # avoiding any dependency on host initialization implementation.
                cpu = argv[:]
                cpu[0] = "CPU_DRIVER"
                cpu[8] = "CPU"
                if state:
                    cpu[11] = "initial.raw"
                else:
                    cpu.insert(11, "initial.raw")
                extract = f"import numpy as np; np.load({name!r}, allow_pickle=False)['initial'].tofile('initial.raw')"
                replay = ("python3 -c " + shlex.quote(extract) + " && "
                          + '"${SOLID_CPU_DRIVER:?Set SOLID_CPU_DRIVER to the absolute CPU driver path}" '
                          + shlex.join(cpu[1:]))
                job["chunks"].append({"file": name, "phase": phase, "start_sweep": start,
                                      "production_start": job["production_done"], "sweeps": count,
                                      "seconds": result["seconds"], "seed": job["seed"],
                                      "cpu_replay": replay, "gpu_argv": argv})
                job[key] += count
                _json_write(manifest_path, manifest)
                volume.commit()
                written += 1
                print(json.dumps({"checkpoint": name, "progress": _campaign_progress(manifest, written)}), flush=True)
    return _campaign_progress(manifest, written)


def _campaign_progress(manifest, written):
    settings = manifest["settings"]
    complete = sum(job["equil_done"] == settings["equil"] and job["production_done"] == settings["sweeps"]
                   for job in manifest["jobs"])
    return {"run": settings["run"], "complete": complete == len(manifest["jobs"]),
            "jobs_complete": complete, "jobs_total": len(manifest["jobs"]), "chunks_written": written,
            "volume": VOLUME_NAME, "manifest": settings["run"] + "/manifest.json",
            "npz_paths": [settings["run"] + "/" + job["chunks"][-1]["file"]
                          for job in manifest["jobs"] if job["chunks"]][-8:]}


def _gpu_call(config, gpu):
    hard_timeout, _, rate = GPU_LIMITS[gpu]
    started = time.monotonic()
    deadline = started + hard_timeout - 20
    result = {"gpu": gpu, "action": config["action"], "ok": False, "rows": []}
    try:
        build_volume.reload()
        marker = _build_path(config["build_digest"]) / "build.json"
        if not marker.is_file():
            raise RuntimeError("Compiled backend missing; CPU compilation must succeed before GPU invocation")
        result["build_digest"] = config["build_digest"]
        result["sms"] = _device()
        result["clocks_before"] = _clocks(result["rows"], deadline)
        _verify(config, result["rows"], deadline)
        if config["action"] in ("verify", "bench"):
            smoke = {**config, "action": "campaign", "run": "cuda-smoke-" + uuid.uuid4().hex,
                     "L": 8, "beta": "3.2", "betas": ["3.2"], "chains": 2, "batches": 2,
                     "mode": "both", "equil": 2, "sweeps": 4, "chunk_sweeps": 2,
                     "seed": 987654321, "window": 32, "windows": [32], "max_chunks": 2,
                     "history_every": 1, "max_history_samples": 100000,
                     "rho": 0.75, "rho1": 0.3, "rho2": 0.45}
            first = _campaign(smoke, result["sms"], result["rows"], deadline)
            result["smoke"] = {"initial": first}
            if first["complete"] or first["chunks_written"] != 2:
                raise RuntimeError("Production smoke did not reach its planned resume boundary")
            smoke["max_chunks"] = 10
            resumed = _campaign(smoke, result["sms"], result["rows"], deadline)
            result["smoke"]["resumed"] = resumed
            if not resumed["complete"]:
                raise RuntimeError("Resumed production smoke did not finish")
        if config["action"] == "bench":
            _benchmark(config, result["sms"], result["rows"], deadline)
        elif config["action"] == "campaign":
            result["progress"] = _campaign(config, result["sms"], result["rows"], deadline)
        if gpu == "L40S" and config["action"] == "bench":
            result["final_windows"] = _final_gate(config, result["rows"], deadline)
        result["clocks_after"] = _clocks(result["rows"], deadline)
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = time.monotonic() - started
    result["cost"] = {"gpu_rate_usd_s": rate, "gpu_elapsed_usd": result["elapsed_seconds"] * rate,
                      "cpu_cores": 2, "memory_gib": 4, "hard_timeout_seconds": hard_timeout,
                      "work_shutdown_headroom_seconds": 20, "scaledown_seconds": 2,
                      "reserved_usd": GPU_LIMITS[gpu][1],
                      "compute_elapsed_estimate_usd": result["elapsed_seconds"] * (rate + 2 * 0.0000131 + 4 * 0.00000222),
                      "note": "Elapsed estimate excludes startup/image build; reservation is the conservative budget debit."}
    return result


@app.function(image=image, gpu="L4", cpu=2, memory=4096, timeout=360,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/runs": volume, "/builds": build_volume})
def run_l4(config):
    return _gpu_call(config, "L4")


@app.function(image=image, gpu="L40S", cpu=2, memory=4096, timeout=240,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/runs": volume, "/builds": build_volume})
def run_l40s(config):
    return _gpu_call(config, "L40S")


@app.local_entrypoint()
def main(budget_usd: float, action: str = "verify", gpu: str = "L4", windows: str = "1,32,64,128,256",
         seconds: float = 1.0, l: int = 21, beta: str = "3.2", run: str = "cuda-campaign",
         betas: str = "3.2", chains: int = 16384, batches: int = 0, mode: str = "both",
         sweeps: int = 1000, equil: int = 0, seed: int = 1, window: int = 128,
         chunk_sweeps: int = 32, max_chunks: int = 1, history_every: int = 1,
         max_history_samples: int = 100000, rho: float = 0.75, rho1: float = 0.3,
         rho2: float = 0.45, calls: int = 1,
         ledger: str = str(ROOT / ".modal_cuda_budget.json"),
         results: str = str(ROOT / "modal_cuda_results")):
    """One staged verify/bench call, or bounded resumable campaign calls.

    --batches is chains per GPU batch (0 = SM count); --calls authorizes at most
    that many serialized calls, each reserved before invocation. Campaign both
    creates independent local and nonlocal jobs. To resume, repeat the same
    campaign settings/run; calls/max-chunks can change. CPU replay commands in
    the downloaded manifest expect SOLID_CPU_DRIVER to point to the CPU driver.
    """
    if action not in ("verify", "bench", "campaign") or gpu not in GPU_LIMITS:
        raise ValueError("action must be verify/bench/campaign; gpu must be L4/L40S")
    chosen = [int(value) for value in windows.split(",")]
    if not chosen or len(set(chosen)) != len(chosen) or any(value not in WINDOWS for value in chosen):
        raise ValueError("windows must be distinct values from 1,32,64,128,256")
    if mode not in ("local", "nonlocal", "both") or window not in WINDOWS:
        raise ValueError("Invalid mode/window")
    if not 4 <= l <= 21 or any(value <= 0 for value in (chains, sweeps, chunk_sweeps, max_chunks, history_every, max_history_samples, calls)):
        raise ValueError("L must be 4..21 and counts must be positive")
    if equil < 0 or batches < 0 or not 0 <= seed < 2**64 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid equil/batches/seed/seconds")
    if not math.isfinite(budget_usd) or budget_usd <= 0:
        raise ValueError("budget-usd must be finite and positive")
    if action != "campaign" and calls != 1:
        raise ValueError("verify/bench uses exactly one staged GPU call")
    if not run or run in (".", "..") or Path(run).name != run:
        raise ValueError("run must be one directory name")
    beta_list = [value.strip() for value in betas.split(",")]
    if any(not value or not math.isfinite(float(value)) or float(value) < 0 for value in [beta, *beta_list]):
        raise ValueError("betas must be finite and nonnegative")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (rho, rho1, rho2)) or rho1 + rho2 > 1:
        raise ValueError("Invalid densities")
    if not (ROOT / "solid/cuda_fixtures.bin").is_file():
        raise FileNotFoundError("First run: make -C solid cuda_check_prepare && ./solid/cuda_check_prepare --prepare solid/cuda_fixtures.bin")
    config = {"action": action, "windows": [window] if action == "campaign" else chosen,
              "seconds": seconds, "L": l, "beta": beta, "run": run, "betas": beta_list,
              "chains": chains, "batches": batches, "mode": mode, "sweeps": sweeps, "equil": equil,
              "seed": seed, "window": window, "chunk_sweeps": chunk_sweeps, "max_chunks": max_chunks,
              "history_every": history_every, "max_history_samples": max_history_samples,
              "rho": rho, "rho1": rho1, "rho2": rho2}
    ledger_path = Path(ledger).expanduser().resolve()
    destination = Path(results).expanduser().resolve() / run
    destination.mkdir(parents=True, exist_ok=True)
    function = run_l4 if gpu == "L4" else run_l40s
    # Also serialize distinct CLI processes that share the budget ledger.
    with _lock(str(ledger_path) + ".invocation.lock"):
        source_digest = _source_digest(ROOT / "solid")
        compile_reservation = _reserve(ledger_path, budget_usd, "native-build:" + TOOLCHAIN_ID + ":" + source_digest,
                                       0.02, once=True)
        build_record = {"reservation": compile_reservation, "source_digest": source_digest}
        build_output = destination / ("build-" + source_digest + ".json")
        _json_write(build_output, build_record)
        try:
            build_record["result"] = _bounded_call(compile_backend, source_digest, 120)
            _json_write(build_output, build_record)
            print(json.dumps({"build_result_file": str(build_output), **build_record}), flush=True)
        except Exception as exc:
            build_record["error"] = f"{type(exc).__name__}: {exc}"
            _json_write(build_output, build_record)
            print(json.dumps({"build_result_file": str(build_output), **build_record}), flush=True)
            raise
        config["build_digest"] = source_digest
        for _ in range(calls):
            reservation = _reserve(ledger_path, budget_usd, gpu + ":" + action, GPU_LIMITS[gpu][1])
            output = destination / (reservation["id"] + ".json")
            record = {"reservation": reservation, "build_reservation": _build_reservation,
                      "compile_reservation": compile_reservation, "config": config}
            _json_write(output, record)
            try:
                result = _bounded_call(function, config, GPU_LIMITS[gpu][0])
                record["result"] = result
                _json_write(output, record)
                print(json.dumps({"result_file": str(output), **record}), flush=True)
            except Exception as exc:
                record["invocation_error"] = f"{type(exc).__name__}: {exc}"
                _json_write(output, record)
                print(json.dumps({"result_file": str(output), **record}), flush=True)
                raise
            if not result["ok"]:
                raise RuntimeError(result["error"])
            if action != "campaign" or result["progress"]["complete"]:
                break
