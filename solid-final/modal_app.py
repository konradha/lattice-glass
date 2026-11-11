"""Explicit-budget CUDA campaigns; importing locally requires run admission flags.

modal run solid-final/modal_app.py --budget-usd 5 --config campaign.json \
    --gpu L40S --preset depth --seconds 600
Resume with the same flags plus --resume-id ID and the printed ledger path.
Inspect/download: modal volume get lattice-solid-final-runs ID ./download
No historical optimization budget/date controller is used here.
"""

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import modal

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/opt/solid-final")
CUDA_BASE = "nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43"
TOOLCHAIN = "cuda1281-boost183-numpy226-v1-" + CUDA_BASE.split("@sha256:")[1]
OUTPUT_VOLUME = "lattice-solid-final-runs"
BUILD_VOLUME = "lattice-solid-final-builds"
GPU_RATES = {"L4": .000222, "L40S": .000542, "A100-40GB": .000583, "H100": .001097}
CPU_RATE = .0000131
MEMORY_RATE = .00000222
BUILD_SECONDS = 900
STARTUP_SECONDS = 60
QUEUE_SECONDS = 120
STOP_SECONDS = 2  # Modal's minimum scaledown window; no warm/buffer containers.
WORK_HEADROOM = 60
IMAGE_ALLOWANCE_USD = 1.0
FLAGS = {"CXX": "g++", "NVCC": "nvcc", "BOOST_INC": "/usr/include", "EXTRA_NVCCFLAGS": ""}
NATIVE_INPUTS = ("cuda/Makefile", "cuda/driver.cpp", "cuda/backend.hpp", "cuda/check.cpp",
                 "cuda/fast_empty_window.cu", "cuda/packed_type_commit.cu", "cuda/packed_fast_empty.cu",
                 "common/lattice.hpp", "common/sweeps_cpu.hpp", "common/efma.hpp")
BINARIES = ("driver-depth", "driver-throughput", "driver-ampere")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _native_digest(root):
    digest = hashlib.sha256(_canonical({"toolchain": TOOLCHAIN, "flags": FLAGS}))
    for name in NATIVE_INPUTS:
        data = (root / name).read_bytes()
        digest.update(name.encode() + b"\0" + len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()


def _code_digest(root):
    # Runtime source binding only: these files are mounted, never baked into the image.
    return _hash({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in ("campaign.py", "driver.py", "modal_app.py",
                               "cpu/selection.json", "metal/selection.json", "cuda/selection.json")})


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _config(path, preset):
    sys.path.insert(0, str(ROOT))
    from campaign import normalize_config
    config = normalize_config({**json.loads(Path(path).expanduser().read_text()), "preset": preset})
    if any(mode not in ("local", "nonlocal") for mode in config["modes"]):
        raise ValueError("CUDA campaigns support physical local/nonlocal only")
    if not 4 <= config["L"] <= 21 or config["chunk_sweeps"] > 4096:
        raise ValueError("CUDA requires L=4..21 and chunk_sweeps<=4096")
    return config


def _admit():
    # Modal materializes images before calling main: admission MUST precede Image/Volume definitions.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", choices=tuple(GPU_RATES), default="L40S")
    parser.add_argument("--preset", choices=("auto", "depth", "throughput", "ampere"), default="")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--resume-id", default="")
    parser.add_argument("--ledger", default="")
    args, _ = parser.parse_known_args()
    if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        raise ValueError("--budget-usd must be finite and positive")
    if not 1 <= args.seconds <= 3600:
        raise ValueError("--seconds must be 1..3600")
    run_id = args.resume_id or "solid-final-" + uuid.uuid4().hex
    if not re.fullmatch(r"solid-final-[0-9a-f]{32}", run_id):
        raise ValueError("invalid resume id")
    preset = ("ampere" if args.gpu == "A100-40GB" else "depth") if args.preset in ("", "auto") else args.preset
    if preset == "ampere" and args.gpu != "A100-40GB":
        raise ValueError("ampere preset is reserved for A100-40GB")
    config = _config(args.config, preset)
    native_digest = _native_digest(ROOT)
    code_digest = _code_digest(ROOT)
    settings = {"config": config, "gpu": args.gpu, "native_digest": native_digest, "code_digest": code_digest}
    settings_hash = _hash(settings)
    hard_timeout = args.seconds + WORK_HEADROOM
    # Full reservation retained on errors/cancellation; estimates are NOT an invoice or provider spending cap.
    reserve = (IMAGE_ALLOWANCE_USD + 2 * (BUILD_SECONDS + STARTUP_SECONDS + STOP_SECONDS) * (4 * CPU_RATE + 8 * MEMORY_RATE)
               + 2 * (hard_timeout + STARTUP_SECONDS + STOP_SECONDS) * (GPU_RATES[args.gpu] + 2 * CPU_RATE + 4 * MEMORY_RATE))
    ledger_path = Path(args.ledger or f".solid-final-{run_id}.ledger.json").expanduser().resolve()
    attempt = uuid.uuid4().hex
    with _lock(str(ledger_path) + ".lock"):
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            if (ledger["run_id"], ledger["settings_hash"], ledger["budget_usd"]) != (run_id, settings_hash, args.budget_usd):
                raise ValueError("ledger run/settings/budget mismatch; resume with original settings and cap")
        else:
            ledger = {"schema": "solid-final.modal-ledger.v1", "run_id": run_id,
                      "settings_hash": settings_hash, "budget_usd": args.budget_usd, "reservations": []}
        total = sum(row["reserved_usd"] for row in ledger["reservations"])
        if total + reserve > args.budget_usd:
            raise ValueError(f"budget admission denied: ${total + reserve:.6f} reserved exceeds ${args.budget_usd:.6f}")
        ledger["reservations"].append({"attempt": attempt, "reserved_usd": reserve, "utc_unix": time.time(),
                                       "seconds": args.seconds, "hard_timeout": hard_timeout, "status": "admitted"})
        _json_write(ledger_path, ledger)
    request = {**settings, "settings_hash": settings_hash, "run_id": run_id, "attempt": attempt,
               "seconds": args.seconds, "hard_timeout": hard_timeout, "resume": bool(args.resume_id),
               "reserved_usd": reserve, "budget_usd": args.budget_usd, "ledger": str(ledger_path)}
    print(json.dumps({"admitted": request, "volume": OUTPUT_VOLUME}), flush=True)
    return request


ADMISSION = _admit() if modal.is_local() else None
app = modal.App("lattice-solid-final-production")
image = (modal.Image.from_registry(CUDA_BASE, add_python="3.11")
         .apt_install("build-essential", "libboost1.83-dev")
         .pip_install("numpy==2.2.6")
         .add_local_dir(str(ROOT), str(SOURCE), copy=False,
                        ignore=["**/build/**", "**/driver", "**/driver-*", "**/check-*", "**/*.o", "**/*.npz",
                                "**/__pycache__/**", "**/.DS_Store", "artifacts/**", "evidence/**",
                                "benchmarks/**", "third_party/**"]))
builds = modal.Volume.from_name(BUILD_VOLUME, create_if_missing=True)
runs = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)
starts = modal.Dict.from_name("lattice-solid-final-worker-starts", create_if_missing=True)


def _build_path(digest):
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid native digest")
    return Path("/builds") / TOOLCHAIN / digest


def _bounded(function, request, timeout, phase):
    call = function.spawn(request)
    print(json.dumps({"run_id": request["run_id"], "function_call_id": call.object_id}), flush=True)
    try:
        expires = time.monotonic() + QUEUE_SECONDS
        while starts.get(request["attempt"] + ":" + phase) is None:
            if time.monotonic() >= expires:
                raise TimeoutError(f"No {phase} worker-start acknowledgement within {QUEUE_SECONDS}s")
            time.sleep(.5)
        return call.get(timeout=timeout + 30)
    except BaseException:
        call.cancel(terminate_containers=True)
        raise


@app.function(image=image, cpu=(4, 4), memory=(8192, 8192), timeout=BUILD_SECONDS,
              startup_timeout=STARTUP_SECONDS, retries=0, min_containers=0, max_containers=1,
              buffer_containers=0, scaledown_window=STOP_SECONDS, volumes={"/builds": builds})
def compile_backend(request):
    """CPU-only compilation; never invokes a native executable."""
    started = time.monotonic()
    starts.put(request["attempt"] + ":build", {"started_unix": time.time()})
    digest = request["native_digest"]
    if _native_digest(SOURCE) != digest:
        raise ValueError("mounted native inputs changed after admission")
    builds.reload()
    destination = _build_path(digest)
    marker = destination / "build.json"
    if marker.is_file():
        result = json.loads(marker.read_text())
        if result["native_digest"] != digest or result["toolchain"] != TOOLCHAIN:
            raise ValueError("native build cache marker mismatch")
        for name, expected in result["binaries"].items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
                raise ValueError("native build cache executable mismatch")
        return {**result, "cached": True, "call_seconds": time.monotonic() - started}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        for name in NATIVE_INPUTS:
            target = scratch / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, target)
        process = subprocess.run(["make", "-C", str(scratch / "cuda"), *BINARIES,
                                  *(f"{key}={value}" for key, value in FLAGS.items())],
                                 capture_output=True, text=True, timeout=BUILD_SECONDS - 45)
        result = {"native_digest": digest, "toolchain": TOOLCHAIN, "flags": FLAGS,
                  "seconds": time.monotonic() - started, "exit_code": process.returncode,
                  "stdout": process.stdout, "stderr": process.stderr,
                  "resources": {"cpu_cores": 4, "memory_gib": 8, "gpu": None}}
        result["resources"]["elapsed_compute_estimate_usd"] = result["seconds"] * (4 * CPU_RATE + 8 * MEMORY_RATE)
        result["resources"]["note"] = "Elapsed estimate only, not provider invoice; full reservation retained."
        if process.returncode:
            raise RuntimeError(json.dumps(result))
        result["binaries"] = {name: hashlib.sha256((scratch / "cuda" / name).read_bytes()).hexdigest() for name in BINARIES}
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".compile-") as staging_name:
            staging = Path(staging_name)
            for name in BINARIES:
                shutil.copy2(scratch / "cuda" / name, staging / name)
            _json_write(staging / "build.json", result)
            os.replace(staging, destination)
        builds.commit()
    return {**result, "cached": False}


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
    if sms.value <= 0:
        raise RuntimeError("CUDA reported a nonpositive SM count")
    return sms.value


def _clocks():
    process = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.current.memory,clocks.max.sm,power.limit",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
    return {"exit_code": process.returncode, "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(4096, 4096), timeout=3660,
              startup_timeout=STARTUP_SECONDS, retries=0, min_containers=0, max_containers=1,
              buffer_containers=0, scaledown_window=STOP_SECONDS, volumes={"/builds": builds, "/runs": runs})
def execute_campaign(request):
    started = time.monotonic()
    starts.put(request["attempt"] + ":gpu", {"started_unix": time.time()})
    result = {"run_id": request["run_id"], "attempt": request["attempt"], "ok": False}
    run_dir = Path("/runs") / request["run_id"]
    attempts = run_dir / "modal-attempts"
    try:
        builds.reload()
        runs.reload()
        if _native_digest(SOURCE) != request["native_digest"] or _code_digest(SOURCE) != request["code_digest"]:
            raise ValueError("mounted source changed after admission")
        metadata = run_dir / "modal-settings.json"
        if request["resume"]:
            prior = json.loads(metadata.read_text())
            if prior["settings_hash"] != request["settings_hash"]:
                raise ValueError("resume source/settings mismatch")
        elif run_dir.exists():
            raise ValueError("fresh run id already exists")
        else:
            _json_write(metadata, {key: request[key] for key in ("settings_hash", "config", "gpu", "native_digest", "code_digest")})
        binary_dir = _build_path(request["native_digest"])
        build = json.loads((binary_dir / "build.json").read_text())
        native = binary_dir / ("driver-" + request["config"]["preset"])
        if hashlib.sha256(native.read_bytes()).hexdigest() != build["binaries"][native.name]:
            raise ValueError("compiled executable digest mismatch")
        sms = _device()
        result.update(sms=sms, clocks_before=_clocks(), gpu=request["gpu"], preset=request["config"]["preset"],
                      launch={"window": 256, "threads": 256, "ppt": 1, "profile": False},
                      native_binary_sha256=build["binaries"][native.name])
        config = dict(request["config"])
        if config["batch_chains"] == 0:
            config["batch_chains"] = sms
        runtime_config = run_dir / "runtime-config.json"
        if runtime_config.exists() and json.loads(runtime_config.read_text()) != config:
            raise ValueError("resume runtime configuration/SM count changed")
        _json_write(runtime_config, config)
        _json_write(attempts / (request["attempt"] + ".admission.json"), request)
        runs.commit()  # Run id/settings are durable before native work begins.
        sys.path.insert(0, str(SOURCE))
        from campaign import execute_campaign as run_campaign
        with (attempts / (request["attempt"] + ".stdout.log")).open("w") as stdout, \
             (attempts / (request["attempt"] + ".stderr.log")).open("w") as stderr, \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result["progress"] = run_campaign(
                config=config, backend="cuda", native_driver=str(native),
                output_dir=str(run_dir / "campaign"), max_wall_seconds=request["seconds"],
                resume=request["resume"], commit_hook=runs.commit)
        result["ok"] = True
        result["clocks_after"] = _clocks()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.monotonic() - started
        result.update(elapsed_seconds=elapsed, resources={"cpu_cores": 2, "memory_gib": 4, "gpu": request["gpu"],
                      "gpu_rate_usd_s": GPU_RATES[request["gpu"]], "hard_timeout_seconds": request["hard_timeout"],
                      "reserved_usd": request["reserved_usd"],
                      "elapsed_compute_estimate_usd": elapsed * (GPU_RATES[request["gpu"]] + 2 * CPU_RATE + 4 * MEMORY_RATE),
                      "note": "Not an invoice; excludes unobserved startup/image/storage/network charges. Reservation is retained in full."})
        # A failed settings check must not overwrite another run's campaign state.
        _json_write(attempts / (request["attempt"] + ".result.json"), result)
        runs.commit()
    return result


@app.local_entrypoint()
def main(budget_usd: float, config: str, gpu: str = "L40S", preset: str = "", seconds: int = 600,
         resume_id: str = "", ledger: str = ""):
    """One bounded campaign segment; resume explicitly, never retry automatically."""
    request = ADMISSION
    if request is None:
        raise RuntimeError("local budget admission was not performed")
    status = {"status": "failed"}
    try:
        status["build"] = _bounded(compile_backend, request, BUILD_SECONDS, "build")
        exact_gpu = "H100!" if request["gpu"] == "H100" else request["gpu"]
        result = _bounded(execute_campaign.with_options(gpu=exact_gpu, timeout=request["hard_timeout"]),
                          request, request["hard_timeout"], "gpu")
        status.update(status="returned", result=result)
        print(json.dumps(result, indent=2), flush=True)
        print(f"Run {request['run_id']}; ledger {request['ledger']}; outputs: modal volume get {OUTPUT_VOLUME} {request['run_id']} ./download", flush=True)
        if not result["ok"]:
            raise RuntimeError(f"campaign failed; inspect durable output for {request['run_id']}")
    finally:
        with _lock(request["ledger"] + ".lock"):
            record = json.loads(Path(request["ledger"]).read_text())
            for row in record["reservations"]:
                if row["attempt"] == request["attempt"]:
                    row.update(status)
            _json_write(request["ledger"], record)
