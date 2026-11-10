#!/usr/bin/env python3
"""Sustained throughput benchmark for the solid CPU and Metal drivers.

One driver process runs at a time.  Each case performs an untimed warmup (the
driver's EQUIL_SWEEPS) followed by a timed interval of --sweeps sweeps.  The
driver reports wall time for the timed interval; process startup, warmup, and
.npz writing are outside it.  The driver's internal chunking and snapshot
copies are inside the interval.

A sweep issues exactly L**3 proposals per chain in both nonlocal and local
mode, so throughput is

    attempts/s per chain = sweeps * L**3 / wall
    aggregate attempts/s = chains * attempts/s per chain
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "bench" / "scale.csv"
BACKENDS = ("cpu", "metal")
MODES = ("nonlocal", "local")
ATTEMPT_FACTOR = {"nonlocal": 1, "local": 1}
U64_MAX = (1 << 64) - 1
INT_MAX = (1 << 31) - 1

CSV_FIELDS = (
    "backend",
    "mode",
    "chains",
    "L",
    "sweeps",
    "warmup",
    "attempts_per_sweep",
    "wall_s",
    "attempts_per_chain_s",
    "aggregate_attempts_s",
    "status",
    "error",
)

DECIMAL_RE = re.compile(r"^[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
DONE_RE = re.compile(
    r"done L=(?P<L>\d+) chains=(?P<chains>\d+) sweeps=(?P<sweeps>\d+) "
    r"beta=(?P<beta>\S+) wall=(?P<wall>\d+(?:\.\d+)?)s"
)


@dataclass(frozen=True)
class Case:
    backend: str
    mode: str
    chains: int


@dataclass
class Config:
    backends: Tuple[str, ...]
    modes: Tuple[str, ...]
    chains: Tuple[int, ...]
    L: int
    beta: str
    rho: float
    rho1: float
    rho2: float
    sweeps: int
    warmup: int
    seed: int
    timeout: float
    csv_path: Path
    drivers: Dict[str, Path]
    driver_env: Dict[str, str]
    keep_workdir: bool


@dataclass
class Result:
    case: Case
    attempts_per_sweep: int
    wall_s: Optional[float] = None
    attempts_per_chain_s: Optional[float] = None
    aggregate_attempts_s: Optional[float] = None
    status: str = "ok"
    error: str = ""


def _env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure sustained solid sampler throughput for the CPU and Metal "
            "drivers.  Cases run sequentially, one driver process at a time."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backends",
        default=_env_default("SOLID_BENCH_BACKENDS", ",".join(BACKENDS)),
        metavar="LIST",
        help="comma-separated backends to run (cpu, metal); env SOLID_BENCH_BACKENDS",
    )
    parser.add_argument(
        "--modes",
        default=_env_default("SOLID_BENCH_MODES", ",".join(MODES)),
        metavar="LIST",
        help="comma-separated move modes (nonlocal, local); env SOLID_BENCH_MODES",
    )
    parser.add_argument(
        "--chains",
        default=_env_default("SOLID_BENCH_CHAINS", "8,64,128,256"),
        metavar="LIST",
        help="comma-separated chain counts; env SOLID_BENCH_CHAINS",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=_env_default("SOLID_BENCH_L", "21"),
        metavar="N",
        help="lattice side length (4..32); env SOLID_BENCH_L",
    )
    parser.add_argument(
        "--beta",
        default=_env_default("SOLID_BENCH_BETA", "2.0"),
        metavar="VALUE",
        help="inverse temperature, passed verbatim to the driver; env SOLID_BENCH_BETA",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=_env_default("SOLID_BENCH_RHO", "0.75"),
        metavar="VALUE",
        help="total occupied fraction; must equal rho1+rho2; env SOLID_BENCH_RHO",
    )
    parser.add_argument(
        "--rho1",
        type=float,
        default=_env_default("SOLID_BENCH_RHO1", "0.30"),
        metavar="VALUE",
        help="type-1 occupied fraction; env SOLID_BENCH_RHO1",
    )
    parser.add_argument(
        "--rho2",
        type=float,
        default=_env_default("SOLID_BENCH_RHO2", "0.45"),
        metavar="VALUE",
        help="type-2 occupied fraction; env SOLID_BENCH_RHO2",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        default=_env_default("SOLID_BENCH_SWEEPS", "1000"),
        metavar="N",
        help="timed sweeps per chain; env SOLID_BENCH_SWEEPS",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=_env_default("SOLID_BENCH_WARMUP", "200"),
        metavar="N",
        help="untimed warmup sweeps, passed as EQUIL_SWEEPS; env SOLID_BENCH_WARMUP",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_default("SOLID_BENCH_SEED", "1"),
        metavar="N",
        help="driver seed; env SOLID_BENCH_SEED",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_env_default("SOLID_BENCH_TIMEOUT", "3600"),
        metavar="SECONDS",
        help="per-case driver timeout; env SOLID_BENCH_TIMEOUT",
    )
    parser.add_argument(
        "--cpu-driver",
        default=_env_default("SOLID_CPU_DRIVER", str(ROOT / "cpu" / "driver")),
        metavar="PATH",
        help="CPU driver executable; env SOLID_CPU_DRIVER",
    )
    parser.add_argument(
        "--metal-driver",
        default=_env_default("SOLID_METAL_DRIVER", str(ROOT / "metal" / "driver")),
        metavar="PATH",
        help="Metal driver executable; env SOLID_METAL_DRIVER",
    )
    parser.add_argument(
        "--csv",
        default=_env_default("SOLID_BENCH_CSV", str(DEFAULT_CSV)),
        metavar="PATH",
        help="CSV output path; env SOLID_BENCH_CSV",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment variable for every driver process; repeatable",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="keep temporary work directories (run.npz files) for failed cases",
    )
    return parser


def parse_choice_list(text: str, allowed: Tuple[str, ...], label: str) -> Tuple[str, ...]:
    values = [item.strip().lower() for item in text.split(",")]
    if not values or any(not item for item in values):
        raise ValueError(
            f"{label} list must be a comma-separated subset of: {', '.join(allowed)}"
        )
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(
            f"unknown {label}(s): {', '.join(unknown)}; allowed: {', '.join(allowed)}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label} in list: {text!r}")
    return tuple(values)


def parse_chains(text: str) -> Tuple[int, ...]:
    parts = [item.strip() for item in text.split(",")]
    if not parts or any(not item for item in parts):
        raise ValueError("--chains must be a comma-separated list of positive integers")
    values: List[int] = []
    for item in parts:
        try:
            value = int(item, 10)
        except ValueError:
            raise ValueError(f"invalid chain count {item!r}; expected a positive integer")
        if value <= 0:
            raise ValueError(f"invalid chain count {value}; must be positive")
        if value > INT_MAX:
            raise ValueError(f"chain count {value} exceeds the driver's int range ({INT_MAX})")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate chain count in {text!r}")
    return tuple(values)


def parse_driver_env(items: List[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--env has an empty key: {item!r}")
        if "\x00" in key or "\x00" in value:
            raise ValueError("--env keys and values cannot contain NUL bytes")
        if key in env:
            raise ValueError(f"duplicate --env key {key!r}")
        env[key] = value
    return env


def resolve_driver(text: str, backend: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise ValueError(
            f"{backend} driver not found: {path} "
            f"(build it with: make -C {ROOT / backend})"
        )
    if not os.access(str(path), os.X_OK):
        raise ValueError(f"{backend} driver is not executable: {path}")
    return path


def build_config(args: argparse.Namespace) -> Config:
    backends = parse_choice_list(args.backends, BACKENDS, "backend")
    modes = parse_choice_list(args.modes, MODES, "mode")
    chains = parse_chains(args.chains)

    if args.L < 4 or args.L > 32:
        raise ValueError(
            f"L must be in [4, 32] because driver coordinates are packed in 5 bits; got {args.L}"
        )

    beta = args.beta.strip()
    if not DECIMAL_RE.match(beta):
        raise ValueError(f"beta must be a non-negative decimal number; got {args.beta!r}")
    beta_value = float(beta)
    if not math.isfinite(beta_value):
        raise ValueError(f"beta must be finite; got {args.beta!r}")

    for label, value in (("rho", args.rho), ("rho1", args.rho1), ("rho2", args.rho2)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be a finite non-negative number; got {value}")
    if args.rho1 + args.rho2 > 1.0 + 1e-9:
        raise ValueError(
            f"rho1+rho2 must not exceed 1; got {args.rho1 + args.rho2:.12g}"
        )
    if abs(args.rho - (args.rho1 + args.rho2)) > 1e-9:
        raise ValueError(
            "rho must equal rho1+rho2; "
            f"got rho={args.rho:.12g}, rho1+rho2={args.rho1 + args.rho2:.12g}"
        )

    N = args.L ** 3
    n1 = int(math.floor(args.rho1 * N + 0.5))
    n2 = int(math.floor(args.rho2 * N + 0.5))
    if n1 + n2 > N:
        raise ValueError(
            f"rho1/rho2 round to {n1}+{n2} occupied sites, exceeding N={N}"
        )
    if n1 + n2 == 0:
        raise ValueError("rho1/rho2 produce an empty occupied set")

    if args.sweeps < 1 or args.sweeps > U64_MAX:
        raise ValueError(f"sweeps must be in [1, {U64_MAX}]; got {args.sweeps}")
    if args.warmup < 0 or args.warmup > U64_MAX:
        raise ValueError(f"warmup must be in [0, {U64_MAX}]; got {args.warmup}")
    if args.sweeps + args.warmup > U64_MAX:
        raise ValueError("sweeps + warmup exceeds the driver's uint64 sweep index range")
    last_sweep = args.warmup + args.sweeps - 1
    max_proposal_index = (2 * last_sweep + 1) * N + (N - 1)
    if max_proposal_index > U64_MAX:
        raise ValueError(
            "warmup + sweeps is too large for the driver's 64-bit proposal index range"
        )
    if args.seed < 0 or args.seed > U64_MAX:
        raise ValueError(f"seed must be in [0, {U64_MAX}]; got {args.seed}")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise ValueError(f"timeout must be a finite positive number; got {args.timeout}")

    driver_text = {"cpu": args.cpu_driver, "metal": args.metal_driver}
    drivers = {backend: resolve_driver(driver_text[backend], backend) for backend in backends}
    driver_env = parse_driver_env(args.env)

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    else:
        csv_path = csv_path.resolve()
    if csv_path.exists() and csv_path.is_dir():
        raise ValueError(f"CSV path is a directory: {csv_path}")
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"cannot create CSV directory {csv_path.parent}: {exc}")

    return Config(
        backends=backends,
        modes=modes,
        chains=chains,
        L=args.L,
        beta=beta,
        rho=args.rho,
        rho1=args.rho1,
        rho2=args.rho2,
        sweeps=args.sweeps,
        warmup=args.warmup,
        seed=args.seed,
        timeout=args.timeout,
        csv_path=csv_path,
        drivers=drivers,
        driver_env=driver_env,
        keep_workdir=args.keep_workdir,
    )


def iter_cases(cfg: Config):
    for backend in cfg.backends:
        for mode in cfg.modes:
            for chains in cfg.chains:
                yield Case(backend=backend, mode=mode, chains=chains)


def _tail(text: str, limit: int = 240) -> str:
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return collapsed[-limit:]


def run_case(cfg: Config, case: Case) -> Result:
    attempts_per_sweep = (cfg.L ** 3) * ATTEMPT_FACTOR[case.mode]
    result = Result(case=case, attempts_per_sweep=attempts_per_sweep)
    try:
        workdir = Path(
            tempfile.mkdtemp(prefix=f"solid-scale-{case.backend}-{case.mode}-")
        )
    except OSError as exc:
        result.status = "error"
        result.error = f"cannot create work directory: {exc}"
        return result
    kind = "CPU" if case.backend == "cpu" else "GPU"
    args = [
        str(cfg.drivers[case.backend]),
        str(cfg.L),
        cfg.beta,
        repr(cfg.rho),
        repr(cfg.rho1),
        repr(cfg.rho2),
        str(cfg.sweeps),
        str(case.chains),
        kind,
        case.mode,
        str(cfg.warmup),
        "--seed",
        str(cfg.seed),
    ]
    env = os.environ.copy()
    env.update(cfg.driver_env)

    try:
        try:
            proc = subprocess.Popen(
                args,
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            result.status = "error"
            result.error = f"cannot launch driver: {exc}"
            return result

        try:
            stdout, stderr = proc.communicate(timeout=cfg.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            result.status = "error"
            result.error = f"timed out after {cfg.timeout:g}s"
            return result
        except BaseException:
            proc.kill()
            proc.communicate()
            raise

        if proc.returncode != 0:
            detail = _tail(stderr or stdout)
            result.status = "error"
            result.error = f"driver exited with status {proc.returncode}"
            if detail:
                result.error += f": {detail}"
            return result
        if stderr.strip():
            result.status = "error"
            result.error = f"unexpected driver stderr: {_tail(stderr)}"
            return result

        match = DONE_RE.search(stdout)
        if match is None:
            result.status = "error"
            result.error = f"driver did not report wall time: {_tail(stdout) or 'no output'}"
            return result
        got = (int(match.group("L")), int(match.group("chains")), int(match.group("sweeps")))
        expected = (cfg.L, case.chains, cfg.sweeps)
        if got != expected:
            result.status = "error"
            result.error = f"driver reported L/chains/sweeps {got}, expected {expected}"
            return result

        wall_s = float(match.group("wall"))
        if wall_s <= 0.0:
            result.status = "error"
            result.error = (
                f"driver reported non-positive wall time ({wall_s:.3f}s); "
                "increase --sweeps"
            )
            return result

        result.wall_s = wall_s
        result.attempts_per_chain_s = float(attempts_per_sweep * cfg.sweeps) / wall_s
        result.aggregate_attempts_s = result.attempts_per_chain_s * case.chains
        return result
    finally:
        if cfg.keep_workdir and result.status != "ok":
            print(f"kept workdir: {workdir}", file=sys.stderr)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def result_row(cfg: Config, result: Result) -> Dict[str, object]:
    case = result.case
    return {
        "backend": case.backend,
        "mode": case.mode,
        "chains": case.chains,
        "L": cfg.L,
        "sweeps": cfg.sweeps,
        "warmup": cfg.warmup,
        "attempts_per_sweep": result.attempts_per_sweep,
        "wall_s": "" if result.wall_s is None else f"{result.wall_s:.3f}",
        "attempts_per_chain_s": (
            "" if result.attempts_per_chain_s is None else f"{result.attempts_per_chain_s:.6f}"
        ),
        "aggregate_attempts_s": (
            "" if result.aggregate_attempts_s is None else f"{result.aggregate_attempts_s:.6f}"
        ),
        "status": result.status,
        "error": result.error,
    }


def print_progress(result: Result) -> None:
    case = result.case
    label = f"{case.backend}/{case.mode}/chains={case.chains}"
    if result.status == "ok":
        print(
            f"[{label}] wall={result.wall_s:.3f}s "
            f"chain={result.attempts_per_chain_s / 1e6:.2f}M/s "
            f"aggregate={result.aggregate_attempts_s / 1e6:.2f}M/s",
            file=sys.stderr,
        )
    else:
        print(f"[{label}] FAILED: {result.error}", file=sys.stderr)


def print_table(rows: List[Result]) -> None:
    if not rows:
        print("no completed cases")
        return
    header = (
        f"{'backend':<7} {'mode':<8} {'chains':>6} {'wall_s':>9} "
        f"{'chain M/s':>10} {'agg M/s':>10} {'status':<5}"
    )
    print(header)
    print("-" * len(header))
    for result in rows:
        case = result.case
        if result.status == "ok":
            wall = f"{result.wall_s:9.3f}"
            per_chain = f"{result.attempts_per_chain_s / 1e6:10.2f}"
            aggregate = f"{result.aggregate_attempts_s / 1e6:10.2f}"
        else:
            wall = f"{'-':>9}"
            per_chain = f"{'-':>10}"
            aggregate = f"{'-':>10}"
        print(
            f"{case.backend:<7} {case.mode:<8} {case.chains:6d} "
            f"{wall} {per_chain} {aggregate} {result.status:<5}"
        )


def print_errors(rows: List[Result]) -> None:
    failed = [result for result in rows if result.status != "ok"]
    if not failed:
        return
    print("\nerrors:", file=sys.stderr)
    for result in failed:
        case = result.case
        print(
            f"  {case.backend} {case.mode} chains={case.chains}: {result.error}",
            file=sys.stderr,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = build_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    cases = list(iter_cases(cfg))
    print("solid-opt2 sustained scale benchmark")
    print(
        f"  L={cfg.L} beta={cfg.beta} rho={cfg.rho:g} "
        f"rho1={cfg.rho1:g} rho2={cfg.rho2:g} seed={cfg.seed}"
    )
    print(
        f"  warmup={cfg.warmup} sweeps, timed={cfg.sweeps} sweeps, "
        f"proposals/sweep={cfg.L ** 3}"
    )
    print(
        f"  cases={len(cases)} backends={','.join(cfg.backends)} "
        f"modes={','.join(cfg.modes)} chains={','.join(str(c) for c in cfg.chains)}"
    )
    for backend in cfg.backends:
        print(f"  {backend} driver: {cfg.drivers[backend]}")
    print(f"  csv: {cfg.csv_path}")
    print()

    rows: List[Result] = []
    interrupted = False
    try:
        with cfg.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            handle.flush()
            for case in cases:
                result = run_case(cfg, case)
                rows.append(result)
                writer.writerow(result_row(cfg, result))
                handle.flush()
                print_progress(result)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted; partial results retained", file=sys.stderr)
    except OSError as exc:
        parser.exit(2, f"{parser.prog}: error: cannot write CSV {cfg.csv_path}: {exc}\n")

    print()
    print_table(rows)
    print_errors(rows)
    print(f"\ncsv: {cfg.csv_path}")

    if interrupted:
        print("interrupted before all cases completed", file=sys.stderr)
        return 130
    failed = sum(1 for result in rows if result.status != "ok")
    if failed:
        print(f"{failed}/{len(rows)} case(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
