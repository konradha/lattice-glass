#!/usr/bin/env python3
"""Distributional CPU/Metal gate for the solid sampler.

Use this gate when the local-update schedule is algorithmically reordered
(checkerboard/colored schedules, grouped or reordered proposals) and the CPU
and Metal realizations no longer map the same RNG stream to the same moves, so
bitwise equality is the wrong criterion. For layout, compiler, precision and
other implementation changes use check_correctness.py instead, except for the
Metal L=21 local phase, which is the 21-colour schedule and is covered here.
See README.md for the change-class table.

What is checked:

  * both backends start from independent draws of the same initial-state
    distribution D0: uniform placement of n1 type-1 and n2 type-2 sites on the
    torus, with neighbor occupancy masks matching lattice.hpp;
  * identical bath parameters: L, beta, rho/rho1/rho2, update mode, burn-in and
    production sweeps, asserted from the driver metadata;
  * the driver discards the burn-in; a stationarity diagnostic additionally
    compares early and late production windows within each backend;
  * energy, occupancy, type-occupancy and local-environment distributions are
    compared across backends on the production window.

Statistics are permutation tests on chain-level functionals. Chains are the
independent units, so whole chains are relabeled across backends; this is
exact under the null and needs no distributional approximation. The joint test
is a union-intersection (min-p) combination over six functionals: the energy
distribution (mean, sd, quantiles), the density field L2 distance and maximum
site deviation, the type-1 field L2 distance and maximum site deviation, and
the local-environment histogram. This is a practical Geweke-style
joint-distribution check: the null is that the two kernels sample the same
joint law, and the test has power against changes in the energy law, density
field, type field, or local-environment distribution.
Simulation-based calibration is not applicable to this driver because the bath
(L, beta, densities) is fixed rather than drawn from a prior; run SBC
separately if a parameter-sampling driver is introduced.

Acceptance rates are schedule diagnostics. A reordered or colored schedule
legitimately changes per-move acceptance while preserving the stationary law
(each color updates against a different stale state), so acceptance is
reported with effect sizes and a permutation p-value but does not fail the gate
by default (--acceptance-gate warn|fail|off).

The gate compares two backends, not two algorithms: two implementations of the
same wrong kernel agree. An update-order change still needs a documented
stationarity argument, and tiny-lattice exact references where available. No
tempering, replica exchange, or ensembling is used or assumed.

Exit codes: 0 pass, 1 distribution mismatch, 2 setup error or inconclusive.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

GATE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLID_DIR = os.path.dirname(GATE_DIR)

MODE_CODES = {"both": 0, "nonlocal": 1, "local": 2}
ENERGY_NAMES = ("mean", "sd", "q10", "q25", "q50", "q75", "q90")
ACCEPT_NAMES = ("local", "nonlocal")
ENV_NAMES = tuple("type%d_n%d" % (t, k) for t in (1, 2) for k in range(7))
GATING = ("energy", "occupancy_l2", "occupancy_max", "type1_l2",
          "type1_max", "environment")


class GateError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _executable(path, label):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise GateError("%s: no such file: %s" % (label, path))
    if not os.access(path, os.X_OK):
        raise GateError("%s: not executable: %s" % (label, path))
    return path


def _candidates():
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


def find_driver(backend, explicit):
    env_names = (["SOLID_CPU_DRIVER"] if backend == "CPU"
                 else ["SOLID_METAL_DRIVER"])
    env_names.append("SOLID_DRIVER")
    label = "%s driver" % backend
    if explicit:
        return _executable(explicit, label)
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return _executable(value, "%s ($%s)" % (label, name))
    candidates = _candidates()[backend]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return os.path.abspath(path)
    tried = "\n  ".join(candidates)
    flag = "--cpu-driver" if backend == "CPU" else "--metal-driver"
    raise GateError(
        "no %s found; pass %s or set $%s.\ntried:\n  %s\nbuild with: "
        "make -C %s" % (label, flag, env_names[0], tried,
                         os.path.join(SOLID_DIR, backend.lower())))


def _probe_usage(driver, backend):
    return [driver, "4", "2.0", "0.75", "0.30", "0.45", "1", "1",
            backend, "both", "0"]


def probe(driver, backend, timeout):
    """Return (ok, message). Rejects CPU fallback instead of passing silently."""
    with tempfile.TemporaryDirectory(prefix="solid-probe-") as work:
        try:
            proc = subprocess.run(
                _probe_usage(driver, backend), cwd=work, capture_output=True,
                text=True, timeout=timeout)
        except OSError as exc:
            return False, "cannot execute: %s" % exc
        except subprocess.TimeoutExpired:
            return False, "probe timed out"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, (tail[-1] if tail else "exit %d" % proc.returncode)
        if backend == "GPU" and "using CPU" in proc.stderr:
            return False, ("binary silently falls back to CPU; rebuild the "
                           "Metal backend with: make -C %s"
                           % os.path.join(SOLID_DIR, "metal"))
        if not os.path.exists(os.path.join(work, "run.npz")):
            return False, "probe produced no run.npz"
    return True, ""


# ---------------------------------------------------------------------------
# Initial states
# ---------------------------------------------------------------------------

def pack_state(types, L):
    """Pack site types into driver bytes: type | neighbor_mask << 2.

    Flat index order is x + L*y + L*L*z, matching lattice.hpp. A reshaped
    (L, L, L) array therefore has axes (z, y, x).
    """
    N = L * L * L
    occupied = (types != 0).astype(np.uint8).reshape((L, L, L))
    mask = np.roll(occupied, -1, axis=2)          # +x
    mask |= np.roll(occupied, 1, axis=2) << 1     # -x
    mask |= np.roll(occupied, -1, axis=1) << 2    # +y
    mask |= np.roll(occupied, 1, axis=1) << 3     # -y
    mask |= np.roll(occupied, -1, axis=0) << 4    # +z
    mask |= np.roll(occupied, 1, axis=0) << 5     # -z
    return (types.astype(np.uint8) | (mask.reshape(N) << 2)).astype(np.uint8)


def check_packing(types, packed, L):
    """Independent loop check of pack_state, mirroring lattice.hpp::init."""
    def occupied(x, y, z):
        return int(types[x + L * y + L * L * z]) != 0

    for z in range(L):
        for y in range(L):
            for x in range(L):
                site = x + L * y + L * L * z
                mask = 0
                for direction, (a, b, c) in enumerate((
                        ((x + 1) % L, y, z), ((x - 1) % L, y, z),
                        (x, (y + 1) % L, z), (x, (y - 1) % L, z),
                        (x, y, (z + 1) % L), (x, y, (z - 1) % L))):
                    if occupied(a, b, c):
                        mask |= 1 << direction
                want = int(types[site] & 3) | (mask << 2)
                if int(packed[site]) != want:
                    raise GateError(
                        "internal state packing error at site %d: %d != %d"
                        % (site, int(packed[site]), want))


def make_states(L, n1, n2, chains, rng, self_check):
    """Draw `chains` independent states from D0 and pack them for STATE_RAW."""
    N = L * L * L
    packed = np.empty((chains, N), dtype=np.uint8)
    for chain in range(chains):
        types = np.zeros(N, dtype=np.uint8)
        order = rng.permutation(N)
        types[order[:n1]] = 1
        types[order[n1:n1 + n2]] = 2
        packed[chain] = pack_state(types, L)
        if self_check and chain == 0:
            check_packing(types, packed[chain], L)
    return packed


def derive_seed(master, backend, run):
    salt = 0x9E3779B9 if backend == "CPU" else 0x85EBCA6B
    value = master * 0x100000001B3 + salt + 0x10000 * run
    return 1 + value % (2 ** 63 - 1)


# ---------------------------------------------------------------------------
# Driver invocation and validation
# ---------------------------------------------------------------------------

def run_driver(driver, backend, args, seed, state_path, workdir, timeout):
    cmd = [driver, str(args.L), args.beta, args.rho, args.rho1, args.rho2,
           str(args.sweeps), str(args.chains), backend, args.mode,
           str(args.burn_in), state_path, "--seed", str(seed),
           "--history-every", "1", "--max-history-samples",
           str(args.sweeps)]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        raise GateError("%s run timed out after %.0fs" % (backend, timeout))
    except OSError as exc:
        raise GateError("%s driver failed to start: %s" % (backend, exc))
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise GateError("%s driver exit %d: %s"
                        % (backend, proc.returncode,
                           tail[-1] if tail else "no output"))
    if backend == "GPU" and "using CPU" in proc.stderr:
        raise GateError("Metal driver fell back to CPU: %s"
                        % proc.stderr.strip().splitlines()[-1])
    path = os.path.join(workdir, "run.npz")
    if not os.path.exists(path):
        raise GateError("%s driver produced no run.npz" % backend)
    with np.load(path) as npz:
        data = {key: np.array(npz[key]) for key in npz.files}
    os.unlink(path)
    return data, wall


def validate(data, args, n1, n2, N, packed, seed):
    for key in ("initial", "equilibrated", "final", "snapshots", "energy",
                "accept_local", "accept_nonlocal", "snapshot_sweeps",
                "history_sweeps", "history_every", "seed", "scalars", "meta"):
        if key not in data:
            raise GateError("driver output missing %r" % key)
    want_meta = [args.L, args.chains, args.sweeps, MODE_CODES[args.mode],
                 args.burn_in]
    meta = [int(value) for value in data["meta"]]
    if meta != want_meta:
        raise GateError("driver metadata %r != requested %r"
                        % (meta, want_meta))
    want_scalars = np.array([float(args.beta), float(args.rho),
                             float(args.rho1), float(args.rho2)])
    if not np.array_equal(data["scalars"], want_scalars):
        raise GateError("driver scalars %r != requested %r"
                        % (data["scalars"].tolist(), want_scalars.tolist()))
    if int(data["seed"][0]) != seed:
        raise GateError("driver used seed %d, requested %d"
                        % (int(data["seed"][0]), seed))
    if int(data["history_every"][0]) != 1:
        raise GateError("driver decimated the history; cannot compare samples")
    sweeps = data["history_sweeps"]
    if (sweeps.size != args.sweeps or int(sweeps[0]) != 1
            or int(sweeps[-1]) != args.sweeps):
        raise GateError("unexpected history_sweeps coverage")
    if data["initial"].shape != packed.shape:
        raise GateError("initial state shape %r != %r"
                        % (data["initial"].shape, packed.shape))
    if data["initial"].tobytes() != packed.tobytes():
        raise GateError("driver did not load the generated initial states")
    types = data["initial"] & 3
    if int((types == 1).sum(1).min()) != n1:
        raise GateError("type-1 count differs from n1=%d" % n1)
    if n2 and int((types == 2).sum(1).min()) != n2:
        raise GateError("type-2 count differs from n2=%d" % n2)
    if data["energy"].shape != (args.chains, args.sweeps):
        raise GateError("energy history shape %r"
                        % (data["energy"].shape,))
    energy = data["energy"]
    if not np.isfinite(energy).all() or np.any(energy < 0):
        raise GateError("energy history is not finite and non-negative")
    if data["snapshots"].shape[1:] != (args.chains, N):
        raise GateError("snapshot shape %r" % (data["snapshots"].shape,))


# ---------------------------------------------------------------------------
# Chain-level observables
# ---------------------------------------------------------------------------

def neighbor_counts(occupied, L, rows, N):
    grid = occupied.reshape(rows, L, L, L).astype(np.uint8)  # (rows, z, y, x)
    counts = (np.roll(grid, -1, axis=3) + np.roll(grid, 1, axis=3) +
              np.roll(grid, -1, axis=2) + np.roll(grid, 1, axis=2) +
              np.roll(grid, -1, axis=1) + np.roll(grid, 1, axis=1))
    return counts.reshape(rows, N)


def summarize(data, args):
    L = args.L
    sweeps = args.sweeps
    N = L * L * L
    energy = data["energy"].astype(np.float64)
    chains, samples = energy.shape
    quantiles = np.quantile(energy, (0.10, 0.25, 0.50, 0.75, 0.90),
                            axis=1).T
    energy_features = np.column_stack(
        (energy.mean(1), energy.std(1, ddof=1), quantiles))
    accept = np.column_stack((
        data["accept_local"].astype(np.float64).mean(1),
        data["accept_nonlocal"].astype(np.float64).mean(1)))

    raw = data["snapshots"].astype(np.uint8)
    snapshot_sweeps = data["snapshot_sweeps"]
    first = max(int(sweeps // 8), 1)
    selected = snapshot_sweeps >= first
    if not selected.any():
        selected[-1] = True
    configs = np.concatenate(
        (raw[selected], data["final"][None].astype(np.uint8)), axis=0) & 3
    count = configs.shape[0]
    occupancy = (configs != 0).mean(0).astype(np.float64)
    type1 = (configs == 1).mean(0).astype(np.float64)

    flat = configs.transpose(1, 0, 2).reshape(chains * count, N)
    counts = neighbor_counts(flat != 0, L, chains * count, N)
    environment = np.zeros((chains * count, 14), dtype=np.float64)
    for kind in (1, 2):
        base = (kind - 1) * 7
        for row in range(chains * count):
            local = counts[row][flat[row] == kind]
            if local.size:
                environment[row, base:base + 7] = np.bincount(
                    local.astype(np.intp), minlength=7)
    environment = environment.reshape(chains, count, 14).sum(1)
    for kind in (1, 2):
        block = environment[:, (kind - 1) * 7:kind * 7]
        total = block.sum(1, keepdims=True)
        environment[:, (kind - 1) * 7:kind * 7] = np.divide(
            block, total, out=np.zeros_like(block), where=total > 0)

    half = samples // 2
    drift_energy = (energy[:, half:].mean(1) - energy[:, :half].mean(1)
                    if half >= 1 else np.zeros(chains))
    mid = max(int(sweeps // 2), 1)
    early = (snapshot_sweeps >= first) & (snapshot_sweeps < mid)
    late = snapshot_sweeps >= mid
    drift_occupancy = None
    if early.any() and late.any():
        early_occ = ((raw[early] & 3) != 0).mean(0)
        late_occ = ((raw[late] & 3) != 0).mean(0)
        drift_occupancy = late_occ - early_occ
    return {
        "energy": energy_features,
        "accept": accept,
        "env": environment,
        "occ": occupancy,
        "type1": type1,
        "dE": drift_energy,
        "docc": drift_occupancy,
        "configs": int(count),
    }


# ---------------------------------------------------------------------------
# Permutation statistics
#
# Chains are the independent units. The two-sample tests relabel whole chains
# across backends, which is exact under the null (same joint law) and assumes
# nothing about the shape of the energy or occupancy distributions.
# ---------------------------------------------------------------------------

def _diff_vec(values, idx0, total, n):
    """Mean(CPU) - mean(Metal) for chain-level values, rows idx0 are CPU."""
    n0 = idx0.size
    n1 = n - n0
    sum0 = values[idx0].sum(0)
    return sum0 / n0 - (total - sum0) / n1


def _diff_sumsq(values, idx0, total, n):
    diff = _diff_vec(values, idx0, total, n)
    return float(np.sum(diff * diff))


def _standardized(block):
    """Divide each column by its total across-chain sd, dropping constants."""
    sd = block.std(0, ddof=1)
    keep = sd > 1e-12
    if not keep.any():
        return block[:, :0], sd[keep], keep
    return block[:, keep] / sd[keep], sd[keep], keep


def _obs_p(null, observed):
    return (1.0 + float(np.count_nonzero(null >= observed))) / (null.size + 1.0)


def _perm_p(null):
    ordered = np.sort(null)
    ge = null.size - np.searchsorted(ordered, null, side="left")
    return (1.0 + ge) / (null.size + 1.0)


def minp_joint(null, observed):
    """Union-intersection (min-p) test calibrated on the shared permutations."""
    obs_p = {name: _obs_p(null[name], observed[name]) for name in observed}
    combined = np.vstack([_perm_p(null[name]) for name in observed]).min(axis=0)
    joint = (1.0 + float(np.count_nonzero(
        combined <= min(obs_p.values())))) / (combined.size + 1.0)
    return joint, obs_p


def permutation_tests(summ, n_cpu, perms, rng):
    n = summ["energy"].shape[0]
    standardized = {}
    for name in ("energy", "accept", "env"):
        standardized[name] = _standardized(summ[name])
    totals = {name: standardized[name][0].sum(0) for name in standardized}
    occupancy_total = summ["occ"].sum(0)
    type1_total = summ["type1"].sum(0)

    def statistics(idx0):
        out = {}
        for name, (block, _sd, _keep) in standardized.items():
            key = "environment" if name == "env" else name
            out[key] = (_diff_sumsq(block, idx0, totals[name], n)
                        if block.shape[1] else 0.0)
        diff = _diff_vec(summ["occ"], idx0, occupancy_total, n)
        out["occupancy_l2"] = float(np.sum(diff * diff))
        out["occupancy_max"] = float(np.max(np.abs(diff)))
        diff = _diff_vec(summ["type1"], idx0, type1_total, n)
        out["type1_l2"] = float(np.sum(diff * diff))
        out["type1_max"] = float(np.max(np.abs(diff)))
        return out

    base = np.arange(n_cpu)
    observed = statistics(base)
    draws = np.argsort(rng.random((perms, n)), axis=1)[:, :n_cpu]
    null = {name: np.empty(perms) for name in observed}
    for perm in range(perms):
        current = statistics(draws[perm])
        for name, value in current.items():
            null[name][perm] = value

    effects = {}
    for name, raw_names in (("energy", ENERGY_NAMES),
                            ("accept", ACCEPT_NAMES),
                            ("env", ENV_NAMES)):
        block, _sd, keep = standardized[name]
        names = [label for label, flag in zip(raw_names, keep) if flag]
        differences = (_diff_vec(block, base, totals[name], n)
                       if names else np.zeros(0))
        effects[name] = {"names": names, "d": differences}

    occupancy_diff = _diff_vec(summ["occ"], base, occupancy_total, n)
    type1_diff = _diff_vec(summ["type1"], base, type1_total, n)
    effects["occupancy"] = {
        "rms": float(np.sqrt(np.mean(occupancy_diff ** 2))),
        "max": float(np.max(np.abs(occupancy_diff))),
        "null95": float(np.quantile(null["occupancy_max"], 0.95)),
        "mean_shift": float(np.mean(occupancy_diff)),
    }
    effects["type1"] = {
        "rms": float(np.sqrt(np.mean(type1_diff ** 2))),
        "max": float(np.max(np.abs(type1_diff))),
        "null95": float(np.quantile(null["type1_max"], 0.95)),
        "mean_shift": float(np.mean(type1_diff)),
    }
    return observed, null, effects


def _signflip_stats(signs, values):
    """Sum of squared column means after random sign flips, one value per row."""
    chains = values.shape[0]
    out = np.empty(signs.shape[0])
    chunk = max(1, 2000000 // max(values.shape[1], 1))
    for start in range(0, signs.shape[0], chunk):
        stop = min(start + chunk, signs.shape[0])
        means = (signs[start:stop] @ values) / chains
        out[start:stop] = np.sum(means * means, axis=1)
    return out


def stationarity(drift_energy, drift_occupancy, perms, rng):
    """Geweke-style split diagnostic: early vs late production window.

    Per-chain early/late differences are exchangeable about zero if the chain
    is stationary over the production window. A significant joint result means
    the burn-in is too short (or mixing is too slow) to compare ensembles.
    """
    chains = drift_energy.size
    sd = drift_energy.std(ddof=1)
    zscore = drift_energy / sd if sd > 0 else np.zeros(chains)
    signs = rng.integers(0, 2, size=(perms, chains), dtype=np.int8) * 2 - 1
    null = {"energy": _signflip_stats(signs, zscore[:, None])}
    observed = {"energy": float(zscore.mean() ** 2)}
    rms_occupancy = None
    if drift_occupancy is not None and drift_occupancy.size:
        null["occupancy"] = _signflip_stats(signs, drift_occupancy)
        mean_diff = drift_occupancy.mean(0)
        observed["occupancy"] = float(np.sum(mean_diff * mean_diff))
        rms_occupancy = float(np.sqrt(np.mean(mean_diff * mean_diff)))
    joint, obs_p = minp_joint(null, observed)
    return {
        "p": joint,
        "p_energy": obs_p["energy"],
        "p_occupancy": obs_p.get("occupancy"),
        "d_z": float(zscore.mean()),
        "rms_occupancy": rms_occupancy,
    }


# ---------------------------------------------------------------------------
# CLI and report
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Distributional CPU/Metal equivalence gate.")
    bath = parser.add_argument_group("bath and schedule")
    bath.add_argument("--L", type=int, default=21,
                      help="linear size (default 21; Metal's coloured local "
                           "schedule exists only at L=21)")
    bath.add_argument("--beta", default="2.0", help="inverse temperature text")
    bath.add_argument("--rho", default="0.75", help="total density")
    bath.add_argument("--rho1", default="0.30", help="type-1 density")
    bath.add_argument("--rho2", default="0.45", help="type-2 density")
    bath.add_argument("--mode", choices=("both", "nonlocal", "local"),
                      default="both", help="proposal set per sweep")
    bath.add_argument("--sweeps", type=int, default=2048,
                      help="production sweeps after burn-in (default 2048)")
    bath.add_argument("--burn-in", type=int, default=2048,
                      help="discarded equilibration sweeps (default 2048 at "
                           "L=21; energy relaxes over ~1000 sweeps)")
    bath.add_argument("--chains", type=int, default=16,
                      help="independent chains per run (default 16)")
    bath.add_argument("--runs", type=int, default=4,
                      help="driver runs per backend (default 4)")
    stats = parser.add_argument_group("statistics")
    stats.add_argument("--seed", type=int, default=20261110,
                       help="master seed; streams are derived per backend/run")
    stats.add_argument("--perms", type=int, default=2000,
                       help="permutation replicates (default 2000)")
    stats.add_argument("--alpha", type=float, default=0.01,
                       help="gate level for distributional tests (default 0.01)")
    stats.add_argument("--acceptance-gate", choices=("warn", "fail", "off"),
                       default="warn",
                       help="acceptance-rate policy (default warn)")
    stats.add_argument("--stationarity-alpha", type=float, default=1e-3,
                       help="level for the split stationarity check")
    stats.add_argument("--require-stationary",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="treat a failed stationarity check as inconclusive")
    drivers = parser.add_argument_group("drivers")
    drivers.add_argument("--cpu-driver")
    drivers.add_argument("--metal-driver")
    drivers.add_argument("--timeout", type=float, default=3600.0,
                         help="per-run timeout in seconds")
    drivers.add_argument("--skip-if-no-metal", action="store_true",
                         help="exit 0 with SKIP when Metal is unavailable")
    output = parser.add_argument_group("output")
    output.add_argument("--workdir", help="keep run artifacts here")
    output.add_argument("--json", dest="json_path",
                        help="write a JSON summary to this path")
    return parser


def _validate_args(args):
    if args.L < 4:
        raise GateError("--L must be >= 4")
    if not 16 <= args.sweeps <= 1000000:
        raise GateError("--sweeps must be in [16, 1000000] so both backends "
                        "share the same snapshot plan")
    if args.burn_in < 0:
        raise GateError("--burn-in must be non-negative")
    if args.chains < 2:
        raise GateError("--chains must be >= 2")
    if args.runs < 1:
        raise GateError("--runs must be >= 1")
    if args.chains * args.runs < 8:
        raise GateError("need at least 8 chains per backend "
                        "(--chains x --runs)")
    if not 0.0 < args.alpha < 1.0:
        raise GateError("--alpha must be in (0, 1)")
    if not 0.0 < args.stationarity_alpha < 1.0:
        raise GateError("--stationarity-alpha must be in (0, 1)")
    if args.perms < max(199, int(1.0 / args.alpha)):
        raise GateError("--perms is too small to resolve --alpha")
    if args.seed < 0:
        raise GateError("--seed must be non-negative")
    try:
        beta = float(args.beta)
        rho = float(args.rho)
        rho1 = float(args.rho1)
        rho2 = float(args.rho2)
    except ValueError:
        raise GateError("--beta/--rho/--rho1/--rho2 must be numeric")
    if not all(math.isfinite(value) for value in (beta, rho, rho1, rho2)):
        raise GateError("bath parameters must be finite")
    if abs(rho1 + rho2 - rho) > 1e-8:
        raise GateError("rho1 + rho2 must equal rho (got %r)" % (rho1 + rho2))
    if min(rho1, rho2) < 0 or rho <= 0:
        raise GateError("densities must be non-negative and rho > 0")
    N = args.L ** 3
    if N > 9261:
        raise GateError("Metal kernel supports N <= 9261 (L <= 21)")
    n1 = int(math.floor(rho1 * N + 0.5))
    n2 = int(math.floor(rho2 * N + 0.5))
    if n1 + n2 > N:
        raise GateError("round(rho1*N) + round(rho2*N) exceeds N=%d" % N)
    if n1 + n2 <= 0:
        raise GateError("empty occupied set")
    return {"N": N, "n1": n1, "n2": n2}


def _d_value(effects, block, key):
    names = effects[block]["names"]
    if key not in names:
        return None
    return float(effects[block]["d"][names.index(key)])


def _max_d(effects, block):
    values = effects[block]["d"]
    if values.size == 0:
        return 0.0, "-"
    index = int(np.argmax(np.abs(values)))
    return float(values[index]), effects[block]["names"][index]


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        geometry = _validate_args(args)
    except GateError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    N = geometry["N"]
    n1 = geometry["n1"]
    n2 = geometry["n2"]

    try:
        cpu_driver = find_driver("CPU", args.cpu_driver)
    except GateError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    try:
        metal_driver = find_driver("GPU", args.metal_driver)
    except GateError as exc:
        if args.skip_if_no_metal:
            print("SKIP: Metal backend unavailable: %s" % exc)
            return 0
        print("error: %s" % exc, file=sys.stderr)
        return 2
    print("CPU driver:   %s" % cpu_driver)
    print("Metal driver: %s" % metal_driver)
    for driver, backend in ((cpu_driver, "CPU"), (metal_driver, "GPU")):
        ok, why = probe(driver, backend, args.timeout)
        if not ok:
            if backend == "GPU" and args.skip_if_no_metal:
                print("SKIP: Metal backend unavailable: %s" % why)
                return 0
            print("error: %s probe failed: %s" % (backend, why),
                  file=sys.stderr)
            return 2

    workdir = os.path.abspath(args.workdir) if args.workdir else \
        tempfile.mkdtemp(prefix="solid-stat-")
    if args.workdir:
        os.makedirs(workdir, exist_ok=True)
    try:
        pooled, drifts, walls = {}, {}, {}
        signature = None
        for backend, driver in (("CPU", cpu_driver), ("GPU", metal_driver)):
            label = "CPU" if backend == "CPU" else "Metal"
            runs = []
            drifts[label] = []
            walls[label] = 0.0
            for run in range(args.runs):
                initial_rng = np.random.default_rng(np.random.SeedSequence([
                    args.seed & 0xFFFFFFFF,
                    0xC0FFEE if label == "CPU" else 0xBEEF, run]))
                packed = make_states(args.L, n1, n2, args.chains, initial_rng,
                                     self_check=(run == 0))
                state_path = os.path.join(workdir, "init_%s_%d.bin"
                                          % (label, run))
                packed.tofile(state_path)
                seed = derive_seed(args.seed, label, run)
                data, wall = run_driver(driver, backend, args, seed,
                                        state_path, workdir, args.timeout)
                validate(data, args, n1, n2, N, packed, seed)
                run_signature = (
                    tuple(int(value) for value in data["meta"]),
                    tuple(float(value) for value in data["scalars"]),
                    int(data["history_every"][0]),
                    tuple(int(value) for value in data["snapshot_sweeps"]))
                if signature is None:
                    signature = run_signature
                elif run_signature != signature:
                    raise GateError(
                        "bath/schedule metadata differs between runs: %r vs %r"
                        % (run_signature, signature))
                summary = summarize(data, args)
                runs.append(summary)
                drifts[label].append((summary["dE"], summary["docc"]))
                walls[label] += wall
                print("  %s run %d/%d: %.2fs" % (label, run + 1, args.runs,
                                                 wall))
            pooled[label] = {
                key: np.concatenate([run[key] for run in runs], axis=0)
                for key in ("energy", "accept", "env", "occ", "type1")
            }
            pooled[label]["configs"] = runs[0]["configs"]

        keys = ("energy", "accept", "env", "occ", "type1")
        combined = {
            key: np.concatenate([pooled["CPU"][key], pooled["Metal"][key]],
                                axis=0)
            for key in keys
        }
        n_cpu = pooled["CPU"]["energy"].shape[0]
        stats_started = time.perf_counter()
        perm_rng = np.random.default_rng(np.random.SeedSequence(
            [(args.seed + 1) & 0xFFFFFFFF, 0x5EED]))
        observed, null, effects = permutation_tests(
            combined, n_cpu, args.perms, perm_rng)
        joint_p, component_p = minp_joint(
            {name: null[name] for name in GATING},
            {name: observed[name] for name in GATING})
        accept_p = _obs_p(null["accept"], observed["accept"])

        station = {}
        for label in ("CPU", "Metal"):
            drift_energy = np.concatenate([item[0] for item in drifts[label]])
            drift_occ = [item[1] for item in drifts[label]
                         if item[1] is not None]
            drift_occupancy = (np.concatenate(drift_occ, axis=0)
                               if drift_occ else None)
            stat_rng = np.random.default_rng(np.random.SeedSequence([
                (args.seed + 2) & 0xFFFFFFFF,
                0x1234 if label == "CPU" else 0x4321]))
            station[label] = stationarity(drift_energy, drift_occupancy,
                                          args.perms, stat_rng)

        stats_wall = time.perf_counter() - stats_started
        accept_warn = (args.acceptance_gate == "warn"
                       and accept_p < args.alpha)
        failed = joint_p < args.alpha
        if args.acceptance_gate == "fail" and accept_p < args.alpha:
            failed = True
        inconclusive = (args.require_stationary and any(
            station[label]["p"] < args.stationarity_alpha
            for label in station))
        verdict = ("FAIL" if failed else
                   "INCONCLUSIVE" if inconclusive else "PASS")

        print("\nbath: L=%d beta=%s rho=%s (rho1=%s rho2=%s) mode=%s"
              % (args.L, args.beta, args.rho, args.rho1, args.rho2,
                 args.mode))
        print("      burn-in=%d production=%d sweeps; %d runs x %d chains per "
              "backend; %d occupancy configurations per chain (snapshots plus "
              "final)" % (args.burn_in, args.sweeps, args.runs, args.chains,
                          pooled["CPU"]["configs"]))
        print("      initial states: independent draws, %d type-1 + %d type-2 "
              "sites (D0)" % (n1, n2))
        print("      alpha=%.3g permutations=%d acceptance-policy=%s"
              % (args.alpha, args.perms, args.acceptance_gate))

        print("\n  backend  runs  chains  E[mean]+-SE          E[sd]    "
              "acc_local  acc_nonlocal")
        for label in ("CPU", "Metal"):
            e0 = pooled[label]["energy"][:, 0]
            chains = e0.size
            se = e0.std(ddof=1) / math.sqrt(chains)
            print("  %-7s %5d %7d  %10.3f +- %-7.3f  %8.3f  %9.5f  %9.5f"
                  % (label, args.runs, chains, e0.mean(), se,
                     pooled[label]["energy"][:, 1].mean(),
                     pooled[label]["accept"][:, 0].mean(),
                     pooled[label]["accept"][:, 1].mean()))

        print("\n  component                statistic       p       effect size")
        for name in GATING:
            title = {"energy": "energy distribution",
                     "occupancy_l2": "occupancy field L2",
                     "occupancy_max": "occupancy field max",
                     "type1_l2": "type-1 field L2",
                     "type1_max": "type-1 field max",
                     "environment": "local environment"}[name]
            if name == "energy":
                d_mean = _d_value(effects, "energy", "mean")
                d_sd = _d_value(effects, "energy", "sd")
                d_max, d_name = _max_d(effects, "energy")
                detail = ("d(mean)=%+.3f d(sd)=%+.3f max|d|=%.3f (%s)"
                          % (d_mean or 0.0, d_sd or 0.0, d_max, d_name))
            elif name == "environment":
                d_max, d_name = _max_d(effects, "env")
                detail = "max|d|=%.3f (%s)" % (d_max, d_name)
            elif name.startswith("occupancy"):
                occ = effects["occupancy"]
                detail = ("rms=%.4f max=%.4f (null95=%.4f)"
                          % (occ["rms"], occ["max"], occ["null95"]))
            else:
                typ = effects["type1"]
                detail = ("rms=%.4f max=%.4f (null95=%.4f)"
                          % (typ["rms"], typ["max"], typ["null95"]))
            print("  %-23s T=%-11.4g %7.4f  %s"
                  % (title, observed[name], component_p[name], detail))
        print("  %-23s %-13s %7.4f  %s"
              % ("joint (min-p, 6)", "", joint_p,
                 "PASS" if joint_p >= args.alpha else "FAIL"))

        d_local = _d_value(effects, "accept", "local")
        d_nonlocal = _d_value(effects, "accept", "nonlocal")
        accept_note = {"warn": "[WARN]", "fail": "[FAIL]",
                       "off": "[reported]"}[args.acceptance_gate] \
            if accept_p < args.alpha else ""
        print("  %-23s T=%-11.4g %7.4f  d(local)=%+.3f d(nonlocal)=%+.3f %s"
              % ("acceptance", observed["accept"], accept_p,
                 d_local or 0.0, d_nonlocal or 0.0, accept_note))

        print()
        for label in ("CPU", "Metal"):
            stats = station[label]
            if stats["p_occupancy"] is None:
                detail = "d_z(energy)=%+.3f" % stats["d_z"]
            else:
                detail = ("d_z(energy)=%+.3f rms(density)=%.4f"
                          % (stats["d_z"], stats["rms_occupancy"]))
            state = ("FAIL" if stats["p"] < args.stationarity_alpha
                     else "ok")
            print("  stationarity %-5s p=%.4f (energy %.4f, density %s) %s "
                  "[%s]" % (label, stats["p"], stats["p_energy"],
                            "%.4f" % stats["p_occupancy"]
                            if stats["p_occupancy"] is not None else "n/a",
                            detail, state))
        if inconclusive:
            print("  note: a stationarity check failed; increase --burn-in or "
                  "--sweeps before trusting the comparison")

        print("\nstatistical gate: %s (alpha=%.3g, B=%d)"
              % (verdict, args.alpha, args.perms))
        print("runtime: CPU %.1fs Metal %.1fs (statistics %.1fs)"
              % (walls["CPU"], walls["Metal"], stats_wall))
        if args.workdir:
            print("artifacts: %s" % workdir)

        result = {
            "config": {
                "L": args.L, "beta": args.beta, "rho": args.rho,
                "rho1": args.rho1, "rho2": args.rho2, "mode": args.mode,
                "sweeps": args.sweeps, "burn_in": args.burn_in,
                "chains": args.chains, "runs": args.runs, "seed": args.seed,
                "perms": args.perms, "alpha": args.alpha,
                "acceptance_gate": args.acceptance_gate,
                "stationarity_alpha": args.stationarity_alpha,
                "require_stationary": args.require_stationary,
                "n1": n1, "n2": n2,
                "snapshots_per_chain": pooled["CPU"]["configs"],
            },
            "drivers": {"cpu": cpu_driver, "metal": metal_driver},
            "backends": {
                label: {
                    "chains": int(pooled[label]["energy"].shape[0]),
                    "energy_mean": float(pooled[label]["energy"][:, 0].mean()),
                    "energy_mean_se": float(
                        pooled[label]["energy"][:, 0].std(ddof=1)
                        / math.sqrt(pooled[label]["energy"].shape[0])),
                    "energy_sd": float(
                        pooled[label]["energy"][:, 1].mean()),
                    "accept_local": float(
                        pooled[label]["accept"][:, 0].mean()),
                    "accept_nonlocal": float(
                        pooled[label]["accept"][:, 1].mean()),
                    "occupancy_mean": float(pooled[label]["occ"].mean()),
                    "wall_seconds": walls[label],
                } for label in ("CPU", "Metal")
            },
            "components": {
                name: {"statistic": float(observed[name]),
                       "p": float(component_p[name])}
                for name in GATING
            },
            "joint_p": float(joint_p),
            "statistics_seconds": stats_wall,
            "effects": {
                "energy": dict(zip(effects["energy"]["names"],
                                   [float(v) for v in
                                    effects["energy"]["d"]])),
                "acceptance": dict(zip(effects["accept"]["names"],
                                       [float(v) for v in
                                        effects["accept"]["d"]])),
                "environment_max_abs_d": _max_d(effects, "env")[0],
                "occupancy": effects["occupancy"],
                "type1": effects["type1"],
            },
            "acceptance": {"p": float(accept_p),
                           "policy": args.acceptance_gate,
                           "warn": bool(accept_warn)},
            "stationarity": station,
            "verdict": verdict,
            "exit_code": 1 if failed else 2 if inconclusive else 0,
        }
        if args.json_path:
            try:
                with open(args.json_path, "w", encoding="utf-8") as handle:
                    json.dump(_jsonable(result), handle, indent=2,
                              sort_keys=True)
                    handle.write("\n")
            except OSError as exc:
                raise GateError("cannot write %s: %s" % (args.json_path, exc))
            print("json: %s" % os.path.abspath(args.json_path))
    except GateError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    finally:
        if not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    return 1 if failed else 2 if inconclusive else 0


if __name__ == "__main__":
    sys.exit(main())