#!/usr/bin/env python3
"""Deterministic CPU CLI regressions for seeds and bounded production history."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np


DRIVER = Path(__file__).resolve().with_name("driver")
L = 4
CHAINS = 2
HISTORY_KEYS = ("energy", "accept_local", "accept_nonlocal")
PHYSICAL_KEYS = (
    "initial", "equilibrated", "final", "snapshots", "snapshot_sweeps",
    "scalars", "meta",
)
KEYS = PHYSICAL_KEYS + HISTORY_KEYS + ("seed", "history_every", "history_sweeps", "start_sweep")
MODES = {"both": 0, "nonlocal": 1, "local": 2}
SEEDS = (1, (1 << 32) + 1, (1 << 63) + 1, (1 << 64) - 1)


def equal(actual, expected, label):
    np.testing.assert_array_equal(actual, expected, err_msg=label)


def same_fields(actual, expected, keys, label):
    for key in keys:
        equal(actual[key], expected[key], f"{label}: {key}")


def run_case(*, sweeps=17, mode="both", equil=0, seed=None,
             history_every=None, max_samples=None, raw=None, start_sweep=None):
    command = [str(DRIVER), str(L), "1.25", "0.75", "0.30", "0.45",
               str(sweeps), str(CHAINS), "CPU", mode, str(equil)]
    with tempfile.TemporaryDirectory(prefix="solid-production-") as directory:
        work = Path(directory)
        if raw is not None:
            (work / "initial.raw").write_bytes(raw)
            command.append("initial.raw")
        for option, value in (("--seed", seed),
                              ("--history-every", history_every),
                              ("--max-history-samples", max_samples),
                              ("--start-sweep", start_sweep)):
            if value is not None:
                command += [option, str(value)]
        result = subprocess.run(command, cwd=work, text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(
                f"driver exited {result.returncode}: {' '.join(command)}\n"
                f"{result.stdout}{result.stderr}"
            )
        with np.load(work / "run.npz", allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}

    effective_seed = 1 if seed is None else seed
    requested = 1 if history_every is None else history_every
    cap = 100000 if max_samples is None else max_samples
    interval = max(requested, (sweeps + cap - 1) // cap)
    timestamps = list(range(interval, sweeps + 1, interval))
    if sweeps and (not timestamps or timestamps[-1] != sweeps):
        timestamps.append(sweeps)
    for key in ("seed", "history_every", "history_sweeps"):
        assert data[key].dtype == np.dtype("uint64"), f"{key} must be uint64"
    equal(data["seed"], np.array([effective_seed], dtype=np.uint64), "seed metadata")
    equal(data["history_every"], np.array([interval], dtype=np.uint64),
          "effective history interval")
    equal(data["history_sweeps"], np.array(timestamps, dtype=np.uint64),
          "production-relative history timestamps")
    equal(data["meta"], np.array([L, CHAINS, sweeps, MODES[mode], equil],
                                  dtype=np.uint64), "legacy five-element metadata")
    for key in HISTORY_KEYS:
        assert data[key].shape == (CHAINS, len(timestamps)), (key, data[key].shape)
    assert len(timestamps) <= cap, "history exceeds configured sample cap"
    for key in ("initial", "equilibrated", "final"):
        assert data[key].shape == (CHAINS, L ** 3), (key, data[key].shape)
    assert data["snapshots"].shape == (len(data["snapshot_sweeps"]), CHAINS, L ** 3)
    return data


def check_seeds():
    default = run_case()
    explicit = run_case(seed=1)
    same_fields(default, explicit, KEYS, "default seed is one")
    seeded = [explicit] + [run_case(seed=seed) for seed in SEEDS[1:]]
    replay = run_case(seed=SEEDS[-1])
    same_fields(replay, seeded[-1], KEYS, "same unsigned 64-bit seed replays exactly")
    for index, first in enumerate(seeded):
        for second in seeded[index + 1:]:
            assert not np.array_equal(first["initial"], second["initial"]), (
                "fixed seeds, including high-bit differences, must change initialization"
            )

    # Import exactly the same packed state for every seed. Initialization cannot
    # explain any differences here: only the per-proposal stream can change it.
    raw = default["initial"].tobytes()
    for mode in MODES:
        proposals = [run_case(seed=seed, mode=mode, raw=raw) for seed in SEEDS]
        for data in proposals:
            equal(data["initial"], default["initial"], f"{mode}: imported initial state")
            equal(data["equilibrated"], default["initial"], f"{mode}: no equilibration")
        repeated = run_case(seed=SEEDS[-1], mode=mode, raw=raw)
        same_fields(repeated, proposals[-1], KEYS, f"{mode}: imported-state replay")
        for index, first in enumerate(proposals):
            for second in proposals[index + 1:]:
                assert not np.array_equal(first["final"], second["final"]), (
                    f"{mode}: fixed seeds must change proposals from identical raw state"
                )
    print("OK: default seed, full-width seed replay, initialization and imported-state proposals")


def compare_sampled(full, sampled, label):
    same_fields(sampled, full, PHYSICAL_KEYS + ("seed",), label)
    columns = sampled["history_sweeps"].astype(np.intp) - 1
    for key in HISTORY_KEYS:
        equal(sampled[key], full[key][:, columns], f"{label}: sampled {key}")


def check_sampling():
    for mode in MODES:
        full = run_case(mode=mode, equil=5, seed=SEEDS[1])
        # Explicit expected boundaries also keep the oracle independent of the
        # general timestamp calculation in run_case.
        cases = (
            ({"max_samples": 17}, 1, list(range(1, 18))),
            ({"max_samples": 16}, 2, [2, 4, 6, 8, 10, 12, 14, 16, 17]),
            ({"max_samples": 8}, 3, [3, 6, 9, 12, 15, 17]),
            ({"history_every": 5}, 5, [5, 10, 15, 17]),
            ({"history_every": 2, "max_samples": 4}, 5, [5, 10, 15, 17]),
            ({"max_samples": 1}, 17, [17]),
            ({"history_every": 23}, 23, [17]),
        )
        for options, interval, timestamps in cases:
            sampled = run_case(mode=mode, equil=5, seed=SEEDS[1], **options)
            label = f"{mode}: {options}"
            equal(sampled["history_every"], [interval], label)
            equal(sampled["history_sweeps"], timestamps, label)
            compare_sampled(full, sampled, label)
        print(f"OK: {mode} full/decimated trajectories, cap boundaries and final partial interval")


def check_equilibration():
    for mode in MODES:
        # The first eight sweeps of the uninterrupted run are equilibration in
        # the split run. Their histories must not enter the production archive.
        continuous = run_case(sweeps=17, mode=mode, seed=SEEDS[-1])
        split = run_case(sweeps=9, equil=8, mode=mode, seed=SEEDS[-1])
        equal(split["initial"], continuous["initial"], f"{mode}: split initial state")
        at_eight = np.flatnonzero(continuous["snapshot_sweeps"] == 8)
        assert at_eight.size == 1, "missing deterministic sweep-eight snapshot"
        equal(split["equilibrated"], continuous["snapshots"][at_eight[0]],
              f"{mode}: equilibrated state is uninterrupted sweep eight")
        equal(split["final"], continuous["final"], f"{mode}: same total physical evolution")
        for key in HISTORY_KEYS:
            equal(split[key], continuous[key][:, 8:], f"{mode}: equilibration excluded from {key}")
        sparse = run_case(sweeps=9, equil=8, mode=mode, seed=SEEDS[-1], max_samples=2)
        compare_sampled(split, sparse, f"{mode}: sampled after discarded equilibration")

        for equil in (0, 8):
            empty = run_case(sweeps=0, equil=equil, mode=mode, seed=SEEDS[-1],
                             history_every=3, max_samples=1)
            equal(empty["equilibrated"], split["equilibrated"] if equil else split["initial"],
                  f"{mode}: zero production with equil={equil}")
            equal(empty["final"], empty["equilibrated"], f"{mode}: no production updates")
            equal(empty["snapshot_sweeps"], np.empty(0, dtype=np.uint64),
                  f"{mode}: no production snapshots")
    print("OK: discarded equilibration histories, uninterrupted evolution and zero production")


def check_resume():
    for mode in MODES:
        initial = run_case(sweeps=0, mode=mode, seed=SEEDS[-1])["initial"].tobytes()
        # Crossing 2^32 also catches accidental narrowing of the counter offset.
        offset = (1 << 32) - 4
        continuous = run_case(sweeps=17, mode=mode, seed=SEEDS[-1],
                              raw=initial, start_sweep=offset)
        first = run_case(sweeps=8, mode=mode, seed=SEEDS[-1],
                         raw=initial, start_sweep=offset)
        resumed = run_case(sweeps=9, mode=mode, seed=SEEDS[-1],
                           raw=first["final"].tobytes(), start_sweep=offset + 8)
        equal(resumed["final"], continuous["final"], f"{mode}: exact counter-based resume")
        for key in HISTORY_KEYS:
            equal(resumed[key], continuous[key][:, 8:], f"{mode}: resumed {key}")
    print("OK: raw-state resume preserves full-width proposal counters")


def main():
    check_seeds()
    check_sampling()
    check_equilibration()
    check_resume()
    print("All CPU production regressions passed.")


if __name__ == "__main__":
    main()
