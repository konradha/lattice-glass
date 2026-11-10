#!/usr/bin/env python3
"""Replay downloaded Modal CUDA campaign chunks against this branch's CPU model."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="downloaded directory containing manifest.json")
    parser.add_argument("--driver", type=Path,
                        default=Path(__file__).resolve().parents[1] / "solid" / "driver")
    args = parser.parse_args()
    driver = args.driver.resolve()
    manifest = json.loads((args.campaign / "manifest.json").read_text())
    checked = 0
    fields = ("initial", "equilibrated", "final", "energy", "accept_local",
              "accept_nonlocal", "scalars", "meta", "seed", "history_every",
              "history_sweeps", "start_sweep")
    for job in manifest["jobs"]:
        previous = None
        for chunk in job["chunks"]:
            path = args.campaign / chunk["file"]
            with np.load(path, allow_pickle=False) as archive:
                gpu = {key: archive[key].copy() for key in archive.files}
            if previous is not None:
                np.testing.assert_array_equal(gpu["initial"], previous,
                                              err_msg=f"{path.name}: checkpoint continuity")
            command = list(chunk["gpu_argv"])
            if len(command) < 11 or command[8] != "CUDA":
                raise ValueError(f"Not a CUDA driver invocation: {path}")
            command[0], command[8] = str(driver), "CPU"
            with tempfile.TemporaryDirectory(prefix="cuda-cpu-replay-") as work:
                work = Path(work)
                gpu["initial"].tofile(work / "initial.raw")
                if len(command) > 11 and not command[11].startswith("--"):
                    command[11] = "initial.raw"
                else:
                    command.insert(11, "initial.raw")
                subprocess.run(command, cwd=work, check=True, capture_output=True, text=True)
                with np.load(work / "run.npz", allow_pickle=False) as cpu:
                    for field in fields:
                        np.testing.assert_array_equal(gpu[field], cpu[field],
                                                      err_msg=f"{path.name}: {field}")
                    # Before one million sweeps all snapshot boundaries are deterministic.
                    if int(gpu["meta"][2]) <= 1000000:
                        for field in ("snapshot_sweeps", "snapshots"):
                            np.testing.assert_array_equal(gpu[field], cpu[field],
                                                          err_msg=f"{path.name}: {field}")
            previous = gpu["final"]
            checked += 1
            print(f"PASS {path.name}", flush=True)
    if not checked:
        raise ValueError("No completed CUDA chunks to replay")
    print(json.dumps({"cpu_replayed_chunks": checked, "ok": True}))


if __name__ == "__main__":
    main()
