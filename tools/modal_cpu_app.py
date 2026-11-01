import concurrent.futures as cf
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

app = modal.App("lattice-cpu-campaign")
CPU = int(os.environ.get("MODAL_CPU", "32"))
LADDERS = int(os.environ.get("MODAL_LADDERS", "32"))
HISTORY_EVERY = int(os.environ.get("HISTORY_EVERY", "100"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "1000"))
image = (
    modal.Image.debian_slim()
    .apt_install("build-essential")
    .add_local_dir("cpu", "/root/cpu", copy=True)
    .run_commands("cd /root/cpu && make clean && make CXX=g++ lattice_single")
)
volume = modal.Volume.from_name("lattice-runs", create_if_missing=True)


def ladder_done(run_dir: Path) -> bool:
    done = run_dir / "done.txt"
    return done.exists() and "complete=1" in done.read_text()


@app.function(image=image, cpu=CPU, memory=65536, volumes={"/runs": volume}, timeout=3600)
def run_ladders(run_name: str, ladder_ids: list[int], cfg_json: str):
    cfg = json.loads(cfg_json)
    volume.reload()
    root = Path("/runs") / run_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign.json").write_text(json.dumps(cfg, indent=2) + "\n")
    binary = "/root/cpu/lattice_single"
    procs = []
    status = {}
    for lid in ladder_ids:
        run_dir = root / f"ladder_{lid:06d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if ladder_done(run_dir):
            status[lid] = "done"
            continue
        args = [
            binary,
            "--run-dir", str(run_dir),
            "--beta", str(cfg["beta"]),
            "--sweeps", str(cfg["sweeps"]),
            "--sample-every", str(cfg["sample_every"]),
            "--checkpoint-every", str(cfg["checkpoint_every"]),
            "--history-every", str(cfg["history_every"]),
            "--local", str(cfg["local"]),
            "--seed", str(cfg["seed"] + 104729 * lid),
        ]
        if (run_dir / "checkpoint.bin").exists():
            args.append("--resume")
        log = (run_dir / "run.log").open("w")
        procs.append((lid, subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT), log))
        status[lid] = "running"

    stop_commit = threading.Event()
    def committer():
        while not stop_commit.wait(10.0):
            try:
                volume.commit()
            except Exception:
                pass
    commit_thread = threading.Thread(target=committer, daemon=True)
    commit_thread.start()
    for lid, proc, log in procs:
        rc = proc.wait()
        log.close()
        status[lid] = "ok" if rc == 0 else f"failed:{rc}"
    stop_commit.set()
    commit_thread.join(timeout=10.0)
    volume.commit()
    complete = sum(1 for lid in ladder_ids if ladder_done(root / f"ladder_{lid:06d}"))
    return {"run": run_name, "ladders": len(ladder_ids), "complete": complete, "status": status}


def launch_dataset():
    prefix = os.environ.get("RUN_PREFIX", "supercooled_v1")
    ladders = int(os.environ.get("LADDERS_PER_BETA", "10000"))
    group = int(os.environ.get("GROUP_SIZE", "64"))
    workers = int(os.environ.get("MAX_CONCURRENT", "16"))
    betas = [2.0 * i / 9.0 for i in range(10)]
    base_cfg = {
        "sweeps": int(os.environ.get("SWEEPS", "100000")),
        "sample_every": 10,
        "checkpoint_every": CHECKPOINT_EVERY,
        "history_every": HISTORY_EVERY,
        "local": 50,
        "seed": 1,
    }
    jobs = []
    for j, beta in enumerate(betas):
        run_name = f"{prefix}/b{j:02d}_b{beta:.3f}"
        cfg = dict(base_cfg, beta=beta)
        for start in range(0, ladders, group):
            ids = list(range(start, min(start + group, ladders)))
            jobs.append((run_name, ids, json.dumps(cfg)))
    print(f"dataset jobs={len(jobs)} betas={betas} ladders_per_beta={ladders} workers={workers}", flush=True)
    failures = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_ladders.remote, run_name, ids, cfg)
                for run_name, ids, cfg in jobs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                res = fut.result()
                print(f"[{i}/{len(futs)}] {res['run']} complete={res['complete']}/{res['ladders']}", flush=True)
            except Exception as exc:
                failures += 1
                print(f"[{i}/{len(futs)}] FAILED {exc!r}", flush=True)
                if os.environ.get("STOP_ON_ERROR") == "1":
                    break
    print(f"dataset done failures={failures}", flush=True)
    return failures


@app.local_entrypoint()
def main():
    if os.environ.get("DATASET") == "1":
        launch_dataset()
        return
    run_name = os.environ.get("RUN_NAME", "fixed_beta")
    cfg = {
        "beta": float(os.environ.get("BETA", "2.0")),
        "sweeps": int(os.environ.get("SWEEPS", "100000")),
        "sample_every": 10,
        "checkpoint_every": CHECKPOINT_EVERY,
        "history_every": HISTORY_EVERY,
        "local": 50,
        "seed": 1,
    }
    result = run_ladders.remote(run_name, list(range(LADDERS)), json.dumps(cfg))
    print(json.dumps(result, indent=2))
