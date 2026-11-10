import concurrent.futures as cf
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

app = modal.App("lattice-cpu-campaign")
CPU = int(os.environ.get("MODAL_CPU", "8"))
LADDERS = int(os.environ.get("MODAL_LADDERS", "32"))
HISTORY_EVERY = int(os.environ.get("HISTORY_EVERY", "100"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "10000"))
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


@app.function(image=image, cpu=CPU, memory=2048, volumes={"/runs": volume}, timeout=3600)
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
        if cfg.get("adaptive_burn_in"):
            args += [
                "--adaptive-burn-in",
                "--burn-min", str(cfg["burn_min"]),
                "--burn-max", str(cfg["burn_max"]),
                "--post-burn", str(cfg["post_burn"]),
                "--burn-acf-every", str(cfg["burn_acf_every"]),
                "--burn-max-lag", str(cfg["burn_max_lag"]),
                "--burn-check-every", str(cfg["burn_check_every"]),
                "--burn-zero-window", str(cfg["burn_zero_window"]),
            ]
        args += ["--max-snapshots", str(cfg.get("max_snapshots", 0))]
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

    # One packed config file per group. Retries overwrite the same group tag,
    # so a resumed run cannot create duplicates.
    pack_dir = root / "packed"
    pack_dir.mkdir(parents=True, exist_ok=True)
    tag = f"group_{min(ladder_ids):06d}"
    bin_path = pack_dir / f"{tag}.bin"
    idx_path = pack_dir / f"{tag}.csv"
    nbytes = 15625
    with bin_path.open("wb") as fb, idx_path.open("w") as fi:
        fi.write("ladder_id,offset,bytes,sweeps_done,burn_in_resolved,burn_sweeps,energy,status\n")
        off = 0
        for lid in ladder_ids:
            d = root / f"ladder_{lid:06d}"
            cp = d / "configs" / "config.bin"
            if not cp.exists():
                continue
            data = cp.read_bytes()
            if len(data) != nbytes:
                continue
            meta = {}
            dp = d / "done.txt"
            if dp.exists():
                for line in dp.read_text().splitlines():
                    k, _, v = line.partition("=")
                    meta[k] = v
            fb.write(data)
            fi.write(f"{lid},{off},{len(data)},{meta.get('sweeps_done','')},"
                     f"{meta.get('burn_in_resolved','')},{meta.get('burn_sweeps','')},"
                     f"{meta.get('energy','')},{status.get(lid,'')}\n")
            off += len(data)
    stop_commit.set()
    commit_thread.join(timeout=10.0)
    volume.commit()
    complete = sum(1 for lid in ladder_ids if ladder_done(root / f"ladder_{lid:06d}"))
    return {"run": run_name, "ladders": len(ladder_ids), "complete": complete, "status": status}


def launch_dataset():
    prefix = os.environ.get("RUN_PREFIX", "supercooled_v1")
    ladders = int(os.environ.get("LADDERS_PER_BETA", "10000"))
    group = int(os.environ.get("GROUP_SIZE", "8"))
    workers = int(os.environ.get("MAX_CONCURRENT", "16"))
    betas = [2.0 * i / 9.0 for i in range(10)]
    base_cfg = {
        "sweeps": int(os.environ.get("SWEEPS", "100000")),
        "sample_every": 10,
        "checkpoint_every": CHECKPOINT_EVERY,
        "history_every": HISTORY_EVERY,
        "local": int(os.environ.get("LOCAL_PCT", "0")),
        "adaptive_burn_in": os.environ.get("ADAPTIVE", "1") == "1",
        "burn_min": int(os.environ.get("BURN_MIN", "5000")),
        "burn_max": int(os.environ.get("BURN_MAX", "300000")),
        "post_burn": int(os.environ.get("POST_BURN", "0")),
        "burn_acf_every": 10,
        "burn_max_lag": int(os.environ.get("BURN_MAX_LAG", "1000")),
        "burn_check_every": int(os.environ.get("BURN_CHECK_EVERY", "5000")),
        "burn_zero_window": 5,
        "max_snapshots": int(os.environ.get("MAX_SNAPSHOTS", "0")),
        "seed": 1,
    }
    jobs_by_beta = []
    for j, beta in enumerate(betas):
        run_name = f"{prefix}/b{j:02d}_b{beta:.3f}"
        cfg = dict(base_cfg, beta=beta)
        q = []
        for start in range(0, ladders, group):
            ids = list(range(start, min(start + group, ladders)))
            q.append((run_name, ids, json.dumps(cfg)))
        jobs_by_beta.append(q)
    # Round-robin across betas: slow high-beta chains must not be queued behind
    # 9000 already-finished low-beta chains.
    jobs = []
    for i in range(max(len(q) for q in jobs_by_beta)):
        for q in jobs_by_beta:
            if i < len(q):
                jobs.append(q[i])
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
    if os.environ.get("PACK", "1") == "1":
        for j, beta in enumerate(betas):
            run_name = f"{prefix}/b{j:02d}_b{beta:.3f}"
            try:
                print("pack", pack_beta.remote(run_name), flush=True)
            except Exception as exc:
                print("pack FAILED", run_name, repr(exc), flush=True)
    return failures


@app.function(image=image, cpu=2, memory=4096, volumes={"/runs": volume}, timeout=3600)
def pack_beta(run_name: str):
    import csv
    volume.reload()
    root = Path("/runs") / run_name
    out_path = root / "dataset.bin"
    idx_path = root / "index.csv"
    total = 0
    resolved = 0
    seen = set()
    with out_path.open("wb") as fb, idx_path.open("w", newline="") as fi:
        w = csv.writer(fi)
        w.writerow(["ladder_id", "offset", "bytes", "sweeps_done",
                    "burn_in_resolved", "burn_sweeps", "energy", "status"])
        for p in sorted((root / "packed").glob("group_*.bin")):
            ip = p.with_suffix(".csv")
            if not ip.exists():
                continue
            data = p.read_bytes()
            for r in csv.DictReader(ip.open()):
                lid = int(r["ladder_id"])
                if lid in seen:
                    continue
                seen.add(lid)
                o = int(r["offset"])
                n = int(r["bytes"])
                fb.write(data[o:o + n])
                w.writerow([lid, total, n, r["sweeps_done"],
                            r["burn_in_resolved"], r["burn_sweeps"],
                            r["energy"], r["status"]])
                total += n
                resolved += 1 if r["burn_in_resolved"] == "1" else 0
    volume.commit()
    return {"run": run_name, "configs": len(seen), "resolved": resolved, "bytes": total}


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
        "local": int(os.environ.get("LOCAL_PCT", "0")),
        "adaptive_burn_in": os.environ.get("ADAPTIVE", "1") == "1",
        "burn_min": int(os.environ.get("BURN_MIN", "5000")),
        "burn_max": int(os.environ.get("BURN_MAX", "300000")),
        "post_burn": int(os.environ.get("POST_BURN", "0")),
        "burn_acf_every": 10,
        "burn_max_lag": int(os.environ.get("BURN_MAX_LAG", "1000")),
        "burn_check_every": int(os.environ.get("BURN_CHECK_EVERY", "5000")),
        "burn_zero_window": 5,
        "max_snapshots": int(os.environ.get("MAX_SNAPSHOTS", "0")),
        "seed": 1,
    }
    result = run_ladders.remote(run_name, list(range(LADDERS)), json.dumps(cfg))
    print(json.dumps(result, indent=2))
