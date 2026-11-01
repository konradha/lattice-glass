#!/usr/bin/env python3
"""Launch and resume many independent PT ladders on a local machine."""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path


def ladder_args(binary, run_dir, cfg, ladder_id, resume):
    args = [
        binary,
        "--run-dir", str(run_dir),
        "--ntemps", str(cfg["ntemps"]),
        "--beta-min", str(cfg["beta_min"]),
        "--beta-max", str(cfg["beta_max"]),
        "--sweeps", str(cfg["sweeps"]),
        "--exchange-every", str(cfg["exchange_every"]),
        "--sample-every", str(cfg["sample_every"]),
        "--checkpoint-every", str(cfg["checkpoint_every"]),
        "--local", str(cfg["local"]),
        "--seed", str(cfg["seed"] + 104729 * ladder_id),
        "--snapshot-first", str(cfg.get("snapshot_first", 10000)),
        "--snapshot-growth", str(cfg.get("snapshot_growth", 1.15)),
        "--max-snapshots", str(cfg.get("max_snapshots", 200)),
    ]
    if resume:
        args.append("--resume")
    return args


def is_done(run_dir):
    done = Path(run_dir) / "done.txt"
    if not done.exists():
        return False
    text = done.read_text()
    return "complete=1" in text


def run_one(binary, root, cfg, ladder_id):
    run_dir = Path(root) / f"ladder_{ladder_id:06d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_done(run_dir):
        return ladder_id, "done", 0.0
    resume = (run_dir / "checkpoint.bin").exists()
    args = ladder_args(binary, run_dir, cfg, ladder_id, resume)
    proc = subprocess.run(args, capture_output=True, text=True)
    status = "resumed" if resume else "fresh"
    if proc.returncode != 0:
        return ladder_id, f"FAILED: {proc.stderr[-500:]}", 0.0
    return ladder_id, status, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--ladders", type=int, default=24)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--binary", default=str(Path(__file__).resolve().parents[1] / "cpu" / "lattice_pt"))
    ap.add_argument("--config", help="JSON file with ntemps, beta_min, beta_max, sweeps, ...")
    ap.add_argument("--ntemps", type=int, default=4)
    ap.add_argument("--beta-min", type=float, default=0.5)
    ap.add_argument("--beta-max", type=float, default=2.0)
    ap.add_argument("--sweeps", type=int, default=1000)
    ap.add_argument("--exchange-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--local", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
    else:
        cfg = {
            "ntemps": args.ntemps, "beta_min": args.beta_min, "beta_max": args.beta_max,
            "sweeps": args.sweeps, "exchange_every": args.exchange_every,
            "sample_every": args.sample_every, "checkpoint_every": args.checkpoint_every,
            "local": args.local, "seed": args.seed,
        }
    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign.json").write_text(json.dumps(cfg, indent=2) + "\n")

    todo = [i for i in range(args.ladders) if not is_done(root / f"ladder_{i:06d}")]
    print(f"ladders={args.ladders} pending={len(todo)} workers={args.workers}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args.binary, root, cfg, i) for i in todo]
        for f in cf.as_completed(futs):
            lid, status, _ = f.result()
            print(f"ladder_{lid:06d} {status}", flush=True)
    done = sum(1 for i in range(args.ladders) if is_done(root / f"ladder_{i:06d}"))
    print(f"complete={done}/{args.ladders}")
    return 0 if done == args.ladders else 1


if __name__ == "__main__":
    raise SystemExit(main())
