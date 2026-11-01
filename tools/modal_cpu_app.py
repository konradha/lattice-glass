import json
import os
import subprocess
from pathlib import Path

import modal

import os

app = modal.App("lattice-cpu-campaign")
CPU = int(os.environ.get("MODAL_CPU", "32"))
LADDERS = int(os.environ.get("MODAL_LADDERS", "32"))
image = (
    modal.Image.debian_slim()
    .apt_install("build-essential")
    .add_local_dir("cpu", "/root/cpu", copy=True)
    .run_commands("cd /root/cpu && make clean && make CXX=g++ lattice_pt")
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
    binary = "/root/cpu/lattice_pt"
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
            "--ntemps", str(cfg["ntemps"]),
            "--beta-min", str(cfg["beta_min"]),
            "--beta-max", str(cfg["beta_max"]),
            "--sweeps", str(cfg["sweeps"]),
            "--exchange-every", str(cfg["exchange_every"]),
            "--sample-every", str(cfg["sample_every"]),
            "--checkpoint-every", str(cfg["checkpoint_every"]),
            "--local", str(cfg["local"]),
            "--seed", str(cfg["seed"] + 104729 * lid),
        ]
        if (run_dir / "checkpoint.bin").exists():
            args.append("--resume")
        log = (run_dir / "run.log").open("w")
        procs.append((lid, subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT), log))
        status[lid] = "running"
    for lid, proc, log in procs:
        rc = proc.wait()
        log.close()
        status[lid] = "ok" if rc == 0 else f"failed:{rc}"
    volume.commit()
    complete = sum(1 for lid in ladder_ids if ladder_done(root / f"ladder_{lid:06d}"))
    return {"run": run_name, "ladders": len(ladder_ids), "complete": complete, "status": status}


@app.function(image=image, timeout=600)
def inspect_image():
    import subprocess
    r = subprocess.run(["bash", "-lc", "find /root -maxdepth 3 -type f | head -50; echo ---; ls -l /root/cpu || true; echo ---; cd /root/cpu && make CXX=g++ lattice_pt 2>&1 | tail -30"],
                       capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)
    return r.stdout + r.stderr


@app.local_entrypoint()
def main():
    run_name = os.environ.get("RUN_NAME", "smoke32")
    cfg = {
        "ntemps": 4,
        "beta_min": 0.5,
        "beta_max": 2.0,
        "sweeps": int(os.environ.get("SWEEPS", "1000")),
        "exchange_every": 10,
        "sample_every": 10,
        "checkpoint_every": 1000,
        "local": 50,
        "seed": 1,
    }
    result = run_ladders.remote(run_name, list(range(LADDERS)), json.dumps(cfg))
    print(json.dumps(result, indent=2))
