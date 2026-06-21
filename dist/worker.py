"""Worker agent for the distributed NH lattice-glass PT campaign.

A worker is preemptible cattle. It loops:

    1. lease one PT ladder from the coordinator (HTTP ``/claim``);
    2. pull that ladder's checkpoint from the store (resume), if any;
    3. run ``experiments/production_sampler`` for a bounded wall-clock window
       (the lease minus a safety margin), checkpointing as it goes;
    4. push the updated checkpoint + per-temperature config ``.npy`` files back
       to the store, regardless of how the child exited;
    5. report progress via ``/heartbeat`` and finalize via ``/complete``.

Preemption is handled two ways. A ``SIGTERM``/``SIGINT`` sets a stop flag, and
the coordinator can revoke the lease (another worker reclaimed it). Either way
the child sampler is terminated cleanly and its partial results are flushed to
the store so the next worker resumes exactly where this one stopped.

Standard library only; talks to the coordinator with urllib (no requests).
"""

import argparse
import glob
import json
import os
import platform
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request

from dist.store import FilesystemStore

# Network calls are bounded so an unreachable coordinator fails fast, never hangs.
_HTTP_TIMEOUT = 30
# Grace given to a terminated child before we SIGKILL it.
_TERMINATE_GRACE = 10.0
# Sampler per-checkpoint progress line: "... sweeps=<N> collected[cold]=<C>/<T> ...".
_PROGRESS_RE = re.compile(r"sweeps=(\d+) collected\[cold\]=(\d+)/(\d+)")


def parse_progress(stdout_text):
    """Return ``(sweeps, configs_done, done)`` parsed from sampler stdout.

    Scans for checkpoint lines of the form
    ``... sweeps=<N> collected[cold]=<C>/<TARGET> ...`` and reports the values
    from the LAST one seen; ``done`` is True iff that last line contains the
    `` DONE`` marker. Returns ``(0, 0, False)`` when no checkpoint line is
    present (e.g. empty or garbage input).
    """
    sweeps, configs, done = 0, 0, False
    for line in stdout_text.splitlines():
        m = _PROGRESS_RE.search(line)
        if m:
            sweeps = int(m.group(1))
            configs = int(m.group(2))
            done = " DONE" in line
    return sweeps, configs, done


def http_post_json(url, obj, timeout=_HTTP_TIMEOUT):
    """POST ``obj`` as JSON to ``url``; return parsed JSON. Raise on non-200."""
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError("POST %s -> HTTP %s" % (url, resp.status))
        return json.loads(resp.read().decode("utf-8"))


def http_get_json(url, timeout=_HTTP_TIMEOUT):
    """GET ``url``; return parsed JSON. Raise on non-200."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError("GET %s -> HTTP %s" % (url, resp.status))
        return json.loads(resp.read().decode("utf-8"))


class Worker:
    """Leases PT ladders and drives the sampler for each, syncing to a store.

    Parameters
    ----------
    coordinator_url : str
        Base URL of the coordinator HTTP API, e.g. ``http://host:8080``.
    store : Store
        Durable home for checkpoints + config artifacts (``dist.store``).
    binary : str
        Path to the ``production_sampler`` executable.
    workdir : str
        Scratch directory; each ladder gets ``workdir/<ladder_id>/``.
    lease_secs, poll_secs, margin_secs, checkpoint_secs : numbers
        Lease length requested per claim; idle poll interval when no ladder is
        available; wall-clock margin reserved before the lease expires; and an
        optional override for the sampler's checkpoint cadence.
    """

    def __init__(self, coordinator_url, store, binary, workdir,
                 lease_secs=3600, poll_secs=30, margin_secs=60,
                 checkpoint_secs=None):
        self.coordinator_url = coordinator_url.rstrip("/")
        self.store = store
        self.binary = binary
        self.workdir = workdir
        self.lease_secs = lease_secs
        self.poll_secs = poll_secs
        self.margin_secs = margin_secs
        self.checkpoint_secs = checkpoint_secs
        self.worker_id = "%s:%d" % (socket.gethostname(), os.getpid())
        self.arch = platform.machine()
        # Set by main()'s signal handlers (and observed by the heartbeat thread)
        # to request a clean, prompt shutdown of the current run.
        self.stop = threading.Event()

    # -- coordinator RPCs ------------------------------------------------------

    def claim(self):
        """Lease one ladder; return its fields dict, or ``None`` if none free."""
        resp = http_post_json(self.coordinator_url + "/claim", {
            "worker_id": self.worker_id,
            "arch": self.arch,
            "lease_secs": self.lease_secs,
        })
        return resp.get("ladder")

    # -- sampler invocation ----------------------------------------------------

    def _build_argv(self, ladder, ckpt, out_prefix):
        """Assemble the ``production_sampler`` command line for ``ladder``.

        ``--wall-secs`` is the lease minus the safety margin, floored at a small
        positive value: a non-positive wall would make the sampler WALL-STOP on
        its very first iteration (it tests ``elapsed > wall`` after each round).
        """
        wall = max(1.0, self.lease_secs - self.margin_secs)
        if self.checkpoint_secs is not None:
            ckpt_secs = self.checkpoint_secs
        else:
            ckpt_secs = max(30.0, (self.lease_secs - self.margin_secs) / 3.0)
        return [
            self.binary,
            "--L", str(ladder["L"]),
            "--n-temps", str(ladder["n_temps"]),
            "--beta-min", str(ladder["beta_min"]),
            "--beta-max", str(ladder["beta_max"]),
            "--seed", str(ladder["seed"]),
            "--warmup", str(ladder["warmup"]),
            "--sample-every", str(ladder["sample_every"]),
            "--exchange-every", str(ladder["exchange_every"]),
            "--configs-per-temp", str(ladder["target_configs"]),
            "--ckpt", ckpt,
            "--out-prefix", out_prefix,
            "--wall-secs", str(wall),
            "--checkpoint-secs", str(ckpt_secs),
        ]

    def run_ladder(self, ladder):
        """Resume + run one ladder to its bounded window, then sync results."""
        ladder_id = ladder["ladder_id"]
        if not os.path.exists(self.binary):
            raise FileNotFoundError("sampler binary not found: %s" % self.binary)

        d = os.path.join(self.workdir, ladder_id)
        os.makedirs(d, exist_ok=True)
        ckpt = os.path.join(d, "ckpt.bin")
        # Pull the stored checkpoint to resume; False simply means start fresh.
        self.store.get_checkpoint(ladder_id, ckpt)

        out_prefix = os.path.join(d, "cfg")
        argv = self._build_argv(ladder, ckpt, out_prefix)
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        # Shared progress, seeded with the registry's resume baseline so the
        # first heartbeat never regresses progress (the coordinator sets, not
        # max()es, heartbeat counts).
        lock = threading.Lock()
        latest = [int(ladder.get("sweeps_done") or 0),
                  int(ladder.get("configs_done") or 0), False]
        done_event = threading.Event()      # set by main thread once child exits
        revoked_event = threading.Event()   # set by heartbeat on lease loss

        def _terminate_child():
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=_TERMINATE_GRACE)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass

        def _heartbeat_loop():
            interval = max(5.0, self.lease_secs / 3.0)
            last_beat = time.monotonic()
            # Poll at a fine granularity so stop/exit are noticed promptly, but
            # only emit a heartbeat once every `interval` seconds.
            while not done_event.wait(timeout=0.5):
                if self.stop.is_set():
                    _terminate_child()
                    return
                if time.monotonic() - last_beat < interval:
                    continue
                last_beat = time.monotonic()
                with lock:
                    sw, cf = latest[0], latest[1]
                try:
                    resp = http_post_json(self.coordinator_url + "/heartbeat", {
                        "worker_id": self.worker_id,
                        "ladder_id": ladder_id,
                        "sweeps_done": sw,
                        "configs_done": cf,
                        "lease_secs": self.lease_secs,
                    })
                except Exception:
                    # Transient coordinator hiccup: keep the child running and
                    # retry on the next beat rather than discarding live work.
                    continue
                if resp.get("revoked"):
                    revoked_event.set()
                    _terminate_child()
                    return

        hb = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb.start()

        captured = []
        try:
            for raw in proc.stdout:
                captured.append(raw)
                m = _PROGRESS_RE.search(raw)
                if m:
                    with lock:
                        latest[0] = int(m.group(1))
                        latest[1] = int(m.group(2))
                        latest[2] = " DONE" in raw
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            rc = proc.wait()
            done_event.set()
            hb.join()

        # Flush whatever the child produced back to the store, win or lose, so a
        # successor resumes from the furthest checkpoint reached.
        if os.path.isfile(ckpt):
            self.store.put_checkpoint(ladder_id, ckpt)
        for path in sorted(glob.glob(os.path.join(d, "cfg_seed*_T*.npy"))):
            self.store.put_config(ladder_id, os.path.basename(path), path)

        sweeps, configs, parsed_done = parse_progress("".join(captured))

        # On lease revocation another worker already owns this ladder; posting
        # /complete here would clear their ownership in the registry. Stop quietly.
        if revoked_event.is_set():
            return

        # DONE only when the child exited cleanly AND reported all temps full.
        # Anything else (wall-stop, SIGTERM, error) is released for retry.
        final_done = rc == 0 and parsed_done
        http_post_json(self.coordinator_url + "/complete", {
            "worker_id": self.worker_id,
            "ladder_id": ladder_id,
            "sweeps_done": sweeps,
            "configs_done": configs,
            "done": final_done,
        })

    # -- main loop -------------------------------------------------------------

    def run(self, once=False):
        """Claim-and-run loop until stopped (or after one ladder if ``once``)."""
        while True:
            if self.stop.is_set():
                return
            ladder = self.claim()
            if ladder is None:
                if once or self.stop.is_set():
                    return
                # Interruptible idle wait: a SIGTERM during the poll is prompt.
                self.stop.wait(timeout=self.poll_secs)
                continue
            self.run_ladder(ladder)
            if once or self.stop.is_set():
                return


def main(argv=None):
    parser = argparse.ArgumentParser(description="PT-ladder campaign worker")
    parser.add_argument("--coordinator", required=True, help="coordinator base URL")
    parser.add_argument("--store-root", required=True, help="FilesystemStore root")
    parser.add_argument("--binary", required=True, help="path to production_sampler")
    parser.add_argument("--workdir", required=True, help="scratch dir for ladders")
    parser.add_argument("--lease-secs", type=float, default=3600)
    parser.add_argument("--poll-secs", type=float, default=30)
    parser.add_argument("--margin-secs", type=float, default=60)
    parser.add_argument("--checkpoint-secs", type=float, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    store = FilesystemStore(args.store_root)
    worker = Worker(
        args.coordinator, store, args.binary, args.workdir,
        lease_secs=args.lease_secs, poll_secs=args.poll_secs,
        margin_secs=args.margin_secs, checkpoint_secs=args.checkpoint_secs,
    )

    def _handle(signum, frame):
        worker.stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    worker.run(once=args.once)


if __name__ == "__main__":
    main()
