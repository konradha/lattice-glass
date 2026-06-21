"""Lease-based bag-of-tasks coordinator for the NH lattice-glass PT sampler.

One PT ladder = one work unit. The coordinator keeps a registry of work units
in sqlite and serves a tiny HTTP API to a fleet of preemptible workers:

    POST /claim      -> lease one eligible ladder to a worker
    POST /heartbeat  -> renew a lease and report progress
    POST /complete   -> release a ladder (done, or pending for continuation)
    GET  /status     -> campaign-wide totals + per-ladder snapshot

Robustness to worker death comes from lease expiry: a leased ladder whose
`lease_expiry` has passed is reclaimable by the next claimant. Workers are
cattle; the registry is the single source of truth.

The :class:`Coordinator` wraps all logic against sqlite and is fully usable
without HTTP, which keeps it deterministically testable via an injected clock.
"""

import argparse
import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Columns returned to a worker when it is granted a ladder.
_CLAIM_FIELDS = (
    "ladder_id",
    "L",
    "n_temps",
    "beta_min",
    "beta_max",
    "seed",
    "warmup",
    "sample_every",
    "exchange_every",
    "target_configs",
    "sweeps_done",
    "configs_done",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ladders(
    ladder_id      TEXT PRIMARY KEY,
    L              INT,
    n_temps        INT,
    beta_min       REAL,
    beta_max       REAL,
    seed           INT,
    warmup         INT,
    sample_every   INT,
    exchange_every INT,
    target_configs INT,
    status         TEXT,
    owner          TEXT,
    lease_expiry   REAL,
    sweeps_done    INT,
    configs_done   INT,
    arch           TEXT
)
"""


class Coordinator:
    """Sqlite-backed registry of PT-ladder work units with lease semantics.

    Parameters
    ----------
    db_path : str
        Path to the sqlite database file (created if absent).
    lease_default_secs : float
        Default lease duration when a claim/heartbeat omits ``lease_secs``.
    clock : callable
        Zero-arg callable returning the current time as a float. Injected so
        expiry behaviour is deterministically testable.
    """

    def __init__(self, db_path, lease_default_secs=3600, clock=time.time):
        self._clock = clock
        self._lease_default_secs = lease_default_secs
        # check_same_thread=False so the ThreadingHTTPServer worker threads may
        # share one connection; isolation_level=None puts us in autocommit mode
        # so we control transactions explicitly (BEGIN IMMEDIATE for atomic
        # read-modify-write claims). A process-wide lock serializes access to
        # the shared connection.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.isolation_level = None
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.execute(_SCHEMA)

    # -- campaign registration -------------------------------------------------

    def upsert_campaign(self, spec):
        """Insert any ladders in ``spec`` not already present; return the count
        inserted. Existing ladders are left fully untouched (idempotent): their
        status and progress are never reset by a re-upsert.
        """
        ladders = spec.get("ladders", [])
        with self._lock:
            before = self._conn.total_changes
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for lad in ladders:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO ladders(
                            ladder_id, L, n_temps, beta_min, beta_max, seed,
                            warmup, sample_every, exchange_every, target_configs,
                            status, owner, lease_expiry, sweeps_done,
                            configs_done, arch)
                        VALUES(?,?,?,?,?,?,?,?,?,?, 'pending', NULL, NULL, 0, 0, NULL)
                        """,
                        (
                            lad["ladder_id"],
                            lad["L"],
                            lad["n_temps"],
                            lad["beta_min"],
                            lad["beta_max"],
                            lad["seed"],
                            lad["warmup"],
                            lad["sample_every"],
                            lad["exchange_every"],
                            lad["target_configs"],
                        ),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            return self._conn.total_changes - before

    # -- worker lifecycle ------------------------------------------------------

    def claim(self, worker_id, arch, lease_secs=None):
        """Atomically lease one eligible ladder to ``worker_id``.

        Eligible = pending, or a leased ladder whose lease has expired (stale
        reclaim), restricted to ladders whose arch is unset or matches ``arch``.
        Pending ladders are preferred over reclaimed ones, then lowest
        ``sweeps_done``, then ladder_id (deterministic). Returns the granted
        ladder's fields as a dict, or ``None`` if nothing is eligible.
        """
        now = self._clock()
        lease = lease_secs or self._lease_default_secs
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    """
                    SELECT ladder_id FROM ladders
                    WHERE (status = 'pending'
                           OR (status = 'leased' AND lease_expiry < ?))
                      AND (arch IS NULL OR arch = ?)
                    ORDER BY (status = 'pending') DESC, sweeps_done ASC, ladder_id ASC
                    LIMIT 1
                    """,
                    (now, arch),
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None
                ladder_id = row["ladder_id"]
                self._conn.execute(
                    """
                    UPDATE ladders
                       SET status = 'leased',
                           owner = ?,
                           lease_expiry = ?,
                           arch = COALESCE(arch, ?)
                     WHERE ladder_id = ?
                    """,
                    (worker_id, now + lease, arch, ladder_id),
                )
                granted = self._conn.execute(
                    "SELECT %s FROM ladders WHERE ladder_id = ?" % ", ".join(_CLAIM_FIELDS),
                    (ladder_id,),
                ).fetchone()
                self._conn.execute("COMMIT")
                return dict(granted)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def heartbeat(self, worker_id, ladder_id, sweeps_done, configs_done, lease_secs=None):
        """Renew a lease and record progress if ``worker_id`` still owns the
        leased ladder. If the lease has been reclaimed (owner changed / not
        leased), report ``revoked`` and leave progress untouched.
        """
        now = self._clock()
        lease = lease_secs or self._lease_default_secs
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT owner, status FROM ladders WHERE ladder_id = ?",
                    (ladder_id,),
                ).fetchone()
                if row is not None and row["owner"] == worker_id and row["status"] == "leased":
                    self._conn.execute(
                        """
                        UPDATE ladders
                           SET lease_expiry = ?, sweeps_done = ?, configs_done = ?
                         WHERE ladder_id = ?
                        """,
                        (now + lease, sweeps_done, configs_done, ladder_id),
                    )
                    self._conn.execute("COMMIT")
                    return {"ok": True, "revoked": False}
                self._conn.execute("COMMIT")
                return {"ok": True, "revoked": True}
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def complete(self, worker_id, ladder_id, sweeps_done, configs_done, done):
        """Release a ladder. ``done`` -> status 'done'; otherwise 'pending'
        (released for continuation by another worker). Progress is monotonic:
        a stale worker reporting lower counts cannot regress the record.
        Accepted whenever ``ladder_id`` exists, even past lease expiry.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT sweeps_done, configs_done FROM ladders WHERE ladder_id = ?",
                    (ladder_id,),
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return {"ok": False}
                new_sweeps = max(row["sweeps_done"] or 0, sweeps_done)
                new_configs = max(row["configs_done"] or 0, configs_done)
                status = "done" if done else "pending"
                self._conn.execute(
                    """
                    UPDATE ladders
                       SET sweeps_done = ?, configs_done = ?, status = ?,
                           owner = NULL, lease_expiry = NULL
                     WHERE ladder_id = ?
                    """,
                    (new_sweeps, new_configs, status, ladder_id),
                )
                self._conn.execute("COMMIT")
                return {"ok": True}
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def status(self):
        """Return campaign-wide totals plus a per-ladder snapshot."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT ladder_id, status, owner, sweeps_done, configs_done,
                       target_configs, arch
                  FROM ladders
                 ORDER BY ladder_id ASC
                """
            ).fetchall()
        ladders = [dict(r) for r in rows]
        totals = {"pending": 0, "leased": 0, "done": 0, "configs_done_sum": 0}
        for lad in ladders:
            if lad["status"] in totals:
                totals[lad["status"]] += 1
            totals["configs_done_sum"] += lad["configs_done"] or 0
        return {"totals": totals, "ladders": ladders}

    def close(self):
        with self._lock:
            self._conn.close()


# -- HTTP layer ----------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Thin JSON HTTP adapter; the Coordinator lives on ``self.server``."""

    server_version = "PTCoordinator/1.0"

    @property
    def _coord(self):
        return self.server.coordinator

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            body = self._read_json()
        except Exception:
            self._send_json({"error": "bad request body"}, 400)
            return
        try:
            if self.path == "/claim":
                ladder = self._coord.claim(
                    body["worker_id"], body.get("arch"), body.get("lease_secs")
                )
                self._send_json({"ladder": ladder})
            elif self.path == "/heartbeat":
                self._send_json(
                    self._coord.heartbeat(
                        body["worker_id"],
                        body["ladder_id"],
                        body["sweeps_done"],
                        body["configs_done"],
                        body.get("lease_secs"),
                    )
                )
            elif self.path == "/complete":
                self._send_json(
                    self._coord.complete(
                        body["worker_id"],
                        body["ladder_id"],
                        body["sweeps_done"],
                        body["configs_done"],
                        body["done"],
                    )
                )
            else:
                self._send_json({"error": "unknown route"}, 400)
        except KeyError as exc:
            self._send_json({"error": "missing field %s" % exc}, 400)

    def do_GET(self):
        if self.path == "/status":
            self._send_json(self._coord.status())
        else:
            self._send_json({"error": "unknown route"}, 400)

    def log_message(self, *args):  # silence default stderr access logging
        pass


def serve(coordinator, host, port):
    """Serve ``coordinator`` over HTTP until interrupted."""
    server = ThreadingHTTPServer((host, port), _Handler)
    server.coordinator = coordinator
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="PT-ladder lease coordinator")
    parser.add_argument("--db", required=True, help="sqlite database path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--campaign", default=None, help="JSON campaign to upsert at startup")
    parser.add_argument("--lease-default-secs", type=float, default=3600)
    args = parser.parse_args()

    coordinator = Coordinator(args.db, lease_default_secs=args.lease_default_secs)
    if args.campaign:
        with open(args.campaign) as fh:
            spec = json.load(fh)
        inserted = coordinator.upsert_campaign(spec)
        print("upserted %d ladders" % inserted)
    print("coordinator listening on %s:%d" % (args.host, args.port))
    serve(coordinator, args.host, args.port)


if __name__ == "__main__":
    main()
