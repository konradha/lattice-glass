#!/usr/bin/env bash
#
# launch.sh -- start and supervise N production_sampler workers on this host.
#
# Each worker leases a PT ladder from the coordinator, drives the sampler for a
# bounded lease window, checkpoints, and pushes the checkpoint + configs to the
# store. Workers are stateless cattle: this script wraps each one in a respawn
# loop so a crashed or preempted worker restarts automatically, and a single
# SIGINT/SIGTERM tears the whole fleet down cleanly (forwarding the signal so
# in-flight workers checkpoint before exit). Run ./dist/bootstrap.sh first to
# build the binary.
#
# Required env:
#   COORDINATOR_URL   coordinator base URL, e.g. http://10.0.0.5:8080
#   STORE_ROOT        checkpoint/config store root (shared mount or synced dir)
#
# Optional env (defaults):
#   N_WORKERS        (= CPU count)     workers to run on this host
#   LEASE_SECS       (= 3600)          lease window requested per ladder
#   WORKDIR          (= /tmp/nhlg_work) scratch + per-worker logs
#   POLL_SECS        (= 30)            idle poll interval when no ladder is free
#   RESPAWN_BACKOFF  (= 5)             seconds to wait before respawning a worker
set -euo pipefail

# Job control so each background respawn loop becomes its own process group;
# that lets the shutdown trap signal the loop AND its sampler child as a unit.
set -m

: "${COORDINATOR_URL:?set COORDINATOR_URL to the coordinator base URL, e.g. http://10.0.0.5:8080}"
: "${STORE_ROOT:?set STORE_ROOT to the checkpoint/config store root}"

N_WORKERS=${N_WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)}
LEASE_SECS=${LEASE_SECS:-3600}
WORKDIR=${WORKDIR:-/tmp/nhlg_work}
POLL_SECS=${POLL_SECS:-30}
RESPAWN_BACKOFF=${RESPAWN_BACKOFF:-5}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

BINARY="$REPO_ROOT/experiments/production_sampler"
if [ ! -x "$BINARY" ]; then
    echo "ERROR: sampler binary not found at $BINARY" >&2
    echo "       Run ./dist/bootstrap.sh on this host first." >&2
    exit 1
fi

mkdir -p "$WORKDIR"

# Respawn loop for a single worker: keep relaunching until this script is torn
# down. The short backoff stops a worker that dies instantly (e.g. coordinator
# briefly unreachable) from busy-looping.
run_worker() {
    local i="$1"
    local wlog="$WORKDIR/w$i.log"
    mkdir -p "$WORKDIR/w$i"
    while true; do
        echo "[launch $(date -u +%H:%M:%S)] (re)starting worker w$i" >>"$wlog"
        python3 -m dist.worker \
            --coordinator "$COORDINATOR_URL" \
            --store-root "$STORE_ROOT" \
            --binary "$BINARY" \
            --workdir "$WORKDIR/w$i" \
            --lease-secs "$LEASE_SECS" \
            --poll-secs "$POLL_SECS" \
            >>"$wlog" 2>&1 || true
        echo "[launch $(date -u +%H:%M:%S)] worker w$i exited; respawning in ${RESPAWN_BACKOFF}s" >>"$wlog"
        sleep "$RESPAWN_BACKOFF"
    done
}

PIDS=()

shutdown() {
    trap - INT TERM
    echo ""
    if [ "${#PIDS[@]}" -gt 0 ]; then
        echo "[launch] stopping ${#PIDS[@]} worker(s) ..."
        for pid in "${PIDS[@]}"; do
            # Signal the whole process group (respawn loop + sampler child) so a
            # running worker gets SIGTERM and can checkpoint; fall back to the
            # bare pid if the group send is rejected.
            kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        done
    fi
    wait 2>/dev/null || true
    echo "[launch] all workers stopped."
    exit 0
}
trap shutdown INT TERM

echo "[launch] host=$(hostname) arch=$(uname -m)"
echo "[launch] starting $N_WORKERS worker(s)"
echo "[launch]   coordinator : $COORDINATOR_URL"
echo "[launch]   store root  : $STORE_ROOT"
echo "[launch]   workdir     : $WORKDIR"
echo "[launch]   lease/poll  : ${LEASE_SECS}s / ${POLL_SECS}s"
echo "[launch]   logs        : $WORKDIR/w<i>.log"

for ((i = 0; i < N_WORKERS; i++)); do
    run_worker "$i" &
    PIDS+=("$!")
done

echo "[launch] launched worker loops with PIDs: ${PIDS[*]}"
echo "[launch] send SIGINT (Ctrl-C) or SIGTERM to stop all workers."

# Supervise until a signal arrives; the respawn loops never exit on their own.
wait
