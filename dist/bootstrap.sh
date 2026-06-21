#!/usr/bin/env bash
#
# bootstrap.sh -- turn a fresh OpenStack VM into an NH lattice-glass worker host.
#
# Run this ONCE per VM (or as a golden-image build step). It compiles the
# production_sampler binary on THIS host so the `-march=native` baked into the
# Makefile's CXXFLAGS targets the local CPU, then sanity-checks that the Python
# campaign package imports. After it succeeds, start workers with ./dist/launch.sh.
#
# Idempotent: `make` is incremental and there are no destructive steps, so this
# is safe to re-run (e.g. after a `git pull`).
#
# Optional env:
#   CXX   C++ compiler to use (default: clang++). The build picks up -march=native.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

CXX="${CXX:-clang++}"
BINARY="$REPO_ROOT/experiments/production_sampler"

echo "[bootstrap] repo root : $REPO_ROOT"
echo "[bootstrap] arch      : $(uname -m)"
echo "[bootstrap] python3   : $(python3 --version 2>&1)"
echo "[bootstrap] compiler  : $CXX (per-host build picks up -march=native)"

# Build the sampler for this host. make is incremental, so a no-op rebuild is cheap.
echo "[bootstrap] building experiments/production_sampler ..."
make experiments/production_sampler CXX="$CXX"

if [ ! -x "$BINARY" ]; then
    echo "[bootstrap] ERROR: $BINARY is missing or not executable after the build." >&2
    echo "[bootstrap]        Check the compiler ($CXX) and the make output above." >&2
    exit 1
fi
echo "[bootstrap] binary    : $BINARY"

# The worker drives the sampler through the dist package; make sure it imports
# from the repo root (where this package lives).
echo "[bootstrap] verifying 'import dist' from repo root ..."
python3 -c 'import dist'
echo "[bootstrap] dist package import OK"

echo "[bootstrap] done. This host is ready."
echo "[bootstrap] start workers with: COORDINATOR_URL=http://<coord>:8080 STORE_ROOT=/mnt/shared/store ./dist/launch.sh"
