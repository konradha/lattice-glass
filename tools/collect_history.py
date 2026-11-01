#!/usr/bin/env python3
"""Collect per-chain history (and optionally final configs) into an NPZ dataset."""
import argparse
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<QI4xd")   # magic, version, beta (+ padding)
RECORD = struct.Struct("<Qi iq")   # sweep, energy, accepted_since, accepted_total


def read_history(path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        return None
    magic, version, beta = HEADER.unpack_from(data, 0)
    if magic != 0x4C474831 or version != 1:
        return None
    n = (len(data) - HEADER.size) // RECORD.size
    out = np.zeros((n, 4), dtype=np.int64)
    for i in range(n):
        s, e, a, t = RECORD.unpack_from(data, HEADER.size + i * RECORD.size)
        out[i] = (s, e, a, t)
    return beta, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--configs", action="store_true")
    args = ap.parse_args()
    root = Path(args.run_root)
    ladders = sorted(root.glob("ladder_*"))
    histories = []
    betas = []
    chain_ids = []
    for ladder in ladders:
        h = read_history(ladder / "history.bin")
        if h is None:
            continue
        beta, rec = h
        histories.append(rec)
        betas.append(beta)
        chain_ids.append(int(ladder.name.split("_")[-1]))
    if not histories:
        raise SystemExit("no history.bin found")
    max_len = max(len(h) for h in histories)
    fields = ("sweep", "energy", "accepted_since", "accepted_total")
    arrays = {}
    for col, name in enumerate(fields):
        arr = np.zeros((len(histories), max_len), dtype=np.int64)
        for i, h in enumerate(histories):
            arr[i, :len(h)] = h[:, col]
        arrays[name] = arr
    arrays["lengths"] = np.array([len(h) for h in histories], dtype=np.int32)
    arrays["beta"] = np.array(betas, dtype=np.float64)
    arrays["chain_ids"] = np.array(chain_ids, dtype=np.int64)
    out = Path(args.out) if args.out else root / "history.npz"
    if args.configs:
        configs = []
        for ladder in ladders:
            p = ladder / "configs" / "config.bin"
            if p.exists():
                configs.append(np.frombuffer(p.read_bytes(), dtype=np.uint8).copy())
        if configs:
            arrays["configs"] = np.stack(configs)
    np.savez(out, **arrays)
    print(f"chains={len(histories)} max_records={max_len} out={out}")


if __name__ == "__main__":
    main()
