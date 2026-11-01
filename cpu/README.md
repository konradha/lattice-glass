# CPU fixed-beta campaign library

L=25 lattice glass with:
- packed 1-byte state (2-bit type, 6-bit neighbor mask)
- arithmetic neighbors, no neighbor table
- Metropolis local and nonlocal swaps (nonlocal Manhattan distance >= 3)
- exact incremental energy
- fixed beta per chain, independent chains
- per-chain history: sweep, energy, accepted, accepted_total
- log-spaced full checkpoints (restartable snapshots)
- atomic, checksummed checkpoints and resume
- local-move validation on saved configs

No parallel tempering. No reweighting.

Measured on Apple M3, single core: **46.4 ns/proposal**, **0.725 ms/sweep**.

## Build

```bash
cd cpu
make            # lattice_single
make test       # kernel, local moves, checkpoint, history, histogram
make bench      # proposal benchmark
```

## One chain

```bash
./cpu/lattice_single --run-dir runs/ladder_000000 \
  --beta 2.0 --sweeps 100000 \
  --sample-every 10 --checkpoint-every 1000 --history-every 100 \
  --local 50 --seed 1
```

Resume:

```bash
./cpu/lattice_single --run-dir runs/ladder_000000 --beta 2.0 \
  --sweeps 200000 --resume
```

Outputs:

```
run-dir/
  manifest.txt
  checkpoint.bin          # latest, atomic replace
  checkpoint.bin.prev     # previous good checkpoint
  history.bin             # per-chain time series
  snapshots/              # log-spaced full restarts
  configs/config.bin      # final 1-byte state
  hist.csv                # energy histogram
  done.txt
```

`history.bin` records `(sweep, energy, accepted_since, accepted_total)`.
On resume the file is truncated to the checkpoint sweep and continues.

## Many chains

```bash
python3 tools/run_campaign.py --run-root runs/beta2 \
  --ladders 128 --workers 64 --beta 2.0 \
  --sweeps 100000 --history-every 100 --checkpoint-every 1000
```

Completed chains are skipped. Incomplete chains resume from `checkpoint.bin`.

Collect the per-chain history:

```bash
python3 tools/collect_history.py --run-root runs/beta2 --configs
python3 tools/collect.py --run-root runs/beta2
```

## Modal

`tools/modal_cpu_app.py` builds the library in the image, mounts the
`lattice-runs` Volume, and runs one fixed-beta chain per core.

```bash
MODAL_CPU=64 MODAL_LADDERS=64 BETA=2.0 SWEEPS=100000 \
  RUN_NAME=beta2_100k micromamba run -n rsmi modal run tools/modal_cpu_app.py
```

Resume with the same command; completed chains are skipped.

## Thermalization gate

Do not release configs from sweep count alone. Require:

1. energy plateau after burn-in,
2. independent-chain energy agreement,
3. local-move energy drift on a validation subset,
4. stable energy histogram.

The history and snapshots are the evidence.
