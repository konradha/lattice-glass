# CPU PT campaign library

L=25 lattice glass with:
- packed 1-byte state (2-bit type, 6-bit neighbor mask)
- arithmetic neighbors, no neighbor table
- Metropolis local and nonlocal swaps (nonlocal Manhattan distance >= 3)
- exact incremental energy
- parallel tempering with adjacent exchange
- energy histograms and histogram reweighting
- atomic, checksummed checkpoints and resume
- log-spaced restartable snapshots
- one PT ladder per process; many ladders per machine

Measured on Apple M3, single core:
- 46.4 ns/proposal for the far-swap engine
- about 65 ns/proposal for a 4-temperature PT ladder including exchange and sampling

## Build

```bash
cd cpu
make            # lattice_pt
make test       # kernel, checkpoint, PT, reweighting tests
make bench      # proposal benchmark
```

## One ladder

```bash
./cpu/lattice_pt --run-dir runs/ladder_000000 \
  --ntemps 8 --beta-min 0.1 --beta-max 3.5 \
  --sweeps 10000000 --exchange-every 10 --sample-every 10 \
  --checkpoint-every 1000 --local 50 --seed 1
```

Resume:

```bash
./cpu/lattice_pt --run-dir runs/ladder_000000 ... --sweeps 20000000 --resume
```

Outputs:

```
run-dir/
  manifest.txt
  checkpoint.bin          # latest, atomic replace
  checkpoint.bin.prev     # previous good checkpoint
  snapshots/              # log-spaced full restarts
  configs/beta_<i>.bin    # raw 1-byte states, one per temperature
  hist.csv                # energy histograms per beta
  done.txt
```

Checkpoints contain the full state, xoshiro RNG state, energy, betas, histograms,
and cadence. `--resume` validates the FNV-1a checksum and the header, and falls
back to `.prev` if the latest checkpoint is corrupt.

## Many ladders

```bash
python3 tools/run_campaign.py --run-root runs/campaign \
  --ladders 128 --workers 64 \
  --ntemps 8 --beta-min 0.1 --beta-max 3.5 \
  --sweeps 1000000 --checkpoint-every 1000 --local 50
```

Completed ladders are skipped. Incomplete ladders resume from `checkpoint.bin`.

Collect and reweight:

```bash
python3 tools/collect.py --run-root runs/campaign
python3 tools/reweight.py --hist runs/campaign/combined_hist.csv \
  --targets 0,0.389,0.778,1.167,1.556,1.944,2.333,2.722,3.111,3.5
```

## Modal

`tools/modal_cpu_app.py` builds the library in the image, mounts the
`lattice-runs` Volume, and runs one ladder per core.

```bash
MODAL_CPU=32 MODAL_LADDERS=32 RUN_NAME=smoke32 SWEEPS=1000 \
  micromamba run -n rsmi modal run tools/modal_cpu_app.py
```

Resume is the same command with the same `RUN_NAME`; completed ladders are
skipped and incomplete ladders resume.

## Notes

- PT costs `ntemps` sweeps per ladder. A 10-temperature ladder costs about
  10x a single chain. Reweighting lets you simulate fewer temperatures and
  interpolate the target grid.
- The target grid `np.linspace(0, 3.5, 10)` is
  `0, 0.389, 0.778, 1.167, 1.556, 1.944, 2.333, 2.722, 3.111, 3.5`.
- For beta=0, the equilibrium distribution is uniform; sample it directly or
  reweight from beta_min.
- Checkpoint checksums detect corruption, not tampering. Files are 0600 and
  written by atomic rename.
