# Verification and validation after optimization

Run every step from the repository root.

## 1. Build

```bash
make -C solid clean all
```

The Metal kernel is compiled at driver start. A build or MSL error stops the run.

## 2. Short correctness gate

```bash
make -C solid test
make -C solid test-long
```

This runs the CPU driver and the Metal driver with the same seed and compares:

- initial, equilibrated, and final state
- snapshots and snapshot sweeps
- total energy history
- local and nonlocal acceptance histories
- scalars and metadata

All fields must be bitwise equal.

## 3. High-beta gate

This catches rare EFMA, RNG, and cut errors.

```bash
mkdir -p /tmp/efma_cpu /tmp/efma_gpu
(cd /tmp/efma_cpu && /Users/fritz/code/glasses/lattice-glass/solid/driver \
  21 2.5 0.75 0.30 0.45 160000 8 CPU nonlocal 1000)
(cd /tmp/efma_gpu && /Users/fritz/code/glasses/lattice-glass/solid/driver \
  21 2.5 0.75 0.30 0.45 160000 8 GPU nonlocal 1000)
python3 - <<'PY'
import numpy as np
c = np.load('/tmp/efma_cpu/run.npz')
g = np.load('/tmp/efma_gpu/run.npz')
keys = ['initial','equilibrated','final','snapshots','snapshot_sweeps',
        'energy','accept_local','accept_nonlocal','scalars','meta']
bad = [k for k in keys if not np.array_equal(c[k], g[k])]
assert not bad, bad
print('high-beta bitwise PASS')
PY
```

## 4. EFMA invariants

The implementation must keep these properties:

- `T(dE) = floor(2^64 * exp(-beta*dE))`, split into two u32 words.
- Beta is parsed from decimal text and evaluated with at least 128-bit precision.
- `R_hi < T_hi` accepts, `R_hi > T_hi` rejects.
- The second word is drawn only when `R_hi == T_hi`.
- No float operation occurs in the accept step.
- Site and partner indices use exact Lemire rejection.
- CPU and Metal use the same word stream and proposal index.
- The cut commits accepted proposals and stops at any valid proposal within
  torus L1 distance 2 of an earlier accepted endpoint, including no-ops.

## 5. Campaign validation

Use `solid/aging_driver.py` for each backend and mode. Run betas sequentially;
each driver invocation runs 8 chains in parallel.

```bash
python3 solid/aging_driver.py \
  --betas '0.01,0.29428571428571426,0.5785714285714286,0.8628571428571429,1.1471428571428572,1.4314285714285715,1.7157142857142858,2.0' \
  --L 21 --rho 0.75 --rho1 0.30 --rho2 0.45 \
  --chains 8 --backend CPU --mode nonlocal \
  --equilibration '1000,1000,1000,1000,1000,1000,2767,10773' \
  --sweeps '1000,1000,1000,1000,1000,1000,2767,10773' \
  --output campaigns_linspace/cpu_nonlocal.npz \
  --workdir campaigns_linspace/cpu_nonlocal \
  --driver solid/driver --max-parallel 1
```

Repeat with the local schedule and with `--backend GPU`:

- local equilibration: `100,100,100,170,493,1591,6923,36324`
- local production: `200,200,200,339,985,3181,13846,72648`

A valid campaign has equal CPU and GPU fields for every beta and mode.

## 6. Performance check

```bash
/usr/bin/time -p solid/driver 21 2.0 0.75 0.30 0.45 1000 8 GPU nonlocal
/usr/bin/time -p solid/driver 21 2.0 0.75 0.30 0.45 1000 1 CPU nonlocal
```

Use the same command with `local` and `both`. Report per-chain and aggregate
proposal rates. Do not compare an 8-chain GPU aggregate with a 1-chain CPU run.

## 7. Plot regeneration

```bash
micromamba run -n rsmi python solid/plot_paper.py \
  --input campaigns_linspace/gpu_nonlocal.npz \
  --output-dir plots_linspace/gpu_nonlocal_paper --skip-fig2
```
