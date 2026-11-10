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

## 8. CUDA and Modal

CPU physics is authoritative at clean-solid `50fb57a`, including any-pair
nonlocal proposals. CUDA supports L4 through L21, with L8/L21 specializations.
Acceptance uses the CPU-built uint64 threshold table; no device `exp`.

```bash
make -C solid HAVE_CUDA=1 HAVE_METAL=0
make -C solid cuda_check_prepare
./solid/cuda_check_prepare --prepare solid/cuda_fixtures.bin --final-seed 15637821895788547078
modal run tools/modal_cuda_app.py --budget-usd 1.0 --action bench --gpu L40S --seconds 0.5
```

The pinned toolchain image is built once. Source/fixtures are mounted; native
builds run without a GPU and persist in `lattice-cuda-builds`. Python tool edits
do not invalidate native builds. Reservations are enforced before compute.
Queue plus execution waits are bounded; call-graph snapshots are not used as a
reliable indication that a container has started.

CUDA driver selection is `CUDA`, not the Metal `GPU` selector. Use
`--cuda-window 256` for the measured best configuration. `--start-sweep` resumes
the full uint64 proposal counter from an imported raw state.

Campaign example (explicit budget authorizes future compute; not run as part of
the development budget):

```bash
modal run tools/modal_cuda_app.py --budget-usd 1.0 --action campaign --gpu L40S \
  --run sample --betas 2.8,3.2 --mode nonlocal --chains 16384 --batches 284 \
  --sweeps 100000 --equil 10000 --window 256 --chunk-sweeps 4096 --max-chunks 64
```

Repeat identical settings to resume; `--calls` controls the number of authorized
bounded calls. Run local dynamics separately with `--mode local --chains 32`.
Check downloaded Volume data with
`python3 tools/check_cuda_campaign.py DOWNLOADED_RUN_DIRECTORY`.

### Exercised results, L40S, L21, beta 3.2

| Move | CPU core, M attempts/s | Isolated GPU chain, M/s | 142 chains, M/s per chain | Best card aggregate, G/s |
| --- | ---: | ---: | ---: | ---: |
| Local | 7.85 | 84.09 | 77.60 | 20.10 (568 chains) |
| Nonlocal | 7.60 | 121.83 | 110.28 | 27.14 (284 chains) |

All windows 1/32/64/128/256 passed CPU-byte, per-sweep energy/counter, and trace
comparisons before timing. Multi-chain grids included 142/284/568/1136 chains.
Memcheck, racecheck, and synccheck were clean for both moves and all windows.
Selected W256 passed 100,000,278 proposals per move with held-out chain 1,000,000.
Six real Modal checkpoint chunks, including resume, were downloaded and exactly
replayed against this branch's CPU implementation.

Rates count every attempt, including no-ops/rejections. Timings are CUDA events
after 1024 warm-up sweeps, with at least 0.5 measured kernel seconds per shape;
they exclude transfers/checkpointing and are not equilibrium certification.
At 142 simultaneous chains, one billion sweeps per chain projects to 33.15 hours
local or 23.33 hours nonlocal, kernel-only. The two-hour target was not reached.
Only L40S was runtime-validated; sm89/sm90 compilation is not an H100 measurement.

Native-source digest:
`9a5af33e466b8ad30ea2e18e87b2637a59ea70f9f24a28b71499ae94b1e720c1`.
Raw receipts and CPU replay: `modal_cuda_results/`; conservative development
reservations total USD 0.99, not an invoice. No CUDA changes were pushed.

## 9. Integration with solid-opt at 070c2d6

The upstream addition of `solid-opt/` leaves the GPU-validated native source
digest above unchanged. Both sm89/sm90 CUDA targets rebuild successfully.
CPU production regressions, including full-width seeds and counter-based resume,
pass after the merge.

Original and optimized CPU backends matched all 13 shared output fields in 36
cases: L4/L8/L21, beta 2.8/3.2, local/nonlocal/both, two full-width seeds, three
chains, equilibration, and decimated histories. The optimized CPU also reproduced
both complete downloaded CUDA smoke campaigns, including checkpoint boundaries,
energies, acceptance histories, and snapshots.

Nonlocal sampling is two independent uniform site draws. Self-pairs and
same-state pairs consume attempts without changing the lattice; there is no
nearest-neighbor or distance-two exclusion. The CPU reference traces actually
compared with CUDA contain 79 accepted adjacent nonlocal moves and 379 accepted
distance-two moves across the reference cases. These are coverage counts, not
independent statistical samples. CUDA uses the adjacent correction at distance
one and removes the double-counted occupied-common-neighbor terms at distance two.
No additional CUDA discrepancy was found.

The CPU rates in section 8 are for the original `solid/` reference, not the newly
added optimized backend. This integration used no additional GPU compute.
