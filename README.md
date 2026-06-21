### Lattice glass in 3d
This is a WIP repository to reproduce [this work](https://arxiv.org/abs/2003.02872).

More information to follow.

build and run as
```bash
make
export OMP_NUM_THREADS=4
./to_omp 5.0 0.58 0.29 output.npy
```

Arguments are `beta rho rho1 out_file`, where `rho` is total particle
density and `rho1` is the type-1 particle density.

Deterministic verification only:
```bash
make check
```

`make check` exercises exact species marginalisation and balanced-cluster
detailed-balance invariants on small fixtures. It is not a production physics
run.

Bounded balanced-cluster pilot:
```bash
make pilot
```

The pilot reports closure, abandonment, rejection, acceptance, size-bucket,
boundary-size, and energy-delta summaries on a small seeded lattice with
`L >= 8`. It is a diagnostic smoke run, not a production sampling campaign.

Kernel comparison harness (two same-temperature replicas):
```bash
make
./to_omp --mode compare --kernel heatbath --beta 2.0 --rho 0.75 --rho1 0.30 \
  --sweeps 40000 --warmup 10000 --sample-every 50 --heatbath-every 10
```

`--kernel` is `base | heatbath | cluster`. Compare mode reports local
acceptance, species heat-bath and cluster usage, and the integrated
autocorrelation time and effective sample count of both the energy and the
Rao-Blackwellised replica overlap. Build with `make CPPFLAGS=-DL=10` (or any
even `L`) to change the lattice size.

Cluster diagnostics (free vs reference-coupled disagreement set):
```bash
make experiments/cluster_diagnostics
./experiments/cluster_diagnostics --L 10 --beta 3.0 --epsilon 0.5
```

Franz-Parisi efficiency harness with epsilon(T)-to-percolation calibration:
```bash
make experiments/fp_efficiency
./experiments/fp_efficiency --L 10 --beta 3.0 --target-d 0.12 \
  --production-sweeps 200000 --sample-every 4
```

The harness couples two same-temperature replicas to a common quenched
reference, calibrates the field `epsilon` so the inter-replica disagreement
fraction sits below the 3D site-percolation threshold (~0.3116), and reports
integrated autocorrelation time and ESS-per-second for the base sampler versus
the base+balanced-cluster sampler at matched wall-clock. Pass an explicit
`--epsilon` to skip calibration. These are pilot-scale runs; a publishable
efficiency claim needs multiple seeds, a temperature scan, and size scaling.

Informed (locally-balanced) vs blind nonlocal occupancy swap:
```bash
make experiments/informed_swap_efficiency
./experiments/informed_swap_efficiency --L 8 --beta 2.0 \
  --production 120000 --sample-every 8
python experiments/run_informed_sweep.py --workers 4        # temperature scan
python experiments/run_informed_sweep.py --hard --workers 4 # hard-window budgets
python experiments/analyze_informed.py --csv experiments/informed_results.csv
```

The occupancy bottleneck is the nonlocal swap with a UNIFORM destination, whose
acceptance collapses at low T. `informed_swap.h` keeps the exact move set but
draws the destination vacancy with the Zanella locally-balanced weight
`sqrt(exp(-beta dE))` (exact dE, no surrogate; MH-corrected to
`min(1, Z_i/Z_j)`). The harness compares the trusted all-pairs swap, the blind
occupancy swap, and the informed occupancy swap, reporting occupancy-move
acceptance and the integrated autocorrelation time / ESS-per-second of the
energy and the pure-occupancy bond count.

Findings (multi-seed, Madras-Sokal tau-resolved): informed restores acceptance
monotonically (~9x at T=1.0 to ~53x at T=0.4) and accelerates occupancy
decorrelation per sweep by 4-28x at L=8 (~67x at L=10 -- the advantage GROWS
with system size). Versus the blind occupancy swap it wins per second from
T<=0.67 down through T=0.4 (~2.5x at T=0.4, 6 seeds), and at T=0.5 it edges past
even the trusted full nonlocal swap per second (L=8 1.17x, L=10 1.05x). Two
limits remain: (i) at T=0.4 the full-set O(N_v) cost keeps informed below the
cheap full swap per second (~0.3x) despite the per-sweep win -- a capped /
multiple-try candidate set is the lever to convert it; (ii) at T~0.33 the
occupancy tau is unresolved even at 1.2M sweeps, so the residual slowdown is
barrier-limited (RFOT), not proposal-limited. Informed proposals fix proposal
quality, not thermodynamic barrier crossing.

Capped kernel (`informed_mtm_swap_sweep`, `--mtm-k k`): the full-set proposal is
O(N_v) per move; the Multiple-Try-Metropolis cap scores only `k` random
candidate vacancies, restoring O(N) per sweep (measured: L8->L20 cost x17 for
x15.6 sites, vs the full-set's x244). It is exact (the reverse-weight identity
and the bond-mean stationarity gate hold). Performance is k-dependent and k is
the knob: cost grows ~linearly in k while the per-sweep gain saturates, so small
k wins. At L=10, T=0.5 the optimum is k~4-8, giving ~2x ESS/s over blind --
matching the full-set kernel but at O(N) cost, so it scales. At L=20, T=0.5
(k=4) it is ~2x over blind on energy and ~1x on the bond observable: T=0.5 is the
easy end where blind still mixes, so the margin is thin and single-seed-noisy.
The capped kernel's real payoff is the low-T regime (T<=0.4), where blind tau
explodes and the per-sweep gain is largest; quantifying that needs the long
deep-T runs and is not yet done. Honest status: correct and O(N), a clean win at
L=10, marginal at L=20/T=0.5, deep-T win projected but unverified.