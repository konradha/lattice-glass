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
