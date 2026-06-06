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