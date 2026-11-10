lattice-glass sampler with a CPU reference. WIP: Metal + CUDA backends for max depth and breadth

(deep into glassy regime, for as many samples in parallel as possible)

```bash
make -C solid clean all
```

Metal is optional. On macOS it is enabled when the framework is present.
Otherwise `make` prints a warning and builds the CPU backend:

```bash
make -C solid HAVE_METAL=0
```


```bash
make -C solid test
make -C solid test-long
```


## Run

```bash
solid/driver 21 2.0 0.75 0.30 0.45 1000 8 GPU nonlocal 100
solid/driver 21 2.0 0.75 0.30 0.45 1000 8 GPU local 100
```

Arguments: `L beta rho rho1 rho2 sweeps chains CPU|GPU mode equil [state.raw]`.

