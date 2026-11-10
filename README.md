lattice-glass sampler with a CPU reference. WIP: Metal + CUDA backends for max depth and breadth

(deep into glassy regime, for as many samples in parallel as possible)

```bash
make -C solid clean all
```

GPU backends are optional

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

Arguments: `L beta rho rho1 rho2 sweeps chains CPU|GPU mode equil [state.raw]`
followed by optional `--seed N`, `--history-every N`, `--max-history-samples N`.

Use a different `--seed` for each independent batch; the default is `1` for
reproducible replay. The seed controls initialization and proposal randomness.
`solid/aging_driver.py` forwards the same options.

Histories default to at most 100,000 samples per chain, not one allocation per
requested sweep. The effective interval is
`max(history_every, ceil(sweeps / max_history_samples))`; defaults are `1` and
`100000`. The final sweep is always recorded. Short runs retain every sweep.
`history_sweeps` gives the production-relative sweep of each energy/acceptance
column; `history_every` and `seed` are also saved in the NPZ.
Equilibration histories are discarded, and temporary GPU histories are chunked.

CPU-only regression checks: `make -C solid test-production`.

