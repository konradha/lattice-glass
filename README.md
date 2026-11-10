# lattice-glass — clean-solid

EFMA lattice-glass sampler with a CPU reference and a Metal backend.

## Build

```bash
make -C solid clean all
```

Requires macOS Metal, `clang++`, and Boost headers.

## Verify

```bash
make -C solid test
make -C solid test-long
```

See `solid/VALIDATION.md` for the high-beta gate, campaign validation, and
performance checks.

## Run

```bash
solid/driver 21 2.0 0.75 0.30 0.45 1000 8 GPU nonlocal 100
solid/driver 21 2.0 0.75 0.30 0.45 1000 8 GPU local 100
```

Arguments: `L beta rho rho1 rho2 sweeps chains CPU|GPU mode equil [state.raw]`.

## Layout

- `solid/` — driver, EFMA core, Metal kernel, correctness and plot tools
- `plots_linspace/` — campaign figures
- `tools/` — campaign helpers
- `third_party/` — metal-cpp
