# Gates

Two gates protect the solid sampler in `solid-opt2`. Both run from this
directory, need only Python 3 and numpy plus the built drivers, and compare the
CPU backend against the Metal backend. Neither uses tempering, replica
exchange, or ensembling.

## Which gate applies to which change

| Change class | Gate | Criterion |
|---|---|---|
| Compiler flags, memory layout, packed tables, fast paths, plumbing, precision-preserving refactors | `check_correctness.py` on schedule-identical paths | Every saved field bitwise equal; use `statistical.py` for the L21 coloured local path |
| Local-update order or a colored/checkerboard schedule, where CPU and Metal realize different move groupings or a different RNG-to-move map | `statistical.py` | Joint energy/occupancy distribution indistinguishable; effect sizes reported |
| Ordering change that still maps the same RNG stream to the same moves on both backends | `check_correctness.py` first, then `statistical.py` | Bitwise equality of the new schedule, plus a distributional check |
| Documentation, benchmarks, analysis scripts that do not touch sampler code | none | — |

Rubric for an ordering change:

1. Document the schedule, why it preserves the stationary distribution, and
   what parallel realization CPU and Metal use for it.
2. If the RNG stream still maps to the same moves on both backends, the exact
   gate must pass; the statistical gate is then an additional check.
3. If bitwise equality is not achievable (for example the CPU walks colors in a
   different order than a warp-parallel Metal kernel), the statistical gate is
   the backends' equivalence criterion. It does not certify absolute
   correctness: two implementations of the same wrong kernel agree. The
   stationarity argument and, where available, an exact tiny-lattice reference
   are still required.
4. Record the statistical gate's effect sizes and p-values with the change.

The statistical gate compares two ensembles drawn from the same initial-state
distribution under identical bath parameters. It is a backend-equivalence test,
not a test against the analytic target.

## Exact gate (`check_correctness.py`)

```bash
cd solid-opt2/gate
python3 check_correctness.py            # schedule-identical cases
python3 check_correctness.py --long     # adds 1000-sweep cases
python3 check_correctness.py --schedule reference   # pre-coloured baseline
```

Each case runs the CPU binary on the CPU backend, then the Metal binary on the
GPU backend, with the same seed and bath, and requires every field of
`run.npz` to match exactly: `initial`, `equilibrated`, `final`,
`snapshot_sweeps`, `snapshots`, `energy`, `accept_local`, `accept_nonlocal`,
`scalars`, `meta`, `seed`, `history_every`, `history_sweeps`. Use
`--seed N` to change the shared seed (default 1).

Because the Metal local phase at L=21 is the 21-colour edge schedule, it is a
different trajectory from the CPU reference local scan. `--schedule colored`
(the default) therefore runs only schedule-identical paths: L4 both/local,
L21 nonlocal, and `--long` uses L21 nonlocal and L4 both. `--schedule
reference` runs the pre-coloured case set, including L21 local and both, and is
correct only when both backends implement the reference scan. The coloured
L21 local path is validated by `statistical.py`.

### Driver resolution

First hit wins:

| Backend | Options |
|---|---|
| CPU | `--cpu-driver`, `$SOLID_CPU_DRIVER`, `$SOLID_DRIVER`, `<solid-opt2>/cpu/driver`, `<solid-opt2>/metal/driver`, `./driver` |
| Metal | `--metal-driver`, `$SOLID_METAL_DRIVER`, `$SOLID_DRIVER`, `<solid-opt2>/metal/driver`, `./driver` |

The CPU run always uses the CPU binary and the GPU run the Metal binary.
Before running cases, each driver gets a one-sweep probe. The Metal probe
rejects a binary built with `SOLID_HAVE_METAL=0`, because that binary prints a
warning and silently falls back to CPU; without the probe the gate could pass
by comparing CPU with itself. If Metal is unavailable, the gate exits 2 with
the probe error, unless `--skip-metal` is given, in which case it prints
`SKIP` and exits 0.

Exit codes: 0 all fields equal, 1 mismatch, 2 setup or driver error.

## Statistical gate (`statistical.py`)

```bash
cd solid-opt2/gate
python3 statistical.py
python3 statistical.py --L 10 --sweeps 512 --burn-in 512 \
    --chains 8 --runs 2 --perms 500 --json summary.json   # quick smoke
```

Defaults: L=21, beta=2.0, rho=0.75 (rho1=0.30, rho2=0.45), mode `both`,
production 2048 sweeps, burn-in 2048 sweeps, 16 independent chains x 4 runs per
backend (64 chains each), 2000 permutations, alpha 0.01. L=21 is the default
because it is the only size where Metal runs the 21-colour local schedule; at
L=21 the CPU's reference local scan and the Metal coloured schedule are exactly
the comparison of interest. Runtime is roughly one to two minutes on an
8-core laptop, mostly driver time. The L=10 command above is a fast
implementation-only smoke test where both backends use the reference schedule.

### Protocol

1. Both backends receive independent draws from the same initial-state
   distribution `D0`: `n1 = round(rho1*N)` type-1 sites and
   `n2 = round(rho2*N)` type-2 sites placed uniformly without replacement on
   the torus, with neighbor occupancy masks computed exactly as in
   `lattice.hpp`. Every chain, run, and backend gets its own draw; the master
   seed is split into initial-state and driver seeds.
2. Both backends run with identical bath parameters, mode, burn-in, and
   production sweeps. This is asserted from each `run.npz`: `meta`, `scalars`,
   history coverage and decimation, state-file load, and type counts. Seeds
   differ by backend and run, so the chains are statistically independent.
3. The driver discards the burn-in. Energy and acceptance histories cover
   production sweeps 1..`sweeps` only.
4. A stationarity diagnostic splits each chain's production window into early
   and late halves and sign-flips per-chain differences across chains. It is a
   Geweke-style split check on energy and density. If it rejects at
   `--stationarity-alpha` (default 1e-3), the gate reports `INCONCLUSIVE`
   (exit 2) rather than giving a meaningless comparison. `--no-require-stationary`
   downgrades this to a printed warning.

### Statistics

Chains are the independent units. Two-sample tests relabel whole chains
between backends, which is exact under the null (same joint law) and makes no
distributional assumptions. Six chain-level functionals are tested:

| Functional | Statistic | Reported effect size |
|---|---|---|
| Energy distribution | sum of squared standardized mean differences over mean, sd, q10, q25, q50, q75, q90 | per-feature standardized difference `d` |
| Occupancy field | sum of squared per-site differences in site-occupancy probability | rms and max absolute difference, permutation 95th percentile of max |
| Occupancy field (max) | maximum per-site absolute difference | same |
| Type-1 field | as occupancy, for the type-1 indicator | rms, max, null 95th percentile |
| Type-1 field (max) | maximum per-site absolute difference | same |
| Local environment | sum of squared standardized differences over the histogram of occupied-neighbor counts (0..6) for type-1 and type-2 sites | max absolute `d` |

Occupancy and type fields are averaged over production snapshots from
`snapshot_sweeps >= sweeps/8` plus the final state, so each chain contributes
several configurations without treating them as independent replicates.

The joint test combines the six component p-values by a union-intersection
(min-p) rule calibrated on the shared permutations. This is a practical
Geweke-style joint-distribution check: the null is that both kernels sample
the same joint law of energy, occupancy, type occupancy, and local
environment. It has power against changes in any of those laws; a change that
preserves all six functionals while altering only residual dependence between
them would not be detected. Simulation-based calibration is not applicable to
this driver because the bath is fixed rather than drawn from a prior; add it
separately if a parameter-sampling driver is introduced.

Acceptance rates (`accept_local`, `accept_nonlocal`) are schedule diagnostics.
A colored or reordered schedule legitimately changes per-move acceptance while
preserving the stationary law, so acceptance is reported with standardized
effect sizes and a permutation p-value but does not fail the gate by default:

- `--acceptance-gate warn` (default): report and flag `[WARN]` below alpha.
- `--acceptance-gate fail`: treat a significant acceptance difference as a
  failure. Use only when the schedule is supposed to leave acceptance
  unchanged.
- `--acceptance-gate off`: report effect sizes and p without a verdict note.

### Output and exit codes

The report prints per-backend energy, acceptance, and chain counts; one row
per functional with statistic, p-value, and effect sizes; the joint p-value;
the acceptance row; and the stationarity diagnostics. Example tail:

```text
  joint (min-p, 6)                    0.7321  PASS
  acceptance                T=0.81      0.4425  d(local)=+0.031 d(nonlocal)=-0.014
  stationarity CPU   p=0.4021 (energy 0.3550, density 0.5011) d_z(energy)=+0.021 rms(density)=0.0018 [ok]
  statistical gate: PASS (alpha=0.01, B=2000)
```

Exit codes: 0 pass, 1 distribution mismatch (or acceptance failure under
`--acceptance-gate fail`), 2 setup error or inconclusive stationarity. A
significant joint p-value is a real backend disagreement at the stated
effect sizes; failing the acceptance policy with a small effect size means the
schedule differs more than expected.

`--json PATH` writes config, backends, component statistics and p-values,
effect sizes, stationarity, verdict, and exit code.

### Tuning and limits

- Power grows with `--chains`, `--runs`, and `--sweeps`, because the null
  width is set by between-chain variation. Long autocorrelation widens the
  null, not the test's validity; increase chains or production sweeps rather
  than lowering alpha.
- `--mode both` is the default because the shared nonlocal moves mix globally
  and keep the ensemble stationary. `--mode local` isolates the coloured local
  phase and doubles the local proposal count, but local-only dynamics mix
  slowly; expect to raise `--burn-in` and `--sweeps` if the stationarity
  diagnostic rejects.
- `--perms` bounds p-value resolution; keep `perms >= 1/alpha`. The default
  2000 resolves 0.01 comfortably.
- The Metal kernel supports `N <= 9261`, so the gate requires `--L <= 21`.
  L=21 is also the only size where the coloured local schedule is used; a
  lower L compares the reference scan on both backends and is an
  implementation smoke test, not a coloured-schedule test.
- Occupancy snapshots are powers of two, so the gate requires
  `--sweeps <= 1000000`. Beyond that the driver snapshots on wall-clock time
  and the backends' occupancy windows would differ.
- The probe rejects a Metal binary without Metal; `--skip-if-no-metal` prints
  `SKIP` and exits 0 for CPU-only environments.
- Driver resolution is the same as for the exact gate, with
  `--cpu-driver` / `--metal-driver` and the `SOLID_CPU_DRIVER`,
  `SOLID_METAL_DRIVER`, `SOLID_DRIVER` environment variables.

## Building the drivers

```bash
make -C solid-opt2/cpu
make -C solid-opt2/metal
```

The Metal driver resolves its MSL source from the path baked in at build time
(`SOLID_MSL_PATH`). Rebuild after editing `kernels.metal`; a stale binary can
fail at runtime on a freshly edited kernel.



