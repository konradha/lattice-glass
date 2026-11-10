# Sustained scale benchmark

`scale.py` measures sustained proposal throughput for the CPU and Metal solid
drivers. It runs one driver process at a time and sweeps the chain counts
8, 64, 128, and 256 over both backends and both move modes (`nonlocal` and
`local`).

## Build

```bash
make -C solid-opt2/cpu
make -C solid-opt2/metal
```

The CPU driver must be built with its default `SOLID_HAVE_METAL=0`; the Metal
driver must be built with `SOLID_HAVE_METAL=1`. The harness passes `CPU` to the
CPU driver and `GPU` to the Metal driver. If a Metal driver was built without
Metal support, it falls back to the CPU and writes a warning to stderr; the
harness treats that warning as a failed case.

## Run

```bash
python3 solid-opt2/bench/scale.py
```

Defaults:

- `L=21`, `beta=2.0`, `rho=0.75`, `rho1=0.30`, `rho2=0.45`, `seed=1`
- backends `cpu,metal`; modes `nonlocal,local`; chains `8,64,128,256`
- `200` untimed warmup sweeps and `1000` timed sweeps per chain
- CSV output at `solid-opt2/bench/scale.csv`
- progress and error details on stderr; the summary table on stdout

Exit status is `0` when every case succeeds, `1` when any case fails, `2` for
invalid arguments or an unusable driver/CSV path, and `130` when interrupted.
Completed cases are flushed to the CSV as they finish, so an interrupted run
leaves partial results.

## What is measured

For each case the harness runs:

1. An untimed warmup, passed to the driver as `EQUIL_SWEEPS`.
2. A timed interval of `--sweeps` sweeps, from which the driver reports wall time.

The driver's wall time excludes process startup, kernel compilation, warmup,
and `.npz` writing. It includes the driver's internal sweep chunking and
snapshot copies, so the reported rate is the driver's sustained rate for the
chosen case rather than a bare kernel time. The driver prints wall time with
millisecond resolution; a timed interval that rounds to `0.000s` is reported as
an error, so increase `--sweeps` for short runs.

The CSV file is overwritten on each run. Use `--csv` to write elsewhere.

Every sweep issues exactly `L**3` proposals per chain in either mode, so:

```text
attempts/s per chain = sweeps * L**3 / wall
aggregate attempts/s  = chains * attempts/s per chain
```

The CSV records the raw rates; the table shows them in millions of attempts per
second. `attempts_per_sweep` is recorded explicitly so post-processing does not
need to know the mode.

CSV columns:

```text
backend,mode,chains,L,sweeps,warmup,attempts_per_sweep,wall_s,
attempts_per_chain_s,aggregate_attempts_s,status,error
```

`wall_s` is the driver-reported timed wall; `status` is `ok` or `error`; `error`
is a short single-line diagnostic for failed cases.

## Concurrency

Cases always run sequentially. The script never runs a CPU and Metal case at the
same time, and never runs two Metal processes concurrently.

For CPU cases the driver receives `NUM_CORES = chains` and creates
`min(chains, 64)` worker threads inside that one process. Chain counts above the
machine's logical CPU count therefore oversubscribe only because that chain
count was explicitly requested. Do not run two `scale.py` instances at once on
the same machine; Metal must stay one process at a time.

Each case gets its own temporary working directory, so the driver's `run.npz`
never collides between cases or with another run. Successful work directories
are deleted. With `--keep-workdir`, the directory for a failed case is kept and
its path is printed for inspection.

## Options

| Option | Environment variable | Default | Meaning |
|---|---|---|---|
| `--backends LIST` | `SOLID_BENCH_BACKENDS` | `cpu,metal` | Backends to run |
| `--modes LIST` | `SOLID_BENCH_MODES` | `nonlocal,local` | Move modes to run |
| `--chains LIST` | `SOLID_BENCH_CHAINS` | `8,64,128,256` | Chain counts to run |
| `--L N` | `SOLID_BENCH_L` | `21` | Lattice side length, 4..32 |
| `--beta VALUE` | `SOLID_BENCH_BETA` | `2.0` | Inverse temperature, passed verbatim |
| `--rho VALUE` | `SOLID_BENCH_RHO` | `0.75` | Total occupied fraction |
| `--rho1 VALUE` | `SOLID_BENCH_RHO1` | `0.30` | Type-1 occupied fraction |
| `--rho2 VALUE` | `SOLID_BENCH_RHO2` | `0.45` | Type-2 occupied fraction |
| `--sweeps N` | `SOLID_BENCH_SWEEPS` | `1000` | Timed sweeps per chain |
| `--warmup N` | `SOLID_BENCH_WARMUP` | `200` | Untimed warmup sweeps |
| `--seed N` | `SOLID_BENCH_SEED` | `1` | Driver seed |
| `--timeout S` | `SOLID_BENCH_TIMEOUT` | `3600` | Per-case wall timeout |
| `--cpu-driver PATH` | `SOLID_CPU_DRIVER` | `solid-opt2/cpu/driver` | CPU executable |
| `--metal-driver PATH` | `SOLID_METAL_DRIVER` | `solid-opt2/metal/driver` | Metal executable |
| `--csv PATH` | `SOLID_BENCH_CSV` | `solid-opt2/bench/scale.csv` | CSV output file |
| `--env KEY=VALUE` | none | none | Extra driver environment; repeatable |
| `--keep-workdir` | none | off | Keep work directories for failed cases |

All relative driver and CSV paths are resolved against the current working
directory. The `--env` option is applied on top of the current environment for
every driver process.

## Validation

The harness fails with a usage error before running anything when:

- a backend or mode is unknown or listed twice;
- a chain count is not a positive integer or is listed twice;
- `L` is outside `[4, 32]` (driver coordinates are packed in 5 bits);
- `beta` is not a finite non-negative decimal number;
- `rho`, `rho1`, or `rho2` is not finite and non-negative;
- `rho != rho1 + rho2`, `rho1 + rho2 > 1`, or the rounded occupation is empty
  or exceeds `L**3`;
- `sweeps < 1`, `warmup < 0`, or `sweeps + warmup` makes the driver's sweep or
  proposal index exceed 64 bits;
- `seed` is outside `[0, 2**64 - 1]`, or `timeout` is not finite and positive;
- a selected driver is missing or not executable, or the CSV directory cannot
  be created or the CSV path is a directory.

Per-case failures do not abort the sweep. A non-zero driver exit, timeout,
unexpected stderr output, a missing or malformed `done ... wall=...` line, or a
reported `L`/`chains`/`sweeps` mismatch produces an error row, the remaining
cases still run, and the final exit status is `1`.

## Examples

```bash
# CPU only
python3 solid-opt2/bench/scale.py --backends cpu

# Metal only, with a longer timed interval
python3 solid-opt2/bench/scale.py --backends metal --sweeps 5000 --warmup 500

# Custom chain counts and driver locations
python3 solid-opt2/bench/scale.py \
  --chains 8,64 \
  --cpu-driver solid-opt2/cpu/driver \
  --metal-driver solid-opt2/metal/driver \
  --csv /tmp/scale.csv

# Same knobs through environment variables
SOLID_BENCH_BACKENDS=cpu \
SOLID_BENCH_CHAINS=8,256 \
SOLID_BENCH_SWEEPS=2000 \
SOLID_CPU_DRIVER=solid-opt2/cpu/driver \
python3 solid-opt2/bench/scale.py

# Pass an extra variable to every driver process
python3 solid-opt2/bench/scale.py --env FOO=bar

# Keep run.npz files for failed cases
python3 solid-opt2/bench/scale.py --keep-workdir
```
