> # Kernel benchmark

Time-boxed comparison of move kernels for the NH lattice glass.

## Build

On this macOS host the micromamba clang fails at link time. Use the system
clang:

```bash
/usr/bin/clang++ -O3 -mcpu=native -std=c++17 -DNDEBUG bench.cpp -o bench
```

On Linux:

```bash
clang++ -O3 -march=native -std=c++17 -DNDEBUG bench.cpp -o bench
```

## Run

```bash
./bench L proposals beta reps
```

Example: `./bench 25 1500000 3.5 3`.

## What it compares

- `packed/mt/exp`: current design. 2-bit packed lattice, `std::mt19937`,
  `std::exp` per accepted proposal, affected-set recompute.
- `byte/mt/exp`: byte types, 6-bit neighbor masks, `local_energy[3][64]`,
  delta computed without mutation, commit only on acceptance.
- `byte/*/table`: same kernel with a per-beta `exp` table.
- `byte/xo/table`: xoshiro256++ with Lemire bounded integers.
- `byte/ph/table`: Philox4x32-10 with Lemire bounded integers.
- `byte/ph/table/local` and `.../nonlocal`: move-mix extremes.

The benchmark checks the incremental delta against a full-energy recompute
for 20,000 random swaps before timing. It reports the acceptance rate from the
last repetition.
