# solid-opt

Performance-oriented clone of the solid sampler. No tempering, replica exchange,
or ensembling. The proposal kernel is unchanged: any two distinct sites for
nonlocal moves; nearest neighbors for local moves; MF acceptance is EFMA.

## Layout

```text
solid-opt/
  cpu/    CPU driver and shared lattice/EFMA headers
  metal/  Metal driver and kernel
  gate/   CPU/GPU equivalence checks
```

Build:

```bash
make -C solid-opt/cpu
make -C solid-opt/metal
```

## Optimizations in this branch

CPU:
- Incremental energy history; no full lattice scan after each sweep.
- Precomputed neighbor, packed-coordinate, and Lemire rejection tables.
- Fixed affected-set paths for local moves and nonlocal moves with L1 > 2.

Metal:
- Fixed affected-set path for far nonlocal pairs; dedup only for near pairs.
- Warp-summed energy history; no full-state rescans.
- Shared EFMA threshold table and precomputed coordinate/neighbor tables.
- `-O2` plus native, unrolling, and safe integer flags.

## Gates

Implementation changes must preserve the Markov kernel. The current CPU and Metal
paths are checked bitwise:

```bash
cd solid-opt/gate
python3 check_correctness.py --long
```

The gate compares initial, equilibrated, and final states, snapshots, energy,
acceptance histories, scalars, metadata, seed, and decimation fields.

Algorithmic changes would be allowed only with a stationarity gate: Geweke’s
joint-distribution test or SBC, plus exact tiny-lattice references. No such change
is present in this directory.

## Measured rates

L21, beta 2.0, 1000 sweeps:

| Backend | mode | per chain | 8-chain aggregate |
|---|---|---:|---:|
| CPU | nonlocal | ~12.2 M/s | ~98 M/s |
| CPU | local | ~11.7 M/s | ~94 M/s |
| Metal | nonlocal | ~7.0 M/s | ~56 M/s |
| Metal | local | ~5.2 M/s | ~42 M/s |

Metal aggregate scales to roughly 467 M/s nonlocal at 128 chains.
