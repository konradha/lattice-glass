# solid-opt2

Performance-oriented clone of the solid sampler. No tempering, replica exchange,
or ensembling.

## Layout

```text
solid-opt2/
  cpu/    CPU driver and shared lattice/EFMA headers
  metal/  Metal driver and kernel
  gate/   bitwise and statistical gates
  bench/  sustained scale benchmark
```

Build:

```bash
make -C solid-opt2/cpu
make -C solid-opt2/metal
```

## Kernel

- Nonlocal: any two distinct sites; equal types are no-op; exact ΔE and EFMA.
- Local: nearest-neighbor swaps.
- CPU local uses the reference random-edge ordering.
- Metal local at L21 uses a validated 21-colour edge schedule. It changes the
  trajectory and requires the statistical gate. Other L values use the exact cut.
- Metal nonlocal uses a persistent 4-warp/128-thread threadgroup per chain and an
  exact cross-warp prefix cut. Re-proposal after a cut preserves the sequential
  trajectory.
- Thresholds, coordinate/neighbor tables, and state stay resident per segment.

## Gates

Exact gate for implementation changes:

```bash
cd solid-opt2/gate
python3 check_correctness.py --schedule colored --long
```

Statistical stationarity gate for the coloured local schedule:

```bash
cd solid-opt2/gate
python3 statistical.py --mode both
```

The exact gate passed. The statistical gate passed at L21, beta 2.0, 64 chains
per backend, alpha 0.01: joint p=0.692, both stationarity checks ok.

High-beta gate: L21, beta 2.5, 160k sweeps, 8 chains: CPU 88.3 s, Metal 74.8 s;
nonlocal fields bitwise equal.

## Sustained rates

L21, beta 2.0, 1000 timed sweeps, 200 warmup:

| backend | mode | 8 chains | 64 | 128 | 256 |
|---|---|---:|---:|---:|---:|
| CPU | nonlocal M/s | 202 | 208 | 196 | 187 |
| CPU | local M/s | 133 | 158 | 163 | 163 |
| Metal | nonlocal M/s | 115 | 381 | 437 | 458 |
| Metal | local M/s | 223 | 593 | 620 | 604 |

Per-chain rates at 8 chains: CPU nonlocal 25.3M/s, local 16.6M/s; Metal
nonlocal 14.4M/s, local 27.9M/s.

The benchmark writes `bench/scale.csv`. Raw campaign NPZ files are local-only.
Capped campaign wall: CPU 149.6 s, Metal 145.6 s. Nonlocal campaign is bitwise
equal; Metal L21 local is the coloured schedule and is covered by the statistical
gate, not bitwise equality.
