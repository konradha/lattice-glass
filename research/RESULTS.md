# Kernel research results

Time-boxed pass. Single thread. Apple M3, 8 cores, clang 21.1.0.
The benchmark is in `research/kernel_bench/`. Build and run commands are there.

## Correctness

The byte/mask kernel passed 20,000 exact comparisons:
incremental ΔE equals full-energy difference, and swapping the same pair twice
restores the original energy. The packed baseline is included only as a
performance reference.

## L=25, beta=3.5, 1.5M proposals, best of 3

| Variant | ns/proposal | Mprop/s | acceptance |
|---|---:|---:|---:|
| packed / mt19937 / exp / mixed | 115.24 | 8.68 | 2.29% |
| byte / mt19937 / exp / mixed | 70.93 | 14.10 | 2.29% |
| byte / mt19937 / table / mixed | 70.87 | 14.11 | 2.29% |
| byte / xoshiro256++ / table / mixed | 50.92 | 19.64 | 2.22% |
| byte / Philox4x32-10 / table / mixed | 60.64 | 16.49 | 2.19% |
| byte / Philox / table / local only | 54.56 | 18.33 | 2.37% |
| byte / Philox / table / nonlocal only | 50.98 | 19.61 | 2.27% |

The byte/mask kernel is 1.6x faster than packed with the same RNG.
With xoshiro it is 2.26x faster. With Philox it is 1.90x faster.

## Size dependence, beta=3.5

| L | packed / mt | byte / mt | byte / xoshiro | byte / Philox |
|---:|---:|---:|---:|---:|
| 8 | 108.51 | 67.21 | 48.47 | 57.52 |
| 12 | 112.86 | 69.79 | 49.21 | 58.05 |
| 20 | 111.02 | 70.26 | 49.23 | 59.49 |
| 25 | 115.24 | 70.93 | 50.92 | 60.64 |

Cost per proposal is nearly flat. The state fits cache. L=25 is not a
throughput problem for the kernel.

## Beta dependence, L=25

At beta=1.0 the acceptance is 5.5% and the timings are within 2% of beta=3.5.
Acceptance is low in both cases because the benchmark starts from a random
configuration. These numbers measure move cost, not equilibrium acceptance.

## Checkpoint I/O

Atomic write: temp file, `fsync`, `rename`, directory `fsync`, on local APFS.

| Size | median | p99 |
|---:|---:|---:|
| 2 KiB | 161.8 us | 524.7 us |
| 8 KiB | 159.8 us | 532.8 us |
| 64 KiB | 168.2 us | 549.4 us |
| 1 MiB | 537.1 us | 13.66 ms |

The metadata operation dominates up to 64 KiB. A 16 KiB checkpoint is cheap.
S3 is the slow path; upload on a timer, not every checkpoint.

## Campaign arithmetic at L=25

One sweep = 15,625 proposed swaps.

| Kernel | ms/sweep | 1e8 sweeps | 1e9 sweeps |
|---|---:|---:|---:|
| packed / mt | 1.80 | 50.0 h | 20.8 days |
| byte / Philox | 0.947 | 26.3 h | 11.0 days |
| byte / xoshiro | 0.796 | 22.1 h | 9.2 days |

For 10,000 chains on 10,000 cores, wall time equals the per-chain time.
Core-hours: 263k for 1e8, 2.63M for 1e9 with Philox.
At USD 0.03/core-hour that is about USD 7.9k and USD 79k.
At USD 0.06/core-hour it is about USD 15.8k and USD 158k.

## Recommendations

1. Use byte types, 6-bit neighbor masks, `local_energy[3][64]`, and the
   affected-set delta. Commit the precomputed affected state only on
   acceptance. This avoids mutate/revert on the 97-98% rejected proposals.
2. Use Philox4x32-10 with key `(run_seed, chain_id)` and counter equal to the
   move index. Resume needs no RNG state. It costs about 19% versus xoshiro at
   L=25. That is acceptable for a fault-tolerant 10k-core campaign.
   Use xoshiro256++ only if throughput is the dominant constraint.
3. Keep the per-beta `exp` table. The measured difference is negligible at 2%
   acceptance, but the table removes `exp` from the hot path at high
   acceptance and makes acceptance deterministic.
4. Use a 50/50 local/nonlocal mix for annealing. Acceptance and cost are close.
   Tune after the pilot. Nonlocal is slightly cheaper per proposal.
5. Keep byte arrays in memory. Pack to 2-bit only for checkpoints and
   snapshots. The packed in-memory path is the slow path.
6. Do not use `std::mt19937`. It is slower and its stream serialization is not
   portable.

## Caveats

- These are relative numbers on one machine. No SIMD, no prefetch, no
  multi-threaded run.
- The acceptance is from a random start, not from equilibrium.
- The L=25 lattice is 15,625 sites. A two-bit checkpoint is about 3.9 KiB;
  a byte checkpoint is 15.6 KiB. Either is fine.
- Statistical reproducibility is the target. Integer ΔE plus a per-stage table
  is enough. Full bitwise replay across architectures is not claimed.
