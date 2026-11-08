# Local dynamics at 1e9 sweeps in 2 h

## Target

- L=21, N=9261 sites.
- One sweep = N local Metropolis pair moves.
- 1e9 sweeps = 9.261e12 local proposals per config.
- 2 h = 7200 s.
- Required per config = 1.286e9 local proposals/s.
- 1000 configs = 1.286e12 proposals/s aggregate.

The target is a per-config rate floor, not a total-work target. Splitting work across more devices does not help one config unless within-config parallelism exists.

## Verdict

- GPU: most credible. One config per SM, modeled near 1.9 h per config. 1000 configs need roughly 8 H100-class GPUs for 2 h. No GPU benchmark exists yet.
- FPGA: not viable for thousands of configs at this rate. One config per engine is port-bound near 250M/s at 250 MHz, so it needs ~6 independent lanes per config. Shared-proposal SIMD couples configs and violates independence.
- CPU: one config is 50+ h on one core. It cannot meet 2 h without within-config parallelization.

## Exact local dependency

Local move: random site a, random neighbor b, reject equal types, Metropolis swap.

Affected set: endpoints plus their neighbors. For adjacent a,b on a cubic lattice this is 12 distinct sites. Two moves commute exactly iff all endpoint pairs have torus Manhattan distance >= 3.

The CPU sequential kernel is the reference. The parallel colored kernel has the same stationary distribution but a different trajectory and a different integrated autocorrelation time. If dynamics or tau are measured, this is a different model. If only equilibrium observables are needed, it is valid.

## Two correctness contracts

Keep these separate:

1. CPU-exact: same proposal order, accept/reject results, state, and RNG state as `cpu/lattice.hpp`.
2. Equilibrium-exact: same stationary distribution; proposal order and dynamics may change.

The current golden model is not a local reference. Its `evaluate` rejects every adjacent pair as null, so it cannot test local moves. A local golden model is required before GPU/FPGA work.

## Candidate 21-color schedule

Astra proposed a fixed schedule:

- For positive x edges anchored at (x,y,z), color = (x + 3y + 5z) mod 21.
- The 12 affected offsets have distinct residues.
- Each color holds 21^2 = 441 proposals.
- 21 colors for one axis = 9261 proposals.
- Choose the axis uniformly, independent of state. All three axes make three sweeps.
- Select forward or reverse phase order with equal probability to make the sweep reversible.

This is a candidate, not yet proven. The coloring must be checked on the move-conflict graph, not the site graph.

Correct conflict rule: two moves may run concurrently only if their 12-site affected sets are disjoint. For bonds e,f, that is min torus Manhattan distance between endpoints >= 3. Ordinary checkerboard and ordinary edge coloring are insufficient.

All pairs of the 27,783 undirected bonds need checking. If the 21-color construction fails any pair, it is invalid.

A valid colored kernel preserves the fixed-composition Boltzmann distribution. It does not preserve the original random-site/random-direction trajectory or tau.

## GPU design

- One config per CTA/SM. 512 threads, not one warp.
- State: one byte per site, 9.26 KB per config. 1000 configs = 9.3 MB total.
- Load config state into shared memory. Do not stream from HBM.
- One 21-color sweep has 21 barriers and 9261 proposals.
- Required time is 7.2 microseconds per sweep = 1.286e9 proposals/s/config.
- Per-config counter-based RNG. No shared proposals, no common random numbers.
- No per-proposal atomics; color writes are disjoint.

All GPU timing below is a model, not a measurement. One config on one H100 SM is modeled near 1.9 h. 1000 configs at 132 SMs/GPU = 7.6 waves. One GPU is modeled near 14.5 h; eight GPUs near 1.8 h.

All 1,000 configs in 2 h require 1.286e12 proposals/s aggregate: 160.8G/s per GPU on 8 GPUs, or 1.22G/s per SM. State is only 9.26 MB for 1,000 configs, so capacity is not the limit. The limit is the serial RNG/read/energy/accept/write path and internal state traffic.

Source-level state traffic is about 11 bytes/proposal; at full campaign rate that is about 14 TB/s, before transaction amplification. HBM cannot carry it. State must remain in SMEM/shared memory; HBM is only checkpoint/streaming.

Estimated campaign: 8 H100-class GPUs, about 2 h, 1000 configs at a time. 3000 configs need three waves, about 5.4 h, or 24 GPUs for 2 h. No GPU benchmark exists yet.

## FPGA design

- One config per engine: seven banks give at most two addresses per bank for the 12-site closure. Two ports per bank, one read and one write pass. At 250 MHz this is ~250M proposals/s/config, not 1.286G.
- Six independent proposal lanes per config are required at 250 MHz. 1000 configs then need 6000 lanes, plus state and port arbitration. This does not fit or cost less than GPU.
- Site-transposed SIMD packs many configs per word, but all lanes then share proposal addresses. That couples configs and destroys independent RNG streams. Reject.
- Full multi-pumping does not remove the same-config port and dependency limit enough to close 5x.
- Do not buy VU47P hardware for this target.

## Cheapest falsifying experiments

1. Single-thread CPU local kernel, L=21, state in L1, 1e7 proposals. Measure ns/proposal. Expected 4-8 ns. If >0.777 ns, serial cannot meet 2 h.
2. GPU one-config-per-SM kernel, 1e7 proposals, full 12-site closure, shared-memory state. Measure proposals/s. Kill if below 1.286e9/s/config.
3. GPU eight-config microbenchmark for 60 s. Project 1000 configs and compare with 2 h.
4. Checkerboard/colored kernel vs sequential CPU at L=8: energy distribution, acceptance, and tau ratio. If tau differs materially, the parallel kernel cannot reproduce dynamical results.
5. Counter RNG: verify 64-bit counter headroom for 9.261e12 proposals and bit-exact restart.

## Kill criteria

- Kill the GPU route if one config on one SM cannot reach 1.286e9 proposals/s with exact batch writes.
- Kill the FPGA route if one config needs more than one engine at the target rate, or if configs must share proposal addresses.
- Kill any route if the required observable is the sequential trajectory or tau, not just the equilibrium distribution.
- Kill the 21-color kernel if any same-color bond pair has a non-disjoint 12-site affected set.
- Kill any GPU/FPGA route until a local C++ golden model and color-exhaustiveness test exist.

## Path forward

1. Measure CPU local ns/proposal.
2. Build one GPU CTA for one config, colored batches, shared-memory state, counter RNG.
3. Measure one config rate and tau agreement.
4. Scale to 8 configs, then 1000 configs across 8 GPUs.
5. Use FPGA only as a separate single-config instrument if its one-config rate is useful. It is not the thousands-config engine.

## Sources

- cpu/lattice.hpp local move and affected set.
- Golden exactness work in FPGA_WIDE_LEARNINGS.md.
- Xoshiro/xoroshiro RNG.
- CUDA shared-memory/CTA model.
- Xilinx UG573/UG901 for BRAM port limits.
- Metropolis/Hastings; parallel MCMC and prefetching literature.


## Gate 1 results (2026-11-08)

- `golden/local_model.hpp`: exact L=21 local state, 12-site affected set, delta, commit, Metropolis step.
- `golden/local_tests.cpp`: PASS. 55,566 directed bonds; delta vs full energy; fixed composition; 441-move class; 100,000 random proposals.
- `golden/color_validate.cpp`: PASS. 27,783 bonds; 63 classes; 6,112,260 same-class pairs; zero conflicts.

The 21-color schedule is valid per axis. Axis sweeps must be serialized; cross-axis same-color conflicts exist. It remains a different scan from the original random-site/random-direction kernel.
