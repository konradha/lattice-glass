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

- GPU: viable. One config per SM. ~1.9 h per config. 1000 configs need roughly 8 H100-class GPUs for 2 h.
- FPGA: not viable for thousands of configs at this rate. One config per engine is port-bound near 250M/s at 250 MHz, so it needs ~6 independent lanes per config. Shared-proposal SIMD couples configs and violates independence.
- CPU: one config is 50+ h on one core. It cannot meet 2 h without within-config parallelization.

## Exact local dependency

Local move: random site a, random neighbor b, reject equal types, Metropolis swap.

Affected set: endpoints plus their neighbors. For adjacent a,b on a cubic lattice this is 12 distinct sites. Two moves commute exactly iff all endpoint pairs have torus Manhattan distance >= 3.

The CPU sequential kernel is the reference. The parallel colored kernel has the same stationary distribution but a different trajectory and a different integrated autocorrelation time. If dynamics or tau are measured, this is a different model. If only equilibrium observables are needed, it is valid.

## GPU design

- One config per CTA/SM. 256 threads, not one warp.
- State: one byte per site, 9.26 KB per config. 1000 configs = 9.3 MB total.
- Load config state into shared memory. Do not stream from HBM.
- Build conflict-free batches of local swaps. Exact bound: at most 771 moves per batch; practical packing gives about 13 batches per sweep.
- Round barrier after each batch. Distribution exact; trajectory differs from sequential CPU.
- Per-config counter-based RNG. No shared proposal stream. No common random numbers.
- No per-proposal atomics; batch writes are disjoint.
- Per-config rate target: 1.286e9 proposals/s. One H100 SM is near 1.9 h for one config.
- 1000 configs at 132 SMs/GPU = 7.6 waves. One GPU ~14.5 h. Eight GPUs ~1.8 h.

Estimated campaign: 8 H100-class GPUs, about 2 h, thousands of configs only if 1000 at a time. 3000 configs need about 3 waves of 8 GPUs, ~5.4 h, or 24 GPUs for 2 h.

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
