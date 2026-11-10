# Solid backend contract

This file freezes the CPU reference and the Metal fast kernel. A change to any item
below changes the sampled chain. Re-run `make test` and `make test-long` after a change.

## State
- One byte per site.
- Bits 0..1: type. 0 is vacant, 1 is type 1, 2 is type 2.
- Bits 2..7: bit `d` is the count of occupied neighbors in direction `d`.
- Six directions: +x, -x, +y, -y, +z, -z on a periodic cubic lattice.

## Energy
- Type 1 target coordination is 3. Type 2 target coordination is 5.
- Site energy is 0 for a vacant site. Otherwise it is `(count - target)^2`.
- Total energy is the sum over all sites.
- `energy.npy` stores total energy after every sweep. It is not incremental.

## RNG and index sampling
- Counter-based SplitMix64 word stream keyed by `(seed, chain, proposal index, mode)`.
- Two consecutive u32 words come from the high and low halves of one 64-bit mix.
- Proposal index is an explicit integer. It is not a sequential stream state.
- Nonlocal sweep `s` uses indices `(2*s)*N ... (2*s)*N + N - 1`.
- Local sweep `s` uses indices `(2*s+1)*N ... (2*s+1)*N + N - 1`.
- Mode `both` runs nonlocal then local for the same sweep number.
- Site, partner, and direction use exact Lemire rejection:
  `i = mulhi(R, n)`, reject while `low32(R*n) < (2^32 mod n)`.
- No floating-point index conversion is permitted.

## Proposals
- Nonlocal: draw `a` and `b` by exact Lemire. Reject when `a == b` or when torus
L1 distance is less than 3. The two affected neighborhoods are then independent.
- Local: draw `a` by exact Lemire, direction from exact Lemire with `n = 6`, and
swap `a` with that neighbor.
- A proposal with equal endpoint types is a no-op. It consumes its index and
changes no state.

## EFMA acceptance
- Compute the exact integer energy delta over the deduplicated affected set.
- Accept immediately when `dE <= 0`.
- For `dE > 0`, use `T(dE) = floor(2^64 * exp(-beta*dE))`, split into
`T_hi * 2^32 + T_lo`.
- Evaluate the table once per beta with at least 128-bit precision. Parse beta
from its decimal text. Do not round beta through fp64.
- Draw `R_hi`. Accept if `R_hi < T_hi`; reject if `R_hi > T_hi`.
- On the tie `R_hi == T_hi`, draw `R_lo` lazily and accept iff `R_lo < T_lo`.
- No `exp`, float, or int-to-float conversion is allowed in the accept step.
- The realized acceptance differs from exact `min(1, exp(-beta*dE))` by at most
`2^-64` per uphill proposal.

## Parallel cut
- The fast kernel proposes up to 32 updates per round.
- Commit the longest prefix where accepted proposals are pairwise disjoint.
- The cut also stops before a valid proposal whose endpoint is within torus L1
distance 2 of an earlier accepted endpoint in the same round.
- A valid no-op also stops the cut. The no-op does not change state.
- Rejected and invalid proposals do not stop the cut.
- After a cut, unchanged suffix proposals are drawn again from the same indices.

## Scheduling and output
- A segment runs one Metal dispatch. There is no host synchronization per sweep.
- Snapshots occur at planned sweep boundaries. Planned boundaries are powers of
two through the requested sweep count.
- The driver copies state after a segment, then the host writes snapshots.
- CPU recomputes total energy after every sweep.
- GPU accumulates accepted deltas with a warp reduction after every round, then
writes total energy after every sweep. Initial GPU energy is a warp reduction.

## Verification
- `make test`: short CPU/GPU bitwise cases for L4 and L21, all modes.
- `make test-long`: adds L21, 1000 sweeps, 8 chains, nonlocal and local.
- Compared fields: initial, equilibrated, final, snapshot sweeps, snapshots,
energy, local acceptance, nonlocal acceptance, scalars, and metadata.
