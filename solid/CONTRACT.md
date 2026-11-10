# Solid backend contract

This file defines the CPU reference, Metal fast kernel, and output contract.
History sampling must not change the sampled chain. Re-run `make test` and
`make test-long` after a change.

## State
- One byte per site.
- Bits 0..1: type. 0 is vacant, 1 is type 1, 2 is type 2.
- Bits 2..7: bit `d` is the count of occupied neighbors in direction `d`.
- Six directions: +x, -x, +y, -y, +z, -z on a periodic cubic lattice.

## Energy
- Type 1 target coordination is 3. Type 2 target coordination is 5.
- Site energy is 0 for a vacant site. Otherwise it is `(count - target)^2`.
- Total energy is the sum over all sites.
- `energy.npy` stores total energy at the production sweeps in `history_sweeps.npy`.

## RNG and index sampling
- Counter-based SplitMix64 word stream keyed by `(seed, chain, proposal index, mode)`.
- Two consecutive u32 words come from the high and low halves of one 64-bit mix.
- Proposal index is an explicit integer. It is not a sequential stream state.
- Nonlocal sweep `s` uses indices `(2*s)*N ... (2*s)*N + N - 1`.
- Local sweep `s` uses indices `(2*s+1)*N ... (2*s+1)*N + N - 1`.
- Mode `both` runs nonlocal then local for the same sweep number.
- Driver `--seed` defaults to 1 and controls both initialization and proposals.
  Use distinct seeds for independent batches. The exact uint64 seed is saved.
- Site, partner, and direction use exact Lemire rejection:
  `i = mulhi(R, n)`, reject while `low32(R*n) < (2^32 mod n)`.
- No floating-point index conversion is permitted.

## Proposals
- Nonlocal: draw any `a != b` by exact Lemire. Do not reject neighbor or
distance-2 pairs. The affected set must be deduplicated before the energy delta.
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
- Execution uses at most 4096 sweeps per Metal dispatch, including equilibration.
  There is no host synchronization per sweep.
- Snapshots occur at planned sweep boundaries. Planned boundaries are powers of
two through the requested sweep count.
- The driver copies state after a segment, then the host writes snapshots.
- CPU recomputes total energy after every sweep.
- GPU accumulates accepted deltas with a warp reduction after every round, then
writes total energy after every sweep. Initial GPU energy is a warp reduction.
- `--history-every` defaults to 1; `--max-history-samples` defaults to 100000.
  Both must be positive. The effective interval is
  `max(history_every, ceil(production_sweeps / max_history_samples))`.
- Store energy and both acceptance fractions at positive interval multiples and
  at the final production sweep. Zero production sweeps produces empty histories.
- History arrays have shape `(chains, samples)`. `history_sweeps` contains the
  one-based production-relative timestamps. `history_every` and `seed` each have
  shape `(1,)`, dtype uint64. The existing five-element `meta` is unchanged.
- Equilibration histories are discarded; GPU temporary histories are bounded by
  the dispatch chunk size. Sampling and chunk boundaries do not change RNG indices.

## Verification
- `make test`: short CPU/GPU bitwise cases for L4 and L21, all modes.
- `make test-long`: adds L21, 1000 sweeps, 8 chains, nonlocal and local.
- `make test-production`: CPU-only seed replay and history-sampling regressions.
- Compared fields: initial, equilibrated, final, snapshot sweeps, snapshots,
energy, local acceptance, nonlocal acceptance, scalars, metadata, seed, history
interval, and history sweep timestamps.

## CUDA exactness argument
- The CPU implementation on clean-solid is authoritative, including its fixed
  per-proposal SplitMix64/Lemire word stream and integer EFMA table. CUDA must
  not change proposal selection, no-op accounting, or acceptance thresholds.
- Cache each site's type, occupied-neighbor count `m`, and
  `S = sum_j occupied(j) * (m_j - ell_j)` over its six neighbors. Unlike-type
  particle swaps change only the two endpoint site energies. For particle-hole
  moves separated by more than two, the neighbor energy changes sum to
  `-2*S_a + m_a + 2*S_b + m_b`. For adjacent endpoints (local or nonlocal), exclude
  the departing particle from the arrival neighborhood, giving
  `(m_b-1-ell)^2-(m_a-ell)^2-2*S_a+m_a+2*(S_b-m_a+ell)+m_b-1`.
- At distance two, subtract two for each occupied common neighbor from the
  separated-endpoint formula: its occupancy change is zero, whereas the two
  independently evaluated quadratic changes incorrectly contribute two.
  Count distinct common neighbors explicitly, including periodic L4 cases.
- On acceptance, update type/count in the deduplicated endpoint neighborhoods.
  For each changed `q_j = occupied(j)*(m_j-ell_j)`, add its exact integer
  difference to each neighboring `S`. Cooperative shared-memory atomic adds
  commute; all readers wait until the update is complete.
- A speculative proposal reads cached endpoint values. An accepted predecessor
  can change these only within distance two of an endpoint. Cut before the first
  such dependency, including rejected and no-op proposals. Accepted proposals
  before that cut have their serial outcomes. Commit them in order, retain the
  suffix's proposal indices, and synchronize before the next evaluation.
- Separate local and nonlocal template instantiations may use different window
  sizes and launch occupancy; none of these choices may alter the serial chain.
- Ordered commit may iterate set bits of the acceptance ballots instead of every
  proposal: ascending warp number and ascending bit number preserve serial order.
- A one-warp block may replace block barriers with full-mask `__syncwarp`, since
  every physical lane participates; larger blocks retain `__syncthreads`.
- Dependency radius can follow the fields actually read: a no-op reads endpoint
  types (radius zero); unlike particles read types/counts (radius one); a
  particle-hole move reads `S` (radius two). A predecessor changing the endpoint
  types is caught even if it changes which of these cases applies on replay.
