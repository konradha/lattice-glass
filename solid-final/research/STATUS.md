# solid-opt3 follow-up status — 2026-11-11T11:18:41.443063+00:00

## Closed paid phase
- All five requested follow-up slices executed/evaluated. No new throughput winner. No pushes.
- Conservative cumulative commitment USD14.845438183698; follow-up USD0.849390955694, including the failed timing call's full USD0.50032836 liability. Not an invoice; no speculative refunds.
- Initial follow-up USD0.90 admission allowance was transparently raised to USD0.96 solely to fit the final proper shutdown envelope. Actual closed-call commitment remains below USD0.90; original USD15 hard cap never raised.
- All three own Apps verified STOPPED/empty tasks. Persistent paid-work closure now rejects image/CPU/GPU admission before materialization, even if a caller supplies a positive budget. No further paid calls under this authorization.

## Best measured rates remain unchanged
- L40S C07 single-chain local169.672M/nonlocal207.956M attempts/s. At142chains:142.069M/197.339M per chain. Historical C02 card frontier:40.839G/48.379G at1136chains.
- Actual C07 W256 profile at1/142chains:27–30%issue-active,20–25%barrier stall, zero local sectors; stall statistics are not wall-time/Amdahl fractions.
- C14 sparse+lazy-incidence engine passed selected normal1M/full100ktrace/multi/saturation, all three sanitizers both moves, and fresh100M both moves at W32/T32,W32/T256,W256/T256. It is slower:137.650M/165.648M single,116.167M/158.301M at142chains. No global/all-shape/H100 promotion.
- Lookahead comparison at beta.01/1.5/3.2 favored W256/T256 over W32/T32 for each tested physical move. No adaptive-window policy kept. Work/aging differs across beta groups; no equilibrium or causal beta-only speedup claim.
- H100 not purchased: no demonstrated per-chain gain/resource payoff justified its cost after the sparse regression. Static sm90 compilation is not a runtime measurement.

## Actual bounded production/recovery proof
- Receipt801576c74a0d452289c28cc178d8fdc7: ten betas linspace(.01,2,10),20cells,6chains each/120total,30s requested with native120/worker180bounds and4096upperchunks.
- All20cells terminaldeadline; native wall27.603194546s. All80saved actual probes replayed exactly against frozen CPU. Terminal raw bytes/cursor/seed/energy/both acceptance counts match final probes and observations.
- Fixed checkpoint selection that chose immutable993 over completed1077. New completed-native selection requires terminal markers and actual final-probe identity/hash; incomplete/corrupt data retain immutable fallback. All12regression tests pass.
- Corrected resume-plan-terminal contains all20actual terminal starts; pure planning only, no GPU-resume execution. Default16384-chain-per-beta/move current-source production plan also passed locally, not executed at that size.
- This is not the original600s campaign, full-trajectory replay or equilibration. Original failed600s result remains explicit in REPORT.

## Retained evidence
REPORT.md; NOTES.md; results.tsv; proofs/C07_winner_profiling.md; proofs/C13.md andC14.md; proofs/CAMPAIGN_SCHEDULING.md; artifacts/followup-*-summary.json; artifacts/followup-shutdown-pilot/verification.json andterminal-checkpoint-proof.json; actual receipts; source-bound production recipes.

## Next funded priorities, not running
1. Reduce actual C07 instruction/dependency cost; profile its remaining non-barrier stalls rather than infer unused issue as free speedup.
2. Examine exact one-warp/hardware-reduction selection and ILP before more sparse metadata. Any gain needs new gates/timing, not static SASS inference.
3. Measure host/publication/chunk amortization end-to-end before another full600s mixed-beta run. Never shrink690/900bounds or change physical-local dynamics to colored updates.
