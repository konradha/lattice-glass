# Solid opt3 baseline exactness argument

This document precedes the baseline implementation. It describes the inherited
CPU-random-neighbor CUDA algorithm, not a performance result or any of O01–O19.
The frozen opt2 CPU headers are authoritative. Gates and sanitizers must still
exercise the actual compiled binary; these arguments do not replace them.

## Cache and exact energy difference

A shared uint32 cell stores type in bits 0–1, occupied-neighbor count m in
bits 2–4, and signed S in bits 16–31. Define ell(1)=3, ell(2)=5,
q_i=occupied(i)*(m_i-ell(type_i)), S_i=sum over neighbors j of q_j.
Each occupied site's energy is q_i squared; a vacant site's energy is zero.
Counts are 0–6, |q|<=5 and |S|<=30, so these fields have ample exact range.

Unlike occupied endpoint types swap without changing occupancy. Their exact
delta is 2*(ell_b-ell_a)*(m_b-m_a). For a particle/hole exchange, orient a as
the particle, b as the hole, and let ell be the particle target. For separation
above two the delta is

    (m_b-ell)^2-(m_a-ell)^2-2*S_a+m_a+2*S_b+m_b.

For adjacent endpoints (including a nonlocal draw that happens to be adjacent),
the arrival count excludes the departing particle, and its neighbor sum excludes
the departing particle's old q. The exact delta is

    (m_b-1-ell)^2-(m_a-ell)^2-2*S_a+m_a
      +2*(S_b-m_a+ell)+m_b-1.

At distance two, every occupied common neighbor experiences zero net occupancy
change. The separated formula erroneously assigns it the sum of two quadratic
changes, two too large. Subtract two once per distinct occupied common neighbor.
For L>=4 each six-neighbor list has distinct sites, including the periodic L4
case. No adjacent/distance-two move is filtered, redrawn, or approximated.

## Commit and physical ownership

An accepted exchange changes types at its endpoints and counts only in their
neighbor union. Deduplicate those at most fourteen sites. Each unique owner
computes its exact low-field replacement and dq from old type/count, adds the
low-field difference, then atomically scatters dq*65536 to the six adjacent S
fields. Low fields before/after remain 0–31, so the modular low-field addition
cannot carry or borrow into S. Signed S arithmetic is represented modulo 2^32.
Full-word atomic reads/adds are mandatory: a low-field owner may share a physical
uint32 with another owner's S scatter. No non-atomic access overlaps this phase.
Scatters commute. A barrier completes each accepted commit before the next;
accepted moves are committed in increasing proposal order, not concurrently.

## Dependency, resurrection, and serial prefix

Every round speculates on one immutable cache state. A committed endpoint
changes endpoint types at radius zero, counts at radius one and S at radius two.
A no-op reads endpoint types, so uses radius zero. Unlike occupied swaps read
types/counts, so use radius one. Particle/hole proposals read S, so use radius
two; their distance-two common-neighbor type reads lie within that bound too.
Invalid a==b is state-independent and needs no dependency. Rejected proposals
are readers exactly like accepted proposals. A no-op can become a move and a
rejection can become accepted; neither may be omitted from invalidation.

Cut before the first valid proposal whose read radius intersects an earlier
speculatively accepted endpoint. Every proposal before the cut has the same
read values and outcome as serial CPU execution by induction. The cut is at
least one. Commit accepted members of that prefix in serial order. Regenerate
the suffix from unchanged full-width proposal keys on the updated state. This
terminates and exactly preserves all outcomes, including resurrection. It does
not require all speculatively accepted suffix proposals to remain accepted.

## Synchronization and RNG

All physical threads, including inactive lanes for W1 and tail rounds, execute
all full-mask ballots/shuffles and barriers. Blocks of 32 use explicit full-mask
syncwarp before reading published metadata and after every commit, including
empty-accept rounds. Multiwarp blocks use syncthreads at those same boundaries.
There are no speculative reads while commits execute. Independent CTAs own
disjoint chain state. Launch supports W1/32/64/128/256, T32/64/128/256, W<=T,
and ppt1 only; unsupported configurations throw rather than pretending support.

SplitMix counter words, high-then-low order, Lemire rejection, proposal key bit
allocation, separate nonlocal/local index ranges and lazy low acceptance word
match opt2. Host decimal EFMA builds exact integer thresholds; the device uses
integer comparison only, with no exp/log/pow. Sweep/start/seed/chain IDs retain
uint64 arithmetic (including the CPU's modular key-index arithmetic). The
low-level `CudaBackend::run` accepts starts whose requested sweeps cross
UINT64_MAX: each `start+s` and derived proposal index wraps modulo 2^64, exactly
as in the frozen CPU reference. Splitting such a run into checkpoint calls
must preserve that same modular trajectory. The production driver's separate
nonwrapping progress checks remain a checkpoint-bookkeeping restriction, not
a restriction on the low-level backend or its boundary correctness gates.
Inherited beta-zero threshold wrap is not changed or claimed validated.

## Streams, bounded production, and observations

Each backend owns one nonblocking stream. All transfers, kernels, and events use
that stream; run waits only for it. Device buffers grow only when reserve needs
a larger bound, and the batch driver reserves every backend before launching
host workers. Every worker owns a backend and bounded segment result. Rolling
observations retain bounded histories/snapshots; checkpoints retain two raw
slots. Only bounded real small continuation probes retain incoming/outgoing
bytes and every-sweep integer measurements for independent CPU replay. Ordinary
production gaps are not claimed replay-verified. Atomic file replacement
publishes complete observations, probes, and checkpoint/manifest files.

Profiling is a separate template instantiation. clock64 intervals are CTA elapsed
stage intervals, including the barriers used to delimit them, not instruction
counts or per-thread summed work. Round covers a complete replay/prefix round;
component intervals need not sum to Round. measured=false means stage counters
are unavailable, not measured zero. SM ID is read from the actual executing CTA.
CUDA event times exclude upload/download and remain separate from production
wall envelopes: concurrent event durations overlap and must not be summed into
wall throughput. Profile runs are never ordinary-rate measurements.

## Native production and replay conventions

`--jobs-tsv FILE --seconds SECONDS --output-dir DIR` uses the exact twelve-column
shared-contract header. Additional positive retention controls are:

* `--max-history-samples` defaults to 4096 (minimum 2).
* `--max-snapshots` defaults to 128 (minimum 64).
* `--probe-every` defaults to 64 production chunks.
* `--probe-sweeps` defaults to 4 (maximum 4096).
* `--max-replay-chunks` defaults to 32 rotating periodic probe slots.
* `--observations-every` defaults to 16 production chunks between publications.

The obsolete `--max-chunks` stop flag is removed, not silently ignored.
Reaching observation/probe retention bounds never stops production. Chunk
sweeps remain 1–4096. The common deadline is checked before each ordinary chunk;
a running chunk finishes and a final small continuation probe is published, so
the wall envelope can exceed requested time by that work/publication. Choose
short measured chunks. All engines/device reservations precede worker creation.

`DIR/id/observations.npz` is atomically replaced, not accumulated. Its incoming
bytes are the actual campaign initial bytes, its final bytes/current next sweep
are actual, and `meta` sweeps cover the entire campaign including probes.
Integer fields `energy_int`, `accept_local_count`, and `accept_nonlocal_count`
retain exact measurements at the reported absolute history timestamps. The
initial absolute cadence comes from the TSV. When its bounded history fills,
double the cadence and retain only samples aligned to the new cadence. Continue
sampling for the full duration; always publish the actual final measurement.
The reported `history_every` is the current cadence, not an invented dense
history or an average over discarded sweeps.

Snapshots are actual dispatch-boundary bytes. Dispatches are split to land
exactly on relative-to-campaign-origin targets `n*h` (n=1..10) and `(10*m+r)*h`
(m in {1,2,3,10,20,30}, r=1..4), for powers of ten h. At h>=10, waits
10h/20h/30h versus h/2h/3h have eight exact common positive lags:
h/10,2h/10,3h/10,4h/10,h,2h,3h,4h. The fractional-h targets come from the
next finer decimal grid, with integer sweep boundaries; no interpolation is used.
The fixed primary group is the largest scale h=10^k>=10 whose three waiting
pairs and all eight lag targets remain in saved snapshots, with 34h<=elapsed.
Select it by timestamps only, never observed overlap values; older groups are
descriptive. Chronologically latest bounded snapshots plus original incoming
state are retained, with actual final state included at publication. Retention
and elapsed time may leave no complete primary group; report that limitation,
not a substituted earlier-aging result. The schedule alone does not prove relaxation.
All stored batch snapshot/history timestamps are absolute; subtract start_sweep
when interpreting the campaign-relative observation schedule.

`probe-first.npz`, `probe-last.npz`, and rotating `probe-000000.npz` slots retain
small actual continuations of the same backend/chain/index, never independently
regenerated initial states or shadow chains. They store actual incoming/outgoing
bytes, all per-sweep integer measurements, seed, full-width start/next, launch
and model metadata. Probe sweeps contribute to campaign progress and observations.
Sort probe files by stored start, not slot name. Independently replay every
retained probe from its own actual incoming bytes. Only directly adjacent
retained intervals support continuity assertions; gaps between probes remain
explicitly unreplayed, and sparse snapshots are state evidence, not replay
coverage. Mandatory CUDA correctness gate counts remain unchanged.

Standalone compatibility retains production-relative history/snapshot times;
`history_sweeps_absolute` is provided in both paths. The standalone output
remains `run.npz` in its working directory. Its history cap defaults to 100000;
power-of-two snapshots plus final need at most 65 states for any uint64 duration.

`manifest.tsv` lists progress, statuses, and checkpoint TSV paths. A checkpoint
has exact decimal seed, beta text, model/launch, next sweep, and `state`, naming
one of two bounded alternating raw slots. Publish the inactive raw slot before
atomically replacing the TSV, preserving the old exact checkpoint during an
interrupted write. The TSV's state slot is authoritative. Resume supplies it
and next sweep in a new jobs TSV and a fresh output directory. This is atomic
publication against process interruption, not fsync/power-loss durability.

`run.tsv` records native source digest/build flags and actual Linux executable
path, size, mtime, and compiler. It does not claim those attributes are a binary
hash; the external controller must record SHA256 of the actual binary. Kernel
event seconds and wall seconds have distinct labels. Profile-mode observations
are marked and are not normal throughput. The RNG stage measures proposal
index words; lazy acceptance words remain in LoadDelta, whose elapsed interval
also includes publication/barrier overhead.

## Isolated execution-model compositions

Candidate-specific arguments are written before each isolated candidate source. Their status is evidence-scoped; source review/compilation is not a GPU correctness or performance gate.

- C03 depth composition: `proofs/C03_fast_composite.md`.
- C04 wide selective repair: `proofs/C04_wide_selective_repair.md`.
- C05 fast wide selective repair: `proofs/C05_fast_wide_repair.md`.
- C06 fast virtual-proposal lookahead: `proofs/C06_fast_ilp.md`.
- C07 no-accepted-writer shortcut: `proofs/C07_fast_empty_window.md`; selected full byte/trace/multi/saturation plus all3 L40S sanitizers precede its measured rates.
- C08 packed fast no-writer composition: `proofs/C08_packed_fast_empty.md`; actual A100 selected W256/T256/PPT1 normal/trace/multichain/all3-sanitizer and held-out100M gates passed before timing, receipt0e81177be78643e1a82985e7e60dc8d4. Untested shapes/profile/cards are not covered.
- C09 exact modular key strength reduction: `proofs/C09_shared_key_product.md`; shared product is CTA-owned and published before evaluation, updated before the existing end join. RNG words are unchanged by ring distributivity; no GPU result inferred.
- Grouped-load compiler fusion is an observed negative diagnostic in `proofs/H02_grouped_loads.md`, not a Monte Carlo speedup.
- C10 direct field-class minima: `proofs/C10_direct_field_claims.md`; source/dual-architecture compilation only, no GPU correctness or performance claim.
- C11 sparse ordered repair: `proofs/C11_sparse_ordered_repair.md`; the pre-code invariant and synchronization argument replaces global prefix cuts with endpoint incidence and only accepted/dirty work.
- Register-cap pruning: `proofs/O18_register_caps.md`; any normal L21 spill rejects a candidate before GPU qualification.
- Fixed-S production and recovery: `proofs/PRODUCER.md` and `proofs/PRODUCTION_PLAN.md`; actual CPU/GPU byte/recovery evidence is separately recorded in NOTES.
- C12 packed sparse repair: `proofs/C12_packed_sparse_repair.md`; reuse C08's bounded16bit logical cells/full-word atomic updates and C11's serial-outcome induction. Two uint16 next links share one word owned by their original proposal's single generating thread; all traversal follows the publication fence. The approximately114KB CTA/two-resident-Hopper-chain benefit is only a resource hypothesis until actual qualification.

## Complete CUDA correctness ladder from a fresh checkout

From the repository root, first choose a genuinely unused final seed and chain
and export them as `FINAL_SEED` and `FINAL_CHAIN`. Do not copy an earlier run's
values: the builder rejects the frozen known development/historical seed list,
but cannot establish global freshness for caller-selected counters. The seed
must be a decimal uint64 and the chain must be in `[1000000, 2147483647]`.
One command prepares both fixtures remotely and runs the complete existing CUDA
correctness ladder, for the baseline's full supported launch matrix:

```sh
python3 solid-opt3/gate/check_all.py --all \
  --budget-usd 15 --ledger solid-opt3/artifacts/budget.json \
  --gpu A100-40GB --backend cuda/baseline.cu \
  --extra-nvccflags="-gencode arch=compute_80,code=sm_80" \
  --sanitizer-timeout 120 --racecheck-timeout 900 \
  --prepare-final final.bin \
  --final-seed "${FINAL_SEED:?Choose a genuinely fresh decimal uint64}" \
  --final-chain "${FINAL_CHAIN:?Choose a genuinely fresh chain >=1000000}"
```

Prerequisites are Python 3, the external Modal Python SDK/`modal` CLI on PATH
(or `--modal /path/to/modal`, alternatively `MODAL_BIN`), authenticated Modal
credentials with permission to create/use this app's image, Volumes and Dict,
network access to Modal/the pinned image/package repositories, and access to
the selected GPU. No local CUDA installation, local GPU, prebuilt executable,
downloaded fixture or prior gate receipt is needed for `--all`. The existing
Modal image supplies CUDA 12.8.1, `nvcc`, `compute-sanitizer`, GNU C++/make,
Boost 1.83 headers and NumPy 2.2.6. The CPU worker builds the four native
executables before any GPU allocation. Nsight tools and the separate full-H
audit executable are not prerequisites for this command.

The command writes one exclusive-create packet and invokes Modal once. Its CPU
prefix runs the unchanged physics audit, prepares `development.bin`, and
prepares the distinct fresh `final.bin`. Its GPU sequence is device identity,
development smoke, profile-smoke, memcheck/racecheck/synccheck for both moves,
single-chain (including existing proposal-trace and counter-boundary checks),
multichain and saturation, followed by held-out smoke, all three sanitizers for
both moves, and final. All commands retain their existing finite bounds and
stop the packet on failure. The CPU preflight models future gate ordering;
the GPU admits final only after the real successful earlier command rows,
including the CPU audit and development single/multi/saturation/sanitizers.
No synthetic receipt authorizes final. Combined packets retain `stage=proof`
to admit the existing profile-smoke exception, while invoking Modal with
`--final` to use the existing final-budget admission.

The defaults cover both local/nonlocal laws and the fourteen Cartesian shapes
with W in `{1,32,64,128,256}`, T in `{32,64,128,256}`, PPT=1 and W<=T.
The native single gate compares the existing first 100000 proposal records per
chain (including proposal fields, delta and accept/no-op flags), checkpoint
bytes, and every-sweep integer energies/acceptances; it is not a new separate
`fulltrace` stage or an all-proposals trace of each long trajectory. Final uses
the unchanged L21/beta3.2 held-out fixtures with at least 100 million proposals
per move and the existing trace/checkpoint coverage. Narrowing
`--windows/--threads/--ppt/--move` narrows the claim. Selecting another backend
does not prove that it supports the baseline matrix; unsupported launches must
fail, and untested profile modes/cards/shapes remain unqualified.

The normalized `--extra-nvccflags` and all native input bytes determine
`expected_source_digest` at packet construction. Modal rejects subsequent
source changes, builds/cache-selects with those actual flags, and binds command
receipts to the source digest, actual executable hashes, GPU and launch/move
config. A100 requires explicit sm80 code as shown; changing backend, flags or
card requires matching new proofs, not reuse of a different native identity.

Racecheck has its own optional wall-time bound so the full matrix can retain
the longer observed race-analysis allowance without assigning900seconds to
every memcheck/synccheck command. All original commands, fixtures, seeds and
error/hazard rejection remain unchanged. Assigning900seconds to all twelve
sanitizer commands in combined mode exceeds Modal's7200second packet limit;
the builder rejects that envelope rather than reducing any check.

Development-only remains the default (optionally `--fixture LOCAL_FILE`);
final-only remains `--prepare-final FILE --final-seed ... --final-chain ...`
without `--all`, with `--receipts` naming actual successful development
receipt JSON files. Final-only receipts must match the current source/flags
and GPU and cover the requested config; Modal independently verifies the
durable originals and native binary hashes. `--all` rejects fixture reuse and
requires explicit final counters and a final basename distinct from
`development.bin`. `--packet-only` only constructs the real packet and reports
its command; it performs no build, fixture preparation or correctness check.

An explicit finite positive budget no greater than USD15 is mandatory. The
ledger is this worktree's existing `solid-opt3/artifacts/budget.json`; another
filename does not create a new allowance. Existing remaining-balance,
development/final reserve, USD14 operational ceiling, fixed campaign window
ending 2026-11-11T09:02:41Z, acquisition/startup and command/packet timeout guards
remain authoritative. A fresh checkout is not a fresh campaign allowance;
expired/exhausted admission must fail, not reset the ledger or retry. This
command neither runs benchmarks nor launches production, and constructing it
is not evidence that any GPU ladder or campaign succeeded.

The independent **million-proposal per-proposal full-H audit is not invoked**.
Its executable/build, external raw-state provenance (where used), commands and
exact scope remain separately documented in `gate/FULL_H_AUDIT.md`. CPU fixture
energy reconstruction and CUDA proposal traces above must not be relabeled as
that separate audit.

