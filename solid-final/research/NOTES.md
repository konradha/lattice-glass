# solid-opt3 durable operating memory

## Window, authority, and current state

Campaign: `solid-opt3-20261110T190241Z`, integration worktree
`/home/fritz/code/lattice-glass-solid-opt3`, intended branch `opt/solid-opt3`,
upstream `eaef7d4`. UTC window **2026-11-10 19:02:41 through 2026-11-11
09:02:41**. These are fourteen genuine optimization hours, not a sleeping scheduler
or an instruction to burn remaining budget. Parent alone builds, tests, runs
Modal, and performs git operations. No builds, tests, installations, Modal
launches, commits, or measurements were performed by the tool-authoring worker.
At this handoff, its new campaign compute spend is **USD0**; the ledger is
created only by the first real admission. Never copy an old ledger here.

Frozen reference: latest opt2 **CPU random-neighbor trajectory for both moves**,
not original ROOT/Philox and absolutely not opt2's 21-colour Metal-local schedule.
Counter SplitMix64, exact Lemire draws and exact host decimal64 EFMA are frozen.
Nonlocal draws two independent sites; equal sites are invalid but counted, equal
types are no-ops. Local draws a uniform site and independent uniform direction6.
No distance exclusion/redraw, schedule change, device exp/log/pow, or fast math.
L21, rho=.75, rho1=.30, rho2=.45, ell1=3, ell2=5, A=1. Both-mode means NL then
local, not production's two separate ladders. All seeds and sweep counters are
full uint64. CPU beta0 threshold wrap is inherited and documented, not repaired
to rescue a gate; production begins at .01.

Latest opt2 campaign summary used L21, eight independent chains/beta,
linspace(.01,2,8), separate local/nonlocal. Its nonlocal comparison was bitwise;
its local Metal comparison used a different coloured schedule and was
statistical. Its old CPU149.6s/Metal145.6s are **not** opt3 CUDA measurements.
Strong useful learning: unlike-particle/type-only commits deserve investigation.
Rejected/no-op proposals can resurrect; every such dependency must invalidate.
Logical halfword separation does not establish physical 32-bit write ownership.

`experiments.json` enumerates O01..O19 with conditional predictions and proof
obligations. Except supported baseline launch shapes, proposed variant paths are
**intended, absent/unimplemented candidates**, not nineteen implemented kernels.
Every measurement, gain, hash and check starts empty/unrun. Establish real
baseline/gates and stage fractions before choosing the next implementation.
No performance or equilibrium claim is made by these tools.

## Parent-observed development evidence

The first actual receipt, run `d8dbfebfef7548408636450ec936e0a9`
(native source prefix `ab01`), failed the newly added native uint64 overflow
guard; it contains **no valid performance timings**. Its retained budget was
USD.05 image +.00421473418227 CPU +1.12646016 GPU =
**USD1.18067489418** before the second call, not an invoice.
The corrected native build completed in26.77s; reported native build digest:
`46ecbcc16951e4385657297407f47122dee53f068d1c4dc8f94d2fa4765facb7`.

Independent CPU random-trajectory audits passed one million proposals per
separate move, checking full H, two-cell cache and unchanged CPU authority:
34.94s/34.46s. Evidence: `artifacts/full-h-random-summary.json`.
An actual beta3.2 input aged by58192 CPU nonlocal sweeps has SHA256
`3df82625de565d96d72fff37319b4782f528ef7ef2fe2061ec8fdd077992104a`;
aged audits were still in flight at this update. The corrected bounded Modal
packet **passed actual normal and profile smoke across all14 shapes**:
`artifacts/receipts/1332f18ad17949bfa3a951ba0467888f.json`,
app `ap-tgFVryMbTFE8kYWKJuo9hX`. Full sanitizers/single/multi gates remain
pending; this smoke success does not admit diagnostics or rates. Actual GPU
throughput remains **unmeasured**, and counter permissions remain unknown.
These are parent-reported observations, not checks performed by this worker.


## Fixed image, source identity, and CPU-before-GPU

Use the existing authenticated executable
`/home/fritz/code/lattice-glass/lattice-gpu/.venv/bin/modal` (SDK1.6.3), not a new
venv. Image is pinned CUDA12.8.1 Ubuntu24.04 digest
`4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43`, Python3.11,
build-essential, Boost1.83, numpy2.2.6. Source and fixtures use **copy=False**
mounts. Python/docs/artifacts do not change the native digest or toolchain image.

Native SHA256 covers selected relative `.cu`, `cuda/backend.hpp`,
`cuda/driver.cpp`, `gate/check.cpp`, the three CPU headers, `Makefile`, normalized
compiler flags and pinned toolchain identity, only. Native flags permit pinned
CXX=g++, NVCC=nvcc, BOOST_INC=/usr/include and explicit EXTRA_NVCCFLAGS; fast math
is refused. Any native edit/flag change requires new own-binary proofs.
`expected_source_digest`, when supplied, rejects edits after packet creation.

CPU function runs `make cpu`, then `make all` **without any GPU allocation**,
with -j2, and stores all four binaries plus source/toolchain/binary hashes in
Volume `lattice-cuda-builds/TOOLCHAIN/DIGEST`. It checks executable hashes on
cache hits. CPU commands must form a packet prefix; they finish and commit
before GPU spawn. `cpu_check --prepare @run:dev.bin` followed by GPU
`cuda_check --check @run:dev.bin ...` is supported. Native build logs and command
results commit incrementally. No make/compiler command exists in GPU execution.
A source/backend requiring additional native helper files must first extend the
explicit shared digest/copy contract; silently hashing all Python is forbidden.

## Exact invocation and packet interface

Every compute invocation requires an explicit finite positive total campaign
budget <=15, GPU choice, packet and this ledger:

```
modal run solid-opt3/modal_app.py --budget-usd 11 --packet PACKET.json \
  --gpu L40S --ledger solid-opt3/artifacts/budget.json
```

Add `--final` for held-out final gates or final production. `--budget-usd` is the
cumulative authorized ceiling, not a separate allowance per command. Only the
canonical new ledger path is accepted, including when explicitly specified by
an absolute path. Old schemas/campaign IDs are refused. A lower ceiling is
allowed and respected. Omitted, NaN, infinity, zero, negative, and >15 are
refused **before defining/materializing the image**.

Packet fields:

```json
{
  "backend": "cuda/baseline.cu",
  "native_flags": {"EXTRA_NVCCFLAGS": ""},
  "stage": "proof",
  "config": {"windows": [32], "threads": [32], "ppt": [1],
             "moves": ["local", "nonlocal"], "profile": false},
  "mounts": [{"local": "/absolute/dev.bin", "name": "dev.bin"}],
  "commands": [{"binary": "cuda_check",
    "args": ["--check", "@fixture:dev.bin", "--stage", "smoke",
             "--windows", "32", "--threads", "32", "--ppt", "1",
             "--move", "local", "--profile", "0"], "timeout": 60}],
  "hard_timeout": 90,
  "requires": [],
  "proof_fixture_sha256": []
}
```

This single-command example is smoke only, not permission to benchmark.
Allowed stages: proof, single, multi, perf, scaling, final, pilot, campaign,
diagnostic. Commands have exactly binary/args/timeout; ordinary binaries are
cpu_check, cpu_driver, cuda_check, driver, compute-sanitizer. Diagnostic-only
hardware/inspector commands are specified below. No shell. Argument values are strings and
flags take separated values. Sanitizer invocation uses
`["--tool","racecheck","--error-exitcode","99","@binary:cuda_check", ...]`.
Warnings/hazards or absence of an explicit clean sanitizer summary fail even
when exit status is zero. An exit-zero native gate must emit JSONL proof.

Remapping is explicit, not string replacement of arbitrary local paths:

* `@fixture:NAME` -> read-only `/fixtures/NAME`, declared in mounts; local mount
  paths resolve relative to the packet file. All bytes are hashed and rechecked.
* `@run:REL` -> writable `/runs/RUN_ID/REL`, same durable directory across CPU
  and GPU phases. No `..` or absolute suffix. Output parents can be created by
  the native tool; do not overwrite another run's directory.
* `@binary:cuda_check` -> exact cached native executable (also other binary names).
* Bare absolute command arguments are refused; use these explicit remappings.

`hard_timeout` bounds the GPU function, integer30..7200. Sum of GPU command
bounds must leave at least20s shutdown/publication headroom. CPU bound defaults
to sum(prefix timeouts)+270s (240s build plus publication), with another180s for
optional hardware compilation. Optional cpu_hard_timeout must cover at least
prefix+250s (plus180s when hardware is requested) and be <=14400. CPU-only
packets still compile and account CPU work, but never request a GPU.
Prefer small development packets, not the maximum allowed deadline.

Production/pilot uses driver `--jobs-tsv @fixture:jobs.tsv --seconds N
--output-dir @run:production`. Driver jobs TSV schema is exactly
id,beta,mode,chains,seed,window,threads,ppt,history_every,chunk_sweeps,initial,
start_sweep (TAB separated). Its **contents** are not token-remapped: resume
initial paths are `/fixtures/JOB.raw`, explicitly declared and hashed mounts;
`-` requests CPU initialization. Actual TSV launch shapes/moves must be proved,
uint64 seed/start are checked, and no requested move may silently disappear.
Pilot is development-budget driver work with all development gates; campaign
requires final proofs, `--final`, and a real600s duration. Timeout must leave
at least30s beyond duration; e.g. timeout650, hard_timeout900. Scientific
selection/concurrency and actual checkpoint CPU replay belong to campaign.py.

## Proof/rate admission and durable results

Each successful command receipt hashes its complete canonical JSON, source
hash, all four actual executable hashes, requested GPU, exact config, argv,
stdout/stderr, exit status, actual elapsed, actual profile flag, gate class and
fixture digest. Native command environment also receives SOLID_SOURCE_HASH,
SOLID_BINARY_HASH, SOLID_BACKEND, SOLID_BUILD_FLAGS. Clock evidence is raw
`nvidia-smi` name/UUID/driver/current+max clocks/power CSV before and after.
Long benchmark/driver commands also collect actual `clock_samples` every5s
(bounded2s probe); probes may perturb timing. No interpolated/nominal clock is
presented as a sampled clock, and no clock/occupancy value is fabricated.

Continue a stage with
`requires:[{"run_id":"32hex","command":0,"receipt_sha256":"64hex"}]` and an
explicit `proof_fixture_sha256` list containing referenced fixture hashes.
CPU and GPU both load the durable receipts, recompute receipt hashes and require
successful gates with **exact source+binary+GPU identity**. A target may declare
windows/threads/ppt/moves subsets of a proved config; profile and all other
config keys/values must still match exactly. Gate receipt hashes bind their
original configs. No check/count/tolerance is reduced; gates cover their full
declared shape axes. Put auxiliary metadata outside config.
A native `--move both` gate is accepted only for config containing both separate
moves; it covers both plus native combined-mode cases. A normal proof packet's
extra profile1 smoke is classified `profile_smoke`, cannot serve as normal
proof, and needs preceding normal smoke. Actual profiling throughput needs its
own complete profile=true gate identity, never an ordinary rate claim.

Smoke must precede long gates. Rates/pilots require successful smoke, single,
multi, saturated for every requested move, CPU audit, and memcheck/racecheck/
synccheck for both. Final checks additionally require this complete development
proof set; final production additionally requires held-out final proofs.
Failed/missing/tampered/mismatched proof stops admission. First failed command
stops the packet before later gates or rates. CPU preflight validates ordering
and prerequisites before GPU allocation; only actual GPU success receipts count.

Volume `lattice-solid-opt3-runs` stores RUN_ID/packet.json,
RUN_ID/cpu-result.json, RUN_ID/command-NNN.json, RUN_ID/result.json, production
outputs. Every command result and intermediate result is committed as it
finishes, including nonzero/timeout output. Native child process groups are
killed on command timeout. Per-command deadlines leave publication headroom;
a provider/container kill can still prevent publication of the **in-flight**
command, but already committed receipts survive.
Long driver commands also publish Volume snapshots while native work runs:
first after2s, then every10s. Each stable native checkpoint.tsv/raw pair is
copied with before/after metadata equality into immutable
`production/durable-checkpoints/JOB/generation-*/`; latest.json points to it.
Two generations/job are retained. An8s-bounded commit and its actual status are
recorded in driver-publication.json/command checkpoint_publications. Publication
failure stops native work and fails the command; no silent600s commit gap.
Cancellation may lose the current in-flight copy, not a prior committed usable
raw/metadata generation. Use these generations for recovery if native rolling
files or their last manifest were interrupted.
Local receipt wrapper is `solid-opt3/artifacts/receipts/RUN_ID.json`; partial result JSON is downloaded
under the sibling RUN_ID directory on success or client failure. Download data:

```
modal volume get lattice-solid-opt3-runs RUN_ID LOCAL_DIRECTORY
```

Result fields: run_id, ok, stage, backend, source_digest, binary_sha256 (map),
gpu, config, rows, errors, clocks_before/after (command objects; stdout is CSV),
proof_receipts, elapsed_seconds and cost. Native benchmark JSONL is preserved
in each row's stdout. Binary identity for benchmark is cuda_check, production
is driver; never substitute one for the other.

## Optional diagnostic packets (not ordinary MC throughput)

These use the existing app, fixed image, ledger and one-GPU-call lifecycle.
They are optional, after actual successful development proofs; they do not
delay baseline gates. The pinned image already supplied
`/usr/local/cuda/bin/ncu`, version2025.1.1.0 build35528883, in real CPU build
receipt `d8dbfebfef7548408636450ec936e0a9`. Counter permission is **unknown**.
No profiler install, image change, new physics digest or MC rebuild is needed.

Generate an unexecuted packet, then run it through the usual budget admission:

```
python solid-opt3/orchestrate.py packet --stage diagnostic --diagnostic-tool both \
  --backend cuda/baseline.cu --gpu L40S --windows 128 --threads 128 --ppt 1 \
  --move both --sweeps 1 --ncu-warmup 8192 --ncu-timeout 120 \
  --receipts ACTUAL_PROOF.json ACTUAL_SINGLE.json ACTUAL_MULTI.json \
  --output diagnostics.json
python solid-opt3/orchestrate.py run --packet diagnostics.json --budget-usd 11 \
  --gpu L40S --ledger solid-opt3/artifacts/budget.json
```

`--diagnostic-tool ncu|hardware|both` defaults to ncu. NCU uses W128 and exactly
chains1/8/64/284, one separate collection per requested move; threads128 or256,
ppt1, beta3.2 by default, four native fixed-work repeats. Only the selected
matching kernel launch is profiled. `--ncu-profile 0|1` defaults0; this is tool
instrumentation metadata, not the packet's normal proof config. Profile1 also
needs actual `profile_smoke` receipt(s). `config.profile` remains false and all
four MC binary hashes, native source, GPU, shape/move identity are unchanged.
Normal smoke/single/multi/saturated, audit and all three sanitizers are required;
missing prerequisites fail CPU preflight before allocating a GPU.

The packet schema is the existing schema plus `"stage":"diagnostic"`,
`"diagnostic":true`, `"ordinary_throughput":false`,
`"mc_throughput_claim":false`. Its commands remain exactly binary/args/timeout:

```json
{"binary":"inspector","args":["ncu","@binary:cuda_check",
 "--output","@run:diagnostics/ncu-local","--mode","local","--profile","0",
 "--threads","128","--warmup","8192","--sweeps","1","--beta","3.2",
 "--timeout","120"],"timeout":130}
{"binary":"hardware","args":["--case","rng","--seed","17","--iterations","4096",
 "--entries","4096","--threads","128","--blocks","1","--ilp","1",
 "--warmup","1","--repeats","1","--local","0"],"timeout":30}
```

The wrapper injects inspector `--expected-sha256` from the actual validated
cuda_check ELF and the fixed NCU path. Arbitrary executables, shell commands,
profiler executable overrides, output paths outside this run, and unlisted
options are refused. NCU timeout bounds its **whole** collection, including
discovery and up to four chain counts, not each subprocess independently.
Supported packet ranges: NCU timeout1..880s, sweeps1..16, warmup0..8192;
outer command deadline >=collection+10s and <=900s. Hardware deadline <=30s,
iterations1..65536, entries power-of-two32..4096, threads32/64/128/256,
blocks0..284 (0 requests actual SM count), ILP1/4/8, warmup0..1, repeats1..4,
local0/1, uint64 seed. These are hard ceilings, not measured durations.

Hardware generation defaults to shared32/shared16/packed8/integer/rng;
integer emits separate ILP1/4/8 commands (seven commands total).
`--hardware-cases shared32,rng`, `--hardware-iterations 4096` and
`--hardware-blocks 1` select short cases. CPU prepare optionally compiles this
isolated subdirectory Makefile before any GPU allocation. Its separate
`/builds/profile-tools/TOOLCHAIN/TOOL_DIGEST` cache binds hardware.cu/Makefile,
actual compiler versions and explicit target/optimization flags, not Python.
The hardware ELF and `.build.json` sidecar are independently hashed and checked
on CPU and GPU. The inspector is explicitly mounted copy=False and its payload
SHA is rechecked separately; it never enters the image or MC native cache key.
Receipts bind tool source, compiler, ELF, sidecar and inspector hashes separately
from unchanged source+four-MC-binary proof identity.

With defaults and both moves, estimated worst configured bounds are:
NCU-only: two130s commands, GPU hard290s, CPU hard270s;
hardware-only: seven30s commands, GPU hard240s, CPU hard450s;
both: nine commands, GPU hard500s, CPU hard450s. Existing reservations additionally
cover acquisition (CPU60s, L40S/H10090s or L445s) and62s lifecycle uncertainty
per worker, plus the unchanged conservative image allowance. These are
planning envelopes, **not observed execution time or measured cost**.

All diagnostic rows externally carry instrumentation ncu/hardware,
ordinary_throughput=false and mc_throughput_claim=false, even if a native
benchmark prints profile:false. `collect` leaves rate empty and keeps actual
proof hashes, cost, tool identity and diagnostic records in metric_json without
changing the TSV schema. Raw NCU CSV/reports and hardware JSONL live below
RUN_ID/diagnostics on the runs Volume; long diagnostics publish while running
(first2s then every10s), with the existing bounded commit/failure discipline.
Missing tool, counters, permission or supported metrics yield explicit unavailable
rows, never fake zero counters or successful measurements. The first unsupported
NCU result prevents all subsequent NCU permission probes in that packet; these
commands get explicit skipped-unavailable receipts. Native sampler/semantic
errors and timeouts still stop the packet. Download the actual raw reports with
the existing `modal volume get`, not a rebuilt substitute ELF.

## Budget and acquisition lifecycle

Limits are hard15, operational stop14, development admission11, at least3 kept
for final work. Before **each** image materialization, CPU invocation and GPU
invocation, a new exclusive-fcntl ledger transaction reserves its conservative
envelope. CPU2 is requested **and capped**, RAM4096MiB requested/capped,
min_containers0, max_containers1, buffer0, scaledown_window2 (SDK rejects0/1),
retries0. No idle service or automatic retry/fallback. Native builds never hold
GPU. A shared local GPU lock serializes independent CLI processes; CPU coding,
analysis and preparation can proceed independently.

L4 acquisition45s, L40S/H100 acquisition90s. The first worker operation records a
**durable Dict worker-start handshake**. Client observes that acknowledgement,
not call graphs/task_id, before waiting for bounded hard_timeout+60 completion.
On **every BaseException**, including builtin TimeoutError/KeyboardInterrupt,
client calls cancel(terminate_containers=True). Regardless of cancellation RPC
acknowledgement, a client exception leaves persistent `gpu_owner` unresolved:
later GPU admission refuses even after the OS lock is released. The claim stores
actual own app ID, invocation, reservation and FunctionCall ID. Only a direct
terminal FunctionCall.get result releases normal execution ownership. Otherwise
run `python solid-opt3/orchestrate.py close-gpu --ledger
solid-opt3/artifacts/budget.json --app-id ap-OWN_ID` using the authenticated venv.
It targets only the recorded own app, requests stop once, and requires actual
AppGetInfo STOPPED plus an empty authoritative TaskList before releasing
ownership. No elapsed-time, call-graph or cancellation-ack inference; failures
keep ownership and money charged. Closure is control-only, never a compute
retry or refund. L4 historically queued20min; a new L40S packet is an explicit
decision only after verified closure, never an automatic fallback.

Reservation seconds = configured execution bound + acquisition bound +62s
startup/shutdown uncertainty. Rates USD/s: L4 .000222, L40S .000542,
H100 .001097; CPU .0000131/core, RAM .00000222/GiB. CPU and RAM are included
for failed, startup, shutdown, CPU-only and GPU work. Image materialization
retains a separate >=.05 allowance using a1200s CPU/RAM envelope, even if cached.
This fixed SDK image-builder allowance is conservative planning, **not a
provider-side spending cap or an observed invoice**; inspect provider billing
independently if available. New source mounts do not invalidate the fixed image.

Failed/canceled/unobservable calls initially keep their entire original
reservation. Observed successful calls settle unused envelope to min(reservation,
(full client elapsed+62s)*configured resource rates). A returned **functional
error** is not automatically released: the explicit control-only
`reconcile-returned-error --ledger solid-opt3/artifacts/budget.json --receipt
ACTUAL_WRAPPER.json --inspect-only` command defaults to read-only inspection.
It captures the actual stored result SHA, matching historical/current direct
FunctionCall ownership, unchanged reservation and observed full-client elapsed,
fresh own-app STOPPED/empty TaskList, exact FunctionCall app/function metadata,
complete own-function input histories (including variants), and TaskGetInfo for
every discovered container. All must be finished and owned by the same app;
provider definitions must confirm CPU/RAM caps, retries0, max_containers1,
min/buffer0 and exactly one actual configured GPU container/call.
Only explicit `--apply` can reconcile, and missing metadata blocks it. Nothing
stops/retries a call or accepts canceled/unknown work.

The upper bound uses the **whole app** through actual container termination:
min(reservation, (max(max(App.stopped_at, all Task.finished_at)-App.created_at,
full-client elapsed)+62s)*configured rates). No scaledown-window billing-cutoff
assumption, worker-body/native-duration billing guess, or invoice claim.
See `proofs/BUDGET_SETTLEMENT.md` for parent verification requirements. If complete
container/configuration metadata is unavailable, retain the full liability;
missing current/partial provider billing intervals never establish zero cost.

Original reserved_usd, failed status/detail, receipts and physics.ok are never
changed. An auditable `reconciliations` entry records each distinct receipt and
provider proof; only `charged_upperbound_usd` may decrease, never below the
observed elapsed-rate product. Future errors require the same explicit evidence.
Image/CPU allowances remain unchanged. `committed_upperbound_usd` is used for
admission/spend; `reserved_total_usd` is historical envelopes, not an amount to sum
with it. Do not manually erase unresolved liabilities to authorize more work.

## Controller and next integration work

`python solid-opt3/orchestrate.py list` shows all19 ideas. `digest --backend`
computes native-only identity. `packet --stage ... --gpu ... --output PATH`
writes an **unexecuted** proof/single/multi/perf/scaling/final packet. Pass
`--receipts PATH...` from preceding actual wrapper/command receipts. Perf is
single-chain first; scaling takes an explicit measured chain-count list. Shape
axes and moves can be split into budget-sized packets. Final packet generation
needs fresh final-seed and final-chain>=1e6 and refuses historical seeds
15637821895788547078 /4585611281380521951. Never choose held-out IDs in advance
for development. `run --packet ... --budget-usd POSITIVE --gpu ... --ledger ...`
invokes authenticated Modal once and appends actual receipts to results.tsv.
`collect RECEIPT...` can recover rows without launching compute.

`abba --a A.json --b B.json --budget-usd ... --gpu ... --ledger ...` executes
four matched, already-gated unprofiled **A/B/B/A packets**, fail-fast. Each
native bench command itself requests four fixed-work same-aged-checkpoint
repeats; this is four packets, not a claim that sixteen inner repeats were four.
A/B packet launch/workload args/config/checkpoint bytes must match exactly;
source/backend and own-binary proof identities differ as intended. Native
benchmark resets same aged checkpoint/start each repeat. No timed trajectories,
performance predictions passed off as observations, or automatic promotion.

results.tsv contains only actual hashed receipts (including failures). Rates
are blank on failed/profile rows; measured gain remains blank until analysis
computes it. Conservative per-invocation charge repeats on its rows: deduplicate
run_id to sum spend. Do not count static SASS site counts as dynamic work or
profile stage zeros as measurements. Parent must verify syntax/native CLI/SDK
integration once, then establish baseline CPU audit+fixtures, smoke+sanitizers,
single/multi/saturated, fixed-work single-chain timings, measured scaling,
proofs for kept variants and held-out final cases before ten-minute production.
No checks have yet been exercised by this worker.

Old dead ends: historical17h ROOT used different physics/Philox; reuse reasoning
not timings/keys. CPU wider-cache hurt32KB L1; XOR bank permutation is neutral;
blanket register caps32/48 spilled,64 often. No evidence RNG deferral alone is5x.
NVBit standalone image once rebuilt ELF nondeterministically: a profiler binary
must prove **its own** bytes. Do not spend the new budget on optional tooling
before measuring the new implementation.

Final scientific target remains separate local/nonlocal ladders,
linspace(.01,beta_max,10), >=8 independent chains per beta/move, concurrency
chosen from actual scaling,600s actual production, bounded snapshots and exact
CPU replay from **downloaded actual initial bytes**/seed/start. Energy flattening
or speed does not establish equilibrium. Pilot beta_max with energy-block
stationarity and C(t;tw) versus C(t;tw/10), several waiting times/independent
chains; inconclusive is valid. User's local tau~8000 atbeta2.5 is ground-truth
context, not a request to rerun confirmation. Paper finite-size/VFT extrapolation
is not an actual L21 relaxation measurement.

Generic packet, benchmark and jobs capacity is **window <= threads*ppt**.
Shape expansion filters within the PPT loop, so W256/T64/PPT4 is retained for
a future implementation's real gates/timing. Baseline still explicitly rejects
PPT!=1; generic capacity is not a claim that a backend implements a shape.

The extended09:02:41Z deadline keeps start and all budget limits unchanged.
Parent owns actual ledger metadata; `_read` rejects stale/foreign window values
instead of silently rewriting a real ledger. CPU build receipts record optional
`tooling.ncu_path` and actual `ncu --version` output when found; absence is not a
failure and no installation is attempted. Final gate packet construction takes
actual preceding development receipts; it omits only the extra profile1 smoke
under profile=false final config, never real profile=true validation.

## Actual baseline closure and first measurements, after22:20Z

- Corrected source native digest `1ed16973b807e7b06508fc1346b9376d8a22e22b8b16a5eb39fa9cafc78d7627`. Receipt `1332f18ad17949bfa3a951ba0467888f`: normal434/profile434 smoke checks, all14 shapes, passed. Receipt `3657a27f8eab4cf4a88553c6dd46616a`: memcheck154 each move passed; racecheck120s envelope expired after42 valid checkpoints, no completed hazard result. Keep its full liability; no kernel change inferred from timeout.
- Receipt `386b972ae58248fd85bab6b1992da6c3`: **all unchanged remaining gates passed**. Racecheck both154, zero hazards/errors/warnings,502.434/501.709s; synccheck both154, zero errors,8.757/8.741s. Full single714, multi168, saturated112 checks,71.209/35.149/3.726s. Saturated includes8*142=1136 chains. Changed only sanitizer CPU-worker count2 and execution bounds900s, not fixtures/shapes/seeds/tolerances/reference. GPUbody1150.206s; settled conservative cumulativeUSD3.082768, not invoice.
- First ordinary timing receipt `a8ba2c6573524a69aaa232add5f7e751`, L40S beta3.2,8192 warmup, four reset-to-same-aged-checkpoint repeats of8192 sweeps. MedianM proposals/s: physical-local W128/T12884.7074, W128/T25683.6079, W256/T256105.5272; nonlocal106.8861/105.0977/134.4039 respectively. Local aged fingerprintcee9a9aa6dcf135b matched allshapes andaccepted count234032 acrossrepeats; noncryptographicfingerprint is not a byteproof. Nativegateproofs are separate above. Summary at artifacts/baseline-single-summary.json. CallGPUbody35.371s; cumulative conservativeUSD3.202986, not invoice.
- W128 profile=true gets its OWN fullselectedshape gates before clock64 stage runs at1,8,64,284 chains (packet baseline-profile-complete-and-stages.json). Profile rates remainnull. Normal scaling packet prepared for1,8,64,142,284,568 across the same3 shapecombinations; no scaling numbers yet.
- Standalone O14 repair/O13 parallelcommit/O19 halfpotential/O07 necessarydistance owners were released only after actual baselinecorrectnessgreen. Each wrote proof first; no builds/checks byagents. Parentbuild/gate/measure remains mandatory. No commonnative changes.
- New exact algebra: allH deltas even by occupied-edge doublecount/composition. Minimalpositive2, not1. Cache K=m(m+1)/2-n(ell+1)m+S lies[-45,39]. Farhole delta/2=Kb-Ka-ell*mb, adjacent subtractma+mb-2ell, distance2 subtractoccupiedcommonneighbors. Parent mathenumeration235298far/adjidentities passed, notGPU validation. See O19proof. O07's wrap-safe linear-index necessary-distance mask has33possibleL21 absdifferences/1160B, uniformcandidateprob.0048067; onlynegative skips exactgeometry. Mathematical sparsity is not a measured gain.
- Newusernorthstar: continue BOTH physical-local andnonlocal singlechain AND manychainfrontiers until deadline/budget; no earlytargetstop. Final09:02:41Z andUSD15 unchanged. No pushes.

## O20 separate colored transport decision (before implementation)

The optional `colored-local` experiment is not the physical-local engine and
cannot supply its proof gates, proposal-rate columns, relaxation clock or C(t).
The ideal fixed-edge Metropolis kernels preserve Gibbs measure under the
state-independent upstream axis/forward-or-reverse schedule. The frozen
quantized EFMA law is not literally that ideal law: same-schedule finite-horizon
coupling bounds divergence by M*2^-64, not stationary-law error or equivalence
between schedules. Empirical independent-terminal-configuration comparisons and
aging evidence remain separate prerequisites for any scientific promotion.

Transport design: diagnostic-only aliases `cpu_colored` and `colored` invoke
`colored_cpu` and `colored_gpu`. CPU fixture preparation is a real CPU prefix;
both native executables compile in the CPU worker before GPU allocation via
`profile/Makefile.colored`. A separate colored cache binds `profile/colored.cu`,
that Makefile, all three frozen CPU headers, explicit compiler flags, actual
compiler versions and pinned toolchain identity. Its marker and sidecar bind
both actual ELF hashes. This leaves the selected physical backend digest and
four physical executable hashes unchanged; Python and documentation do not
enter either native digest. Source mounts remain copy=False.

Normal smoke/single/multi/saturated, CPU audit and all three normal sanitizers
must already pass for the actual physical binary/GPU identity before any
colored command. A colored packet prepares its own oracle fixture, checks its
own complete schedule/first100k trace and runs memcheck/racecheck/synccheck
before any bounded four-repeat colored bench. Colored success receipts retain
`gate:null`, `diagnostic:true`, `sampler:colored-local`,
`physical_local_dynamics:false`, `ordinary_throughput:false` and
`mc_throughput_claim:false`; own-check admission is never a physical proof.
The sanitizer wrapper is stripped by the controller and invokes
compute-sanitizer with error-exitcode99 on the actual validated colored GPU ELF;
the existing explicit-clean-summary check is reused. Any failure stops later
commands. Budget admission, CPU resource caps, GPU locking, acquisition bounds,
per-command deadlines and publication headroom remain unchanged.

This is a pre-code design decision, not evidence of implementation correctness,
a completed stationarity gate, GPU performance, or exercised transport.

Planned CLI: `packet --stage diagnostic --diagnostic-tool colored` retains
normal `--windows/--threads/--ppt/--move` solely for physical proof identity.
Separate `--colored-threads` selects own-schedule check and benchmark launches;
`--colored-chains` defaults to 1,142,284,568 and `--colored-profile` to0.
Ordinary `--beta/--warmup/--sweeps/--command-timeout/--cpu-timeout` bound actual
colored work. The generated packet includes prepare, check, all three sanitizer
checks and four-repeat benches. Parent can narrow chain counts/work/deadlines
before admission. No terminal configurations or stationarity verdict are
fabricated by this transport.

Colored cache publication uses one atomic directory containing both ELF files,
a hash-binding `colored.build.json` sidecar and a final `tool.json` marker.
Every reuse checks the marker's canonical source/compiler/flag digest, expected
cache path, sidecar hash and exact sidecar content, ELF magic and both ELF
SHA256 values; GPU execution repeats verification without invoking a compiler.
Colored check/sanitizer receipts are tracked only inside their diagnostic
packet and must agree on fixture bytes, profile and covered colored threads
before benchmarks. They are never accepted by physical `requires` loading.

### Colored transport implementation handoff (unexercised)

The above optional path is now wired in modal_app.py and orchestrate.py.
`--colored-threads` accepts a distinct CSV subset of32,64,128,256;
`--colored-chains` accepts counts1..568; `--colored-profile` accepts0/1.
Own sweeps are bounded1..8192, warmup0..8192, prepare/command deadlines<=900s.
Defaults produce one CPU prepare, one own check, three own sanitizer checks,
and four benches at chains1,142,284,568 with colored threads128/profile0.
Only GPU command bounds contribute to GPU hard_timeout; CPU hard_timeout
defaults to prepare timeout+450s for the physical-cache/colored-cache phases.
The existing7200s GPU cap and20s publication margin still reject oversize
packets; narrow chain/thread/work selections or individual packet deadlines
instead of bypassing those limits.

Example packet construction (not executed):

```
python solid-opt3/orchestrate.py packet --stage diagnostic --diagnostic-tool colored \
  --backend cuda/baseline.cu --gpu L40S --windows 128 --threads 128 --ppt 1 \
  --move both --colored-threads 128 --colored-chains 1,142,284,568 \
  --beta 3.2 --warmup 1 --sweeps 10 --command-timeout 120 --cpu-timeout 900 \
  --receipts ACTUAL_NORMAL_PROOF_RECEIPTS.json --output colored-packet.json
```

The normal proof shape arguments are independent from colored launch threads.
CPU preparation and GPU checking reuse only the colored fixture and own-schedule
oracle; `gate:null` remains enforced by classification. Benchmark admission
requires earlier successful own check plus all three explicit-clean sanitizers
covering the same fixture bytes, profile and colored threads. Their receipts
remain separate from physical proof receipt lists. All four physical native
hashes/source digest are unchanged by this optional auxiliary path.

Parent must exercise Python syntax/packet admission, normal-path regression,
CPU cache build/reuse/tamper rejection, own CPU/GPU byte/trace checks, all three
sanitizers and bounded diagnostic collection on actual binaries before use.
No builds, tests, formatters, Modal calls or transport checks were performed by
this worker. The transport does not implement or claim an empirical stationarity
verdict, unchanged physical dynamics or a physical-MC throughput result.

## Actual subsequent observations, after 2026-11-11T00:03Z

- No pushes. Local branch `opt/solid-opt3-O14-selective-repair`, commit595e5cb, records the passing normal-all14/selected-sanitizer experiment and its rejected W256 measurements; fast-forwarded into opt/solid-opt3. Archived candidate is not a production promotion.
- Baseline W256 scaling receipt5861cd7f1a1840c8a12c2c99de6afa2c: at284chains, local79.012M/chain,22.439G/card; nonlocal111.055M/chain,31.540G/card. At568chains, local40.802M/chain,23.175G/card; nonlocal56.286M/chain,31.970G/card. Single-chain baseline105.5272M local,134.4039M nonlocal. These are attempted proposals, not decorrelated configurations.
- First four standalone trials passed full normal all14-shape trajectories and representative W256/T256 all3 sanitizers before timing. O14 repair89.025/115.079M single local/nonlocal loses; baseline reevaluation overhead only roughly2%, not a2x opportunity at W256. O13 parallel accepted-prefix commit100.167/117.088M loses single-chain; local card near-neutral, nonlocal loses. O07 necessary-distance filter98.346/127.070M loses. O19 half-potential105.533/134.863M near-neutral (+~.4%nonlocal), not promoted without matched ABBA. Actual receipts/rates are now in their configs and REPORT; profile/all-shape-sanitizer/other-card coverage is explicitly unverified.
- Selected-shape exploratory ladder now keeps every original L/beta/id/trace/counter case at W256/T256, followed by all3 sanitizers then four-repeat rankings; fail-fast. It does not claim untested shapes. Final retained composites still require universal/profile/cross-card/held-out gates.
- Seven selected packets initially mixed mount alias development.bin with @fixture:fixture.bin. All failed before native launch, not CUDA correctness. Fixed every reference to the actual mounted name and asserted aliases locally. Corrected O02 receipt eb95498c91534b72ae45b565ea1b984e passed smoke11 per move, single51, multi12, saturated8, and all3 sanitizer tools both moves. Single102.722M local/132.059M nonlocal;284chain77.932M local/110.778M nonlocal: not a win.
- Corrected O01 receipt40ab477b198b43429e43338d9094f974 passed the same selected ladder and all3 sanitizers. Single106.049M local/136.280M nonlocal;284chain79.159M local/112.435M nonlocal. Roughly+.5%local/+1.4%nonlocal versus prior baseline; provisional, not a demonstrated multi-x gain or production promotion. O03 receiptf9b4b1c41f924217be85b9bdea90a08e also passed its selected ladder/rank packet; analysis pending.
- O12 packed cache receipt9da66af46e3449e282f749453de12fb6 passed selected exactness and all3 sanitizers before timing. Single104.377M local/132.379M nonlocal;284chain77.433M local/108.333M nonlocal. Slight loss at two CTAs/SM; four-CTA occupancy benefit still requires568/1136-chain measurements. Do not infer all-card benefit from packing.
- New O10 wide-window source905ddcc97951e0d82549715d8ba78fa6ea889f5994703c746bc2d7eed77e8ea7 compiled216 functions across both architectures. Actual SASS inspection found no local resource bytes/LDL/STL in normal L21 modes for any compiled capacity/launch-bound pair. This is static resource evidence, not a runtime speed claim.
- O10 receipt77bf6aca0ec2484b9dcf914508a45b24 passed normal smoke/single/multi/saturation and both memchecks for W512/T512,W512/T1024,W1024/T1024. Local racecheck120s expired; no completed clean result, no throughput accepted. Remaining same cases queued with300s race deadlines. Earlier CPU-only admissions rejected an old proof scope and missing explicit proof fixture SHA; no GPU was requested on those errors. Keep scopes strict: prefix a real CPU audit for expanded axes and declare unchanged development fixture SHA5b42f75650ff030456f76ba55fbcf248fc03aeb8c52f6fc2c172d951e23b6752 when referencing prior GPU gates.
- O09 empty-window, O10 ILP/lookahead, O12 packed cache, O15 epoch claims, O16 RNG pipeline, O17 exact limb RNG and O06b global coordinates all completed CPU-only Modal dual-architecture builds/audits in parallel. Their source digests/receipts are in artifacts/gates and local kernel objects compile. This does not imply GPU correctness.
- Parent wrote proof then implemented O09b accepted-only shared metadata publication: every shared metadata reader is indexed through an accepted mask, while private rejected/no-op readers and all dependency cuts remain unchanged. Parent also proved/wrote O18b a512-byte exact shared threshold prefix with global fallback. Both local sm89/sm90 compilation passed; neither has GPU evidence yet.
- Actual NCU collection works. Offline parser now handles wide671-column CSV plus units row and bare/typed0/1 kernel spellings; saved first local one-sweep report yields660 counter records. Actual issue16.47%, eligible.16warp/activecycle, active1warp/scheduler,2102 shared-load bank conflicts, zero shared-store conflicts. This short capture includes initialization/export and is not a pure hot-loop roofline. Original no_counter_values_collected label remains in the raw receipt; derived result retains provenance. SASS baseline39regs/no spills; real microbenchmarks found packed8 random logical loads slower than32-bit, not a universal packing result.
- Standalone optional CPU fast port passed its full frozen-reference S/K cache matrix, including full-width counter boundaries (artifact753). Pinned core1/O3native/L21/beta3.2/warm8192/kernel-only10s: S43.984M local/45.887M nonlocal; K45.275M local/46.859M nonlocal. Same per-move aged checkpoint for S/K; time-limited timed work differs. Original O2 CPU wall+NPZ measurements are not identical timing methodology. Measured records at artifacts/cpu-fast-measured.json; frozen headers unchanged.
- Colored diagnostic CPU/GPU executables compile locally; own CPU oracle fixture36 cases prepared. No colored GPU gate, sanitizer, statistical stationarity verdict or performance exists yet. Never relabel this schedule as physical-local dynamics.
- Provider accounting review distinguished functional errors from unclosed calls. Actual authenticated FunctionCall histories, function resource/retry definitions, all TaskGetInfo finished timestamps, exact own App STOPPED/empty TaskList and direct result SHA support conservative whole-app-through-last-container plus62s bounds. All seven alias failures reconciled explicitly; original reservations, failed status/errors and limits remain. CPU containers use provider GPU label CPU with count0; only that/empty label is accepted. Negative metadata/SHA/app/config cases rejected in parent throwaway checks. No invoice claim. Other unresolved/unchecked historical liabilities remain fully charged. Last observed conservative snapshot9.936405USD included in-flight reservations; always read live ledger before admission.
- Correct runner syntax: orchestrate.py run has no --modal switch; it uses its pinned MODAL constant. campaign.py run/download do expose --modal. Eight unsupported-flag attempts failed locally before any reservation/app; no GPU claims. Image/source hashing remains native-only and pinned.
- Actual180s baseline pilot receipt05aca736725b456ebd2f8961c8fa1fad succeeded:8 jobs/64chains, roughly1.05–1.26M sweeps/chain across beta2.5/2.8/3.2/3.4 and both physical moves. Downloaded real data; all retained4-sweep probes replayed byte-identically on CPU, with unreplayed gaps explicitly listed. First download hit SDK destination file/directory race; precreating destination directory fixes the real CLI surface. Retried bulk download succeeded.
- Every pilot scientific verdict is inconclusive. Energy alone is insufficient; predeclared primary h=10000 group has lags only up to40000 and shows aging/insufficient decay. beta_max remains null; do not lower margins, postselect later groups or announce equilibrium. Next scientific decision: a fresh600s pilot at beta1.5/2.0/2.5/2.8,8 chains per move/beta, new master seed930272, aiming to observe another predeclared waiting-time scale while covering clearly easier values. Plan/run still pending; final ten-point600s campaign remains separate.

## Publication deadline decision, before the controller fix

Actual600s lower-beta pilot98730c92bf9a400b8b6d0737fc9fe734 was killed after169.096s, not by MC failure: periodic runs.commit exceeded the hardcoded8s wait after thirteen successful publications. Native stdout/stderr had no physics error; the failing publication is retained as failed. The run is not a completed600s pilot.

The packet already has driver command690s and worker780s bounds. Replace the incidental8s commit deadline with min(60s, remaining worker budget minus10s), and await the publisher for min(65s, remaining budget minus5s) at shutdown. Commit exceptions/timeouts remain fatal; no retry, swallowed failure, fabricated durability or native/kernel/hash/gate change. This fixes an incorrectly short provider-I/O deadline within existing admission envelopes. Parent must exercise actual publication before another long pilot/final campaign. Saved partial checkpoints can be downloaded and CPU-replayed; uncompleted sweeps must never be invented.

Actual C04 CPU-only preparation f1b4cbc73f5440c6b7e656e64c0ab705 failed solely because the CUDA make subprocess had an incidental120s limit. Local full dual-architecture object compilation took119.59s, and actual CPU-container build_rows records timeout120s, not compiler diagnostics. Decision: allow native build subprocess420s under the unchanged overall CPU900s/resource envelope; _execute still enforces the existing absolute deadline and publication margin. No CUDA source, compiler flags, fixture, physics or proof changes are needed.

## Frontier and budget state after 2026-11-11T01:08Z

- Actual C03 selected W256/T256 all-gates/all3-sanitizers pass e397495ef48843499682e1708b8a0b42: single155.065943M local/171.019242M nonlocal. At142chains137.964689M local/166.715802M nonlocal per chain;284chains33.763985G local/39.906597G nonlocal per card;568chains34.409227G/40.140393G. Best current depth, not universally promoted.
- Actual C02 packed+type selected pass562e53f4ccd74420b5b08ae2fa14fc5b:568chains39.958646G local/46.654498G nonlocal;1136chains40.839028G/48.378777G. Best card throughput, lower per-chain depth. Single133.627680M/161.449222M. These are distinct Pareto points.
- C01 wide+type W512/T512 passes but loses to O03/C03: single126.358550M/158.357263M;284chains21.891748G/29.035535G. O10 wide900s gate receipt69a3608c55124c9fb7808e0af0bf694b completed race322.596/348.325s,33checks each, sync33each. All three larger-thread shapes lose; W1024/T1024 single72.620M local/100.399M nonlocal. No larger-window speedup claimed. Virtual PPT2/4 packet was budget-refused before GPU allocation, remains untested.
- CPU occupied-type fast path passed existing full S/K reference matrix (artifact869). Same pinned-core1/O3native/warm8192/kernel10s method: K47.899628M local/51.113403M nonlocal; S43.043221M/44.093244M regresses in this run. Exact source3e26cc3be810ccad8fefbc34df3a5c7214f19464f2b3b404d5b698d75b34c394. Use K as equally-clever CPU comparator with timing/seed caveats, not old O2 wall+NPZ rates.
- C04 wide selective repair dual-arch compile and CPU-only native-cache/audit passed d65b15ad120b44fb9e9722f8b37b6637 after fixing only the120s compiler deadline. All compiled fixed-L21 normal capacities show zero LOCAL/LDL/STL. W2048 shared94724B plus static fits L40S; W4096 dynamic152324B needs a larger device such as a queried H100. Tiny W2048/T512/PPT4 single-chain exact gate queued before any performance claim.
- Publication60s-budget fix actual45s smoke f2fe53a598a548e5a6484a1e432b3771 passed,45.854s native wall,8jobs,three committed checkpoint publications. Interrupted600s pilot partial data downloaded and CPU-replayed successfully; it is still only169s, not600s.
- Provisional actual-pilot beta selection is2.0 from nonlocal1.5/2.0 compatible stationarity/decay evidence, not equilibrium proof. Nonlocal2.5/2.8 and all physical-local controls remain inconclusive. Artifacts/provisional-beta-selection.json; do not erase those limits or relabel the partial run complete.
- L4 trial74f3fff3fe204128bacdf862e5c57bb8 obtained no durable GPU-worker acknowledgement within45s. No L4 correctness/performance result. Exact own app ap-X9saCNAYzzDZdqh3PQ8UiD is authoritatively STOPPED with empty TaskList after explicit close; original unresolved/canceled liability remains fully charged. Do not repeat the unavailable-card queue.
- V2 task-only GPU accounting was reviewed and rejected: input histories do not enumerate every historical startup/replacement allocation. It was never applied to the ledger; V1 whole-app accounting restored exactly. Do not revive that unproven premise.
- Conservative snapshot01:08:31Z:10.803738USD, including76 independent.05 image allowances=3.80USD,CPU.318439,GPU6.685299. No cap raised;15hard/14operational/11development/3finalreserve retained. These are planning/resource envelopes, not invoice.
- Parent ACTUALLY fetched authenticated app-filtered WorkspaceBillingReport over COMPLETE hours19:00Z–01:00Z for exactly76 own apps.77rows cover72apps and total3.71999518USD: L40S3.38666311,CPU.24442124,Memory.08264465,NetworkEgress.00626614. Missing/current-hour apps are NOT zero. Raw and summary artifacts retained. A separate strict explicit metered-error reconciliation path is being reviewed; no metered credits applied yet, no arbitrary local-file refund, CPU/image liabilities remain unchanged.
- Grouped scalar4/vector4 hardware helpers compiled; identical CPU checksum15902321798514708071 for2097152logical word reads/524288groups. No GPU measurement yet. Controller now accepts these explicit case names; physical native hashes unchanged.
- Six-hour milestone reached; this is NOT the stopping condition. Continue toward09:02:41Z or genuine budget exhaustion, preserving final two-hour verification window from07:02Z.

## Explicit internal reserve reallocation decision, before mutation

The user hard budget remains15USD and the operational GPU stop remains14USD. The earlier11USD development/3USD final split was this assistant's planning allocation, not permission to erase liabilities or a separate user limit. After selected final-candidate sanitizer measurements and successful publication smoke, reallocate that internal split to12USD development/2USD final. Preserve every reservation, charged amount, failure, unknown-L4 liability, image allowance, timestamp and ownership record; append a policy-change audit entry. No metered/report credit and no hard/operational-cap increase.

The2USD final reserve is for actual retained-source all-supported-shape normal/profile equivalence, warp/CTA representative all3 sanitizers for both moves, fresh held-out1e8 comparisons on L40S/H100, clean local clone/build, final600s production and checkpoint/resume smoke. Existing selected L40S sanitizers are already measured green; final time window still starts07:02Z. Use bounded commands based on observed runtime, not weakened fixtures or tolerances. If hardware is unavailable, record the exact unexercised criterion rather than claim it passed. This planning reallocation keeps total operational allowance14USD and permits remaining correctness-gated hypotheses instead of unsafe accounting refunds.

The proposed metered-error refund was also rejected: Modal documents collection delays and supplies no ingestion-finality watermark. Complete hour buckets/positive rows are useful cost diagnostics, not proof of finalized billing. New inspection is read-only; V1 lifecycle accounting remains the only implemented debit-reduction rule.

## Frontier, CPU evidence and card-allocation stop, 2026-11-11T02:40Z

- C07 `cuda/fast_empty_window.cu` native8020c79afcbb7af5d82234c89a6e458fa95d5b43493fe0ff5c3a9774636481e8 passed selected W256T256 smoke/single/fulltrace/multi/saturation and all3 sanitizers BOTH moves before rates, receiptbcb0075d40664e588f9435232fc4653d. Single169.671703Mlocal/207.955519MNL;142chains142.068618Mlocal/197.339042MNLperchain. Race29.669/30.725s. A preceding call3647394228b34051aa807c4aff3e4ab4 prepared/cache/audited successfully but GPU admission refused; no GPU check/rate from that refusal. Retry used unchanged fixtures/checks with deadlines based on observed neighboring runtimes, not weaker tests. Budget committed11.926300202USD afterward.
- All six remaining selected trials passed full selected ladder/all3sanitizers: O09emptywindow wins; O09bmetadata neutral; O15epochclaim loses; O16pipeline nonlocal loses; O06bglobaltables loses; O18bsharedthreshold neutral. Actual summaries `artifacts/O##-unitSM-summary.json`.
- C05/C06/C08 dual-arch objects and inspections succeed, zero LOCAL/LDL/STL on normalL21. C05W2048/MaxThreads1024 registers56local/55NL sm89,54/53sm90. C08 sm89regs40, sm90local48/NL47. Reviews found no evidence-backed defect; not GPU verification. C09new exact modular shared-index-product proof/source/object/SASS created, no GPU gate/rate.
- H02 grouped scalar4 and vector4 BOTH compile to one LDS.128 with identical opcode histograms (sm89regs24/sm90regs22). CPU checksums match15902321798514708071. This is a useful negative comparison; do not add artificial dependencies to manufacture a gain or claim GPU timings. Raw packed8 microbench already measured4.7284Glogicalreads/s versus5.5128Gshared32, not MC proposals.
- Generated results.tsv projection compacted80,402,615→1,920,080bytes, preserving all931rows and SHA1947f4ba2aca806aa1ee94d87a2e3f3e86d03b4164decd0bb9dbc2f450fd06f9 of every nonmetric column. Raw receipts unchanged. Recollect actual old42ae... receipt appended0rows. Collector now references immutable raw diagnostic reports/records rather than embedding them repeatedly.
- Actual CPU acquisition/resume/absolute-history merge agrees byte-for-byte with original monolithic340sweep/8chain reference on state, metadata, energies, counts and common snapshots. Proof `artifacts/cpu-relaxation-acquisition-check.json`. CPU-only fixed/predeclared1Msweep trajectories now run: NL8chains at2.0/2.5/2.8/3.2 (cores2–5), local32chains at2.0/2.5 (cores6–7). Jobsbg290–295auto-deliver; no polling. No Modal spend; helpers enforce180spersegment/global09:02:41Z. Use unchanged scientific diagnostics, no false GPU-rate/eq claims.
- **Stop further L40S calls.** Fresh authenticated complete-hour report through02:00Z gives L40Sresource3.93725580USD /0.000542USD/s = at least7264.309594s (2.017864h), excluding possible delayed charges. The original2hcard allocation has already been exceeded by at least64.31s; earlier worker-time-only accounting missed allocation lifecycle overhead. This is an allocation overrun, not a USD15 breach. It cannot be undone or relabelled. Add an admission stop in the locked ledger; preserve all charges/history. Finish reachable final proof on L4/H100 and state any unexercised L40S1e8 criterion explicitly. Never claim it passed.
- Reported resource costs are diagnostics, not finalized billing or permission for credits. Operational14/hard15 stay unchanged. Start actual missing-L4 final hardware qualification early (sourceC07frozen, developmentfixture, no timing) with45sacquisition; protected final funds only. L4 last failed01:03, so this is a bounded new availability interval, not an unbounded repeat. Final600sproduction still required; choose L4 if allocation succeeds. Do not spend further protected funds on generic research variants without a final-retention/verification purpose.


## Bounded final-card fallback decision

Both L4 probes failed acquisition; latestb19596e0869c4b70960cfcaab4b3ea79 explicitly closedAppap-erVsktywss40YDGFOElp5j withSTOPPED/emptycontainers. No L4 correctness/rate is claimed; full canceled reservation stays charged. L40S remains admission-stopped after its2h allocation overrun.

Add explicitly selected **A100-40GB** as an additional final-verification/production fallback, not a substitute claim for L4/L40S/H100 observations. Primary sources https://modal.com/docs/guide/gpu and https://modal.com/pricing confirm the selection and0.000583USD/s GPU rate. All existing user dollar caps/stops stay unchanged. Build the selected unchanged physics source with an explicit extra sm80 gencode in native flags; this has its own digest/proofs. Tiny exact gates precede all larger checks/timing. No image recipe changes and no credits.

For requested H100 measurements use provider `H100!`, preserving the public logical selector/rate `H100`: current Modal docs otherwise permit automatic H200 upgrades. Log actual device identity; never label an H200 measurement as H100. Keep actual H100 allocated time within15min and use the new fallback for600sproduction if needed.

Compiler evidence also rules out blind micro-tuning: C07 already contains native REDUX.MIN.S32 and REDUX.SUM, so replacing manual source reductions is not assumed to save hot instructions. C09shared-key product increases whole-function static IMAD/LDS/STS/BAR counts and may increase registers; no GPU speedup or retention is claimed. Image-ID-only deduplication was independently investigated and rejected because SDK metadata does not prove exactly-once build/billing history. No image allowance refunded.

## Actual final qualification and production integration, 2026-11-11T03:58Z

- A100 fallback actually allocated NVIDIA A100-SXM4-40GB,108SM. C08 packed+fast+empty selected W256T256/PPT1 passed every normal/100k-trace/multichain/saturation gate, all3 sanitizers both physical moves, and reserved100M-proposal fixtures for both moves before timing. Receipt0e81177be78643e1a82985e7e60dc8d4, native5e323f37ca6cf5e31f254b32ce69baf97e09a45dd40a2cf61ec5624b1c6afa4e, explicitsm80extra flags. Single95.556175Mlocal/116.816217MNL. At432chains20.423742Glocal/25.708913GNL/card;648chains19.315582G/26.168522G. Never compare these as algorithm ratios to L40S. All-shape/profile/H100 coverage remains pending.
- Reserved final fixture SHAce5f0523c7c387921d43e9ec2dab5aba8ad2b4144c73cd7509283edf3574ac5d, seed6768922282260232020, chain1000000, L21beta3.2,100Mproposals each move. Only these two held-out fixtures are claimed; not three held-out IDs. L40S held-out remains unexercised because that card is stopped.
- C05 wide repair W2048T512/PPT4 also passed all selected normal/trace/multi/saturation, all3 sanitizers BOTH moves and the same final held-out fixtures on A100. Receipt9b0e6759bc7047f9af8a27785f901963. Race175.203/198.921s, final7.217s. Its FIRST timing command hit12s after only2of4 repeats; this is a benchmark deadline failure, not a failed physics/sanitizer gate. No accepted four-repeat throughput, no promotion. Do not mislabel it a racecheck failure or cite partial timing as a completed experiment.
- Conservative committed envelope12.839558012USD after the A100 calls, not an invoice. L40S admission veto remains locked. No image/metered credits. Further GPU work must be actual final qualification/production inside operational14USD; development is exhausted.
- Completed original-CPU nonlocal1Msweep/8chain diagnostics at2.0/2.5/2.8/3.2 are ALL inconclusive under unchanged predeclared margins. At2.0 decay/energy look compatible but waiting-time200k uncertainty exceeds.05; do not erase that result or assert equilibrium. Larger32chain beta2.0 uses the SAME original seed1131250365872000, retaining the first8 chain identities. New beta1.5/32chain control seed1131250365872011. Original local32chain2.0/2.5 remain running. Pure CPU acquisition/resume is independently byte-checked, no Modal spend.
- Real saved GPU checkpoint exposed two Python recovery errors: native history decimation256→512 was wrongly rejected; high uint64 sweep origins were wrongly constrained by proposal-index arithmetic although native counters intentionally wrap. campaign.py now accepts only power-of-two cadence multiples, preserves actual cadence, and uses native sweep-range bounds. Four deterministic regression tests pass. Actual old180s GPU production now produces a complete8job resume packet; no GPU trajectory or test tolerance changed.
- Fixed-S production now has pure producer.py plus proof-bound production_plan.py and source-mounted Modal transport. CPU 12jobs (2betas,2moves,5chains in2+2+1batches), E11/S37/chunk16/historycap4 completed through a real interrupted/resumed boundary; full monolithic CPU byte/energy/counter comparison is next. GPU executor still unexercised. Source identity explicitly distinguishes controller native digest from embedded NPZ Make hash; driver SHA and both identities survive resume. No native file/image recipe changed.

## Production surface fixes observed and verified, 2026-11-11T04:45Z

- Read-only production review found detached native children could survive the Modal producer group's timeout. Native children now inherit the outer group, with direct-PID kill/reap for the inner deadline. Real frozen CPU driver cancellation tests pass for outer SIGKILL and inner TimeoutExpired; artifactproducer-cancellation-check.json. No mocked native execution.
- First actual fixed-production GPU smoke bf46e8e6b66147cc864e1a9851fa7380 failed because Modal Volume rejects os.link. This is a filesystem capability failure, not a CUDA physics result. Replace both immutable NPZ/raw publishers with exclusive-create streams; publish the manifest only after those streams close. Partial unreferenced files are not committed progress. CPU portability smoke preserves rawSHA bc6c8e339ab725fe6425853c232dc4a19ad9a1469b5afcb15297f3fd3d73e15d. Corrected actual A100 smoke e7fe69f0ee714ac5a43cc13b8df2d44e passes, two committed chunks through absolute sweep27, full-width seed15492044448512513609; worker12.635s. Downloaded data and cross-oracle/recovery are next.
- Closed-error V1 reconciliation of bf46... actually applied after authenticated STOPPED/emptytasks plus complete call/resource/lifecycle metadata: original GPUreservation.16811776 retained, conservativecharge.057536548581. No metered/image credit. Use the existing Modal venv Python for SDK-dependent reconcile commands; systempython lacks Modal. Latest authoritative budget.snapshot committed13.051597415USD. Bare elapsed-products sums are NOT spending; they omit mandatory charged bounds/allowances.
- Actual A100 flag propagation bug found before any long campaign: campaign.py constructed native_flags={} despite selected proof using explicit sm80. Derive normalized flags from actual immutable proof wrappers, require consistent flags and a local native digest match, propagate them into packet/plan and set expected_source_digest. No compiler/image/native/reference change. Planner must pass an actual A100 smoke before the600s run.
- C10 direct field claims proof was written before a new isolated candidate. It removes wide pairwise accepted-predecessor scans with three complete field-class minima, reuses consumed queue storage for warp-cooperative writers, includes no-accept suffix finalization and an8bit epoch whose rollover is forced by unchanged million-proposal gates. sm89/sm90 object builds161.79s, actualSHA5f948a978b84b29ea7f7c317f3462876c626e2228e0a3ad4d511897ec1a3d1b1;288 compiled functions. Resource analysis pending. It is NOT GPU-verified, measured, retained or production-supported for wide shapes. No current native ABI or old proof changed.
- New32chain nonlocal beta2.5 CPU control uses SAME earlier seed1131250365872001, core4, fixed original1Msweep schedule. This increases power without changing seeds/margins or erasing the8chain inconclusive result. Expected completion before final data selection, not a GPU throughput experiment.


## 2026-11-11 06:10Z — sparse execution model, actual spill fix, stronger relaxation evidence

- C11 implements immutable endpoint incidence, unique dirty queues, cached original acceptance masks and minimum-original-index ordered repair, with warp0 committing while other warps inspect metadata only. C12 composes that state machine with C08 packed cache operations and paired-owner uint16 links. Independent actual-source reviews found no defects; neither review constitutes runtime proof.
- Forceinline alone did NOT solve the actual sm80 local stack: still32 registers/STACK16/10LDL/4STL. Explicit `launch_bounds(MaxThreads,1)` did: actual C11 and C12 objects each have zero STACK/LOCAL/LDL/STL for all nine normal L21/CAP2048/MaxThreads1024 mode/architecture entries. C11 object4dd186e9646a8260eed08310993baaa9b23fd2c924b446697134031e7646b90a; C12 object7ade01d625c53eb05520d34ee9101fa429eb98c13f27e83cd3af8d17625f3ab1. Targeted raw SASS/resources/summary artifacts retained; no GPU or occupancy claim follows.
- C11 final-candidate A100 invocation is active: unchanged CPU audit, tiny/single/fulltrace/multi/saturation, all three sanitizers for both moves, heldout100M, then normal four-repeat scaling at1/8/64/108/216 chains. Native digestb7d5e0ee3886758bae5183188f8bfa8c0e3bd06dab0fc8db0989df41bcd9cfa3. GPU wall-time abort bounds protect production funds; check arguments, fixtures, seeds and tolerances are unchanged. A timeout is unqualified, never an accepted performance result.
- The wide W2048/T512/PPT4 CPU production ABI actually completed12jobs/960chainsweeps and matched the existing, not regenerated, monolithic frozen CPU oracles: all280 energy/acceptance observations,60 common snapshots and final bytes. `artifacts/producer-wide-cpu-check.json`.
- Original CPU32-chain nonlocal β1.5 and2.0 runs each completed1Msweeps and passed the unchanged predeclared late waiting-time/energy/decay diagnostics. This means compatibility with stationary relaxation, NOT equilibrium proof. β2.0 original8 chains are byte-identical to the first8 of32, including218 saved snapshots and all common observations; count32, not40. `artifacts/cpu-relaxation-expanded-prefix-check.json`.
- Local32-chain β2.0 is compatible with stationary relaxation, τ bracket[3000,4000]sweeps. Localβ2.5 remains inconclusive and right-censored beyond40000sweeps. Nonlocalβ2.5/32 is still running. Do not transfer either move's relaxation result to the other.
- Budget13.120610774899376 currently includes the new CPU/image reservation; pending C11 GPU envelope is422allocated-seconds ×0.00061808USD/s =0.26082976USD. Keep approximately0.61USD available for the real600s experiment plus fresh/resumed producer verification. No new L40S or repeated unavailable L4 acquisition, no speculative image/billing credits. H100 remains unmeasured.
- Deadline09:02:41Z; final verification reservation starts07:02Z. Next: consume the actual C11 outcome, finalize empirical βmax after the32-chain evidence, run the actual600s ten-beta campaign and GPU checkpoint/resume, then verify saved bytes and spending. No pushes.

## 2026-11-11 07:10Z — final allocation executing, not yet a completed campaign

- C11 final-candidate receipt1ba190f801a84045b3bdce92605c20c9 passed normal smoke/single/fulltrace/multi/saturation and both memchecks, then local racecheck timed out45s. No race clearance, heldout or timing follows. Archived local branch `opt/solid-opt3-C11-unretained`, commit3c6e1d3. C12 is static/compiled-only, branch `opt/solid-opt3-C12-unqualified`, commit0f72bc7.
- Restored only our native launch-guard change. Current C08 native digest is exactly the previously fully qualified5e323f37ca6cf5e31f254b32ce69baf97e09a45dd40a2cf61ec5624b1c6afa4e, with explicit extra sm80 flags. Wider native ABI remains reproducible on the unretained idea branches. No physics/header/gate/fixture change.
- Authenticated C05 returned-error reconciliation applied only after complete own-app definitions/input histories/container closure/resource/retry evidence. Conservative committed bound became13.354836041498006USD. Failed status and original reservation remain; no image, canceled-call or invoice credit was invented.
- Production review found two actual blockers before paid execution: CPU preflight hashed future owned-run recovery files, and a six-second fresh probe could not meet the deliberately conservative30s first-launch allowance. Fixed only the former's admission timing: preflight defers owned-run hashes, runtime still hashes actual files and enforces all proofs. Fresh producer now40s/timeout70, resume50s/timeout80. The real-filesystem/function-AST smoke passed preflight deferral, runtime missing-file refusal, immutable-fixture refusal and actual CPU manifest/raw hashing; no import-time Modal budget admission was performed by that smoke.
- Final probe CPU fresh/restart actually completed all4jobs and12chainsweeps: beta1.5, both moves,3chains in2+1batches, production2/equil0/chunk1, master9223372036854775965, restart at absolute1. Active raw SHAe83fae27f580eb2fd13998b15acb0b601ce9e62b2b1fa403158920bc2fceae88. GPU version is executing, not yet claimed.
- Nonlocal32-chain beta2.5 completed1Msweeps and remains inconclusive under unchanged criteria: late overlap has not demonstrably decayed near zero. The actual GPU pilot selector therefore freezes beta_max2.0; beta1.5/2.0 passed, beta2.5/2.8 did not. Local2.5 remains separately censored. No VFT extrapolation or universal tau multiplier substitutes for these observations.
- Actual final Modal invocation started07:10Z: C08/A100, original source-bound normal/sanitizer/heldout proofs, fresh producer then process-restarted producer then native `--seconds600`, ten betas linspace(.01,2,10), both moves,21chains per cell/420total. New campaign master4123495003391909354. Packet `artifacts/final-campaign-600/packet-with-resume.json`, provenance `bundle-plan.json`; requested worker800s, CPU250s, native630s. Pilot-calibrated chunks are approximately1424–1575sweeps, not the invalid8192 bound considered before planning; estimated output1,384,730,000bytes. Mounts use the actual `local`/`name` schema.
- Development GPU work is stopped to protect this allocation. Next: consume its actual result, verify downloaded producer bytes/energies/counters and retained campaign probes, diagnose all20cells, finish the fresh-clone build and local archival/reporting. Deadline09:02:41Z; no pushes.

## Final operational stop and verified handoff

- Conservative committed ceiling is13.996047228003794USD, not an invoice. Only0.003952771996205584USD remains under the mandatory14USD operational stop, below any admitted allocation. The final failed allocation retains its full liability: authenticated reconciliation twice failed on provider FunctionCallGetInfo resource exhaustion. No missing metadata, image cache, canceled call or invoice assumption was converted into credit.
- Actual final receipt02fd2c40c8b14c0bb72d41032e0a5628: fresh fixed-sweep producer4.197s and resumed producer11.570s completed all4jobs/12chainsweeps, both moves and2+1batches. Every checkpoint's bytes, integer/normalized energies and acceptance counts matched monolithic frozen CPU runs; artifactproducer-gpu-final-replay-check.json. Default16384-chain-per-beta/move planning also executed locally without compute, using C08's actual selected proof; artifactproduction16384-plan-final. It did not run16384chains.
- The separate native600s command timed out at630s. Actual manifest has18deadline cells around603–622s and two stillrunning cells: local-b00(last durable wall564.38s), local-b06(584.83s). No run.tsv exists. Do not call this a clean completed ten-minute campaign. All38 retained replay probes match CPU, with unreplayed gaps disclosed; diagnostics report13compatible/seveninconclusive and never equilibrium proof. All20actual checkpoints produced a pure resume plan successfully, not executed.
- Do not squeeze the existing planner's690s native/900s worker envelopes down to630s/800s again merely to fit a balance. The observed low-beta kernel times are much slower than >=1.5 pilot extrapolation, and the mixed-stream/host-publication path needs measured end-to-end tuning. No unmeasured I/O attribution or hypothetical speedup is promoted.
- Four independent full-H arithmetic audits passed1M proposals each: random-startL21/beta3.2 local/nonlocal, and actual agedL21/beta2.0 local/nonlocal states. Inputs remain labeled aged/diagnostic-compatible, not certified equilibrium. Full records:full-h-final-four-cases.json. Coloured CPU fixture generation passed36cases through1M and100ktrace each; GPU, sanitizers, stationarity and performance remain unqualified, never physical-local.
- Public gate/check_all.py now has one--all packet, A100/actualnativeflags and optionalracecheck-only bound. CLI construction plus the actual Modal validator function produced/accepted23commands, including3CPU preparations and12sanitizer calls, then heldout; source5e323f37ca6cf5e31f254b32ce69baf97e09a45dd40a2cf61ec5624b1c6afa4e, GPU6360s/CPU2250s envelope. No checks were removed to fit7200s. This is packet/admission evidence, not another executed GPU ladder. Independent full-H audit remains a separate documented tool.
- Production frontend validation now matches the restored proven native ABI: W1/32/64/128/256, T32/64/128/256, PPT1, W<=T. The earlier wide-CPU experiment is unretained; wider CUDA research remains on its idea branches. Final narrowed CPU producer completed4jobs and matched all frozen final-state oracles.
- Genuine no-hardlinks clone0195700 compiled retained C08 forsm80/sm89/sm90 and bothCPU executables; inclusive clone+build29.86s. Clone CPU audit passed34398cases; all4recovery regressions passed. Artifactclean-clone-final-evidence.json. Local973d... Make hash is not the remote controller digest or remote binary hash. Subsequent changes are evidence/catalog/documentation, not the native engine.
- Twenty-two measured experiments now have individual local commits/branches; separate unqualified branches preserve O17/C04/C06/C09/C10/register caps/coloured work, in addition to earlierC05/C07/C08/C11/C12 archives. Experiments.json now points to actual files/current evidence rather than intended absent stubs. Old proofs are preimplementation arguments; actual execution scope is in configurations, immutable receipts and REPORT.
- Own final Appap-uAhETW9QWtywkjLwKuQEEZ was authoritatively verifiedAPP_STATE_STOPPED with emptyTaskList at08:37:10Z. No idle final GPU container and no pushes. H100/L4 runtime remain unmeasured;1G/chain and beta3.4 equilibration are not achieved claims. Next research priorities require a new authorized budget, not a silent override of this ledger.

## Renewed five-point follow-up: bounded authorization

- User explicitly renewed execution of profiling, sparse repair, acceptance-regime windows, campaign shutdown and conditional H100. Decision: at most0.90USD additional committed work, not another15USD allowance. Every old reservation/charge remains unchanged at13.996047228003794USD; cumulative operational ceiling14.896047228003795USD, original hard ceiling15USD. Compute deadline13:00Z,2026-11-11. New development share0.70USD, publication/final reserve0.20USD.
- Saved prior ledger with canonical SHA inartifacts/budget-before-five-point-followup.json. Current ledger records exact user request, prior limits/deadline and inherited stop records. The prior L40S stop remains in history; renewed work has a separate1200s total admitted lifecycle ceiling. A100 also1200s, H100450s only if justified, unavailableL4zero. New stop records still bind. Neither a new file nor a deleted ledger resets the allowance.
- Actual local financial smoke passed six cases: extra-cap rejection, L4 refusal, card-envelope refusal, newly imposed card-stop refusal, positive local-only accounting preserving every prior row, and missing-ledger refusal. No Modal allocation was used by this smoke. Artifactfollowup-budget-guard-smoke.json.
- Three independent changes prepared in parallel: current-winner Nsight controls (no native bytes changed), compact sparse operating specialization (C13, unqualified), and adaptive campaign scheduling in an isolated worktree. C13's82,424B L21dynamic shared size is the same capacity already present in C11's small-window branch; it is NOT a new8x saving or measured gain. It additionally fixesPPT1 and omits unused wide instantiations. Existing broad C11 remains archived/available.
- Run the current frozen C07 profile before integrating any driver changes, preserving its exact original proof identity. Qualification always precedes ordinary timing. No H100 purchase merely for headline FLOPS; no blind time-envelope shrinking and no new orchestration layer.

## Follow-up completed: measured limits, no new throughput winner

- Actual C07 L40S profile receipt53dc7b2cdac54ff3af1e75f9bd9c6cb8: W256/T256,1and142chains, both physical moves,1024timed after8192warmup. Source8020c79... matches the frozen pre-driver-change native revision c0d8e69. Fifteen of17requested metrics available. Issue-active27–30%, barrier-stall20–25%, zero measured local sectors,14.7–17.7dynamic warp instructions/proposal. Stall percentages are not wall-time/Amdahl fractions; no claim that removing barriers gives5x. Artifactfollowup-C07-profile-summary.json.
- C13 isolates an already existing C11 small-window capacity plus fixedPPT1; it was locally compiled, not GPU-qualified. Branchopt/solid-opt3-C13-unqualified at8f976a9. C14 adds lazy incidence construction only when a first accepted writer exists and omits earlier finalized nodes. Full normal selected L21SASS sm89/sm90 has zeroSTACK/LOCAL/LDL/STL/MUFU; actual registers40/42 for the physical modes. No sm80C14 or H100runtime claim.
- C14 actual receipt16ecda16b46e4655a4a0cebc4f59ad2d, native429d2c2034ea4bd629a51eae015e812a33bb827ae413f6078a9f314e075cc641: selected W32/T32,W32/T256,W256/T256 pass normalmillion/full100ktrace/multi/saturation, all3sanitizers both moves, and fresh100M both moves/all3shapes before timing. New heldout seed3483929462277233793/chain1000002 was prepared once after source freeze. Frozen traces prove at least349constructed lazy epochs, crossing rollover. No fixture, seed, tolerance or CPU law was changed.
- Sparse engine is slower: W256single137.650298Mlocal/165.647879MNL; at142chains116.167271Mlocal/158.301232MNL. C07remains169.671703M/207.955519Msingle and142.068618M/197.339042Mat142. C14notkeptasspeedwinner; local branchopt/solid-opt3-C14-qualified-negative at a8743bb. The first beta.01W32local128warm/128timed command exceeded10s; wholepacketfailed, no accepted rate for that command, fullGPUliability.50032836USDretained.
- Subsequent exact-source beta.01/1.5single-chain16warm+16timed four-repeat experiments completed: W256wins both laws/alltestedpairs, with identical aged fingerprints and acceptance counts within each pair. Beta.01W256~1.099Mlocal/1.069MNL; beta1.5~24.687M/27.487M. Do not compare their differently aged states to the8192-warm deep-beta workload as a causal beta-only effect. No adaptive-window policy kept. Artifactfollowup-acceptance-window-summary.json.
- Final paid receipt801576c74a0d452289c28cc178d8fdc7 succeeded: a30srequestedpilot, tenbetaslinspace(.01,2,10), bothlaws,6chains/cell,20cells/120chains, intentionally forced4096upperchunks. Adaptive native code completed all20cells asdeadline in27.603194546swall, command28.332s. Proper native120/worker180bounds retained; this is NOT a600scampaign. All80retained actualprobes passed frozenCPUreplay; all20terminalrawstates/seed/cursor/energy/counts match finalprobe andobservations. Gaps are not replayed; no equilibration or matched campaign-speed claim.
- Actual data exposed checkpoint_pair preferring old immutable993 over completed1077. Fixed to order terminal-native and durable candidates by counters, verify terminal markers plus final-probe hash/seed/cursor, and retain immutable fallback. TwelveCPUregressionspass. Correctedactual20cellresumeplan isartifacts/followup-shutdown-pilot/resume-plan-terminal; oldresume-plan issuperseded. NewplanwasnotGPUexecuted. Terminalproof and80fileCPUverification are alongside it.
- Current qualified C14/default16384production plan generated withoutcompute atartifacts/production16384-plan-followup. Updatedproofs/PRODUCTION_PLAN.md usescurrentnative429dandactualpassingcommands; olderC07/C08proofs cannot attach tonewdriverhash. The historicalspeedwinner remains available on its frozen revisions rather than falsely inheriting qualification.
- Conservative follow-up commitment0.849390955694USD, aggregate14.845438183698USD, not aninvoice. Initial.90followup admission cap was explicitly revised to.96only for thelast~.22002upper-envelopecall; original15USDhard cap neverraised. FailedC14reconcile hit providerFunctionCallGetInforesource-limit; no refundcredited. Every originalrecordretained. Completed lifecycle accounting releases only the existing successful-call charged bound, never speculative billing.
- H100purchase rejected: sparsecandidatealreadyregressed, larger shared capacityalone is not evidence of per-chain gain, and no remaining proper qualification allocation fits. All3ownedfollow-upApps authoritativelySTOPPED/emptytasks; latestap-wpIZFkzqo7uKyNB9k3xh9b verified10:45:48Z. Paid phase is nowpersistentlyclosed BEFOREimage/CPU/GPUadmission, with a localrefusalproof. No pushes, no idlecontainers, no furtherModalcompute under thisauthorization.
