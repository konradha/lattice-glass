# solid-opt3: measurement protocol and scientific limits

## Campaign facts and observed baseline

Authorized campaign window: **2026-11-10 19:02:41 UTC–2026-11-11 09:02:41 UTC**
(14 hours, extended by the user's latest instruction).
Hard Modal budget USD15; operational stop USD14. The internal planning split
was explicitly reallocated from11development/3final to12development/2final,
without reducing historical liabilities or increasing either user cap.
Costs in the ledger are conservative elapsed/reservation bounds, not invoices.
No historical campaign ledger is part of this budget.

The original L40S2h allocation was exceeded before provider-lifecycle accounting
caught it: authenticated complete-hour GPU resource cost through02:00Z is
USD3.93725580 / USD0.000542s^-1 = at least7264.309594s. This is at least64.31s
over2h, not a USD15 breach. Further L40S admission is now blocked in the locked
ledger. Unexercised final L40S criteria must remain explicitly unverified.
Latest parent-reported conservative campaign envelope is USD13.996047228003794,
including the full failed final-run liability. Two authenticated attempts to
reconcile that run were blocked by provider `FunctionCallGetInfoResourceExhausted`;
no speculative credits were applied. The earlier authenticated C05 whole-app
resource reconciliation is not an invoice or a refund inferred from absent
image records; strict older-pilot reconciliation was idempotent, not new funds.
No further GPU calls are affordable under the USD14 operational stop.
The USD15 hard cap and L40S admission stop remain unchanged.

**Single-chain and aggregate L40S and A100 throughput are measured**.
L40S has separate depth(C07) and peak-card(C02) frontiers; A100 has its own
selected, held-out-qualified C08 scaling packet. H100 and L4 rates remain
unmeasured. Scientific evidence remains limited: beta_max2.0 is admitted by
the actual partial-pilot selector, not equilibrium proof. The final A100
`--seconds600` packet **failed its630s command timeout**: retained probes and
some diagnostics passed, but clean all-cell completion is not established.
The earlier interrupted lower-beta pilot also remains incomplete.

Historical baseline successful receipt:
`artifacts/receipts/a8ba2c6573524a69aaa232add5f7e751.json`
(native rows under `.result.rows`). L21,beta3.2,one chain,seed1,
8192 warmup sweeps followed by four fixed8192-sweep repetitions from identical
aged bytes; each timed repetition attempts75866112 proposals including no-ops.
Reported rates below are four-repeat medians of normal unprofiled CUDA-event
throughput, excluding warmup, construction and transfers—not CPU-inclusive wall.

| Window / threads / PPT | Local M attempts/s | Nonlocal M attempts/s |
|---|---:|---:|
| 128 / 128 / 1 | 84.7074 | 106.8861 |
| 128 / 256 / 1 | 83.6079 | 105.0977 |
| 256 / 256 / 1 | **105.5272** | **134.4039** |

Native source SHA256:
`1ed16973b807e7b06508fc1346b9376d8a22e22b8b16a5eb39fa9cafc78d7627`;
checker binary SHA256:
`c164af6f14a1519b28be98e0fd718bc45c716c5c650bcd046ce56ebe69aae182`.
Per-move checkpoint fingerprints and acceptance counts agree across these
shapes/repetitions (local fingerprint cee9a9aa6dcf135b,234032 accepts;
nonlocal89cd770fadc48e35,123987 accepts). These are fixed-seed shape
observations, not a claimed A/B/B/A optimization promotion or equilibration.
The receipt's actual sampled SM-clock endpoints are210 MHz before and2520 MHz
after, memory9001 MHz at both; neither endpoint is an in-kernel clock trace.
Stage cycles are null because these are unprofiled observations. The parent
reports39 registers/zero local allocation for normal code on both target
architectures; this is static compiler metadata, not dynamic instructions,
and supplies no evidence for blind register caps.

Additional actual scaling receipt:
`artifacts/receipts/5861cd7f1a1840c8a12c2c99de6afa2c.json`; summary
`artifacts/baseline-scaling-summary.json` binds its receipt SHA256
`387754f45caeb9c147619f7ff2b73e1d1ce27858d385b951405b4332f6032368`.
Same native source/checker as above, L21,beta3.2,8192 warmup and four fixed8192-
sweep repeats. W256/T256/PPT1 results (M=10^6,G=10^9 attempted proposals/s):

| Chains | Local M/s/chain | Local G/s/card | Nonlocal M/s/chain | Nonlocal G/s/card |
|---:|---:|---:|---:|---:|
| 142 | 88.832670 | 12.614239 | 127.579555 | 18.116297 |
| 284 | 79.012250 | 22.439479 | 111.055150 | 31.539663 |
| 568 | 40.801794 | 23.175419 | 56.285746 | 31.970304 |

Each saturated observation used142 actual SM IDs. Higher aggregate throughput
is not single-chain latency improvement or evidence of faster relaxation.
These measurements motivate campaign concurrency; mixed-beta stream throughput
remains a separate production measurement.

Actual instrumented W128/T128 results are in
`artifacts/baseline-stage-summary.json`, receipt
`9b00d07dd0e34ab9b4d895295c38e3bd` (summary-bound receipt SHA256
`46973d21105ab5aa24b8214a006b96a24d359695ed617be7f8096e7d29d415ed`).
The following one-chain numbers are **CTA elapsed stage cycles per finalized
proposal**, including instrumentation/barriers, not dynamic instructions:

| Move | RNG | Address | Load/delta | Conflict | Commit | Round |
|---|---:|---:|---:|---:|---:|---:|
| local | 2.974 | 5.560 | 4.828 | 8.001 | 10.874 | 34.314 |
| nonlocal | 2.888 | 1.045 | 6.383 | 7.117 | 6.924 | 26.481 |

At284 chains, round intervals are37.962(local)/28.374(nonlocal) cycles/finalized
proposal. Component intervals need not sum to round. Profile throughput fields
are explicitly null. These W128 fractions are not a measured decomposition
of the faster W256 path. Actual NCU counter access subsequently succeeded:
the corrected offline parser recovered660 counters from the saved wide CSV
(`artifacts/ncu-mangled-derived.json`). Its one-sweep W128 local kernel
includes initialization/export:197156warp instructions,2102shared-load
bank conflicts,16.47% issue-active and sampled2.519GHz are not a steady-state
hot-loop roofline. The initial parser's no-counter label is not a permissions fact.

Parent-reported CPU baseline observations (not re-executed by this worker):
untouched opt2 CPU, `-O2 -march=native`, pinned core0, L21,
warmup8192 sweeps, measured50000 sweeps =463050000 attempts.
These CPU observations predate the GPU receipt above; their wall includes NPZ writing.

| Move | beta | Inclusive wall s | Observed M attempts/s |
|---|---:|---:|---:|
| local | 2.8 | 16.639 | 27.8292 |
| local | 3.2 | 16.451 | 28.1472 |
| nonlocal | 2.8 | 14.321 | 32.3336 |
| nonlocal | 3.2 | 14.620 | 31.6724 |

Separate actual **optional standalone packed-cache CPU candidates**, not a
replacement for the frozen CPU oracle: parent-built `-O3 -march=native
-mtune=native`, pinned core1. The parent reports469 oracle-check JSONL records,
including L4/8/21,10^4/10^5/10^6 and full64-bit wrap cases; the actual terminal
record in `artifact://753` is `cpu_fast_check_complete,ok:true` for source
`174744a4b99b8e6529ad98fc87dcdb9c83f225424e5acf0edf4113dde833c4d7`.
Original authoritative CPU headers remain unchanged.

Actual benchmark rows are the first four records of `artifact://758`.
L21,beta3.2,chain0,seed15637821895788547078,8192 warmup sweeps; each
timed interval requests10 s and finishes its current sweep:

| Cache | Local M attempts/s | Nonlocal M attempts/s |
|---|---:|---:|
| S | 43.98434688 | 45.88658372 |
| K | 45.27468923 | 46.85901513 |

Timer scope is kernel-only `steady_clock` including sweep timer checks,
excluding initialization/thresholds/warmup/export/NPZ. S/K share per-move
incoming aged fingerprints but their time-limited attempted work differs
(local47495/48888 sweeps; nonlocal49549/50599). These are **not** fixed-work
A/B/B/A speedups, and neither the O2 CPU wall+NPZ baseline nor seed1 GPU
benchmarks provide a fully matched comparator. CPU clock/cycle fields are
null. The historical seed above is development evidence, not a new held-out
final key.

Updated CPU comparator: `artifacts/cpu-type-fast-measured.json`, source
`3e26cc3be810ccad8fefbc34df3a5c7214f19464f2b3b404d5b698d75b34c394`.
The parent's full frozen-oracle matrix ends with `ok:true` in `artifact://869`;
the oracle headers remain unchanged. Same O3/native/core1, L21,beta3.2,
historical development seed,8192 warmup and10s kernel-only method:

| Type-fast cache | Local M attempts/s | Nonlocal M attempts/s |
|---|---:|---:|
| S | 43.043221 | 44.093244 |
| K | **47.899628** | **51.113403** |

K is the current measured CPU comparator, not the slower historical O2
wall+NPZ baseline. S regressed in this actual run; no frequency measurement
is available to assign a cause. These time-based CPU runs still have different
timed work from fixed-work GPU repeats and use a different development seed.
Any contextual GPU/CPU comparison must retain those limitations, not claim a
fixed-seed A/B/B/A speedup.

Parent-reported trial decisions: O14/O13/O07 lost; O19 was near-neutral
(approximately0.4% nonlocal) and was not promoted. No broader claim is
substituted for the actual correctness/performance receipts. The initial
chunk8192 pilot failed native parameter validation before kernel execution;
the corrected4096,180s baseline pilot subsequently ran, as documented below.

### Selected-W256 candidate evidence

Each candidate below passed its selected normal W256/T256/PPT1 smoke,
single/full-trace, multi and saturated checks plus memcheck/racecheck/synccheck
for both moves before its four-repeat benchmarks. This is **selected packet
scope**, not proof for other shapes, profile paths, final held-out keys or
universal production promotion. Exact rows, scope and proof hashes are in
`artifacts/O03-ranking-summary.json`, O04/O05/O06/O08 equivalents.
Rates are median M attempted proposals/s **per chain**:

| Idea | Local1 chain | Nonlocal1 chain | Local284 chains | Nonlocal284 chains |
|---|---:|---:|---:|---:|
| O03 type commit | **134.821645** | **163.616129** | **105.044138** | **133.582996** |
| O06 local addressing | 116.103376 | 134.416588 | 85.705857 | 111.056644 |
| O05 field conflicts | 106.806380 | 137.778165 | 80.106046 | 114.431180 |
| O04 fixed commit | 106.388511 | 133.623104 | 79.157567 | 110.807390 |
| O08 threshold | 104.293986 | 133.105540 | 78.085305 | 110.154598 |

O03 was the first large measured win (subsequently exceeded by C03): relative to the
recorded W256 baseline medians, about27.8% local and21.7% nonlocal at one chain.
At284 chains its actual card rates are29.832535G/s local and37.937571G/s nonlocal.
Its receipt is `f9b4b1c41f924217be85b9bdea90a08e`, source SHA256
`e80b6d90e195f242bb1ad10d3afd0156163b1dc362f97e43e8424683a5ee278c`.
O06's local gain is about10.0% single-chain/8.47% at284, with nonlocal neutral.
O05's nonlocal gains are about2.51%/3.04%; O04 is near-neutral and O08 loses.
These are baseline-relative ranking observations, not a claim that A/B/B/A
ordering has been established from consecutive four-repeat candidate packets.

Other actual receipt IDs: O06 `d8c8685709b74ea68128fcec66670f1b`;
O05 `81f1579dcf584e8799af90e90954bd26`;
O04 `5ea0a8b9e15049ecbd385694df43fe64`;
O08 `7257774a8af44656b25e4d7a9021f93f`.
Their summaries bind each source independently. Neither single-chain wins
nor higher aggregate rates establish thermodynamic relaxation.

### Current selected L40S depth frontier: C07

C07 composes the already measured fast type/address/RNG/field mechanisms
with the exact no-accepted-writer window shortcut. Receipt
`bcb0075d40664e588f9435232fc4653d` passed unchanged selected W256/T256/PPT1
smoke/single/full-trace/multi/saturation and all3 sanitizers for both moves
before four-repeat ordinary CUDA-event timing. No untested profile, shape,
held-out or cross-card promotion is inferred.

| Chains | Local M/s/chain | Local G/s/card | Nonlocal M/s/chain | Nonlocal G/s/card |
|---:|---:|---:|---:|---:|
| 1 | 169.671703 | 0.169672 | 207.955519 | 0.207956 |
| 142 | 142.068618 | 20.173744 | 197.339042 | 28.022144 |

Exact source digest:
`8020c79afcbb7af5d82234c89a6e458fa95d5b43493fe0ff5c3a9774636481e8`.
The preceding call364739... only prepared the CPU cache; GPU admission was
refused. Successful retry changed deadlines from observed runtimes, not
fixtures, check counts, tolerances, seeds or physics. Summary:
`artifacts/C07-unitSM-summary.json`.

C07 is NVIDIA L40S/142SM evidence at L21,beta3.2,seed1,8192 warmup sweeps
and four fixed8192-sweep repetitions from identical aged bytes. Ordinary
CUDA-event timing excludes initialization, warmup and transfers. Its one-chain
local/nonlocal medians are1.608×/1.547× the recorded baseline W256 medians;
these are descriptive packet-median ratios, not observed A/B/B/A ordering.
Against the equally clever CPU K+type-fast comparator they are3.542×/4.069×,
but the CPU's O3/native/core1,10s kernel timer and different seed/work make
those contextual ratios, not matched algorithm speedups. The frozen O2
wall+NPZ baseline is not substituted as the current CPU comparator.


### Earlier C03/C02 Pareto measurements

`artifacts/C03-ranking-summary.json` and `artifacts/C02-ranking-summary.json`
record all selected normal gates and three sanitizers passing before timing.
Both are W256/T256/PPT1 evidence only, with no untested-shape/profile/final-key
promotion inferred. C03 was the earlier depth frontier, now exceeded by C07;
packed+type C02 still leads the measured peak aggregate card rate.
Rates below are four-repeat medians, with per-chain and card units separate:

| Candidate | Chains | Local M/s/chain | Local G/s/card | Nonlocal M/s/chain | Nonlocal G/s/card |
|---|---:|---:|---:|---:|---:|
| C03 | 1 | **155.065943** | 0.155066 | **171.019242** | 0.171019 |
| C03 | 142 | 137.964689 | 19.590986 | 166.715802 | 23.673644 |
| C03 | 284 | 118.887271 | 33.763985 | 140.516187 | 39.906597 |
| C03 | 568 | 60.579625 | 34.409227 | 70.669706 | 40.140393 |
| C02 | 568 | 70.349729 | 39.958646 | 82.138201 | 46.654498 |
| C02 | 1136 | 35.949849 | **40.839028** | 42.586952 | **48.378777** |

C03 receipt `e397495ef48843499682e1708b8a0b42`, source
`1708ac11977dcafa3604d556bf130b6299ec0ff65e10ca313e7d889e9deb0c43`;
C02 receipt `562e53f4ccd74420b5b08ae2fa14fc5b`, source
`58e15c455fcbef643682e68f3b43e631a26be8e0ad7e17ab91d1badd8dc91303`.
The1136-chain C02 points lack a matching baseline1136-chain observation;
no absent baseline ratio is invented. Greater card throughput at1136 comes
with lower per-chain throughput than568 and does not improve single-chain
relaxation depth.

C01's selected W512 type-commit path also passed but is slower than the W256
type frontiers: single-chain local126.358550/nonlocal158.357263M/s;
284-chain local77.083621/nonlocal102.237798M/s/chain.
Receipt `080f205405ee48e2a4dd4b1660f17d38`,
`artifacts/C01-ranking-summary.json`. All measured O10-wide baseline larger-
thread shapes lost to the baseline W256 reference; the complete selected
wide sanitizer coverage passed in receipt `69a3608c55124c9fb7808e0af0bf694b`
(`artifacts/O10-wide-ranking-summary.json`), including33 racecheck checks per
move taking322.596s local/348.325s nonlocal. Those are sanitizer durations,
not throughput. Virtual PPT2/4 were not GPU-admitted at that earlier update.
The later actual C05 wide/PPT4 A100 attempt below is retained separately.

### Actual A100 selected frontier and incomplete wide timing

`artifacts/C08-A100-final-summary.json`, receipt
`0e81177be78643e1a82985e7e60dc8d4`, measures **NVIDIA A100-SXM4-40GB,
108SM**, backend `cuda/packed_fast_empty.cu`, selected W256/T256/PPT1.
Source/build digest:
`5e323f37ca6cf5e31f254b32ce69baf97e09a45dd40a2cf61ec5624b1c6afa4e`;
checker SHA256:
`d5962f8e723e4223f4eefa2a7d972398813cd7725c37941acbcd8d50e0ac6e27`.
The digest includes `EXTRA_NVCCFLAGS=-gencode arch=compute_80,code=sm_80`;
it is not the L40S source/build identity. Selected normal smoke,
single/full-trace, multi and saturation gates, memcheck/racecheck/synccheck
for both moves, and fresh reserved held-out10^8-proposal checks for each move
all passed before timing. This does not qualify other launch shapes,
profile paths, hardware, or scientific equilibration.

Rates are four-repeat medians, L21,beta3.2,seed1,8192 warmup and fixed8192
timed sweeps/repetition (75866112 attempts/chain), identical incoming aged
bytes within each configuration. Timing is normal unprofiled CUDA events,
excluding construction, warmup and transfers—not inclusive production wall.

| Chains | Local M/s/chain | Local G/s/card | Nonlocal M/s/chain | Nonlocal G/s/card |
|---:|---:|---:|---:|---:|
| 1 | 95.556175 | 0.095556 | 116.816217 | 0.116816 |
| 8 | 85.846427 | 0.686771 | 113.359278 | 0.906874 |
| 64 | 80.125281 | 5.128018 | 110.906031 | 7.097986 |
| 108 | 80.124414 | 8.653437 | 110.903290 | 11.977555 |
| 216 | 68.359918 | 14.765742 | 91.893064 | 19.848902 |
| 432 | 47.277181 | **20.423742** | 59.511374 | 25.708913 |
| 648 | 29.807997 | 19.315582 | 40.383522 | **26.168522** |

Local peaks at432 measured chains and then declines; nonlocal peaks at648,
but432 already supplies98.24% of that observed peak at greater per-chain
depth. Neither point is a measurement at16384 simultaneous chains.
The sampled SM-clock endpoints are210 MHz before/300 MHz after, memory1215
MHz at both;1410 MHz is the reported nominal/max SM clock, not a measured
in-kernel clock. Registers, shared bytes and occupancy remain null in this
summary. There is no same-card A100 baseline ablation, so L40S-versus-A100
rates confound card and backend/build and are not algorithmic speedups.

The earlier **C05 A100 packet was not a successful timing packet**:
receipt `9b0e6759bc7047f9af8a27785f901963`, backend
`cuda/fast_wide_repair.cu`, W2048/T512/PPT4, digest
`331e1272b8dfcdcae589d133a8512bd439d660a17264452828baf8c3b54fbdcc`
(also includes the extra sm80 flag). All its selected normal/trace/multi/
saturation, three sanitizers and held-out checks for both moves passed.
The first local timing command expired at its12s deadline after two complete
of four requested repeats (retained rates35.269723/35.269955M/s); truncated
stdout is not a completed extra repetition. There is **no accepted four-repeat
C05 rate**, no promotion and no completed later scaling/nonlocal timing.
This was a timing-command timeout, not a CUDA correctness or sanitizer failure.

The current measured Pareto choices therefore remain source/card-specific:
L40S C07 for one-chain and142-chain depth, L40S C02 for peak card throughput
(1136 chains,40.839028G local/48.378777G nonlocal), and A100 C08 for the
separately qualified A100 curve above. C03 remains useful intermediate scaling
evidence, not the current depth winner. H100 and L4 have **no accepted actual
rate** in this update; no extrapolated hardware rating is presented as measured.

### Retained optimization hypotheses versus measurements

Qualitative motivations below are hypotheses, not reconstructed numerical
predictions. The corresponding `results.tsv` records still label numerical
`predicted_gain` as `unmeasured baseline`, leave `measured_gain` empty and mark
`kept=not-promoted`; those bookkeeping fields must not override actual retained
summary rates or be interpreted as a final production promotion.
All rows below are actual selected W256/T256/PPT1 L40S packets with normal
gates/all three sanitizers before timing; ratios are against the recorded
baseline, not matched A/B/B/A effects.

| Idea / qualitative hypothesis | Measured local/nonlocal ratio,1 chain | Measured local/nonlocal ratio,142 chains | Observation |
|---|---:|---:|---|
| O09, skip exact windows with no accepted writer | 1.081 / 1.227 | 1.027 / 1.193 | Useful, especially nonlocal |
| O09b, accepted-writer metadata reduces conflict work | 0.996 / 0.998 | 0.997 / 0.999 | Near-neutral/slower |
| O15, epoch claims reduce conflict bookkeeping | 0.645 / 0.716 | 0.614 / 0.693 | Clear regression |
| O16, pipeline RNG work | 1.004 / 0.956 | 1.021 / 0.961 | Small local gain, nonlocal regression |
| O18b, shared thresholds reduce lookup cost | 1.000 / 1.002 | 1.000 / 1.002 | Near-neutral |
| O06b, coordinate table reduces address arithmetic | 0.972 / 0.956 | 1.000 / 0.957 | No general win |
| C07, compose fast mechanisms with exact empty-window shortcut | 1.608 / 1.547 | 1.599 / 1.547 | Current measured L40S depth frontier |

Sources and immutable receipt IDs are bound by
`artifacts/{O09,O09b,O15,O16,O18b,O06b,C07}-unitSM-summary.json`.
The type-commit O03 win and packed+type C02 aggregate win above show that
compositions can help, but gains do not add or multiply predictably: C07 was
measured, not assigned the product of isolated gains. Retained failed/neutral
trials, C05's incomplete timing and unexecuted hardware/prototypes are not
silently converted into successful optimization evidence.

Later **local compilation/static inspection only** retained C11 and C12
objects targeting sm80/sm89/sm90. Each targeted summary contains nine selected
normal L21 kernels (three move modes on three architectures), all with
`LOCAL=0,STACK=0,LDL=0,STL=0`:
`artifacts/sass-C11-minblock-targeted/summary.json`, object SHA256
`4dd186e9646a8260eed08310993baaa9b23fd2c924b446697134031e7646b90a`;
`artifacts/sass-C12-targeted/summary.json`, object SHA256
`7ade01d625c53eb05520d34ee9101fa429eb98c13f27e83cd3af8d17625f3ab1`.
The launch-bound change is `__launch_bounds__(MaxThreads,1)`, not a demonstrated
GPU speedup. Forceinline alone had **not** eliminated sm80 mode2 accesses:
`artifacts/sass-C11-inline-targeted/summary.json` records STACK16,10 static
LDL sites and4 static STL sites despite LOCAL0. Thus LOCAL0 alone was not
proof of no stack traffic. These are static object/resource observations,
not executed dynamic instruction counts, throughput or cross-card promotion.
The subsequent actual C11 A100 packet, receipt
`1ba190f801a84045b3bdce92605c20c9`, digest
`b7d5e0ee3886758bae5183188f8bfa8c0e3bd06dab0fc8db0989df41bcd9cfa3`,
passed selected normal smoke/single/full-trace/multi/saturation and both
memchecks. Local racecheck then exceeded45s and stopped the packet.
There is **no complete racecheck/synccheck/held-out qualification or timing
result**, and no five-point C11 rate is accepted. A timeout is not a proven
race or a passing sanitizer. C12 remains compiled/static-only.
The unretained C11 branch `opt/solid-opt3-C11-unretained` at `3c6e1d3`
and unqualified C12 branch `opt/solid-opt3-C12-unqualified` at `0f72bc7`
preserve those experiments, not promoted backends. The measured tables
and conditional costs remain unchanged; sm90 compilation does not make
H100 measured.

### Conditional kernel-only workload costs

These are **arithmetic projections, not executed campaigns or invoices**,
using measured L21,beta3.2 card peaks and GPU-only prices L40S
USD0.000542/s and A100 USD0.000583/s. For one separate move:

```
N(S) = 16384 chains * S sweeps/chain * 9261 attempts/sweep
     = 151732224 * S attempts
seconds = N(S) / (measured card G attempts/s * 10^9)
GPU USD = seconds * GPU USD/s
```

N is1.51732224×10^13,1.51732224×10^14 and1.51732224×10^15 attempts for
S=10^5,10^6,10^7 respectively. Each cell is **seconds / GPU USD**;
rounding uses the displayed peak rates40.839028/48.378777G/s for L40S C02
and20.423742/26.168522G/s for A100 C08.

| Card / backend / move / measured peak chains | S=10^5 | S=10^6 | S=10^7 |
|---|---:|---:|---:|
| L40S / C02 / local /1136 | 371.54 /0.2014 | 3715.37 /2.0137 | 37153.73 /20.1373 |
| L40S / C02 / nonlocal /1136 | 313.63 /0.1700 | 3136.34 /1.6999 | 31363.39 /16.9990 |
| A100 / C08 / local /432 | 742.92 /0.4331 | 7429.21 /4.3312 | 74292.08 /43.3123 |
| A100 / C08 / nonlocal /648 | 579.83 /0.3380 | 5798.27 /3.3804 | 57982.72 /33.8039 |

These assume the measured homogeneous-beta peak can be sustained over batched
work, without tails or rate changes. They are not measured16384-chain scaling
and not a universal beta-ladder cost guarantee. For both separate moves with
the same16384 chains and S each, add the two projected times/costs; do not
apply either move's rate to the other. CPU/control/image builds, initialization,
warmup, transfers, checkpoint publication, output, idle billed GPU intervals,
scientific analysis and other service overhead are excluded. Actual ledger
reservations/provider lifecycle charges remain separate, with no implied
budget credits or authorization to resume L40S admission.

Depth is a different constraint. A **single physical-local chain** advanced
10^9 sweeps needs9.261×10^12 attempts. Using each card's measured one-chain
local rate, not its card aggregate, gives:

| Card / backend | Local M attempts/s/chain | Projected kernel seconds | Hours | GPU-only USD |
|---|---:|---:|---:|---:|
| L40S / C07 | 169.671703 | 54581.88 | 15.162 | 29.5834 |
| A100 / C08 | 95.556175 | 96916.81 | 26.921 | 56.5025 |

Finishing that chain in2h requires
`10^9 * 9261 / 7200 = 1.28625×10^9 attempts/s/chain`, about7.58× the
measured L40S depth rate or13.46× A100. Concurrency raises card throughput
but does not divide one exact chain's sequential trajectory among chains.
Attempted proposals include rejects/no-ops; neither proposals nor sweeps are
independent samples. Effective independent sample cost requires actual
move/beta/observable-specific relaxation evidence, not these kernel rates.

### Actual first pilot: no selected beta_max

Baseline receipt `05aca736725b456ebd2f8961c8fa1fad` requested180s on L40S,
using chunk4096, eight independent chains at each beta2.5/2.8/3.2/3.4 for
both separate moves. Its actual command elapsed182.174s, including command
setup/output; worker elapsed187.056s is a different envelope. No sum of
overlapping job CUDA times is substituted for either wall measurement.

Downloaded evidence is under `artifacts/pilot-baseline-4096/`.
`verification.json` reports `ok:true` for all47 actual retained short probe
files across the eight jobs, replayed on CPU from actual incoming bytes.
Explicit unreplayed production gaps remain recorded; this is not a claim
that all intervening proposals were CPU-replayed.

`diagnostics.json` gives **inconclusive for all eight jobs**. Their primary
group was the predeclared timestamp-selected h10000 group: waits100000,
200000,300000 versus10000,20000,30000, at the fixed eight lags1000–4000
and10000–40000 sweeps. Every job's energy block bands fit the declared
energy-equivalence margin, yet all primary groups failed waiting-time
compatibility and demonstrable late overlap decay. Thus energy flattening
alone would have produced a scientifically unsupported conclusion.

Nonlocal beta2.5 has observed mean C=e^-1 sampling brackets4000–10000 sweeps,
but these are **not equilibrium tau estimates** while aging/decay requirements
fail. Other nonlocal betas and all local jobs have no observed crossing
through the sampled40000-sweep lag limit. No fit extrapolation rescues them.
`beta-selection.json` correctly records
`status=inconclusive,beta_max:null,equilibrium_established:false`,
with selection move nonlocal and local thermalization explicitly unestablished.

The follow-on lower-beta1.5/2/2.5/2.8 pilot requested600s but **did not complete**:
receipt `98730c92bf9a400b8b6d0737fc9fe734` failed with
`checkpoint-publication-failed` after169.096s command time
(174.223s worker envelope). The parent identified the periodic Volume
commit's8s timeout: an infrastructure/publication failure, not an observed
Monte Carlo mismatch. It is not a successful600s campaign.

Actual recovered files remain under `artifacts/pilot-low-600/`:
`partial-verification.json` passes all16 retained probe files, explicitly
not the un-replayed gaps. `partial-diagnostics.json` reports nonlocal beta1.5
and2.0 compatible with stationary relaxation under the predeclared diagnostic,
**not equilibrium proof**. Other nonlocal points and every local point remain
inconclusive. These are partial saved-window observations, not600s results or
a newly declared final beta_max. No criteria, tolerances or seeds were changed.

The parent bounded publication commit/join handling at60s; subsequent actual
45s publication smoke `f2fe53a598a548e5a6484a1e432b3771` passed with three
recorded commits (46.247s command,71.961s worker envelope). This demonstrates
that smoke's publication path only; it does not retroactively complete the
interrupted600s pilot, establish a long-run publication guarantee, or certify
equilibrium. The analysis worker did not execute these runs or alter campaign
code/criteria.

The earlier actual authoritative CPU nonlocal controls at beta2.0/2.5/2.8/3.2,
eight independent chains each through10^6 sweeps, are **all inconclusive**
in `artifacts/cpu-relaxation-nonlocal-b{2.0,2.5,2.8,3.2}-c8/diagnostics.json`.
All fail primary late-group aging/equivalence at the unchanged margin;
beta2.5/2.8/3.2 also lack demonstrable near-zero decay at multiple waiting
times. All retain `equilibrium_established=false`. Acquisition/continuation
agreement with the unmodified CPU driver is separately recorded in
`artifacts/cpu-relaxation-acquisition-check.json`, not equilibrium evidence.
Subsequently completed32-chain,10^6-sweep CPU-reference controls use the same
fixed saved-time schedule, diagnostic policy and unchanged margins. Their
actual `diagnostics.json` files under
`artifacts/cpu-relaxation-{nonlocal-b1.5,nonlocal-b2.0,nonlocal-b2.5,local-b2.0,local-b2.5}-c32/`
report:

| CPU move / beta | Chains | Diagnostic verdict | Primary late-group observed C=e^-1 bracket, sweeps |
|---|---:|---|---|
| nonlocal /1.5 | 32 | compatible with stationary relaxation, not equilibrium proof | [0,1000] at all three waits |
| nonlocal /2.0 | 32 | compatible with stationary relaxation, not equilibrium proof | [0,1000] at all three waits |
| nonlocal /2.5 | 32 | inconclusive: no demonstrated near-zero late decay | [4000,10000] at all three waits |
| physical-local /2.0 | 32 | compatible with stationary relaxation, not equilibrium proof | [3000,4000] at all three waits |
| physical-local /2.5 | 32 | inconclusive: late aging/equivalence and decay fail | right-censored beyond40000 at all three waits |

These are the timestamp-selected h10000 primary groups, with waits100000,
200000,300000 sweeps; brackets are sampling intervals, not fitted precise
taus or equilibrium certificates. All five still explicitly have
`equilibrium_established=false`. The original eight nonlocal beta2.0 **and
beta2.5** chains are byte-identical subsets of their respective32, not eight
additional independent chains: neither group can be pooled as40 chains.
`artifacts/cpu-relaxation-expanded-prefix-check.json` proves equality of all
common physical observations and218 saved snapshots for both betas; each
expansion supplies24 additional independent chains. The larger beta2.0
compatible verdict does not erase its earlier eight-chain inconclusive result.

Nonlocal beta1.5/2.0 `provenance.json` records9545.171/9543.354s summed original
CPU subprocess wall respectively, including process overhead; the parent
reports whole-job envelopes approximately9548–9551s. These scopes differ,
and neither is a GPU rate or a completed600s GPU campaign. Nonlocal beta2.5
completed its1Msweep/32-chain acquisition in a parent-reported9512.06s
whole-job envelope (provenance:9506.748s summed CPU subprocess wall).
Its unchanged diagnostic passes waiting-time compatibility but fails
demonstrable late decay; the observed [4000,10000] crossings are **not
equilibrium tau estimates**. Neither CPU controls nor CUDA qualification
retroactively turn the169s interrupted pilot into600s production.

### Actual beta admission and failed final A100 campaign

`artifacts/final-beta-selection.json` now records
`status=selected_from_actual_pilots,beta_max=2.0`, using the actual retained
`artifacts/pilot-low-600/partial-diagnostics.json` and unchanged nonlocal
selection rule. It accepts contiguous tested beta1.5/2.0 and rejects2.5/2.8.
At beta2.0 its observed upper crossing bracket1000 sweeps divided by the
actual pilot stream rate2532.250197 sweeps/s/chain gives0.394906s projected
wall, below600s. This is a pilot-rate admission projection, not measured
final-concurrency latency, a universal tau multiplier or equilibrium proof.
`equilibrium_established=false` and
`local_thermalization_established=false` remain explicit; the separate CPU
local beta2.0 compatibility does not certify all local ladder points.

The parent launched an actual **600s native final production request at
07:10 UTC**, on A100 using C08's exact native digest
`5e323f37ca6cf5e31f254b32ce69baf97e09a45dd40a2cf61ec5624b1c6afa4e`
and existing selected normal/sanitizer/held-out proof
`0e81177be78643e1a82985e7e60dc8d4`. Only the driver guard was restored;
the unretained wide experiments are not substituted into that identity.
`artifacts/final-campaign-600/bundle-plan.json` specifies L21, both separate
moves, ten betas exactly `linspace(.01,2,10)`,21 chains per beta/move cell
(420 total), W256/T256/PPT1 and master seed4123495003391909354.
Allocation uses the actual432-chain C08 scaling points; all20 beta/move
jobs run concurrently. It is not a measured mixed-stream scaling curve.
The same allocation ran a fresh/restarted four-job producer probe before
the separate native600s production command.

**Final actual packet status: FAILED**, receipt
`02fd2c40c8b14c0bb72d41032e0a5628`. The producer commands succeeded in
4.197s fresh/11.570s resumed. Their separate actual full CPU replay proof,
`artifacts/producer-gpu-final-replay-check.json`, is `ok:true` for all four
jobs/both moves and12 chain-sweeps, comparing every checkpoint byte,
energy and acceptance count to the monolithic frozen CPU. That proof
explicitly does not claim the production command succeeded.

The native `--seconds600` command exceeded the **parent-chosen630s command
bound**, returning a TimeoutExpired wrapper after630.103s. The base campaign
plan's normal native bound was690s; only this budget-constrained bundle
overrode it to630s. This was an orchestration/time-envelope miss, not a
demonstrated physical-law or byte-trajectory mismatch. The downloaded root
`production/manifest.tsv` has18 cells marked `deadline`, with per-cell walls
603.496–622.578s, and two still marked `running`: local-b00 at564.382s and
local-b06 at584.832s. There is **no production/run.tsv** and no clean all-cell
600s shutdown receipt. Eighteen deadline statuses do not make the enclosing
failed packet a successful campaign.

Actual selected retained production observations (21 chains per cell):

| Move / beta | Manifest next sweep | Native kernel/event seconds | Per-cell wall seconds | Retained status |
|---|---:|---:|---:|---|
| physical-local /.01 | 9454 | 241.026455 | 564.381884 | running |
| physical-local /2.0 | 18520 | 5.760337 | 603.612895 | deadline |
| nonlocal /.01 | 9386 | 282.454285 | 620.704987 | deadline |
| nonlocal /2.0 | 36089 | 39.586605 | 603.617058 | deadline |

The chunk-sizing extrapolation from measured pilot betas>=1.5 did not bound
the low-beta workload: beta.01 has very large native kernel/event intervals.
These are real mixed-stream production observations, **not** a repeatable
per-SM kernel benchmark or a universal beta throughput law. Per-cell wall
includes host I/O and concurrent work; event intervals and walls have
different scopes, overlapping cell times must not be summed as global wall,
and inclusive manifest sweep rates must not replace the fixed-work CUDA
benchmark frontier. C08's beta3.2 peak cost projections above remain
conditional, not validated ladder cost predictions.

Downloaded-data analysis nevertheless produced real limited evidence:
`artifacts/final-campaign-600/verification.json` is `ok:true` for **38 retained
four-sweep probe files across20 jobs**. Eighteen jobs have first and last
probes with explicit intervening unreplayed gaps; local-b00/local-b06 retain
only their initial four-sweep probe in this verification. Their empty
`unreplayed_intervals` arrays describe that retained probe span only, not
full replay to manifest sweeps9454/18904. Passing all retained files does not
CPU-replay the full production trajectories or prove terminal publication.

`artifacts/final-campaign-600/diagnostics.json` reports13 compatible-with-
stationary-relaxation-not-equilibrium-proof cells and7 inconclusive cells:

- Physical-local:5 compatible (beta.231111–1.115556); local-b00 andlocal-b06
  have insufficient retained diagnostic coverage, and beta1.557778/1.778889/
  2.0 fail late aging/equivalence and demonstrable decay.
- Nonlocal:8 compatible (beta.01–1.557778); beta1.778889 fails late aging/
  equivalence and decay, while beta2.0 lacks demonstrable near-zero late decay.

All20 retain `equilibrium_established=false`. The running local cells'
diagnostics end at saved sweep4 despite much later manifest progress.
Thus beta_max2.0 was honestly admitted from the earlier partial pilot, but
the retained final nonlocal beta2.0 observations do **not** validate its late
decay under this final concurrency/coverage. Replay and diagnostic commands
exiting0 are analysis completion, not clean production success, full
trajectory replay, local thermalization or an equilibrium claim.

The original O2 wall+NPZ CPU table is parent-reported, not a matched GPU
speedup comparison. No source/binary hashes or operating-clock samples
accompanied those original O2 observations; the newer packed-candidate source
hash is explicitly recorded above.

Earlier parent-reported correctness/design facts included four million full-H
CPU proposals—one million each for local/nonlocal from random input and from
58192-sweep CPU-aged input. **Aged is not a thermalization claim.**
The first GPU attempt failed an added overflow guard; the guard was corrected,
but no GPU throughput or completed full GPU gate is inferred from that fix.
All mandatory physical CUDA gates still precede performance/scientific use.

### Final independent full-H audit and exploratory colored scope

The later actual `artifacts/full-h-final-four-cases.json` is `ok:true` for
four separate L21 **arithmetic-only** audits, one million proposals each:

| Incoming state | beta | Separate move cases | Audited proposals per case |
|---|---:|---|---:|
| CPU random | 3.2 | physical-local, nonlocal | 1000000 |
| External aged raw bytes | 2.0 | physical-local, nonlocal | 1000000 |

Each case records `status=passed,arithmetic_only=true,equilibrium_claim=false`,
actual input/final/executable hashes and frozen authoritative-header hashes.
The two external aged inputs have separate provenance/byte identities; they
are not relabeled as equilibrium inputs. Independent full-H arithmetic checks
are distinct from byte-trajectory fixtures and GPU sanitizer gates. They do
**not** establish beta0 Gibbs stationarity, equilibration, independent samples
or a GPU throughput result.

The parent also reports actual exploratory **colored CPU fixture generation**
passing36 cases of10^6 proposals each, with100000 traced proposals per case;
the retained final fixture is `artifacts/colored-cpu-final-fixture.bin`.
This is colored CPU generator/fixture evidence only: no corresponding colored
GPU qualification, sanitizers, stationarity or timing is claimed. Colored
dynamics are never physical-local or a replacement for the frozen random-edge
oracle, and these fixture counts do not demonstrate equilibrium at beta3.4.

### Final local admission and reproducibility evidence

`artifacts/check-all-local-validation.json` records actual public CLI packet
construction plus the actual Modal validator function's AST validation:
`ok:true`,23 commands including3 CPU and12 sanitizer commands, all original
checks retained, C08 native digest
`5e323f37ca6cf5e31f254b32ce69baf97e09a45dd40a2cf61ec5624b1c6afa4e`,
GPU hard bound6360s and CPU hard bound2250s. The public all-check path supports
A100 with the extra sm80 NVCC flag and a racecheck-only timeout override.
This is **construction/admission proof without a Modal reservation or compute**,
not a newly executed full GPU matrix or a fix that retroactively completes
C11's racecheck or the failed600s final campaign.

Final producer admission is narrowed to the restored native launch support:
W1/32/64/128/256, T32/64/128/256, PPT1. Wide experimental archives are not
retained as admitted production launch paths. The parent's actual narrowed
CPU four-job scenario matched the frozen oracle finals; its manifest under
`artifacts/final-producer-narrow-cpu/` records all four jobs at next sweep2,
with2+1 chains for each separate move. This local producer evidence is
distinct from the actual A100 resumed-producer proof above.

`artifacts/local-idea-archive.json` preserves22 measured O/C experiments on
separate local `opt/solid-opt3-*-measured` branches, with per-entry files and
commit output. The unretained/unqualified experiments remain separately
labeled; a branch name is not promotion or added performance evidence.
No pushes were made. The genuine single-branch, `--no-hardlinks` clean clone
of commit `0195700` at `/tmp/solid-opt3-clean-final-20261111` now **passed**;
`artifacts/clean-clone-final-evidence.json` records C08 sm80/sm89/sm90 plus
CPU compilation in29.86s, the clone's CPU audit (34398 full-H cases,
14718 normal accepts,9312 uphill cases,168 adapter draws), and all four
`check_campaign.py` recovery regressions passing. This is local build/CPU
evidence, with no local GPU execution or beta0 Gibbs validation.
Its local Make source hash
`973d2920f45f4c19e98382d0d0930efa9e8f9ac9b2dc311cc7b51369188b801c`
differs from the remote native controller digest `5e323f37...` because local
build flags/Boost include path differ; no equality of local and remote binary
SHA256 identities is claimed.

Pure `campaign.resume` also successfully prepared a20-cell recovery plan/
packet from the **actual interrupted production's immutable checkpoint pairs**,
retained at `artifacts/final-campaign-600/recovered-resume-plan/`.
Its plan binds original metadata/state hashes and exact next sweep/chunk
counters, including local-b00 sweep9454 andlocal-b06 sweep18904. This recovery
packet was **not executed** and does not complete either missing terminal cell
or replay their intervening trajectories. The19 CLI registry entries now
record actual paths/status metadata, not fictional intended implementations;
metadata alone supplies no missing GPU qualification or scientific result.
These local successes add no GPU measurements or budget credits. Spend remains
USD13.996047228003794 with the full failed liability retained, and the failed
final packet with only18/20 deadline cells remains the production outcome.

Before any GPU scientific data, the parent exercised a **timestamp-only
schedule model**, not Monte Carlo data: at snapshot cap128 and endpoints
10^5/10^6/10^7 sweeps, the old grid admitted zero full three-wait groups,
while the revised grid admitted3/2/2. Latest waits were respectively
{10^4,2×10^4,3×10^4}, {10^5,2×10^5,3×10^5}, and
{10^6,2×10^6,3×10^6}, compared with their exact /10 references.
Available common-lag counts17/13/11 include the fixed eight-lag subset used
below. These are saved-time coverage facts only, not measured relaxation,
equilibration, throughput or a reason to change scientific tolerances/seeds.

The historical `solid-opt2/campaigns/summary.md` records L21, eight independent
chains at each of `linspace(.01,2,8)`, separate nonlocal/local campaigns, CPU
149.6 s and Metal 145.6 s. Its L21 Metal-local trajectory is **colored**, not the
CPU random-edge trajectory. Those historical walls are neither opt3 benchmarks
nor a matched baseline for the new physical-local implementation.
The historical CPU driver's reported wall is inclusive of NPZ writing (the
timestamp follows ZIP close), despite the older bench README's exclusion
claim. It is not directly comparable to opt3 CUDA-event-only kernel time.

Frozen opt3 dynamics are the latest opt2 CPU random-neighbor trajectory for
**both** separate moves. State uses packed bytes, fixed integer energy,
counter SplitMix/Lemire proposals and host decimal64 EFMA thresholds. Local
means uniform site plus independent uniform direction among six; nonlocal
means two independent uniform sites, including invalid same-site attempts and
no-ops. A single-mode sweep is L³ proposals per chain; rejected/no-op attempts
are counted. Never replace local with a mixed, colored or nonlocal schedule.
L21, rho=.75, site densities .30/.45, ell=3/5 and A=1 are fixed here.
Any newly proposed colored sampler is only a separately labeled exploratory
hypothesis, not the physical-local CPU oracle or a replacement for either
primary separate-move campaign. Current plans use independent authoritative
CPU initialization for local controls; nonlocal warm-starting is **not**
implemented or claimed. If added later, same-beta independent preparation,
actual incoming bytes, seed/index and preparation provenance must be recorded.

## Real receipts and a single scaling visualization

`analyze.py RECEIPT... --output-dir solid-opt3/artifacts/analysis` accepts native
JSONL `kind=benchmark` rows and enclosing Modal result/command receipts with
JSONL stdout. Failed enclosing packets/commands are excluded. Missing values
stay missing: there are no generated timings or synthesized hardware counters.

Native rows use `move,L,beta,chains,window,threads,ppt,profile,repeat,repeats,
seed,start_sweep,warmup_sweeps,sweeps_per_chain,attempts_per_chain,
aggregate_attempts,cuda_seconds,checkpoint_fnv1a64,name,sms,clock_khz,smids`.
The explicit checkpoint FNV identifier is **noncryptographic**, and not a
replacement for receipt/source/binary SHA256. Wrapper `source_digest` and
`binary_sha256.{cuda_check,driver}` bind the actual native artifacts.
The analysis preserves available registers/shared memory/occupancy metadata;
absence is not reported as zero. `clock_khz` is the nominal device property,
not a sampled operating clock. Wrapper `clocks_before/after.stdout` contains
actual nvidia-smi CSV when acquisition succeeded.

Ordinary throughput is computed only from positive measured CUDA-event
seconds and the exact attempted-work count:

- per-chain attempts/s = L³ × sweeps per chain / CUDA seconds;
- aggregate attempts/s = chains × per-chain attempts/s;
- sweep latency = CUDA seconds / sweeps per chain (single-chain latency is the
  chains=1 point, not aggregate rate divided by occupancy).

Curves retain separate moves, model/age/seed, source/binary, card, launch shape
and fixed work; repetitions report median and observed range, not a fabricated
confidence interval. `analysis.json`, `scaling.tsv` and **one** `scaling.svg`
contain both per-chain and aggregate scaling, with chains=1 included whenever
measured. Profiling rows are excluded from ordinary rates. Profile cycles are
instrumented CTA elapsed stage intervals (RNG/address/load-delta/conflict/
commit/round), including stage synchronization overhead, not dynamic
instructions/proposal. Component intervals need not sum to the round interval.

Matched comparisons require identical seed, aged checkpoint identifier,
start sweep, beta, move, chain count and attempted work, with at least two
observations per variant and actual hashes. `matched_speedups` records the
observed order and sets `abba_observed` only for actual A/B/B/A input order.
An unmatched median ratio is never substituted for this comparison. Supply
receipts in chronological order for that ordering check. Profile times and
aged/time-limited production trajectories are not optimization A/B timings.

## Campaign tools and bounded artifacts

`campaign.py` exposes `plan`, `run`, `download`, `resume`, `verify`, `diagnose`
and `select-beta`. `plan` never launches compute. It requires measured,
hashed, gated scaling for **both** moves, a selected source/card, and real gate
receipts. The packet validator independently enforces full proof coverage and
the final fresh proof before final production.
Packets preserve the exact proof `config` identity, including `profile:false`,
and carry `proof_fixture_sha256` from every referenced successful gate row.
Selected jobs must be covered by that unchanged launch/move configuration;
proofs with differing configuration identities are not silently combined.

Launch selection uses measured >=160-chain points with >=2 repetitions. For
each move it selects the lowest chain count within 95% of its observed
aggregate peak and distributes the selected concurrency across twenty final
beta/move jobs, with at least eight chains/job. This is a declared allocation
heuristic from measured single-shape scaling, **not** a claim that the mixed
multi-stream workload has already been measured. All jobs run simultaneously
on independent host threads/nonblocking streams; they are not twenty serial
eight-SM beta runs. Each job's move remains separate.

Protocols:

- `--protocol known`: exactly `np.linspace(.01,2,8)`, eight chains/beta/move,
  both separate moves, L21; default 150 s envelope, explicitly overridable.
  This reproduces the known experiment's model/ladder coverage, not its
  colored Metal-local schedule or historical timing assertion. CPU replay
  uses the real downloaded initial bytes.
- `--protocol pilot`: beta 2.5, 2.8, 3.2, 3.4, eight independent chains per
  beta and each move; default 30 s common envelope. This cheap reconnaissance
  spans beta2.5 upward. Insufficient evidence remains inconclusive. Longer
  or resumed pilots are justified by actual diagnostics and remaining budget,
  not by an assumed success temperature.
  Optional `--pilot-betas CSV` supplies distinct finite positive betas sorted
  numerically, for pilot plans only; the default grid above is unchanged.
- `--protocol final`: exactly `np.linspace(.01,beta_max,10)`, >=8 independent
  chains per beta and each separate move, **600 s common production envelope**.
  Requires CPU-selected `--beta-selection` from actual **nonlocal** pilot
  relaxation evidence within the600 s wall criterion. Both moves remain in
  the final ladder; local aging/relaxation is reported separately, not
  represented as thermalized because nonlocal tau fits the envelope. There is
  no hardcoded successful beta_max; untested ladder points remain unverified.

Seeds are full uint64 decimal integers derived reproducibly by domain-separated
SHA256 from an explicitly supplied uint64 master seed, protocol and unique
job ID; collisions are rejected/rederived. Chains use distinct native chain
keys within each job. Seeds never pass through float or uint32.

TSV columns are exactly:

```
id beta mode chains seed window threads ppt history_every chunk_sweeps initial start_sweep
```

Those columns are tab-separated. `initial=-` requests the authoritative CPU
initializer; resume mounts actual raw bytes at `/fixtures/JOB.raw`, preserving
seed and exact next uint64 sweep. Packet mount substitution acts on command
arguments, not TSV contents. The selected native backend is shared by the
multi-job binary; window/threads/ppt are selected separately per move.

Cold-start pilot/known defaults are256 sweeps/chunk and history cadence16.
Final planning requires repeatable `--pilot-production DIR/production`
arguments pointing to real native `jobs.tsv`/`manifest.tsv` outputs. Each
beta/move's chunk uses the slower observed neighboring-beta stream rate at
the selected launch, targeting `.75s` (`--chunk-seconds` within.5–1s), capped
at4096 sweeps and decimated to about16 energy observations/chunk. This is a
declared sizing heuristic, not a synthetic performance curve or a promise
of equal wall time at changed concurrency. `plan.json/chunk_planning`
retains actual manifest hashes and rates used. Production uses rolling actual
checkpoints, bounded observations and sparse **actual4-sweep replay probes**
that advance the continuing backend; it does not archive every production
chunk or stop early when a retention cap fills.
The native command starts `driver --jobs-tsv @fixture:jobs.tsv --seconds SECONDS
--output-dir @run:production` and adds `--max-history-samples H`,
`--max-snapshots S`, `--max-replay-chunks R`, `--probe-every64`,
`--probe-sweeps4`, and `--observations-every16` (flags/values are separated in
the actual packet). Defaults H4096/S128/R32 are reduced conservatively within
the explicit total disk bound (default2048 MiB), never below H64/S64/R4;
an impossible bound rejects planning rather than truncating600 s production.
`plan.json/retention` and `estimated_output_upper_bytes` record admission.
Native history cadence doubles as needed; saved timestamps/cadence, not
assumed uniform every-sweep observations, govern analysis. RAM is bounded by
native chunking/retention; diagnostics admit at most256 MiB retained-array-plus-
input bytes per job and4096 snapshots, plus Python/NumPy bookkeeping overhead.
More data than the bound is inconclusive, not manufactured relaxation evidence.

Expected native output:

- root `jobs.tsv` and `manifest.tsv`; manifest columns
  `id,beta,mode,chains,seed,next_sweep,chunks,kernel_seconds,wall_seconds,status,checkpoint`;
- `production/JOB/initial.raw`, two bounded checkpoint raw slots
  `checkpoint-0.raw`/`checkpoint-1.raw`, and atomic `checkpoint.tsv` key/value
  metadata including `state` (the authoritative slot) and `next_sweep`;
- actual `production/JOB/probe-first.npz`, `probe-last.npz` and rotating bounded
  `probe-NNNNNN.npz` slots: `initial,final` uint8(chain,L³),
  `snapshots` uint8(2,chain,L³), absolute uint64 `snapshot_sweeps`, absolute
  `history_sweeps`/`history_sweeps_absolute`, uint32 `energy_int,
  accept_local_count,accept_nonlocal_count` (chain,sample), compatible float
  energy/acceptance arrays, uint64 `seed,start_sweep,next_sweep,history_every`,
  `meta=[L,chains,sweeps,mode,0]` and `scalars=[beta,rho,rho1,rho2]`.
- `production/JOB/observations.npz`, preferred for diagnostics, uses the same
  NPZ state/model/history fields across the whole real interval, with decimated
  energy and bounded absolute-age snapshots. Probe histories save every sweep;
  their `history_every=1` need not equal the production job's cadence. Consumers
  order probes by saved start/next sweep, not rotating slot filename.
  Sparse replay intervals are not synthetic reconstructions of omitted work.
  Snapshot targets relative to campaign origin include `n*h`, n1..10,
  and `(10*m+r)*h`, m in{1,2,3,10,20,30}, r1..4, h powers10.
  These real targets supply the fixed late waiting-group comparisons below;
  a plain decimal-log grid supplies tw/10 references but generally lacks
  matching positive-lag states. Native `run.tsv` records the snapshot schedule
  and primary-group rule; no extra NPZ metadata is required.

`resume --plan PLAN --production DOWNLOADED/production --output-dir NEW`
prefers verified immutable recovery generations under
`production/durable-checkpoints/JOB/generation-SWEEP-CHUNKS/`. The worker
publishes those stable original metadata/raw pairs during long driver calls;
`snapshot.json` binds job ID, exact next sweep, state filename/byte size,
state SHA256 and original checkpoint.tsv SHA256. Resume enumerates generations
and chooses the highest valid exact counter pair, rejecting corrupted newer
generations rather than combining their files. It does not depend on a
possibly incomplete `latest.json` pointer. If no durable generation exists,
the ordinary metadata-selected native raw slot is used, not a guessed alias.
`plan.json/recovery` records actual source paths/hashes, next sweep and
rejected newer candidates. Resume writes a fresh packet/output run, preserves
actual seed/index/bytes and never regenerates an incoming state or overwrites
prior probes. Receipts from all segments must be retained together for
end-to-end coverage and cost accounting.

`run --packet PACKET --gpu CARD --budget-usd POSITIVE [--final]` invokes the
new Modal API. The ledger reserves every actual invocation; there is no
implicit retry. CPU build admission is separately controlled by the parent.
`download --run-id ID --output-dir DIR` retrieves the real Volume packet with
`modal volume get lattice-solid-opt3-runs ID DIR`. Download itself does not
start a compute container. `verify --plan PLAN --production DIR/production
--cpu-driver solid-opt3/build/cpu_driver --output postverify.json` replays every
actual retained probe from its saved initial bytes and exact seed/start,
requires the exact planned job set, compares deterministic states/model fields
and integer/float energy/counts at every saved absolute history timestamp,
and checks byte continuity between adjacent retained chunks. Its coverage
explicitly lists omitted intervals; passing replay does not claim those gaps
were CPU-replayed. CPU replay requests history cadence1 to
avoid standalone-relative versus batch-absolute cadence ambiguity. The native
standalone output is cwd/run.npz; its `history_sweeps` remains relative for
compatibility, while `history_sweeps_absolute` binds replay sampling. A replay
failure is not a successful result. Gate proof and postverification do not
establish equilibrium.

## Scientific diagnostics and evidence-driven beta_max

`campaign.py diagnose --plan PLAN --production DIR/production --output diagnostics.json`
operates only on saved real chunks. `diagnostics.py CHUNKS... --output FILE`
is also available for one job. Exact state continuity and identity are checked;
a missing chunk, bad timestamp, changed species count, insufficient chains,
missing waiting times or memory bound yields **inconclusive**.

Chains, not individual sites or correlated sweeps, are independent error units.
The implementation resamples whole chains (4096 deterministic bootstrap draws)
for99% simultaneous absolute-error bands within each block/curve comparison.
These are finite-sample approximate bootstrap intervals, not a rigorous
finite-sample equilibrium certificate or a global familywise guarantee across
all jobs. Default diagnostic tolerances are explicit engineering decisions,
not paper-derived constants: energy difference .01/site, normalized overlap
waiting-time difference .05, late overlap upper band <.1. They are reported
with every result and should be fixed before inspecting the relevant data.

1. Split the later half of the saved energy interval into four equal-duration
   blocks, with >=4 saved samples/block. Compare all six per-chain block-mean
   differences. Require the entire simultaneous band inside ±.01/site; merely
   failing to reject drift is not treated as positive evidence.
2. Compute the normalized **species** overlap excluding vacant sites from the
   reference particle sum. For actual counts n1,n2, sites V, and occupied M:
   `q0=(n1²+n2²)/(M*V)` and
   `C=(same-species occupied-reference matches/M - q0)/(1-q0)`.
   This is1 at identical states and tends to0 for independent uniform
   configurations at the same conserved composition. Do not count matching
   vacancies; do not use q0=.3²+.45² without dividing by occupancy.
3. **Predeclared before GPU scientific data:** choose the largest fully
   observed decimal scale h>=10, with34h<=saved elapsed sweeps, using only
   saved timestamps. Require all reference/target states for the three waits
   10h,20h,30h versus h,2h,3h and the eight exact common positive lags
   h/10,2h/10,3h/10,4h/10,h,2h,3h,4h. No state interpolation or selection
   based on observed C is permitted. This is the sole primary late waiting
   group; older fully observed groups are descriptive and their cold-start
   aging does not veto the late criterion. Require the unchanged paired-chain
   difference bands inside ±.05 and upper overlap bands below.1 at the last
   three lags of every primary comparison. Missing group, late aging,
   uncertainty or nondecaying overlap is inconclusive even if energy is flat.
4. Report only an observed sampling bracket for the first mean C=e^-1
   crossing. If there is no crossing, report right-censoring at the largest
   measured lag; do not fabricate tau by fitting a plateau.

The strongest output is
`compatible_with_stationary_relaxation_not_equilibrium_proof`, with
`equilibrium_established=false` **always**. Optional hot/aged independent
initialization comparisons would be additional evidence, not silently assumed.
`select-beta DIAGNOSTICS... --output beta-selection.json` takes the highest
contiguous tested positive beta with compatible **nonlocal** energy/aging/
overlap-decay evidence and three measured C=e^-1 crossing brackets from the
predeclared primary late waiting group only.
The largest observed bracket upper endpoint must fit600 s using the actual
pilot's inclusive per-chain sweep rate. The emitted wall projection is a
measurement-based admission heuristic, **not** measured final-concurrency
performance, a fixed10tau/50tau prescription or an equilibrium certificate.
`diagnose` attaches the actual native manifest rate and its hash; absent
rate/crossing/decay evidence yields inconclusive rather than an extrapolated
tau. Local diagnostics are retained separately and do not determine beta_max;
`local_thermalization_established=false` remains explicit. Both primary moves
still run for600 s at ten betas. Actual lower-beta evidence may set a lower
limit, with no presumed successful beta2.5 threshold. If no candidate is
acceptable, output `status=inconclusive,beta_max=null`. The user-reported local
tau≈8000 sweeps at beta2.5 is context, not an opt3 measurement or a result
rerun merely to confirm that observation.

## Primary literature: sizes, relaxation and VFT

- Nishikawa & Hukushima, *Lattice glass model in three spatial dimensions*,
  [arXiv:2003.02872v2](https://arxiv.org/html/2003.02872v2), Eq.(2), Figs.2–3:
  occupied-reference normalized species autocorrelation, tw versus tw/10
  agreement, and tau_alpha defined by C(tau_alpha)=e^-1. The displayed dynamics
  use **L20,6000 particles**, not L21. Their waiting times range10²–10⁸ sweeps
  with temperature; no universal10tau or50tau prescription is given. Fig.3's
  VFT guide uses T_VFT/A=.177(6); in standard notation
  tau=tau0 exp[B/(T-T_VFT)] (equivalently D*T_VFT in the numerator), with fit
  parameters specific to dynamics, size and temperature range. The text
  explicitly notes that low-temperature tau increases with system size and
  the fitted T_VFT would shift upward for larger systems. This is not an L21
  prediction, a measured opt3 relaxation time or a proven singularity.
- Nishikawa & Berthier, *Collective relaxation dynamics in a three-dimensional
  lattice glass model*, [arXiv:2307.08110v2](https://arxiv.org/html/2307.08110v2),
  Eq.(2), Fig.1 and supplement: same collective species overlap convention;
  time unit L³ attempted moves; **L20 except L10 at T=.26**. The paper
  separates nonlocal equilibrium preparation from local physical dynamics,
  uses an Arrhenius high-temperature reference (not a universal transferable
  VFT fit), and reports incompletely decayed overlap at low temperatures.
  Its supplement explicitly warns that incomplete decay causes systematic
  low-frequency spectral error. L10 low-T behavior and L20 relaxation curves
  cannot be relabeled as L21 measurements. Neither paper licenses replacing
  physical local dynamics by nonlocal or colored dynamics in this campaign.
  The2023 text describes choosing sites with different occupations/types;
  the2020 description distinguishes particle swaps and moves to vacancies.
  These descriptions alone do not establish the same unconditional proposal/
  no-op clock as the frozen CPU law here. The papers supply observable
  definitions and qualitative size-dependent context, not a byte-trajectory
  authority or a directly transferable opt3 VFT/relaxation calibration.

## Final evidence path

After actual parent runs, regenerate the single results bundle:

```
python solid-opt3/analyze.py RECEIPT_JSON_OR_JSONL... \
  --output-dir solid-opt3/artifacts/final-analysis \
  --ideas-tsv solid-opt3/results.tsv \
  --budget solid-opt3/artifacts/budget.json \
  --diagnostics ACTUAL_DIAGNOSTICS_JSON
```

`final-analysis/RESULTS.md` includes only actual receipt-derived scaling,
matched speedups, available clocks/profiles, supplied actual budget bounds,
scientific diagnostics and recorded predicted-versus-measured idea rows.
Missing measurements stay unmeasured. `analysis.json` retains raw observations
and excluded-row reasons; `scaling.tsv` is machine-readable and `scaling.svg`
is the only visualization. Actual production global wall time must be read
from worker/manifest receipts; overlapping per-job CUDA seconds must not be
summed or called global elapsed wall time. The parent appends final results,
actual coverage, scientific limitations, costs and hourly status. The measured
performance evidence is explicitly separated from actual, still-inconclusive scientific evidence.

## Follow-up measurements — 2026-11-11T10:45:06Z

This timestamped continuation preserves the historical results above. Sources:
`artifacts/followup-C07-profile-summary.json` and
`artifacts/followup-C14-summary.json`. The bounded final pilot is still running
at this update; neither its completion nor its scientific outcome is claimed.

### Actual current-winner hot-kernel profile

C07 profiling receipt `53dc7b2cdac54ff3af1e75f9bd9c6cb8` uses source digest
`8020c79afcbb7af5d82234c89a6e458fa95d5b43493fe0ff5c3a9774636481e8`,
W256/T256/PPT1, both physical move laws, and1/142 chains. Each captured timed
kernel runs1024 sweeps from the8192-sweep aged checkpoint; the matching warmer
is skipped. Shared-state initialization, sweep histories and final export
inside the captured kernel remain included. These are actual NCU diagnostics,
**not ordinary throughput measurements**.

| Move | Chains | Issue-active (%) | Barrier-stall (%) | Warp instructions/proposal |
|---|---:|---:|---:|---:|
| Local | 1 | 27.83 | 23.48 | 16.447 |
| Local | 142 | 27.08 | 25.14 | 17.718 |
| Nonlocal | 1 | 29.89 | 20.40 | 14.688 |
| Nonlocal | 142 | 29.84 | 20.43 | 14.761 |

All four captures report **zero local-memory load/store sectors** and zero
shared-store bank conflicts. Shared-load bank conflicts are0.162–0.167 per
attempt. Active warps are2.0 per active SMSP cycle, eligible warps0.31–0.36;
short-scoreboard stall is7.90–9.69%, long-scoreboard0.05–0.06%, and
MIO-throttle0.12–0.14%. Names/units and original counter values remain in the
summary; the barrier numbers are
`smsp__warp_issue_stalled_barrier_per_warp_active.pct`, while issue-active is
`smsp__issue_active.avg.pct_of_peak_sustained_active`.

**[INFERENCE]** In these captures, reducing spill traffic is not the evident
next lever: none was observed. Ordered synchronization, dependent shared-data
work and insufficient ready warps are more plausible constraints than a
large long-latency local/global-memory scoreboard bottleneck. This is a
diagnosis to guide exact algorithms, not proof that any particular instruction
sequence is removable. Warp-stall percentages are sampled warp-state
statistics, **not wall-time fractions**, mutually exclusive runtime stages or
Amdahl fractions. Likewise,27–30% issue-active is not a directly recoverable
70–73% of application wall time.

A conditional issue-count ceiling would divide the device's sustained
warp-instruction issue capacity by the measured warp instructions/proposal,
only if the same instruction mix/count and relevant sustained issue capacity
apply. Dependencies, instruction-class limits, synchronization, occupancy,
replay and included non-hot-loop work prevent that ceiling from establishing
an achievable rate. These counters alone establish neither hardware-limit
proximity nor a numerical speedup available from removing barriers.

### C14 exact sparse/acceptance-regime continuation: correct selected scope, slower

Receipt `16ecda16b46e4655a4a0cebc4f59ad2d`, native source digest
`429d2c2034ea4bd629a51eae015e812a33bb827ae413f6078a9f314e075cc641`,
retains passing immutable commands even though the aggregate packet failed.
The parent verified W32/T32, W32/T256 and W256/T256 for both moves: normal
million-proposal checks with the full existing100000-proposal trace, multi-chain
and saturation gates, all three sanitizers, and fresh100-million-proposal
held-out checks using seed3483929462277233793 and chain1000002. This qualifies
only those selected shapes/laws and that tested native identity; it does not
promote untested shapes, cards or acceptance regimes. Frozen CPU physics,
RNG/EFMA, fixtures, gate tolerances and physical local/nonlocal laws are not
replaced by a faster different sampler.

The eight completed ordinary beta3.2 timing points below each contain all four
repeats. M/s means million attempted proposals per second per chain; G/s means
billion attempted proposals per second for the card. Correctness coverage of
W32/T256 does not imply a completed timing point for that shape.

| Move | Chains | W/T | M/s/chain | G/s/card |
|---|---:|---|---:|---:|
| Local | 1 | 32/32 | 32.678135 | 0.032678 |
| Local | 1 | 256/256 | 137.650298 | 0.137650 |
| Nonlocal | 1 | 32/32 | 34.096871 | 0.034097 |
| Nonlocal | 1 | 256/256 | 165.647879 | 0.165648 |
| Local | 142 | 32/32 | 30.991558 | 4.400801 |
| Local | 142 | 256/256 | 116.167271 | 16.495752 |
| Nonlocal | 142 | 32/32 | 33.759756 | 4.793885 |
| Nonlocal | 142 | 256/256 | 158.301232 | 22.478775 |

C14's W256 points are below the recorded C07 ordinary points above
(local169.671703/142.068618 M/s/chain and
nonlocal207.955519/197.339042 M/s/chain at1/142 chains). This is a comparison
of recorded completed measurements, not a new matched A/B/B/A claim or an
NCU-derived throughput comparison. C14 is **not kept as the speed winner**.
Its lazy-epoch reference-build counter has a reported minimum349, but that
mechanism count is not a performance gain.

The first beta0.01 local, one-chain W32 four-repeat benchmark, command20,
exceeded its10-second bound. **No accepted measurement exists for that point**;
the timeout is not a rate, a low-acceptance extrapolation, or evidence that all
prior passing commands failed. Aggregate packet success remains false.

### Conditional hardware and still-open shutdown/accounting

No H100 runtime or rate was measured. The parent ruled out purchasing an H100
run for the slower sparse engine; compilation or a conditional hardware
ceiling is not a substitute for such a measurement. C07 remains the recorded
L40S depth choice, with the historical C02 peak-card frontier unchanged.

The last bounded30-second,20-cell/120-chain pilot is **running, not complete**.
Its actual result, provider closure and final accounting must be appended from
the returned evidence. The initial follow-up USD0.90 cap was explicitly revised
to USD0.96 only for this final approximately USD0.22002 bounded call; total
configured worst liability remains below USD14.95 and the original hard USD15
limit. These are authorization/envelope bounds, not finalized spending.
The failed C14 call's **full USD0.50032836 liability remains retained** because
provider metadata resource-limit errors blocked reconciliation; no inferred
refund is credited. The C14 summary's USD14.726575549653047 cumulative committed
value is its at-return snapshot, not the final campaign cost. This update
claims neither completed pilot shutdown nor finalized campaign spending.

## Follow-up returned evidence — 2026-11-11T10:51:38Z

This later update supersedes the **running/pending accounting** status at
10:45:06Z only to the extent of the returned evidence below. It does not
rewrite the historical failed600-second campaign.

### Measured acceptance-regime lookahead comparison

`artifacts/followup-acceptance-window-summary.json` records successful receipt
`801576c74a0d452289c28cc178d8fdc7`, source digest
`429d2c2034ea4bd629a51eae015e812a33bb827ae413f6078a9f314e075cc641`.
All eight single-chain beta0.01/1.5 benchmark commands completed their four
ordinary fixed-work repeats. Every point uses16 warmup and16 timed sweeps.
The table groups the eight measured medians into four matched beta/move pairs;
units are million attempted proposals/s/chain.

| Beta | Move | W32/T32 M/s | W256/T256 M/s | Acceptance fraction, both shapes |
|---:|---|---:|---:|---:|
| 0.01 | Local | 0.579414 | 1.098896 | 0.627902 |
| 0.01 | Nonlocal | 0.585561 | 1.069495 | 0.624757 |
| 1.5 | Local | 11.828523 | 24.687113 | 0.029418 |
| 1.5 | Nonlocal | 13.785291 | 27.486585 | 0.027474 |

Within every beta/move pair, both shapes use the same aged checkpoint
fingerprint and return identical accepted counts. The fingerprints are,
respectively, `27af08330130257b`, `9e5d78bd19ff0a43`,
`6e794e50cb80dcb7` and `562900e1c73a9fdd`; these are native noncryptographic
fingerprints, not substituted SHA256 identities. W256/T256 wins every tested
pair, including the higher-acceptance beta0.01 cases. Together with the
completed deep-beta C14 points, this supplies **no measured reason to keep an
adaptive-window policy**; none is kept. The comparison changes threads along
with window and therefore does not isolate window size alone.

These16-sweep-aged states are not the deep-beta8192-sweep-aged states. Rates
across the two workloads cannot be read as a causal beta-only effect or a
matched same-state speedup. No untested beta, concurrency, shape, stationary
regime or equilibrium claim follows. The earlier timed-out10-second command
remains failed; these later completed bounded commands do not relabel it.

### Native bounded pilot returned successfully; postverification pending

The same receipt is aggregate **OK**. Its native command0 exited0 with
30 seconds requested,20 cells and120 chains; command elapsed time was28.332s
and reported native wall time27.6032s. All eight following benchmark commands
also succeeded. This is the actual successful short native-pilot command,
not completion of the original600-second run or evidence of600-second
scientific relaxation. Dataset download and postverification remain pending
at this update: **no CPU replay, terminal-manifest validation or resume check
is claimed here**.

The parent verified all three owned follow-up apps in `APP_STATE_STOPPED`.
For the latest app `ap-wpIZFkzqo7uKyNB9k3xh9b`, provider evidence at
2026-11-11T10:45:48Z reports zero tasks. No further GPU calls are authorized;
H100 was skipped and no pushes were made.

Returned conservative cumulative committed cost is **USD14.845438183697965**;
the follow-up portion is **USD0.8493909556941706**, below its explicitly
revised USD0.96 cap and the original USD15 campaign hard limit. This includes
the failed C14 call's full USD0.50032836 liability: blocked provider-metadata
reconciliation still supplies no refund. These are conservative committed
accounting figures, not an asserted final provider invoice.

### Downloaded pilot verification and recovery correction

The pending postverification above is now complete. All20downloaded manifest
rows are `deadline`, with no running/failed cell, and the native run.tsv reports
27.603194546s for the30s requested envelope. All80retained actual four-sweep
probes passed frozen-CPU state, energy and acceptance-counter replay. For every
cell, final raw checkpoint bytes/cursor/seed match the actual final probe and
observations; terminal integer energies and both acceptance counts also agree.
Evidence: `artifacts/followup-shutdown-pilot/verification.json` and
`terminal-checkpoint-proof.json`. Gaps between probes are not replayed.

This check exposed a real recovery bug: the reader always preferred a durable
publication even when a newer completed native pair existed. For local-b00 it
selected sweep993 instead of1077. `campaign.py::checkpoint_pair` now orders
completed-native and immutable candidates by sweep/chunk counters, admits a
completed native pair only with run/terminal-manifest evidence and a matching
actual final-probe seed/cursor/state hash, and retains immutable fallback for
incomplete or inconsistent data. All12recovery/planning regressions pass,
including four new progress-selection/corruption cases.

The corrected pure `resume-plan-terminal/packet.json` contains all20actual
terminal checkpoints. Its GPU execution is not claimed; the earlier
`resume-plan/` is superseded. Default16384-chain-per-beta/move production
planning also passed with the current qualified C14 source and matching
immutable proofs, without launching that default-sized campaign. The current
recipe is in `proofs/PRODUCTION_PLAN.md`. Old C07/C08 proofs remain bound to
their frozen native revisions; changing the driver changes the proof digest.

This proves short-pilot shutdown and the stated saved-data/recovery behavior,
not the original600s run, full-trajectory replay, equilibrium, or improved
end-to-end campaign throughput. The original failed600s result is unchanged.
All paid admission is explicitly closed before further image/CPU/GPU
materialization; every historical charge remains. Conservative cumulative
commitment rounds toUSD14.845, includingUSD0.849for this follow-up and the
failed timing packet's full unreconciled reservation. No H100 allocation or
push occurred. C07 remains the measured L40S depth winner; this follow-up did
not establish a new speed record.
