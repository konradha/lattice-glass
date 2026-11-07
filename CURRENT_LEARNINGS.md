# Current learnings

Date: 2026-11-05

## 1. Original implementation

The original CPU implementation is present. It is not named `lattice_glass.cpp` in this tree.

```text
cpu/lattice.hpp
cpu/main.cpp
cpu/lattice_single
```

`./cpu/lattice_single --validate-only --seed 1` returns:

```text
kernel_verify=PASS
```

The CPU README reports 20.9 ns/proposal on an Apple M3 and 35.3 ns/proposal on a Modal AMD core. The production state is L=25, fixed composition, 2-bit type plus 6-bit neighbor mask, 26 fixed nonlocal shifts, and xoshiro256++.

## 2. Golden C++ model

`golden/lattice_model.hpp`, `golden/lattice_model.cpp`, `golden/tests.cpp`, and `golden/validate.cpp` are in the repo. The Python window model was removed.

Contract that is currently implemented:

- Five xoshiro128++ lanes per substream.
- `W` substreams per chain, with proposal `k` using substream `k mod W`.
- `i` and `j` come from the two 16-bit halves of lane 0 by multiply-shift.
- `u` comes from lanes 1 and 2.
- Lanes 3 and 4 step but are reserved.
- Lane 0 is splitmix64-seeded; lanes 1-4 are jump-separated.
- Substreams are jump-separated.
- Null proposals: `i == j`, equal content, or periodic Manhattan distance <= 2.
- Acceptance: `dE <= 0` or `u < T[dE]`.
- `T` is loaded from the host table.
- State: `s`, `c`, energy, proposal/null/accept counters.
- Checkpoint: state, counters, all substream states, and `k`.

## 3. Tests that pass

```text
cases=10160 radius1_negative=1
samples=2000 G=6.3174 df=6 p=0.611725
```

- Window property covers W in {1,2,4,8,16}, L in {7,21,28}, and beta in {0,0.5,3.0,3.3}.
- The radius-1 negative control fails as required.
- Checkpoint round-trip and invariant checks pass.
- Beta=0 neighbor-count distribution passes.
- Golden CLI: 58 ns/proposal single chain, 34 ns/proposal with 4 chains.

## 4. The stale-read bug

The first window model checked hazards only against accepted candidates. That is wrong. A rejected candidate can read state written by an earlier accepted candidate. Serial execution would re-evaluate it. The window must stop and retry it.

The corrected rule is implemented in the C++ golden model and `rtl/opt4`. The golden property test now passes.

## 5. RTL status

`rtl/opt4` now has:

- the stale-read hazard rule,
- a neighbor ROM,
- 16 substreams x 5 xoshiro128++ lanes x 4 words,
- host loading of the same RNG state as the golden model,
- the golden 16-bit multiply-shift and 64-bit `u`,
- debug selection of queue entries,
- a direct RTL-versus-golden testbench.

The RTL/golden test still fails.

```text
win 1  rtl_h=2 ref_h=2 head=2
pre q0 mismatch win 2 head=2
RTL  i=113 j=306 u=6052307426033430627
ref  i=49  j=101 u=14479885225695173485
```

Window 1 matches. Window 2 begins with the wrong queue head. The RTL queue entry that should be retained after the commit is not the queue entry the golden model retains. This is a concrete COMMIT/suffix bug.

## 6. What is not established

- No FPGA samples.
- No equilibrium samples.
- No RTL-versus-golden equality.
- No RTL throughput measurement.
- No L=7 exact enumeration. Fixed-composition L=7 is too large for the current enumerator.
- No CPU comparison at beta 1.5, 2.22, 2.5.
- No PractRand, BigCrush, or cross-chain correlation.
- No >=1e9-proposal property run. Do not start one locally.
- No two-initial-condition convergence, tau_int, or log-binning.
- No spaced-sample metadata or 10-minute readout implementation.

## 7. Performance facts

- CPU original: about 20.9 ns/proposal on M3.
- Serial RTL at 250 MHz: 4 ns/proposal, about 5.9x one CPU core.
- A single chain is serial. Additional engines make independent replicas; they do not accelerate one chain.
- 30x per chain needs 1.27 GHz at one proposal per cycle, or a correct speculative engine at about 5.1 proposals per cycle.
- 50x needs about 2.1 GHz or about 8.5 proposals per cycle.
- The W=16 window property passes in C++, but RTL is not yet equal to it.

## 8. Next concrete fix

The RTL COMMIT path must perform one consistent queue operation:

1. Retire exactly `h` entries.
2. Move the unchanged suffix to the front.
3. Fill exactly `h` new entries at the tail.
4. Never let the new fill overwrite a retained suffix entry.
5. Compare all queue entries before and after commit against the golden model.

The first failing position is at window 2, queue head 2. Fix that before any further optimization or sampling work.

## RTL/golden Step 0 status

Short RTL-versus-golden equality now passes:

```text
L=7  W=16 beta=0.000 windows=40 mism=0
L=7  W=16 beta=3.000 windows=40 mism=0
L=7  W=16 beta=3.300 windows=40 mism=0
L=21 W=16 beta=3.000 windows=20 mism=0
L=21 W=16 beta=2.800 windows=20 mism=0
```

The stale `ev_acc` problem was fixed by registering the EVAL acceptance
vector once and using the registered vector in COMMIT and hazard detection.
The earlier window-14 hazard divergence is gone.

The >=1e8-proposal RTL equality run was not run locally. The current
Verilator harness is too slow for that scope, and local hero runs are explicitly
out of scope.

## Literature

`FPGA_LITERATURE.md` contains the bounded 15-source table.

## Optimization loop

Not started. Step 0 long-run equality is still open.

## Vivado and toolchain lessons (2026-11-05)

1. Bit-exact in Verilator does not mean synthesizable. Vivado rejected dynamic loop bounds and dynamic array shifts that Verilator accepted for 10^7 proposals: `for ii < run_h`, `qa[ii+run_h]`, `for ii < ev_h_q[ctx]`.
2. Unreachable phases still cost synthesis. The old `EVAL`, `COMMIT`, and `DRAIN` branches were never entered, but Vivado still tried to synthesize their dynamic loops. They had to be deleted.
3. The W=4 shift path is now explicit: `case(run_h)` with fixed indices. That is synthesis-friendly but hard-codes W=4; W=2/W=8 need their own shift cases.
4. Remote toolchain: Vivado 2024.1 at `/opt/Xilinx/Vivado/2024.1`, source `settings64.sh`. Verilator 4.038 is too old for this RTL; Verilator 5.028 built from source to `~/verilator`.
5. Remote simulation: L=21, C=3, 10,000 windows/context, `mism=0`, `cycles/window=1.000`, wall 6.82 s.
6. Local approved gate: L=21, C=3, 1,000,000 windows/context, `mism=0`, `cycles/window=1.000`, wall 957.91 s, about 12M proposals.
7. The first OOC attempts failed before place/route with Synth 8-3380. No resource or Fmax result is claimed until OOC completes.
8. OOC part: `xcvu47p-fsvh2892-2-e`. Script: `rtl/cslow/ooc_xcvu47p.tcl`.

## Vivado RAM inference warning (2026-11-05)

- Vivado did not infer URAM from the 3-D bank arrays. It inferred registers:
  - typ_bank: 55,566 registers
  - cnt_bank: 83,349 registers
  - thr: 98,496 registers
- One engine is therefore ~237k registers. The 45-engine/FPGA model assumes replicated URAM copies that the current RTL does not instantiate.
- Until the banks are explicit RAM instances with the required read/write ports, the 45-engine and 4.5e10/s figures are unsupported.
- No throughput claim.

## Session close 2026-11-05

### Verified
- L=21, C=3, W=4, 1,000,000 windows/context, local Verilator: mism=0, cycles/window=1.000, 957.91 s.
- L=21, C=3, W=4, 10,000 windows, remote Verilator 5.028: mism=0, 1.000, 6.82 s.
- Trace-diff C=3 short: 240 events, no mismatch.
- Literal 1e8 gate was replaced by user-approved 1e7; this run is ~12M proposals, not 1e8.

### Vivado findings
- Vivado 2024.1 at /opt/Xilinx/Vivado/2024.1; source settings64.sh.
- OOC part xcvu47p-fsvh2892-2-e.
- First OOC failed on dynamic loop bounds/array shifts (Synth 8-3380).
- Dead EVAL/COMMIT/DRAIN phases still synthesized; removed.
- Replacing the 9,261x6 nn_rom with direct nbr() calls made synthesis reach "done synthesizing module" quickly.
- Vivado still inferred registers, not URAM:
  typ_bank 55,566; cnt_bank 83,349; thr 98,496; ~237k registers/engine.
  That allows only ~4-5 engines/FPGA, not 45.
- Three OOC jobs then stalled for >1 h with no utilization/timing report. No Fmax.
- 3-copy 4.5e10 model is invalid: 8 parallel reads/bank need 8 read ports.
- URAM-only 4 copies: ~1.7e10/s, 136 chains, ~20.6 h.
- Hybrid URAM+BRAM 8 copies: ~5.3e10/s, C=3 159 chains, ~7.7 h. Estimate only.

### State
- No measured throughput.
- No full-device build.
- No OOC resource/Fmax result.
- Target not demonstrated.

### Restart
- Remote host was shut down after logs were copied.
- Local code: rtl/cslow/lattice_engine.sv, tb_cslow.cpp, perf_model.cpp, ooc_*.tcl.
- Next: explicit packed-context URAM/BRAM memory with 8 read ports and scheduled write-back.

## New host failure (2026-11-06)

- Host ec2-54-162-243-179: 8 vCPU, 30 GB RAM, Vivado 2024.1.
- Verilator 5.028 brought up; L=21 C=3 10k windows: mism=0, 1.000 cycle/window, 7.03 s.
- OOC synthesis ran >3 h, one core, ~13 GB RAM; no utilization or timing report.
- A parallel L=7 synthesis added ~11 GB and left ~4 GB free; it was stopped.
- Host then became unreachable: SSH "Connection timed out during banner exchange". Likely OOM/swap death.
- No AWS CLI locally; host needs console reboot. No OOC/Fmax result.
- Next host must have >30 GB RAM, or only one Vivado synthesis at a time.

## Explicit-memory engine (2026-11-06)

- New file: rtl/cslow/lattice_engine_mem.sv, C=1.
- One read/write port per bank, explicit 5-bit bank words, FSM.
- Functional: L=21, 1,000 windows, mism=0, 155.475 cycles/window.
- Synthesis: 120,543 LUTs, 48,588 FF, 0 BRAM, 0 URAM, 666 LUTRAM.
- Synthesis time: ~11 min, no stall.
- Post-synth WNS at 4.0 ns clock: -15.688 ns -> ~50.8 MHz estimated. No place/route.
- This is the first correct explicit-memory design; it is not fast.

## Process research and verified gates (2026-11-06)

Artifacts:
- PROCESS_PLAYBOOK.md: gate ladder L0-L5, commands, stop rules.
- research/vivado_incremental.md
- research/verilator_timing.md
- research/fast_synthesis_loops.md
- research/formal_equivalence.md
- research/hls_flow.md
- research/ci_orchestration.md
- research/prototyping_shells.md
- research/perf_modeling_mcmc.md
- research/agentic_hdl.md

Verified locally today:
- L0 Verilator lint: rc=0, 81 warnings, no errors, 0.2 s.
- L1 Yosys per-array probe on current dynamic-bank read/write shape: 0 RAM cells, 40 DFFs, 1.4 s. The dynamic-bank read forbids RAM inference.
- L2 parallel reader: L=21, C=1, 10,000 windows, mism=0, 41.256 cycles/window.

Process corrections:
- create_clock was after synth_design; XDC must be read before synthesis.
- runtimeoptimized caused huge area in published study; do not use as default.
- Vivado synthesis uses at most 4 effective threads; RAM size matters more than vCPU.
- Incremental synthesis is incompatible with OOC; top is one module.
- Only post-route Vivado STA is trustworthy for Fmax.
- One remote synthesis queue and 48 GiB free-RAM guard.

Current design facts:
- Serial explicit engine: 156.74 cycles/window, 120,543 LUT, WNS -15.688 ns.
- Parallel reader: 41.256 cycles/window, 143,085 LUT, WNS -34.117 ns (~26 MHz synth estimate), 0 BRAM/URAM.
- Parallel net throughput gain ~1.7x, not 3.8x, because period worsened.

## L1 follow-up (2026-11-06)

- Current dynamic-bank state shape: Yosys probe 0 RAM cells, 40 DFF, 1.4 s. L1 RED.
- Seven named banks with two read expressions per bank: 35 DFF, no RAM. Still RED.
- One-read-port-per-bank probe: Yosys full-synth stat was empty/inconclusive; do not claim RAM inference from it.
- Conclusion: the named-bank rewrite is not yet proven; run the L1 probe on the actual rewritten module before Vivado.

## Named-bank + parallel apply (2026-11-06)

- Seven named banks, one muxed read port and one write source per bank.
- Parallel apply: 2 cycles per accepted move (7 banks/cycle), no per-site read.
- L2: L=21 C=1, 10,000 windows, mism=0, 26.513 cycles/window.
- OOC: 42,994 LUT, 2,474 FF, 7 RAMB18, 0 URAM; WNS -32.908 ns @ 4 ns -> ~27.1 MHz synth estimate.
- Per-engine at 27.1 MHz: 4/26.513*27.1M = 4.1M proposals/s.
- 30 engines/FPGA by LUT: ~123M/s/FPGA; 8 FPGAs ~1G/s.
- 1e11/s total means 1.25e10/s per FPGA: still ~100x short.

## Padded L=32 / 8-bank engine (2026-11-06)

- Internal NP=32768, 8 banks x 4096, word 20 bits with coords; coords derived by shifts/masks.
- 8-bank colour collides at 21-torus seams; read/apply phase counts are data dependent (1.8% extra cycles).
- L2: 10,000 windows, mism=0, 27.982 cycles/window.
- L0: rc=0, 0 errors.
- OOC: 27,147 LUT, 3,613 FF, 16xRAMB36 + 8xRAMB18; WNS -8.242 ns @ 4 ns -> ~81.7 MHz synth estimate.
- Per engine at 81.7 MHz: 4/27.982*81.7M = 11.7M proposals/s.
- Net vs previous: 2.85x proposals/s per engine; LUT 35,464 -> 27,147.
- ~48 engines/FPGA by LUT; ~560M/s/FPGA; 8 FPGAs ~4.5G/s. Target 1.25e10/FPGA still ~22x short.

## One-proposal generation + EVAL split (2026-11-07)

- GEN state: one gen_prop/cycle; LUT 29,282 -> 14,818; cycles 27.982 -> 31.966; post-route Fmax 96.1 MHz.
- EVAL0/EVAL1 split: cycles 31.966 -> 35.966; post-route Fmax 115.6 MHz; LUT 14,882.
- Current best per engine: 4/35.966*115.6M = 12.86M proposals/s.
- Engines/FPGA by LUT: ~87; ~1.12G/s/FPGA; 8 FPGAs ~9.0G/s total.
- 2e10 total is ~2.2x away; 1e11 total ~11x away. Target 1.35e10/FPGA is still ~12x away.
- No full-device/1e10 checkpoint; no beta throughput.

## Best engine after pipeline exploration (2026-11-07)

- One-proposal GEN + EVAL0/EVAL1 split.
- L2: L=21 C=1, 10,000 windows, mism=0, 35.966 cycles/window.
- Post-route: 14,882 LUT, 3,709 FF, 115.6 MHz, WNS -4.647 ns.
- Per engine: 4/35.966*115.6M = 12.86M proposals/s.
- ~87 engines/FPGA by LUT -> ~1.12G/s/FPGA; 8 FPGAs ~9.0G/s total.
- EVAL2 accept split improved Fmax to 121.7 MHz but reduced per-engine to 12.18M; reverted.
- 2e10 total is ~2.2x away; 1e11 total ~11x. Target 1.35e10/FPGA still ~12x.
- No place-route of a multi-engine design, no 1e10 checkpoint, no beta throughput.

## Two-stage generation + accept (2026-11-07)

- One-proposal GEN split into RNG step + coordinate decode; EVAL accept stage registered.
- L2: 10,000 windows, mism=0, 43.950 cycles/window.
- Post-route: 15,278 LUT, 3,867 FF, 151.1 MHz, WNS -2.617 ns.
- Per engine: 4/43.950*151.1M = 13.75M proposals/s.
- ~85 engines/FPGA by LUT -> ~1.17G/s/FPGA; 8 FPGAs ~9.4G/s total.
- 2e10 total ~2.1x away. Target 1.35e10/FPGA ~11x away.
- Remaining path q_reg -> de_fin (EVAL1 dE); further split costs cycles and net gain <1.2.

## Read/eval overlap + hazard split (2026-11-07)

- Read engine runs one proposal ahead with double-buffered site values.
- HAZ split into conflict-register + priority stages.
- L2: 10,000 windows, mism=0, 39.950 cycles/window.
- Post-route: 15,824 LUT, 4,120 FF, 160.8 MHz, WNS -2.218 ns.
- Per engine: 4/39.950*160.8M = 16.1M proposals/s.
- ~82 engines/FPGA by LUT -> ~1.32G/s/FPGA; 8 FPGAs ~10.6G/s total.
- 2e10 total ~1.8x away; 1.35e10/FPGA ~10x; 1e11 total ~9.4x.
- Critical path now all_vals -> de_fin (EVAL dE).

## W=8 experiment (2026-11-07)

- W=8 shifts implemented; first 10 windows pass, window 11 h differs (RTL 2 vs golden 8), state diverges.
- Rejected; W=4 best restored and revalidated: 10,000 windows, mism=0, 39.950 cycles/window.
- Current best: 15,824 LUT, 4,120 FF, 160.8 MHz, 16.1M proposals/s/engine.

## 16-bank experiment (2026-11-07)

- Geometry bank=(x+y+2z)&15, index=addr>>4; max 2 sites/bank.
- L2: 10,000 windows, mism=0, 44.014 cycles/window.
- Post-route: 20,872 LUT, 128.4 MHz, 11.67M proposals/s/engine.
- Worse than 8-bank 16.1M; rejected. 8-bank best restored and revalidated (39.950 cycles, mism=0).

## Final architecture bound (2026-11-07)

- Best verified: 39.950 cycles/window, 15,824 LUT, 160.8 MHz, 16.1M proposals/s/engine.
- 82 engines/FPGA => 1.32G/s/FPGA; 8 FPGAs => 10.6G/s total.
- Target 1.35e10/FPGA = 84 proposals/cycle; current ~8.2 => 10.2x gap.
- 2e10 total ~1.9x; 1e11 total ~9.4x.
- 16-bank (11.67M/s) and W=8 (incorrect) rejected.
- Per-chain 1e9/24h requires 11.6k sweeps/s; current 1.74k => 6.7x.
- Remaining work is area/port architecture and a multi-engine hardware build; no further scheduling-only gain found.

## W=8 corrected, rejected (2026-11-07)

- Root cause of W8 mismatch: h=1..3 shift cases were W4-only; fixed for W8.
- W8 with radius-3 hazard: 10,000 windows mism=0, 73.025 cycles/window.
- Post-route: 14,008 LUT, 124.4 MHz, 13.63M proposals/s/engine.
- W4 remains better: 39.950 cycles, 160.8 MHz, 16.1M/s.
- Production restored to W4/radius2; 10,000 windows mism=0.

## Beta sweep, best engine (2026-11-07)

1000 windows, C=1, W=4, L=21. Acceptance counts change by only a few proposals between beta 2.8 and 3.3; the golden model shows the same effect (2428 vs 2426 accepted of 10,000).

| beta | cycles/window | acceptance | M/s/engine @160.8 MHz |
|---|---|---|---|
| 2.8 | 40.695 | 0.3036 | 15.8 |
| 3.0 | 40.695 | 0.3036 | 15.8 |
| 3.3 | 40.695 | 0.3036 | 15.8 |

The measured edge is 40.695 cycles/window at these betas; the 10,000-window run measured 39.950. This is one engine, not full-device throughput.

## Context packing C=2 not viable without rewrite (2026-11-07)

- `thr_we` writes `thr[thr_addr]` and ignores `thr_ctx`.
- State arrays and RNG/queues are likewise single-context.
- C=2 in the testbench duplicates logic but not context-indexed state; it cannot be bit-exact for distinct betas.
- A real C=2 needs per-context state and threshold storage; not implemented. C=1 stays the selected design.

## Parallel window evaluation (2026-11-07)

`Windowed::commit_parallel` evaluates all W proposals against the pre-window state. It is bit-exact with serial `commit`: 200-window equality for L=7/21, W=1..32, beta=0..3.3. Reason: an accepted move only changes its 14-site neighborhood; a later proposal disjoint from every prior accepted move has identical dE, and an intersecting proposal triggers the existing hazard cut before its evaluation is used.

Measured hazard cut, L=21, beta=2.8, 20,000 windows:

| W | h/window | h/W |
|---|---|---|
| 4 | 3.99 | 0.997 |
| 8 | 7.94 | 0.993 |
| 16 | 15.71 | 0.982 |
| 32 | 30.44 | 0.951 |
| 64 | 56.52 | 0.883 |
| 128 | 94.18 | 0.736 |
| 256 | 135.66 | 0.530 |

Cycle breakdown, current W=4, L=21, 40.6 cycles/window: GEN 11.1, EVAL 8.0, READ 4.0, WAIT 4.0, HAZ 5.2, FINISH 2.0, IDLE 4.8.

A wide read engine with B=W banks, 14-cycle read, and ~4-cycle write projects:

| W | h/window | cycles/window | M/s/chain @200 MHz | x CPU |
|---|---|---|---|---|
| 16 | 15.7 | ~20 | 157 | 3.3 |
| 32 | 30.4 | ~21 | 290 | 6.1 |
| 64 | 56.5 | ~21 | 538 | 11.3 |
| 128 | 94.2 | ~21 | 897 | 18.8 |
| 256 | 135.7 | ~20 | 1357 | 28.4 |

These were projections. The wide core is now built and bit-exact.

## Wide-window RTL, measured (2026-11-07)

`rtl/cslow/wide_core.sv`: W=16, B=64, L=21. `rtl/cslow/tb_wide.cpp`: 1000 windows against `commit_parallel(2)`, full state, h, and accepted mask.

| W | cycles/window | proposals/cycle | M/s @200 MHz | x CPU |
|---|---|---|---|---|
| 16 | 9.471 | 1.69 | 338 | 7.1 |
| 32 | 13.115 | 2.44 | 488 | 10.2 |
| 64 | 18.909 | 3.39 | 677 | 14.2 |
| 128 | 28.549 | 4.48 | 897 | 18.8 |
| 256 | 46.350 | 5.52 | 1105 | 23.1 |
| 512 | 79.149 | 6.47 | 1294 | 27.0 |

All rows: `mism=0`, 1000 windows. Golden exactness separately passed 10,000 windows across 56 W/beta/L combinations. Pre-Fmax, pre-synthesis values; Fmax is not yet known.
