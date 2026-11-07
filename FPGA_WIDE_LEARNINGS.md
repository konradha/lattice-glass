# Wide-window FPGA sampler: learnings

Status: single-chain datapath and RNG are bit-exact. Full-device throughput and Fmax are not proven. No completion claim.

## 1. What was proven

### Exact parallel window evaluation

A window of W proposals can be evaluated against the pre-window state, then committed via the existing radius-2 hazard cut.

Reason:
- An accepted move changes only its 14-site neighborhood.
- A later proposal that is disjoint from all prior accepted moves has the same dE.
- A later proposal that intersects a prior accepted move stops the window before its evaluation is used.

This is exact. It is not a new MCMC kernel.

Evidence:
- `golden/lattice_model.hpp`: `commit_parallel`.
- `golden/tests.cpp`: 10,000 windows, W=4..256, L=7/21, beta=0/2.8/3.0/3.3, zero mismatch.
- `golden/wide_stats.cpp`: 420,000 windows, zero mismatch.

Hazard cut at L=21, beta=2.8:

| W | h/window | h/W |
|---:|---:|---:|
| 4 | 3.988 | 0.997 |
| 8 | 7.952 | 0.994 |
| 16 | 15.722 | 0.983 |
| 32 | 30.496 | 0.953 |
| 64 | 56.645 | 0.885 |
| 128 | 96.204 | 0.752 |
| 256 | 138.142 | 0.540 |

### Measured wide core

`rtl/cslow/wide_core.sv`, W=16/B=64, 1000 windows, full state, h, accepted mask:

| W | core cycles/window | exact |
|---:|---:|---|
| 16 | 9.471 before BRAM fix; 10.471 after | yes |
| 32 | 13.115 before BRAM fix | yes |
| 64 | 18.909 before BRAM fix | yes |
| 128 | 28.549 | yes |
| 256 | 46.350 | yes |
| 512 | 79.149 | yes |

### Full chain with sliding RNG

`rtl/cslow/wide_engine.sv` includes exact sliding refill:

| W | full-chain cycles/window | windows | exact |
|---:|---:|---:|---|
| 16 | 12.471 | 1000 | yes |
| 64 | 22.670 | 100 | yes |

`wide_engine` was 26.153 cycles/window with serial refill. Parallel next-substream refill removed the h-cycle penalty.

## 2. Fast feedback loop

Use this order. Each step answers one question.

1. **C++ golden test**
   - Question: is the algorithm exact?
   - Tool: `golden/tests`, `golden/wide_stats`.
   - Cost: seconds.
   - Gate: zero mismatch before any RTL work.

2. **Verilator lint/build**
   - Question: does the RTL elaborate with no warnings?
   - Tool: `verilator --cc --exe --build`.
   - Gate: no latches, no width warnings, no packed-array misuse.
   - Do not suppress warnings. Three of four failures were real semantic bugs.

3. **Short exactness run**
   - Question: does the RTL match the golden state, h, and mask?
   - Tool: `tb_wide`, `tb_engine`, `tb_rng`.
   - Use 1 to 10 windows first. Full-state scans are expensive.
   - Gate: `mism=0`.

4. **Long exactness run**
   - Question: does the stream stay exact after many windows?
   - W=16: 1000 windows is practical.
   - W=64: 1000 windows exceeded 300 s. Use 100 windows plus sparse hashes.
   - Gate: `mism=0`.

5. **Synthesis-only OOC**
   - Question: what is the resource inference and area?
   - Tool: `ooc_wide_synth.tcl`.
   - Gate: no RAM-to-register warning; LUT/FF/BRAM within budget.
   - Do this before place-and-route.

6. **Place-and-route OOC**
   - Question: what is the real Fmax after routing?
   - Tool: `ooc_wide.tcl`.
   - Only for a design that passed synthesis area.
   - Gate: WNS >= 0 at the target clock.

### Testbench cost control

- Comparing all 9,261 sites per window dominates runtime.
- W16/1000 windows took ~50 s; W64/100 windows took ~47 s.
- Add a deterministic state hash for long runs; keep full scans for short runs.
- Download checkpoints and compare offline.
- Divide cycles by the number of windows actually driven. An abort after 64 windows divided by 1000 produced the false 0.602 cycles/window.

## 3. When to synthesize

Do not place-and-route before the memory inference is correct.

Wasted order used here:
1. Full synth and impl of the async-read core.
2. Thirty-minute timeout with no report.
3. Synth-only run.
4. Discovery: `bm_reg with 185600 registers`.

Correct order:
1. Synth-only OOC.
2. Read the synthesis log for `RAM`, `register`, and `LUTRAM` messages.
3. Fix memory style.
4. Re-run synth-only.
5. Only then run implementation for Fmax.

Do not trust Verilator for area. A functionally correct array can synthesize to 185,600 registers if it has an asynchronous read.

## 4. RNG on FPGA

### Golden layout

`golden/lattice_model.hpp` `ChainRng`:
- W substreams.
- Five xoshiro128++ lanes per substream.
- Proposal k uses substream `k mod W`.
- `i` and `j`: lane 0, two 16-bit halves, multiply-shift by N=9261.
- `u`: lanes 1 and 2, 64 bits.
- Lanes 3 and 4 step but are unused.
- Substreams and lanes are jump-separated by the host.

### Sliding window is mandatory

Golden `Windowed` does this:
1. Commit h proposals.
2. Retain proposals h..W-1.
3. Draw only h new proposals.

A block-aligned W-draws-per-window RNG is wrong. `h/W` is about 0.98, so the divergence starts immediately. `tb_engine` was first written block-aligned; that test was invalid.

### Exact parallel refill

At window start the queue covers global indices K..K+W-1. After h:
- Tail slot t uses global index K+W+t.
- Its substream is `(K+t) mod W`.
- Therefore the new proposals are the next draw of substreams K..K+h-1, modulo W.

Implementation:
1. Compute the next proposal for all W substreams in parallel.
2. Append the first h in wrap order.
3. Advance the full 128-bit states of exactly those h substreams.
4. Do not advance states for retained proposals.
5. Set `sub_base = (sub_base + h) mod W`.

This makes refill one cycle. Serial refill costs h cycles and made the full chain 2.3x slower.

### RNG area

- W=16 needs 80 xoshiro128++ lane states.
- W=64 needs 320.
- W=512 needs 2,560.
- Parallel next-proposal logic is proportional to W.
- At large W, RNG area eventually limits scaling.

Evidence:
- `rtl/cslow/wide_rng.sv`: 1000 windows, `mism=0`.
- `rtl/cslow/wide_engine.sv`: sliding exactness, 1000 windows W16, 100 windows W64.

## 5. Verilog and Xilinx mismatches

### Asynchronous read prevents BRAM inference

The first wide core used:

```systemverilog
logic [19:0] bm [0:NB-1][0:BN-1];
assign brd[b] = bm[b][raddr[b]];
```

Vivado reported:

```text
Potential Runtime issue ... RAM bm_reg with 185600 registers
```

Fix:
- One synchronous write port and one registered read port per bank.
- No reset on the memory.
- `(* ram_style = "block" *)` only as a hint.
- Data valid one cycle after the address.
- Share the debug read through the main read port. A second asynchronous debug port can defeat inference.

Cost: one extra cycle. W16 core went from 9.471 to 10.471 cycles/window.

### Latches

Conditionally assigned combinational temporaries infer latches. The first `accf` had latches on `dv/tn/cn/de_c`. Move the score into an automatic function with local variables; assign every output on every path.

### Packed arrays and Verilator

Verilator maps a packed `[W-1:0][14:0]` port to `VlWide` word storage, not a C++ array of lanes.

Wrong:
```cpp
dut->pi[k] = value;  // writes 32-bit word k
```

Correct:
```cpp
wide_put(dut->pi, 15*k, 15, value);
```

The wrong form produced garbage proposals and a fake RTL mismatch.

### Width bugs are semantic bugs

A packed `logic [W-1:0] pu_r` made `pu_r[q]` a 1-bit select. The accept test compared 1 bit of the uniform with a 64-bit threshold. Verilator warned; the warning was the bug.

### Synthesis warnings can be buried

The 185,600-register warning appears in the synth log, not in Verilator. Always grep the synth log for:
- `registers`
- `RAM`
- `LUTRAM`
- `inferred`
- `trimmed`

### Combinational crossbars

Variable-index merge loops become wide mux trees. The bank scheduler is the area hot spot, not the RAM bits. A 64-bank, W=16 core has many variable selects. Synthesis time and LUT use reflect this.

### Timeouts

- A 30-minute synth+impl run timed out before reporting.
- Synth-only also exceeded 30 minutes in one run.
- Treat long synthesis as an area signal, not only a tool problem.

## 6. Speed and parallelization

### Single-chain, measured cycles

| Design | cycles/window | samples/s at 100 MHz | samples/s at 160 MHz |
|---|---:|---:|---:|
| W16 full | 12.471 | 128M | 205M |
| W64 full | 22.670 | 282M | 452M |

200 MHz is not measured. The narrow W=4 core reached 160.8 MHz post-route. The wide core Fmax is unknown.

### Parallel copies

- Each chain needs 64 banks.
- VU47P has 2,688 RAMB36.
- A 145x20 bank is smaller than one RAMB36; packing controls the chain count.
- Conservative bound: 42 chains. Two-bank packing could double it.
- The logic area may bind earlier.

At 42 chains and 100 MHz:
- W16: 5.4G/s/FPGA.
- W64: 11.8G/s/FPGA.

At 42 chains and 160 MHz:
- W16: 8.6G/s/FPGA.
- W64: 19.0G/s/FPGA.

Target is 13.5G/s/FPGA. W64 at 160 MHz can meet it if area permits 42 cores and Fmax holds. This is not measured.

### 1,000 chains

1,000 x 1e9 sweeps is 9.26e15 proposals.

At 8 FPGAs x 42 chains x W64 x 100 MHz:
- Aggregate: 94.7G/s.
- Time: 27.2 h. Miss.

At 160 MHz:
- Aggregate: 151G/s.
- Time: 17.0 h. Passes, if area and Fmax hold.

No full-device build exists.

## 7. Process failures

- Two agents edited `wide_core.sv` at the same time. One left the file mid-edit with undeclared signals. Use one owner per file.
- A cancelled agent must be stopped before a repair agent starts.
- A corrective agent must get the exact failing command and log lines.
- Do not accept a testbench that changes the reference algorithm. Block alignment was a correctness failure, not a test shortcut.
- Do not claim a throughput number from cycles alone. Fmax and area are required.

## 8. Useful commands

```bash
# Golden exactness
cd golden
c++ -O3 -march=native -std=c++17 tests.cpp -o tests
./tests
c++ -O3 -march=native -std=c++17 wide_stats.cpp -o wide_stats
./wide_stats

# Wide core exactness
verilator --cc --exe --build -j 8 -O3 --no-timing --top-module wide_core \
  -CFLAGS "-O3 -std=c++17 -I$PWD -DTB_W=16 -DTB_NB=64" \
  -LDFLAGS "-lm" -Mdir obj_dir_wide -GW=16 -GNB=64 \
  rtl/cslow/wide_core.sv rtl/cslow/tb_wide.cpp
./obj_dir_wide/Vwide_core 1000

# Full chain
verilator --cc --exe --build -j 8 -O3 --no-timing --top-module wide_engine \
  -CFLAGS "-O3 -std=c++17 -I$PWD -DTB_W=16" \
  -LDFLAGS "-lm" -Mdir obj_dir_engine -GW=16 -GNB=64 \
  rtl/cslow/wide_core.sv rtl/cslow/wide_engine.sv rtl/cslow/tb_engine.cpp
./obj_dir_engine/Vwide_engine 1000 2.8

# Synthesis only
vivado -mode batch -source rtl/cslow/ooc_wide_synth.tcl
```

## 9. Sources

| Source | Use |
|---|---|
| Yoshihiko Nishikawa, Koji Hukushima, "Lattice glass model in three spatial dimensions", arXiv:2003.02872, https://arxiv.org/abs/2003.02872 | Model, energy, nonlocal swap definition |
| Metropolis et al., 1953, https://doi.org/10.1063/1.1699114 | Acceptance rule |
| Hastings, 1970, https://doi.org/10.1093/biomet/57.1.97 | General MCMC acceptance |
| Brockwell, 2006, "Parallel Markov chain Monte Carlo simulation by prefetching", https://doi.org/10.1198/106186006X142663 | Prefetching and rollback |
| Byrd, Jarvis, Bhalerao, 2008, "Parallelization of Markov chain generation and its application to the independence sampler", https://doi.org/10.1007/s11222-008-9053-3 | Speculative parallel MCMC |
| Angelino et al., 2014, "Accelerating MCMC via parallel predictive prefetching", arXiv:1403.7265, https://arxiv.org/abs/1403.7265 | Predictive prefetching |
| Jacob, O'Leary, Atchade, 2020, "Unbiased Markov chain Monte Carlo methods with couplings", https://doi.org/10.1111/rssb.12336 | Coupling; not used for throughput |
| Belletti et al., 2015, "Janus II", https://doi.org/10.1016/j.cpc.2015.05.021 | FPGA Monte Carlo architecture |
| Baity-Jesi et al., 2013, "Janus computer", https://doi.org/10.1007/s10773-013-1592-5 | FPGA spin-update architecture |
| Blackman and Vigna, 2018, "Scrambled linear pseudorandom number generators", arXiv:1805.01407, https://arxiv.org/abs/1805.01407 | xoshiro128++ |
| Vigna, 2017, "Further scramblings of Marsaglia's xorshift generators", https://arxiv.org/abs/1404.0390 | xoshiro construction |
| Xilinx UG901, "Vivado Design Suite User Guide: Synthesis", https://docs.amd.com/v/u/en-US/ug901-vivado-synthesis | RAM inference, `ram_style`, synchronous read |
| Xilinx UG573, "UltraScale Architecture Memory Resources", https://docs.amd.com/v/u/en-US/ug573-ultrascale-memory-resources | BRAM/LUTRAM capabilities |
| Xilinx UG949, "Vivado Design Methodology", https://docs.amd.com/v/u/en-US/ug949-vivado-design-methodology | OOC synthesis and implementation flow |
| Xilinx UG903, "Vivado Using Constraints", https://docs.amd.com/v/u/en-US/ug903-vivado-using-constraints | Clock constraints |
| Xilinx UG906, "Vivado Design Analysis and Closure Techniques", https://docs.amd.com/v/u/en-US/ug906-vivado-design-analysis-and-closure-techniques | Timing reports and WNS |
| Xilinx UG575, "UltraScale+ FPGA Packaging and Pinout", https://docs.amd.com/v/u/en-US/ug575-ultrascale-pkg-pinout | VU47P package |
| Verilator manual, https://verilator.org/guide/latest/ | `VlWide` packed-port representation, warning handling |
| Kogge, 1981, "The Architecture of Pipelined Computers" | Multi-ported memory and banked state |

## 10. Open work

- Real Fmax and area for the BRAM wide core.
- Area of the threshold table and bank crossbar.
- Full-device build with a measured chain count.
- Beta 2.8/3.0/3.3 full-chain throughput.
- 1e10-proposal checkpoint equality.

No target completion claim.
