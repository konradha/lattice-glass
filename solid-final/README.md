# solid-final

Selected CPU, Metal and CUDA samplers, with one front door and bounded campaign output.
**Packaging only: no builds, simulations, correctness-suite reruns, benchmarks or Modal jobs were run for this directory.** Historical qualification belongs to the original sources; relocated/adapted host executables have not been newly qualified.

## Start here

Python 3.10+, NumPy; C++17 and header-only Boost (`brew install boost` / `apt install libboost-dev`). Metal requires macOS; CUDA requires CUDA 12.8+. The Metal C++ SDK headers and license are included. No silent GPU-to-CPU fallback.

```sh
python3 -m pip install -r solid-final/requirements.txt
python3 solid-final/driver.py info
python3 solid-final/driver.py build --backend cpu
# Apple GPU; builds the appropriate native executable on first use:
python3 solid-final/driver.py run --backend metal --mode nonlocal \
  --beta 2.5 --chains 256 --sweeps 1000 --seed 123 --output-dir runs/apple
# Inspect existing results on any laptop; no GPU or Modal SDK needed:
python3 solid-final/driver.py inspect runs/apple/run.npz
```

`--dry-run` prints commands/plans without compiling or simulating. `solid-final/driver` also accepts the original positional driver ABI, so existing scripts can use it directly. `make -C solid-final cpu|metal|cuda-depth|cuda-throughput|cuda-ampere` builds only: no PGO training or hidden benchmark.

## Backend selection

- **CPU:** Darwin preserves the measured opt2 physical CPU algorithm and O2/native flags. Other POSIX hosts use the validated opt3 cached K/type-fast algorithm, with a production host adapter. No measured Apple cache speed is asserted.
- **Metal nonlocal:** opt2 persistent exact prefix-cut engine. **Physical local:** opt physical random-edge engine. `--mode both` uses the physical engine for nonlocal then local ordering.
- **Metal `--mode local-colored`:** explicit opt-in, L21 only. Faster equilibrium-update schedule, **not physical local dynamics**; never use its sweep clock for propensity/physical relaxation. Historical stationarity evidence is L21, beta 2.0; it is not a trajectory-equivalence gate or a deep-beta measurement.
- **CUDA `depth`:** C07 `fast_empty_window`, selected L40S single-chain/depth engine.
- **CUDA `throughput`:** C02 `packed_type_commit`, selected L40S whole-card engine; historical saturation point is 1136 chains, not 1136 simultaneously resident CTAs.
- **CUDA `ampere`:** C08 `packed_fast_empty`, selected A100 engine. `auto` on Modal chooses this for A100-40GB, depth otherwise. H100 and L4 runtime performance/qualification are **not** established by these final selections.

Historical measurements, not new package timings:

| Implementation / configuration | Chains | Local M proposals/s/chain | Nonlocal M proposals/s/chain |
|---|---:|---:|---:|
| Apple CPU opt2, beta 2.0 | 8 | 16.60 | 25.30 |
| non-Apple cached CPU, beta 3.2 | 1 | 47.90 | 51.11 |
| L40S C07 depth, beta 3.2 | 1 | 169.67 | 207.96 |
| L40S C02 saturation, beta 3.2 | 1136 | 35.95 | 42.59 |
| A100 C08, beta 3.2 | 1 | 95.56 | 116.82 |

These differ in hardware, beta and timing scope: CPU cache numbers are kernel-only; the historical Apple driver includes NPZ work. Do not infer a controlled cross-platform speedup. Full Metal scaling, CUDA scaling, timing scopes, qualification limits and raw receipts are preserved in `*/selection.json`, `research/REPORT.md`, and `benchmarks/`.

All rates count proposals, including invalid/equal-site draws, no-ops and rejections. **One physical local or nonlocal sweep is L³ proposals: 9261 at L21.** `both` issues 2L³. A proposal rate is not an independent-equilibrium-configuration rate.

## Local or CUDA-host campaigns

```sh
python3 solid-final/driver.py campaign --backend metal \
  --betas 0.01,2.0,2.5 --modes nonlocal --chains 16384 \
  --equil-sweeps 100000 --sweeps 100000 --seed 123 \
  --batch-chains 256 --output-dir runs/ensemble
# Continue exactly from the last committed batch states/counters:
# repeat the same command with --resume
```

CUDA hosts use `--backend cuda --preset depth`, or the whole-card `--preset throughput --batch-chains 1136` on L40S. Automatic batch size is CPU logical workers (capped 64), Metal 256, or the actual CUDA SM count. These are scheduling choices, not universal hardware-optimum claims.

`paper_campaign.json` preserves the **historical rounded ten-beta ladder and sweep schedules from `solid/fold_campaign.py`**, with 16384 nonlocal chains and 32 local chains per beta. Edit the explicit schedule/beta list for another campaign:

```sh
python3 solid-final/driver.py campaign --config solid-final/paper_campaign.json \
  --backend cpu --output-dir runs/paper --max-wall-seconds 600
```

The recipe's local chains use their own physical burn-in. To instead start physical/propensity measurements from nonlocal configurations, use the seeding runner below. Neither the historical recipe nor configured burn-in is a measured guarantee of equilibration, tau, or a maximum accessible beta.

Campaigns use domain-separated full-u64 batch seeds, preserve absolute sweep counters, stream immutable observation shards, and retain only the latest checkpoint per batch. The manifest stays bounded with sweep depth; events stream to JSONL. Global history cadence is `max(history_every,ceil(total_production_sweeps/max_history_samples))`; global power-of-two snapshots and the final state are retained. No billion-element in-memory history. Publication precedes old-checkpoint deletion; an interrupted native chunk rolls back to the last committed state. Resume checks settings, batch identities, native executable hashes and checkpoint hashes. Keep native binaries/build volumes for exact resume.

## Modal, explicitly budgeted

Install `modal` and authenticate normally on the client. **These commands launch paid work; none was invoked while packaging.**

```sh
python3 solid-final/driver.py modal --budget-usd 5 \
  --config solid-final/paper_campaign.json --gpu L40S --preset depth --seconds 600
# Use the printed run id and ledger, with unchanged settings/cap:
python3 solid-final/driver.py modal --budget-usd 5 \
  --config solid-final/paper_campaign.json --gpu L40S --preset depth --seconds 600 \
  --resume-id solid-final-PRINTED_ID --ledger PRINTED_LEDGER
modal volume get lattice-solid-final-runs solid-final-PRINTED_ID ./download
```

Pinned CUDA base **and digest**, CPU-only native build cached on `lattice-solid-final-builds`, mounted runtime sources, no Python-tool-driven image rebuilds and no benchmark archive upload. Every invocation requires positive budget admission; cumulative conservative reservations remain on failures. Queue waits/timeouts are bounded, with no automatic retries/warm containers. The ledger is an estimate/admission control, **not a provider invoice or account-level spending cap**. Outputs/checkpoints go to `lattice-solid-final-runs`; logs include actual SM count and clocks. A returned deadline is resumable progress, not a completed campaign.

## Export, physical seeding, existing analysis

```sh
python3 solid-final/runners/export_configs.py --manifest runs/ensemble/manifest.json \
  --mode nonlocal --stage equilibrated --output-dir exports/equilibrium
# Memory-mappable configurations.npy, total energies.npy, batch seeds and native row ids.
python3 solid-final/runners/local_from_nonlocal.py --input exports/equilibrium \
  --backend metal --configs 32 --replicas 8 --sweeps 100000 \
  --seed 456 --output-dir runs/propensity
```

Each initial configuration is repeated `replicas` times with distinct native chain RNG keys. The physical local runner also reads existing native/aging NPZ files, retains replica mapping, and supports `--dry-run`. It runs on the selected local machine/CUDA host; it does not silently upload seed files or launch Modal.

`runners/aging_driver.py` is the existing beta/explicit-fitted-tau aggregate runner, routed through the new front door. Use explicit `--sweeps` and `--equilibration`, or supply fitted tau parameters; no paper fit is invented. `runners/plot_paper.py` is the preserved paper-style postprocessor (optional `matplotlib`). These native/aging NPZ files are different from streaming campaign observation shards; export configurations with `export_configs.py`, not by concatenating shards as if each were a fresh trajectory.

## Physics and evidence

Frozen CPU headers match upstream `eaef7d4` and opt3 byte-for-byte. Independent nonlocal site draws include counted identical-site invalid proposals; equal types are no-ops. Physical local uses uniform nearest-neighbor proposals. Packed canonical state, composition, integer delta energy, CounterSplitMix/Lemire proposal mapping, and precomputed 64-bit EFMA thresholds are unchanged. No exponential in proposal hot paths. Supported public range L4–21, beta>0; the inherited beta=0 EFMA edge is deliberately not advertised. Physical Metal's preserved kernel is limited to exclusive next sweep <=2³²; the facade enforces nonwrapping proposal counters for all backends.

`PROVENANCE.json` identifies the package and source snapshots. `research/` is the original optimization report/proofs/notes/results, not a claim that all experiments are selected backends. `benchmarks/INDEX.txt`, `manifest.json`, per-file manifests and exclusions preserve the raw evidence, including failures and incomplete campaigns. Extraction example:

```sh
python3 solid-final/benchmarks/extract.py --root opt3 \
  --path solid-opt3/artifacts/cpu-type-fast-measured.json --output /tmp/cpu-measured.json
```
