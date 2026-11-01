# Distributed design

## Fixed campaign parameters

- L=25, N=15,625 sites.
- beta_max=3.5.
- One sweep = L^3 = 15,625 proposed swap moves.
- Target: 10,000 chains sweeping at the same moment.
- Sweep budget: 1e8 to 1e9 per chain.
- Statistical reproducibility, not bitwise reproducibility.
- No PT, no collapsed species, no Fourier normalizer, no cluster moves.
- No AI agents in the runtime or the control plane. This is an HPC batch
  problem. The control plane is the Kubernetes Job controller, Karpenter, S3,
  and a small Python submit/collect script.

## Execution model

One chain per pod. One pod per core. One process per pod. No threads, no MPI,
no shared memory.

A chain is identified by the Kubernetes Indexed Job completion index
`0..9999`. The seed is derived from `(run_id, chain_id)`. The chain runs until
its sweep budget is met, checkpointing as it goes.

The scheduler is the Kubernetes Job controller. There is no central lease
service and no consensus protocol. S3 is the source of truth. A read-only
status script may aggregate progress; it has no authority.

## Kubernetes

Use one Indexed Job:

```yaml
completionMode: Indexed
completions: 10000
parallelism: 10000
backoffLimitPerIndex: 20
restartPolicy: Never
terminationGracePeriodSeconds: 300
```

Each pod reads `JOB_COMPLETION_INDEX` and uses it as `chain_id`.

- Requests and limits: 1 CPU, 64 MiB memory, Guaranteed QoS.
- Dedicated node pool with `cpuManagerPolicy: static` for CPU pinning.
- Karpenter with multiple instance types and `capacity-optimized` spot.
- `topologySpreadConstraints` across zones.
- The binary traps `SIGTERM`, stops after the current sweep, checkpoints, and
  uploads. The 300 s grace period is much longer than the measured I/O.
- If a pod is preempted, Kubernetes replaces it with the same index. The new
  pod reads the latest S3 checkpoint and resumes.
- A watchdog aborts a pod that makes no progress for 30 minutes. The pod exits
  with a retryable code; `backoffLimitPerIndex` retries it from checkpoint.
- A container image must be multi-arch or built per node architecture. Do not
  use `-march=native` in a shared image. Use a baseline ISA, or build in an
  init container if the performance loss is unacceptable.

No microVMs on the critical path. Standard `runc` containers with dedicated
cores are the right tool for a CPU-bound kernel. Kata Containers or Firecracker
add isolation and operational cost, not throughput. Use Kata only if
multi-tenant security requires it, and measure the CPU overhead first.
Firecracker without Kubernetes needs a custom orchestrator; that is not worth
it here. The scaling lever is Karpenter plus spot plus the indexed Job.

## S3 layout

```text
s3://<bucket>/runs/<run_id>/
  manifest.json
  chains/<chain_id>/
    latest.json
    epochs/<epoch>.ckpt
    snapshots/<step>.bin
    progress.json
    done.json
```

- `epochs/<epoch>.ckpt` is immutable and written with `If-None-Match: *`.
- `snapshots/<step>.bin` is immutable.
- `latest.json` is mutable and written with `If-Match` on its ETag.
- `done.json` is immutable and contains the final step, checksum, and epoch.
- `manifest.json` contains L, beta ladder, rho, p1, sweep budget, code version,
  and the parameter hash. Every checkpoint carries that hash.

Checkpoint format: fixed header, little-endian, magic, version, run id,
chain id, epoch, step, beta, RNG state or counter, lattice bytes, SHA-256.
A byte lattice at L=25 is 15.6 KiB; a packed lattice is 3.9 KiB. Either is
fine. Verify the hash on read.

Use `aws-sdk-cpp` in a thin store layer. Keep the kernel independent of S3.
S3 supports conditional writes and checksums. Enable bucket versioning,
SSE, and block public access. Use IRSA with least privilege on the run prefix.
No long-lived keys in the image.

## Checkpoint and upload policy

- Local checkpoint every 60 s, or every 10,000 sweeps, whichever is first.
  Atomic local write is about 160 us and is negligible.
- Upload to S3 every 10 minutes. At 0.947 ms/sweep this is about 634k sweeps.
  A preemption loses at most 10 minutes of work.
- Upload on `SIGTERM` as well, if time remains.
- 10,000 pods x 6 uploads/hour = 16.7 PUT/s. S3 handles this easily.
- Snapshot cadence is log-spaced: first at 1,000 sweeps, growth factor 1.15,
  maximum 200 snapshots. At L=25 this is at most 3.2 MiB per chain with byte
  snapshots, or 0.8 MiB packed. 10,000 chains is 8-32 GiB.

## Resume and duplicate writers

The dangerous case is an old pod that wakes after its replacement has written
newer state. Use a monotonic epoch per attempt.

- A new pod writes `epochs/<new_epoch>.ckpt`, never overwrites an old epoch.
- It updates `latest.json` with `If-Match`. If the ETag changed, it re-reads
  and retries. A stale pod's conditional update fails and it stops.
- On resume, read `latest.json`, fetch that epoch, verify the hash and the
  parameter hash. If invalid, list epochs and use the highest valid one.
- `done.json` is written only after the final checkpoint and snapshot.
- The collector reads only chains with a valid `done.json`. It never mixes
  epochs.

This gives at-least-once execution and exactly-once effect. Duplicate pods
produce different epoch keys; only the highest valid epoch is authoritative.

## Fault injection before the campaign

Test each case on the real cluster:

1. `SIGTERM` during a sweep. The pod checkpoints, uploads, exits 0.
2. `SIGKILL` mid-checkpoint. The temporary file is discarded; the previous
   epoch is used.
3. Node loss. Kubernetes recreates the pod; resume from S3.
4. Duplicate pod with the same index. The conditional `latest.json` update
   rejects the stale epoch.
5. Corrupt `latest.json` or checkpoint. Fall back to the previous epoch.
6. S3 503 or timeout. Retry with backoff; do not lose the local checkpoint.
7. Clock skew and long GC pause. The epoch, not wall time, decides authority.

## Annealing and sampling

- beta_min=0.5 or 1.0; beta_max=3.5; delta_beta=0.1; geometric ladder.
- Per-stage sweeps scale as (L/8)^3 (0.75/rho), with a geometric ramp and a
  longer final stage. At L=25 this is a few hundred thousand sweeps total.
  The main 1e8-1e9 sweeps run at beta_max.
- Log acceptance and energy per stage. The schedule is a heuristic. It does
  not prove equilibrium.
- The production phase writes log-spaced snapshots. Each snapshot is a full
  restartable state. This is the "thermalization series" for each sample.
- Equilibrium diagnostics: energy plateau, acceptance stationarity, and
  between-chain variance across the 10,000 independent chains. A `done.json`
  marker means the budget was met, not that the chain is equilibrated.

## Honest limits

1. No PT. The paper uses exchange Monte Carlo, multiple-temperature
   reweighting, and Wang-Landau for a reason. Local and nonlocal swaps alone
   may not equilibrate beta=3.5. Ten thousand cores buy statistics, not
   barrier crossing. Run a 100-chain pilot and check the energy plateau before
   launching the full campaign.
2. At L=25, 1e9 sweeps per chain is about 11 days of wall time on 10,000
   dedicated cores. Spot churn must be tested.
3. A single Indexed Job with 10,000 pods stresses the Kubernetes control
   plane. Test 1,000 pods first; tune the API server, etcd, and controller
   throughput.
4. `If-Match` and `If-None-Match` must be verified on the chosen S3-compatible
   store. Plain S3 supports them; some gateways do not.
5. Heterogeneous CPUs break a native-ISA image. Use a baseline image or
   per-architecture images. Statistical reproducibility is enough; bitwise
   replay is not promised.
6. Cost: about USD 8k-16k for 1e8 sweeps and USD 80k-160k for 1e9 at
   10,000 cores, before storage and egress. Set the budget before launch.
