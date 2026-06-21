"""Distributed campaign infrastructure for the NH lattice-glass PT sampler.

Modules:
  coordinator -- lease-based bag-of-tasks registry (sqlite + http.server)
  store       -- checkpoint/result store (filesystem now; pluggable for S3/Swift)
  worker      -- agent that leases a ladder, drives experiments/production_sampler,
                 and syncs checkpoints + configs back to the store

One ladder = one PT run = one work unit. Workers are preemptible cattle; a
dropped worker's lease expires and the ladder is reclaimed. Designed for
reclaimable, heterogeneous OpenStack capacity.
"""
