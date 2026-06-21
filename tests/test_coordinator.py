import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dist.coordinator import Coordinator


def make_spec(n, prefix="lad"):
    """Build a campaign spec with ``n`` deterministically-named ladders."""
    return {
        "ladders": [
            {
                "ladder_id": "%s%d" % (prefix, i),
                "L": 8,
                "n_temps": 16,
                "beta_min": 0.1,
                "beta_max": 2.0,
                "seed": 1000 + i,
                "warmup": 100,
                "sample_every": 10,
                "exchange_every": 5,
                "target_configs": 50,
            }
            for i in range(n)
        ]
    }


class CoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.db = os.path.join(self.tmp, "coord.db")
        self.t = [1000.0]
        self.coord = Coordinator(self.db, lease_default_secs=3600, clock=lambda: self.t[0])
        self.addCleanup(self.coord.close)

    def _ladder(self, ladder_id):
        return next(x for x in self.coord.status()["ladders"] if x["ladder_id"] == ladder_id)

    # -- upsert idempotency ----------------------------------------------------

    def test_upsert_inserts_and_is_idempotent(self):
        self.assertEqual(self.coord.upsert_campaign(make_spec(3)), 3)

        # Advance progress on one ladder via claim + heartbeat.
        granted = self.coord.claim("w1", "x86_64")
        self.assertIsNotNone(granted)
        hb = self.coord.heartbeat("w1", granted["ladder_id"], 42, 7)
        self.assertEqual(hb, {"ok": True, "revoked": False})

        # Re-upsert the same spec: nothing new, progress/status preserved.
        self.assertEqual(self.coord.upsert_campaign(make_spec(3)), 0)
        row = self._ladder(granted["ladder_id"])
        self.assertEqual(row["sweeps_done"], 42)
        self.assertEqual(row["configs_done"], 7)
        self.assertEqual(row["status"], "leased")
        self.assertEqual(row["owner"], "w1")

    def test_upsert_adds_only_missing_ladders(self):
        self.assertEqual(self.coord.upsert_campaign(make_spec(2)), 2)
        # Superset spec: 2 existing kept, 2 new inserted.
        self.assertEqual(self.coord.upsert_campaign(make_spec(4)), 2)
        self.assertEqual(len(self.coord.status()["ladders"]), 4)

    # -- claim -----------------------------------------------------------------

    def test_claim_grants_distinct_then_exhausts(self):
        self.coord.upsert_campaign(make_spec(2))

        a = self.coord.claim("w1", "x86_64")
        self.assertIsNotNone(a)
        row_a = self._ladder(a["ladder_id"])
        self.assertEqual(row_a["status"], "leased")
        self.assertEqual(row_a["owner"], "w1")
        self.assertEqual(row_a["arch"], "x86_64")

        b = self.coord.claim("w2", "x86_64")
        self.assertIsNotNone(b)
        self.assertNotEqual(a["ladder_id"], b["ladder_id"])

        # Both leased and unexpired -> nothing eligible.
        self.assertIsNone(self.coord.claim("w3", "x86_64"))

    # -- arch affinity ---------------------------------------------------------

    def test_arch_affinity_blocks_mismatched_reclaim(self):
        self.coord.upsert_campaign(make_spec(1))
        a = self.coord.claim("w1", "x86_64")
        self.assertEqual(a["ladder_id"], "lad0")

        # Expire the lease.
        self.t[0] += 3601

        # aarch64 cannot reclaim an x86_64-pinned ladder.
        self.assertIsNone(self.coord.claim("w2", "aarch64"))

        # x86_64 can.
        reclaimed = self.coord.claim("w3", "x86_64")
        self.assertEqual(reclaimed["ladder_id"], "lad0")
        self.assertEqual(self._ladder("lad0")["owner"], "w3")

    # -- stale lease reclaim ---------------------------------------------------

    def test_stale_lease_reclaim(self):
        self.coord.upsert_campaign(make_spec(1))
        a = self.coord.claim("w1", "x86_64", lease_secs=100)
        self.assertEqual(a["ladder_id"], "lad0")

        # Just before expiry: not reclaimable (lease_expiry == 1100, now 1099).
        self.t[0] += 99
        self.assertIsNone(self.coord.claim("w2", "x86_64"))

        # Past expiry: reclaimable by same arch.
        self.t[0] += 2  # now 1101 > 1100
        reclaimed = self.coord.claim("w2", "x86_64")
        self.assertEqual(reclaimed["ladder_id"], "lad0")
        self.assertEqual(self._ladder("lad0")["owner"], "w2")

    # -- heartbeat -------------------------------------------------------------

    def test_heartbeat_owner_then_revoked(self):
        self.coord.upsert_campaign(make_spec(1))
        self.coord.claim("w1", "x86_64", lease_secs=100)

        hb = self.coord.heartbeat("w1", "lad0", 10, 2, lease_secs=100)
        self.assertEqual(hb, {"ok": True, "revoked": False})
        row = self._ladder("lad0")
        self.assertEqual(row["sweeps_done"], 10)
        self.assertEqual(row["configs_done"], 2)

        # Expire and let another worker reclaim.
        self.t[0] = 1101
        reclaimed = self.coord.claim("w2", "x86_64", lease_secs=100)
        self.assertEqual(reclaimed["ladder_id"], "lad0")

        # Original owner's heartbeat is now revoked and must not change progress.
        hb2 = self.coord.heartbeat("w1", "lad0", 999, 999)
        self.assertEqual(hb2, {"ok": True, "revoked": True})
        row = self._ladder("lad0")
        self.assertEqual(row["sweeps_done"], 10)
        self.assertEqual(row["configs_done"], 2)
        self.assertEqual(row["owner"], "w2")

    # -- complete --------------------------------------------------------------

    def test_complete_release_done_and_monotonic_progress(self):
        self.coord.upsert_campaign(make_spec(1))
        self.coord.claim("w1", "x86_64")

        # done=False releases back to pending and records progress.
        self.assertEqual(self.coord.complete("w1", "lad0", 30, 5, done=False), {"ok": True})
        row = self._ladder("lad0")
        self.assertEqual(row["status"], "pending")
        self.assertIsNone(row["owner"])
        self.assertEqual(row["sweeps_done"], 30)

        # Re-claimable; claim surfaces accumulated progress.
        b = self.coord.claim("w2", "x86_64")
        self.assertEqual(b["ladder_id"], "lad0")
        self.assertEqual(b["sweeps_done"], 30)
        self.assertEqual(b["configs_done"], 5)

        # A stale lower report must not regress progress (max wins).
        self.coord.complete("w2", "lad0", 5, 1, done=False)
        row = self._ladder("lad0")
        self.assertEqual(row["sweeps_done"], 30)
        self.assertEqual(row["configs_done"], 5)

        # done=True finalizes; ladder is no longer claimable.
        self.coord.complete("w2", "lad0", 60, 50, done=True)
        row = self._ladder("lad0")
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["sweeps_done"], 60)
        self.assertEqual(row["configs_done"], 50)
        self.assertIsNone(self.coord.claim("w3", "x86_64"))

    # -- status ----------------------------------------------------------------

    def test_status_totals_add_up(self):
        self.coord.upsert_campaign(make_spec(4))
        self.coord.claim("w1", "x86_64")
        self.coord.claim("w2", "x86_64")

        leased = [x for x in self.coord.status()["ladders"] if x["status"] == "leased"]
        self.assertEqual(len(leased), 2)

        self.coord.complete("w1", leased[0]["ladder_id"], 100, 9, done=True)

        totals = self.coord.status()["totals"]
        self.assertEqual(totals["done"], 1)
        self.assertEqual(totals["leased"], 1)
        self.assertEqual(totals["pending"], 2)
        self.assertEqual(totals["pending"] + totals["leased"] + totals["done"], 4)
        self.assertEqual(totals["configs_done_sum"], 9)


if __name__ == "__main__":
    unittest.main()
