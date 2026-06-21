import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dist.store import FilesystemStore  # noqa: E402
from dist.worker import Worker, http_get_json, parse_progress  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(REPO_ROOT, "experiments", "production_sampler")


class ParseProgressTest(unittest.TestCase):
    """Always-run unit coverage of the stdout progress parser."""

    def test_last_checkpoint_wins_wall_stop(self):
        text = (
            "[rank seed=1] fresh at sweeps=0 (L=6, 3 temps, T in [1.000,3.333])\n"
            "[seed=1] sweeps=10 collected[cold]=1/2 E_cold/site=-1.50000 "
            "exch_acc=0.250 wall=5s\n"
            "[seed=1] sweeps=20 collected[cold]=1/2 E_cold/site=-1.60000 "
            "exch_acc=0.300 wall=12s WALL-STOP\n"
        )
        self.assertEqual(parse_progress(text), (20, 1, False))

    def test_last_checkpoint_done(self):
        text = (
            "[rank seed=1] fresh at sweeps=0 (L=6, 3 temps, T in [1.000,3.333])\n"
            "[seed=1] sweeps=10 collected[cold]=1/2 E_cold/site=-1.50000 "
            "exch_acc=0.250 wall=5s\n"
            "[seed=1] sweeps=20 collected[cold]=2/2 E_cold/site=-1.60000 "
            "exch_acc=0.300 wall=12s DONE\n"
        )
        self.assertEqual(parse_progress(text), (20, 2, True))

    def test_empty_and_garbage(self):
        self.assertEqual(parse_progress(""), (0, 0, False))
        self.assertEqual(parse_progress("nothing here\nstill nothing\n"), (0, 0, False))


@unittest.skipUnless(os.path.exists(BINARY), "sampler not built")
class WorkerEndToEndTest(unittest.TestCase):
    """Drive a real coordinator + sampler through one ladder to DONE."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
        finally:
            s.close()

    def test_lease_run_complete(self):
        port = self._free_port()
        base = "http://127.0.0.1:%d" % port
        db = os.path.join(self.tmp, "coord.db")
        campaign = os.path.join(self.tmp, "campaign.json")
        store_root = os.path.join(self.tmp, "store")
        workdir = os.path.join(self.tmp, "work")

        with open(campaign, "w") as fh:
            json.dump({"ladders": [{
                "ladder_id": "t1", "L": 6, "n_temps": 3,
                "beta_min": 0.3, "beta_max": 1.0, "seed": 1,
                "warmup": 0, "sample_every": 2, "exchange_every": 2,
                "target_configs": 2,
            }]}, fh)

        proc = subprocess.Popen(
            [sys.executable, "-m", "dist.coordinator",
             "--db", db, "--host", "127.0.0.1", "--port", str(port),
             "--campaign", campaign, "--lease-default-secs", "60"],
            cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            # Wait for the coordinator's HTTP server to accept requests.
            deadline = time.time() + 10
            up = False
            while time.time() < deadline:
                if proc.poll() is not None:
                    out = proc.stdout.read() if proc.stdout else ""
                    self.fail("coordinator exited early:\n%s" % out)
                try:
                    http_get_json(base + "/status", timeout=2)
                    up = True
                    break
                except Exception:
                    time.sleep(0.1)
            self.assertTrue(up, "coordinator did not come up in time")

            store = FilesystemStore(store_root)
            worker = Worker(base, store, BINARY, workdir, lease_secs=40, poll_secs=1)
            worker.run(once=True)

            status = http_get_json(base + "/status")
            ladders = {lad["ladder_id"]: lad for lad in status["ladders"]}
            self.assertIn("t1", ladders)
            self.assertEqual(ladders["t1"]["status"], "done")
            self.assertGreaterEqual(ladders["t1"]["configs_done"], 2)

            pulled = os.path.join(self.tmp, "pulled_ckpt.bin")
            self.assertTrue(store.get_checkpoint("t1", pulled))
            self.assertGreaterEqual(len(store.list_configs("t1")), 1)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if proc.stdout:
                proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
