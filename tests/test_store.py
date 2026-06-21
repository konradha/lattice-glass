import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dist.store import FilesystemStore, Store  # noqa: E402


class FilesystemStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.store = FilesystemStore(os.path.join(self.root, "store"))
        # Scratch area for the "local" files workers push/pull.
        self.scratch = os.path.join(self.root, "scratch")
        os.makedirs(self.scratch, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    # -- helpers ----------------------------------------------------------

    def _write(self, name, data):
        path = os.path.join(self.scratch, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _read(self, path):
        with open(path, "rb") as f:
            return f.read()

    # -- abstract base ----------------------------------------------------

    def test_store_is_abstract(self):
        with self.assertRaises(TypeError):
            Store()  # cannot instantiate the ABC directly
        self.assertTrue(issubclass(FilesystemStore, Store))

    # -- checkpoints ------------------------------------------------------

    def test_put_get_checkpoint_roundtrips_bytes(self):
        data = bytes(range(256)) + b"checkpoint-payload"
        src = self._write("ckpt.bin", data)
        self.store.put_checkpoint("L0", src)

        dest = os.path.join(self.scratch, "fetched.bin")
        self.assertTrue(self.store.get_checkpoint("L0", dest))
        self.assertEqual(self._read(dest), data)

    def test_get_checkpoint_absent_returns_false_and_no_file(self):
        dest = os.path.join(self.scratch, "missing.bin")
        self.assertFalse(self.store.get_checkpoint("nope", dest))
        self.assertFalse(os.path.exists(dest))

    def test_put_checkpoint_overwrites(self):
        first = self._write("c1.bin", b"first-version")
        self.store.put_checkpoint("L0", first)
        second = self._write("c2.bin", b"second-version-longer")
        self.store.put_checkpoint("L0", second)

        dest = os.path.join(self.scratch, "out.bin")
        self.assertTrue(self.store.get_checkpoint("L0", dest))
        self.assertEqual(self._read(dest), b"second-version-longer")

    # -- configs ----------------------------------------------------------

    def test_config_put_list_get(self):
        a = self._write("a.npy", b"\x93NUMPY-A-bytes")
        b = self._write("b.npy", b"\x93NUMPY-B-bytes-different-length")
        # Stored out of order to confirm list_configs sorts.
        self.store.put_config("L1", "cfg_T1.0000.npy", b)
        self.store.put_config("L1", "cfg_T0.5000.npy", a)

        self.assertEqual(
            self.store.list_configs("L1"),
            ["cfg_T0.5000.npy", "cfg_T1.0000.npy"],
        )

        dest = os.path.join(self.scratch, "cfg_out.npy")
        self.assertTrue(self.store.get_config("L1", "cfg_T0.5000.npy", dest))
        self.assertEqual(self._read(dest), b"\x93NUMPY-A-bytes")

    def test_list_configs_empty_for_unknown_ladder(self):
        self.assertEqual(self.store.list_configs("ghost"), [])

    def test_get_config_absent_returns_false_and_no_file(self):
        self.store.put_config("L1", "present.npy", self._write("p.npy", b"x"))
        dest = os.path.join(self.scratch, "absent_out.npy")
        self.assertFalse(self.store.get_config("L1", "missing.npy", dest))
        self.assertFalse(os.path.exists(dest))

    def test_put_config_overwrites(self):
        self.store.put_config("L1", "c.npy", self._write("v1", b"old"))
        self.store.put_config("L1", "c.npy", self._write("v2", b"new-bytes"))
        dest = os.path.join(self.scratch, "c_out.npy")
        self.assertTrue(self.store.get_config("L1", "c.npy", dest))
        self.assertEqual(self._read(dest), b"new-bytes")
        # Overwrite must not duplicate the listing entry.
        self.assertEqual(self.store.list_configs("L1"), ["c.npy"])

    # -- traversal guard --------------------------------------------------

    def test_traversal_guard_rejects_bad_names(self):
        src = self._write("ok.npy", b"data")
        dest = os.path.join(self.scratch, "out.npy")
        for bad in ("../evil", "a/b", "..", "", "a\\b", "sub/../x"):
            with self.assertRaises(ValueError):
                self.store.put_config("L1", bad, src)
            with self.assertRaises(ValueError):
                self.store.get_config("L1", bad, dest)
        # A rejected put must not leave the destination behind.
        self.assertFalse(os.path.exists(dest))

    # -- atomicity --------------------------------------------------------

    def test_atomic_put_leaves_no_temp_files(self):
        data = b"payload" * 1000
        self.store.put_checkpoint("L9", self._write("big.bin", data))
        self.store.put_config("L9", "cfg.npy", self._write("cfg.npy", data))

        ladder_dir = os.path.join(self.store.root, "L9")
        ckpt = os.path.join(ladder_dir, "checkpoint.bin")
        self.assertTrue(os.path.isfile(ckpt))
        self.assertEqual(os.path.getsize(ckpt), len(data))

        configs_dir = os.path.join(ladder_dir, "configs")
        leftovers = []
        for d in (ladder_dir, configs_dir):
            for entry in os.listdir(d):
                if ".tmp" in entry:
                    leftovers.append(os.path.join(d, entry))
        self.assertEqual(leftovers, [])

    # -- isolation --------------------------------------------------------

    def test_ladders_are_isolated(self):
        self.store.put_config("A", "shared.npy", self._write("s.npy", b"A-data"))

        self.assertEqual(self.store.list_configs("A"), ["shared.npy"])
        self.assertEqual(self.store.list_configs("B"), [])

        dest = os.path.join(self.scratch, "b_out.npy")
        self.assertFalse(self.store.get_config("B", "shared.npy", dest))
        self.assertFalse(os.path.exists(dest))

        # Checkpoints are likewise per-ladder.
        self.store.put_checkpoint("A", self._write("ack.bin", b"A-ckpt"))
        self.assertFalse(
            self.store.get_checkpoint("B", os.path.join(self.scratch, "b_ckpt.bin"))
        )


if __name__ == "__main__":
    unittest.main()
