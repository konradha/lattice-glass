"""Checkpoint/result store for the distributed PT campaign.

A `Store` is the durable home for a ladder's artifacts: its single
checkpoint file and the per-temperature config `.npy` files produced at each
checkpoint. Workers are reclaimable cattle, so the store is the only thing
that survives a worker dying mid-run -- a fresh worker leases the ladder and
pulls the checkpoint back to resume exactly where the last one stopped.

The interface is deliberately small and path-based (push a local file in,
pull a stored file out) so it maps cleanly onto a shared POSIX mount today
and onto object storage (S3/Swift) later without changing callers.

Standard library only (os, shutil, tempfile, abc).
"""

import abc
import os
import shutil
import tempfile


class Store(abc.ABC):
    """Abstract checkpoint/result store.

    Implementations MUST make `put_*` operations atomic with respect to
    readers: a concurrent `get_*` either sees the complete previous value or
    the complete new value, never a partially written file. Overwrites are
    allowed and replace the prior value wholesale.
    """

    @abc.abstractmethod
    def put_checkpoint(self, ladder_id: str, local_path: str) -> None:
        """Store the file at `local_path` as `ladder_id`'s checkpoint.

        Atomic; overwrites any previously stored checkpoint for the ladder.
        """

    @abc.abstractmethod
    def get_checkpoint(self, ladder_id: str, local_path: str) -> bool:
        """Copy the stored checkpoint for `ladder_id` to `local_path`.

        Returns True on success. Returns False -- and does not create
        `local_path` -- when no checkpoint is stored for the ladder.
        """

    @abc.abstractmethod
    def put_config(self, ladder_id: str, name: str, local_path: str) -> None:
        """Store the file at `local_path` as config artifact `(ladder_id, name)`.

        Atomic; overwriting an existing artifact of the same name is allowed.
        """

    @abc.abstractmethod
    def list_configs(self, ladder_id: str) -> list:
        """Return the sorted list of config names stored for `ladder_id`.

        Returns an empty list when the ladder has no stored configs.
        """

    @abc.abstractmethod
    def get_config(self, ladder_id: str, name: str, local_path: str) -> bool:
        """Copy stored config `(ladder_id, name)` to `local_path`.

        Returns True on success, False when no such config is stored.
        """


def _check_name(name: str) -> None:
    """Reject config names that could escape their ladder's directory.

    A stored `name` is used as a single path component; anything containing a
    path separator or a parent-directory reference is a traversal attempt.
    """
    if not name or name in (".", ".."):
        raise ValueError("invalid config name: %r" % (name,))
    if "/" in name or "\\" in name or os.sep in name:
        raise ValueError("config name must not contain path separators: %r" % (name,))
    if (os.altsep and os.altsep in name) or ".." in name:
        raise ValueError("config name must not contain path separators: %r" % (name,))


class FilesystemStore(Store):
    """`Store` backed by a directory tree on a (possibly shared) filesystem.

    Layout::

        root/<ladder_id>/checkpoint.bin
        root/<ladder_id>/configs/<name>

    Atomic puts are done by copying into a temp file in the *same* directory
    as the destination, flushing+fsync'ing it, then `os.replace`-ing it over
    the destination. `os.replace` is atomic within a filesystem, so a reader
    never observes a half-written file and no partial file is left behind on
    crash (only a discardable temp file).
    """

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    # -- path helpers -----------------------------------------------------

    def _ladder_dir(self, ladder_id: str) -> str:
        return os.path.join(self.root, ladder_id)

    def _checkpoint_path(self, ladder_id: str) -> str:
        return os.path.join(self._ladder_dir(ladder_id), "checkpoint.bin")

    def _configs_dir(self, ladder_id: str) -> str:
        return os.path.join(self._ladder_dir(ladder_id), "configs")

    def _config_path(self, ladder_id: str, name: str) -> str:
        return os.path.join(self._configs_dir(ladder_id), name)

    # -- primitives -------------------------------------------------------

    @staticmethod
    def _atomic_copy(src_path: str, dest_path: str) -> None:
        """Atomically replace `dest_path` with a copy of `src_path`."""
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest_dir, prefix=".tmp.")
        os.close(fd)
        try:
            shutil.copyfile(src_path, tmp)
            f = os.open(tmp, os.O_RDWR)
            try:
                os.fsync(f)
            finally:
                os.close(f)
            os.replace(tmp, dest_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        # Best-effort durability of the rename itself.
        try:
            dir_fd = os.open(dest_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    @staticmethod
    def _fetch(src_path: str, dest_path: str) -> bool:
        """Copy `src_path` to `dest_path`; return False if `src_path` absent."""
        if not os.path.isfile(src_path):
            return False
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        shutil.copyfile(src_path, dest_path)
        return True

    # -- Store interface --------------------------------------------------

    def put_checkpoint(self, ladder_id: str, local_path: str) -> None:
        self._atomic_copy(local_path, self._checkpoint_path(ladder_id))

    def get_checkpoint(self, ladder_id: str, local_path: str) -> bool:
        return self._fetch(self._checkpoint_path(ladder_id), local_path)

    def put_config(self, ladder_id: str, name: str, local_path: str) -> None:
        _check_name(name)
        self._atomic_copy(local_path, self._config_path(ladder_id, name))

    def list_configs(self, ladder_id: str) -> list:
        configs_dir = self._configs_dir(ladder_id)
        try:
            names = os.listdir(configs_dir)
        except FileNotFoundError:
            return []
        return sorted(
            n for n in names if os.path.isfile(os.path.join(configs_dir, n))
        )

    def get_config(self, ladder_id: str, name: str, local_path: str) -> bool:
        _check_name(name)
        return self._fetch(self._config_path(ladder_id, name), local_path)
