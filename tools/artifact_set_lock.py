"""Cooperative locks and crash markers for a GGUF shard set and its derived pack."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Iterable
import fcntl


LOCK_FDS_ENV = "STRATA_ARTIFACT_SET_LOCK_FDS"
PENDING_ENV = "STRATA_ARTIFACT_SET_PENDING"
LOCK_NAME = ".strata-artifact-set.lock"
PENDING_NAME = ".strata-generation-pending"


class ArtifactSetLockError(RuntimeError):
    pass


def lock_path(directory: str | os.PathLike[str]) -> Path:
    resolved = Path(directory).expanduser().resolve()
    return resolved.parent / f".{resolved.name}{LOCK_NAME}"


class _HeldLock:
    def __init__(self, path: Path, fd: int, borrowed: bool):
        self.path = path
        self.fd = fd
        self.borrowed = borrowed

    def close(self) -> None:
        if self.fd >= 0 and not self.borrowed:
            try:
                os.close(self.fd)
                self.fd = -1
            finally:
                self.fd = -1


class ArtifactSetLocks:
    """Locks sources first, then pack directories in path order."""

    def __init__(self, held: list[_HeldLock]):
        self._held = held

    @property
    def inherited_fds(self) -> tuple[int, ...]:
        return tuple(lock.fd for lock in self._held if lock.fd >= 0)

    def inherited_environment(self) -> str:
        return json.dumps(
            {str(lock.path): lock.fd for lock in self._held if lock.fd >= 0},
            separators=(",", ":"),
            sort_keys=True,
        )

    def is_borrowed(self, directory: str | os.PathLike[str]) -> bool:
        path = lock_path(directory)
        return any(lock.path == path and lock.borrowed for lock in self._held)

    def close(self) -> None:
        for lock in reversed(self._held):
            lock.close()
        self._held.clear()

    def __enter__(self) -> "ArtifactSetLocks":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _canonical_directory(directory: str | os.PathLike[str]) -> Path:
    return Path(directory).expanduser().resolve()


def _identity(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino


def _acquire_one(path: Path, exclusive: bool, inherited: dict[str, int]) -> _HeldLock:
    path.parent.mkdir(parents=True, exist_ok=True)
    inherited_fd = inherited.get(str(path))
    if inherited_fd is not None:
        try:
            if _identity(path) != (
                os.fstat(inherited_fd).st_dev,
                os.fstat(inherited_fd).st_ino,
            ):
                raise ArtifactSetLockError(
                    f"inherited artifact lock does not match {path}"
                )
        except OSError as exc:
            raise ArtifactSetLockError(
                f"cannot verify inherited artifact lock {path}: {exc}"
            ) from exc
        return _HeldLock(path, inherited_fd, borrowed=True)

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ArtifactSetLockError(f"cannot open artifact lock {path}: {exc}") from exc
    try:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, operation | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            raise ArtifactSetLockError(
                f"{path}: artifact set is in use by another packer or model load"
            ) from exc
        raise ArtifactSetLockError(f"cannot lock artifact set {path}: {exc}") from exc
    return _HeldLock(path, fd, borrowed=False)


def acquire_artifact_set_locks(
    source_directories: Iterable[str | os.PathLike[str]],
    pack_directories: Iterable[tuple[str | os.PathLike[str], bool]],
    inherited: str | None = None,
    exclusive_sources: bool = False,
) -> ArtifactSetLocks:
    """Acquire source locks, then pack locks (True means exclusive).

    An inherited lock map is used by a packer subprocess. Its file descriptors
    already own the locks, so the child verifies and borrows them.
    """
    inherited_map: dict[str, int] = {}
    if inherited:
        try:
            decoded = json.loads(inherited)
            if not isinstance(decoded, dict):
                raise ValueError("lock map must be an object")
            inherited_map = {str(path): int(fd) for path, fd in decoded.items()}
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactSetLockError(f"invalid inherited artifact lock map: {exc}") from exc

    sources = {_canonical_directory(path) for path in source_directories}
    packs = [
        (_canonical_directory(directory), bool(exclusive))
        for directory, exclusive in pack_directories
    ]
    ordered: dict[Path, bool] = {}
    for directory in sorted(sources):
        ordered[directory] = exclusive_sources
    for directory, exclusive in sorted(packs):
        ordered[directory] = ordered.get(directory, False) or exclusive

    held: list[_HeldLock] = []
    try:
        for directory in sorted(sources):
            held.append(_acquire_one(lock_path(directory), ordered[directory], inherited_map))
        pack_set = {directory for directory, _ in packs} - sources
        for directory in sorted(pack_set):
            held.append(_acquire_one(lock_path(directory), ordered[directory], inherited_map))
    except Exception:
        for lock in reversed(held):
            lock.close()
        raise
    return ArtifactSetLocks(held)


def pending_marker(pack_directory: str | os.PathLike[str]) -> Path:
    return _canonical_directory(pack_directory) / PENDING_NAME


def _fsync_directory(directory: Path) -> None:
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def mark_generation_pending(pack_directory: str | os.PathLike[str]) -> Path:
    directory = _canonical_directory(pack_directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / PENDING_NAME
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(directory)
    return marker


def clear_generation_pending(pack_directory: str | os.PathLike[str]) -> None:
    directory = _canonical_directory(pack_directory)
    marker = directory / PENDING_NAME
    try:
        marker.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(directory)


def done(path: Path) -> bool:
    """Return whether a completed generation marker exists beside ``path``."""
    return path.with_name(path.name + ".done").exists()


def generation_needs_repack(
    pack: Path, shards: list[Path], external_source: bool
) -> bool:
    """Check whether a pack is incomplete or its source is no longer local."""
    return (
        pending_marker(pack).exists()
        or external_source
        or not all(shard.exists() and done(shard) for shard in shards)
    )
