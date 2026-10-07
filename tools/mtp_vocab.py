"""Atomically install a draft vocabulary into a locked MTP runtime directory."""
from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
from pathlib import Path

from artifact_set_lock import (  # noqa: E402
    ArtifactSetLockError,
    LOCK_FDS_ENV,
    acquire_artifact_set_locks,
    clear_generation_pending,
    mark_generation_pending,
    pending_marker,
)

VOCAB_SIZE = 248_320
DESTINATION_NAME = "draft_vocab.bin"


def validate_vocabulary(source: Path) -> bytes:
    raw = source.read_bytes()
    if not raw or len(raw) % 4 or len(raw) > VOCAB_SIZE * 4:
        raise ValueError(
            f"{source}: expected 1 to {VOCAB_SIZE} little-endian int32 token IDs"
        )
    seen: set[int] = set()
    for (token_id,) in struct.iter_unpack("<i", raw):
        if token_id < 0 or token_id >= VOCAB_SIZE:
            raise ValueError(
                f"{source}: token ID {token_id} is outside [0, {VOCAB_SIZE})"
            )
        if token_id in seen:
            raise ValueError(f"{source}: duplicate token ID {token_id}")
        seen.add(token_id)
    return raw


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def install(source: Path, runtime: Path, inherited: str | None = None) -> None:
    locks = acquire_artifact_set_locks(
        [], [(runtime, True)], inherited=inherited
    )
    owns_generation = not locks.is_borrowed(runtime)
    temporary: Path | None = None
    try:
        if owns_generation and pending_marker(runtime).exists():
            raise ArtifactSetLockError(
                f"{pending_marker(runtime)}: an interrupted MTP runtime generation needs recovery"
            )
        raw = validate_vocabulary(source)
        runtime.mkdir(parents=True, exist_ok=True)
        if owns_generation:
            mark_generation_pending(runtime)
        try:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{DESTINATION_NAME}.tmp.", dir=runtime
            )
            temporary = Path(temporary_name)
            with os.fdopen(fd, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, runtime / DESTINATION_NAME)
            _fsync_directory(runtime)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if owns_generation:
            clear_generation_pending(runtime)
    finally:
        locks.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    args = parser.parse_args()
    try:
        install(
            args.source,
            args.runtime,
            inherited=os.environ.get(LOCK_FDS_ENV),
        )
    except (ArtifactSetLockError, OSError, ValueError) as exc:
        print(f"mtp_vocab: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
