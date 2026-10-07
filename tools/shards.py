"""Validate downloaded GGUF shards before they reach the Metal engine."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

from gguf_reader import GGUFFile


class ShardCheckError(ValueError):
    """A model shard is missing, malformed, or shorter than its own tensor directory requires."""


def check_shards(shards: Iterable[str | Path]) -> None:
    """Check that each requested shard exists and contains all tensor bytes declared by its header."""
    for raw_path in shards:
        path = Path(raw_path)
        if not path.exists():
            raise ShardCheckError(f"missing {path}")
        try:
            gguf = GGUFFile(path)
        except (ValueError, struct.error) as error:
            raise ShardCheckError(
                f"{path.name} is not a whole GGUF shard ({error}); delete it and retry the download"
            ) from error
        needed = gguf.data_start + max(
            (tensor.offset + (tensor.expected_bytes() or 0) for tensor in gguf.tensors),
            default=0,
        )
        actual = path.stat().st_size
        if actual < needed:
            raise ShardCheckError(
                f"{path.name} is short: {actual:,} of {needed:,} bytes ({needed - actual:,} missing); "
                "delete it and retry the download"
            )
