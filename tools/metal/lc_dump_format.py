#!/usr/bin/env python3
"""Output formats and graph-node checks of llama.cpp's reference dump tool (lc_dump).

Section 2.5 of docs/mac/09-testing-and-validation.md specifies lc_dump as C++ against llama.h, built with
tools/metal/lc_dump/CMakeLists.txt at the pinned llama.cpp commit. That program is not written in this checkout and is
not built here. This module holds the part of it that the specification states in Python terms: the logits file
(`--dump-logits` format), the residual-ladder file (`--dump-layers` layout, section 9.1) and the graph node names and
per-position sizes the ladder must find. The C++ program, when it is written, must apply the same node check; this
module is the only implementation of that check in the repository.

Standard library only. Files are little-endian. Writers stream to `<path>.partial` and rename it on success, so a
failed run leaves no file under the final name. A truncated file under the final name would be read by tf_compare.py
as a short prompt-path dump, so no partial file may take that name.
"""
from __future__ import annotations

import os
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

N_LAYERS = 48
HIDDEN = 2560
STREAMS = 4
SLOT_FLOATS = STREAMS * HIDDEN          # one slot of one position: 4 x 2560 F32
LAYER_SLOTS = N_LAYERS + 1              # slot 0 is the embedding, slot l + 1 is R after layer l
EMBEDDING_NODE = 'hc_init'
LAYER_NODE = 'l_last-{}'


class LcDumpError(ValueError):
    """An input or output of lc_dump does not have the specified form."""


def layer_node_names(n_layers: int = N_LAYERS) -> tuple[str, ...]:
    """The graph nodes of the residual ladder, in slot order."""
    return (EMBEDDING_NODE,) + tuple(LAYER_NODE.format(layer) for layer in range(n_layers))


def check_graph_nodes(element_counts: Mapping[str, int], n_layers: int = N_LAYERS) -> None:
    """Fail if a ladder node is missing or does not hold one slot (4 x 2560 values) per position."""
    for name in layer_node_names(n_layers):
        if name not in element_counts:
            raise LcDumpError(f"graph node {name!r} is missing; the llama.cpp graph may have changed")
        if element_counts[name] != SLOT_FLOATS:
            raise LcDumpError(f"graph node {name!r} holds {element_counts[name]} values per position, "
                              f"expected {SLOT_FLOATS}")


def layers_record_bytes(n_layers: int = N_LAYERS) -> int:
    return (n_layers + 1) * SLOT_FLOATS * 4


def layers_position_count(path: str | Path, n_layers: int = N_LAYERS) -> int:
    """The number of whole positions in a `--dump-layers` file; fails on a truncated file."""
    size = Path(path).stat().st_size
    record = layers_record_bytes(n_layers)
    if size == 0:
        raise LcDumpError(f"{path}: the ladder file is empty")
    if size % record:
        raise LcDumpError(f"{path}: {size} bytes is not a whole number of {record}-byte position records")
    return size // record


def _temporary(path: Path) -> Path:
    return path.with_name(path.name + '.partial')


def write_logits(path: str | Path, n_vocab: int, rows: Iterable[Sequence[float]]) -> int:
    """Write `int32 n_vocab, int32 rows` then the rows as F32 (the `--dump-logits` format). Returns the row count."""
    if n_vocab < 2:
        raise LcDumpError(f"vocabulary must have at least two entries, got {n_vocab}")
    target = Path(path)
    temporary = _temporary(target)
    count = 0
    try:
        with temporary.open('wb') as stream:
            stream.write(b'\x00' * 8)
            for row in rows:
                if len(row) != n_vocab:
                    raise LcDumpError(f"row {count} has {len(row)} logits, expected {n_vocab}")
                stream.write(struct.pack(f'<{n_vocab}f', *row))
                count += 1
            if count == 0:
                raise LcDumpError('no rows to write')
            stream.seek(0)
            stream.write(struct.pack('<ii', n_vocab, count))
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def write_layers(path: str | Path, positions: Iterable[Mapping[str, Sequence[float]]],
                 n_layers: int = N_LAYERS) -> int:
    """Write the residual ladder: per position, slot 0 is `hc_init`, slot l + 1 is `l_last-<l>` (section 9.1).

    Each position is a mapping from node name to its F32 values. A missing or resized node fails before the
    final path is written. Returns the number of positions.
    """
    target = Path(path)
    temporary = _temporary(target)
    count = 0
    try:
        with temporary.open('wb') as stream:
            for position in positions:
                check_graph_nodes({name: len(values) for name, values in position.items()
                                   if name in layer_node_names(n_layers)}, n_layers)
                for name in layer_node_names(n_layers):
                    values = position[name]
                    stream.write(struct.pack(f'<{SLOT_FLOATS}f', *values))
                count += 1
        if count == 0:
            raise LcDumpError('no positions to write')
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count
