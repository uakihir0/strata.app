"""Verify that unconverted native-pack float tensors preserve their GGUF bytes."""
from __future__ import annotations

import hashlib
import io
import json
import re
import struct
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gguf_reader as G
from pack_format import (
    NATIVE_PLE_KEY_NAME, NATIVE_PLE_KEY_TYPES, NOT_IN_PACK, form_of, is_expert,
)

CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_SAMPLE_COUNT = 32
FLOAT_KINDS = {"2", "3", "4", "5"}
KIND_SOURCE_TYPE = {"2": "F32", "3": "F32", "4": "BF16", "5": "F16"}
DESTINATION_KIND = {"F32": "2", "BF16": "4"}
SOURCE_FLOAT_TYPES = {"F32", "BF16", "F16"}
TENSORS_EXCLUDED_FROM_PACK = NOT_IN_PACK
CATALOGUE_MODELS = {
    ("Qwen3.8-Flash-Next GSQ-RCO", 41): "Q2_0",
    ("Qwen3.8-Flash-Next GSQ-RCO", 20): "IQ2_XS",
    ("Qwen3.8-Flash-Next GSQ-RCO", 23): "IQ3_XXS",
    ("Qwen3.8-Flash-Next GSQ-RCO", 26): "IQ3_S",
    ("Qwen3.8-Flash-Next GSQ-RCO Coder", 31): "Coder IQ1_M",
    ("Swift Flash Next", 20): "Swift IQ2_XS",
    ("Swift Flash Next", 23): "Swift IQ3_XXS",
}


class DenseGGUFError(ValueError):
    """A pack cannot be proven byte-consistent with its GGUF sources."""


@dataclass(frozen=True)
class IndexRow:
    name: str
    file_id: int
    kind: str
    source_offset: int
    source_bytes: int
    destination_bytes: int
    ne0: int
    ne1: int
    line: int


@dataclass(frozen=True)
class VerificationReport:
    tensors_checked: int
    bytes_checked: int
    conversions_checked: int
    conversion_bytes_checked: int
    sampled: bool
    tensor_names: tuple[str, ...]
    catalogue_model: str | None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DenseGGUFError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, DenseGGUFError) as exc:
        raise DenseGGUFError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DenseGGUFError(f"{path}: expected a JSON object")
    return value


def _read_manifest(
    pack_dir: Path, source_dir: Path
) -> tuple[list[Path], dict[str, dict[str, Any]], bool]:
    manifest_path = pack_dir / "conversions.json"
    manifest = _read_json(manifest_path)
    if type(manifest.get("schema")) is not int or manifest["schema"] != 1:
        raise DenseGGUFError(f"{manifest_path}: schema must be integer 1")
    if type(manifest.get("compat_bf16")) is not bool:
        raise DenseGGUFError(f"{manifest_path}: compat_bf16 must be a boolean")

    source_records = manifest.get("source_shards")
    if not isinstance(source_records, list) or not source_records:
        raise DenseGGUFError(f"{manifest_path}: source_shards must be a non-empty array")
    if not source_dir.is_dir():
        raise DenseGGUFError(f"GGUF source directory does not exist: {source_dir}")

    source_paths: list[Path] = []
    seen_shards: set[str] = set()
    for index, record in enumerate(source_records):
        if not isinstance(record, dict):
            raise DenseGGUFError(f"{manifest_path}: source_shards[{index}] must be an object")
        name = record.get("name")
        size = record.get("size")
        if (not isinstance(name, str) or not name or Path(name).name != name or
                "/" in name or "\\" in name):
            raise DenseGGUFError(f"{manifest_path}: source_shards[{index}].name must be a filename")
        if name in seen_shards:
            raise DenseGGUFError(f"{manifest_path}: duplicate source shard {name}")
        if type(size) is not int or size <= 0:
            raise DenseGGUFError(f"{manifest_path}: source_shards[{index}].size must be a positive integer")
        seen_shards.add(name)
        source_path = source_dir / name
        try:
            actual_size = source_path.stat().st_size
        except OSError as exc:
            raise DenseGGUFError(f"missing source shard {source_path}: {exc}") from exc
        if not source_path.is_file():
            raise DenseGGUFError(f"source shard is not a regular file: {source_path}")
        if actual_size != size:
            raise DenseGGUFError(
                f"{source_path.name}: size is {actual_size} B; conversions.json records {size} B"
            )
        source_paths.append(source_path)

    conversion_rows = manifest.get("tensors")
    if not isinstance(conversion_rows, list):
        raise DenseGGUFError(f"{manifest_path}: tensors must be an array")
    conversions: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(conversion_rows):
        if not isinstance(record, dict):
            raise DenseGGUFError(f"{manifest_path}: tensors[{index}] must be an object")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise DenseGGUFError(f"{manifest_path}: tensors[{index}].name must be a non-empty string")
        if name in conversions:
            raise DenseGGUFError(f"{manifest_path}: duplicate conversion for {name}")
        if not all(isinstance(record.get(key), str) and record[key] for key in
                   ("shard", "src_type", "dst_type", "method")):
            raise DenseGGUFError(f"{manifest_path}: conversion for {name} has invalid shard or types")
        if record["src_type"] == record["dst_type"]:
            raise DenseGGUFError(f"{manifest_path}: conversion for {name} does not change tensor type")
        source_hash = record.get("source_sha256")
        if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
            raise DenseGGUFError(f"{manifest_path}: conversion for {name} has an invalid source_sha256")
        conversions[name] = record
    return source_paths, conversions, manifest["compat_bf16"]


def _load_sources(source_paths: list[Path]) -> dict[str, tuple[Path, G.GGUFFile, G.TensorInfo]]:
    files: list[tuple[Path, G.GGUFFile]] = []
    for path in source_paths:
        try:
            files.append((path, G.GGUFFile(path)))
        except (OSError, ValueError, struct.error) as exc:
            raise DenseGGUFError(f"cannot read GGUF header {path}: {exc}") from exc

    shard_count = len(files)
    first_metadata = files[0][1].metadata
    declared_count = first_metadata.get("split.count", 1)
    if type(declared_count) is not int or declared_count != shard_count:
        raise DenseGGUFError(
            f"{files[0][0].name}: split.count is {declared_count!r}; expected {shard_count} source shard(s)"
        )
    if shard_count > 1 and "general.architecture" not in first_metadata:
        raise DenseGGUFError(f"{files[0][0].name}: first split shard has no general.architecture")

    total_tensors = 0
    declared_tensors = first_metadata.get("split.tensors.count")
    if declared_tensors is not None and type(declared_tensors) is not int:
        raise DenseGGUFError(f"{files[0][0].name}: split.tensors.count must be an integer")
    if shard_count > 1 and declared_tensors is None:
        raise DenseGGUFError(f"{files[0][0].name}: split.tensors.count is missing")

    source_tensors: dict[str, tuple[Path, G.GGUFFile, G.TensorInfo]] = {}
    for number, (path, gguf) in enumerate(files):
        metadata = gguf.metadata
        actual_count = metadata.get("split.count", 1)
        if shard_count > 1 and "split.no" not in metadata:
            raise DenseGGUFError(f"{path.name}: split.no is missing from a multi-shard source")
        split_no = metadata.get("split.no", 0)
        if (type(actual_count) is not int or actual_count != shard_count or
                type(split_no) is not int or split_no != number):
            raise DenseGGUFError(
                f"{path.name}: split metadata does not identify shard {number + 1} of {shard_count}"
            )
        if declared_tensors is not None and metadata.get("split.tensors.count") != declared_tensors:
            raise DenseGGUFError(f"{path.name}: split.tensors.count differs from shard 1")
        total_tensors += len(gguf.tensors)
        for tensor in gguf.tensors:
            if tensor.name in source_tensors:
                raise DenseGGUFError(f"duplicate source tensor {tensor.name} in {path.name}")
            source_tensors[tensor.name] = (path, gguf, tensor)
    if declared_tensors is not None and total_tensors != declared_tensors:
        raise DenseGGUFError(
            f"GGUF shards contain {total_tensors} tensors; split.tensors.count is {declared_tensors}"
        )
    return source_tensors


def _read_index(path: Path) -> dict[str, IndexRow]:
    rows: dict[str, IndexRow] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DenseGGUFError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 19:
            raise DenseGGUFError(f"{path}:{line_number}: expected 19 index fields, found {len(fields)}")
        name = fields[0]
        if not name or name in rows:
            raise DenseGGUFError(f"{path}:{line_number}: empty or duplicate tensor name {name!r}")
        try:
            numbers = [int(value, 10) for value in fields[1:]]
        except ValueError as exc:
            raise DenseGGUFError(f"{path}:{line_number}: non-integer index field") from exc
        if any(value < 0 for value in numbers):
            raise DenseGGUFError(f"{path}:{line_number}: negative index field")
        file_id, kind_id, source_offset, source_bytes = numbers[:4]
        destination_bytes = numbers[5]
        ne0, ne1 = numbers[6:8]
        kind = str(kind_id)
        if kind != "0" and kind not in FLOAT_KINDS:
            raise DenseGGUFError(f"{path}:{line_number}: unsupported native-pack kind {kind}")
        rows[name] = IndexRow(
            name, file_id, kind, source_offset, source_bytes, destination_bytes, ne0, ne1, line_number
        )
    if not rows:
        raise DenseGGUFError(f"{path}: no tensor rows")
    return rows


def _validate_native_experts_marker(path: Path) -> None:
    """Reject empty or malformed expert tables before treating a pack as complete."""
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError) as exc:
        raise DenseGGUFError(f"cannot read {path}: {exc}") from exc
    if not lines:
        raise DenseGGUFError(f"{path}: native expert table is empty")

    header = re.match(
        r"^# strata native experts v([34]):.*\(n_expert ([1-9][0-9]*), total ([1-9][0-9]*);",
        lines[0],
    )
    if header is None:
        raise DenseGGUFError(f"{path}: expected a v3 or v4 native-experts header")
    version, expert_count, total_bytes = int(header.group(1)), int(header.group(2)), int(header.group(3))
    expected_offset = 0
    for expected_layer, line in enumerate(lines[1:]):
        fields = line.split()
        if len(fields) not in (8, 9):
            raise DenseGGUFError(f"{path}:{expected_layer + 2}: expected 8 fields and an optional shard field")
        try:
            values = [int(value, 10) for value in fields[:8]]
        except ValueError as exc:
            raise DenseGGUFError(f"{path}:{expected_layer + 2}: non-integer expert-table field") from exc
        if any(value < 0 for value in values):
            raise DenseGGUFError(f"{path}:{expected_layer + 2}: negative expert-table field")
        layer, _gu_type, _d_type, offset, blob_bytes, gate_offset, up_offset, down_offset = values
        if layer != expected_layer:
            raise DenseGGUFError(f"{path}:{expected_layer + 2}: layer rows must be contiguous from zero")
        if offset != expected_offset or blob_bytes == 0:
            raise DenseGGUFError(f"{path}:{expected_layer + 2}: invalid expert blob offset or size")
        if min(gate_offset, up_offset, down_offset) == 0:
            raise DenseGGUFError(f"{path}:{expected_layer + 2}: expert source offsets must be positive")
        if len(fields) == 9:
            shards = fields[8].split(",")
            if (version == 3 and len(shards) != 1) or (version == 4 and len(shards) not in (1, 3)):
                raise DenseGGUFError(f"{path}:{expected_layer + 2}: invalid v{version} shard field")
            if any(shard and Path(shard).name != shard for shard in shards):
                raise DenseGGUFError(f"{path}:{expected_layer + 2}: shard field must contain filenames")
        expected_offset += blob_bytes * expert_count
    if len(lines) == 1:
        raise DenseGGUFError(f"{path}: native expert table has no layer rows")
    if expected_offset != total_bytes:
        raise DenseGGUFError(
            f"{path}: header total is {total_bytes} bytes; layer rows describe {expected_offset} bytes"
        )


def _validate_native_experts_against_sources(path: Path, source_paths: list[Path]) -> None:
    """Require the expert table to exactly describe the GGUF shards, using the packer's layout rules."""
    try:
        import iq_pack
    except ImportError as exc:
        raise DenseGGUFError(f"cannot load the native expert layout rules: {exc}") from exc

    try:
        model = iq_pack.Model(source_paths[0])
        with redirect_stdout(io.StringIO()):
            result = iq_pack.expert_layout(model, source_paths[0])
    except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        raise DenseGGUFError(f"cannot derive native_experts.txt from the GGUF shards: {exc}") from exc
    if isinstance(result, str):
        raise DenseGGUFError(f"cannot derive native_experts.txt from the GGUF shards: {result}")
    expected_text = result[1]
    try:
        actual_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DenseGGUFError(f"cannot read {path}: {exc}") from exc
    if actual_text != expected_text:
        raise DenseGGUFError(f"{path}: expert table does not match the layout derived from the GGUF shards")


def _validate_float_rows(
    pack_dir: Path,
    dense_path: Path,
    rows: dict[str, IndexRow],
    conversions: dict[str, dict[str, Any]],
    source_tensors: dict[str, tuple[Path, G.GGUFFile, G.TensorInfo]],
) -> tuple[
    list[tuple[IndexRow, Path, G.GGUFFile, G.TensorInfo]],
    list[tuple[IndexRow, Path, G.GGUFFile, G.TensorInfo, dict[str, Any]]],
]:
    dense_size = dense_path.stat().st_size
    float_rows: list[tuple[IndexRow, Path, G.GGUFFile, G.TensorInfo]] = []
    converted_rows: list[tuple[IndexRow, Path, G.GGUFFile, G.TensorInfo, dict[str, Any]]] = []

    # iq_pack.index_standalone indexes every non-expert float except the PLE table.
    # Check the source inventory too, so deleting a row cannot shrink the comparison set.
    for name, (_, _, tensor) in source_tensors.items():
        if (tensor.type_name in SOURCE_FLOAT_TYPES and not is_expert(name) and
                name not in TENSORS_EXCLUDED_FROM_PACK):
            row = rows.get(name)
            if row is None or row.kind == "0":
                raise DenseGGUFError(f"{name}: source float tensor has no dense index row")

    # Quantized source tensors can also require a pack form. In particular, removing their
    # conversion record and marking the index row native must not bypass this contract.
    for name, (_, _, tensor) in source_tensors.items():
        expected_form = form_of(name)
        if expected_form is None or is_expert(name) or name in TENSORS_EXCLUDED_FROM_PACK:
            continue
        row = rows.get(name)
        if row is None:
            raise DenseGGUFError(f"{name}: required {expected_form} pack-form tensor has no index row")
        native_ple_key = name == NATIVE_PLE_KEY_NAME and tensor.type_name in NATIVE_PLE_KEY_TYPES
        if row.kind == "0" and not native_ple_key:
            raise DenseGGUFError(f"{name}: required {expected_form} pack form is marked as native")

    for name, conversion in conversions.items():
        row = rows.get(name)
        if row is None or row.kind == "0":
            raise DenseGGUFError(f"conversions.json lists {name}, which has no float index row")
        if name not in source_tensors:
            raise DenseGGUFError(f"conversions.json lists {name}, which is missing from the GGUF shards")

    for row in rows.values():
        if row.kind == "0":
            continue
        if row.file_id != 0:
            raise DenseGGUFError(f"{row.name}: float tensor references file {row.file_id}, not dense.bin")
        if row.source_bytes <= 0 or row.source_offset + row.source_bytes > dense_size:
            raise DenseGGUFError(
                f"{row.name}: dense.bin range [{row.source_offset}, "
                f"{row.source_offset + row.source_bytes}) exceeds {dense_size} B"
            )
        source = source_tensors.get(row.name)
        if source is None:
            raise DenseGGUFError(f"{row.name}: float index row is missing from the GGUF shards")
        source_path, gguf, tensor = source
        if tensor.shape and (row.ne0 != tensor.shape[0] or
                             row.ne1 != (tensor.shape[1] if len(tensor.shape) > 1 else 0)):
            raise DenseGGUFError(f"{row.name}: index shape differs from the GGUF tensor table")
        expected_bytes = tensor.expected_bytes()
        if expected_bytes is None:
            raise DenseGGUFError(f"{row.name}: unsupported or invalid GGUF tensor byte geometry")
        absolute_start = gguf.data_start + tensor.offset
        if absolute_start < gguf.data_start or absolute_start + expected_bytes > source_path.stat().st_size:
            raise DenseGGUFError(f"{row.name}: GGUF tensor range exceeds {source_path.name}")

        conversion = conversions.get(row.name)
        if conversion is not None:
            expected_form = form_of(row.name)
            if expected_form is None or conversion["dst_type"] != expected_form:
                raise DenseGGUFError(
                    f"{row.name}: conversion destination {conversion['dst_type']} "
                    f"does not match the pack form {expected_form}"
                )
            if conversion["src_type"] == expected_form:
                raise DenseGGUFError(
                    f"{row.name}: conversions.json lists a tensor that already has its pack form"
                )
            if conversion["shard"] != source_path.name:
                raise DenseGGUFError(
                    f"{row.name}: conversion names shard {conversion['shard']}, but tensor is in {source_path.name}"
                )
            if conversion["src_type"] != tensor.type_name:
                raise DenseGGUFError(
                    f"{row.name}: conversion source type is {conversion['src_type']}, "
                    f"but the GGUF stores {tensor.type_name}"
                )
            expected_kind = DESTINATION_KIND.get(conversion["dst_type"])
            if conversion["dst_type"] == "F16":
                expected_kind = "3" if tensor.type_name == "F32" else "5"
            if expected_kind is None or row.kind != expected_kind:
                raise DenseGGUFError(
                    f"{row.name}: conversion to {conversion['dst_type']} disagrees with index kind {row.kind}"
                )
            element_count = 1
            for dimension in tensor.shape:
                element_count *= dimension
            destination_size = {"F32": 4, "BF16": 2, "F16": 2}.get(conversion["dst_type"])
            if destination_size is None:
                raise DenseGGUFError(
                    f"{row.name}: unsupported conversion destination {conversion['dst_type']}"
                )
            expected_destination_bytes = element_count * destination_size
            expected_dense_bytes = expected_bytes if row.kind == "3" else expected_destination_bytes
            if (row.source_bytes != expected_dense_bytes or
                    row.destination_bytes != expected_destination_bytes):
                raise DenseGGUFError(
                    f"{row.name}: conversion byte sizes disagree with its source and destination types"
                )
            _verify_source_hash(
                source_path,
                absolute_start,
                expected_bytes,
                conversion["source_sha256"],
                row.name,
            )
            converted_rows.append((row, source_path, gguf, tensor, conversion))
            continue

        expected_form = form_of(row.name)
        if expected_form is not None and tensor.type_name != expected_form:
            raise DenseGGUFError(
                f"{row.name}: missing required conversion from {tensor.type_name} to {expected_form}"
            )
        expected_type = KIND_SOURCE_TYPE[row.kind]
        if row.kind == "3":
            raise DenseGGUFError(f"{row.name}: index kind 3 requires a conversion record")
        if tensor.type_name != expected_type:
            raise DenseGGUFError(
                f"{row.name}: unlisted index kind {row.kind} expects {expected_type}, "
                f"but the GGUF stores {tensor.type_name}"
            )
        if row.source_bytes != expected_bytes:
            raise DenseGGUFError(
                f"{row.name}: dense.bin has {row.source_bytes} bytes; the GGUF tensor has {expected_bytes}"
            )
        if row.destination_bytes != expected_bytes:
            raise DenseGGUFError(
                f"{row.name}: index destination has {row.destination_bytes} bytes; expected {expected_bytes}"
            )
        float_rows.append((row, source_path, gguf, tensor))

    return float_rows, converted_rows


def _verify_source_hash(path: Path, offset: int, size: int, expected_hash: str, name: str) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            source.seek(offset)
            remaining = size
            while remaining:
                chunk = source.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    raise DenseGGUFError(f"{name}: short read while hashing converted source tensor")
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError as exc:
        raise DenseGGUFError(f"{name}: cannot hash converted source tensor in {path.name}: {exc}") from exc
    if digest.hexdigest() != expected_hash:
        raise DenseGGUFError(f"{name}: conversions.json source_sha256 does not match {path.name}")


def _compare_tensor(
    dense_file,
    source_file,
    row: IndexRow,
    source_path: Path,
    gguf: G.GGUFFile,
    tensor: G.TensorInfo,
) -> int:
    source_offset = gguf.data_start + tensor.offset
    compared = 0
    while compared < row.source_bytes:
        size = min(CHUNK_BYTES, row.source_bytes - compared)
        dense_file.seek(row.source_offset + compared)
        source_file.seek(source_offset + compared)
        dense_bytes = dense_file.read(size)
        gguf_bytes = source_file.read(size)
        if len(dense_bytes) != size or len(gguf_bytes) != size:
            raise DenseGGUFError(f"{row.name}: short read while comparing {source_path.name}")
        if dense_bytes != gguf_bytes:
            mismatch = next(i for i, (left, right) in enumerate(zip(dense_bytes, gguf_bytes)) if left != right)
            dense_value = dense_bytes[mismatch]
            gguf_value = gguf_bytes[mismatch]
            raise DenseGGUFError(
                f"{row.name}: byte mismatch at tensor offset {compared + mismatch} "
                f"(dense.bin=0x{dense_value:02x}, {source_path.name}=0x{gguf_value:02x})"
            )
        compared += size
    return compared


def _compare_conversion(
    dense_file,
    source_file,
    row: IndexRow,
    source_path: Path,
    gguf: G.GGUFFile,
    tensor: G.TensorInfo,
    conversion: dict[str, Any],
    compat_bf16: bool,
) -> int:
    if row.kind == "3":
        # Kind 3 keeps the source F32 bytes and lets the loader narrow them to F16.
        return _compare_tensor(dense_file, source_file, row, source_path, gguf, tensor)

    try:
        import iq_pack
    except ImportError as exc:
        raise DenseGGUFError(f"cannot load the pack conversion rules: {exc}") from exc

    geometry = G.BLOCK_GEOMETRY.get(tensor.type_name)
    if geometry is None:
        raise DenseGGUFError(f"{row.name}: unsupported source type for conversion: {tensor.type_name}")
    block_bytes = geometry[1]
    chunk_bytes = max(block_bytes, CHUNK_BYTES // block_bytes * block_bytes)
    source_offset = gguf.data_start + tensor.offset
    source_compared = 0
    dense_compared = 0
    destination_compared = 0
    while source_compared < (tensor.expected_bytes() or 0):
        size = min(chunk_bytes, tensor.expected_bytes() - source_compared)
        source_file.seek(source_offset + source_compared)
        source_raw = source_file.read(size)
        if len(source_raw) != size:
            raise DenseGGUFError(f"{row.name}: short read while converting {source_path.name}")
        try:
            result = iq_pack.convert(
                row.name, tensor.type_name, iq_pack.np.frombuffer(source_raw, dtype=iq_pack.np.uint8), compat_bf16
            )
        except (ImportError, KeyError, ValueError, RuntimeError) as exc:
            raise DenseGGUFError(f"{row.name}: cannot reproduce pack conversion: {exc}") from exc
        if isinstance(result, str):
            raise DenseGGUFError(
                f"{row.name}: conversion cannot be reproduced with compat_bf16={compat_bf16}: {result}"
            )
        kind, expected_raw, destination_bytes, record = result
        if record is None or kind != row.kind:
            raise DenseGGUFError(
                f"{row.name}: pack conversion produced index kind {kind}, expected {row.kind}"
            )
        if (record.get("src_type") != conversion["src_type"] or
                record.get("dst_type") != conversion["dst_type"]):
            raise DenseGGUFError(f"{row.name}: conversion record does not match the reproduced conversion")
        if destination_bytes < 0 or not expected_raw:
            raise DenseGGUFError(f"{row.name}: pack conversion produced an empty or invalid chunk")
        dense_file.seek(row.source_offset + dense_compared)
        actual_raw = dense_file.read(len(expected_raw))
        if len(actual_raw) != len(expected_raw):
            raise DenseGGUFError(f"{row.name}: short read in dense.bin while checking converted bytes")
        if actual_raw != expected_raw:
            mismatch = next(i for i, (left, right) in enumerate(zip(actual_raw, expected_raw)) if left != right)
            raise DenseGGUFError(
                f"{row.name}: converted dense.bin mismatch at tensor offset {dense_compared + mismatch} "
                f"(dense.bin=0x{actual_raw[mismatch]:02x}, expected=0x{expected_raw[mismatch]:02x})"
            )
        source_compared += size
        dense_compared += len(expected_raw)
        destination_compared += destination_bytes

    if dense_compared != row.source_bytes or destination_compared != row.destination_bytes:
        raise DenseGGUFError(
            f"{row.name}: converted payload totals disagree with index sizes "
            f"(dense {dense_compared}/{row.source_bytes}, destination "
            f"{destination_compared}/{row.destination_bytes})"
        )
    return dense_compared


def verify_pack(
    pack_dir: Path | str,
    source_dir: Path | str,
    *,
    sample_count: int | None = None,
    allow_pending: bool = False,
) -> VerificationReport:
    """Compare pack floats; allow_pending is reserved for the active packer's post-build check."""
    pack_dir = Path(pack_dir)
    source_dir = Path(source_dir)
    if sample_count is not None and (type(sample_count) is not int or sample_count <= 0):
        raise DenseGGUFError("sample_count must be a positive integer or None")
    pending_path = pack_dir / ".strata-generation-pending"
    if pending_path.exists() and not allow_pending:
        raise DenseGGUFError(f"pack generation is incomplete: {pending_path}")
    dense_path = pack_dir / "dense.bin"
    index_path = pack_dir / "index.txt"
    completion_path = pack_dir / "native_experts.txt"
    for path in (dense_path, index_path, completion_path):
        if not path.is_file():
            raise DenseGGUFError(f"required pack file is missing: {path}")
    _validate_native_experts_marker(completion_path)

    source_paths, conversions, compat_bf16 = _read_manifest(pack_dir, source_dir)
    source_tensors = _load_sources(source_paths)
    _validate_native_experts_against_sources(completion_path, source_paths)
    rows = _read_index(index_path)
    eligible, converted_rows = _validate_float_rows(pack_dir, dense_path, rows, conversions, source_tensors)
    if not eligible:
        raise DenseGGUFError(f"{pack_dir}: no unlisted float tensors to compare")

    if sample_count is None or sample_count >= len(eligible):
        selected = eligible
        sampled = sample_count is not None
    else:
        selected = sorted(
            eligible,
            key=lambda item: (
                hashlib.sha256(f"{item[1].name}\0{item[0].name}".encode("utf-8")).digest(),
                item[0].name,
            ),
        )[:sample_count]
        sampled = True

    bytes_checked = 0
    conversion_bytes_checked = 0
    try:
        with ExitStack() as stack:
            dense_file = stack.enter_context(dense_path.open("rb"))
            source_files = {path: stack.enter_context(path.open("rb")) for path in source_paths}
            for row, source_path, gguf, tensor, conversion in converted_rows:
                conversion_bytes_checked += _compare_conversion(
                    dense_file, source_files[source_path], row, source_path, gguf, tensor, conversion, compat_bf16
                )
            for row, source_path, gguf, tensor in selected:
                bytes_checked += _compare_tensor(
                    dense_file, source_files[source_path], row, source_path, gguf, tensor
                )
    except OSError as exc:
        raise DenseGGUFError(f"cannot read pack/source bytes: {exc}") from exc

    first_source = next(iter(source_tensors.values()), None)
    catalogue_model = None
    if first_source is not None:
        metadata = first_source[1].metadata
        catalogue_model = CATALOGUE_MODELS.get(
            (metadata.get("general.name"), metadata.get("general.file_type"))
        )

    return VerificationReport(
        tensors_checked=len(selected),
        bytes_checked=bytes_checked,
        conversions_checked=len(converted_rows),
        conversion_bytes_checked=conversion_bytes_checked,
        sampled=sampled,
        tensor_names=tuple(row.name for row, *_ in selected),
        catalogue_model=catalogue_model,
    )


def discover_model_packs(models_root: Path | str, *, expected_count: int = 7) -> list[tuple[Path, Path]]:
    """Return the catalogue's completed pack directories below a STRATA_TEST_MODELS root."""
    root = Path(models_root)
    if not root.is_dir():
        raise DenseGGUFError(f"STRATA_TEST_MODELS is not a directory: {root}")
    found: list[tuple[Path, Path]] = []
    for model_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name):
        pack_root = model_dir / "pack"
        if not pack_root.exists():
            continue
        if not pack_root.is_dir():
            raise DenseGGUFError(f"model pack path is not a directory: {pack_root}")
        direct_index = pack_root / "index.txt"
        nested = sorted(
            (item for item in pack_root.iterdir() if item.is_dir() and (item / "index.txt").exists()),
            key=lambda item: item.name,
        )
        if direct_index.exists() and nested:
            raise DenseGGUFError(f"{pack_root}: both direct and nested native packs are present")
        if direct_index.exists():
            found.append((pack_root, model_dir))
            continue
        if nested:
            if len(nested) != 1:
                raise DenseGGUFError(f"{pack_root}: expected one native pack, found {len(nested)}")
            found.extend((item, model_dir) for item in nested)
        elif any(pack_root.iterdir()):
            raise DenseGGUFError(f"{pack_root}: pack exists but index.txt is missing")
    if len(found) != expected_count:
        raise DenseGGUFError(
            f"found {len(found)} native packs under {root}; expected {expected_count} catalogue packs"
        )
    return found


def verify_model_packs(
    models_root: Path | str,
    *,
    expected_count: int = 7,
) -> list[tuple[Path, VerificationReport]]:
    """Run the full comparison for each catalogue pack under a model-data root."""
    results: list[tuple[Path, VerificationReport]] = []
    for pack_dir, source_dir in discover_model_packs(models_root, expected_count=expected_count):
        results.append((pack_dir, verify_pack(pack_dir, source_dir)))
    found_models: dict[str, Path] = {}
    for pack_dir, report in results:
        if report.catalogue_model is None:
            raise DenseGGUFError(
                f"{pack_dir}: GGUF metadata does not identify one of the seven catalogue models"
            )
        previous = found_models.get(report.catalogue_model)
        if previous is not None:
            raise DenseGGUFError(
                f"duplicate catalogue model {report.catalogue_model}: {previous} and {pack_dir}"
            )
        found_models[report.catalogue_model] = pack_dir
    missing = sorted(set(CATALOGUE_MODELS.values()) - set(found_models))
    if missing:
        raise DenseGGUFError(f"catalogue packs are missing: {', '.join(missing)}")
    return results
