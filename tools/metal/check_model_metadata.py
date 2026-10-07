#!/usr/bin/env python3
"""Exercise the production GGUF reader against compact headers from every model snapshot."""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import test_model_headers as catalog


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_snapshot(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def encode_metadata(metadata: dict[str, str | int]) -> bytes:
    encoded = bytearray()
    for key, value in metadata.items():
        encoded.extend(gguf_string(key))
        if isinstance(value, str):
            encoded.extend(struct.pack("<I", 8))  # GGUF metadata STRING
            encoded.extend(gguf_string(value))
        else:
            encoded.extend(struct.pack("<II", 4, value))  # GGUF metadata UINT32
    return bytes(encoded)


def write_header(path: Path, tensors: list[dict], metadata: dict[str, str | int]) -> None:
    body = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
    body.extend(encode_metadata(metadata))
    for tensor in tensors:
        body.extend(gguf_string(tensor["name"]))
        shape = tensor["ne"]
        body.extend(struct.pack("<I", len(shape)))
        body.extend(struct.pack("<" + "Q" * len(shape), *shape))
        body.extend(struct.pack("<IQ", tensor["id"], 0))
    body.extend(b"\0" * (-len(body) % 32))
    path.write_bytes(body)


def shard_metadata(kv: dict, shard_no: int) -> dict[str, str | int]:
    keys = (
        "general.architecture",
        "qwen4exp.block_count",
        "qwen4exp.embedding_length",
        "qwen4exp.attention.head_count",
        "qwen4exp.attention.head_count_kv",
        "qwen4exp.expert_count",
        "qwen4exp.expert_used_count",
        "split.count",
        "split.no",
        "split.tensors.count",
    )
    result = {key: kv[key] for key in keys if key in kv}
    result.setdefault("split.no", shard_no)
    return result


def write_pair(directory: Path, first: dict, second: dict,
               first_kv: dict, second_kv: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    first_path = directory / "model-00001-of-00002.gguf"
    second_path = directory / "model-00002-of-00002.gguf"
    write_header(first_path, first["tensors"], shard_metadata(first_kv, 0))
    write_header(second_path, second["tensors"], shard_metadata(second_kv, 1))
    return first_path


def invoke(binary: Path, first_path: Path, experts: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), "--check-model-snapshot", str(first_path), str(experts)],
        capture_output=True,
        check=False,
        text=True,
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_model_metadata.py METAL_WEIGHT_STORE_TEST FIXTURE_DIR", file=sys.stderr)
        return 2
    binary = Path(sys.argv[1]).resolve()
    fixture_dir = Path(sys.argv[2]).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)

    checked = 0
    with tempfile.TemporaryDirectory(prefix="strata-model-metadata-") as temporary:
        root = Path(temporary)
        for number, (model_name, shards) in enumerate(catalog.MODELS.items()):
            first = read_snapshot(fixture_dir, shards[0])
            second = read_snapshot(fixture_dir, shards[1])
            experts = catalog.SHARD1_METADATA[shards[0]][1]
            pair = write_pair(root / f"valid-{number}", first, second,
                              first["kv"], second["kv"])
            result = invoke(binary, pair, experts)
            if result.returncode:
                raise RuntimeError(f"{model_name} snapshot rejected by C++ reader: {result.stderr}{result.stdout}")
            checked += 1

        first_name, second_name = catalog.MODELS["Q2_0"]
        first = read_snapshot(fixture_dir, first_name)
        second = read_snapshot(fixture_dir, second_name)
        first_kv = dict(first["kv"])
        second_kv = dict(second["kv"])
        mutations: list[tuple[str, object, object]] = [
            ("general.architecture", "llama", 512),
            ("qwen4exp.block_count", 47, 512),
            ("qwen4exp.embedding_length", 2561, 512),
            ("qwen4exp.attention.head_count", 23, 512),
            ("qwen4exp.attention.head_count_kv", 3, 512),
            ("qwen4exp.expert_count", 256, 512),
            ("qwen4exp.expert_used_count", 9, 512),
            ("split.count", 1, 512),
            ("split.no", 1, 512),
            ("split.tensors.count", 1223, 512),
        ]
        for number, (key, changed, experts) in enumerate(mutations):
            changed_first = dict(first_kv)
            changed_second = dict(second_kv)
            if key.startswith("split."):
                changed_first[key] = changed
                changed_second[key] = changed
            else:
                changed_first[key] = changed
            pair = write_pair(root / f"invalid-{number}", first, second,
                              changed_first, changed_second)
            result = invoke(binary, pair, experts)
            diagnostic = result.stderr + result.stdout
            if result.returncode == 0 or key not in diagnostic:
                raise RuntimeError(
                    f"C++ reader did not reject changed {key} with a key-specific diagnostic: "
                    f"status={result.returncode}, output={diagnostic!r}"
                )
            checked += 1

        result = subprocess.run([str(binary), "--test-model-metadata"],
                                capture_output=True, check=False, text=True)
        if result.returncode:
            raise RuntimeError(f"synthetic WeightStore metadata tests failed: {result.stderr}{result.stdout}")
        checked += 1

    print(f"C++ GGUF metadata validation passed: {checked} snapshot and synthetic cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
