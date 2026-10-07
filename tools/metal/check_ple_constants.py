#!/usr/bin/env python3
"""Compare the PLE artifact metadata with the kernel and shared Metal constants."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import test_model_headers as catalog


METADATA_FIELDS = {
    "qwen4exp.embedding_length": "embedding_length",
    "qwen4exp.embedding_length_per_layer_input": "embedding_length_per_layer_input",
    "qwen4exp.ple.layers": "ple_layers",
    "qwen4exp.ple.ngram_size": "ngram_size",
    "qwen4exp.ple.heads_per_ngram": "heads_per_ngram",
    "qwen4exp.ple.conv_kernel": "conv_kernel",
    "qwen4exp.ple.eos_token_id": "eos_token_id",
    "qwen4exp.ple.image_token_id": "image_token_id",
    "qwen4exp.ple.layer_multipliers": "layer_multipliers",
    "qwen4exp.ple.head_vocab_sizes": "head_vocab_sizes",
    "qwen4exp.ple.head_offsets": "head_offsets",
}


def read_json(path: Path) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def same_json_value(actual, expected) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_json_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def compare_metadata(kv: dict, constants: dict, label: str) -> None:
    for metadata_key, constant_key in METADATA_FIELDS.items():
        if metadata_key not in kv:
            raise AssertionError(f"{label}: missing metadata key {metadata_key}")
        if not same_json_value(kv[metadata_key], constants[constant_key]):
            raise AssertionError(
                f"{label}: {metadata_key} differs: fixture={kv[metadata_key]!r}, "
                f"engine={constants[constant_key]!r}"
            )


def validate_tokenizer_eos(kv: dict, constants: dict, label: str) -> None:
    tokenizer_eos = kv.get("tokenizer.ggml.eos_token_id")
    if type(tokenizer_eos) is not int:
        raise AssertionError(f"{label}: tokenizer.ggml.eos_token_id must be an integer")
    if tokenizer_eos != 248046 or tokenizer_eos == constants["eos_token_id"]:
        raise AssertionError(
            f"{label}: tokenizer EOS must remain 248046 and differ from PLE EOS"
        )


def validate_internal_contract(constants: dict) -> None:
    multipliers = constants["layer_multipliers"]
    vocab_sizes = constants["head_vocab_sizes"]
    offsets = constants["head_offsets"]
    heads = constants["ple_heads"]
    if len(multipliers) != constants["ngram_size"]:
        raise AssertionError("layer_multipliers length must equal ngram_size")
    if len(vocab_sizes) != heads or len(offsets) != heads:
        raise AssertionError("head_vocab_sizes and head_offsets must have one entry per PLE head")
    running_offset = 0
    for head, (vocab_size, offset) in enumerate(zip(vocab_sizes, offsets, strict=True)):
        if offset != running_offset:
            raise AssertionError(f"head_offsets[{head}] is not the prefix sum of head_vocab_sizes")
        running_offset += vocab_size
    if running_offset > constants["table_rows"]:
        raise AssertionError("head_vocab_sizes exceed PLE_TABLE_ROWS")
    if constants["table_rows"] - running_offset != 90:
        raise AssertionError("the pinned PLE table must retain its 90 trailing rows")
    if heads != (constants["ngram_size"] - 1) * constants["heads_per_ngram"]:
        raise AssertionError("PLE_HEADS does not follow the ngram/head geometry")
    expected_history = (constants["conv_kernel"] - 1) * constants["ngram_size"]
    if constants["metal_history_rows"] != expected_history:
        raise AssertionError("PLE_HISTORY_ROWS does not follow the PLE convolution geometry")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_ple_constants.py METAL_WEIGHT_STORE_TEST FIXTURE_DIR", file=sys.stderr)
        return 2
    binary = Path(sys.argv[1]).resolve()
    fixture_dir = Path(sys.argv[2]).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)

    result = subprocess.run(
        [str(binary), "--print-ple-constants"], capture_output=True, check=False, text=True
    )
    if result.returncode:
        raise RuntimeError(f"PLE constant emitter failed: {result.stderr}{result.stdout}")
    constants = json.loads(result.stdout)

    paired_values = (
        ("embedding_length", "metal_embedding_length"),
        ("embedding_length_per_layer_input", "metal_embedding_length_per_layer_input"),
        ("ngram_size", "metal_ngram_size"),
        ("heads_per_ngram", "metal_heads_per_ngram"),
        ("ple_heads", "metal_ple_heads"),
        ("conv_kernel", "metal_conv_kernel"),
        ("ple_layers", "metal_ple_layers"),
        ("table_rows", "metal_table_rows"),
        ("eos_token_id", "metal_eos_token_id"),
    )
    for kernel_key, metal_key in paired_values:
        if constants[kernel_key] != constants[metal_key]:
            raise AssertionError(f"kernel and shared Metal constants differ: {kernel_key}/{metal_key}")
    validate_internal_contract(constants)

    models_checked = 0
    tensor_tables_checked = 0
    for model_name, shard_names in catalog.MODELS.items():
        snapshots = [read_json(fixture_dir / name) for name in shard_names]
        first_kv = snapshots[0]["kv"]
        compare_metadata(first_kv, constants, f"{model_name} shard 1")
        validate_tokenizer_eos(first_kv, constants, model_name)
        if constants["image_token_id"] != first_kv["qwen4exp.ple.image_token_id"]:
            raise AssertionError(f"{model_name}: image token ID differs from PLE metadata")

        tables = [
            tensor
            for snapshot in snapshots
            for tensor in snapshot["tensors"]
            if tensor["name"] == "per_layer_token_embd.weight"
        ]
        if len(tables) != 1:
            raise AssertionError(
                f"{model_name}: expected one per_layer_token_embd.weight table, found {len(tables)}"
            )
        table = tables[0]
        expected_shape = [constants["embedding_length_per_layer_input"], constants["table_rows"]]
        if table["ne"] != expected_shape:
            raise AssertionError(
                f"{model_name}: PLE table shape {table['ne']!r} does not match {expected_shape!r}"
            )
        models_checked += 1
        tensor_tables_checked += 1

    model_name, shard_names = next(iter(catalog.MODELS.items()))
    mutated = read_json(fixture_dir / shard_names[0])
    mutated["kv"]["qwen4exp.ple.layer_multipliers"][0] += 1
    try:
        compare_metadata(mutated["kv"], constants, f"{model_name} mutated snapshot")
    except AssertionError as error:
        if "qwen4exp.ple.layer_multipliers" not in str(error):
            raise
    else:
        raise AssertionError("mutated PLE metadata was not rejected")

    wrong_type = read_json(fixture_dir / shard_names[0])
    wrong_type["kv"]["qwen4exp.ple.layers"] = [True]
    try:
        compare_metadata(wrong_type["kv"], constants, f"{model_name} mistyped snapshot")
    except AssertionError as error:
        if "qwen4exp.ple.layers" not in str(error):
            raise
    else:
        raise AssertionError("boolean metadata was incorrectly accepted as integer 1")

    wrong_tokenizer_type = read_json(fixture_dir / shard_names[0])
    wrong_tokenizer_type["kv"]["tokenizer.ggml.eos_token_id"] = 248046.0
    try:
        validate_tokenizer_eos(
            wrong_tokenizer_type["kv"], constants, f"{model_name} mistyped tokenizer EOS"
        )
    except AssertionError as error:
        if "tokenizer.ggml.eos_token_id" not in str(error):
            raise
    else:
        raise AssertionError("floating-point tokenizer EOS was incorrectly accepted")

    print(
        f"PLE constants passed: {models_checked} model headers, {tensor_tables_checked} tables, "
        f"{len(constants['layer_multipliers'])} multipliers, {constants['ple_heads']} heads, "
        "and named value and metadata-type mutations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
