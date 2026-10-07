import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "model-headers"
SPIKES = ROOT / "docs" / "mac" / "spikes" / "model-headers"
MODELS = {
    "Q2_0": ("q2_0-s1.json", "q2_0-s2.json"),
    "IQ2_XS": ("IQ2_XS-s1.json", "q2_0-s2.json"),
    "IQ3_XXS": ("IQ3_XXS-s1.json", "q2_0-s2.json"),
    "IQ3_S": ("IQ3_S-s1.json", "q2_0-s2.json"),
    "Coder IQ1_M": ("coder-s1.json", "q2_0-s2.json"),
    "Swift IQ2_XS": ("swift-IQ2_XS-s1.json", "swift-IQ2_XS-s2.json"),
    "Swift IQ3_XXS": ("swift-IQ3_XXS-s1.json", "swift-IQ3_XXS-s2.json"),
}
SHARD_LAYOUT = {
    # Physical shard size from docs/mac/03-model-reference.md §2.4,
    # followed by data_start from §3.4 and split.no from §2.5.
    "q2_0-s1.json": (37_623_740_192, 11_024_672, 0),
    "IQ2_XS-s1.json": (39_225_954_592, 11_024_672, 0),
    "IQ3_XXS-s1.json": (47_039_860_096, 11_024_256, 0),
    "IQ3_S-s1.json": (54_817_524_224, 11_024_384, 0),
    "coder-s1.json": (29_608_446_496, 11_024_416, 0),
    "q2_0-s2.json": (28_800_138_432, 192, 1),
    "swift-IQ2_XS-s1.json": (39_788_473_344, 10_967_808, 0),
    "swift-IQ2_XS-s2.json": (28_363_693_824, 56_320, 1),
    "swift-IQ3_XXS-s1.json": (39_785_790_560, 10_966_240, 0),
    "swift-IQ3_XXS-s2.json": (36_180_282_560, 57_920, 1),
}
SHARD1_METADATA = {
    "q2_0-s1.json": (41, 512),
    "IQ2_XS-s1.json": (20, 512),
    "IQ3_XXS-s1.json": (23, 512),
    "IQ3_S-s1.json": (26, 512),
    "coder-s1.json": (31, 256),
    "swift-IQ2_XS-s1.json": (20, 512),
    "swift-IQ3_XXS-s1.json": (23, 512),
}
REQUIRED_METADATA = {
    "general.architecture": "qwen4exp",
    "qwen4exp.block_count": 48,
    "qwen4exp.embedding_length": 2560,
    "qwen4exp.attention.head_count": 24,
    "qwen4exp.attention.head_count_kv": 2,
    "qwen4exp.expert_used_count": 10,
    "split.count": 2,
    "split.tensors.count": 1224,
}
PLE_SHARD2_TENSOR = {
    "name": "per_layer_token_embd.weight",
    "ne": [160, 320_001_536],
    "type": "IQ4_NL",
    "id": 20,
    "offset": 0,
    "bytes": 28_800_138_240,
}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_fixture(name: str) -> dict:
    return json.loads(
        (FIXTURES / name).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )


def run_spike(script: str, *args: str) -> str:
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="strata-model-header-") as temporary:
        env["TMPDIR"] = temporary
        result = subprocess.run(
            [sys.executable, str(SPIKES / script), *args],
            cwd=FIXTURES,
            env=env,
            capture_output=True,
            check=True,
            text=True,
        )
    return result.stdout


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def run_ggufhdr(path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(SPIKES / "ggufhdr.py"), str(path)],
        cwd=FIXTURES,
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(result.stdout, object_pairs_hook=_unique_json_object)


def normalize_swift_temp_path(output: str) -> str:
    return re.sub(
        r"(?m)^(=+) .+/swift-merged\.json tensors ",
        r"\1 <TMP>/swift-merged.json tensors ",
        output,
    )


class ModelHeaderFixtures(unittest.TestCase):
    def test_fixture_json_rejects_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key: 'kv'"):
            json.loads(
                '{"kv": {}, "kv": {}}',
                object_pairs_hook=_unique_json_object,
            )

    def test_all_models_have_1224_unique_tensors_across_their_shards(self):
        expected_files = {name for shards in MODELS.values() for name in shards}
        self.assertEqual(expected_files, {p.name for p in FIXTURES.glob("*.json")})

        for model, shards in MODELS.items():
            with self.subTest(model=model):
                tensors = []
                for name in shards:
                    data = load_fixture(name)
                    self.assertIn("kv", data)
                    self.assertIn("data_start", data)
                    self.assertIn("tensors", data)
                    tensors.extend(tensor["name"] for tensor in data["tensors"])
                self.assertEqual(1224, len(tensors))
                self.assertEqual(len(tensors), len(set(tensors)))

    def test_shard_sizes_and_tensor_totals_match_the_specification(self):
        for name, (file_size, data_start, split_no) in SHARD_LAYOUT.items():
            with self.subTest(shard=name):
                data = load_fixture(name)
                tensor_bytes = sum(tensor["bytes"] for tensor in data["tensors"])
                self.assertEqual(data_start, data["data_start"])
                self.assertEqual(file_size - data_start, tensor_bytes)
                self.assertEqual(file_size, data["data_start"] + tensor_bytes)
                self.assertEqual(2, data["kv"]["split.count"])
                self.assertEqual(1224, data["kv"]["split.tensors.count"])
                self.assertEqual(split_no, data["kv"]["split.no"])

    def test_fixed_model_metadata_matches_section_2_5(self):
        for name, (file_type, expert_count) in SHARD1_METADATA.items():
            with self.subTest(shard=name):
                kv = load_fixture(name)["kv"]
                for key, expected in REQUIRED_METADATA.items():
                    self.assertEqual(expected, kv.get(key), key)
                self.assertEqual(file_type, kv.get("general.file_type"), "general.file_type")
                self.assertEqual(expert_count, kv.get("qwen4exp.expert_count"), "qwen4exp.expert_count")
                self.assertEqual(640, kv.get("qwen4exp.expert_feed_forward_length"))
                self.assertEqual(640, kv.get("qwen4exp.expert_shared_feed_forward_length"))
                self.assertNotIn("general.alignment", kv)
                for key in (
                    "qwen4exp.rope.scaling.type",
                    "qwen4exp.rope.scaling.factor",
                    "qwen4exp.rope.scaling.original_context_length",
                ):
                    self.assertNotIn(key, kv)

    def test_shared_ista_coder_shard2_contains_the_ple_tensor(self):
        data = load_fixture("q2_0-s2.json")
        self.assertEqual([PLE_SHARD2_TENSOR], data["tensors"])
        self.assertEqual(2, data["kv"]["split.count"])
        self.assertEqual(1224, data["kv"]["split.tensors.count"])
        self.assertEqual(1, data["kv"]["split.no"])

    def test_ggufhdr_parses_a_small_generated_header(self):
        metadata = b"".join(
            (
                _gguf_string("general.architecture")
                + struct.pack("<I", 8)
                + _gguf_string("qwen4exp"),
                _gguf_string("general.alignment")
                + struct.pack("<II", 4, 64),
                _gguf_string("probe.array")
                + struct.pack("<IIQii", 9, 5, 2, 7, -2),
            )
        )
        tensor = (
            _gguf_string("probe.weight")
            + struct.pack("<IQQIQ", 2, 2, 3, 0, 0)
        )
        header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 3) + metadata + tensor
        header_end = len(header)
        expected_data_start = (header_end + 63) // 64 * 64

        with tempfile.TemporaryDirectory(prefix="strata-gguf-header-") as temporary:
            path = Path(temporary) / "probe.gguf"
            path.write_bytes(header)
            parsed = run_ggufhdr(path)

        self.assertEqual(3, parsed["version"])
        self.assertEqual(header_end, parsed["header_end"])
        self.assertEqual(expected_data_start, parsed["data_start"])
        self.assertEqual(
            {
                "general.architecture": "qwen4exp",
                "general.alignment": 64,
                "probe.array": [7, -2],
            },
            parsed["kv"],
        )
        self.assertEqual(
            [
                {
                    "name": "probe.weight",
                    "ne": [2, 3],
                    "type": "F32",
                    "id": 0,
                    "offset": 0,
                    "bytes": 24,
                }
            ],
            parsed["tensors"],
        )

    def test_ista_summary_reproduces_from_fixtures(self):
        files = ("q2_0-s1.json", "IQ2_XS-s1.json", "IQ3_XXS-s1.json", "IQ3_S-s1.json", "coder-s1.json")
        actual = run_spike("summary.py", *files)
        expected = (SPIKES / "summary.txt").read_text(encoding="utf-8")
        self.assertEqual(expected, actual)

    def test_swift_summary_reproduces_from_fixtures(self):
        pairs = (
            "swift-IQ2_XS-s1.json+swift-IQ2_XS-s2.json",
            "swift-IQ3_XXS-s1.json+swift-IQ3_XXS-s2.json",
        )
        actual = normalize_swift_temp_path(run_spike("swiftsum.py", *pairs))
        expected = normalize_swift_temp_path(
            (SPIKES / "swift-summary.txt").read_text(encoding="utf-8")
        )
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
