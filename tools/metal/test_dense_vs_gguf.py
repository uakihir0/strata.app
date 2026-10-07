from __future__ import annotations

import contextlib
import io
import json
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import gguf_reader as G  # noqa: E402
from dense_vs_gguf import CATALOGUE_MODELS, DenseGGUFError, verify_model_packs, verify_pack  # noqa: E402


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def write_gguf(
    path: Path,
    tensors: list[tuple[str, bytes, int, list[int]]],
    *,
    split_no: int | None = 0,
    split_count: int = 1,
    total_tensors: int | None = None,
    include_architecture: bool = True,
    catalogue_identity: tuple[str, int] | None = None,
) -> None:
    metadata: list[tuple[str, int, bytes]] = [
        ("general.alignment", 4, struct.pack("<I", 32)),
        ("split.count", 2, struct.pack("<H", split_count)),
        ("split.tensors.count", 5, struct.pack("<i", total_tensors or len(tensors))),
    ]
    if split_no is not None:
        metadata.insert(2, ("split.no", 2, struct.pack("<H", split_no)))
    if include_architecture:
        metadata.append(("general.architecture", 8, _gguf_string("qwen4exp")))
        if catalogue_identity is not None:
            model_name, file_type = catalogue_identity
            metadata.extend([
                ("general.name", 8, _gguf_string(model_name)),
                ("general.file_type", 4, struct.pack("<I", file_type)),
            ])
    encoded_metadata = b"".join(
        _gguf_string(key) + struct.pack("<I", value_type) + value
        for key, value_type, value in metadata
    )

    tensor_offsets: list[int] = []
    body = bytearray()
    for name, raw, type_id, shape in tensors:
        offset = _align(len(body), 32)
        body.extend(b"\0" * (offset - len(body)))
        tensor_offsets.append(offset)
        body.extend(raw)
    encoded_tensors = bytearray()
    for (name, _raw, type_id, shape), offset in zip(tensors, tensor_offsets, strict=True):
        encoded_tensors.extend(_gguf_string(name))
        encoded_tensors.extend(struct.pack("<I", len(shape)))
        encoded_tensors.extend(struct.pack(f"<{len(shape)}Q", *shape))
        encoded_tensors.extend(struct.pack("<IQ", type_id, offset))

    header = (
        b"GGUF"
        + struct.pack("<IQQ", 3, len(tensors), len(metadata))
        + encoded_metadata
        + encoded_tensors
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(header)
        output.write(b"\0" * (_align(len(header), 32) - len(header)))
        output.write(body)


def _pack_row(name: str, kind: int, offset: int, size: int, ne0: int, ne1: int = 0) -> str:
    fields = [0, kind, offset, size, 0, size, ne0, ne1, 0, 0, 1] + [0] * 7
    return name + " " + " ".join(str(value) for value in fields)


def make_fixture(
    model_dir: Path,
    *,
    float_count: int = 40,
    converted: bool = True,
    split: bool = False,
    catalogue_identity: tuple[str, int] | None = None,
    omit_first_split_no: bool = False,
    quantized_form_native_row: bool = False,
    native_ple_key_type: str | None = None,
) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = model_dir / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    raw_tensors: list[tuple[str, bytes, int, list[int]]] = []
    pack_tensors: list[tuple[str, bytes, int, int, int]] = []
    conversions: list[dict[str, str]] = []

    for number in range(float_count):
        name = f"tensor_{number:03d}"
        raw = struct.pack("<I", 0x3F800000 + number)
        raw_tensors.append((name, raw, 0, [1]))
        pack_tensors.append((name, raw, 2, 1, 0))

    bf16_raw = b"\x80\x3f"
    f16_raw = b"\x00\x3c"
    raw_tensors.extend([
        ("tensor_bf16", bf16_raw, 30, [1]),
        ("tensor_f16", f16_raw, 1, [1]),
    ])
    pack_tensors.extend([
        ("tensor_bf16", bf16_raw, 4, 1, 0),
        ("tensor_f16", f16_raw, 5, 1, 0),
    ])

    if converted:
        name = "blk.0.ffn_gate_inp.weight"
        source_raw = struct.pack("<II", 0x3F800000, 0x40000000)
        packed_raw = b"\x80\x3f\x00\x40"
        raw_tensors.append((name, source_raw, 0, [2, 1]))
        pack_tensors.append((name, packed_raw, 4, 2, 1))
        conversions.append({
            "name": name,
            "shard": "model.gguf" if not split else "model-00001-of-00002.gguf",
            "src_type": "F32",
            "dst_type": "BF16",
            "method": "fixture f32->bf16 conversion",
            "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        })
    else:
        name = "blk.0.ffn_gate_inp.weight"
        source_raw = b"\x80\x3f\x00\x40"
        raw_tensors.append((name, source_raw, 30, [2, 1]))
        pack_tensors.append((name, source_raw, 4, 2, 1))

    # A tiny expert inventory lets the fixture build the real v3/v4 table from its own GGUF metadata.
    for role in ("gate", "up", "down"):
        raw_tensors.append((f"blk.0.ffn_{role}_exps.weight", struct.pack("<f", 1.0), 0, [1, 1, 1]))
    if quantized_form_native_row:
        name = "blk.0.ffn_gate_inp_shexp.weight"
        raw_tensors.append((name, bytes(18), 2, [32]))  # Q4_0, one block, requires a BF16 pack form.
        pack_tensors.append((name, b"", 0, 32, 0))
    if native_ple_key_type is not None:
        encodings = {"Q2_0": (42, 18, 64), "Q8_0": (8, 34, 32)}
        if native_ple_key_type not in encodings:
            raise ValueError(f"unsupported native PLE-key fixture type: {native_ple_key_type}")
        type_id, size, elements = encodings[native_ple_key_type]
        name = "blk.1.ple_key.weight"
        raw_tensors.append((name, bytes(size), type_id, [elements]))
        pack_tensors.append((name, b"", 0, elements, 0))

    if split:
        first_name = "model-00001-of-00002.gguf"
        second_name = "model-00002-of-00002.gguf"
        first_tensors = raw_tensors[::2]
        second_tensors = raw_tensors[1::2]
        write_gguf(
            model_dir / first_name,
            first_tensors,
            split_no=None if omit_first_split_no else 0,
            split_count=2,
            total_tensors=len(raw_tensors),
            include_architecture=True,
            catalogue_identity=catalogue_identity,
        )
        write_gguf(
            model_dir / second_name,
            second_tensors,
            split_no=1,
            split_count=2,
            total_tensors=len(raw_tensors),
            include_architecture=False,
        )
        for conversion in conversions:
            conversion["shard"] = first_name
        source_names = [first_name, second_name]
    else:
        write_gguf(model_dir / "model.gguf", raw_tensors, catalogue_identity=catalogue_identity)
        source_names = ["model.gguf"]

    dense = bytearray()
    rows: list[str] = []
    for name, raw, kind, ne0, ne1 in pack_tensors:
        offset = _align(len(dense), 64)
        dense.extend(b"\0" * (offset - len(dense)))
        rows.append(_pack_row(name, kind, offset, len(raw), ne0, ne1))
        dense.extend(raw)
    (pack_dir / "dense.bin").write_bytes(dense)
    (pack_dir / "index.txt").write_text(
        "# strata pack index v3 -- generated by test fixture\n"
        "# align 64 pool 0 tensors 0\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": 1,
        "tool": "tools/iq_pack.py",
        "compat_bf16": False,
        "source_shards": [
            {"name": name, "size": (model_dir / name).stat().st_size}
            for name in source_names
        ],
        "tensors": conversions,
    }
    (pack_dir / "conversions.json").write_text(json.dumps(manifest), encoding="utf-8")
    import iq_pack
    original_check_split = iq_pack.check_split
    try:
        if omit_first_split_no:
            # Build marker bytes from this intentionally invalid split fixture; verify_pack must reject its metadata first.
            iq_pack.check_split = lambda _files: None
        with contextlib.redirect_stdout(io.StringIO()):
            first_shard = model_dir / source_names[0]
            native_experts = iq_pack.expert_layout(iq_pack.Model(first_shard), first_shard)
    finally:
        iq_pack.check_split = original_check_split
    if isinstance(native_experts, str):
        raise AssertionError(f"could not generate fixture expert table: {native_experts}")
    (pack_dir / "native_experts.txt").write_text(native_experts[1], encoding="utf-8")
    return model_dir, pack_dir


class DenseVsGGUFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_compare_and_deterministic_sample_skip_converted_tensor(self):
        model_dir, pack_dir = make_fixture(self.root / "model")

        first = verify_pack(pack_dir, model_dir, sample_count=10)
        second = verify_pack(pack_dir, model_dir, sample_count=10)
        full = verify_pack(pack_dir, model_dir)

        self.assertTrue(first.sampled)
        self.assertEqual(first.tensor_names, second.tensor_names)
        self.assertEqual(first.tensors_checked, 10)
        self.assertEqual(first.conversions_checked, 1)
        self.assertEqual(first.conversion_bytes_checked, 4)
        self.assertEqual(full.tensors_checked, 42)
        self.assertEqual(full.conversions_checked, 1)
        self.assertEqual(full.conversion_bytes_checked, 4)
        self.assertEqual(full.bytes_checked, 164)
        self.assertNotIn("blk.0.ffn_gate_inp.weight", full.tensor_names)

    def test_sampled_and_full_compare_catch_a_flipped_dense_byte(self):
        model_dir, pack_dir = make_fixture(self.root / "model")
        report = verify_pack(pack_dir, model_dir, sample_count=32)
        rows = {
            line.split()[0]: int(line.split()[3])
            for line in (pack_dir / "index.txt").read_text().splitlines()
            if line and not line.startswith("#")
        }
        tensor_name = report.tensor_names[0]
        dense_path = pack_dir / "dense.bin"
        with dense_path.open("r+b") as dense:
            dense.seek(rows[tensor_name])
            byte = dense.read(1)
            dense.seek(rows[tensor_name])
            dense.write(bytes([byte[0] ^ 0x01]))

        with self.assertRaisesRegex(DenseGGUFError, rf"{tensor_name}: byte mismatch at tensor offset 0"):
            verify_pack(pack_dir, model_dir, sample_count=32)
        with self.assertRaisesRegex(DenseGGUFError, rf"{tensor_name}: byte mismatch at tensor offset 0"):
            verify_pack(pack_dir, model_dir)

    def test_full_compare_reads_tensors_from_each_split_shard(self):
        model_dir, pack_dir = make_fixture(self.root / "split-model", float_count=4, split=True)

        report = verify_pack(pack_dir, model_dir)

        self.assertEqual(report.tensors_checked, 6)
        self.assertEqual(report.bytes_checked, 20)
        self.assertEqual(report.conversions_checked, 1)
        self.assertEqual(report.conversion_bytes_checked, 4)

    def test_conversions_must_name_a_valid_float_index_row(self):
        model_dir, pack_dir = make_fixture(self.root / "model")
        manifest_path = pack_dir / "conversions.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tensors"].append(dict(manifest["tensors"][0]))
        manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(DenseGGUFError, "duplicate conversion for blk.0.ffn_gate_inp.weight"):
            verify_pack(pack_dir, model_dir)

    def test_conversion_cannot_be_used_to_hide_an_unconverted_tensor(self):
        model_dir, pack_dir = make_fixture(self.root / "model", converted=False)
        manifest_path = pack_dir / "conversions.json"
        manifest = json.loads(manifest_path.read_text())
        source_path = model_dir / "model.gguf"
        gguf = G.GGUFFile(source_path)
        tensor = next(tensor for tensor in gguf.tensors if tensor.name == "tensor_000")
        offset = gguf.data_start + tensor.offset
        with source_path.open("rb") as source:
            source.seek(offset)
            source_hash = hashlib.sha256(source.read(tensor.expected_bytes())).hexdigest()
        manifest["tensors"].append({
            "name": "tensor_000",
            "shard": source_path.name,
            "src_type": "F32",
            "dst_type": "F32",
            "method": "forged no-op conversion",
            "source_sha256": source_hash,
        })
        manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(DenseGGUFError, "does not change tensor type"):
            verify_pack(pack_dir, model_dir)

    def test_conversion_source_hash_must_match_the_gguf_tensor(self):
        model_dir, pack_dir = make_fixture(self.root / "model")
        source_path = model_dir / "model.gguf"
        gguf = G.GGUFFile(source_path)
        tensor = next(tensor for tensor in gguf.tensors if tensor.name == "blk.0.ffn_gate_inp.weight")
        offset = gguf.data_start + tensor.offset
        with source_path.open("r+b") as source:
            source.seek(offset)
            byte = source.read(1)
            source.seek(offset)
            source.write(bytes([byte[0] ^ 1]))

        with self.assertRaisesRegex(DenseGGUFError, "source_sha256 does not match"):
            verify_pack(pack_dir, model_dir)

    def test_conversion_record_must_match_the_tensor_pack_form(self):
        model_dir, pack_dir = make_fixture(self.root / "model")
        manifest_path = pack_dir / "conversions.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tensors"][0]["dst_type"] = "F16"
        manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(
            DenseGGUFError,
            "conversion destination F16 does not match the pack form BF16",
        ):
            verify_pack(pack_dir, model_dir)

    def test_quantized_form_tensor_cannot_be_marked_native_to_skip_conversion(self):
        model_dir, pack_dir = make_fixture(
            self.root / "model", converted=False, quantized_form_native_row=True
        )

        with self.assertRaisesRegex(
            DenseGGUFError,
            "blk.0.ffn_gate_inp_shexp.weight: required BF16 pack form is marked as native",
        ):
            verify_pack(pack_dir, model_dir)

    def test_native_quantized_ple_key_encodings_are_explicit_exceptions(self):
        for encoding in ("Q2_0", "Q8_0"):
            with self.subTest(encoding=encoding):
                model_dir, pack_dir = make_fixture(
                    self.root / encoding, native_ple_key_type=encoding
                )

                report = verify_pack(pack_dir, model_dir)

                self.assertEqual(report.tensors_checked, 42)

    def test_native_expert_table_must_match_source_gguf_layout(self):
        mutations = ("role type", "expert count", "source shard offset")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                model_dir, pack_dir = make_fixture(self.root / mutation.replace(" ", "-"))
                table_path = pack_dir / "native_experts.txt"
                lines = table_path.read_text().splitlines()
                if mutation == "role type":
                    fields = lines[1].split()
                    fields[1] = "999"
                    lines[1] = " ".join(fields)
                elif mutation == "expert count":
                    expected_header = "(n_expert 1, total 12;"
                    self.assertIn(expected_header, lines[0])
                    lines[0] = lines[0].replace(
                        expected_header, "(n_expert 2, total 24;", 1
                    )
                else:
                    fields = lines[1].split()
                    fields[5] = str(int(fields[5]) + 32)
                    lines[1] = " ".join(fields)
                table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(
                    DenseGGUFError,
                    "expert table does not match the layout derived from the GGUF",
                ):
                    verify_pack(pack_dir, model_dir)

    def test_missing_required_conversion_record_is_rejected(self):
        model_dir, pack_dir = make_fixture(self.root / "model")
        manifest_path = pack_dir / "conversions.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tensors"] = []
        manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(
            DenseGGUFError,
            "blk.0.ffn_gate_inp.weight: missing required conversion from F32 to BF16",
        ):
            verify_pack(pack_dir, model_dir)

    def test_non_compat_manifest_refuses_lossy_bf16_conversion(self):
        model_dir, pack_dir = make_fixture(self.root / "model")
        source_path = model_dir / "model.gguf"
        gguf = G.GGUFFile(source_path)
        tensor = next(tensor for tensor in gguf.tensors if tensor.name == "blk.0.ffn_gate_inp.weight")
        offset = gguf.data_start + tensor.offset
        with source_path.open("r+b") as source:
            source.seek(offset)
            source.write(struct.pack("<I", 0x3F800001))
        manifest_path = pack_dir / "conversions.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tensors"][0]["source_sha256"] = hashlib.sha256(
            struct.pack("<II", 0x3F800001, 0x40000000)
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(DenseGGUFError, "compat_bf16=False.*use --compat-bf16"):
            verify_pack(pack_dir, model_dir)

    def test_compat_conversion_bytes_are_recomputed(self):
        model_dir, pack_dir = make_fixture(self.root / "model")
        source_path = model_dir / "model.gguf"
        gguf = G.GGUFFile(source_path)
        tensor = next(tensor for tensor in gguf.tensors if tensor.name == "blk.0.ffn_gate_inp.weight")
        source_offset = gguf.data_start + tensor.offset
        with source_path.open("r+b") as source:
            source.seek(source_offset)
            source.write(struct.pack("<I", 0x3F800001))
        manifest_path = pack_dir / "conversions.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["compat_bf16"] = True
        manifest["tensors"][0]["source_sha256"] = hashlib.sha256(
            struct.pack("<II", 0x3F800001, 0x40000000)
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        dense_path = pack_dir / "dense.bin"
        row = next(
            line.split()
            for line in (pack_dir / "index.txt").read_text().splitlines()
            if line.startswith("blk.0.ffn_gate_inp.weight ")
        )
        with dense_path.open("r+b") as dense:
            dense.seek(int(row[3]))
            dense.write(b"\x81\x3f")

        with self.assertRaisesRegex(DenseGGUFError, "converted dense.bin mismatch"):
            verify_pack(pack_dir, model_dir)

    def test_incomplete_or_pending_pack_is_rejected(self):
        model_dir, pack_dir = make_fixture(self.root / "model")
        completion = pack_dir / "native_experts.txt"
        completion.unlink()
        with self.assertRaisesRegex(DenseGGUFError, "required pack file is missing: .*native_experts.txt"):
            verify_pack(pack_dir, model_dir)

        completion.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(DenseGGUFError, "native expert table is empty"):
            verify_pack(pack_dir, model_dir)

        _complete_model_dir, complete_pack_dir = make_fixture(self.root / "complete-model")
        completion.write_text(
            (complete_pack_dir / "native_experts.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (pack_dir / ".strata-generation-pending").write_text("interrupted\n", encoding="utf-8")
        with self.assertRaisesRegex(DenseGGUFError, "pack generation is incomplete"):
            verify_pack(pack_dir, model_dir)

    def test_split_source_requires_split_number_metadata(self):
        model_dir, pack_dir = make_fixture(
            self.root / "split-model", float_count=4, split=True, omit_first_split_no=True
        )

        with self.assertRaisesRegex(DenseGGUFError, "split.no is missing from a multi-shard source"):
            verify_pack(pack_dir, model_dir)

    def test_full_compare_rejects_a_missing_source_float_row(self):
        model_dir, pack_dir = make_fixture(self.root / "model", converted=False)
        index_path = pack_dir / "index.txt"
        lines = index_path.read_text().splitlines()
        index_path.write_text("\n".join(line for line in lines if not line.startswith("tensor_000 ")) + "\n")

        with self.assertRaisesRegex(DenseGGUFError, "tensor_000: source float tensor has no dense index row"):
            verify_pack(pack_dir, model_dir)

    def test_full_compare_rejects_a_source_float_marked_as_native(self):
        model_dir, pack_dir = make_fixture(self.root / "model", converted=False)
        index_path = pack_dir / "index.txt"
        lines = index_path.read_text().splitlines()
        for number, line in enumerate(lines):
            if line.startswith("tensor_000 "):
                fields = line.split()
                fields[2] = "0"
                lines[number] = " ".join(fields)
                break
        else:
            self.fail("fixture did not contain tensor_000")
        index_path.write_text("\n".join(lines) + "\n")

        with self.assertRaisesRegex(DenseGGUFError, "tensor_000: source float tensor has no dense index row"):
            verify_pack(pack_dir, model_dir)

    def test_source_shard_size_must_match_manifest(self):
        model_dir, pack_dir = make_fixture(self.root / "model")
        manifest_path = pack_dir / "conversions.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_shards"][0]["size"] += 1
        manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(DenseGGUFError, "size is .* conversions.json records"):
            verify_pack(pack_dir, model_dir)

    def test_model_root_requires_and_checks_all_seven_catalogue_packs(self):
        models_root = self.root / "models"
        catalogue = list(CATALOGUE_MODELS.items())
        for number, (identity, _model_name) in enumerate(catalogue):
            make_fixture(
                models_root / f"model-{number}",
                float_count=1,
                converted=False,
                catalogue_identity=identity,
            )

        results = verify_model_packs(models_root)
        self.assertEqual(len(results), 7)
        self.assertEqual(sum(report.tensors_checked for _, report in results), 28)
        self.assertEqual(
            {report.catalogue_model for _, report in results},
            set(CATALOGUE_MODELS.values()),
        )

        with self.assertRaisesRegex(DenseGGUFError, "expected 8 catalogue packs"):
            verify_model_packs(models_root, expected_count=8)

    def test_model_root_rejects_duplicate_or_unknown_catalogue_models(self):
        identities = list(CATALOGUE_MODELS)
        duplicate_root = self.root / "duplicate-models"
        for number in range(7):
            identity = identities[0] if number == 6 else identities[number]
            make_fixture(
                duplicate_root / f"model-{number}",
                float_count=1,
                converted=False,
                catalogue_identity=identity,
            )
        with self.assertRaisesRegex(DenseGGUFError, "duplicate catalogue model"):
            verify_model_packs(duplicate_root)

        unknown_root = self.root / "unknown-models"
        for number, identity in enumerate(identities):
            if number == 6:
                identity = ("Uncatalogued model", 999)
            make_fixture(
                unknown_root / f"model-{number}",
                float_count=1,
                converted=False,
                catalogue_identity=identity,
            )
        with self.assertRaisesRegex(DenseGGUFError, "does not identify one of the seven catalogue models"):
            verify_model_packs(unknown_root)


if __name__ == "__main__":
    unittest.main()
