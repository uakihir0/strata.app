import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.metal import make_prompts as mp


class CanonicalIdentity(unittest.TestCase):
    def test_hash_is_over_the_canonical_string_not_the_file(self):
        self.assertEqual(mp.ids_hash([1, 22, 333]), hashlib.sha256(b"1,22,333").hexdigest())

    def test_ids_file_is_one_line_of_commas(self):
        self.assertEqual(mp.parse_ids_file(b"1,22,333\n"), [1, 22, 333])

    def test_line_end_conversion_keeps_the_identity(self):
        self.assertEqual(mp.parse_ids_file(b"1,22,333\r\n"), [1, 22, 333])

    def test_non_canonical_or_multi_line_ids_files_are_rejected(self):
        for bad in (b"01,2\n", b"1, 2\n", b"1,2\n3\n", b"1,2", b"-1\n", b"1,,2\n", b"\n"):
            with self.subTest(bad=bad), self.assertRaises(mp.MakePromptsError):
                mp.parse_ids_file(bad)


class CutSearch(unittest.TestCase):
    def test_keeps_target_minus_head_and_tail(self):
        ids = mp.cut_files("x", 10, [1, 2], list(range(100, 200)), [3], lambda c: True)
        self.assertEqual(len(ids), 10)
        self.assertEqual(ids[:2], [1, 2])
        self.assertEqual(ids[-1], 3)

    def test_steps_back_one_token_at_a_time_until_a_cut_round_trips(self):
        calls = []

        def round_trip(ids):
            calls.append(len(ids))
            return len(ids) <= 9

        ids = mp.cut_files("x", 10, [1, 2], list(range(100, 200)), [3], round_trip)
        self.assertEqual(len(ids), 9)
        self.assertEqual(calls, [10, 9])

    def test_sixteenth_step_is_the_last_one_tried(self):
        ids = mp.cut_files("x", 1024, [], list(range(2000)), [], lambda c: len(c) == 1008)
        self.assertEqual(len(ids), 1008)

    def test_gives_up_after_seventeen_attempts_with_the_documented_message(self):
        calls = []

        def never(ids):
            calls.append(len(ids))
            return False

        with self.assertRaises(mp.MakePromptsError) as raised:
            mp.cut_files("ca-4k", 1024, [], list(range(2000)), [], never)
        self.assertEqual(len(calls), 17)
        self.assertIn("make_prompts: ca-4k: no cut within 16 tokens of 1024 round-trips", str(raised.exception))

    def test_too_few_file_tokens_is_a_clear_error(self):
        with self.assertRaisesRegex(mp.MakePromptsError, "the sources hold 5 file tokens, the set needs 10"):
            mp.cut_files("ca-1k", 10, [], [0] * 5, [], lambda c: True)


class PackAndSources(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_tokenizer_files_are_named(self):
        with self.assertRaisesRegex(mp.MakePromptsError, "lacks vocab.json"):
            mp.load_pack(self.root)

    def test_template_change_changes_the_tokenizer_hash_only_in_the_first_hash(self):
        pack_dir = mp.write_synthetic_pack(self.root / "p")
        first = mp.load_pack(pack_dir)
        (pack_dir / "tokenizer" / "chat_template.jinja").write_text("{{ messages|length }}", encoding="utf-8")
        second = mp.load_pack(pack_dir)
        self.assertNotEqual(first.sha256, second.sha256)
        self.assertEqual(first.aux_sha256, second.aux_sha256)

    def test_pre_tokenizer_other_than_qwen35_is_refused(self):
        pack_dir = mp.write_synthetic_pack(self.root / "p")
        cfg = pack_dir / "tokenizer" / "tokenizer.json"
        data = json.loads(cfg.read_text(encoding="utf-8"))
        data["pre"] = "llama3"
        cfg.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(mp.MakePromptsError, "only qwen35 is"):
            mp.load_pack(pack_dir)

    def test_missing_upstream_checkout_fails_before_any_build(self):
        with self.assertRaisesRegex(mp.MakePromptsError, "upstream Strata checkout not found"):
            mp.read_sources(self.root / "absent", self.root / "also-absent")

    def test_file_text_puts_each_file_under_a_header(self):
        self.assertEqual(mp.files_text((("src/a.py", b"x = 1"), ("src/b.py", b"y = 2\n"))),
                         "=== src/a.py ===\nx = 1\n=== src/b.py ===\ny = 2\n")

    def test_non_utf8_source_is_refused(self):
        with self.assertRaisesRegex(mp.MakePromptsError, "is not UTF-8 text"):
            mp.files_text((("src/a.py", b"\xff\xfe"),))

    def test_chat_transcript_needs_twenty_user_turns(self):
        turns = []
        for _ in range(19):
            turns += [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        with self.assertRaisesRegex(mp.MakePromptsError, "20 user turns"):
            mp._check_tf_chat({"schema": 1, "messages": turns})


class Output(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_names_that_escape_the_output_folder_are_not_set_names(self):
        self.assertIsNone(mp.SET_NAME.match("../x"))
        self.assertIsNone(mp.SET_FILE.match("../x.ids"))
        self.assertIsNone(mp.SET_FILE.match("sub/x.ids"))
        with self.assertRaisesRegex(mp.MakePromptsError, "refusing to write unexpected name"):
            mp.write_files(self.root / "out", {"../evil.ids": b"1\n"})

    def test_write_refuses_to_follow_a_symlink(self):
        out = self.root / "out"
        out.mkdir()
        victim = self.root / "victim.ids"
        victim.write_bytes(b"keep\n")
        os.symlink(victim, out / "ca-4k.ids")
        with self.assertRaisesRegex(mp.MakePromptsError, "symlink"):
            mp.write_files(out, {"ca-4k.ids": b"1\n"})
        self.assertEqual(victim.read_bytes(), b"keep\n")

    def test_second_tokenizer_writes_to_its_own_subfolder(self):
        out = self.root / "out"
        out.mkdir()
        manifest = {"schema": 1, "builder_version": 1, "sets": [{"tokenizer_sha256": "a" * 64}]}
        (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(mp.resolve_out(out, "a" * 64), out)
        self.assertEqual(mp.resolve_out(out, "b" * 64), out / ("b" * 12))


class BuildAndCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_self_test_passes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mp.self_test(), 0)

    def test_manifest_entry_has_the_fields_the_spec_names(self):
        pack = mp.load_pack(mp.write_synthetic_pack(self.root / "p"))
        src = mp.load_src(mp.SRC_DIR)
        _, manifest, _ = mp._rebuild(pack, src, mp.synthetic_sources())
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual(manifest["builder_version"], mp.BUILDER_VERSION)
        for entry in manifest["sets"]:
            self.assertEqual(set(entry), {"name", "tokens", "ids_sha256", "context", "source",
                                          "tokenizer_sha256", "tokenizer_aux_sha256"})
            self.assertEqual(entry["tokenizer_sha256"], pack.sha256)

    def test_check_detects_a_changed_ids_digit_and_a_changed_manifest(self):
        pack = mp.load_pack(mp.write_synthetic_pack(self.root / "p"))
        src = mp.load_src(mp.SRC_DIR)
        _, _, files = mp._rebuild(pack, src, mp.synthetic_sources())
        out = self.root / "out"
        mp.write_files(out, files)
        self.assertEqual(mp._compare_against_fresh(out, files), [])

        ids_path = out / "tf-code.ids"
        raw = bytearray(ids_path.read_bytes())
        raw[0] = ord("9") if raw[0] != ord("9") else ord("8")
        ids_path.write_bytes(bytes(raw))
        errors = mp._compare_against_fresh(out, files)
        self.assertTrue(any("tf-code.ids" in e for e in errors), errors)

        ids_path.write_bytes(files["tf-code.ids"])
        manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["sets"][0]["tokens"] += 1
        (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIn("MANIFEST.json differs from the rebuild", mp._compare_against_fresh(out, files))

    def test_check_fails_on_a_whitespace_only_change_to_the_manifest(self):
        pack = mp.load_pack(mp.write_synthetic_pack(self.root / "p"))
        _, _, files = mp._rebuild(pack, mp.load_src(mp.SRC_DIR), mp.synthetic_sources())
        out = self.root / "out"
        mp.write_files(out, files)
        manifest = out / "MANIFEST.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        self.assertIn("MANIFEST.json differs from the rebuild", mp._compare_against_fresh(out, files))

    def test_check_reports_an_unexpected_set_file(self):
        pack = mp.load_pack(mp.write_synthetic_pack(self.root / "p"))
        _, _, files = mp._rebuild(pack, mp.load_src(mp.SRC_DIR), mp.synthetic_sources())
        out = self.root / "out"
        mp.write_files(out, files)
        (out / "stray.ids").write_bytes(b"1\n")
        self.assertTrue(any("stray.ids" in e for e in mp._compare_against_fresh(out, files)))


if __name__ == "__main__":
    unittest.main()
