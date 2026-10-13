import io
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tools.metal import lc_dump_format as lc
from tools.metal.tf_compare import compare, load, main


class _Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def assert_no_output(self, path):
        self.assertFalse(path.exists(), f'{path} must not exist after a failed write')
        self.assertFalse(Path(f'{path}.partial').exists(), 'the partial file must be removed')


class LogitsFormat(_Base):
    def test_header_and_rows_follow_the_dump_logits_layout(self):
        rows = [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]
        path = self.root / 'lc-tf-test.logits'
        self.assertEqual(lc.write_logits(path, 3, rows), 2)
        data = path.read_bytes()
        self.assertEqual(data[:8], struct.pack('<ii', 3, 2))
        self.assertEqual(len(data), 8 + 2 * 3 * 4)
        np.testing.assert_array_equal(np.frombuffer(data[8:], dtype='<f4').reshape(2, 3), rows)
        self.assertFalse(Path(f'{path}.partial').exists())

    def test_full_vocabulary_rows_have_the_documented_size(self):
        # Section 2.5: a logits row is 248,320 x 4 bytes.
        path = self.root / 'full.logits'
        row = [0.0] * 248320
        lc.write_logits(path, 248320, [row, row])
        self.assertEqual(path.stat().st_size, 8 + 2 * 248320 * 4)
        dump = load(path)
        self.assertEqual((dump.vocab_size, dump.declared_rows, dump.logits.shape), (248320, 2, (2, 248320)))

    def test_logits_file_is_read_by_tf_compare_and_its_cli(self):
        reference = self.root / 'lc-tf-doc.logits'
        test = self.root / 'tf-doc.logits'
        ids = self.root / 'tf-doc.ids'
        lc.write_logits(reference, 3, [[2, 1, 0], [0, 3, 1], [1, 0, 2]])
        lc.write_logits(test, 3, [[2, 1, 0], [6, 2, 0], [1, 0, 2]])
        ids.write_text('0 2 1', encoding='utf-8')
        # Row 2 would need ids[3], which is absent, so only rows 0 and 1 are compared.
        result = compare(load(reference), load(test), [0, 2, 1])
        self.assertEqual(result['rows_compared'], 2)
        self.assertEqual([(item['position'], item['reference_top1'], item['test_top1'])
                          for item in result['disagreements']], [(1, 1, 0)])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(reference), str(test), '--ids', str(ids)]), 0)

    def test_invalid_logits_input_fails_without_output(self):
        path = self.root / 'bad.logits'
        with self.assertRaisesRegex(lc.LcDumpError, 'row 1 has 2 logits, expected 3'):
            lc.write_logits(path, 3, [[1, 2, 3], [1, 2]])
        self.assert_no_output(path)
        with self.assertRaisesRegex(lc.LcDumpError, 'no rows to write'):
            lc.write_logits(path, 3, [])
        self.assert_no_output(path)
        with self.assertRaisesRegex(lc.LcDumpError, 'at least two entries'):
            lc.write_logits(path, 1, [[1.0]])
        self.assert_no_output(path)

    def test_interrupted_logits_write_removes_the_partial_file(self):
        def rows():
            yield [1.0, 2.0, 3.0]
            raise RuntimeError('evaluation stopped')

        path = self.root / 'interrupted.logits'
        with self.assertRaisesRegex(RuntimeError, 'evaluation stopped'):
            lc.write_logits(path, 3, rows())
        self.assert_no_output(path)

    def test_final_name_appears_only_after_the_last_row(self):
        path = self.root / 'streamed.logits'
        seen = []

        def rows():
            for index in range(3):
                seen.append(path.exists())
                yield [float(index), 0.0, 1.0]

        lc.write_logits(path, 3, rows())
        self.assertEqual(seen, [False, False, False])
        self.assertEqual(load(path).declared_rows, 3)

    def test_failed_write_keeps_an_earlier_file_at_the_same_name(self):
        path = self.root / 'kept.logits'
        lc.write_logits(path, 3, [[1.0, 2.0, 3.0]])
        before = path.read_bytes()
        with self.assertRaisesRegex(lc.LcDumpError, 'expected 3'):
            lc.write_logits(path, 3, [[1.0, 2.0]])
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(Path(f'{path}.partial').exists())


class LadderFormat(_Base):
    @staticmethod
    def position(index):
        # Slot k of position p holds p * 100 + k, so reading the file back proves the slot order.
        return {name: np.full(lc.SLOT_FLOATS, index * 100 + slot, dtype='<f4')
                for slot, name in enumerate(lc.layer_node_names())}

    def test_node_names_and_record_size_follow_section_9_1(self):
        names = lc.layer_node_names()
        self.assertEqual(len(names), lc.LAYER_SLOTS)
        self.assertEqual(lc.LAYER_SLOTS, 49)
        self.assertEqual(names[0], 'hc_init')
        self.assertEqual(names[1], 'l_last-0')
        self.assertEqual(names[48], 'l_last-47')
        self.assertEqual(lc.SLOT_FLOATS, 4 * 2560)
        self.assertEqual(lc.layers_record_bytes(), 49 * 4 * 2560 * 4)

    def test_layers_file_has_the_slot_layout_and_reads_back(self):
        path = self.root / 'lc-ladder-test.layers'
        self.assertEqual(lc.write_layers(path, [self.position(0), self.position(1)]), 2)
        self.assertEqual(path.stat().st_size, 2 * 49 * 4 * 2560 * 4)
        self.assertEqual(lc.layers_position_count(path), 2)
        slots = np.fromfile(str(path), dtype='<f4').reshape(2, 49, 4 * 2560)
        for position in range(2):
            for slot in range(49):
                self.assertTrue(np.all(slots[position, slot] == position * 100 + slot),
                                f'position {position}, slot {slot}')
        self.assertFalse(Path(f'{path}.partial').exists())

    def test_renamed_graph_node_fails_and_leaves_no_output(self):
        for old, new in (('hc_init', 'hc_init_v2'), ('l_last-7', 'l_out-7')):
            with self.subTest(node=old):
                renamed = self.position(0)
                renamed[new] = renamed.pop(old)
                path = self.root / f'renamed-{old}.layers'
                with self.assertRaisesRegex(lc.LcDumpError, f"graph node '{old}' is missing"):
                    lc.write_layers(path, [renamed])
                self.assert_no_output(path)

    def test_renamed_node_in_a_later_position_leaves_no_output(self):
        renamed = self.position(1)
        renamed['l_last-30x'] = renamed.pop('l_last-30')
        path = self.root / 'later.layers'
        with self.assertRaisesRegex(lc.LcDumpError, "graph node 'l_last-30' is missing"):
            lc.write_layers(path, [self.position(0), renamed])
        self.assert_no_output(path)

    def test_node_with_another_shape_fails_and_leaves_no_output(self):
        position = self.position(0)
        position['l_last-3'] = np.zeros(lc.SLOT_FLOATS - 1, dtype='<f4')
        path = self.root / 'resized.layers'
        with self.assertRaisesRegex(lc.LcDumpError,
                                    "graph node 'l_last-3' holds 10239 values per position, expected 10240"):
            lc.write_layers(path, [position])
        self.assert_no_output(path)

    def test_no_positions_fails_without_output(self):
        path = self.root / 'none.layers'
        with self.assertRaisesRegex(lc.LcDumpError, 'no positions to write'):
            lc.write_layers(path, [])
        self.assert_no_output(path)

    def test_interrupted_layers_write_removes_the_partial_file(self):
        def positions():
            yield self.position(0)
            raise RuntimeError('evaluation stopped')

        path = self.root / 'interrupted.layers'
        with self.assertRaisesRegex(RuntimeError, 'evaluation stopped'):
            lc.write_layers(path, positions())
        self.assert_no_output(path)

    def test_final_name_appears_only_after_the_last_position(self):
        path = self.root / 'streamed.layers'
        seen = []

        def positions():
            for index in range(2):
                seen.append(path.exists())
                yield self.position(index)

        lc.write_layers(path, positions())
        self.assertEqual(seen, [False, False])
        self.assertEqual(lc.layers_position_count(path), 2)

    def test_failed_write_keeps_an_earlier_file_at_the_same_name(self):
        path = self.root / 'kept.layers'
        lc.write_layers(path, [self.position(0)])
        before = path.read_bytes()
        renamed = self.position(1)
        renamed['hc_init_v2'] = renamed.pop('hc_init')
        with self.assertRaisesRegex(lc.LcDumpError, "graph node 'hc_init' is missing"):
            lc.write_layers(path, [renamed])
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(Path(f'{path}.partial').exists())

    def test_reader_rejects_truncated_and_empty_ladders(self):
        path = self.root / 'ladder.layers'
        lc.write_layers(path, [self.position(0), self.position(1)])
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(lc.LcDumpError, 'not a whole number'):
            lc.layers_position_count(path)
        path.write_bytes(b'')
        with self.assertRaisesRegex(lc.LcDumpError, 'empty'):
            lc.layers_position_count(path)


if __name__ == '__main__':
    unittest.main()
