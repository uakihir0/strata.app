import argparse
import io
import json
import math
import os
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from tools.metal.tf_compare import compare, load, load_ids, main, parse_margins, parse_rows


class TfCompare(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dump(self, name, rows, declared_rows=None):
        values = np.asarray(rows, dtype='<f4')
        if values.ndim != 2:
            raise AssertionError('test rows must be a matrix')
        declared_rows = values.shape[0] if declared_rows is None else declared_rows
        path = self.root / name
        path.write_bytes(struct.pack('<ii', values.shape[1], declared_rows) + values.tobytes())
        return path

    def test_load_uses_file_length_for_short_prompt_path_dump(self):
        path = self.dump('short.logits', [[1, 2, 0], [0, 1, 3]], declared_rows=5)
        dump = load(path)
        self.assertEqual(dump.vocab_size, 3)
        self.assertEqual(dump.declared_rows, 5)
        self.assertEqual(dump.first_position, 3)
        np.testing.assert_array_equal(dump.logits, [[1, 2, 0], [0, 1, 3]])

    def test_payload_must_be_whole_rows(self):
        path = self.root / 'ragged.logits'
        path.write_bytes(struct.pack('<ii', 3, 2) + np.asarray([1, 2, 0], dtype='<f4').tobytes() + b'\x00' * 5)
        with self.assertRaisesRegex(ValueError, 'not a whole number of 3-logit rows'):
            load(path)

    def test_file_cannot_hold_more_rows_than_the_header_declares(self):
        path = self.dump('overfull.logits', [[1, 2, 0], [0, 1, 3]], declared_rows=1)
        with self.assertRaisesRegex(ValueError, 'header advertises 1 rows but file contains 2'):
            load(path)

    def test_invalid_headers_are_rejected(self):
        for vocab, declared in ((1, 1), (3, 0)):
            with self.subTest(vocab=vocab, declared=declared):
                path = self.root / f'bad-{vocab}-{declared}.logits'
                path.write_bytes(struct.pack('<ii', vocab, declared) + np.zeros(vocab, dtype='<f4').tobytes())
                with self.assertRaisesRegex(ValueError, 'invalid header'):
                    load(path)

    def test_infinity_is_rejected_like_nan(self):
        path = self.dump('inf.logits', [[np.inf, 1, 0], [0, 3, 1]], declared_rows=2)
        with self.assertRaisesRegex(ValueError, 'NaN or infinity'):
            load(path)

    def test_row_and_margin_arguments_are_validated(self):
        for value in ('3:3', '-1:2', '1', 'a:b'):
            with self.subTest(rows=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_rows(value)
        self.assertEqual(parse_rows('1:4'), (1, 4))
        for value in ('0.5', '1,0.5', 'nan,1', '-0.1,1', '0.5,1,2'):
            with self.subTest(margins=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_margins(value)
        self.assertEqual(parse_margins('0.25,2'), (0.25, 2.0))

    def test_margin_bands_are_half_open(self):
        # Reference margins are exactly 0.5, 1.0 and 0.4: the boundary value belongs to the upper band.
        reference = load(self.dump('ref.logits', [[1.5, 1.0, 0.0], [2.0, 1.0, 0.0], [0.4, 0.0, -1.0]]))
        test = load(self.dump('test.logits', [[1.5, 1.0, 0.0], [2.0, 1.0, 0.0], [0.4, 0.0, -1.0]]))
        result = compare(reference, test, [0, 0, 0, 0])
        self.assertEqual([(band['range'], band['rows']) for band in result['reference_margin_bands']],
                         [('[0,0.5)', 1), ('[0.5,1)', 1), ('[1,infinity)', 1)])

    def test_token_id_parser_rejects_empty_comma_fields(self):
        path = self.root / 'tokens.ids'
        path.write_text('0, 2 1', encoding='utf-8')
        self.assertEqual(load_ids(path), [0, 2, 1])
        for text in ('12,,45', '12, ,45', ',12', '12,'):
            with self.subTest(text=text):
                path.write_text(text, encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'empty comma-separated token ID'):
                    load_ids(path)

    def test_single_row_marks_standard_error_unavailable(self):
        rows = [[2, 1, 0]]
        reference = self.dump('ref.logits', rows)
        test = self.dump('test.logits', rows)
        ids = self.root / 'tokens.ids'
        ids.write_text('0 2', encoding='utf-8')
        result = compare(load(reference), load(test), [0, 2])
        self.assertIsNone(result['paired_delta_nats_per_token']['standard_error'])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main([str(reference), str(test), '--ids', str(ids)]), 0)
        self.assertIn('standard error unavailable for one row', stdout.getvalue())

    def test_oversized_token_id_is_a_clean_input_error(self):
        reference = self.dump('ref.logits', [[2, 1, 0], [0, 3, 1]])
        test = self.dump('test.logits', [[2, 1, 0], [0, 3, 1]])
        ids = self.root / 'tokens.ids'
        ids.write_text(f'0 {1 << 100} 1', encoding='utf-8')
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(main([str(reference), str(test), '--ids', str(ids)]), 2)
        self.assertIn(f'token ID {1 << 100} at position 1 is outside the vocabulary range 0:3', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())

    def test_equal_dumps_report_zero_kl_and_paired_delta(self):
        rows = [[2, 1, 0], [0, 3, 1], [1, 0, 2], [4, 1, 0]]
        ref = load(self.dump('ref.logits', rows))
        test = load(self.dump('test.logits', rows))
        result = compare(ref, test, [0, 2, 1, 0, 2], margins=(1.5, 2.5))
        self.assertEqual(result['rows_compared'], 4)
        self.assertEqual(result['same_top1'], 4)
        self.assertAlmostEqual(result['kl_nats']['mean'], 0.0, places=12)
        self.assertAlmostEqual(result['paired_delta_nats_per_token']['mean'], 0.0, places=12)
        self.assertEqual([band['rows'] for band in result['reference_margin_bands']], [2, 1, 1])
        self.assertAlmostEqual(result['perplexity']['reference'], result['perplexity']['test'], places=12)

    def test_nontrivial_metrics_match_hand_calculated_values(self):
        reference_rows = [[2, 0], [0, 2]]
        test_rows = [[1, 0], [0, 3]]
        reference = load(self.dump('ref.logits', reference_rows))
        test = load(self.dump('test.logits', test_rows))
        result = compare(reference, test, [1, 1, 0])

        def log_probabilities(row):
            maximum = max(row)
            normalizer = maximum + np.log(sum(np.exp(value - maximum) for value in row))
            return np.asarray(row, dtype=np.float64) - normalizer

        ref_logp = [log_probabilities(row) for row in reference_rows]
        test_logp = [log_probabilities(row) for row in test_rows]
        target_logp = ([ref_logp[0][1], ref_logp[1][0]], [test_logp[0][1], test_logp[1][0]])
        row_kl = [sum(np.exp(ref_logp[row]) * (ref_logp[row] - test_logp[row])) for row in range(2)]
        deltas = [target_logp[0][row] - target_logp[1][row] for row in range(2)]
        self.assertAlmostEqual(result['kl_nats']['mean'], float(np.mean(row_kl)), places=12)
        self.assertAlmostEqual(result['kl_nats']['max'], float(max(row_kl)), places=12)
        self.assertAlmostEqual(result['perplexity']['reference'], float(np.exp(-np.mean(target_logp[0]))), places=12)
        self.assertAlmostEqual(result['perplexity']['test'], float(np.exp(-np.mean(target_logp[1]))), places=12)
        self.assertAlmostEqual(result['paired_delta_nats_per_token']['mean'], float(np.mean(deltas)), places=12)
        self.assertAlmostEqual(result['paired_delta_nats_per_token']['standard_error'],
                               float(np.std(deltas, ddof=1) / np.sqrt(2)), places=12)

    def test_finite_logits_that_overflow_perplexity_fail_cleanly(self):
        rows = [[3e38, -3e38]]
        reference = load(self.dump('ref.logits', rows))
        test = load(self.dump('test.logits', rows))
        with self.assertRaisesRegex(ValueError, 'non-finite metric'):
            compare(reference, test, [0, 1])

    def test_disagreement_and_row_range_are_reported(self):
        ref_rows = [[2, 1, 0], [0, 3, 1], [1, 0, 2], [4, 1, 0]]
        test_rows = [[2, 1, 0], [6, 2, 0], [1, 0, 2], [4, 1, 0]]
        ref = load(self.dump('ref.logits', ref_rows))
        test = load(self.dump('test.logits', test_rows))
        result = compare(ref, test, [0, 2, 1, 0, 2], rows=(1, 3))
        self.assertEqual(result['rows_compared'], 2)
        self.assertEqual(result['same_top1'], 1)
        self.assertEqual(result['disagreements'][0]['position'], 1)
        self.assertEqual(result['disagreements'][0]['reference_top1'], 1)
        self.assertEqual(result['disagreements'][0]['test_top1'], 0)

    def test_argmax_tie_selects_lowest_token_id(self):
        ref = load(self.dump('ref.logits', [[1, 1, 0]]))
        test = load(self.dump('test.logits', [[0, 2, 1]]))
        result = compare(ref, test, [0, 2])
        self.assertEqual(result['disagreements'][0]['reference_top1'], 0)
        self.assertEqual(result['disagreements'][0]['test_top1'], 1)

    def test_misaligned_and_nonfinite_dumps_fail(self):
        ref = load(self.dump('ref.logits', [[2, 1, 0], [0, 3, 1]], declared_rows=4))
        shifted = load(self.dump('shifted.logits', [[2, 1, 0], [0, 3, 1]], declared_rows=5))
        with self.assertRaisesRegex(ValueError, 'same token positions'):
            compare(ref, shifted, [0, 2, 1, 0, 2])
        bad_path = self.dump('bad.logits', [[np.nan, 1, 0], [0, 3, 1]], declared_rows=4)
        with self.assertRaisesRegex(ValueError, 'NaN or infinity'):
            load(bad_path)

    def test_streams_rows_across_chunk_boundaries_and_checks_target_ids(self):
        reference_rows = np.tile(np.asarray([5, 4, 3, 2, 1], dtype=np.float32), (70, 1))
        test_rows = reference_rows.copy()
        for position in (15, 32, 64):
            test_rows[position] = [0, 10, 3, 2, 1]
        ids = [0] + [position % 5 for position in range(70)]
        reference = load(self.dump('ref.logits', reference_rows))
        test = load(self.dump('test.logits', test_rows))
        result = compare(reference, test, ids)
        self.assertEqual(result['rows_compared'], 70)
        self.assertEqual(result['same_top1'], 67)
        self.assertEqual([row['position'] for row in result['disagreements']], [15, 32, 64])
        with self.assertRaisesRegex(ValueError, 'outside the vocabulary range'):
            compare(reference, test, [0, 8] + [0] * 69)

    def test_cli_writes_a_json_report(self):
        rows = [[2, 1, 0], [0, 3, 1]]
        reference = self.dump('ref.logits', rows)
        test = self.dump('test.logits', rows)
        ids = self.root / 'tokens.ids'
        ids.write_text('0 2 1', encoding='utf-8')
        output = self.root / 'result.json'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(reference), str(test), '--ids', str(ids), '--json', str(output)]), 0)
        result = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(result['same_top1'], 2)
        self.assertEqual(result['rows_compared'], 2)
        original_ids = ids.read_text(encoding='utf-8')
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(reference), str(test), '--ids', str(ids), '--json', str(ids)]), 2)
        self.assertEqual(ids.read_text(encoding='utf-8'), original_ids)
        hard_link = self.root / 'reference-hard-link.logits'
        os.link(reference, hard_link)
        original_logits = reference.read_bytes()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(reference), str(test), '--ids', str(ids), '--json', str(hard_link)]), 2)
        self.assertEqual(reference.read_bytes(), original_logits)
        self.assertEqual(hard_link.read_bytes(), original_logits)

    def test_rows_without_a_next_token_id_are_omitted(self):
        rows = [[2, 1, 0], [0, 3, 1], [1, 0, 2]]
        reference = load(self.dump('ref.logits', rows))
        test = load(self.dump('test.logits', rows))
        # Row p predicts ids[p + 1]. With three IDs, row 2 would need ids[3] and is omitted.
        result = compare(reference, test, [0, 2, 1])
        self.assertEqual(result['rows_compared'], 2)
        self.assertEqual(result['last_position'], 1)
        with self.assertRaisesRegex(ValueError, 'no selected dump row has a corresponding next-token ID'):
            compare(reference, test, [0])

    def test_dumps_must_have_the_same_vocabulary(self):
        reference = load(self.dump('ref.logits', [[2, 1, 0], [0, 3, 1]]))
        test = load(self.dump('test.logits', [[2, 1, 0, 0], [0, 3, 1, 0]]))
        with self.assertRaisesRegex(ValueError, r'vocabulary sizes differ \(3 vs 4\)'):
            compare(reference, test, [0, 2, 1])

    def test_negative_token_id_is_outside_the_vocabulary(self):
        reference = self.dump('ref.logits', [[2, 1, 0], [0, 3, 1]])
        test = self.dump('test.logits', [[2, 1, 0], [0, 3, 1]])
        ids = self.root / 'tokens.ids'
        ids.write_text('0 -1 1', encoding='utf-8')
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(reference), str(test), '--ids', str(ids)]), 2)
        self.assertIn('token ID -1 at position 1 is outside the vocabulary range 0:3', stderr.getvalue())

    def test_short_dumps_start_at_their_first_position(self):
        # Both dumps come from a prompt path: the header declares 5 rows and 2 are present, so row 0 is position 3.
        reference = load(self.dump('ref.logits', [[2, 1, 0], [0, 3, 1]], declared_rows=5))
        test = load(self.dump('test.logits', [[2, 1, 0], [6, 2, 0]], declared_rows=5))
        self.assertEqual(reference.first_position, 3)
        ids = [0, 1, 2, 0, 2, 1]
        result = compare(reference, test, ids)
        self.assertEqual(result['first_position'], 3)
        self.assertEqual(result['rows_compared'], 2)
        # Row 1 is position 4 and predicts ids[5]; its reference top-1 is 1 and its test top-1 is 0.
        self.assertEqual([(item['position'], item['reference_top1'], item['test_top1'])
                          for item in result['disagreements']], [(4, 1, 0)])
        # --rows counts rows of the present dump, so row 1 alone is position 4.
        second = compare(reference, test, ids, rows=(1, 2))
        self.assertEqual((second['first_position'], second['rows_compared']), (4, 1))

    def test_test_side_ties_select_the_lowest_token_id(self):
        tied = load(self.dump('tied.logits', [[1, 1, 0]]))
        same = compare(tied, load(self.dump('same.logits', [[1, 1, 0]])), [0, 2])
        self.assertEqual(same['same_top1'], 1)
        result = compare(load(self.dump('ref.logits', [[0, 2, 1]])), tied, [0, 2])
        self.assertEqual(result['disagreements'][0]['reference_top1'], 1)
        self.assertEqual(result['disagreements'][0]['test_top1'], 0)

    def test_disagreement_reports_margin_and_next_token_log_probabilities(self):
        reference = load(self.dump('ref.logits', [[2, 1, 0]]))
        test = load(self.dump('test.logits', [[0, 2, 1]]))
        item = compare(reference, test, [0, 2])['disagreements'][0]
        self.assertAlmostEqual(item['reference_margin'], 1.0, places=12)
        # The next token is ids[1] = 2. Both rows have the same normaliser, log(e^2 + e + 1).
        normaliser = math.log(math.exp(2) + math.exp(1) + 1)
        self.assertAlmostEqual(item['reference_next_logp'], 0.0 - normaliser, places=12)
        self.assertAlmostEqual(item['test_next_logp'], 1.0 - normaliser, places=12)

    def test_median_and_p99_follow_the_linear_percentile(self):
        # Row 0 has KL 0. Row 1 has KL k between reference (1/2, 1/2) and test (1/(1+e), e/(1+e)).
        # With two values the median is k/2, and NumPy's linear 99th percentile is 0.99 k.
        reference = load(self.dump('ref.logits', [[0.0, 0.0], [0.0, 0.0]]))
        test = load(self.dump('test.logits', [[0.0, 0.0], [0.0, 1.0]]))
        p = (0.5, 0.5)
        q = (1 / (1 + math.e), math.e / (1 + math.e))
        k = sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q))
        kl = compare(reference, test, [0, 0, 0])['kl_nats']
        self.assertAlmostEqual(kl['mean'], k / 2, places=12)
        self.assertAlmostEqual(kl['median'], k / 2, places=12)
        self.assertAlmostEqual(kl['p99'], 0.99 * k, places=12)
        self.assertAlmostEqual(kl['max'], k, places=12)

    def test_margin_bands_come_from_the_reference_only(self):
        reference = load(self.dump('ref.logits', [[1.2, 1.0, 0.0]]))  # reference margin 0.2
        test = load(self.dump('test.logits', [[5.0, 0.0, 0.0]]))  # test margin 5.0
        result = compare(reference, test, [0, 0])
        self.assertEqual([(band['range'], band['rows']) for band in result['reference_margin_bands']],
                         [('[0,0.5)', 1), ('[0.5,1)', 0), ('[1,infinity)', 0)])

    def test_log_softmax_stays_finite_for_large_logits(self):
        # exp(1000) overflows float64, so only the max-shifted form gives finite probabilities.
        reference = load(self.dump('ref.logits', [[1000.0, 999.0, 0.0]]))
        test = load(self.dump('test.logits', [[1001.0, 999.0, 0.0]]))
        result = compare(reference, test, [0, 0])

        def log_softmax(row):
            row = np.asarray(row, dtype=np.float64)
            return row - (row.max() + np.log(np.exp(row - row.max()).sum()))

        # KL is unchanged by shifting both rows by the same constant, so the shifted rows give the expected value.
        ref_logp = log_softmax([0.0, -1.0, -1000.0])
        test_logp = log_softmax([1.0, -1.0, -1000.0])
        expected = float(np.sum(np.exp(ref_logp) * (ref_logp - test_logp)))
        self.assertGreater(expected, 0.05)
        self.assertAlmostEqual(result['kl_nats']['mean'], expected, places=12)
        self.assertTrue(math.isfinite(result['perplexity']['reference']))

    def test_row_range_must_lie_inside_the_dumps(self):
        dump = load(self.dump('ref.logits', [[2, 1, 0], [0, 3, 1]]))
        with self.assertRaisesRegex(ValueError, 'row range 0:3 is outside the 2 rows'):
            compare(dump, dump, [0, 2, 1], rows=(0, 3))

    def test_cli_prints_the_default_margin_bands(self):
        rows = [[1.2, 1.0, 0.0], [2.0, 1.0, 0.0]]  # reference margins 0.2 and 1.0
        reference = self.dump('ref.logits', rows)
        test = self.dump('test.logits', rows)
        ids = self.root / 'tokens.ids'
        ids.write_text('0 0 0', encoding='utf-8')
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main([str(reference), str(test), '--ids', str(ids)]), 0)
        self.assertIn('margin [0,0.5): 1/1 same', stdout.getvalue())
        self.assertIn('margin [1,infinity): 1/1 same', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
