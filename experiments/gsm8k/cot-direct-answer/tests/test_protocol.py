import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import compare
import pipeline
import prepare
import run_stage
from common import fingerprint, digest_file


def save(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(json.dumps(r) for r in value) if jsonl else json.dumps(value)) + '\n')


class ProtocolTests(unittest.TestCase):
    def test_exact_derivation_preserves_every_other_field(self):
        prefix = 'Question: demo\nAnswer: 2 + 3 = 5.\n\nQuestion: '
        rows = [{'id': 'a', 'content': prefix + 'target?', 'gold': '9', 'metadata': {'x': 3}}]
        result = prepare.derive(rows, prefix, 'Direct answer.')
        self.assertEqual(result, [dict(rows[0], content='Direct answer.\n\n' + rows[0]['content'])])
        self.assertTrue(rows[0]['content'].startswith(prefix))

    def test_unexpected_original_prefix_rejected(self):
        with self.assertRaises(AssertionError):
            prepare.derive([{'id': 'a', 'content': 'changed'}], 'original', 'instruction')

    def test_aggregate_only_has_no_fabricated_paired_statistics(self):
        result = compare.contrast(None, np.array([1, 0, 1]), 3, 1, 100, 42)
        self.assertFalse(result['paired_available'])
        self.assertAlmostEqual(result['delta'], 1 / 3)
        for k in ['ci_low', 'ci_high', 'wrong_to_correct', 'correct_to_wrong', 'both_correct', 'both_wrong']:
            self.assertIsNone(result[k])

    def test_paired_changes_use_signed_differences(self):
        result = compare.contrast([1, 1, 0, 0], np.array([0, 1, 1, 0], dtype=np.uint8), 4, 2, 100, 42)
        self.assertEqual(result['delta'], 0)
        for k in ['wrong_to_correct', 'correct_to_wrong', 'both_correct', 'both_wrong']:
            self.assertEqual(result[k], 1)
        self.assertLessEqual(result['ci_low'], 0)
        self.assertGreaterEqual(result['ci_high'], 0)

    def test_missing_evidence_falls_back_without_running_anything(self):
        panel, sources, reason = compare.historical_panel(None, 'base', {}, {})
        self.assertIsNone(panel)
        self.assertEqual(sources, {})
        self.assertTrue(reason)

    def test_historical_pairing_requires_hashes_and_full_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, expected = [], {}
            for i, split in enumerate(['train', 'test']):
                ident = str(i)
                r = {'id':ident, 'split':split, 'stage':'base', 'phase':'formal', 'gold':'3',
                     'question_hash':fingerprint(ident), 'continuation':'3', 'extracted_answer':'3', 'correct':True}
                r['record_hash'] = fingerprint(r)
                rows.append(r)
                expected[ident] = r
            file = root / 'outputs/formal/base/batch_x.jsonl'
            save(file, rows, jsonl=True)
            ref = {'sources':{str(file.relative_to(root)):digest_file(file)},
                   'summary':[{'split':s,'stage':'base','n':1,'correct':1} for s in ['train','test']]}
            panel, _, reason = compare.historical_panel(root, 'base', expected, ref)
            self.assertEqual(set(panel), set(expected))
            self.assertIsNone(reason)
            save(file, rows[:1], jsonl=True)
            self.assertIsNone(compare.historical_panel(root, 'base', expected, ref)[0])
            ref['sources'][str(file.relative_to(root))] = digest_file(file)
            self.assertIsNone(compare.historical_panel(root, 'base', expected, ref)[0])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = {'inference': {'max_tokens':1024}, 'models':[{'stage':'base'}]}
        save(self.root / 'experiment.json', self.cfg)
        self.source = {'id':'a', 'gold':'7', 'split':'train', 'question_hash':'q',
                       'prompt_token_hash':'tokens', 'prompt_text_hash':'text'}
        save(self.root / 'data/tokenized/base/pilot.jsonl', [self.source], True)
        self.record = dict(self.source, phase='pilot', stage='base', continuation='7', extracted_answer='7',
                           correct=True, answer_only=True, output_tokens=1, output_token_ids=[7],
                           truncated=False, finish_reason='stop', config_hash=fingerprint(self.cfg))
        self.record['record_hash'] = fingerprint(self.record)
        self.file = self.root / 'outputs/pilot/base/batch_a.jsonl'
        save(self.file, [self.record], True)
        self.patches = [patch.object(pipeline, 'ROOT', self.root), patch.object(run_stage, 'ROOT', self.root)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.temp.cleanup()

    def test_completed_run_does_not_touch_gpu_or_output(self):
        before = self.file.read_bytes()
        with patch('sys.argv', ['run_stage.py','--phase','pilot','--stage','base']), patch.object(run_stage, 'gpu_busy', side_effect=AssertionError('GPU must not be queried')):
            run_stage.main()
        self.assertEqual(before, self.file.read_bytes())

    def test_changed_config_rejected_before_resume(self):
        save(self.root / 'experiment.json', {'inference':{'max_tokens':32}})
        with self.assertRaises(AssertionError):
            pipeline.validate_stage('pilot', 'base')

    def test_changed_prompt_rejected_before_resume(self):
        save(self.root / 'data/tokenized/base/pilot.jsonl', [dict(self.source,prompt_token_hash='different')], True)
        with self.assertRaises(AssertionError):
            pipeline.validate_stage('pilot', 'base')

    def test_duplicate_result_rejected(self):
        save(self.root / 'outputs/pilot/base/batch_duplicate.jsonl', [self.record], True)
        with self.assertRaises(AssertionError):
            pipeline.validate_stage('pilot', 'base')


if __name__ == '__main__':
    unittest.main()
