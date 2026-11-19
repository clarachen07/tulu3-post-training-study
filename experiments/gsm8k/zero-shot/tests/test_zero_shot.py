import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compare
import prepare
import run_stage
from common import fingerprint, write_json, write_jsonl


class ZeroShotTests(unittest.TestCase):
    prefix = "\n\n".join("Question: Demo {}?\nAnswer: 7".format(i) for i in range(8)) + "\n\nQuestion: "

    def test_removes_only_examples_and_preserves_original_target_bytes(self):
        row = {"id": "problem", "content": self.prefix + "A  target?\n",
               "question": "A target?", "question_hash": fingerprint("A target?"), "gold": "8"}
        self.assertEqual(prepare.derive([row], self.prefix), [dict(row, content="Question: A  target?\n")])
        self.assertTrue(row["content"].startswith(self.prefix))

    def test_rejects_unexpected_prefix_and_question(self):
        with self.assertRaises(AssertionError):
            prepare.derive([{"id": "a", "content": "wrong"}], self.prefix)
        with self.assertRaises(AssertionError):
            prepare.derive([{"id": "a", "content": self.prefix + "Changed?", "question": "Expected?"}], self.prefix)

    def test_all_historical_pairs_and_signed_changes(self):
        # Opposite signs between test and train surface accidental unsigned subtraction.
        expected = {"a": {"split": "test"}, "b": {"split": "train"}}
        new = {stage: {"a": {"correct": False}, "b": {"correct": True}} for stage in compare.STAGES}
        old = {i: {"correct": {group: {s: i == "a" for s in compare.STAGES}
                                for group in compare.GROUPS}} for i in expected}
        rows = compare.contrasts(expected, new, old, 20, 42)
        self.assertEqual(len(rows), 24)
        for row in rows:
            sign = 1 if row["split"] == "train" else -1
            self.assertEqual((row["delta"], row["ci_low"], row["ci_high"]), (sign,) * 3)
            self.assertEqual(row["wrong_to_correct"], int(sign > 0))
            self.assertEqual(row["correct_to_wrong"], int(sign < 0))

    def test_completed_pilot_reuses_records_without_querying_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"inference": {}, "models": [{"stage": "base"}]}
            write_json(root / "experiment.json", config)
            source = {"id": "a", "gold": "7", "split": "train", "question_hash": "q",
                      "prompt_token_hash": "tokens", "prompt_text_hash": "text"}
            write_jsonl(root / "data/tokenized/base/pilot.jsonl", [source])
            record = dict(source, phase="pilot", stage="base", continuation="7", extracted_answer="7",
                          correct=True, answer_only=True, output_tokens=1, output_token_ids=[7],
                          truncated=False, finish_reason="stop", config_hash=fingerprint(config))
            record["record_hash"] = fingerprint(record)
            path = root / "outputs/pilot/base/batch_a.jsonl"
            write_jsonl(path, [record])
            before = path.read_bytes()
            with patch.object(run_stage, "ROOT", root), patch("shared_pipeline.ROOT", root), \
                 patch("sys.argv", ["run_stage.py", "--phase", "pilot", "--stage", "base"]), \
                 patch.object(run_stage, "gpu_busy", side_effect=AssertionError("GPU checked on resume")):
                run_stage.main()
                altered = dict(source, prompt_token_hash="changed")
                inputs = root / "data/tokenized/base/pilot.jsonl"
                inputs.unlink()
                write_jsonl(inputs, [altered])
                with self.assertRaises(AssertionError):
                    run_stage.main()
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
