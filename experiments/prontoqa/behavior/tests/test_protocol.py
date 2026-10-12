"""Independent checks against the actual frozen corpus and prompt protocol."""
from __future__ import annotations

import ast
from collections import Counter
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tulu_behavior.common import ROOT, fingerprint, load_config, physical_wrappers, read_jsonl, sha256_file
from tulu_behavior.data import prepare
from tulu_behavior.prompts import build_prompt
from tulu_behavior.scoring import score_response, validate_item


ORIGINAL_ITEMS_SHA256 = "aeb3adcb45fdb6e7c823ba9feea6901ea2b37fc642db7aad2a9cf9f51675ae80"


class ActualProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "data/formal.jsonl").exists():
            raise unittest.SkipTest("Full frozen corpus is local-only; see experiments/prontoqa/README.md")
        cls.config = load_config(ROOT)
        cls.manifest = json.loads((ROOT / "provenance/data.json").read_text())
        cls.formal = read_jsonl(ROOT / "data/formal.jsonl")
        cls.pilot = read_jsonl(ROOT / "data/pilot.jsonl")
        cls.demos = read_jsonl(ROOT / "data/demonstrations.jsonl")

    def test_actual_counts_and_split_disjointness(self):
        self.assertEqual(Counter(item["hop"] for item in self.formal), {1: 1440, 3: 1440, 5: 1440})
        self.assertEqual(Counter((item["hop"], item["answer"]) for item in self.pilot),
                         {(hop, label): 8 for hop in (1, 3, 5) for label in ("True", "False")})
        self.assertEqual([item["answer"] for item in self.demos], ["True", "False", "True", "False"])
        self.assertEqual([item["hop"] for item in self.demos], [3] * 4)
        all_items = self.formal + self.pilot + self.demos
        self.assertEqual(len(all_items), 4372)
        for field in ("sample_id", "text"):
            self.assertEqual(len({item[field] for item in all_items}), len(all_items))
        # This is stronger than full question+query deduplication: the same
        # theory cannot appear in another pool with an inverted query.
        pools = [self.formal, self.pilot, self.demos]
        for first in range(3):
            for second in range(first + 1, 3):
                self.assertFalse({item["question"] for item in pools[first]}
                                 & {item["question"] for item in pools[second]})

    def test_original_bytes_fields_ids_and_order_are_preserved(self):
        original_path = ROOT.parent / "data/items.jsonl"
        self.assertEqual(sha256_file(original_path), ORIGINAL_ITEMS_SHA256)
        original = read_jsonl(original_path)
        retained = [item for item in self.formal if item["hop"] == 3]
        self.assertEqual(len(original), len(retained))
        for before, after in zip(original, retained):
            self.assertEqual(before, {key: after[key] for key in before})
        self.assertEqual(self.manifest["original_3hop"]["sha256"], ORIGINAL_ITEMS_SHA256)

    def test_actual_artifact_hashes_and_all_gold_structures(self):
        self.assertEqual(self.manifest["data_fingerprint"], fingerprint(self.manifest["identity"]))
        for relative, expected in self.manifest["files"].items():
            self.assertEqual(sha256_file(ROOT / relative), expected, relative)
        for item in self.formal + self.pilot + self.demos:
            self.assertEqual(validate_item(item), item["validation"], item["sample_id"])

    def test_target_gold_cannot_affect_either_prompt_condition(self):
        for hop in (1, 3, 5):
            for label in ("True", "False"):
                item = next(item for item in self.formal if item["hop"] == hop and item["answer"] == label)
                hidden_changed = copy.deepcopy(item)
                hidden_changed["answer"] = "TARGET_ANSWER_MUST_NOT_BE_READ"
                hidden_changed["chain_of_thought"] = ["TARGET_PROOF_MUST_NOT_BE_READ"]
                hidden_changed["validation"] = {"secret": "TARGET_VALIDATION_MUST_NOT_BE_READ"}
                for condition in ("cot", "direct"):
                    prompt = build_prompt(item, self.demos, condition)
                    self.assertEqual(prompt, build_prompt(hidden_changed, self.demos, condition))
                    self.assertEqual(prompt.count("\nA:\n"), 5)
                    self.assertEqual(prompt.rsplit("\n\nQ: ", 1)[1], item["text"] + "\nA:\n")
                    self.assertNotIn("TARGET_", prompt)

    def test_actual_demonstrations_are_the_only_completed_answers(self):
        item = self.formal[0]
        for condition in ("cot", "direct"):
            prompt = build_prompt(item, self.demos, condition)
            blocks = prompt.split("\n\nQ: ")
            self.assertEqual(len(blocks), 6)
            for block, demo in zip(blocks[1:5], self.demos):
                question, answer = block.split("\nA:\n")
                self.assertEqual(question, demo["text"])
                scored = score_response(demo, answer, condition)
                self.assertTrue(scored["answer_correct"])
                self.assertTrue(scored["format_compliant"])
                if condition == "cot":
                    self.assertTrue(scored["joint_success"])
                else:
                    self.assertEqual(answer, "Answer: " + demo["answer"])
            self.assertTrue(blocks[-1].endswith("\nA:\n"))

    def test_completed_data_reuse_does_not_depend_on_generator_python(self):
        before = sha256_file(ROOT / "provenance/data.json")
        with patch("tulu_behavior.data.platform.python_version", return_value="3.10.21"), \
             patch("tulu_behavior.data.run_generator", side_effect=AssertionError("Must not regenerate")):
            self.assertEqual(prepare(ROOT), self.manifest)
        self.assertEqual(sha256_file(ROOT / "provenance/data.json"), before)
        for module in ("data.py", "prompts.py", "tokenize.py", "scoring.py"):
            ast.parse((ROOT.parent / "tulu_behavior" / module).read_text(), feature_version=(3, 10))

    def test_declared_matrix_matches_actual_counts_and_base_alias(self):
        stages = [model["stage"] for model in self.config["models"]]
        self.assertEqual(stages, ["base", "sft", "dpo", "rlvr"])
        wrappers = sum(len(physical_wrappers(stage)) for stage in stages)
        self.assertEqual(wrappers, 7)
        expected = self.config["expected"]
        self.assertEqual(wrappers * 2 * 3, expected["physical_cells_formal"])
        self.assertEqual(4 * 2 * 2 * 3, expected["logical_cells_formal"])
        self.assertEqual(len(self.formal) * wrappers * 2, expected["physical_generations_formal"])
        self.assertEqual(len(self.pilot) * wrappers * 2, expected["physical_generations_pilot_initial"])
        self.assertEqual(expected["formal_items"], len(self.formal))
        self.assertEqual(expected["pilot_items"], len(self.pilot))


if __name__ == "__main__":
    unittest.main()
