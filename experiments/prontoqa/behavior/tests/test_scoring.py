from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from tulu_behavior.scoring import (
    ground_proof_steps, parse_statement, score_response, validate_item,
)


def make_item(*, false=False, negative_target=False):
    predicate = "not opaque" if negative_target else "opaque"
    query_predicate = ("opaque" if negative_target else "not opaque") if false else predicate
    item = {
        "question": "Every wumpus is a yumpus. Each yumpus is a zumpus. "
                    f"Zumpuses are {predicate}. Dumpuses are not opaque. Fae is a wumpus.",
        "query": f"True or false: Fae is {query_predicate}.",
        "answer": "False" if false else "True",
        "hop": 3,
        "chain_of_thought": [
            "Fae is a wumpus.", "Every wumpus is a yumpus.", "Fae is a yumpus.",
            "Each yumpus is a zumpus.", "Fae is a zumpus.",
            f"Zumpuses are {predicate}.", f"Fae is {predicate}.",
        ],
    }
    item["text"] = item["question"] + " " + item["query"]
    return item


def response(item, steps=None, answer=None):
    if steps is None:
        steps = ground_proof_steps(item)
    if answer is None:
        answer = item["answer"]
    return "Reasoning:\n" + "\n".join(steps) + "\nAnswer: " + answer


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.item = make_item()

    def test_gold_metadata_and_reduced_proof(self):
        metadata = validate_item(self.item)
        self.assertEqual(metadata["min_hops"], 3)
        self.assertEqual(metadata["rule_count"], 4)
        self.assertEqual(metadata["gold_ground_step_count"], 4)
        for steps in (self.item["chain_of_thought"], ground_proof_steps(self.item)):
            result = score_response(self.item, response(self.item, steps), "cot")
            self.assertTrue(result["joint_success"])
            self.assertTrue(result["format_compliant"])

    def test_every_query_and_target_polarity(self):
        for false in (False, True):
            for negative in (False, True):
                with self.subTest(false=false, negative=negative):
                    item = make_item(false=false, negative_target=negative)
                    validate_item(item)
                    self.assertTrue(score_response(item, response(item), "cot")["joint_success"])

    def test_wrong_label_does_not_make_correct_proof_wrong(self):
        result = score_response(self.item, response(self.item, answer="False"), "cot")
        self.assertFalse(result["answer_correct"])
        self.assertTrue(result["proof_success"])
        self.assertFalse(result["joint_success"])

    def test_wrong_fact_then_right_answer(self):
        steps = ground_proof_steps(self.item)
        steps[1] = "Fae is a dumpus."
        result = score_response(self.item, response(self.item, steps), "cot")
        self.assertTrue(result["answer_correct"])
        self.assertFalse(result["proof_success"])
        self.assertTrue(result["has_logic_error"])
        self.assertTrue(result["has_skip"])

    def test_skipped_middle_step_is_not_a_logic_error(self):
        result = score_response(self.item, response(self.item, ["Fae is opaque."]), "cot")
        self.assertTrue(result["has_skip"])
        self.assertFalse(result["has_logic_error"])
        self.assertTrue(result["proof_parseable"])
        self.assertFalse(result["proof_success"])

    def test_false_requires_explicit_negation(self):
        item = make_item(false=True)
        text = "There is no evidence that Fae is not opaque.\nAnswer: False"
        result = score_response(item, text, "cot")
        self.assertTrue(result["answer_correct"])
        self.assertFalse(result["proof_success"])
        self.assertFalse(result["proof_parseable"])

    def test_error_after_correct_conclusion_is_checked(self):
        result = score_response(self.item, response(self.item) + "\nFae is not opaque.", "cot")
        self.assertTrue(result["answer_correct"])
        self.assertTrue(result["diagnostics"]["reached_target"])
        self.assertTrue(result["has_logic_error"])
        self.assertFalse(result["proof_success"])
        self.assertFalse(result["format_compliant"])

    def test_unpunctuated_tail_is_not_lost(self):
        steps = ground_proof_steps(self.item) + ["Fae is small"]
        result = score_response(self.item, response(self.item, steps), "cot")
        self.assertEqual(result["step_count"], 5)
        self.assertTrue(result["has_logic_error"])
        self.assertFalse(result["proof_success"])

    def test_valid_unpunctuated_final_assertion(self):
        steps = [step.rstrip(".") for step in ground_proof_steps(self.item)]
        self.assertTrue(score_response(self.item, response(self.item, steps), "cot")["joint_success"])

    def test_effective_axiom_need_not_be_repeated(self):
        steps = ground_proof_steps(self.item)[1:]
        self.assertTrue(score_response(self.item, response(self.item, steps), "cot")["proof_success"])

    def test_equivalent_rules_and_bullets(self):
        self.assertEqual(parse_statement("Every wumpus is a yumpus."),
                         parse_statement("All wumpuses are yumpuses."))
        steps = [f"{index}. {step}" for index, step in enumerate(ground_proof_steps(self.item), 1)]
        self.assertTrue(score_response(self.item, response(self.item, steps), "cot")["joint_success"])

    def test_redundancy_and_valid_irrelevant_axiom_are_allowed(self):
        steps = ground_proof_steps(self.item)
        steps.insert(1, "Fae is a wumpus.")
        steps.insert(2, "Dumpuses are not opaque.")
        result = score_response(self.item, response(self.item, steps), "cot")
        self.assertTrue(result["joint_success"])
        self.assertEqual(result["diagnostics"]["redundant_count"], 1)

    def test_repeated_invalid_assertion_never_becomes_valid(self):
        result = score_response(self.item, response(self.item, ["Fae is small."] * 2), "cot")
        self.assertEqual(result["diagnostics"]["logic_error_count"], 2)
        self.assertEqual(result["diagnostics"]["redundant_count"], 0)

    def test_rule_shortcut_is_a_skip(self):
        steps = ["Every wumpus is a zumpus."] + ground_proof_steps(self.item)
        result = score_response(self.item, response(self.item, steps), "cot")
        self.assertTrue(result["has_skip"])
        self.assertFalse(result["proof_success"])

    def test_missing_final_conclusion(self):
        result = score_response(self.item, response(self.item, ground_proof_steps(self.item)[:-1]), "cot")
        self.assertTrue(result["proof_parseable"])
        self.assertFalse(result["proof_success"])
        self.assertTrue(result["diagnostics"]["missing_conclusion"])

    def test_correct_proof_without_answer(self):
        result = score_response(self.item, "\n".join(ground_proof_steps(self.item)), "cot")
        self.assertEqual(result["answer_status"], "missing")
        self.assertTrue(result["proof_success"])
        self.assertFalse(result["joint_success"])

    def test_empty_proof_and_refusal(self):
        for text in ("", "Answer: True", "I cannot solve this problem."):
            with self.subTest(text=text):
                result = score_response(self.item, text, "cot")
                self.assertFalse(result["proof_success"])
                self.assertFalse(result["proof_parseable"])

    def test_full_assertion_consumption(self):
        text = response(self.item, ground_proof_steps(self.item) + ["Fae is opaque because magic"])
        result = score_response(self.item, text, "cot")
        self.assertFalse(result["proof_parseable"])
        self.assertFalse(result["proof_success"])

    def test_answer_conflicts_anywhere(self):
        for text in ("Answer: True\nAnswer: False", "False\nAnswer: True",
                     "Answer: True\nFalse", "Answer: True or False"):
            with self.subTest(text=text):
                result = score_response(self.item, text, "direct")
                self.assertEqual(result["answer_status"], "conflicting")
                self.assertIsNone(result["extracted_answer"])
                self.assertFalse(result["answer_correct"])

    def test_malformed_answer_does_not_consume_proof(self):
        text = response(self.item).replace("Answer: True", "Answer: True because Fae is small")
        result = score_response(self.item, text, "cot")
        self.assertEqual(result["answer_status"], "invalid")
        self.assertFalse(result["proof_parseable"])
        self.assertEqual(result["step_count"], 5)

    def test_only_final_unmarked_label_is_accepted(self):
        final = "\n".join(ground_proof_steps(self.item)) + "\nTrue."
        self.assertTrue(score_response(self.item, final, "cot")["joint_success"])
        initial = "True\n" + "\n".join(ground_proof_steps(self.item))
        result = score_response(self.item, initial, "cot")
        self.assertEqual(result["answer_status"], "missing")
        self.assertFalse(result["proof_parseable"])

    def test_repeated_same_answer_is_format_failure(self):
        result = score_response(self.item, "Answer: True\nAnswer: True", "direct")
        self.assertTrue(result["answer_correct"])
        self.assertFalse(result["format_compliant"])

    def test_direct_fields_and_tolerant_label_case(self):
        for text in ("Answer: True", "true", "**Answer: True**", "Final Answer: True."):
            with self.subTest(text=text):
                result = score_response(self.item, text, "direct")
                self.assertTrue(result["answer_correct"])
                self.assertTrue(result["format_compliant"])
                self.assertIsNone(result["proof_success"])
                self.assertIsNone(result["joint_success"])
                self.assertIsNone(result["proof_parseable"])

    def test_direct_explanation_is_answer_correct_but_format_failure(self):
        result = score_response(self.item, "Fae is opaque.\nAnswer: True", "direct")
        self.assertTrue(result["answer_correct"])
        self.assertFalse(result["format_compliant"])

    def test_bad_gold_polarity_hop_and_inconsistency_are_rejected(self):
        for mutation in ("label", "hop", "contradiction", "proof"):
            item = copy.deepcopy(self.item)
            if mutation == "label":
                item["answer"] = "False"
            elif mutation == "hop":
                item["hop"] = 2
            elif mutation == "contradiction":
                item["question"] += " Fae is not opaque."
                item["text"] = item["question"] + " " + item["query"]
            else:
                item["chain_of_thought"] = ["Fae is opaque."]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_item(item)

    def test_all_frozen_items_gold_and_ground_proofs_pass(self):
        corpus = Path(__file__).resolve().parents[2] / "data/items.jsonl"
        if not corpus.exists():
            self.skipTest("Frozen parent corpus is unavailable")
        items = [json.loads(line) for line in corpus.read_text().splitlines()]
        self.assertEqual(len(items), 1440)
        for item in items:
            metadata = validate_item(item)
            self.assertEqual(metadata["min_hops"], 3)
            for steps in (item["chain_of_thought"], ground_proof_steps(item)):
                result = score_response(item, response(item, steps), "cot")
                self.assertTrue(result["joint_success"], item["sample_id"])


if __name__ == "__main__":
    unittest.main()
