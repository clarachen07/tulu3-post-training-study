from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import protocol
import report
from protocol import (
    Decoder, EXPECTED_STAGE, FIG2_TASKS, MODEL_NAMES, STAGES, TASKS, atomic_json, encode,
    fingerprint, hops, label_for, prompt_parts, rates, sha256_file, write_gzip,
)
from report import summarize
from runner import Generator, load_tokenizer


def item(i):
    return {
        "x": f"query-{i}", "Fx": f"intermediate-{i}", "Gx": f"reverse-{i}",
        "GFx": f"target-{i}", "FGx": None,
    }


class ProtocolTests(unittest.TestCase):
    def test_task_counts_and_directions(self):
        self.assertEqual(sum(count for count, _, _ in TASKS.values()), 36802)
        self.assertEqual(EXPECTED_STAGE, 140686)
        self.assertEqual(hops("book-author-birthyear"), (("x", "Fx"), ("Fx", "GFx"), ("x", "GFx")))
        self.assertEqual(hops("plus-hundred-times-two")[-3:], (("x", "Gx"), ("Gx", "GFx"), ("x", "GFx")))
        self.assertEqual(len(FIG2_TASKS), 4)

    def test_demo_order_and_answer_boundaries(self):
        rows = [item(i) for i in range(20)]
        rows[5]["Gx"] = rows[0]["x"]
        demos = protocol.sample_contexts(rows)
        self.assertEqual(demos[0], [12, 13, 1, 8, 16, 15, 9, 11, 18, 6])
        fixtures = [rows[i] for i in demos[0]]
        user, prefix, target = prompt_parts("book-author-birthyear", ("x", "GFx"), rows[0], fixtures)
        self.assertTrue(user.startswith("Q: query-12\nA: target-12\n\n"))
        self.assertTrue(user.endswith("\n\nQ: query-0"))
        self.assertEqual(user.count("\nA: "), 10)
        self.assertEqual((prefix, target), ("A: ", "target-0"))
        self.assertEqual(prompt_parts("book-author-birthyear", ("x", "Fx"), rows[0], fixtures)[1:],
                         ("A:", " intermediate-0"))
        self.assertEqual(prompt_parts("word-int-times-two", ("x", "Fx"), rows[0], fixtures)[1:],
                         ("A: ", "intermediate-0"))

    def test_undefined_gap_and_bidirectional_flips(self):
        self.assertIsNone(rates([1, 0, 1, 0], 1)["gap"])
        result = summarize([
            [(True, True), (True, False), (False, True)],
            [(True, True), (True, True), (True, True)],
        ], hops("park-country-capital"))
        self.assertEqual(result["whole"]["gap"], 0.5)
        self.assertEqual(result["tokenwise"]["gap"], 0)
        effect = result["sensitivity"]
        self.assertEqual(effect["all_hops_denominator_change"], -1)
        self.assertEqual(effect["gap_change_pp"], -50)
        self.assertEqual(effect["flips"]["any_answer"],
                         {"unique": 1, "wrong_to_right": 1, "right_to_wrong": 1})


@unittest.skipUnless(os.environ.get("COMPOSITIONALITY_TOKENIZERS"), "requires the pinned tokenizer assets")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(os.environ["COMPOSITIONALITY_TOKENIZERS"])
        cls.tokenizers = {stage: load_tokenizer(directory / MODEL_NAMES[stage]) for stage in STAGES[1:]}

    def test_native_assistant_prefix_preserves_numerical_space(self):
        parts = [
            prompt_parts("book-author-birthyear", hop, item(0), [item(i) for i in range(1, 11)])
            for hop in hops("book-author-birthyear")
        ]
        for stage, tokenizer in self.tokenizers.items():
            with self.subTest(stage=stage):
                encoded = encode(tokenizer, stage, parts)
                for ids, (user, prefix, _) in zip(encoded, parts, strict=True):
                    expected = f"<|user|>\n{user}\n<|assistant|>\n{prefix}"
                    self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), expected)
                    self.assertNotIn(tokenizer.eos_token_id, ids)
                # The same formatter in plain mode lacks chat tokens and includes the tokenizer's BOS.
                self.assertEqual(encode(tokenizer, "base", parts)[0][0], tokenizer.bos_token_id)

    def test_whole_decode_preserves_split_unicode_and_exact_spaces(self):
        tokenizer = self.tokenizers["sft"]
        decode = Decoder(tokenizer)
        value = " 🫨"
        ids = tokenizer.encode(value + "\n\nQ:", add_special_tokens=False)
        raw, whole, tokenwise = decode(ids)
        self.assertEqual(raw, value + "\n\nQ:")
        self.assertEqual(whole, value)
        self.assertNotEqual(tokenwise, whole)
        self.assertEqual(decode([*ids, tokenizer.eos_token_id])[0], raw)

    def test_batch_stops_preserve_eos_and_exclude_later_padding(self):
        import torch
        from transformers import StopStringCriteria

        tokenizer = self.tokenizers["sft"]
        gen = Generator.__new__(Generator)
        gen.torch = torch
        gen.device = "cpu"
        gen.eos = torch.tensor([tokenizer.eos_token_id])
        gen.pad = tokenizer.eos_token_id
        gen.settings = None
        gen.stop = StopStringCriteria(tokenizer, ["\n\n"])
        answer = tokenizer.encode(" Lyon", add_special_tokens=False)[0]
        newline = tokenizer.encode("\n", add_special_tokens=False)[0]
        budget = protocol.MAX_NEW_TOKENS

        class ScriptedModel:
            def generate(self, input_ids, attention_mask, stopping_criteria, **kwargs):
                self_test.assertEqual(attention_mask.tolist(), [[0, 1], [1, 1], [1, 1]])
                script = (
                    [answer, gen.pad] + [gen.pad] * (budget - 2),
                    [answer, newline, newline] + [gen.pad] * (budget - 3),
                    [answer] * budget,
                )
                ids = input_ids
                for next_ids in zip(*script, strict=True):
                    ids = torch.cat((ids, torch.tensor(next_ids)[:, None]), dim=1)
                    stopping_criteria(ids, None)
                return ids

        self_test = self
        gen.model = ScriptedModel()
        records = gen([[0, 0, [1]], [0, 1, [1, 2]], [0, 2, [1, 2]]])
        self.assertEqual(records, [([answer, gen.pad], "eos"), ([answer, newline, newline], "stop"),
                                   ([answer] * budget, "length")])


class ArtifactTests(unittest.TestCase):
    def test_report_replay_and_corruption_detection(self):
        dataset = {task: [item(i) for i in range(4)] for task in FIG2_TASKS}
        visible = {task: (len(dataset[task]), "", TASKS[task][2]) for task in dataset}
        queries = {task: {"fig2": [0, 1, 2]} for task in dataset}
        count = sum(len(rows) * len(hops(task)) for task, rows in dataset.items())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / "data.json", {})
            atomic_json(root / "models.json", {})
            inputs = {"identity": {"data": fingerprint({}), "models": fingerprint({})},
                      "files": {}, "partition": {}}
            outputs = {}
            for stage in STAGES:
                inputs["partition"][stage] = {}
                for task, items in dataset.items():
                    key = f"inputs/{stage}/{task}/000000.json.gz"
                    inputs["partition"][stage][task] = [key]
                    ids, records = [], []
                    for i, row in enumerate(items):
                        for j, hop in enumerate(hops(task)):
                            label = label_for(task, hop, row)
                            whole = label if j < 2 else "wrong"
                            revised = label if i == 0 else whole
                            ids.append([i, j, [1234]])
                            records.append([i, j, [1234], "eos", label, whole, whole, revised])
                    write_gzip(root / key, ids)
                    inputs["files"][key] = sha256_file(root / key)
                    outputs[key.replace("inputs/", "predictions/", 1)] = (key, records)
            atomic_json(root / "inputs.json", inputs)
            run = {"runtime": {"input_manifest_sha256": sha256_file(root / "inputs.json")}, "batch_size": 1}
            atomic_json(root / "pilot.json", run)
            for output, (key, records) in outputs.items():
                write_gzip(root / output, {
                    "run": fingerprint(run), "input_sha256": inputs["files"][key],
                    "rows_sha256": fingerprint(records), "rows": records,
                })
            with (patch.object(report, "TASKS", visible),
                  patch.object(report, "EXPECTED_STAGE", count),
                  patch.object(report, "freeze_data", return_value=(dataset, queries, {})),
                  patch.object(report, "verify_queries")):
                results = report.build_report(root)
                report.verify(root)
                self.assertEqual(results["full"]["base"]["all_tasks_pooled"]["whole"]["composition"], 0)
                self.assertEqual(results["full"]["base"]["all_tasks_pooled"]["tokenwise"]["composition"], 0.25)
                first = root / next(iter(outputs))
                envelope = protocol.read_gzip(first)
                envelope["rows"][0][2] = [999]
                write_gzip(first, envelope)
                with self.assertRaisesRegex(ValueError, "Committed prediction shard"):
                    report.verify(root)


if __name__ == "__main__":
    unittest.main()
