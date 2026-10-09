from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tulu_behavior.common import ROOT, atomic_json, atomic_jsonl, fingerprint, load_config, read_jsonl, sha256_file
from tulu_behavior.data import _collect, generator_arguments, generator_output_name, prepare
from tulu_behavior.prompts import build_prompt, demonstration_proof
from tulu_behavior.tokenize import encode_prompt, prepare_inputs


class FakeTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, None
    model_max_length = 100_000
    special_tokens_map = {"bos_token": "<BOS>", "eos_token": "<EOS>"}
    is_fast = True

    def __init__(self, native=True):
        self.chat_template = "<user>{{content}}</user><assistant>" if native else None
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(("plain", text, kwargs))
        ids = [self.bos_token_id] + [ord(c) + 10 for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.calls.append(("native", messages, tokenize, add_generation_prompt))
        rendered = "<user>" + messages[0]["content"] + "</user><assistant>"
        return [ord(c) + 10 for c in rendered] if tokenize else rendered

    def get_chat_template(self):
        return self.chat_template


class DataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tulu-behavior-data-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = copy.deepcopy(load_config(ROOT))
        self.config["data"].update(count_per_hop=6, pilot_per_hop=2,
                                   original_items="original/items.jsonl", original_manifest="original/data.json")
        corpus = ROOT.parent / "data/items.jsonl"
        if not corpus.exists():
            corpus = Path(__file__).parent / "fixtures/original-items.jsonl"
        self.legacy = read_jsonl(corpus)[:6]
        atomic_jsonl(self.root / "original/items.jsonl", self.legacy)
        atomic_json(self.root / "original/data.json", {
            "files": {"data/items.jsonl": sha256_file(self.root / "original/items.jsonl")}})
        atomic_json(self.root / "experiment.json", self.config)

    def test_real_generator_disjoint_pools_preserves_legacy_and_reuses_cross_environment(self):
        shutil.copytree(ROOT.parent / "vendor/prontoqa", self.root / "vendor/prontoqa")
        before = (self.root / "original/items.jsonl").read_bytes()
        manifest = prepare(self.root)
        formal = read_jsonl(self.root / "data/formal.jsonl")
        pilot = read_jsonl(self.root / "data/pilot.jsonl")
        demos = read_jsonl(self.root / "data/demonstrations.jsonl")
        self.assertEqual(manifest["counts"], {"formal": 18, "pilot": 6, "demonstrations": 4})
        self.assertEqual([x["answer"] for x in demos], ["True", "False", "True", "False"])
        self.assertTrue(all(x["hop"] == 3 for x in demos))
        self.assertEqual(len({x["text"] for x in formal + pilot + demos}), 28)
        self.assertEqual(len({x["sample_id"] for x in formal + pilot + demos}), 28)
        for hop in (1, 3, 5):
            selected = [x for x in pilot if x["hop"] == hop]
            self.assertEqual(sorted(x["answer"] for x in selected), ["False", "True"])
        legacy = [x for x in formal if x["hop"] == 3]
        for original, retained in zip(self.legacy, legacy):
            self.assertEqual(original, {key: retained[key] for key in original})
        self.assertEqual((self.root / "original/items.jsonl").read_bytes(), before)
        self.assertTrue(all(x["validation"]["min_hops"] == x["hop"] for x in formal + pilot + demos))
        with patch("tulu_behavior.data.platform.python_version", return_value="different-runtime"):
            self.assertEqual(prepare(self.root), manifest)
        with (self.root / "data/formal.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(RuntimeError, "changed recorded artifact"):
            prepare(self.root)

    def test_exact_duplicate_replenishment_uses_next_seed(self):
        calls = []

        def generate(root, pool, hop, seed, count, versions, sources):
            calls.append((seed, count))
            names = ["A", "A", "B", "B"] if len(calls) == 1 else ["C", "D"]
            self.assertEqual(len(names), count)
            return {f"example{i + 1}": {"test_example": {
                "question": f"{name} is a wumpus.", "query": f"True or false: {name} is blue.",
                "answer": "True", "chain_of_thought": [f"{name} is blue."],
            }} for i, name in enumerate(names)}, {"seed": seed, "files": {}}

        with patch("tulu_behavior.data.run_generator", side_effect=generate), \
             patch("tulu_behavior.data.validate_item", return_value={}):
            items, info = _collect(self.root, "formal", 1, 4, 100, set(), {}, {})
        self.assertEqual(calls, [(100, 4), (101, 2)])
        self.assertEqual(info["exact_text_duplicates_rejected"], 2)
        self.assertEqual([x["source_example"] for x in items], ["example1", "example3", "example1", "example2"])
        self.assertEqual([x["sample_id"] for x in items], [f"h1_q{i:06d}" for i in range(1, 5)])

    def test_arguments_keep_demonstrations_out_of_generation(self):
        args = generator_arguments(5, 62471905, 6)
        values = dict(zip(args[::2], args[1::2]))
        self.assertEqual(values["--few-shot-examples"], "0")
        self.assertEqual(values["--min-hops"], "5")
        self.assertEqual(values["--max-hops"], "5")
        self.assertEqual(values["--test-hops-diff"], "0")
        self.assertEqual(generator_output_name(5, 62471905), "5hop_preorder_seed62471905.json")

    def test_prompt_uses_same_questions_without_target_gold_leak(self):
        demos = self.legacy[:4]
        target = {**self.legacy[4], "answer": "TARGET_GOLD_SECRET", "chain_of_thought": ["TARGET_PROOF_SECRET"]}
        cot = build_prompt(target, demos, "cot")
        direct = build_prompt(target, demos, "direct")
        for prompt in (cot, direct):
            self.assertNotIn("TARGET_GOLD_SECRET", prompt)
            self.assertNotIn("TARGET_PROOF_SECRET", prompt)
            self.assertEqual(prompt.count("\nA:\n"), 5)
            self.assertTrue(prompt.endswith(target["text"] + "\nA:\n"))
            for demo in demos:
                self.assertIn(demo["text"], prompt)
        self.assertEqual(len(demonstration_proof(demos[0])), 4)
        demo_answer = cot.split("\nA:\n", 1)[1].split("\n\nQ:", 1)[0]
        self.assertNotIn("Each dumpus", demo_answer)
        direct_answers = [chunk.split("\n\nQ:", 1)[0] for chunk in direct.split("\nA:\n")[1:5]]
        self.assertEqual(direct_answers, ["Answer: True", "Answer: False", "Answer: True", "Answer: False"])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_prompt(demos[0], demos, "cot")

    def test_tokenizer_adds_plain_bos_once_and_native_roles_once(self):
        text = "Shared question body"
        plain_tokenizer = FakeTokenizer(native=False)
        plain = encode_prompt(plain_tokenizer, text, "plain", maximum_context=8192, output_reserve=2048)
        self.assertEqual(plain["input_ids"].count(1), 1)
        self.assertTrue(plain_tokenizer.calls[0][2]["add_special_tokens"])
        chat = FakeTokenizer()
        native = encode_prompt(chat, text, "native", maximum_context=8192, output_reserve=2048)
        self.assertEqual(native["prompt_text"], "<user>" + text + "</user><assistant>")
        self.assertEqual(chat.calls[0][1], [{"role": "user", "content": text}])
        self.assertEqual(native["prompt_body_hash"], plain["prompt_body_hash"])
        self.assertEqual(native["prompt_token_hash"], fingerprint(native["input_ids"]))
        self.assertEqual(native["prompt_text_hash"], hashlib.sha256(native["prompt_text"].encode()).hexdigest())
        with self.assertRaisesRegex(ValueError, "context limit"):
            encode_prompt(chat, text, "native", maximum_context=20, output_reserve=10)
        with self.assertRaisesRegex(ValueError, "no native chat template"):
            encode_prompt(plain_tokenizer, text, "native", maximum_context=8192, output_reserve=100)

    def test_tokenization_matrix_alias_and_identity(self):
        # A tiny fixed fixture tests the physical matrix without rerunning the generator.
        for model in self.config["models"]:
            model_path = self.root / "models" / model["stage"]
            model_path.mkdir(parents=True)
            for name in ("tokenizer_config.json", "tokenizer.json"):
                (model_path / name).write_text("{}\n")
            model["path"] = str(model_path)
        self.config["inference"]["max_context_tokens"] = 100_000
        atomic_json(self.root / "experiment.json", self.config)
        for phase in ("pilot", "formal"):
            atomic_jsonl(self.root / f"data/{phase}.jsonl", self.legacy[4:6])
        atomic_jsonl(self.root / "data/demonstrations.jsonl", self.legacy[:4])
        data_manifest = {"counts": {"formal": 2, "pilot": 2, "demonstrations": 4}}
        atomic_json(self.root / "manifests/data.json", data_manifest)
        with patch("tulu_behavior.tokenize.prepare", return_value=data_manifest), \
             patch("tulu_behavior.tokenize.tokenization_versions", return_value={"test": "fake"}), \
             patch("tulu_behavior.tokenize.load_native_tokenizer", side_effect=lambda p: FakeTokenizer(p.name != "base")):
            manifest = prepare_inputs(self.root)
            self.assertEqual(manifest["physical_record_count"], 56)
            self.assertEqual(len(manifest["files"]), 28)
            self.assertFalse((self.root / "data/tokenized/base/formal/cot/native.jsonl").exists())
            records = read_jsonl(self.root / "data/tokenized/sft/formal/cot/native.jsonl")
            self.assertEqual([r["sample_id"] for r in records], [x["sample_id"] for x in self.legacy[4:6]])
            self.assertEqual(records[0]["reserved_output_tokens"], 2048)
            self.assertEqual(records[0]["stage"], "sft")
            self.assertEqual(prepare_inputs(self.root), manifest)
            (self.root / "models/sft/tokenizer_config.json").write_text('{"changed":true}\n')
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                prepare_inputs(self.root)


if __name__ == "__main__":
    unittest.main()
