from __future__ import annotations

import copy
import json
from pathlib import Path
import platform
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tulu_id.common import ROOT, atomic_json, frozen_identity, load_config, read_jsonl, sha256_file
from tulu_id.prepare import (
    generator_arguments, generator_output_name, prepare, sampling_indices,
)
from tulu_id.tokenize import encode_item, tokenize


class FakeTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    model_max_length = 100_000
    special_tokens_map = {"bos_token": "<BOS>", "eos_token": "<EOS>"}
    is_fast = True

    def __init__(self, native=True):
        self.chat_template = "native-template" if native else None
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(("plain", text, kwargs))
        ids = [self.bos_token_id] + [ord(character) + 10 for character in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.calls.append(("chat", messages, tokenize, add_generation_prompt))
        rendered = "<user>\n" + messages[0]["content"] + "\n<assistant>\n"
        return [ord(character) + 10 for character in rendered] if tokenize else rendered

    def get_chat_template(self):
        return self.chat_template

    def convert_ids_to_tokens(self, token):
        return f"token-{token}"

    def decode(self, ids, **kwargs):
        return "".join("<BOS>" if token == 1 else chr(token - 10) for token in ids)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tulu-id-data-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copytree(ROOT / "vendor/prontoqa", self.root / "vendor/prontoqa")
        self.config = copy.deepcopy(load_config(ROOT))
        self.config["data"]["sample_count"] = 6
        self.config["statistics"].update(subsample_repeats=3, subsample_size=5)
        for model in self.config["models"]:
            path = self.root / "models" / model["stage"]
            path.mkdir(parents=True)
            (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            model["path"] = str(path)
        atomic_json(self.root / "experiment.json", self.config)

    def test_original_zero_shot_generator_and_idempotence(self):
        manifest = prepare(self.root)
        items = read_jsonl(self.root / "data/items.jsonl")
        self.assertEqual(len(items), 6)
        self.assertEqual(len({item["text"] for item in items}), 6)
        self.assertEqual([item["sample_id"] for item in items], [f"q{i:06d}" for i in range(1, 7)])
        for item in items:
            self.assertEqual(item["text"], item["question"] + " " + item["query"])
            self.assertEqual(item["hop"], 3)
            self.assertIn(item["answer"], ("True", "False"))
        raw = json.loads((self.root / "data/raw/seed_62471893/3hop_preorder.json").read_text())
        self.assertTrue(all(set(example) == {"test_example"} for example in raw.values()))
        self.assertEqual(manifest["hop_validation"]["internal_num_deduction_steps"], 4)
        before = sha256_file(self.root / "manifests/data.json")
        self.assertEqual(prepare(self.root), manifest)
        self.assertEqual(sha256_file(self.root / "manifests/data.json"), before)
        with (self.root / "data/items.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(RuntimeError, "changed recorded artifact"):
            prepare(self.root)

    def test_seed_increment_fills_only_exact_duplicates(self):
        self.config["data"]["sample_count"] = 4
        self.config["statistics"]["subsample_size"] = 3
        atomic_json(self.root / "experiment.json", self.config)
        seeds_and_counts = []

        def generated(root, data, seed, count, versions, sources):
            seeds_and_counts.append((seed, count))
            names = ["A", "A", "B", "B"] if len(seeds_and_counts) == 1 else ["C", "D"]
            self.assertEqual(len(names), count)
            batch = {f"example{i + 1}": {"test_example": {
                "question": f"{name} is a wumpus.",
                "query": f"True or false: {name} is blue.", "answer": "True",
                "chain_of_thought": [f"{name} is a wumpus.", f"{name} is blue."],
            }} for i, name in enumerate(names)}
            directory = root / "data/raw" / f"seed_{seed}"
            directory.mkdir(parents=True)
            atomic_json(directory / "fixture.json", batch)
            return batch, {"seed": seed, "requested_count": count}

        with patch("tulu_id.prepare.run_generator", side_effect=generated):
            manifest = prepare(self.root)
        self.assertEqual(seeds_and_counts, [(62471893, 4), (62471894, 2)])
        self.assertEqual(manifest["exact_text_duplicates_rejected"], 2)
        items = read_jsonl(self.root / "data/items.jsonl")
        self.assertEqual([item["source_seed"] for item in items], [62471893, 62471893, 62471894, 62471894])
        self.assertEqual([item["source_example"] for item in items], ["example1", "example3", "example1", "example2"])

    def test_sampling_reproducibility_and_independent_rngs(self):
        statistics = self.config["statistics"]
        subsamples, permutations = sampling_indices(6, statistics)
        self.assertEqual(subsamples.shape, (3, 5))
        self.assertEqual(permutations.shape, (3, 6))
        np.testing.assert_array_equal(subsamples[0], [2, 0, 1, 3, 5])
        np.testing.assert_array_equal(permutations[0], [1, 2, 0, 3, 5, 4])
        for row in subsamples:
            self.assertEqual(len(set(row.tolist())), 5)
        for row in permutations:
            self.assertEqual(sorted(row.tolist()), list(range(6)))
        changed = {**statistics, "subsample_seed": statistics["subsample_seed"] + 1}
        _, unchanged_scales = sampling_indices(6, changed)
        np.testing.assert_array_equal(permutations, unchanged_scales)

    def test_arguments_and_nondefault_seed_filename(self):
        arguments = generator_arguments(self.config["data"], 62471894, 7)
        pairs = dict(zip(arguments[::2], arguments[1::2]))
        self.assertEqual(pairs["--model-name"], "json")
        self.assertEqual(pairs["--few-shot-examples"], "0")
        self.assertEqual(pairs["--min-hops"], "3")
        self.assertEqual(pairs["--max-hops"], "3")
        self.assertEqual(pairs["--test-hops-diff"], "0")
        self.assertNotIn("--proofs-only", arguments)
        self.assertEqual(generator_output_name(self.config["data"], 62471894), "3hop_preorder_seed62471894.json")

    def test_frozen_data_reuse_and_configuration_change_rejection(self):
        manifest = prepare(self.root)
        files = {"experiment.json": sha256_file(self.root / "experiment.json")}
        environment = {"python": platform.python_version(), "packages": {}}
        atomic_json(self.root / "manifests/frozen_protocol.json", {
            "files": files, "environment": environment,
            "protocol_digest": frozen_identity(files, environment),
        })
        self.assertEqual(prepare(self.root), manifest)
        self.config["data"]["seed"] += 1
        atomic_json(self.root / "experiment.json", self.config)
        with self.assertRaisesRegex(RuntimeError, "Frozen input/code changed"):
            prepare(self.root)

    def test_tokenization_protocol_no_answer_or_duplicate_bos(self):
        item = {"sample_id": "q000001", "question": "Fae is a wumpus.",
                "query": "True or false: Fae is blue.", "answer": "SECRET_ANSWER",
                "chain_of_thought": ["SECRET_PROOF"]}
        item["text"] = item["question"] + " " + item["query"]
        base = FakeTokenizer(native=False)
        plain = encode_item(base, item, "plain")
        self.assertEqual(plain["input_ids"][0], base.bos_token_id)
        self.assertTrue(base.calls[0][2]["add_special_tokens"])
        chat = FakeTokenizer()
        native = encode_item(chat, item, "native")
        self.assertEqual(chat.calls, [("chat", [{"role": "user", "content": item["text"]}], True, True)])
        self.assertNotIn(chat.bos_token_id, native["input_ids"])
        self.assertEqual(native["readout_index"], len(native["input_ids"]) - 1)
        with self.assertRaisesRegex(ValueError, "no native chat template"):
            encode_item(base, item, "native")
        leaked = {**item, "text": item["text"] + " SECRET_ANSWER"}
        with self.assertRaisesRegex(ValueError, "unexpected content"):
            encode_item(chat, leaked, "native")

    def test_tokenization_manifest_integrity_and_frozen_reuse(self):
        prepare(self.root)
        with patch("tulu_id.tokenize.dependency_versions", return_value={"test": "fake-tokenizer"}), \
             patch("tulu_id.tokenize.load_native_tokenizer", side_effect=lambda path: FakeTokenizer(path.name != "base")):
            manifest = tokenize(self.root)
            self.assertEqual(set(manifest["stages"]), {"base", "sft", "dpo", "rlvr"})
            for stage in manifest["stages"]:
                records = read_jsonl(self.root / "data/tokenized" / f"{stage}.jsonl")
                self.assertEqual(len(records), 6)
                self.assertEqual([r["sample_id"] for r in records], [f"q{i:06d}" for i in range(1, 7)])
            files = {}
            environment = {"python": platform.python_version(), "packages": {}}
            atomic_json(self.root / "manifests/frozen_protocol.json", {
                "files": files, "environment": environment,
                "protocol_digest": frozen_identity(files, environment),
            })
            before = sha256_file(self.root / "manifests/tokenization.json")
            self.assertEqual(tokenize(self.root), manifest)
            self.assertEqual(before, sha256_file(self.root / "manifests/tokenization.json"))
            path = Path(self.config["models"][0]["path"]) / "tokenizer_config.json"
            path.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different source/config/tokenizers/environment"):
                tokenize(self.root)


if __name__ == "__main__":
    unittest.main()
