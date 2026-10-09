import json
import unittest

from ewt_eval.common import ROOT, author_path, read_jsonl
from ewt_eval.prepare import build_messages


@unittest.skipUnless((ROOT / "data/test.jsonl").exists(), "data not prepared")
class ProtocolTests(unittest.TestCase):
    def test_dataset_and_no_target_answer_leakage(self):
        rows = read_jsonl(ROOT / "data/test.jsonl")
        self.assertEqual(len(rows), 2077)
        self.assertEqual(sum(len(s["gold"]) for s in rows), 25094)
        self.assertEqual(len({s["sent_id"] for s in rows}), 2077)
        for s in rows:
            self.assertEqual([m["role"] for m in build_messages(s, "P0")], ["system", "user"])
            for candidate in ["P0", "P1", "P2", "P3"]:
                messages = build_messages(s, candidate)
                self.assertEqual(messages[-1]["role"], "user")
                self.assertNotIn(s["gold_output"], "\n".join(m["content"] for m in messages))

    def test_exact_author_prompt_conversion(self):
        # Compare against the actual author's CLI, not another local renderer.
        import subprocess
        import sys
        author_path()
        subprocess.run([sys.executable, "-m", "llmpp.conllu_to_prompt", str(ROOT / "third_party/llmpp/templates/en-turn1-step3.toml"), str(ROOT / "data/en_ewt-ud-test.conllu"), "LANGUAGE", "English"], cwd=ROOT / "third_party/llmpp", check=True)
        author = read_jsonl(ROOT / "data/en-turn1-step3.test.jsonl")
        own = read_jsonl(ROOT / "data/test.jsonl")
        self.assertEqual(len(author), len(own))
        for a, s in zip(author, own):
            self.assertEqual(a["messages"][:2], s["messages"])
            self.assertEqual(a["messages"][2]["content"], s["gold_output"])

    def test_dev_and_demo_split_separation(self):
        dev = read_jsonl(ROOT / "data/dev128.jsonl")
        self.assertEqual(len(dev), 128)
        self.assertTrue(all(s["split"] == "dev" for s in dev))
        demo = json.loads((ROOT / "data/example.json").read_text())
        self.assertEqual(demo["split"], "train")
        self.assertTrue(6 <= len(demo["gold"]) <= 12)


if __name__ == "__main__":
    unittest.main()
