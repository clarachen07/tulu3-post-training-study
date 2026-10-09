import copy
import json
import tempfile
import unittest
from pathlib import Path

from ewt_eval.common import author_path, latest_results
from ewt_eval.run import repair_crash_tail
from ewt_eval.scoring import author_scores, author_task1, parse_tasks, strict_sentence

GOLD = [
    {"id": 1, "form": "I", "upos": "PRON", "head": 2, "deprel": "nsubj", "space_after": True},
    {"id": 2, "form": "work", "upos": "VERB", "head": 0, "deprel": "root", "space_after": False},
    {"id": 3, "form": ".", "upos": "PUNCT", "head": 2, "deprel": "punct", "space_after": True},
]


def output(gold=GOLD):
    tables = []
    for task in [1, 2, 3]:
        rows = []
        for g in gold:
            row = [g["id"], g["form"], g["upos"]]
            if task >= 2:
                row.append(g["head"])
            if task == 3:
                row.append(g["deprel"])
            rows.append("\t".join(map(str, row)))
        tables.append(f"- Task {task}\n\n" + "\n".join(rows))
    return "\n\n".join(tables)


class ScoringTests(unittest.TestCase):
    def test_perfect_and_punctuation(self):
        r = strict_sentence(GOLD, output())
        for k in ["task1_upos", "task2_uas", "task3_uas", "task3_las", "task3_las_subtypes"]:
            self.assertEqual(r["counts"][k], 3)
        self.assertTrue(r["diagnostics"]["tree"]["valid_tree"])
        self.assertEqual(strict_sentence(GOLD, output(), ignore_punct=True)["counts"]["gold"], 2)

    def test_independent_annotation_errors(self):
        bad = output().replace("1\tI\tPRON\t2\tnsubj", "1\tI\tNOUN\t2\tnsubj")
        c = strict_sentence(GOLD, bad)["counts"]
        self.assertEqual(c["task1_upos"], 3)
        self.assertEqual(c["task3_upos"], 2)
        self.assertEqual(c["task3_las"], 3)
        bad = output().replace("1\tI\tPRON\t2\tnsubj", "1\tI\tPRON\t0\tnsubj")
        c = strict_sentence(GOLD, bad)["counts"]
        self.assertEqual(c["task3_uas"], 2)
        self.assertEqual(c["task3_las"], 2)
        bad = output().replace("2\tnsubj", "2\tobj")
        c = strict_sentence(GOLD, bad)["counts"]
        self.assertEqual(c["task3_uas"], 3)
        self.assertEqual(c["task3_las"], 2)

    def test_subtypes(self):
        c = strict_sentence(GOLD, output().replace("2\tnsubj", "2\tnsubj:pass"))["counts"]
        self.assertEqual(c["task3_las"], 3)
        self.assertEqual(c["task3_las_subtypes"], 2)

    def test_partial_and_empty_fixed_denominator(self):
        partial = output().split("- Task 2")[0].replace("3\t.\tPUNCT", "")
        r = strict_sentence(GOLD, partial, "length")
        self.assertEqual(r["counts"]["gold"], 3)
        self.assertEqual(r["counts"]["task1_upos"], 2)
        self.assertTrue(r["diagnostics"]["truncated"])
        self.assertEqual(strict_sentence(GOLD, "")["counts"], {"gold": 3})
        self.assertEqual(author_task1(GOLD, partial)["content"], 0)

    def test_duplicate_extra_wrong_form(self):
        bad = output().replace("1\tI\tPRON\n", "1\tI\tPRON\n1\tI\tPRON\n", 1)
        self.assertEqual(strict_sentence(GOLD, bad)["counts"]["task1_upos"], 2)
        self.assertFalse(strict_sentence(GOLD, bad)["diagnostics"]["table_complete"]["1"])
        bad = output().replace("3\t.\tPUNCT\n\n- Task 2", "3\t.\tPUNCT\n9\tghost\tNOUN\n\n- Task 2", 1)
        self.assertEqual(strict_sentence(GOLD, bad)["counts"]["task1_upos"], 3)
        self.assertFalse(strict_sentence(GOLD, bad)["diagnostics"]["table_complete"]["1"])
        bad = output().replace("1\tI\tPRON", "1\tYou\tPRON", 1)
        self.assertEqual(strict_sentence(GOLD, bad)["counts"]["task1_upos"], 2)

    def test_invalid_heads_not_repaired(self):
        for head in ["-1", "99", "x", "1"]:
            c = strict_sentence(GOLD, output().replace("1\tI\tPRON\t2\tnsubj", f"1\tI\tPRON\t{head}\tnsubj"))["counts"]
            self.assertEqual(c["task3_uas"], 2)

    def test_roots_and_cycles(self):
        bad = output().replace("1\tI\tPRON\t2\tnsubj", "1\tI\tPRON\t0\tnsubj")
        self.assertTrue(strict_sentence(GOLD, bad)["diagnostics"]["tree"]["multiple_roots"])
        bad = output().replace("2\twork\tVERB\t0\troot", "2\twork\tVERB\t1\troot")
        self.assertTrue(strict_sentence(GOLD, bad)["diagnostics"]["tree"]["cycle"])

    def test_whitespace_policy(self):
        good = output().replace("1\tI\t", "1\tI \t")
        self.assertEqual(strict_sentence(GOLD, good)["counts"]["task3_las"], 3)
        bad = output().replace("2\twork\t", "2\twork \t")
        self.assertEqual(strict_sentence(GOLD, bad)["counts"]["task1_upos"], 2)

    def test_unmarked_and_ambiguous(self):
        unmarked = output()
        for n in [1, 2, 3]:
            unmarked = unmarked.replace(f"- Task {n}\n", "")
        self.assertEqual(strict_sentence(GOLD, unmarked)["counts"]["task3_las"], 3)
        duplicate = output() + "\n\n" + output().split("- Task 2")[0]
        self.assertEqual(parse_tasks(duplicate)[1]["failure"], "ambiguous_task")

    def test_author_parity_and_recovery(self):
        author_path()
        from llmpp.eval import eval as vendor
        sentence = {"sent_id": "toy", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "gold_output": output(), "gold": GOLD}
        for content in [output(), "", output().replace("2\tnsubj", "99\tnsubj"), output().replace("0\troot", "0\tobj")]:
            native = vendor([sentence["messages"] + [{"role": "assistant", "gold": output(), "content": content}]], 0, False, None)
            wrapped = author_scores([sentence], [content])["with_punct"]
            self.assertEqual(native["token"], wrapped["token"])
        self.assertEqual(author_task1(GOLD, output())["correct"], 3)

    def test_resume_crash_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.jsonl"
            p.write_text(json.dumps({"sent_id": "a", "status": "ok"}) + '\n{"sent_id":')
            repair_crash_tail(p)
            self.assertEqual(list(latest_results(p)), ["a"])
            self.assertTrue(list(Path(d).glob("*.crash-tail-*")))
            with p.open("a") as f:
                f.write(json.dumps({"sent_id": "a", "status": "ok"}) + "\n")
            with self.assertRaises(ValueError):
                latest_results(p)


if __name__ == "__main__":
    unittest.main()
