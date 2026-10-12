import argparse
import collections
import json

from .common import ROOT, latest_results, read_jsonl, write_json, write_jsonl
from .scoring import author_scores, author_task1, parse_tasks, strict_sentence, summarize_strict


def score_unit(split, candidate, stage):
    sentences = read_jsonl(ROOT / "data" / f"{split}.jsonl")
    path = ROOT / "outputs" / split / candidate / f"{stage}.jsonl"
    results = latest_results(path)
    assert set(results) == {s["sent_id"] for s in sentences}, "unit is incomplete or has unexpected sentences"
    assert all(r["status"] == "ok" for r in results.values())
    outputs = [results[s["sent_id"]]["output"] for s in sentences]
    per_sentence = []
    summary = {"stage": stage, "candidate": candidate, "split": split, "complete": True}
    for ignore in [False, True]:
        key = "no_punct" if ignore else "with_punct"
        rows = [strict_sentence(s["gold"], o, results[s["sent_id"]]["finish_reason"], ignore) for s, o in zip(sentences, outputs)]
        summary[key] = summarize_strict(rows)
        c = collections.Counter()
        for s, o in zip(sentences, outputs):
            item = author_task1(s["gold"], o, ignore)
            c.update(item)
        summary[key]["task1_author_extension"] = {"counts": dict(c), "f1": 2 * c["correct"] / (c["gold"] + c["content"]) if c["gold"] + c["content"] else 0}
        if not ignore:
            for s, output, row in zip(sentences, outputs, rows):
                per_sentence.append({"sent_id": s["sent_id"], "gold": s["gold"], "parsed_tasks": parse_tasks(output), **row})
    summary["author"] = author_scores(sentences, outputs)
    report = ROOT / "reports" / split / candidate
    write_json(report / f"{stage}.json", summary)
    write_jsonl(report / f"{stage}.sentences.jsonl", per_sentence)
    print(json.dumps({"stage": stage, "candidate": candidate, "metrics": summary["with_punct"]["metrics"], "author": summary["author"]["with_punct"]["metrics"]}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--candidate", default="P0")
    p.add_argument("--stage", required=True)
    args = p.parse_args()
    score_unit(args.split, args.candidate, args.stage)


if __name__ == "__main__":
    main()
