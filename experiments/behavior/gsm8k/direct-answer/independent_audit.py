"""Independent reconstruction: intentionally does not import production scoring/I/O."""
import collections
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def numerical_value(text):
    cleaned = "".join(char for index, char in enumerate(text)
                      if not (char == "," and 0 < index < len(text) - 1
                              and text[index - 1].isdecimal() and text[index + 1].isdecimal()))
    matches = re.findall(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)", cleaned)
    return Decimal(matches[-1]) if matches else None


def format_valid(text):
    text = text.strip()
    prefix = re.match(r"(?:Answer\s*:|Final\s+answer\s*:|The\s+answer\s+is)\s*", text, re.I)
    if prefix:
        text = text[prefix.end():]
    if text.endswith('.'):
        text = text[:-1].rstrip()
    for left, right in [('$$', '$$'), ('$', '$'), (r'\(', r'\)'), (r'\[', r'\]')]:
        if text.startswith(left):
            if not text.endswith(right) or len(text) < len(left) + len(right):
                return False
            text = text[len(left):-len(right)]
            break
    if text.startswith(r'\boxed{') and text.endswith('}'):
        text = text[7:-1].strip()
    if text[:1] in ['+', '-']:
        text = text[1:]
    if not text:
        return False
    pieces = text.split('.')
    if len(pieces) > 2 or (len(pieces) == 2 and not pieces[1].isdecimal()):
        return False
    integer = pieces[0]
    if not integer:
        return len(pieces) == 2
    groups = integer.split(',')
    return all(g.isdecimal() for g in groups) and (len(groups) == 1 or
        (1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])))


def main():
    formal = {r["id"]: r for r in read(ROOT / "data/prepared/formal.jsonl")}
    assert collections.Counter(r["split"] for r in formal.values()) == {"train": 1024, "test": 1319}
    pilot_ids = {r["id"] for r in read(ROOT / "data/prepared/pilot.jsonl")}
    assert len(pilot_ids) == 32 and not pilot_ids & formal.keys()
    registry = json.loads((ROOT / "experiment.json").read_text())
    stats = json.loads((ROOT / "reports/statistics.json").read_text())
    frozen = json.loads((ROOT / "manifests/frozen_protocol.json").read_text())
    for relative_path, expected_hash in frozen["files"].items():
        assert sha256(ROOT / relative_path) == expected_hash, relative_path
    config_hash = object_sha256(registry)
    assets = json.loads((ROOT / "manifests/assets.json").read_text())
    published = json.loads((ROOT / "manifests/published_identity.json").read_text())
    assert published["status"] == "passed"
    verified_shards = 0
    for model in registry["models"]:
        actual = next(m for m in assets["models"] if m["registry"]["stage"] == model["stage"])
        identity = next(m for m in published["models"] if m["stage"] == model["stage"])
        assert actual["registry"] == model and identity["revision"] == model["revision"]
        weights = [f["sha256"] for f in actual["files"] if f["path"].endswith(".safetensors")]
        assert sorted(weights) == sorted(identity["published_sha256"])
        verified_shards += len(weights)
    assert verified_shards == 16
    stages = [m["stage"] for m in registry["models"]]
    panels = {}
    diagnostics = {}
    raw_hashes = {}
    checked = 0
    native_prompt_hashes = None
    for stage in stages:
        prompts = {r["id"]: r for r in read(ROOT / f"data/tokenized/{stage}/formal.jsonl")}
        assert set(prompts) == set(formal)
        if stage != "base":
            hashes = {i: p["prompt_token_hash"] for i, p in prompts.items()}
            if native_prompt_hashes is None:
                native_prompt_hashes = hashes
            else:
                assert hashes == native_prompt_hashes
        for ident, prompt in prompts.items():
            content = formal[ident]["content"]
            manual = content + "\nAnswer:" if stage == "base" else "<|user|>\n" + content + "\n<|assistant|>\n"
            assert prompt["prompt_text"] == manual
            canonical = json.dumps(prompt["prompt_token_ids"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            assert hashlib.sha256(canonical).hexdigest() == prompt["prompt_token_hash"]
        panel = {}
        raw_hashes[stage] = {}
        diag = {}
        for file in (ROOT / f"outputs/formal/{stage}").glob("batch_*.jsonl"):
            assert sha256(file) == stats["sources"][str(file.relative_to(ROOT))]
            for r in read(file):
                ident = r["id"]
                assert ident not in panel
                assert r["stage"] == stage and r["phase"] == "formal"
                assert r["split"] == formal[ident]["split"]
                assert r["config_hash"] == config_hash
                assert r["record_hash"] == object_sha256({k: v for k, v in r.items() if k != "record_hash"})
                assert r["gold"] == formal[ident]["gold"]
                assert r["prompt_token_hash"] == prompts[ident]["prompt_token_hash"]
                val = numerical_value(r["continuation"])
                gold = Decimal(str(r["gold"]).replace(",", ""))
                correct = val is not None and val == gold
                assert correct == r["correct"], ident
                assert (None if r["extracted_answer"] is None else Decimal(r["extracted_answer"])) == val
                assert r["truncated"] == (r["finish_reason"] == "length")
                assert r["output_tokens"] == len(r["output_token_ids"])
                raw_hashes[stage][ident] = r["record_hash"]
                panel[ident] = bool(correct)
                assert format_valid(r["continuation"]) == r["answer_only"]
                diag[ident] = (val is None, r["truncated"], len(r["output_token_ids"]), format_valid(r["continuation"]))
                checked += 1
        assert set(panel) == set(formal)
        panels[stage] = panel
        diagnostics[stage] = diag
    for s in stats["summary"]:
        ids = [i for i, r in formal.items() if r["split"] == s["split"]]
        total = sum(panels[s["stage"]][i] for i in ids)
        assert total == s["correct"] and len(ids) == s["n"]
        assert abs(total / len(ids) - s["accuracy"]) < 1e-15
        diag = [diagnostics[s["stage"]][i] for i in ids]
        assert sum(d[0] for d in diag) == s["no_answer"]
        assert sum(d[1] for d in diag) == s["truncated"]
        assert abs(sum(d[0] for d in diag) / len(ids) - s["no_answer_rate"]) < 1e-15
        assert abs(sum(d[1] for d in diag) / len(ids) - s["truncation_rate"]) < 1e-15
        assert abs(sum(d[2] for d in diag) / len(ids) - s["mean_output_tokens"]) < 1e-12
        assert sum(d[3] for d in diag) == s['answer_only']
        assert sum(d[3] for d in diag) / len(ids) == s['answer_only_rate']
        assert np.median([d[2] for d in diag]) == s['median_output_tokens']
    comparison = json.loads((ROOT / 'reports/comparison.json').read_text())
    baseline_manifest = json.loads((ROOT / 'manifests/baseline.json').read_text())
    for filename, expected_hash in baseline_manifest['files'].items():
        assert sha256(ROOT / filename) == expected_hash
    old_panels = {}
    for stage in stages:
        old_panels[stage] = {}
        for file in (ROOT / f'baseline/outputs/formal/{stage}').glob('batch_*.jsonl'):
            assert sha256(file) == comparison['old_sources'][str(file.relative_to(ROOT))]
            for row in read(file):
                assert row['id'] not in old_panels[stage]
                assert row['gold'] == formal[row['id']]['gold']
                val = numerical_value(row['continuation'])
                assert (val is not None and val == Decimal(str(row['gold']).replace(',', ''))) == row['correct']
                old_panels[stage][row['id']] = row
        assert set(old_panels[stage]) == set(formal)
    assert comparison['new_sources'] == stats['sources']
    for c in comparison['pairs']:
        ids = sorted(i for i, r in formal.items() if r['split'] == c['split'])
        stage = c['stage']
        counts = collections.Counter((old_panels[stage][i]['correct'], panels[stage][i]) for i in ids)
        for key, cell in [('wrong_to_correct', (False, True)), ('correct_to_wrong', (True, False)),
                          ('both_correct', (True, True)), ('both_wrong', (False, False))]:
            assert counts[cell] == c[key]
        assert c['n'] == len(ids)
        for label, corrects, diags in [
            ('old', [old_panels[stage][i]['correct'] for i in ids],
             [(numerical_value(old_panels[stage][i]['continuation']) is None,
               old_panels[stage][i]['finish_reason'] == 'length', len(old_panels[stage][i]['output_token_ids']),
               format_valid(old_panels[stage][i]['continuation'])) for i in ids]),
            ('new', [panels[stage][i] for i in ids], [diagnostics[stage][i] for i in ids])]:
            assert sum(corrects) == c[label + '_correct']
            assert sum(corrects) / len(ids) == c[label + '_accuracy']
            for index, metric in [(0, 'no_answer_rate'), (1, 'truncation_rate'), (3, 'answer_only_rate')]:
                assert sum(d[index] for d in diags) / len(ids) == c[label + '_' + metric]
            assert abs(np.mean([d[2] for d in diags]) - c[label + '_mean_output_tokens']) < 1e-12
            assert np.median([d[2] for d in diags]) == c[label + '_median_output_tokens']
        assert abs((c['new_correct'] - c['old_correct']) / len(ids) - c['delta']) < 1e-15
    examples = json.loads((ROOT / 'reports/comparison_examples.json').read_text())
    expected_cases = []
    for split in ['train', 'test']:
        ids = sorted(i for i, r in formal.items() if r['split'] == split)
        for stage in stages:
            for direction, before, after in [('wrong_to_correct', False, True), ('correct_to_wrong', True, False)]:
                selected = [i for i in ids if old_panels[stage][i]['correct'] == before and panels[stage][i] == after][:2]
                expected_cases.extend((split, stage, direction, i) for i in selected)
    assert [(c['split'], c['stage'], c['direction'], c['id']) for c in examples] == expected_cases
    for c in examples:
        assert c['old'] == old_panels[c['stage']][c['id']]
        assert c['new']['record_hash'] == raw_hashes[c['stage']][c['id']]
        assert object_sha256({k: v for k, v in c['new'].items() if k != 'record_hash'}) == c['new']['record_hash']
        assert c['question'] == formal[c['id']]['question'] and c['gold'] == formal[c['id']]['gold']
    for t in comparison['timing']:
        for label, root in [('old', ROOT / 'baseline'), ('new', ROOT)]:
            jobs = [json.loads(p.read_text()) for p in (root / 'runs').glob('*.json')]
            jobs = [j for j in jobs if j['phase'] == 'formal' and j['stage'] == t['stage']]
            metas = [json.loads(p.read_text()) for p in (root / 'outputs/formal' / t['stage']).glob('*.meta.json')]
            assert abs(sum(j['elapsed_seconds'] for j in jobs) - t[label + '_process_seconds']) < 1e-8
            assert abs(sum(m['generation_seconds'] for m in metas) - t[label + '_generation_seconds']) < 1e-8
            assert sum(m['output_tokens'] for m in metas) == t[label + '_output_tokens']
            raw_total = (sum(d[2] for d in diagnostics[t['stage']].values()) if label == 'new' else
                         sum(len(r['output_token_ids']) for r in old_panels[t['stage']].values()))
            assert raw_total == t[label + '_output_tokens']
            if label == 'new':
                saved_timing = next(x for x in stats['timing'] if x['stage'] == t['stage'])
                assert saved_timing['failed_attempts'] == sum(j['status'] == 'failed' for j in jobs)
                assert saved_timing['output_tokens'] == raw_total
                assert saved_timing['allocated_process_seconds'] == t['new_process_seconds']
                assert saved_timing['committed_generation_seconds'] == t['new_generation_seconds']
    for p in stats["pairs"]:
        counts = collections.Counter((panels[p['from_stage']][i], panels[p['to_stage']][i])
                                     for i, r in formal.items() if r["split"] == p["split"])
        for key, cell in [("wrong_to_correct", (False, True)), ("correct_to_wrong", (True, False)),
                          ("both_correct", (True, True)), ("both_wrong", (False, False))]:
            assert counts[cell] == p[key]
        assert abs((counts[(False, True)] - counts[(True, False)]) / p["n"] - p["delta"]) < 1e-15
    assert checked == 9372
    max_interval_difference = 0.0
    for split_index, split in enumerate(["train", "test"]):
        ids = sorted(i for i, r in formal.items() if r["split"] == split)
        values = np.array([[panels[s][i] for s in stages] for i in ids], dtype=np.uint8)
        rng = np.random.default_rng(registry["bootstrap_seed"] + split_index)
        # One full draw and integer counts, independently of the production chunked means.
        draws = rng.integers(0, len(ids), size=(registry["bootstrap_replicates"], len(ids)))
        counts = values[draws].sum(axis=1, dtype=np.int64)
        bounds = np.percentile(counts / len(ids), [2.5, 97.5], axis=0)
        differences = np.percentile((counts[:, 1:] - counts[:, :-1]) / len(ids), [2.5, 97.5], axis=0)
        old_values = np.array([[old_panels[s][i]['correct'] for s in stages] for i in ids], dtype=np.uint8)
        old_counts = old_values[draws].sum(axis=1, dtype=np.int64)
        protocol_bounds = np.percentile((counts - old_counts) / len(ids), [2.5, 97.5], axis=0)
        for j, stage in enumerate(stages):
            c = next(c for c in comparison['pairs'] if c['split'] == split and c['stage'] == stage)
            max_interval_difference = max(max_interval_difference,
                abs(c['ci_low'] - protocol_bounds[0, j]), abs(c['ci_high'] - protocol_bounds[1, j]))
            saved = next(s for s in stats["summary"] if s["split"] == split and s["stage"] == stage)
            discrepancy = max(abs(saved["ci_low"] - bounds[0, j]), abs(saved["ci_high"] - bounds[1, j]))
            max_interval_difference = max(max_interval_difference, float(discrepancy))
        for j in range(3):
            saved = next(p for p in stats["pairs"] if p["split"] == split and p["from_stage"] == stages[j])
            discrepancy = max(abs(saved["ci_low"] - differences[0, j]), abs(saved["ci_high"] - differences[1, j]))
            max_interval_difference = max(max_interval_difference, float(discrepancy))
    assert max_interval_difference < 1e-12
    target = ROOT / "reports/independent_audit.json"
    result = {"status": "passed", "checked_records": checked, "stage_split_cells": 8, "baseline_checked_records": 9372, "protocol_contrasts": 8, "deterministic_comparison_cases_checked": len(examples),
              "comparison_sha256": sha256(ROOT / "reports/comparison.json"),
              "frozen_files_verified": len(frozen["files"]),
              "published_weight_shards_verified": verified_shards,
              "formal_native_prompts_identical": True,
              "paired_contrasts": 6, "score_reconstruction": "separate regex/Decimal implementation; no production imports",
              "statistics_sha256": hashlib.sha256((ROOT / "reports/statistics.json").read_bytes()).hexdigest(),
              "bootstrap_replicates_per_split": registry["bootstrap_replicates"],
              "maximum_interval_difference": max_interval_difference,
              "scope": "frozen file hashes, raw output hashes, record/config fingerprints, raw answer scoring, completeness, manual prompt reconstruction, token hash, counts, paired changes and full bootstrap intervals independently rerun"}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
