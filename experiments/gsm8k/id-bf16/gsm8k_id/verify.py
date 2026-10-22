"""Independent scoring, token-position, spectrum and selected numerical audits."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from .common import configuration, ROOT, STAGES, digest, read, rows, sha, utc, validate, write


def numerical_value(text):
    # Independent of the production scorer and its comma substitution.
    cleaned = "".join(c for j, c in enumerate(text) if not (
        c == "," and 0 < j < len(text) - 1 and text[j - 1].isdecimal() and text[j + 1].isdecimal()))
    matches = re.findall(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)", cleaned)
    return Decimal(matches[-1]) if matches else None


def independent_twonn(x):
    d = cdist(np.asarray(x, np.float64), np.asarray(x, np.float64))
    d.sort(axis=1)
    r1, r2 = d[:, 1], d[:, 2]
    good = (r1 > 0) & (r2 != r1)
    mu = np.sort(r2[good] / r1[good])
    n = len(mu)
    count = min(int(n * .9), n - 2)
    if count < 2:
        return None
    a = np.log(mu[:count])
    b = -np.log1p(-np.arange(1, count + 1) / n)
    return float(np.dot(a, b) / np.dot(a, a))


def verify(root, *, deep=True):
    root = Path(root)
    frozen = validate(root)
    cfg = configuration(root)
    assert cfg["inference"]["dtype"] == "bfloat16" and cfg["inference"]["device"] == "cuda:0"
    assert cfg["storage_dtype"] == "float32" and cfg["analysis_dtype"] == "float64"
    manifest = read(root / "reports/analysis_manifest.json")
    sources = read(root / "reports/analysis_sources.json")
    assert digest({k: v for k, v in sources.items() if k != "identity"}) == sources["identity"]
    assert manifest["source_identity"] == sources["identity"]
    for receipt in sources["source_receipts"]:
        assert sha(root / receipt["path"]) == receipt["sha256"]
    for rel, expected in manifest["reports"].items():
        assert sha(root / rel) == expected
    behavior = {r["stage"]: r for r in read(root / "reports/statistics.json")["behavior"]}
    checked, coverage = 0, {}
    generated_by_stage = {}
    from transformers import AutoTokenizer
    for stage in STAGES:
        m = next(m for m in cfg["models"] if m["stage"] == stage)
        tok = AutoTokenizer.from_pretrained(m["path"], local_files_only=True, trust_remote_code=False)
        inputs = rows(root / f"data/tokenized/{stage}/formal.jsonl")
        counts = Counter()
        generated_by_stage[stage] = {}
        lengths = []
        assert len(inputs) == 1319
        for source in inputs:
            ident = source["id"]
            r = read(root / f"outputs/formal/{stage}/{ident}.json")
            generated_by_stage[stage][ident] = r
            assert r["identity"] == frozen["identity"] and r["stage"] == stage and r["phase"] == "formal"
            assert (r.get("forward_dtype"), r.get("storage_dtype"), r.get("analysis_dtype")) == ("bfloat16", "float32", "float64")
            assert r["id"] == ident and r["gold"] == source["gold"] and r["prompt_token_ids"] == source["prompt_token_ids"]
            assert r["record_hash"] == digest({k: v for k, v in r.items() if k != "record_hash"})
            v = numerical_value(r["continuation"])
            correct = v is not None and v == Decimal(str(source["gold"]).replace(",", ""))
            assert correct == r["correct"]
            assert v == (None if r["extracted_answer"] is None else Decimal(r["extracted_answer"]))
            sampled = r["output_token_ids"]
            assert len(sampled) == r["output_tokens"] and 1 <= len(sampled) <= 2048
            eos = int(json.loads((Path(m["path"]) / "config.json").read_text())["eos_token_id"])
            content_ids = sampled[:-1] if sampled[-1] == eos else sampled
            decoded = tok.decode(content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            boundary = decoded.find(cfg["inference"]["stop"])
            assert r["continuation"] == (decoded if boundary < 0 else decoded[:boundary])
            assert (r["finish_reason"] == "eos") == (sampled[-1] == eos)
            assert (r["finish_reason"] == "stop_string") == (boundary >= 0)
            assert r["truncated"] == (r["finish_reason"] == "length")
            if r["truncated"]:
                assert len(sampled) == cfg["inference"]["max_new_tokens"]
            effective = r["effective_token_ids"]
            assert effective == content_ids[:len(effective)]
            assert r["effective_output_tokens"] == len(effective)
            if boundary < 0:
                assert effective == content_ids
            else:
                valid_lengths = [j for j in range(len(content_ids) + 1) if decoded.startswith(tok.decode(content_ids[:j], skip_special_tokens=False, clean_up_tokenization_spaces=False)) and len(tok.decode(content_ids[:j], skip_special_tokens=False, clean_up_tokenization_spaces=False)) <= boundary]
                assert len(effective) == max(valid_lengths)
            meta = read(root / f"activations/formal/{stage}/{ident}.json")
            assert (meta.get("forward_dtype"), meta.get("storage_dtype"), meta.get("analysis_dtype")) == ("bfloat16", "float32", "float64")
            pos = [len(source["prompt_token_ids"]) - 1]
            pos += [len(source["prompt_token_ids"]) + int(np.ceil(p * len(effective))) - 1 if effective else None for p in (.25, .5, .75, 1)]
            assert pos == meta["positions"] and meta["record_hash"] == r["record_hash"]
            path = root / f"activations/formal/{stage}/{ident}.npz"
            assert sha(path) == meta["sha256"]
            with np.load(path, allow_pickle=False) as z:
                assert z["hidden"].shape == (len(meta["anchors"]), 32, 4096)
                assert z["hidden"].dtype == np.float32 and np.isfinite(z["hidden"]).all()
            counts["correct"] += correct
            counts["truncated"] += r["truncated"]
            counts["empty_output"] += not effective
            lengths.append(len(sampled))
            checked += 1
        for key in ("correct", "truncated", "empty_output"):
            assert counts[key] == behavior[stage][key]
        assert behavior[stage]["accuracy"] == counts["correct"] / len(inputs)
        np.testing.assert_allclose(behavior[stage]["mean_output_tokens"], np.mean(lengths), atol=1e-12)
        coverage[stage] = dict(counts)
    all_ids = sorted(generated_by_stage[STAGES[0]])
    nonempty = [i for i in all_ids if all(generated_by_stage[s][i]["effective_output_tokens"] > 0 for s in STAGES)]
    complete = [i for i in nonempty if all(not generated_by_stage[s][i]["truncated"] for s in STAGES)]
    assert sources["input_ids"] == all_ids and sources["trajectory_ids"] == nonempty and sources["complete_ids"] == complete
    panels = {"all_input": all_ids, "all_trajectory": nonempty, "common_complete": complete}
    from .common import sample_schedule
    for panel, ids in panels.items():
        expected_schedule = sample_schedule(len(ids), cfg["statistics"])
        if expected_schedule is not None:
            np.testing.assert_array_equal(np.load(root / f"reports/schedules/{panel}.npy", allow_pickle=False), expected_schedule)
    summaries = {(r["panel"], r["stage"], r["anchor"], r["layer"], r["method"]): r for r in read(root / "reports/statistics.json")["summary"]}
    seen = set()
    for receipt in manifest["caches"]:
        path = root / receipt["path"]
        assert sha(path) == receipt["sha256"]
        c = read(path)
        for method in ("twonn", "correlation", "covariance"):
            output = summaries[(receipt["panel"], receipt["stage"], receipt["anchor"], receipt["layer"], method)]
            values = [r["value"] for r in c["records"] if r["method"] == method and r["replicate"] >= 0 and r["status"] == "ok"]
            assert output["n"] == len(panels[receipt["panel"]]) and output["valid_repeats"] == len(values)
            assert output["full"] == next(r["value"] for r in c["records"] if r["method"] == method and r["replicate"] == -1)
            if values:
                np.testing.assert_allclose([output["mean"], output["sd"]], [np.mean(values), np.std(values, ddof=0)], rtol=1e-13)
            else:
                assert output["mean"] is output["sd"] is None
        if path in seen:
            continue
        seen.add(path)
        assert digest(c["records"]) == c["records_hash"] and c["source_identity"] == sources["identity"]
        assert sha(path.with_suffix(".npz")) == receipt["spectrum_sha256"] == c["spectrum_sha256"]
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as spectra:
            for record in c["records"]:
                if record["method"] == "twonn":
                    assert record["n_fit"] == max(0, min(int(.9 * record["n_valid"]), record["n_valid"] - 2))
                    continue
                spectrum = spectra[f"{record['method']}_{record['replicate']}"]
                assert spectrum.shape == (4096,) and np.isfinite(spectrum).all() and np.all(spectrum >= 0)
                assert np.all(np.diff(spectrum) <= 0)
                count = int(np.searchsorted(np.cumsum(spectrum / spectrum.sum()), .9)) + 1
                assert count == record["value"] and record["trace_relative_error"] <= 1e-10
    comparisons, skipped = [], []
    if deep:
        # Prespecified cells, independently recomputed using cdist/direct SVD.
        for stage, anchor, layer in zip(STAGES, (0, 25, 75, 100), (1, 16, 32, 16)):
            panel = "all_input" if anchor == 0 else "all_trajectory"
            ids = sources["input_ids"] if anchor == 0 else sources["trajectory_ids"]
            if len(ids) < 4:
                skipped.append({"stage": stage, "anchor": anchor, "layer": layer, "reason": "insufficient_samples", "n": len(ids)})
                continue
            receipt = next(c for c in manifest["caches"] if (c["stage"], c["panel"], c["anchor"], c["layer"]) == (stage, panel, anchor, layer))
            c = read(root / receipt["path"])
            values = []
            ai = (0, 25, 50, 75, 100).index(anchor)
            for ident in ids:
                with np.load(root / f"activations/formal/{stage}/{ident}.npz") as z:
                    slot = z["anchors"].tolist().index(ai)
                    values.append(z["hidden"][slot, layer - 1, :].astype(np.float64))
            x = np.stack(values)
            estimate = independent_twonn(x)
            expected = next(r["value"] for r in c["records"] if r["replicate"] == -1 and r["method"] == "twonn")
            if expected is None:
                assert estimate is None
            else:
                np.testing.assert_allclose(estimate, expected, rtol=1e-10, atol=1e-10)
            for method in ("correlation", "covariance"):
                centered = x - x.mean(axis=0)
                const = np.all(x == x[0], axis=0)
                centered[:, const] = 0
                if method == "correlation":
                    centered[:, ~const] /= np.std(x[:, ~const], axis=0, ddof=0)
                singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
                ratios = singular ** 2 / np.sum(singular ** 2)
                count = int(np.searchsorted(np.cumsum(ratios), .9)) + 1
                expected_count = next(r["value"] for r in c["records"] if r["replicate"] == -1 and r["method"] == method)
                assert count == expected_count
            comparisons.append({"stage": stage, "anchor": anchor, "layer": layer})
    result = {"status": "passed", "checked_records": checked, "coverage": coverage, "geometry_caches": len(seen),
              "independent_numerical_cells": comparisons, "source_identity": sources["identity"], "verified_utc": utc()}
    result["independent_numerical_cells_skipped"] = skipped
    write(root / "reports/independent_audit.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--no-deep", action="store_true")
    args = p.parse_args()
    with threadpool_limits(limits=4):
        print(verify(args.root, deep=not args.no_deep), flush=True)
