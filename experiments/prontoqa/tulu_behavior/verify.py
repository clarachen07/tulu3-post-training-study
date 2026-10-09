"""Independently verify complete formal outputs and their report exports."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from scipy.stats import binomtest

from .assets import check_assets
from .common import (ROOT, CONDITIONS, STAGES, atomic_json, fingerprint, load_config,
                     physical_wrappers, read_jsonl, sha256_file, utc_now, verify_files)
from .freeze import validate_frozen
from .prompts import build_prompt
from .run import (batch_groups, batch_paths, input_path, output_directory, repetition_diagnostics,
                  split_stopped_output, verify_batch)
from .scoring import score_response
from .tokenize import load_native_tokenizer


LOGICAL_SCORE_FIELDS = ("answer_correct", "proof_success", "joint_success", "answer_status",
                        "format_compliant", "proof_parseable", "has_skip", "has_logic_error", "step_count")
METRICS = ("answer_correct", "proof_success", "joint_success")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _same(actual, expected, message):
    # JSON fingerprints distinguish booleans from integers, unlike Python ==.
    _require(fingerprint(actual) == fingerprint(expected), message)


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _number(actual, expected, message):
    _require(type(actual) in (int, float) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-14), message)


def _verify_source(source, item, demos, identity):
    for field in ("sample_id", "hop", "question", "query", "text", "answer", "chain_of_thought"):
        _same(source.get(field), item[field], f"Frozen input item field differs: {field}")
    for field in ("phase", "stage", "condition", "wrapper"):
        _same(source.get(field), identity[field], f"Frozen input condition differs: {field}")
    ids = source.get("input_ids")
    _require(isinstance(ids, list) and ids and all(type(t) is int and t >= 0 for t in ids), "Invalid frozen input token IDs")
    _same(source.get("attention_mask"), [1] * len(ids), "Frozen input is padded or malformed")
    _same(source.get("input_token_count"), len(ids), "Frozen input token count differs")
    _same(source.get("prompt_token_hash"), fingerprint(ids), "Frozen input token hash differs")
    rendered = source.get("prompt_text")
    _require(isinstance(rendered, str), "Frozen rendered prompt is missing")
    _same(source.get("prompt_text_hash"), _sha_text(rendered), "Frozen rendered prompt hash differs")
    body = build_prompt(item, demos, identity["condition"])
    _same(source.get("prompt_body_hash"), _sha_text(body), "Frozen prompt does not match the shared demonstrations and question")
    if identity["wrapper"] == "plain":
        _same(rendered, body, "Plain prompt differs from the shared body")
    else:
        _require(rendered.count(body) == 1, "Native prompt must contain the shared body exactly once")


def _verify_stop(record, audit, inference, tokenizer=None):
    tokens = record.get("output_token_ids")
    _require(isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens), "Invalid output token IDs")
    _same(record.get("output_tokens"), len(tokens), "Output token count differs from the saved sequence")
    _require(len(tokens) <= record["budget"], "Output exceeds the frozen generation budget")
    eos_ids = audit.get("eos_token_ids")
    _require(isinstance(eos_ids, list) and eos_ids and all(type(t) is int for t in eos_ids), "Missing native EOS audit")
    reason = record.get("finish_reason")
    _require(reason in ("eos", "stop", "length"), "Unknown finish reason")
    _same(record.get("truncated"), reason == "length", "Truncated flag disagrees with finish reason")
    raw, continuation = record.get("raw_continuation"), record.get("continuation")
    _require(isinstance(raw, str) and isinstance(continuation, str), "Raw/scored continuations must be strings")
    _require(not any(t in eos_ids for t in tokens[:-1]), "Generation continued after an EOS token")
    markers = inference["stop_strings"]
    if reason == "length":
        _same(len(tokens), record["budget"], "Length-limited generation did not fill its budget")
        _same(record.get("stop_reason"), "max_new_tokens", "Length stop reason differs")
        _require(tokens[-1] not in eos_ids and not any(m in raw for m in markers), "Length stop conceals a prior EOS/string stop")
        _same(continuation, raw, "Length-limited continuation was altered")
    elif reason == "eos":
        _require(tokens[-1] in eos_ids, "EOS stop is not terminated by a native EOS token")
        _same(record.get("stop_reason"), tokens[-1], "EOS stop reason differs from the saved terminal token")
        _require(raw.startswith(continuation), "EOS continuation is not a prefix of the raw output")
        _require(not any(m in raw for m in markers), "EOS stop conceals an earlier string stop")
    else:
        _require(tokens[-1] not in eos_ids, "String stop incorrectly ends in an EOS token")
        locations = [(raw.find(marker), marker) for marker in markers if marker in raw]
        _require(bool(locations), "String stop marker is absent from raw output")
        position, marker = min(locations)
        _same(record.get("stop_reason"), marker, "Recorded string stop is not the first declared marker")
        _same(continuation, raw[:position], "Scored continuation differs from the declared string-stop prefix")
    if tokenizer is not None:
        decoded_raw, decoded_continuation, marker = split_stopped_output(tokens, tokenizer, eos_ids, markers, reason)
        _same(raw, decoded_raw, "Raw continuation does not decode from its saved token IDs")
        _same(continuation, decoded_continuation, "Scored continuation does not decode from the declared stop")
    _same(record.get("repetition"), repetition_diagnostics(continuation), "Repetition diagnostics differ")


def _logical_row(item, record, template):
    return {
        "sample_id": item["sample_id"], "hop": item["hop"], "label": item["answer"],
        "stage": record["stage"], "condition": record["condition"], "template": template,
        "physical_wrapper": record["wrapper"], "generation_id": record["generation_id"],
        "shared_base_generation": record["stage"] == "base",
        "extracted_answer": record["scoring"].get("extracted_answer"),
        **{key: record["scoring"].get(key) for key in LOGICAL_SCORE_FIELDS},
        **{key: record[key] for key in ("input_tokens", "output_tokens", "finish_reason", "truncated")},
    }


def _verify_csv(path, expected):
    def value(v):
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, sort_keys=True)
        return "" if v is None else str(v)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.DictReader(stream))
    _same(actual, [{k: value(v) for k, v in row.items()} for row in expected], f"CSV export differs from verified rows: {path.name}")


def load_case_review(root, protocol_digest, audit):
    """A case review is attributable to Codex, never represented as human review."""
    root = Path(root)
    pending = {"status": "pending", "reviewer": None, "required_for_pipeline_completion": True}
    path = root / "manifests/case_review.json"
    if not path.is_file():
        return {**pending, "reason": "No Codex case-review record exists."}
    try:
        review = json.loads(path.read_text())
        if not isinstance(review, dict):
            raise ValueError("Case-review record must be an object.")
        required_ids = {case["record"]["generation_id"] for case in audit["cases"]}
        recorded_ids = review.get("reviewed_generation_ids")
        checks = (
            (review.get("status") == "passed", "Case review has not passed."),
            (review.get("reviewer") == "Codex", "Case reviewer is not identified as Codex."),
            (review.get("protocol_digest") == protocol_digest, "Case review refers to a different protocol."),
            (review.get("audit_sample_sha256") == sha256_file(root / "reports/error_audit_sample.json"),
             "Case-review sample hash differs from the current audit sample."),
            (isinstance(review.get("findings"), list), "Case-review findings list is missing."),
            (isinstance(recorded_ids, list) and all(isinstance(gid, str) for gid in recorded_ids),
             "Reviewed generation IDs must be a list of strings."),
        )
        for passed, reason in checks:
            if not passed:
                return {**pending, "reviewer": review.get("reviewer"), "reason": reason}
        if not required_ids <= set(recorded_ids):
            return {**pending, "reviewer": "Codex", "reason": "Case review does not cover every sampled generation."}
        return {"status": "passed", "reviewer": "Codex", "required_for_pipeline_completion": True,
                "reviewed_unique_generations": len(set(recorded_ids)), "sampled_unique_generations": len(required_ids),
                "findings_count": len(review["findings"]), "manifest_sha256": sha256_file(path)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**pending, "reason": f"Case-review evidence is invalid: {exc}"}


def _verify_reports(root, config, frozen, items, physical, raw_files):
    reports = root / "reports"
    manifest_path = root / "manifests/analysis.json"
    _require(manifest_path.is_file(), "Formal analysis manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    _same(manifest.get("protocol_digest"), frozen["protocol_digest"], "Analysis manifest uses another frozen protocol")
    _same(manifest.get("config_sha256"), sha256_file(root / "experiment.json"), "Analysis configuration hash differs")
    _same(manifest.get("physical_generations"), len(physical), "Analysis manifest double-counts or omits physical generations")
    _same(manifest.get("logical_records"), len(items) * 16, "Analysis manifest logical record count differs")
    files = manifest.get("files", {})
    required = {"reports/" + name for name in (
        "analysis.json", "provenance.json", "logical_results.jsonl", "logical_results.csv", "report.html",
        "summary.csv", "primary_tests.csv", "paired_differences.csv", "transitions.csv", "diagnostics.csv",
        "label_summary.csv", "dataset_summary.csv", "error_audit_sample.json",
        "figures/stage_comparison.png", "figures/stage_comparison.svg",
        "figures/difficulty_curves.png", "figures/difficulty_curves.svg")}
    _require(required <= files.keys(), "Analysis manifest omits a required report artifact")
    _require(all(not Path(p).is_absolute() and ".." not in Path(p).parts and p.startswith("reports/") for p in files),
             "Analysis manifest contains a path outside reports")
    verify_files(root, files)
    summary = json.loads((reports / "analysis.json").read_text())
    provenance = json.loads((reports / "provenance.json").read_text())
    _same(summary.get("provenance"), provenance, "Embedded and standalone report provenance differ")
    _same(provenance.get("protocol_digest"), frozen["protocol_digest"], "Report uses another frozen protocol")
    _same(provenance.get("generation"), frozen["identity"]["generation"], "Report generation settings differ")
    _same(provenance.get("frozen_protocol_sha256"), sha256_file(root / "manifests/frozen_protocol.json"), "Report frozen-protocol hash differs")
    _same(provenance.get("config_sha256"), sha256_file(root / "experiment.json"), "Report configuration hash differs")
    _same(provenance.get("dataset_sha256"), sha256_file(root / "data/formal.jsonl"), "Report dataset hash differs")
    _same(provenance.get("raw_batches"), raw_files, "Report provenance omits or invents formal output batches")
    _same(provenance.get("models"), config["models"], "Report model registry differs")
    for field, name in (("analysis_code_sha256", "analyze.py"), ("scoring_code_sha256", "scoring.py")):
        _same(provenance.get(field), sha256_file(root / "tulu_behavior" / name), f"Report {name} provenance differs")
    payload = {k: v for k, v in provenance.items() if k not in ("generated_at_utc", "source_fingerprint")}
    _same(provenance.get("source_fingerprint"), fingerprint(payload), "Report source fingerprint differs")
    _same(summary.get("status"), "complete_matrix_analyzed", "Report does not describe a complete matrix")
    _same(summary.get("n_items"), len(items), "Report item count differs")
    _same(summary.get("physical_generations"), len(physical), "Report physical generation count differs")
    _same(summary.get("logical_records"), len(items) * 16, "Report logical row count differs")
    expected_stats = {
        "bootstrap_replicates": config["statistics"]["bootstrap_replicates"], "seed": config["statistics"]["seed"],
        "confidence_level": config["statistics"]["confidence_level"],
        "bootstrap": "percentile_stratified_paired_shared_indices", "primary": "native/cot/answer_correct/equal_weight_hops",
        "multiplicity": "Holm_family_3",
    }
    _same(summary.get("statistics"), expected_stats, "Report statistics differ from the frozen analysis plan")
    _same(provenance.get("statistics"), expected_stats, "Report provenance statistics differ")
    logical = []
    for item in sorted(items, key=lambda x: (x["hop"], x["sample_id"])):
        for stage in STAGES:
            for condition in CONDITIONS:
                for template in ("native", "plain"):
                    wrapper = "plain" if stage == "base" else template
                    logical.append(_logical_row(item, physical[item["sample_id"], stage, condition, wrapper], template))
    _same(read_jsonl(reports / "logical_results.jsonl"), logical, "Logical report rows do not reference the exact verified physical generations")
    _verify_csv(reports / "logical_results.csv", logical)
    summaries = summary.get("summaries", [])
    expected_keys = {(s, c, t, h, m) for s in STAGES for c in CONDITIONS for t in ("native", "plain")
                     for h in ("all", 1, 3, 5) for m in (METRICS if c == "cot" else METRICS[:1])}
    actual_keys = [(r["stage"], r["condition"], r["template"], r["hop"], r["metric"]) for r in summaries]
    _require(len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys, "Report summary matrix is incomplete or duplicated")
    points = {}
    for row in summaries:
        stage, condition, template, hop, metric = (row[k] for k in ("stage", "condition", "template", "hop", "metric"))
        wrapper = "plain" if stage == "base" else template
        selected = [i for i in items if hop == "all" or i["hop"] == hop]
        values = [physical[i["sample_id"], stage, condition, wrapper]["scoring"][metric] for i in selected]
        estimate = sum(values) / len(values)
        _number(row["estimate"], estimate, "Report success estimate differs from verified per-item scores")
        _same(row["n"], len(selected), "Report success denominator differs")
        _same(row["aggregation"], "equal_weight_hops" if hop == "all" else "within_hop", "Report aggregation policy differs")
        _require(0 <= row["ci_low"] <= row["ci_high"] <= 1, "Report confidence interval is out of range")
        points[stage, condition, template, hop, metric] = estimate
    for row in summary["paired_differences"]:
        before = points[row["before_stage"], row["before_condition"], row["before_template"], row["hop"], row["metric"]]
        after = points[row["after_stage"], row["after_condition"], row["after_template"], row["hop"], row["metric"]]
        _number(row["delta_pp"], 100 * (after - before), "Report paired difference estimate differs")
        _require(-100 <= row["ci_low_pp"] <= row["ci_high_pp"] <= 100, "Paired interval is out of range")
        if row["kind"] == "template_native_minus_plain" and row["before_stage"] == "base":
            _same([row["delta_pp"], row["ci_low_pp"], row["ci_high_pp"]], [0., 0., 0.], "Base template views were treated as independent")
    tests = summary.get("primary_tests", [])
    comparisons = [tuple(pair) for pair in config["statistics"]["primary_comparisons"]]
    _same([(r["before_stage"], r["after_stage"]) for r in tests], comparisons, "Primary comparison family differs")
    raw_p = []
    for row, (before, after) in zip(tests, comparisons):
        left = [physical[i["sample_id"], before, "cot", "plain" if before == "base" else "native"]["scoring"]["answer_correct"] for i in items]
        right = [physical[i["sample_id"], after, "cot", "native"]["scoring"]["answer_correct"] for i in items]
        gained = sum(not a and b for a, b in zip(left, right))
        lost = sum(a and not b for a, b in zip(left, right))
        expected = {"n": len(items), "gained_correct": gained, "lost_correct": lost,
                    "both_correct": sum(a and b for a, b in zip(left, right)),
                    "both_wrong": sum(not a and not b for a, b in zip(left, right)),
                    "discordant": gained + lost, "family_size": 3, "test": "exact_two_sided_mcnemar",
                    "hop": "all", "metric": "answer_correct", "before_condition": "cot", "after_condition": "cot",
                    "before_template": "native", "after_template": "native"}
        for key, value in expected.items():
            _same(row.get(key), value, f"Primary McNemar field differs: {key}")
        p = float(binomtest(gained, gained + lost, .5).pvalue) if gained + lost else 1.
        raw_p.append(p)
        _number(row["p_value"], p, "Primary McNemar p-value differs")
        _number(row["delta_pp"], 100 * (gained - lost) / len(items), "Primary stage difference differs")
    running = 0.
    for rank, index in enumerate(sorted(range(len(raw_p)), key=raw_p.__getitem__)):
        running = min(1., max(running, (len(raw_p) - rank) * raw_p[index]))
        _number(tests[index]["p_holm"], running, "Holm adjustment differs")
        _same(tests[index]["reject_holm_0_05"], running < .05, "Holm rejection flag differs")
    for key, filename in (("summaries", "summary"), ("primary_tests", "primary_tests"),
                          ("paired_differences", "paired_differences"), ("transitions", "transitions"),
                          ("diagnostics", "diagnostics"), ("label_summaries", "label_summary"), ("dataset_summary", "dataset_summary")):
        _verify_csv(reports / f"{filename}.csv", summary[key])
    by_generation = {r["generation_id"]: r for r in physical.values()}
    item_map = {i["sample_id"]: i for i in items}
    audit = json.loads((reports / "error_audit_sample.json").read_text())
    for case in audit["cases"]:
        record = case["record"]
        _require(record["generation_id"] in by_generation, "Audit sample invents a generation")
        _same(record["record_hash"], by_generation[record["generation_id"]]["record_hash"], "Audit generation differs from raw output")
        _same(fingerprint({k: v for k, v in record.items() if k != "record_hash"}), record["record_hash"], "Audit record hash differs")
        _same(case["item"], item_map[record["sample_id"]], "Audit question differs from its formal item")
    current_review = load_case_review(root, frozen["protocol_digest"], audit)
    reported_review = summary.get("case_review", {})
    _same(reported_review.get("status"), current_review["status"],
          "Report case-review state is stale; regenerate the report before final verification")
    if current_review["status"] == "passed":
        _same(reported_review.get("reviewer"), "Codex", "Report misidentifies the case reviewer")
        _same(reported_review.get("manifest_sha256"), current_review["manifest_sha256"], "Report case-review evidence hash differs")
    return {**files, "manifests/analysis.json": sha256_file(manifest_path)}, len(logical)


def verify(root=ROOT, offline=False, require_reports=True, *, write_result=True):
    root = Path(root).resolve()
    config = load_config(root)
    frozen = validate_frozen(root, check_environment=not offline)
    if not offline:
        check_assets(root)
    generation = frozen["identity"]["generation"]
    batch_size, budgets, digest = generation["batch_size"], generation["budgets"], frozen["protocol_digest"]
    data_manifest = json.loads((root / "manifests/data.json").read_text())
    source_path = (root / config["data"]["original_items"]).resolve()
    source_manifest_path = (root / config["data"]["original_manifest"]).resolve()
    _same(sha256_file(source_path), data_manifest["original_3hop"]["sha256"], "Original 3-hop source bytes changed")
    _same(sha256_file(source_manifest_path), data_manifest["original_3hop"]["manifest_sha256"], "Original 3-hop source manifest changed")
    items = read_jsonl(root / "data/formal.jsonl")
    demos = read_jsonl(root / "data/demonstrations.jsonl")
    _same(dict(Counter(i["hop"] for i in items)), {hop: config["data"]["count_per_hop"] for hop in (1, 3, 5)}, "Formal hop counts differ")
    item_map = {i["sample_id"]: i for i in items}
    _require(len(item_map) == len(items), "Formal sample IDs are duplicated")
    original = read_jsonl(source_path)
    retained = [i for i in items if i["hop"] == 3]
    _same([i["sample_id"] for i in retained], [i["sample_id"] for i in original], "Original 3-hop sample IDs/order changed")
    for before, after in zip(original, retained):
        _same({k: after.get(k) for k in before}, before, "Original 3-hop item fields changed")
    expected_count = len(items) * sum(len(physical_wrappers(stage)) for stage in STAGES) * len(CONDITIONS)
    _same(expected_count, config["expected"]["physical_generations_formal"], "Configured physical generation total is inconsistent")
    token_manifest = json.loads((root / "manifests/tokenization.json").read_text())
    expected_batches, expected_metadata = set(), set()
    physical, files, raw_files, cells, stage_audits = {}, {}, {}, Counter(), {}
    generation_ids = set()
    for model in config["models"]:
        stage = model["stage"]
        tokenizer = None if offline else load_native_tokenizer(Path(model["path"]))
        for condition in CONDITIONS:
            for wrapper in physical_wrappers(stage):
                identity = dict(phase="formal", stage=stage, condition=condition, wrapper=wrapper,
                                budget=budgets[condition], batch_size=batch_size, run_digest=digest)
                sources = read_jsonl(input_path(root, "formal", stage, condition, wrapper))
                _same([s["sample_id"] for s in sources], [i["sample_id"] for i in items], "Frozen input item order differs from formal data")
                directory = output_directory(root, "formal", stage, condition, wrapper, budgets[condition], batch_size, digest)
                for start, group in batch_groups(sources, config["inference"]["write_batch_size"]):
                    file, metadata_path = batch_paths(directory, start)
                    expected_batches.add(file)
                    expected_metadata.add(metadata_path)
                    rows = verify_batch(file, metadata_path, group, identity)
                    _require(rows is not None, f"Missing formal output batch: {file}")
                    metadata = json.loads(metadata_path.read_text())
                    _same(metadata.get("identity"), identity, "Batch metadata identity differs")
                    _same(metadata.get("count"), len(group), "Batch metadata count differs")
                    _same(metadata.get("sample_ids"), [s["sample_id"] for s in group], "Batch metadata sample IDs differ")
                    _same(metadata.get("sha256"), sha256_file(file), "Batch metadata digest differs")
                    audit = metadata.get("model_audit", {})
                    for key, expected in (("dtype", "bfloat16"), ("device", "cuda:0"), ("quantization", None),
                                          ("deterministic_algorithms", True), ("allow_tf32", False),
                                          ("attention_implementation", config["inference"]["attention_implementation"])):
                        _same(audit.get(key), expected, f"Generation model audit differs: {key}")
                    if stage in stage_audits:
                        _same(audit, stage_audits[stage], "Model audit changed between formal batches")
                    stage_audits[stage] = audit
                    native_eos = audit.get("native_generation_config", {}).get("eos_token_id")
                    if native_eos is None:
                        native_eos = token_manifest["stages"][stage]["eos_token_id"]
                    eos_ids = native_eos if isinstance(native_eos, list) else [native_eos]
                    _same(audit.get("eos_token_ids"), eos_ids, "Effective EOS IDs differ from the recorded native generation config")
                    for source, record in zip(group, rows):
                        item = item_map[source["sample_id"]]
                        _same({k: record.get(k) for k in identity}, identity, "Generated row inference identity differs")
                        _verify_source(source, item, demos, identity)
                        _same(record.get("hop"), item["hop"], "Generated hop differs from its item")
                        _same(record.get("prompt_text_hash"), source["prompt_text_hash"], "Generated prompt text hash differs")
                        _same(record.get("prompt_token_hash"), fingerprint(source["input_ids"]), "Generated exact-input hash differs")
                        _same(record.get("input_tokens"), len(source["input_ids"]), "Generated input token count differs")
                        _require(len(source["input_ids"]) + budgets[condition] <= config["inference"]["max_context_tokens"], "Frozen generation exceeds context limit")
                        expected_gid = fingerprint([identity, source["sample_id"], source["prompt_token_hash"]])
                        _same(record.get("generation_id"), expected_gid, "Generation ID does not encode its exact item and protocol")
                        _require(expected_gid not in generation_ids, "Duplicate physical generation ID")
                        generation_ids.add(expected_gid)
                        _verify_stop(record, audit, config["inference"], tokenizer)
                        rescored = score_response(item, record["continuation"], condition)
                        _same(record.get("scoring"), rescored, "Saved scoring differs from independent rescoring")
                        incomplete = bool(record["truncated"] and (rescored["extracted_answer"] is None
                                          or rescored["diagnostics"].get("trailing_after_answer", False)))
                        _same(record.get("budget_incomplete"), incomplete, "Budget-incomplete flag differs")
                        key = (item["sample_id"], stage, condition, wrapper)
                        _require(key not in physical, "Duplicate physical item/condition")
                        compact = {k: record[k] for k in ("generation_id", "record_hash", "stage", "condition", "wrapper",
                                                          "input_tokens", "output_tokens", "finish_reason", "truncated")}
                        compact["scoring"] = {k: rescored.get(k) for k in (*LOGICAL_SCORE_FIELDS, "extracted_answer")}
                        physical[key] = compact
                        cells[stage, condition, wrapper, item["hop"]] += 1
                    rel, meta_rel = str(file.relative_to(root)), str(metadata_path.relative_to(root))
                    raw_files[rel] = files[rel] = sha256_file(file)
                    files[meta_rel] = sha256_file(metadata_path)
    actual_batches = set((root / "outputs/formal").rglob("batch_*.jsonl"))
    actual_metadata = set((root / "outputs/formal").rglob("batch_*.meta.json"))
    _require(actual_batches == expected_batches, "Unexpected or missing formal batch files, including possible Base native duplicates")
    _require(actual_metadata == expected_metadata, "Unexpected or missing formal batch metadata files")
    _same(len(physical), expected_count, "Formal physical generation total differs")
    expected_cells = {(s, c, w, h): config["data"]["count_per_hop"]
                      for s in STAGES for c in CONDITIONS for w in physical_wrappers(s) for h in (1, 3, 5)}
    _same({"/".join(map(str, k)): v for k, v in cells.items()},
          {"/".join(map(str, k)): v for k, v in expected_cells.items()}, "Formal physical cell counts differ")
    logical_count = None
    case_review = {"status": "pending", "reviewer": None, "required_for_pipeline_completion": True,
                   "reason": "Formal reports have not yet been verified."}
    if require_reports:
        report_files, logical_count = _verify_reports(root, config, frozen, items, physical, raw_files)
        files.update(report_files)
        case_review = load_case_review(root, digest, json.loads((root / "reports/error_audit_sample.json").read_text()))
        review_path = root / "manifests/case_review.json"
        if review_path.is_file():
            files["manifests/case_review.json"] = sha256_file(review_path)
    result = {
        "schema_version": 1, "status": "passed", "verified_utc": utc_now(), "protocol_digest": digest,
        "scope": "raw_and_reports" if require_reports else "raw_only", "reports_verified": bool(require_reports),
        "offline": bool(offline), "environment_checked": not offline, "model_assets_checked": not offline,
        "output_token_decoding_checked": not offline, "formal_items": len(items), "physical_generations": len(physical),
        "physical_cells": len(cells), "logical_records": logical_count, "committed_batches": len(expected_batches),
        "rescored_generations": len(physical), "original_3hop_preserved": True,
        "cell_counts": {"/".join(map(str, k)): v for k, v in sorted(cells.items())},
        "case_review": case_review, "files": files,
    }
    if write_result:
        atomic_json(root / "manifests/final_verification.json", result)
    return result


def main():
    from .common import add_workspace_arguments, workspace
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_arguments(parser, analysis=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--published", action="store_true", help="Check only the published aggregates and links")
    args = parser.parse_args()
    if args.published:
        from .publication import verify_publication
        result = verify_publication()
        print(json.dumps(result))
    elif args.source_dir is not None:
        from .reanalysis import reanalyze
        result = reanalyze(args.source_dir, args.output_dir)
        print(json.dumps({k: result[k] for k in ("status", "scope", "rescored_generations")}))
    else:
        result = verify(workspace(args), offline=args.offline, require_reports=not args.raw_only)
        print(json.dumps({key: result[key] for key in ("status", "scope", "physical_generations", "rescored_generations", "case_review")}))


if __name__ == "__main__":
    main()
