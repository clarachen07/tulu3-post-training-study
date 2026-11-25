"""Bind a replay to the published GSM8K zero-shot behavior archive."""

from pathlib import Path

from gsm8k_id.common import STAGES, digest, read, rows, sha
from gsm8k_id.verify import numerical_value

PUBLISHED_STATS = "c7ec5341bbf67c36c6c71346ace278ad59b3e0a59713ad892e7a72a1c52ba137"
PUBLISHED_FREEZE = "a628d5eeb1ed9d53f2f8932ca9b786ea7204eade3ca0e7c3c6baac8df470e7fa"
STOP = "\n\nQuestion:"
MAX_TOKENS = 1024
PILOT_N = 8
TEST_N = 1319


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def checked(root, relative, expected):
    path = root / relative
    actual = sha(path)
    require(actual == expected, f"Changed archive file: {relative}: {actual}")
    return path


def indexed(records):
    result = {row["id"]: row for row in records}
    require(len(result) == len(records), "Duplicate question ID")
    return result


def published_statistics(root):
    relative = "reports/statistics.json"
    try:
        path = checked(root, relative, PUBLISHED_STATS)
    except (OSError, ValueError):
        # The repository retains an exact published copy independent of the
        # execution archive. Its digest authenticates the same batch inventory.
        path = checked(Path(__file__).resolve().parents[1] / "zero-shot", relative, PUBLISHED_STATS)
    return read(path)


def archive_inputs(root):
    frozen = read(checked(root, "manifests/frozen_protocol.json", PUBLISHED_FREEZE))["files"]
    cfg = read(checked(root, "experiment.json", frozen["experiment.json"]))
    require(cfg["protocol"] == "gsm8k-tulu-stages-zero-shot-v1" and cfg["demonstrations"] == 0, "Wrong input protocol")
    require(cfg["inference"]["max_tokens"] == MAX_TOKENS and cfg["inference"]["stop"] == [STOP], "Wrong stop contract")
    require(tuple(m["stage"] for m in cfg["models"]) == STAGES, "Unexpected stages")
    assets = read(checked(root, "manifests/assets.json", frozen["manifests/assets.json"]))
    published = read(checked(root, "manifests/published_identity.json", frozen["manifests/published_identity.json"]))
    require(published["status"] == "passed" and assets["config_hash"] == digest(cfg), "Unverified models")

    panels = {}
    for phase in ("pilot", "formal"):
        relative = f"data/prepared/{phase}.jsonl"
        source = indexed(rows(checked(root, relative, frozen[relative])))
        selected = sorted(source)[:PILOT_N] if phase == "pilot" else [
            ident for ident in sorted(source) if source[ident]["split"] == "test"]
        require(len(source) == (32 if phase == "pilot" else 2343), f"Unexpected {phase} input count")
        require(len(selected) == (PILOT_N if phase == "pilot" else TEST_N), f"Unexpected {phase} panel")
        panels[phase] = {}
        for stage in STAGES:
            relative = f"data/tokenized/{stage}/{phase}.jsonl"
            prompts = indexed(rows(checked(root, relative, frozen[relative])))
            require(prompts.keys() == source.keys(), f"Tokenized input coverage: {relative}")
            for ident in selected:
                prompt, item = prompts[ident], source[ident]
                expected = (item["content"] + "\nAnswer:" if stage == "base" else
                            "<|user|>\n" + item["content"] + "\n<|assistant|>\n")
                require(item["content"].startswith("Question: ") and prompt["prompt_text"] == expected,
                        f"Prompt differs for {stage}/{ident}")
                for key in ("id", "split", "gold", "question_hash", "rlvr_row"):
                    require(prompt[key] == item[key], f"Input metadata differs: {stage}/{ident}/{key}")
                tokens = prompt["prompt_token_ids"]
                require(prompt["prompt_text_hash"] == digest(expected) and prompt["prompt_token_hash"] == digest(tokens),
                        f"Input hash differs: {stage}/{ident}")
                require(0 < len(tokens) <= cfg["inference"]["max_model_len"] - MAX_TOKENS, f"Input length: {ident}")
            panels[phase][stage] = {ident: prompts[ident] for ident in selected}
        for ident in selected:
            require(panels[phase]["sft"][ident]["prompt_token_ids"] == panels[phase]["dpo"][ident]["prompt_token_ids"]
                    == panels[phase]["rlvr"][ident]["prompt_token_ids"], f"Native token mismatch: {ident}")
    require(not set(panels["formal"]["base"]) & set(panels["pilot"]["base"]), "Pilot overlaps formal inputs")
    return cfg, panels, assets, published


def archive_outputs(root, cfg, inputs):
    """Validate entire formal batch inventory and the selected archived pilot rows."""
    stats = published_statistics(root)
    expected_sources = stats["sources"]
    observed_sources = {}
    result = {}
    expected_config = digest(cfg)
    for phase, panel in inputs.items():
        result[phase] = {}
        for stage in STAGES:
            selected = panel[stage]
            raw = {}
            paths = sorted((root / f"outputs/{phase}/{stage}").glob("batch_*.jsonl"))
            for path in paths:
                relative = str(path.relative_to(root))
                if phase == "formal":
                    checked(root, relative, expected_sources[relative])
                    observed_sources[relative] = expected_sources[relative]
                for record in rows(path):
                    ident = record["id"]
                    require(ident not in raw, f"Duplicate output: {phase}/{stage}/{ident}")
                    raw[ident] = record
            require(len(raw) == (2343 if phase == "formal" else 32) and selected.keys() <= raw.keys(),
                    f"Archived output coverage differs: {phase}/{stage}")
            result[phase][stage] = {}
            for ident, prompt in selected.items():
                record = raw[ident]
                for key in ("id", "split", "gold", "question_hash", "rlvr_row", "prompt_token_hash", "prompt_text_hash"):
                    require(record[key] == prompt[key], f"Output metadata differs: {stage}/{ident}/{key}")
                require(record["stage"] == stage and record["phase"] == phase and record["config_hash"] == expected_config,
                        f"Wrong output identity: {phase}/{stage}/{ident}")
                require(record["record_hash"] == digest({k: v for k, v in record.items() if k != "record_hash"}),
                        f"Output record hash differs: {stage}/{ident}")
                value = numerical_value(record["continuation"])
                require(value == numerical_value(record["extracted_answer"] or "")
                        and record["correct"] == (value is not None and value == numerical_value(str(prompt["gold"]))),
                        f"Independent scoring differs: {stage}/{ident}")
                require(len(record["output_token_ids"]) == record["output_tokens"] <= MAX_TOKENS,
                        f"Archived output length differs: {ident}")
                require(record["finish_reason"] in ("length", "stop")
                        and record["truncated"] == (record["finish_reason"] == "length"),
                        f"Archived stop condition differs: {ident}")
                result[phase][stage][ident] = record
    require(observed_sources == expected_sources, "Formal batch inventory differs from published statistics")
    published_test = {row["stage"]: row for row in stats["summary"] if row["split"] == "test"}
    for stage, records in result["formal"].items():
        expected = published_test[stage]
        require(len(records) == expected["n"]
                and sum(r["correct"] for r in records.values()) == expected["correct"]
                and sum(r["truncated"] for r in records.values()) == expected["truncated"],
                f"Formal behavior counts differ: {stage}")
    return result
