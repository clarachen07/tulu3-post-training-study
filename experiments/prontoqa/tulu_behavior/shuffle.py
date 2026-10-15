"""Deterministic sentence permutations of complete external preorder evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .common import (
    REPO,
    atomic_json,
    fingerprint,
    load_config,
    publication_code_hashes,
    read_jsonl,
    sha256_file,
    utc_now,
    verify_files,
)
from .freeze import validate_frozen
from .scoring import Atom, Rule, parse_statement, validate_item

VERSION = "sentence-permutation-v1"
SPLITS = ("formal", "pilot", "demonstrations")
FIXED_GENERATION = {"batch_size": 4, "budgets": {"cot": 512, "direct": 32}}


def sentences(question):
    parts = [s.strip() for s in re.findall(r"[^.]+\.", question)]
    if not parts or " ".join(parts) != question:
        raise ValueError(
            "Question must contain complete, unchanged period-terminated sentences"
        )
    return parts


def shuffle_item(item, split, seed):
    parts = sentences(item["question"])
    key = fingerprint([VERSION, seed, split, item["sample_id"]])
    permutation = list(range(len(parts)))
    random.Random(int(key, 16)).shuffle(permutation)
    question = " ".join(parts[index] for index in permutation)
    return {
        **item,
        "question": question,
        "text": question + " " + item["query"],
        "ordering": {
            "version": VERSION,
            "seed": seed,
            "item_seed": key,
            "source_question_sha256": hashlib.sha256(
                item["question"].encode()
            ).hexdigest(),
            "source_text_sha256": hashlib.sha256(item["text"].encode()).hexdigest(),
            "permutation": permutation,
        },
    }


def validate_pair(before, after, split, seed):
    if fingerprint(after) != fingerprint(shuffle_item(before, split, seed)):
        raise RuntimeError(
            f"Derived item or permutation changed: {before['sample_id']}"
        )
    if Counter(sentences(before["question"])) != Counter(sentences(after["question"])):
        raise RuntimeError("Sentence multiset changed")
    if validate_item(after) != before["validation"]:
        raise RuntimeError("Gold structure changed after permutation")


def ordering_summary(items):
    grouped = defaultdict(list)
    for item in items:
        grouped[item["hop"]].append(item)
    result = {}
    for hop, group in sorted(grouped.items()):
        reverse = 0
        positions = Counter()
        unchanged = 0
        for item in group:
            parts = sentences(item["question"])
            formulas = [parse_statement(p) for p in parts]
            indices = {formula: i for i, formula in enumerate(formulas)}
            chain = [
                indices[parse_statement(step)]
                for step in item["chain_of_thought"]
                if isinstance(parse_statement(step), Rule)
            ]
            reverse += int(
                len(chain) > 1 and all(a > b for a, b in itertools.pairwise(chain))
            )
            positions.update(
                str(i + 1)
                for i, formula in enumerate(formulas)
                if isinstance(formula, Atom)
            )
            permutation = item["ordering"]["permutation"]
            unchanged += int(permutation == list(range(len(permutation))))
        result[str(hop)] = {
            "n": len(group),
            "unchanged_questions": unchanged,
            "unchanged_fraction": unchanged / len(group),
            "full_reverse_rule_chain": reverse if hop > 1 else None,
            "full_reverse_rule_chain_fraction": reverse / len(group)
            if hop > 1
            else None,
            "fact_position_counts_1based": dict(positions),
        }
    return result


def baseline_identity(source, config):
    """Validate scientific compatibility; local paths are not model identities."""
    source = Path(source).resolve()
    baseline = load_config(source)
    frozen = validate_frozen(source, check_environment=False)
    if (
        baseline["data"]["ordering"] != "preorder"
        or frozen["identity"]["generation"] != FIXED_GENERATION
    ):
        raise ValueError(
            "Order comparison requires a preorder baseline at batch 4 and budgets 512/32"
        )
    registry = lambda c: [
        {k: m[k] for k in ("stage", "id", "revision", "wrapper")} for m in c["models"]
    ]
    if registry(config) != registry(baseline):
        raise ValueError("Baseline model revisions differ")
    ignored = {
        "original_items",
        "original_manifest",
        "source_behavior_root",
        "source_ordering",
        "shuffle_seed",
        "shuffle_version",
        "shuffle_demonstrations",
        "ordering",
    }
    data = lambda c: {k: v for k, v in c["data"].items() if k not in ignored}
    inference = lambda c: {
        k: v
        for k, v in c["inference"].items()
        if k not in ("budget_candidates", "batch_candidates")
    }
    if data(config) != data(baseline) or inference(config) != inference(baseline):
        raise ValueError("Baseline corpus or inference settings differ")
    stats = lambda c: {
        k: v for k, v in c["statistics"].items() if k != "primary_comparisons"
    }
    if stats(config) != stats(baseline):
        raise ValueError("Baseline statistics differ")
    if (
        config["data"].get("source_ordering") != "preorder"
        or config["data"].get("shuffle_version") != VERSION
        or config["data"].get("shuffle_demonstrations") is not True
        or config["data"].get("shuffle_seed") != 20261014
        or config["inference"]["budget_candidates"] != {"cot": [512], "direct": [32]}
        or config["inference"]["batch_candidates"] != [4]
    ):
        raise ValueError("Shuffled protocol must use the fixed published settings")
    files = {
        name: sha256_file(source / name)
        for name in (
            "experiment.json",
            "manifests/data.json",
            "manifests/frozen_protocol.json",
            *[f"data/{split}.jsonl" for split in SPLITS],
        )
    }
    verify_files(
        source, json.loads((source / "manifests/data.json").read_text())["files"]
    )
    return {"source_protocol_digest": frozen["protocol_digest"], "source_files": files}


def verify_derivation(root, *, baseline_source=None):
    root = Path(root).resolve()
    config = load_config(root)
    source = (
        Path(baseline_source).resolve()
        if baseline_source
        else (root / config["data"]["source_behavior_root"]).resolve()
    )
    if source == root:
        raise ValueError("Shuffled source must be a separate preorder experiment")
    current = baseline_identity(source, config)
    manifest = json.loads((root / "manifests/data.json").read_text())
    identity = manifest["identity"]
    if fingerprint(identity) != manifest["data_fingerprint"]:
        raise RuntimeError("Shuffled derivation identity changed")
    if (
        identity["data_parameters"] != config["data"]
        or identity["source_protocol_digest"] != current["source_protocol_digest"]
    ):
        raise RuntimeError("Shuffled configuration or baseline protocol changed")
    verify_files(source, identity["source_files"])
    verify_files(root, manifest["files"])
    # Historical source files are checked as evidence; no archived code is imported.
    verify_files(root, identity.get("preparation_sources", {}))
    if "baseline_render_sha256" in identity and (
        sha256_file(root / "source-evidence/baseline-report.html")
        != identity["baseline_render_sha256"]
    ):
        raise RuntimeError("Archived baseline rendering evidence changed")
    for split in SPLITS:
        before = read_jsonl(source / f"data/{split}.jsonl")
        after = read_jsonl(root / f"data/{split}.jsonl")
        if len(before) != len(after) or len({r["sample_id"] for r in after}) != len(
            after
        ):
            raise RuntimeError("Shuffled pool count or IDs changed")
        for a, b in zip(before, after):
            validate_pair(a, b, split, config["data"]["shuffle_seed"])
        if ordering_summary(after) != manifest["ordering_summary"][split]:
            raise RuntimeError(
                "Ordering diagnostics differ from the actual permutations"
            )
    return manifest


def prepare_shuffled(root):
    from .data import _commit_jsonl

    root = Path(root).resolve()
    if (root / "manifests/data.json").exists():
        return verify_derivation(root)
    if (root / "manifests/frozen_protocol.json").exists():
        raise RuntimeError("Cannot recreate missing data after protocol freeze")
    config = load_config(root)
    source = (root / config["data"]["source_behavior_root"]).resolve()
    identity = {
        "data_parameters": config["data"],
        **baseline_identity(source, config),
        "preparation_sources": {
            f"tulu_behavior/{n}": sha256_file(root / "tulu_behavior" / n)
            for n in ("data.py", "shuffle.py", "scoring.py")
        },
    }
    files, counts, summaries = {}, {}, {}
    for split in SPLITS:
        before = read_jsonl(source / f"data/{split}.jsonl")
        after = [
            shuffle_item(item, split, config["data"]["shuffle_seed"]) for item in before
        ]
        for a, b in zip(before, after):
            validate_pair(a, b, split, config["data"]["shuffle_seed"])
        path = root / f"data/{split}.jsonl"
        _commit_jsonl(path, after)
        files[f"data/{split}.jsonl"] = sha256_file(path)
        counts[split], summaries[split] = len(after), ordering_summary(after)
    source_manifest = json.loads((source / "manifests/data.json").read_text())
    original_path = (root / config["data"]["original_items"]).resolve()
    original_manifest = (root / config["data"]["original_manifest"]).resolve()
    original = dict(
        source_manifest["original_3hop"],
        configured_path=config["data"]["original_items"],
        manifest_path=config["data"]["original_manifest"],
        all_original_field_values_preserved=False,
        original_nontext_field_values_preserved=True,
        sentence_multisets_preserved=True,
    )
    if (
        sha256_file(original_path) != original["sha256"]
        or sha256_file(original_manifest) != original["manifest_sha256"]
    ):
        raise RuntimeError("Original 3-hop corpus identity differs from baseline")
    manifest = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "identity": identity,
        "data_fingerprint": fingerprint(identity),
        "files": files,
        "counts": counts,
        "original_3hop": original,
        "ordering_summary": summaries,
        "demonstration_order": ["True", "False", "True", "False"],
        "derivation": "One uniform sentence permutation per item; no rejection or regeneration.",
    }
    atomic_json(root / "manifests/data.json", manifest)
    return verify_derivation(root)


def order_workspace(args):
    """Create a portable scientific run; external baseline evidence stays read-only."""
    reference = REPO / "behavior-order-sensitivity"
    default = REPO / "runs/behavior"
    output = (
        REPO / "runs/behavior-order-sensitivity"
        if args.output_dir == default
        else args.output_dir
    )
    root = (args.root or output).resolve()
    if (
        root == REPO
        or REPO.is_relative_to(root)
        or (root.is_relative_to(REPO) and not root.is_relative_to(REPO / "runs"))
    ):
        raise ValueError("Output must be an independent run directory")
    marker = root / "manifests/workspace.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    baseline = getattr(args, "baseline_source", None)
    if baseline is None and previous:
        baseline = Path(previous["baseline_source"])
    if baseline is None:
        raise ValueError(
            "--baseline-source must supply complete external preorder evidence"
        )
    baseline = Path(baseline).resolve()
    if (
        root == baseline
        or baseline.is_relative_to(root)
        or root.is_relative_to(baseline)
    ):
        raise ValueError("Output must be separate from baseline evidence")
    config = load_config(reference)
    source_identity = baseline_identity(baseline, config)
    old = load_config(baseline)
    originals = {
        "original/items.jsonl": (baseline / old["data"]["original_items"]).resolve(),
        "original/data.json": (baseline / old["data"]["original_manifest"]).resolve(),
    }
    model_dir = args.model_dir.resolve()
    config["data"].update(
        source_behavior_root=str(baseline),
        original_items="original/items.jsonl",
        original_manifest="original/data.json",
    )
    for model in config["models"]:
        model["path"] = str(model_dir / model["stage"])
    identity = {
        "code": publication_code_hashes(),
        "model_dir": str(model_dir),
        "baseline_source": str(baseline),
        "baseline_identity": source_identity,
        "static_inputs": {name: sha256_file(path) for name, path in originals.items()},
    }
    if previous is not None:
        if previous != identity or load_config(root) != config:
            raise ValueError(
                "Workspace configuration, code, models or baseline changed"
            )
        verify_files(root, identity["static_inputs"])
        verify_files(
            root,
            {
                n: h
                for n, h in identity["code"].items()
                if n.startswith("tulu_behavior/")
            },
        )
        return root
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output is nonempty without a workspace identity")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".order-", dir=root.parent) as temporary:
        target = Path(temporary) / "run"
        target.mkdir()
        for name in identity["code"]:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, destination)
        shutil.copytree(
            REPO / "vendor",
            target / "vendor",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (target / "provenance").mkdir()
        shutil.copyfile(
            REPO / "provenance/models.json", target / "provenance/models.json"
        )
        for name, source in originals.items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        atomic_json(target / "experiment.json", config)
        verify_files(target, identity["static_inputs"])
        atomic_json(target / "manifests/workspace.json", identity)
        if root.exists():
            root.rmdir()
        os.replace(target, root)
    return root
