"""Data, prompt and scoring contract for arXiv:2510.01685v2, Section 3."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "prontoqa"))

from tulu_behavior.common import atomic_json, fingerprint, sha256_file, verify_files

STAGES = ("base", "sft", "dpo", "rlvr")
MODEL_NAMES = dict(zip(STAGES, (
    "Meta-Llama-3.1-8B", "Llama-3.1-Tulu-3-8B-SFT",
    "Llama-3.1-Tulu-3-8B-DPO", "Llama-3.1-Tulu-3-8B",
)))
REGISTRY = ROOT.parent / "prontoqa" / "provenance/models.json"
PAPER_VERSION = "2510.01685v2"
CODE_REVISION = "f12cef400ff946ab09cee988817daea939436698"
DATA_REVISION = "5b7b70743ff849eae3875dc0c22d5443066b33e9"
DATA_SHA256 = "284d5fbfc6dd7a9fe327d9a39c9ce755402fa9f7180e92a842f88c09f176f079"
DATA_URL = (
    f"https://huggingface.co/datasets/apoorvkh/composing-functions/resolve/{DATA_REVISION}"
    "/data/train-00000-of-00001.parquet?download=true"
)
SOURCE = "train.parquet"
SEED = 0
SHOT_COUNT = 10
MAX_NEW_TOKENS = 20
STOP = "\n\n"
SHARD_SIZE = 1024
FIG2_TASKS = ("antonym-spanish", "plus-hundred-times-two", "park-country-capital", "book-author-birthyear")
FIG2_SIZE = 100

# Counts and random sample inputs from Table 1. The release already has the shuffled order.
TASKS = {
    "antonym-spanish": (2398, "bogus", True),
    "antonym-german": (2398, "philosophical", True),
    "antonym-french": (2398, "excessive", True),
    "book-author-birthyear": (2228, "The Boy in the Striped Pyjamas", False),
    "song-artist-birthyear": (958, "Heartbreak Hotel", False),
    "landmark-country-capital": (1385, "Taq-i Kisra", False),
    "park-country-capital": (743, "Mount Rainier National Park", False),
    "movie-director-birthyear": (2180, "Cape Fear", False),
    "person-university-year": (4992, "Andi Gutmans", False),
    "person-university-founder": (4996, "Ezra Abbot", False),
    "product-company-ceo": (1904, "NES-101", False),
    "product-company-hq": (2276, "Toyota Alphard", False),
    "plus-ten-times-two": (1000, "699", True),
    "plus-hundred-times-two": (1000, "922", True),
    "mod-twenty-times-two": (1000, "891", True),
    "word-int-times-two": (1000, "one hundred and forty-eight", True),
    "word-substring-reverse": (2946, "responsible", True),
    "rgb-rot120-name": (1000, "8a735a", True),
}
FIELDS = ("x", "Fx", "Gx", "GFx", "FGx")
BASE_HOPS = (("x", "Fx"), ("Fx", "GFx"))
REVERSE_HOPS = (("x", "Gx"), ("Gx", "GFx"))
COMPOSITION = ("x", "GFx")
NUMERIC_TASKS = {"plus-ten-times-two", "plus-hundred-times-two", "mod-twenty-times-two"}
YEAR_TASKS = {"book-author-birthyear", "song-artist-birthyear", "movie-director-birthyear",
              "person-university-year"}


def hops(task):
    primitives = BASE_HOPS + REVERSE_HOPS if TASKS[task][2] else BASE_HOPS
    return primitives + (COMPOSITION,)


EXPECTED_STAGE = sum(n * len(hops(task)) for task, (n, _, _) in TASKS.items())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=3, mtime=0) as zipped:
                zipped.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def data_file(root):
    path = root / SOURCE
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, prefix=".dataset-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urllib.request.urlopen(DATA_URL, timeout=60) as response:
                    stream.write(response.read(874326))
                stream.flush()
                if sha256_file(temporary) != DATA_SHA256:
                    raise RuntimeError("Downloaded Parquet differs from the pinned release")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    if sha256_file(path) != DATA_SHA256:
        raise RuntimeError("Local Parquet differs from the pinned release")
    return path


def read_dataset(path):
    import pyarrow.parquet as parquet
    data = {task: [] for task in TASKS}
    table = parquet.read_table(path, columns=["task", *FIELDS])
    for row in table.to_pylist():
        task = row.pop("task")
        data[task].append(row)
    for task, (count, expected_example, _) in TASKS.items():
        rows = data[task]
        if len(rows) != count or len({r["x"] for r in rows}) != count:
            raise ValueError(f"Unexpected count or duplicate query in {task}")
        if random.Random(SEED).sample(rows, 1)[0]["x"] != expected_example:
            raise ValueError(f"Table 1 sample differs for {task}")
        required = {node for hop in hops(task) for node in hop}
        if any(not all(isinstance(row[field], str) and row[field] for field in required) for row in rows):
            raise ValueError(f"Missing required hop node in {task}")
    return data


def sample_contexts(rows):
    """Match the authors' random.choice rejection sampling in released row order."""
    values = [frozenset(row.values()) - {None} for row in rows]
    indices = range(len(rows))
    rng = random.Random(SEED)
    contexts = []
    for index, query_values in enumerate(values):
        selected = []
        seen = {index}
        if len(rows) < SHOT_COUNT + 1:
            raise ValueError("Insufficient candidates for ten demonstrations")
        collisions = 0
        while len(selected) < SHOT_COUNT:
            demo = rng.choice(indices)
            if demo not in seen and query_values.isdisjoint(values[demo]):
                selected.append(demo)
                seen.add(demo)
            else:
                collisions += 1
                if collisions == 10000 and sum(query_values.isdisjoint(v) for v in values) < SHOT_COUNT:
                    raise ValueError(f"Insufficient disjoint demonstrations for row {index}")
        contexts.append(selected)
    return contexts


def freeze_data(root):
    dataset = read_dataset(data_file(root))
    manifest_path = root / "data.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        verify_files(root, manifest["files"])
    else:
        queries = {
            task: {
                "demos": sample_contexts(rows),
                "full": random.Random(SEED).sample(range(len(rows)), len(rows)),
                "fig2": random.Random(SEED).sample(range(len(rows)), FIG2_SIZE) if task in FIG2_TASKS else [],
            }
            for task, rows in dataset.items()
        }
        query_path = root / "queries.json.gz"
        write_gzip(query_path, queries)
        manifest = {
            "paper_version": PAPER_VERSION, "upstream_code": CODE_REVISION,
            "dataset_revision": DATA_REVISION, "dataset_url": DATA_URL,
            "files": {SOURCE: DATA_SHA256, query_path.name: sha256_file(query_path)},
            "seed": SEED, "shots": SHOT_COUNT, "tasks": {task: len(rows) for task, rows in dataset.items()},
            "fig2_examples": {task: queries[task]["fig2"] for task in FIG2_TASKS},
            "requests_per_stage": EXPECTED_STAGE,
        }
        atomic_json(manifest_path, manifest)
    return dataset, read_gzip(root / "queries.json.gz"), manifest


def verify_queries(dataset, queries):
    for task, rows in dataset.items():
        spec = queries[task]
        size = len(rows)
        if spec["demos"] != sample_contexts(rows):
            raise ValueError(f"{task}: demonstrations differ from upstream sampling")
        if spec["full"] != random.Random(SEED).sample(range(size), size):
            raise ValueError(f"{task}: full query order differs")
        expected = random.Random(SEED).sample(range(size), FIG2_SIZE) if task in FIG2_TASKS else []
        if spec["fig2"] != expected:
            raise ValueError(f"{task}: Fig. 2 query selection differs")


def leading_space(task, output):
    if task in NUMERIC_TASKS or (task in YEAR_TASKS and output == "GFx"):
        return False
    return not (task == "word-int-times-two" and output in {"Fx", "GFx", "FGx"})


def prompt_parts(task, hop, query, demonstrations):
    source, target = hop
    if len(demonstrations) != SHOT_COUNT:
        raise ValueError("Expected exactly ten demonstrations")
    block = "".join(f"Q: {row[source]}\nA: {row[target]}\n\n" for row in demonstrations)
    lead = leading_space(task, target)
    return block + f"Q: {query[source]}", "A:" if lead else "A: ", (
        (" " if lead else "") + query[target]
    )


def render_prompts(tokenizer, stage, parts):
    if stage == "base":
        return [user + "\n" + prefix for user, prefix, _ in parts]
    if not tokenizer.chat_template:
        raise ValueError(f"Native chat template missing for {stage}")
    prompts = tokenizer.apply_chat_template(
        [[{"role": "user", "content": user}] for user, _, _ in parts],
        tokenize=False, add_generation_prompt=True,
    )
    suffix = "<|assistant|>\n"
    if any(not prompt.endswith(suffix) for prompt in prompts):
        raise ValueError(f"Unexpected native assistant opening for {stage}")
    return [prompt + prefix for prompt, (_, prefix, _) in zip(prompts, parts, strict=True)]


def encode(tokenizer, stage, parts):
    rendered = render_prompts(tokenizer, stage, parts)
    ids = tokenizer(rendered, add_special_tokens=stage == "base", padding=False,
                    truncation=False, return_attention_mask=False)["input_ids"]
    max_context = min(tokenizer.model_max_length, 131072)
    for row, (_, prefix, _) in zip(ids, parts, strict=True):
        if not row or len(row) + MAX_NEW_TOKENS > max_context:
            raise ValueError("Frozen prompt outside the native context window")
        tail = tokenizer.decode(row, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        expected_suffix = prefix if stage == "base" else "<|assistant|>\n" + prefix
        if not tail.endswith(expected_suffix):
            raise ValueError(f"Tokenizer did not preserve the answer boundary: {tail[-30:]!r}")
    return ids


def label_for(task, hop, item):
    field = hop[1]
    return (" " if leading_space(task, field) else "") + item[field]


class Decoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.single = lru_cache(maxsize=None)(tokenizer.decode)

    def __call__(self, ids):
        raw = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        tokenwise = "".join(self.single(token) for token in ids)
        return raw, raw.split(STOP, 1)[0], tokenwise.split(STOP, 1)[0]


def outcomes(flags):
    return (any(flags[:-1]), all(flags[:-1]), bool(flags[-1]), all(flags))


def rates(counts, total):
    any_hop, all_hops, composition, both = counts
    return {
        "n": total, "any_hop_n": any_hop, "all_hops_n": all_hops,
        "composition_n": composition, "all_hops_and_composition_n": both,
        "any_hop": any_hop / total, "all_hops": all_hops / total,
        "composition": composition / total, "all_hops_and_composition": both / total,
        "absolute_gap": (all_hops - both) / total,
        "composition_given_all_hops": both / all_hops if all_hops else None,
        "gap": (all_hops - both) / all_hops if all_hops else None,
    }
