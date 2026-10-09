import json
import random
import re
import subprocess
import tomllib
import urllib.request
import urllib.error

from .common import ROOT, author_path, config, digest, write_json, write_jsonl

FORMAT_RULE = "Output the three task tables in order, each preceded by its task marker: - Task 1, - Task 2, - Task 3. Separate fields with literal tab characters. Do not add column headers, Markdown tables, code fences, or explanatory prose."
UPOS_TAGS = "ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X"
UPOS_RULE = "For Task 1, use only these Universal Dependencies UPOS tags: " + UPOS_TAGS + "."


def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as f:
            return f.read()
    except urllib.error.URLError:
        # macOS Python's TLS stack can differ from the system curl stack.
        return subprocess.check_output(["curl", "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "15", "--max-time", "120", url])


def main():
    cfg = config()
    commit = subprocess.check_output(["git", "-C", str(ROOT / "third_party/llmpp"), "rev-parse", "HEAD"], text=True).strip()
    assert commit == cfg["author_commit"]
    author_path()
    from llmpp.conllu_to_prompt import convert_lines
    template_bytes = (ROOT / "third_party/llmpp/templates/en-turn1-step3.toml").read_bytes()
    template = tomllib.loads(template_bytes.decode())
    ref = json.loads(fetch("https://api.github.com/repos/UniversalDependencies/UD_English-EWT/git/ref/tags/r2.15"))
    revision = ref["object"]["sha"]
    if ref["object"]["type"] == "tag":
        revision = json.loads(fetch("https://api.github.com/repos/UniversalDependencies/UD_English-EWT/git/tags/" + revision))["object"]["sha"]
    manifests = {}
    datasets = {}
    for split in ["train", "dev", "test"]:
        name = f"en_ewt-ud-{split}.conllu"
        url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{revision}/{name}"
        raw = fetch(url)
        p = ROOT / "data" / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(raw)
        blocks = raw.decode().strip().split("\n\n")
        converted = convert_lines(raw.decode().splitlines(True), template["add_whitespace"])
        assert len(blocks) == len(converted)
        rows = []
        for ordinal, (block, item) in enumerate(zip(blocks, converted)):
            sid = re.search(r"^# sent_id = (.+)$", block, re.M).group(1)
            gold = []
            for line in block.splitlines():
                fields = line.split("\t")
                if fields[0].isdigit():
                    gold.append({"id": int(fields[0]), "form": fields[1], "upos": fields[3], "head": int(fields[6]), "deprel": fields[7], "space_after": "SpaceAfter=No" not in fields[9]})
            n = len(gold)
            assert [t["id"] for t in gold] == list(range(1, n + 1))
            tsv = {}
            for key, fields in [("ORTH", ["orth"]), ("INDEX_ORTH_UPOS", ["id", "orth", "upos"]), ("INDEX_ORTH_UPOS_HEAD", ["id", "orth", "upos", "head"]), ("INDEX_ORTH_UPOS_HEAD_LABEL", ["id", "orth", "upos", "head", "label"])]:
                lines = []
                for t in item["tokens"]:
                    values = {**t, "id": t["id"] + 1, "head": 0 if t["label"] == "root" else t["head"] + 1}
                    lines.append("\t".join(str(values[f]) for f in fields))
                tsv[key] = "\n".join(lines)
            replacements = {"<<<LANGUAGE>>>": "English", "<<<SENTENCE>>>": item["sentence"], "<<<TOKEN_NUM>>>": str(n)}
            replacements.update({f"<<<TOKEN_TSV:{k}>>>": v for k, v in tsv.items()})
            messages = []
            for m in template["message_template"]:
                content = m["content"]
                for key, val in replacements.items():
                    content = content.replace(key, val)
                assert "<<<" not in content
                messages.append({"role": m["role"], "content": content})
            rows.append({"sent_id": sid, "ordinal": ordinal, "split": split, "text": item["sentence"], "messages": messages[:2], "gold_output": messages[2]["content"], "gold": gold})
        datasets[split] = rows
        write_jsonl(ROOT / "data" / f"{split}.jsonl", rows)
        manifests[split] = {"url": url, "sha256": digest(raw), "sentences": len(rows), "words": sum(len(x["gold"]) for x in rows)}
    assert manifests["test"]["sentences"] == 2077 and manifests["test"]["words"] == 25094
    ids = sorted(random.Random(cfg["seed"]).sample(range(len(datasets["dev"])), cfg["dev_sample_size"]))
    write_jsonl(ROOT / "data/dev128.jsonl", [datasets["dev"][i] for i in ids])
    longest = max(datasets["dev"], key=lambda x: len(x["gold"]))
    # Fixed preflight mix, including longest sentence; independent of model behavior.
    probe = [datasets["dev"][ids[i]] for i in [0, 32, 64, 96]] + [longest]
    probe = list({x["sent_id"]: x for x in probe}.values())
    write_jsonl(ROOT / "data/preflight.jsonl", probe)
    eligible = [s for s in datasets["train"] if 6 <= len(s["gold"]) <= 12]
    example = max(eligible, key=lambda x: (len({t["upos"] for t in x["gold"]}), -x["ordinal"]))
    write_json(ROOT / "data/example.json", example)
    write_json(ROOT / "manifests/data.json", {"dataset_revision": revision, "author_commit": commit, "template_sha256": digest(template_bytes), "splits": manifests, "dev128_sent_ids": [datasets["dev"][i]["sent_id"] for i in ids], "preflight_sent_ids": [x["sent_id"] for x in probe], "example_sent_id": example["sent_id"]})
    write_json(ROOT / "manifests/prompts.json", {"P0": {"addition": "", "example": None}, "P1": {"addition": FORMAT_RULE, "example": None}, "P2": {"addition": FORMAT_RULE + "\n" + UPOS_RULE, "example": None}, "P3": {"addition": FORMAT_RULE + "\n" + UPOS_RULE, "example": example["sent_id"]}})
    for name in ["LICENSE.txt", "README.md"]:
        try:
            (ROOT / "data" / ("EWT-" + name)).write_bytes(fetch(f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{revision}/{name}"))
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    print(json.dumps(manifests, indent=2), flush=True)


def build_messages(sentence, candidate):
    specs = json.loads((ROOT / "manifests/prompts.json").read_text())
    spec = specs[candidate]
    messages = [dict(x) for x in sentence["messages"]]
    if spec["addition"]:
        messages[-1]["content"] += "\n" + spec["addition"]
    if candidate == "P3":
        example = json.loads((ROOT / "data/example.json").read_text())
        demo = dict(example["messages"][1])
        demo["content"] += "\n" + spec["addition"]
        messages = [messages[0], demo, {"role": "assistant", "content": example["gold_output"]}, messages[1]]
    assert messages[-1]["role"] == "user"
    return messages


if __name__ == "__main__":
    main()
