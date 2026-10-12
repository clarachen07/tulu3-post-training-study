import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "experiment.json").read_text())


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_results(path):
    """Only retry infrastructure failures. A crash-torn final line is ignored."""
    path = Path(path)
    if not path.exists():
        return {}
    result = {}
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise
        sid = row["sent_id"]
        if sid in result and result[sid].get("status") == "ok":
            raise ValueError(f"duplicate successful inference: {sid}")
        result[sid] = row
    return result


def author_path():
    import sys
    p = str(ROOT / "third_party" / "llmpp")
    if p not in sys.path:
        sys.path.insert(0, p)
