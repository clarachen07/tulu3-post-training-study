"""Small shared I/O and the frozen numerical scoring rule."""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return digest_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")).encode())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def strip_commas(text):
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def extract_answer(continuation):
    numbers = NUMBER.findall(strip_commas(continuation))
    return numbers[-1] if numbers else None


def score(continuation, gold):
    answer = extract_answer(continuation)
    if answer is None:
        return answer, False
    return answer, Decimal(answer) == Decimal(strip_commas(str(gold)).strip())
