import hashlib
import json
import subprocess
import time
import argparse

from .common import ROOT, config, digest, write_json
from .prepare import fetch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-only", action="store_true", help="Fetch public official hashes locally for offline remote verification")
    args = ap.parse_args()
    cfg = config()
    expected_path = ROOT / "manifests/official-models.json"
    if args.fetch_only or not expected_path.exists():
        official = {}
        for model in cfg["models"]:
            official[model["stage"]] = json.loads(fetch(f"https://huggingface.co/api/models/{model['id']}/revision/{model['revision']}?blobs=true"))
        write_json(expected_path, official)
        if args.fetch_only:
            return
    official = json.loads(expected_path.read_text())
    results = []
    for model in cfg["models"]:
        metadata = official[model["stage"]]
        assert metadata["sha"] == model["revision"]
        files = []
        from pathlib import Path
        for entry in metadata["siblings"]:
            name = entry["rfilename"]
            if not (name.endswith(".safetensors") or name in ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json", "special_tokens_map.json"]):
                continue
            path = Path(model["path"]) / name
            assert path.is_file(), f"missing model file {path}"
            if "lfs" in entry:
                expected = entry["lfs"]["sha256"]
                h = hashlib.sha256()
                with path.open("rb") as f:
                    while block := f.read(16 * 1024 * 1024):
                        h.update(block)
                actual = h.hexdigest()
                assert actual == expected, f"weight mismatch: {path}"
                files.append({"file": name, "sha256": actual, "bytes": path.stat().st_size})
            else:
                raw = path.read_bytes()
                actual_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                assert actual_blob == entry["blobId"], f"config/tokenizer mismatch: {path}"
                files.append({"file": name, "sha256": digest(raw), "git_blob": actual_blob, "bytes": path.stat().st_size})
            print(f"verified {model['stage']} {name}", flush=True)
        results.append({**model, "files": files, "verified": True})
        write_json(ROOT / "manifests/weights-progress.json", results)
    code = {}
    for p in sorted((ROOT / "third_party/llmpp").rglob("*")):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts:
            code[str(p.relative_to(ROOT))] = digest(p.read_bytes())
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT / "third_party/llmpp"), "rev-parse", "HEAD"], text=True).strip()
    assert actual_commit == cfg["author_commit"]
    assert not subprocess.check_output(["git", "-C", str(ROOT / "third_party/llmpp"), "status", "--porcelain"], text=True).strip()
    write_json(ROOT / "manifests/weights.json", {"verified": True, "verified_unix": time.time(), "models": results, "author_commit": actual_commit, "author_files": code})


if __name__ == "__main__":
    main()
