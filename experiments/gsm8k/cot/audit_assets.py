"""Hash existing assets without changing model directories."""
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from common import ROOT, digest_file, fingerprint, load_json, utc, write_json


def stat_fields(p):
    s = p.stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "inode": s.st_ino, "device": s.st_dev, "resolved": str(p.resolve())}


def main():
    cfg = load_json(ROOT / "experiment.json")
    path = ROOT / "manifests/assets.json"
    if path.exists():
        audit = load_json(path)
        assert audit["config_hash"] == fingerprint(cfg)
        for m in audit["models"]:
            for file in m["files"]:
                assert stat_fields(Path(file["path"])) == file["after"], file["path"]
        print("Existing asset audit and file stability verified", flush=True)
        return
    models = []
    for model in cfg["models"]:
        p = Path(model["path"])
        weights = sorted(p.glob("*.safetensors"))
        assert len(weights) == 4, (model["stage"], len(weights))
        files = weights + sorted(p.glob("*.json"))
        records = []
        for file in files:
            before = stat_fields(file)
            start = time.monotonic()
            sha = digest_file(file)
            after = stat_fields(file)
            assert before == after, "Asset changed while hashing"
            records.append({"path": str(file), "sha256": sha, "before": before, "after": after,
                            "seconds": time.monotonic() - start})
            write_json(ROOT / "status/asset_progress.json", {"utc": utc(), "stage": model["stage"], "file": file.name})
            print(model["stage"], file.name, sha, flush=True)
        c = load_json(p / "config.json")
        assert c["num_hidden_layers"] == 32 and c["num_attention_heads"] == 32
        models.append({"registry": model, "files": records, "config": c,
                       "revision_provenance": "Existing model registry; current full-file SHA256 recorded separately"})
    versions = {}
    for name in ["torch", "vllm", "transformers", "numpy", "tokenizers"]:
        versions[name] = importlib.metadata.version(name)
    write_json(path, {"created_utc": utc(), "config_hash": fingerprint(cfg), "models": models,
                      "versions": versions, "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()})


if __name__ == "__main__":
    main()
