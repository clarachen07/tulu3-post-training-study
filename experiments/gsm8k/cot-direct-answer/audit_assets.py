"""Hash existing assets without changing model directories."""
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from common import local_path, ROOT, digest_file, fingerprint, load_json, utc, write_json


def stat_fields(p):
    s = p.stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "inode": s.st_ino, "device": s.st_dev, "resolved": str(p.resolve())}


def main():
    cfg = load_json(ROOT / "experiment.json")
    reference = load_json(ROOT / 'reference/cot/assets.json')
    published = load_json(ROOT / 'reference/cot/published_identity.json')
    assert published['status'] == 'passed'
    path = ROOT / "manifests/assets.json"
    if path.exists():
        audit = load_json(path)
        assert audit["config_hash"] == fingerprint(cfg)
        assert audit['versions'] == reference['versions'], 'Inference environment changed'
        for name, version in audit['versions'].items():
            assert importlib.metadata.version(name) == version, name
        for m in audit["models"]:
            for file in m["files"]:
                assert stat_fields(Path(file["path"])) == file["after"], file["path"]
        print("Existing asset audit and file stability verified", flush=True)
        return
    models = []
    for model in cfg["models"]:
        p = local_path(model["path"])
        weights = sorted(p.glob("*.safetensors"))
        assert len(weights) == 4, (model["stage"], len(weights))
        files = weights + sorted(p.glob("*.json"))
        old = next(m for m in reference['models'] if m['registry']['stage'] == model['stage'])
        expected = {Path(f['path']).name: f['sha256'] for f in old['files']}
        assert {f.name for f in files} == set(expected), 'Model asset file set changed'
        records = []
        for file in files:
            before = stat_fields(file)
            start = time.monotonic()
            sha = digest_file(file)
            after = stat_fields(file)
            assert before == after, "Asset changed while hashing"
            assert sha == expected[file.name], 'Historical model/tokenizer identity mismatch: ' + str(file)
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
    assert versions == reference['versions'], ('Inference environment differs from historical baseline', versions)
    for model in models:
        identity = next(m for m in published['models'] if m['stage'] == model['registry']['stage'])
        assert model['registry']['revision'] == identity['revision']
        assert sorted(f['sha256'] for f in model['files'] if f['path'].endswith('.safetensors')) == sorted(identity['published_sha256'])
    write_json(ROOT / 'manifests/published_identity.json', published)
    write_json(ROOT / 'manifests/runtime.json', {'created_utc': utc(),
        'runtime': load_json(ROOT / 'runtime.json'),
        'model_paths': {m['stage']: m['path'] for m in cfg['models']},
        'historical_model_paths': {m['registry']['stage']: m['registry']['path'] for m in reference['models']},
        'interpreter': __import__('sys').executable, 'versions': versions,
        'historical_versions_match': True, 'model_and_tokenizer_hashes_match': True})
    write_json(path, {"created_utc": utc(), "config_hash": fingerprint(cfg), "models": models,
                      "versions": versions, "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()})


if __name__ == "__main__":
    main()
