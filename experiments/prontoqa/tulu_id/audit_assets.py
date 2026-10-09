"""Verify local model snapshots against pinned scientific asset hashes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .common import (ROOT, atomic_json, load_config, sha256_file, utc_now,
                     add_workspace_arguments, workspace)


def fingerprint(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                inode=stat.st_ino, device=stat.st_dev, resolved=str(Path(path).resolve()))


def check_assets(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'manifests/assets.json').read_text())
    if manifest['config_hash'] != sha256_file(root / 'experiment.json'):
        raise RuntimeError('Model audit belongs to another configuration')
    registry = {m['stage']: m for m in json.loads((root / 'provenance/models.json').read_text())['models']}
    for model in manifest['models']:
        expected = registry[model['stage']]
        if model['id'] != expected['id'] or model['revision'] != expected['revision']:
            raise RuntimeError('Model revision identity changed')
        paths = [Path(item['path']) for item in model['files']]
        directory = paths[0].parent
        candidates = sorted(p.name for p in directory.iterdir() if p.name.endswith(('.safetensors', '.safetensors.index.json')))
        wanted = sorted(name for name in expected['files'] if name.endswith(('.safetensors', '.safetensors.index.json')))
        if candidates != wanted:
            raise RuntimeError(f'Weight loader candidates changed: {directory}')
        if {p.name for p in paths} != set(expected['files']):
            raise RuntimeError('Incomplete model asset inventory')
        for item in model['files']:
            if item['sha256'] != expected['files'][Path(item['path']).name] or fingerprint(item['path']) != item['stat']:
                raise RuntimeError(f"Model asset changed: {item['path']}")
    return manifest


def audit(root):
    root = Path(root)
    if (root / 'manifests/assets.json').exists():
        return check_assets(root)
    registry = {m['stage']: m for m in json.loads((root / 'provenance/models.json').read_text())['models']}
    out = dict(created_utc=utc_now(), config_hash=sha256_file(root / 'experiment.json'), models=[])
    for model in load_config(root)['models']:
        expected = registry[model['stage']]
        if any(model[k] != expected[k] for k in ['id', 'revision', 'wrapper']):
            raise RuntimeError('Configured model differs from the pinned release')
        files = []
        for name, value in expected['files'].items():
            path = Path(model['path']) / name
            before = fingerprint(path)
            if sha256_file(path) != value or fingerprint(path) != before:
                raise RuntimeError(f'Pinned model asset mismatch: {name}')
            files.append(dict(path=str(path), sha256=value, stat=before))
        cfg = json.loads((Path(model['path']) / 'config.json').read_text())
        if cfg['num_hidden_layers'] != 32 or cfg['hidden_size'] != 4096:
            raise RuntimeError('Unexpected model architecture')
        out['models'].append(dict(stage=model['stage'], id=model['id'], revision=model['revision'], files=files))
        print(f"{model['stage']}: pinned weights, tokenizer and configuration verified", flush=True)
    atomic_json(root / 'manifests/assets.json', out)
    return check_assets(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_arguments(parser)
    audit(workspace(parser.parse_args()))


if __name__ == '__main__':
    main()
