from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from .common import ROOT, atomic_json, code_files, sha256_file, utc_now, validate_frozen, frozen_identity, add_workspace_arguments, workspace
from .audit_assets import check_assets


def main():
    p = argparse.ArgumentParser(); add_workspace_arguments(p)
    root = workspace(p.parse_args())
    target = root / 'manifests/frozen_protocol.json'
    if target.exists():
        validate_frozen(root); check_assets(root); print('Frozen protocol unchanged'); return
    check_assets(root)
    paths = code_files(root)
    paths += [root / 'manifests/data.json', root / 'manifests/tokenization.json',
              root / 'manifests/assets.json', root / 'provenance/models.json']
    paths += list((root / 'data').rglob('*.jsonl')) + list((root / 'data').glob('*.npy'))
    paths += list((root / 'data/raw').rglob('*.json'))
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(set(paths))}
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'transformers', 'accelerate', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'safetensors']}
    environment = dict(python=platform.python_version(), packages=versions)
    identity = dict(files=files, environment=environment)
    digest = frozen_identity(files, environment)
    atomic_json(target, dict(protocol_digest=digest, created_utc=utc_now(), **identity))
    print(f'Frozen {len(files)} files: {digest}', flush=True)


if __name__ == '__main__': main()
