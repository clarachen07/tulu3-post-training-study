"""Versioned preflights and immutable formal protocol identities."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

from .common import ROOT, atomic_json, code_hashes, fingerprint, sha256_file, utc_now, verify_files

PACKAGES = ("numpy", "scipy", "torch", "transformers", "tokenizers", "accelerate", "safetensors", "matplotlib")


def environment():
    return {"python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES}}


def _prepared_files(root):
    files = code_hashes(root)
    for name in ("data", "tokenization", "assets"):
        path = root / f"manifests/{name}.json"
        files[str(path.relative_to(root))] = sha256_file(path)
        manifest = json.loads(path.read_text())
        if name != "assets":
            files.update(manifest.get("files", {}))
    return files


def preflight(root=ROOT):
    root = Path(root)
    if (root / "manifests/frozen_protocol.json").exists():
        validate_frozen(root)
        return validate_preflight(root)
    payload = {"files": _prepared_files(root), "environment": environment()}
    digest = fingerprint(payload)
    snapshot = root / f"manifests/preflights/{digest}.json"
    if snapshot.exists():
        result = json.loads(snapshot.read_text())
        if result["identity"] != payload:
            raise RuntimeError("Preflight digest collision")
    else:
        result = {"identity": payload, "digest": digest, "created_utc": utc_now()}
        atomic_json(snapshot, result)
    current = root / "manifests/preflight.json"
    if not current.exists() or json.loads(current.read_text()) != result:
        atomic_json(current, result)
    return result


def validate_preflight(root=ROOT, check_environment=True):
    root = Path(root)
    result = json.loads((root / "manifests/preflight.json").read_text())
    if fingerprint(result["identity"]) != result["digest"]:
        raise RuntimeError("Invalid preflight fingerprint")
    verify_files(root, result["identity"]["files"])
    if check_environment and environment() != result["identity"]["environment"]:
        raise RuntimeError("Inference environment changed")
    return result


def freeze(root=ROOT):
    root = Path(root)
    target = root / "manifests/frozen_protocol.json"
    if target.exists():
        return validate_frozen(root)
    pre = validate_preflight(root)
    selection_path = root / "manifests/pilot_selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "passed" or selection["preflight_digest"] != pre["digest"]:
        raise RuntimeError("Pilot did not pass for this exact protocol")
    files = dict(pre["identity"]["files"])
    files["manifests/preflight.json"] = sha256_file(root / "manifests/preflight.json")
    files["manifests/pilot_selection.json"] = sha256_file(selection_path)
    for rel, digest in selection.get("evidence_files", {}).items():
        files[rel] = digest
    payload = dict(files=files, environment=pre["identity"]["environment"], preflight_digest=pre["digest"],
                   generation=selection["generation"])
    result = dict(identity=payload, protocol_digest=fingerprint(payload), frozen_utc=utc_now())
    atomic_json(target, result)
    return validate_frozen(root)


def validate_frozen(root=ROOT, check_environment=True):
    root = Path(root)
    result = json.loads((root / "manifests/frozen_protocol.json").read_text())
    if fingerprint(result["identity"]) != result["protocol_digest"]:
        raise RuntimeError("Invalid frozen protocol fingerprint")
    verify_files(root, result["identity"]["files"])
    if check_environment and environment() != result["identity"]["environment"]:
        raise RuntimeError("Frozen inference environment changed")
    return result


def main():
    import argparse
    from .common import add_workspace_arguments, workspace
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_arguments(parser)
    parser.add_argument("--preflight", action="store_true", help="Freeze prepared inputs before the pilot")
    args = parser.parse_args()
    root = workspace(args)
    result = preflight(root) if args.preflight else freeze(root)
    print(json.dumps({"digest": result.get("protocol_digest", result.get("digest"))}))


if __name__ == "__main__":
    main()
