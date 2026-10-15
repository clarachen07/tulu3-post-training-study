"""Regression coverage for run isolation and retained scientific evidence."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiment_utils import runtime
from experiment_utils.assets import main as preflight_main
from prepare_sources import prepare


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    real = runtime.PROJECT
    for unit in ("patching", "pca"):
        for relative in (
            "configs/experiment.json",
            "configs/model_assets_expected.json",
            "third_party/source_manifest.json",
        ):
            dest = root / unit / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((real / unit / relative).read_bytes())
        package = "truth_patching" if unit == "patching" else "truth_pca"
        for directory in (package, "tests", "third_party/geometry-of-truth"):
            (root / unit / directory).mkdir(parents=True, exist_ok=True)
    (root / "experiment_utils").mkdir()
    paths = {}
    for stage in runtime.STAGES:
        directory = tmp_path / "models" / stage
        directory.mkdir(parents=True)
        paths[stage] = str(directory)
    local = tmp_path / "local.json"
    local.write_text(
        json.dumps(
            {
                "model_paths": paths,
                "device": "cuda:0",
                "gpu_lock": str(tmp_path / "gpu.lock"),
            }
        )
    )
    monkeypatch.setattr(runtime, "PROJECT", root)
    return root, local


@pytest.mark.parametrize("unit", ["patching", "pca"])
def test_fresh_run_isolated_and_resume_config_is_immutable(project, unit):
    root, local = project
    before = (root / unit / "configs/experiment.json").read_bytes()
    run = runtime.prepare_run(unit, "replication", local)
    assert run == root / "runs/replication" / unit
    config = json.loads((run / "configs/experiment.json").read_text())
    assert config["protocol"] == json.loads(before)["protocol"]
    assert all(Path(spec["path"]).is_absolute() for spec in config["models"].values())
    assert (root / unit / "configs/experiment.json").read_bytes() == before
    with pytest.raises(RuntimeError, match="already exists"):
        runtime.prepare_run(unit, "replication", local)
    assert runtime.prepare_run(unit, "replication", local, resume=True) == run
    changed = json.loads(local.read_text())
    changed["device"] = "cuda:1"
    local.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="configuration changed"):
        runtime.prepare_run(unit, "replication", local, resume=True)
    assert (
        json.loads((run / "configs/experiment.json").read_text())["runtime"]["device"]
        == "cuda:0"
    )


@pytest.mark.parametrize(
    "relative",
    [
        "patching",
        "patching/results",
        "pca",
        "pca/results/base",
        "figures",
        "reports",
        "provenance",
    ],
)
def test_published_roots_reject_writes(project, relative):
    root, _ = project
    with pytest.raises(RuntimeError, match="read-only"):
        runtime.protect_published(root / relative)


def test_run_name_cannot_escape_workspace(project):
    _, local = project
    with pytest.raises(ValueError):
        runtime.prepare_run("patching", "../patching", local)


def test_direct_preflight_cannot_write_to_published_root(project, monkeypatch):
    root, _ = project
    monkeypatch.setattr("sys.argv", ["preflight", "--root", str(root / "patching")])
    with pytest.raises(RuntimeError, match="read-only"):
        preflight_main()
    assert not (root / "patching/artifacts/preflight.json").exists()


def test_retained_file_corruption_is_rejected(tmp_path):
    path = tmp_path / "observation.json"
    path.write_bytes(b'{"value": 1}')
    (tmp_path / "provenance").mkdir()
    (tmp_path / "provenance/retained_files.json").write_text(
        json.dumps(
            {
                "files": {
                    "observation.json": {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                }
            }
        )
    )
    assert runtime.verify_retained_files(tmp_path)["passed"]
    path.write_bytes(b'{"value": 2}')
    with pytest.raises(RuntimeError, match="Historical files changed"):
        runtime.verify_retained_files(tmp_path)


def source_root(tmp_path, content):
    (tmp_path / "third_party").mkdir()
    manifest = {
        "commit": "frozen",
        "files": {
            "file": {
                "path": "third_party/file.py",
                "url": "https://example.com/file",
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        },
    }
    (tmp_path / "third_party/source_manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "third_party/file.py"


def test_changed_author_file_is_preserved_without_download(tmp_path):
    file = source_root(tmp_path, b"fixed source")
    file.write_bytes(b"user edit")
    with patch("prepare_sources.urllib.request.urlopen") as fetch:
        with pytest.raises(RuntimeError, match="Refusing changed source"):
            prepare(tmp_path)
        fetch.assert_not_called()
    assert file.read_bytes() == b"user edit"


def test_unexpected_download_is_rejected_before_write(tmp_path):
    file = source_root(tmp_path, b"fixed source")
    with patch("prepare_sources.urllib.request.urlopen") as fetch:
        fetch.return_value.__enter__.return_value.read.return_value = (
            b"changed server response"
        )
        with pytest.raises(RuntimeError, match="digest mismatch"):
            prepare(tmp_path)
    assert not file.exists()
