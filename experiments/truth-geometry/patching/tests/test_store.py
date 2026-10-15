"""Persistence tests exercise restart safety without depending on filenames."""

import json

import pytest

from truth_patching.store import PointStore


@pytest.fixture
def manifest():
    return {"signature": "fixed-protocol-and-model-revision", "stage": "base"}


def point(manifest, layer=1, position=4):
    return {
        "signature": manifest["signature"],
        "stage": manifest["stage"],
        "layer": layer,
        "position": position,
        "scores": {"true": 0.25, "false": -0.75, "difference": 1.0},
    }


def find_json(directory, value):
    matches = [
        path
        for path in directory.rglob("*.json")
        if json.loads(path.read_text()) == value
    ]
    assert len(matches) == 1, "Expected one persisted JSON document with this value"
    return matches[0]


def test_resume_recovers_all_completed_points(tmp_path, manifest):
    store = PointStore(tmp_path, manifest)
    assert store.completed == set()
    records = [point(manifest), point(manifest, layer=2, position=5)]
    for record in records:
        store.write(record)
    assert store.completed == {(1, 4), (2, 5)}

    resumed = PointStore(tmp_path, manifest)
    assert resumed.completed == {(1, 4), (2, 5)}
    find_json(tmp_path, manifest)
    for record in records:
        find_json(tmp_path, record)


def test_duplicate_key_does_not_overwrite_or_rewrite(tmp_path, manifest):
    store = PointStore(tmp_path, manifest)
    original = point(manifest)
    store.write(original)
    path = find_json(tmp_path, original)
    original_bytes = path.read_bytes()
    original_stat = path.stat()
    store.write(dict(original))
    assert store.completed == {(1, 4)}
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert path.stat().st_ino == original_stat.st_ino


def test_conflicting_duplicate_is_rejected_without_overwriting(tmp_path, manifest):
    store = PointStore(tmp_path, manifest)
    original = point(manifest)
    store.write(original)
    path = find_json(tmp_path, original)
    original_bytes = path.read_bytes()
    duplicate = {**original, "scores": {"true": 7.0, "false": 0.0, "difference": 7.0}}
    with pytest.raises(ValueError):
        store.write(duplicate)
    assert path.read_bytes() == original_bytes
    assert store.completed == {(1, 4)}


def test_existing_manifest_signature_mismatch_is_rejected(tmp_path, manifest):
    PointStore(tmp_path, manifest).write(point(manifest))
    with pytest.raises(ValueError):
        PointStore(tmp_path, {**manifest, "signature": "different-protocol"})
    assert PointStore(tmp_path, manifest).completed == {(1, 4)}


def test_wrong_signature_write_is_rejected_without_marking_complete(tmp_path, manifest):
    store = PointStore(tmp_path, manifest)
    wrong = {**point(manifest), "signature": "different-protocol"}
    with pytest.raises(ValueError):
        store.write(wrong)
    assert store.completed == set()
    assert PointStore(tmp_path, manifest).completed == set()


def test_truncated_point_is_rejected_on_resume(tmp_path, manifest):
    record = point(manifest)
    PointStore(tmp_path, manifest).write(record)
    path = find_json(tmp_path, record)
    path.write_text('{"signature": "fixed-protocol-and-model-revision",')
    with pytest.raises(ValueError):
        PointStore(tmp_path, manifest)


def test_wrong_signature_in_existing_point_is_rejected(tmp_path, manifest):
    record = point(manifest)
    PointStore(tmp_path, manifest).write(record)
    path = find_json(tmp_path, record)
    path.write_text(json.dumps({**record, "signature": "another-model"}))
    with pytest.raises(ValueError):
        PointStore(tmp_path, manifest)


def test_truncated_manifest_is_rejected_on_resume(tmp_path, manifest):
    PointStore(tmp_path, manifest)
    path = find_json(tmp_path, manifest)
    path.write_text('{"signature":')
    with pytest.raises(ValueError):
        PointStore(tmp_path, manifest)


def test_failed_atomic_commit_is_not_marked_complete(tmp_path, manifest, monkeypatch):
    store = PointStore(tmp_path, manifest)

    def interrupted_replace(source, destination):
        raise OSError("Simulated interruption before atomic rename")

    with monkeypatch.context() as context:
        context.setattr("truth_patching.store.os.replace", interrupted_replace)
        with pytest.raises(OSError):
            store.write(point(manifest))
    assert store.completed == set()
    resumed = PointStore(tmp_path, manifest)
    assert resumed.completed == set()
    resumed.write(point(manifest))
    assert PointStore(tmp_path, manifest).completed == {(1, 4)}
