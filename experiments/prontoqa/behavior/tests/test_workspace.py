"""Portable run identity, source preservation and explicit reference boundaries."""
import argparse
import json
import shutil
from pathlib import Path
import pytest
from tulu_behavior import common


@pytest.fixture(autouse=True)
def tiny_publication(monkeypatch, tmp_path):
    """Exercise external corpus identity with a small independently hashed archive."""
    original_repo = common.REPO
    repo = tmp_path / 'publication'
    for folder in ('tulu_behavior', 'vendor', 'behavior/tests', 'provenance'):
        shutil.copytree(original_repo / folder, repo / folder)
    for name in ('requirements.txt', 'requirements-analysis.txt', 'behavior/experiment.json'):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_repo / name, target)
    corpus = repo / 'data/items.jsonl'
    corpus.parent.mkdir()
    shutil.copyfile(original_repo / 'behavior/tests/fixtures/original-items.jsonl', corpus)
    evidence = repo / 'behavior/provenance/original_data.json'
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps({'files': {'data/items.jsonl': common.sha256_file(corpus)}}))
    monkeypatch.setattr(common, 'REPO', repo)
    monkeypatch.setattr(common, 'ROOT', repo / 'behavior')


def arguments(tmp_path):
    return argparse.Namespace(root=None, output_dir=tmp_path/'run', model_dir=tmp_path/'models')


def test_fresh_workspace_uses_configurable_models_and_reuses_verified_inputs(tmp_path):
    args = arguments(tmp_path)
    root = common.workspace(args)
    config = common.load_config(root)
    assert [m['path'] for m in config['models']] == [str(args.model_dir/s) for s in common.STAGES]
    assert common.workspace(args) == root
    assert common.sha256_file(root/'original/items.jsonl') == common.sha256_file(common.REPO/'data/items.jsonl')
    assert (root/'vendor/prontoqa/LICENSE').is_file()
    assert not (root/'outputs').exists()
    assert not any(key in config for key in ('remote_root','remote_python'))
    (root/'original/items.jsonl').write_text('corrupted input')
    with pytest.raises(RuntimeError,match='changed recorded artifact'):
        common.workspace(args)


def test_model_directory_and_source_drift_cannot_mix_runs(tmp_path):
    args = arguments(tmp_path)
    root = common.workspace(args)
    args.model_dir = tmp_path/'different-models'
    with pytest.raises(ValueError,match='changed'):
        common.workspace(args)
    args.model_dir = tmp_path/'models'
    (root/'tulu_behavior/scoring.py').write_text('# Changed scientific source\n')
    with pytest.raises(ValueError,match='source changed'):
        common.workspace(args)


@pytest.mark.parametrize('target',['behavior', '.', 'behavior/new', 'tulu_behavior/new'])
def test_reference_and_source_tree_are_never_run_outputs(tmp_path,target):
    args=arguments(tmp_path);args.output_dir=common.REPO/target
    with pytest.raises(ValueError,match='independent'):
        common.workspace(args)


def test_nonempty_unknown_directory_is_preserved(tmp_path):
    args=arguments(tmp_path);args.output_dir.mkdir();existing=args.output_dir/'unique.txt';existing.write_text('Keep me')
    with pytest.raises(ValueError,match='nonempty'):
        common.workspace(args)
    assert existing.read_text()=='Keep me'


def test_external_corpus_must_match_historical_hash(tmp_path):
    args = arguments(tmp_path)
    args.original_items = tmp_path / 'changed-items.jsonl'
    args.original_items.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match='historical SHA-256'):
        common.workspace(args)
