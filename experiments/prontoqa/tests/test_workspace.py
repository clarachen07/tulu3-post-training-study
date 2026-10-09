"""Portable workspace inputs, immutable references and historical identity."""
import argparse
import json
from pathlib import Path
import pytest
from tulu_id import common


@pytest.fixture
def published(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    config = common.load_config(common.ROOT)
    for relative, contents in {
        'experiment.json': json.dumps(config), 'requirements.txt': '',
        'requirements-analysis.txt': '', 'pca/config.json': '{}',
        'tulu_id/example.py': '# Scientific source\n', 'provenance/models.json': '{}',
        'data/items.jsonl': '{"sample_id":"one"}\n',
    }.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    activations = tmp_path / 'release/activations'
    batch = activations / 'base/batch_000000.npz'
    batch.parent.mkdir(parents=True)
    batch.write_bytes(b'hashed activation input')
    scientific = {k: config[k] for k in ['protocol','paper','data','statistics','inference']}
    scientific['models'] = [{k:m[k] for k in ['stage','id','revision','wrapper']} for m in config['models']]
    ledger = dict(original_protocol_digest='historical-forward-identity', scientific_parameters=scientific,
                  input_files={'data/items.jsonl': common.sha256_file(repo / 'data/items.jsonl')},
                  activation_files={'activations/base/batch_000000.npz':common.sha256_file(batch)})
    ledger['identity'] = common.canonical_digest(ledger)
    common.atomic_json(repo / 'provenance/artifacts.json', ledger)
    monkeypatch.setattr(common, 'ROOT', repo)
    args = argparse.Namespace(root=None, output_dir=tmp_path/'run', model_dir=tmp_path/'models',
                              data_dir=repo/'data', activations_dir=activations)
    return repo, args, ledger


def test_published_inputs_copy_separately_and_bind_historical_identity(published):
    repo, args, ledger = published
    work = common.workspace(args, analysis=True)
    assert work != repo
    frozen = common.validate_frozen(work)
    assert frozen['protocol_digest'] == ledger['original_protocol_digest']
    assert frozen['input_identity'] != frozen['protocol_digest']
    assert common.workspace(args, analysis=True) == work
    (work / 'data/items.jsonl').write_text('changed run copy')
    with pytest.raises(RuntimeError, match='Frozen input/code changed'):
        common.validate_frozen(work)
    assert (repo / 'data/items.jsonl').read_text() == '{"sample_id":"one"}\n'


def test_corrupt_release_asset_cannot_commit_a_workspace(published):
    _, args, _ = published
    (args.activations_dir / 'base/batch_000000.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='activation asset changed'):
        common.workspace(args, analysis=True)
    assert not any(args.output_dir.iterdir())


def test_changed_protocol_cannot_reuse_published_activations(published):
    repo, args, _ = published
    config = common.load_config(repo)
    config['statistics']['fit_fraction'] = .8
    common.atomic_json(repo / 'experiment.json', config)
    with pytest.raises(ValueError, match='parameters differ'):
        common.workspace(args, analysis=True)


def test_changed_code_cannot_mix_cached_runs(published):
    repo, args, _ = published
    common.workspace(args, analysis=True)
    (repo / 'tulu_id/example.py').write_text('# Different scientific implementation\n')
    with pytest.raises(ValueError, match='inputs changed'):
        common.workspace(args, analysis=True)
