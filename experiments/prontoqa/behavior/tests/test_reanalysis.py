"""Full external evidence is read-only; reanalysis binds new code separately."""
import json
import pytest
from behavior.tests.test_verify import experiment
from tulu_behavior.common import sha256_file
from tulu_behavior.analyze import analyze
from tulu_behavior.reanalysis import reanalyze


def test_external_reanalysis_preserves_source_and_matches_all_statistics(experiment,tmp_path):
    root=experiment
    analyze(root)
    before={str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file()}
    output=tmp_path.parent/(tmp_path.name+'-analysis')
    result=reanalyze(root,output)
    assert result['rescored_generations']==42
    assert result['saved_scoring_disagreements']==0
    assert all(result['comparison']['tables'].values())
    assert before=={str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file()}
    assert not result['model_assets_checked']
    assert (output/'reports/logical_results.jsonl').is_file()
    marker=json.loads((output/'manifests/reanalysis_workspace.json').read_text())
    assert marker['source_protocol_digest']==result['historical_protocol_digest']
    assert marker['analysis_code']==result['publication_code_files']


def test_stale_presentation_is_recorded_without_repair(experiment,tmp_path):
    root=experiment
    analyze(root)
    page=root/'reports/report.html';page.write_text('A separately edited presentation page')
    result=reanalyze(root,tmp_path.parent/(tmp_path.name+'-stale-analysis'))
    assert result['comparison']['status']=='passed'
    assert not result['comparison']['historical_presentation_page_matches_recorded_hash']
    assert page.read_text()=='A separately edited presentation page'


def test_changed_saved_statistics_prevent_acceptance(experiment,tmp_path):
    root=experiment
    analyze(root)
    path=root/'reports/primary_tests.csv';path.write_text('Corrupt summary')
    with pytest.raises(RuntimeError,match='changed recorded artifact'):
        reanalyze(root,tmp_path.parent/(tmp_path.name+'-bad-analysis'))


def test_source_cannot_be_output(experiment):
    root=experiment
    with pytest.raises(ValueError,match='separate output'):
        reanalyze(root,root)
