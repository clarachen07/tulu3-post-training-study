"""Aggregate verification detects altered counts, tests, chart coordinates and leaks."""
import copy
import json
from pathlib import Path
import re
import shutil
import pytest
from tulu_behavior.common import ROOT,read_jsonl
from tulu_behavior.publication import check_statistics,check_figures,reference_files,published_corpus


@pytest.fixture
def summary():
    return json.loads((ROOT/'reports/analysis.json').read_text())


def test_published_aggregate_matrix_matches_corpus(summary):
    assert len(check_statistics(summary,published_corpus(ROOT)))==128
    check_figures(summary)


@pytest.mark.parametrize('alter',["transition","holm","base_alias"])
def test_semantically_changed_aggregates_are_rejected(summary,alter):
    data=copy.deepcopy(summary)
    if alter=='transition':
        row=data['transitions'][0];row['both_correct']+=1;row['both_wrong']-=1
    elif alter=='holm':
        data['primary_tests'][0]['p_holm']=.99
    else:
        row=next(r for r in data['summaries'] if r['stage']=='base' and r['template']=='native' and r['hop']==1)
        row['estimate']+=1/1440
    with pytest.raises(RuntimeError):
        check_statistics(data,published_corpus(ROOT))


def test_rendered_svg_coordinate_change_is_rejected(summary,tmp_path):
    shutil.copytree(ROOT/'reports/figures',tmp_path/'reports/figures')
    path=tmp_path/'reports/figures/stage_comparison.svg'
    text=path.read_text()
    match=re.search(r'(<g id="stage_answer_correct">\s*<path d="M )([0-9.]+) ([0-9.]+)',text)
    assert match is not None
    text=text[:match.start(3)]+str(float(match.group(3))+5)+text[match.end(3):]
    path.write_text(text)
    with pytest.raises(RuntimeError,match='SVG coordinates differ'):
        check_figures(summary,tmp_path)


def test_publication_rejects_raw_and_runtime_artifacts(tmp_path):
    leak=tmp_path/'reports/logical_results.jsonl';leak.parent.mkdir();leak.write_text('{}\n')
    with pytest.raises(RuntimeError,match='allowlist'):
        reference_files(tmp_path)
