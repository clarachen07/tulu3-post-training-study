"""Pilot gating: preserve entire panels, shared budgets and failure evidence."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from tulu_behavior.common import ROOT,REPO,load_config,atomic_json
from tulu_behavior.pipeline import Pipeline
from tulu_behavior.pilot import budget_decision


def panel(status='passed',budgets=None):
    return {'decision':{'status':status,'next_budgets':budgets or {'cot':512,'direct':32}},
            'batch_size':4,'budgets':budgets or {'cot':512,'direct':32},'preflight_digest':'pre','evidence_files':{}}


@pytest.fixture
def pipeline(tmp_path):
    atomic_json(tmp_path/'experiment.json',load_config(ROOT))
    return Pipeline(tmp_path)


def test_budget_gate_never_reads_correctness():
    class NoScores(dict):
        def __getitem__(self,key):
            if key=='scoring':raise AssertionError('Correctness must not select budgets')
            return super().__getitem__(key)
    budgets={'cot':512,'direct':32}
    rows=[NoScores(condition=c,budget=b,budget_incomplete=False,scoring={'answer_correct':False}) for c,b in budgets.items()]
    assert budget_decision(rows,budgets,load_config()['inference']['budget_candidates'])['status']=='passed'
    rows[0]['budget_incomplete']=True
    assert budget_decision(rows,budgets,load_config()['inference']['budget_candidates'])['next_budgets']=={'cot':1024,'direct':32}


def test_budget_increase_reruns_every_stage_with_shared_tier(pipeline):
    high={'cot':1024,'direct':32}
    with patch('tulu_behavior.freeze.validate_preflight',return_value={'digest':'pre'}), \
         patch.object(pipeline,'run_module',return_value=0) as run, \
         patch('tulu_behavior.pilot.summarize',side_effect=[panel('increase_budget',high),panel('passed',high)]), \
         patch('tulu_behavior.pilot.save_selection',return_value={'status':'passed'}):
        assert pipeline.pilot()['status']=='passed'
    assert len(run.call_args_list)==8
    assert all(call.args[call.args.index('--cot-budget')+1]==1024 for call in run.call_args_list[4:])


def test_oom_starts_complete_smaller_panel(pipeline):
    with patch('tulu_behavior.freeze.validate_preflight',return_value={'digest':'pre'}), \
         patch.object(pipeline,'run_module',side_effect=[42,0,0,0,0]) as run, \
         patch('tulu_behavior.pilot.summarize',return_value=panel()), \
         patch('tulu_behavior.pilot.save_selection',return_value={'status':'passed'}):
        assert pipeline.pilot()['status']=='passed'
    assert len(run.call_args_list)==5
    assert [call.args[call.args.index('--batch-size')+1] for call in run.call_args_list]==[4,2,2,2,2]


def test_maximum_budget_incompleteness_cannot_start_formal(pipeline):
    with patch('tulu_behavior.freeze.validate_preflight',return_value={'digest':'pre'}), \
         patch.object(pipeline,'run_module',return_value=0), \
         patch('tulu_behavior.pilot.summarize',return_value=panel('needs_diagnosis')), \
         patch('tulu_behavior.pilot.save_selection') as save:
        assert pipeline.pilot() is None
        save.assert_not_called()


def test_existing_selection_must_still_pass_completeness(pipeline):
    atomic_json(pipeline.root/'manifests/pilot_selection.json',{'preflight_digest':'pre','status':'passed',
                'evidence_files':{},'generation':{'batch_size':4,'budgets':{'cot':512,'direct':32}}})
    with patch('tulu_behavior.freeze.validate_preflight',return_value={'digest':'pre'}), \
         patch('tulu_behavior.pilot.summarize',return_value=panel('increase_budget')):
        with pytest.raises(RuntimeError,match='no longer matches'):
            pipeline.pilot()


def test_stage_worker_imports_published_code_with_explicit_workspace(pipeline):
    with patch('tulu_behavior.pipeline.subprocess.call',return_value=0) as call:
        pipeline.run_module('run','--stage','base','--phase','pilot')
    command=call.call_args.args[0]
    assert call.call_args.kwargs['cwd']==REPO
    assert command[command.index('--root')+1]==str(pipeline.root)
    assert command[command.index('--model-dir')+1]==str(Path(pipeline.config['models'][0]['path']).parent)
