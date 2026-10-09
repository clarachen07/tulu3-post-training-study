"""Hand-calculated paired differences and scale interpretation."""
import pytest
from tulu_id.summarize import summaries


def fixture():
    config = {'models':[{'stage':'base'}], 'inference':{'layers':3},
              'statistics':{'subsample_repeats':2,'scale_layers':[1,2,3],'scale_k':[1,2]}}
    values = {1:[1,3],2:[3,4],3:[2,2]}
    reps = [dict(stage='base',layer=l,replicate=r,id=v,status='ok') for l,a in values.items() for r,v in enumerate(a)]
    layers = [dict(stage='base',layer=l,id_mean=sum(a)/2,status='ok') for l,a in values.items()]
    scales = [dict(stage='base',layer=l,k=k,n_block=6//k,id_mean=v,id_sd=0 if k==1 else .1,status='ok')
              for k,a in [(1,[1,5,3]),(2,[.5,4,2])] for l,v in enumerate(a,1)]
    return reps,layers,scales,config


def test_selected_peak_differences_are_paired_by_replicate():
    reps,layers,scales,config = fixture()
    curves,sizes = summaries(reps[::-1],layers[::-1],scales[::-1],config)
    row = curves[0]
    assert row['peak_layer']==2
    assert row['peak_minus_first_mean']==row['peak_minus_last_mean']==1.5
    assert row['peak_minus_first_sd']==row['peak_minus_last_sd']==.5
    assert sizes[0]['relative_change_percent']==-50
    assert all(row['middle_above_both_endpoints_all_k'] for row in sizes)


def test_missing_pair_cannot_produce_a_curve_summary():
    reps,layers,scales,config = fixture()
    with pytest.raises(ValueError,match='incomplete'):
        summaries(reps[:-1],layers,scales,config)
