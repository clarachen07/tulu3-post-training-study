"""Reconstruct retained score statistics; no raw-answer rescoring or inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from decimal import Decimal
import numpy as np

ROOT = Path(__file__).resolve().parent
STAGES = ('base', 'sft', 'dpo', 'rlvr')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-11, atol=1e-12), 'Retained numerical statistic differs')


def flag(value):
    require(value in ('True', 'False'), 'Invalid retained boolean')
    return value == 'True'


def bounds(matrix, repeats, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for offset in range(0, repeats, 250):
        indices = rng.integers(0, len(matrix), size=(min(250, repeats-offset), len(matrix)))
        samples.append(matrix[indices].mean(axis=1))
    samples = np.concatenate(samples)
    ci = np.quantile(samples, [.025, .975], axis=0)
    delta = np.quantile(np.diff(samples, axis=1), [.025, .975], axis=0) if matrix.shape[1] > 1 else None
    return ci, delta


def verify(root=ROOT):
    root = Path(root)
    stats = read(root/'reports/statistics.json')
    audit = read(root/'reports/independent_audit.json')
    comparison = read(root/'reports/comparison.json')
    require(audit['status']=='passed' and audit['checked_records']==stats['formal_count']==9372, 'Historical audit coverage differs')
    require(sha(root/'reports/statistics.json')==audit['statistics_sha256'], 'Frozen statistics changed')
    derivation = read(root/'provenance/comparison_derivation.json')
    require(derivation['historical_comparison_sha256']==audit['comparison_sha256'], 'Historical comparison identity differs')
    require(derivation['published_comparison_sha256']==sha(root/'reports/comparison.json'), 'Sanitized comparison identity differs')
    require(derivation['historical_audit_sha256']==sha(root/'provenance/historical_audit.json')==sha(root/'reports/independent_audit.json'), 'Historical audit changed')
    with (root/'reports/per_question.csv').open(encoding='utf-8-sig',newline='') as f:
        questions=list(csv.DictReader(f))
    require(len(questions)==2343 and len({r['id'] for r in questions})==2343, 'Question panel differs')
    lookup={r['id']:r for r in questions}
    for row in questions:
        for stage in STAGES:
            answer=row[stage+'_answer']
            correct=bool(answer) and Decimal(answer)==Decimal(row['gold'].replace(',',''))
            require(correct==flag(row[stage+'_correct']), 'Retained numerical answer differs from score')
            flag(row[stage+'_answer_only'])
    summaries={(r['split'],r['stage']):r for r in stats['summary']}
    require(len(summaries)==len(stats['summary'])==8, 'Summary coverage differs')
    adjacent={(r['split'],r['from_stage'],r['to_stage']):r for r in stats['pairs']}
    require(len(adjacent)==len(stats['pairs'])==6, 'Adjacent comparison coverage differs')
    for si,split in enumerate(('train','test')):
        rows=sorted((r for r in questions if r['split']==split),key=lambda r:r['id'])
        n=1024 if split=='train' else 1319
        require(len(rows)==n, 'Split size differs')
        matrix=np.array([[flag(r[s+'_correct']) for s in STAGES] for r in rows],dtype=np.int8)
        ci,delta_ci=bounds(matrix,stats['bootstrap']['replicates'],stats['bootstrap']['seed']+si)
        for j,stage in enumerate(STAGES):
            saved=summaries[split,stage]
            compliant=np.array([flag(r[stage+'_answer_only']) for r in rows])
            require(saved['n']==n and saved['correct']==int(matrix[:,j].sum()), 'Accuracy counts differ')
            require(saved['answer_only']==int(compliant.sum()), 'Compliance count differs')
            require(saved['correct_and_answer_only']==int((compliant & matrix[:,j].astype(bool)).sum()), 'Correct/compliant count differs')
            require(saved['no_answer']==sum(not r[stage+'_answer'] for r in rows), 'Missing answer count differs')
            for field,count in (('accuracy',saved['correct']),('answer_only_rate',saved['answer_only']),('correct_and_answer_only_rate',saved['correct_and_answer_only']),('no_answer_rate',saved['no_answer'])):
                close(saved[field],count/n)
            close(saved['ci_low'],ci[0,j]);close(saved['ci_high'],ci[1,j])
        for j,(a,b) in enumerate(zip(STAGES,STAGES[1:])):
            saved=adjacent[split,a,b];old,new=matrix[:,j],matrix[:,j+1]
            for field,value in (('both_correct',((old==1)&(new==1)).sum()),('both_wrong',((old==0)&(new==0)).sum()),('wrong_to_correct',((old==0)&(new==1)).sum()),('correct_to_wrong',((old==1)&(new==0)).sum())):
                require(saved[field]==int(value), 'Adjacent paired count differs')
            close(saved['delta'],(new-old).mean());close(saved['ci_low'],delta_ci[0,j]);close(saved['ci_high'],delta_ci[1,j])
    with (root/'reports/comparison_per_question.csv').open(encoding='utf-8-sig',newline='') as f:
        paired=list(csv.DictReader(f))
    require(len(paired)==18744, 'Historical paired panel differs')
    cells={}
    for row in paired:
        key=row['baseline'],row['split'],row['stage'];cell=cells.setdefault(key,{})
        require(row['id'] not in cell and row['id'] in lookup, 'Unexpected/duplicate historical question')
        target=lookup[row['id']]
        require(row['gold']==target['gold'] and row['split']==target['split'] and flag(row['new_correct'])==flag(target[row['stage']+'_correct']), 'Historical pairing target differs')
        flag(row['old_correct']);cell[row['id']]=row
    require(len(cells)==len(comparison['pairs'])==16, 'Historical contrast coverage differs')
    for saved in comparison['pairs']:
        key=saved['baseline'],saved['split'],saved['stage'];cell=cells[key]
        ids=sorted(i for i,r in lookup.items() if r['split']==saved['split'])
        require(sorted(cell)==ids and saved['paired_available'], 'Historical pairing incomplete')
        old=np.array([flag(cell[i]['old_correct']) for i in ids],dtype=np.int8)
        new=np.array([flag(cell[i]['new_correct']) for i in ids],dtype=np.int8)
        baseline=read(root/f"reference/{saved['baseline']}/statistics.json")
        prior=next(r for r in baseline['summary'] if r['split']==saved['split'] and r['stage']==saved['stage'])
        require(saved['n']==len(ids)==prior['n'] and saved['old_correct']==int(old.sum())==prior['correct'] and saved['new_correct']==int(new.sum()), 'Historical marginals differ')
        for field,value in (('both_correct',((old==1)&(new==1)).sum()),('both_wrong',((old==0)&(new==0)).sum()),('wrong_to_correct',((old==0)&(new==1)).sum()),('correct_to_wrong',((old==1)&(new==0)).sum())):
            require(saved[field]==int(value), 'Historical transition differs')
        ci,_=bounds((new-old)[:,None],comparison['bootstrap_replicates'],comparison['bootstrap_seed']+(saved['split']=='test'))
        close(saved['delta'],(new-old).mean());close(saved['ci_low'],ci[0,0]);close(saved['ci_high'],ci[1,0])
    manifest=root/'provenance/publication_manifest.json'
    if manifest.exists():
        for rel,expected in read(manifest)['files'].items():
            target=(root/rel).resolve();require(target.is_relative_to(root.resolve()), 'Invalid publication path')
            require(target.is_file() and sha(target)==expected, 'Publication file changed: '+rel)
    return {'status':'passed','formal_score_records':9372,'question_rows':2343,'summary_cells':8,'historical_paired_contrasts':16,'bootstrap_replicates':10000,'intervals_recomputed_from_scores':True,'raw_answers_rescored':False,'new_model_inference':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    print(json.dumps(verify(parser.parse_args().root),indent=2))
