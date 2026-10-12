"""Read-only verification of retained research results.

Full response rescoring, activation analysis and bootstrap recomputation require
the omitted external evidence and are intentionally not claimed by this check.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('base', 'sft', 'dpo', 'rlvr')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, message):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12), message)


def read(path):
    return json.loads(path.read_text())


def check_gsm8k():
    base = ROOT / 'experiments/gsm8k'
    summaries = {}
    for condition in ('cot', 'direct-answer'):
        folder = base / condition / 'reports'
        evidence = read(folder / 'statistics.json')
        lookup = {(row['split'], row['stage']): row for row in evidence['summary']}
        require(set(lookup) == {(split, stage) for split in ('train', 'test') for stage in STAGES}, 'Incomplete GSM8K matrix')
        require(evidence['formal_count'] == 9372, 'GSM8K physical total differs')
        for (split, stage), row in lookup.items():
            require(row['n'] == (1024 if split == 'train' else 1319), 'GSM8K denominator differs')
            require(0 <= row['correct'] <= row['n'], 'Invalid GSM8K successes')
            close(row['accuracy'], row['correct']/row['n'], 'GSM8K accuracy differs from counts')
            require(0 <= row['ci_low'] <= row['ci_high'] <= 1, 'Invalid saved GSM8K interval')
        pairs = {(r['split'],r['from_stage'],r['to_stage']):r for r in evidence['pairs']}
        require(set(pairs) == {(split,a,b) for split in ('train','test') for a,b in zip(STAGES,STAGES[1:])}, 'Incomplete GSM8K transitions')
        for (split, a, b), row in pairs.items():
            both, gain, loss, wrong = [row[k] for k in ('both_correct','wrong_to_correct','correct_to_wrong','both_wrong')]
            require(all(type(n) is int and n >= 0 for n in (both,gain,loss,wrong)), 'Invalid GSM8K partition')
            require(both+gain+loss+wrong == row['n'], 'GSM8K partition denominator differs')
            require(both+loss == lookup[split,a]['correct'] and both+gain == lookup[split,b]['correct'], 'GSM8K transition marginals differ')
            close(row['delta'], (gain-loss)/row['n'], 'GSM8K stage change differs')
        csv_rows = list(csv.DictReader((folder/'stage_summary.csv').open(encoding='utf-8-sig')))
        require(len(csv_rows)==8, 'GSM8K CSV matrix differs')
        for row in csv_rows:
            saved = lookup[row['split'],row['stage']]
            for key in row:
                if key in ('split','stage'):
                    require(row[key]==saved[key], 'GSM8K CSV category differs')
                else:
                    close(row[key], saved[key], 'GSM8K CSV numeric export differs')
        summaries[condition]=lookup
    contrast = read(base/'direct-answer/reports/comparison.json')['pairs']
    require(len(contrast)==8, 'Incomplete GSM8K prompt contrast')
    for row in contrast:
        key = row['split'],row['stage']
        old,new = summaries['cot'][key], summaries['direct-answer'][key]
        require(row['old_correct']==old['correct'] and row['new_correct']==new['correct'], 'GSM8K prompt contrast marginals differ')
        require(row['both_correct']+row['correct_to_wrong']==old['correct'], 'GSM8K prompt contrast old partition differs')
        require(row['both_correct']+row['wrong_to_correct']==new['correct'], 'GSM8K prompt contrast new partition differs')
        require(sum(row[k] for k in ('both_correct','correct_to_wrong','wrong_to_correct','both_wrong'))==row['n'], 'GSM8K contrast partition differs')
        close(row['delta'],new['accuracy']-old['accuracy'],'GSM8K prompt contrast differs')
    return {'conditions':2, 'summary_cells':16, 'prompt_contrasts':8, 'rescoring_performed':False, 'bootstrap_recomputed':False}


def check_math500():
    from scipy.stats import binomtest
    evidence = read(ROOT/'experiments/math500/results/frozen/three-stage-pass16.json')
    for stage, row in evidence['pooled'].items():
        histogram = {int(k): v for k,v in row['correct_count_histogram'].items()}
        n = row['samples_per_question']
        require(row['questions']==500 and sum(histogram.values())==500, 'MATH500 question count differs')
        require(sum(c*count for c,count in histogram.items())==row['correct_generations'], 'MATH500 success total differs')
        for k_string, saved in row['pass_at'].items():
            k=int(k_string)
            estimate=sum(count*(1-(math.comb(n-c,k)/math.comb(n,k) if n-c>=k else 0)) for c,count in histogram.items())/500*100
            close(saved,estimate, f'MATH500 pass@{k} differs for {stage}')
    paired = evidence['dpo_rlvr_paired_pass_at_16']
    require(sum(paired[k] for k in ('both','rlvr_only','dpo_only','neither'))==500, 'MATH500 paired partition differs')
    require(paired['both']+paired['dpo_only']==evidence['pooled']['dpo']['pass_at_16_questions'], 'MATH500 DPO marginal differs')
    require(paired['both']+paired['rlvr_only']==evidence['pooled']['rlvr']['pass_at_16_questions'], 'MATH500 RLVR marginal differs')
    close(paired['exact_two_sided_p'],binomtest(paired['rlvr_only'],paired['discordant'],.5).pvalue,'MATH500 exact test differs')
    close(paired['bootstrap']['point_pp'],(paired['rlvr_only']-paired['dpo_only'])/500*100,'MATH500 paired point estimate differs')
    return {'pooled_models':len(evidence['pooled']), 'histogram_pass_at_recomputed':True, 'raw_response_scoring':False, 'bootstrap_recomputed':False}


def check_syntax():
    base = ROOT/'experiments/syntax/results/test'
    checked = 0
    for prompt in ('P0', 'P3'):
        models = {}
        for stage in ('base', 'sft', 'dpo', 'final'):
            row = read(base/prompt/f'{stage}.json')
            require(row['complete'] and row['stage']==stage and row['candidate']==prompt and row['split']=='test', 'Syntax completion or coordinate differs')
            for punctuation in ('with_punct', 'no_punct'):
                evidence = row[punctuation]
                counts = evidence['counts']
                require(counts['gold']==(25094 if punctuation=='with_punct' else 21998), 'Syntax token denominator differs')
                require(evidence['diagnostic_counts']['sentences']==2077, 'Syntax sentence denominator differs')
                for metric, saved in evidence['metrics'].items():
                    if metric=='head_consistency':
                        denominator=counts.get('head_consistency_eligible',0)
                        if not denominator:
                            require(saved is None,'Undefined syntax consistency must remain null')
                            continue
                        estimate=counts.get('head_consistent',0)/denominator
                    else:
                        count=counts.get(metric,0)
                        require(0 <= count <= counts['gold'], 'Invalid syntax token count')
                        estimate=count/counts['gold']
                    close(saved,estimate,'Syntax strict metric differs from counts')
            models[stage]=row
            checked+=1
        paired = read(base/prompt/'paired-bootstrap.json')['comparisons']
        require(len(paired)==9,'Syntax comparison family differs')
        for row in paired:
            before=models[row['from']]['with_punct']['metrics'][row['metric']]
            after=models[row['to']]['with_punct']['metrics'][row['metric']]
            close(row['delta_percentage_points'],100*(after-before),'Syntax paired point estimate differs')
            require(row['ci95'][0]<=row['ci95'][1],'Invalid saved syntax interval')
    return {'test_cells':checked,'strict_metrics_checked_against_counts':True,'raw_response_scoring':False,'bootstrap_recomputed':False}


def check_ssp():
    base = ROOT/'experiments/ssp/results/analysis'
    checked=0
    for folder in sorted(base.iterdir()):
        summary=read(folder/'summary.json')
        rows=list(csv.DictReader((folder/'paired_comparisons.csv').open()))
        for endpoint in summary['primary_endpoints']:
            close(endpoint['mean_difference'],endpoint['right_mean']-endpoint['left_mean'],'SSP paired point estimate differs')
            require(endpoint['ci95_low']<=endpoint['ci95_high'],'Invalid saved SSP interval')
            matches=[r for r in rows if r['comparison']==endpoint['comparison'] and r['boundary_mode']==endpoint['boundary_mode'] and int(r['layer'])==endpoint['layer'] and int(r['horizon'])==endpoint['horizon'] and r['metric']==endpoint['metric']]
            require(len(matches)==1, 'SSP endpoint missing or duplicated in CSV')
            for key in ('mean_difference','left_mean','right_mean','ci95_low','ci95_high','n_problems'):
                close(matches[0][key],endpoint[key], 'SSP CSV endpoint differs')
            checked+=1
    return {'paired_endpoints':checked, 'behavior_endpoint':'incomplete', 'bootstrap_recomputed':False}


def main():
    sys.path.insert(0,str(ROOT/'experiments/prontoqa'))
    from tulu_behavior.publication import verify_publication
    result={'status':'passed',
            'prontoqa_behavior':verify_publication(), 'gsm8k':check_gsm8k(),
            'math500':check_math500(), 'syntax':check_syntax(), 'ssp':check_ssp()}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
