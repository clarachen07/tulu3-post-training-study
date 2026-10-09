"""Build and independently check the aggregate-only behavioral publication."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from scipy.stats import binomtest

from .common import (ROOT, REPO, atomic_json, fingerprint, load_config, publication_code_hashes,
                     read_jsonl, sha256_file, utc_now, verify_files)
from .analyze import STAGES, CONDITIONS, TEMPLATES, HOPS, METRICS, _figures, _report, _write_csv
from .reanalysis import TABLE_KEYS
from .verify import _verify_csv, load_case_review

TABLE_NAMES = {"summaries": "summary", "primary_tests": "primary_tests",
               "paired_differences": "paired_differences", "transitions": "transitions",
               "diagnostics": "diagnostics", "label_summaries": "label_summary",
               "dataset_summary": "dataset_summary"}
REFERENCE_FILES = {
    "experiment.json", "README.md", "data/formal.jsonl", "data/pilot.jsonl", "data/demonstrations.jsonl",
    "provenance/data.json", "provenance/original_data.json", "provenance/history.json",
    "provenance/reanalysis.json", "provenance/publication.json", "provenance/consolidation.json", "reports/corpus-index.jsonl", "reports/analysis.json",
    "reports/provenance.json", "reports/report.html", "reports/figures/values.json",
    *[f"reports/{name}.csv" for name in TABLE_NAMES.values()],
    *[f"reports/figures/{name}.{suffix}" for name in ("stage_comparison", "difficulty_curves") for suffix in ("png", "svg")],
}


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _close(a, b, message, *, absolute=1e-12):
    _require(type(a) in (float, int) and math.isfinite(a) and math.isclose(a, b, rel_tol=1e-10, abs_tol=absolute), message)


def published_corpus(root=ROOT):
    """Compact sample/stratum/label index, without questions, proofs or model inputs."""
    path = Path(root) / 'reports/corpus-index.jsonl'
    if path.exists():
        return read_jsonl(path)
    return read_jsonl(Path(root) / 'data/formal.jsonl')


def check_statistics(summary, items):
    """Validate aggregates without reconstructing or inventing per-item scores."""
    counts = Counter(i["hop"] for i in items)
    _require(counts == {1: 1440, 3: 1440, 5: 1440}, "Published corpus has unexpected strata")
    _require(len({i["sample_id"] for i in items}) == 4320, "Duplicate published sample IDs")
    _require(summary["n_items"] == 4320 and summary["physical_generations"] == 60480
             and summary["logical_records"] == 69120, "Physical/logical generation totals differ")
    lookup = {}
    for row in summary["summaries"]:
        key = tuple(row[k] for k in ("stage", "condition", "template", "hop", "metric"))
        _require(key not in lookup, "Duplicate summary cell")
        lookup[key] = row
    expected = {(s, c, t, h, m) for s in STAGES for c in CONDITIONS for t in TEMPLATES
                for h in ("all", *HOPS) for m in (METRICS if c == "cot" else METRICS[:1])}
    _require(set(lookup) == expected, "Summary matrix is incomplete")
    for (s, c, t, h, m), r in lookup.items():
        _require(r["n"] == (4320 if h == "all" else counts[h]), "Summary denominator differs")
        _require(0 <= r["estimate"] <= 1 and 0 <= r["ci_low"] <= r["ci_high"] <= 1, "Invalid success rate or saved interval")
        _close(r["estimate"] * r["n"], round(r["estimate"] * r["n"]), "Success count is not integral")
        _require(r["aggregation"] == ("equal_weight_hops" if h == "all" else "within_hop"), "Aggregation policy differs")
        if h == "all":
            _close(r["estimate"], sum(lookup[s,c,t,hop,m]["estimate"] for hop in HOPS)/3, "Macro estimate differs from hop means")
        if s == "base":
            _require(r == {**lookup[s,c,"plain",h,m], "template": t}, "Base template views differ")
        if m == "joint_success":
            _require(r["estimate"] <= min(lookup[s,c,t,h,"answer_correct"]["estimate"],
                                         lookup[s,c,t,h,"proof_success"]["estimate"]), "Joint successes exceed marginal successes")
    transitions = {}
    for r in summary["transitions"]:
        key = tuple(r[k] for k in ("before_stage", "after_stage", "condition", "template", "hop"))
        _require(key not in transitions, "Duplicate stage transition cell")
        transitions[key] = r
        parts = [r[k] for k in ("both_correct", "gained_correct", "lost_correct", "both_wrong")]
        _require(all(type(n) is int and n >= 0 for n in parts) and sum(parts) == r["n"], "Invalid transition partition")
        a,b,c,t,h = key
        _close((r["both_correct"]+r["lost_correct"])/r["n"], lookup[a,c,t,h,"answer_correct"]["estimate"], "Before-stage transition marginal differs")
        _close((r["both_correct"]+r["gained_correct"])/r["n"], lookup[b,c,t,h,"answer_correct"]["estimate"], "After-stage transition marginal differs")
    expected_transitions = {(a,b,c,t,h) for a,b in zip(STAGES,STAGES[1:]) for c in CONDITIONS for t in TEMPLATES for h in ("all",*HOPS)}
    _require(set(transitions) == expected_transitions, "Transition matrix is incomplete")
    for a,b,c,t,_ in expected_transitions:
        pooled = transitions[a,b,c,t,"all"]
        for field in ("both_correct", "gained_correct", "lost_correct", "both_wrong", "n"):
            _require(pooled[field] == sum(transitions[a,b,c,t,h][field] for h in HOPS), "Pooled transitions differ from hop partitions")
    tests = summary["primary_tests"]
    _require([(r["before_stage"],r["after_stage"]) for r in tests] == list(zip(STAGES,STAGES[1:])), "Primary comparison family differs")
    raw_p = []
    for r in tests:
        transition = transitions[r["before_stage"],r["after_stage"],"cot","native","all"]
        for k in ("n", "gained_correct", "lost_correct", "both_correct", "both_wrong"):
            _require(r[k] == transition[k], "Primary test counts differ from transitions")
        gain,loss = r["gained_correct"],r["lost_correct"]
        p = float(binomtest(gain,gain+loss,.5).pvalue) if gain+loss else 1.
        _close(r["p_value"],p,"Exact McNemar probability differs",absolute=0)
        _close(r["delta_pp"],100*(gain-loss)/r["n"],"Primary change differs from discordant counts")
        _require(r["family_size"] == 3 and r["discordant"] == gain+loss and r["test"] == "exact_two_sided_mcnemar", "McNemar family or test definition differs")
        _require((r["hop"],r["metric"],r["before_condition"],r["after_condition"],r["before_template"],r["after_template"])
                 == ("all","answer_correct","cot","cot","native","native"), "Primary endpoint differs")
        raw_p.append(p)
    running = 0.
    for rank,index in enumerate(sorted(range(3),key=raw_p.__getitem__)):
        running = min(1.,max(running,(3-rank)*raw_p[index]))
        _close(tests[index]["p_holm"],running,"Holm probability differs",absolute=0)
        _require(tests[index]["reject_holm_0_05"] == (running < .05), "Holm decision differs")
    differences = {}
    for r in summary["paired_differences"]:
        key = tuple(r[k] for k in ("kind","before_stage","after_stage","before_condition","after_condition","before_template","after_template","hop","metric"))
        _require(key not in differences,"Duplicate paired difference")
        differences[key] = r
        a = lookup[r["before_stage"],r["before_condition"],r["before_template"],r["hop"],r["metric"]]["estimate"]
        b = lookup[r["after_stage"],r["after_condition"],r["after_template"],r["hop"],r["metric"]]["estimate"]
        _close(r["delta_pp"],100*(b-a),"Paired point estimate differs")
        _require(-100 <= r["ci_low_pp"] <= r["ci_high_pp"] <= 100,"Invalid saved paired interval")
        if r["kind"] == "template_native_minus_plain" and r["before_stage"] == "base":
            _require(r["delta_pp"] == r["ci_low_pp"] == r["ci_high_pp"] == 0,"Base templates incorrectly treated as independent")
    expected_differences = set()
    for c in CONDITIONS:
        for t in TEMPLATES:
            for a,b in zip(STAGES,STAGES[1:]):
                for m in (METRICS if c=='cot' else METRICS[:1]):
                    for h in ('all',*HOPS):
                        expected_differences.add(('adjacent_stage',a,b,c,c,t,t,h,m))
    for s in STAGES:
        for c in CONDITIONS:
            for m in (METRICS if c=='cot' else METRICS[:1]):
                for h in ('all',*HOPS):
                    expected_differences.add(('template_native_minus_plain',s,s,c,c,'plain','native',h,m))
        for t in TEMPLATES:
            for h in ('all',*HOPS):
                expected_differences.add(('cot_minus_direct',s,s,'direct','cot',t,t,h,'answer_correct'))
    _require(set(differences)==expected_differences,'Paired difference matrix is incomplete')
    for r in tests:
        paired=differences['adjacent_stage',r['before_stage'],r['after_stage'],'cot','cot','native','native','all','answer_correct']
        for k in ('delta_pp','ci_low_pp','ci_high_pp'):
            _close(r[k],paired[k],'Primary test and paired difference interval differ')
    labels = Counter((i["hop"],i["answer"]) for i in items)
    label_rows = {}
    for r in summary["label_summaries"]:
        key=tuple(r[k] for k in ('stage','condition','template','hop','metric','label'))
        _require(key not in label_rows,'Duplicate label summary')
        label_rows[key]=r
        n = sum(labels[h,r["label"]] for h in HOPS) if r["hop"] == "all" else labels[r["hop"],r["label"]]
        _require(r["n"] == n,"Label denominator differs")
        _close(r["estimate"]*n,round(r["estimate"]*n),"Label success count is not integral")
    _require(set(label_rows)=={(*key,label) for key in expected for label in ('True','False')},'Label summary matrix is incomplete')
    for key,row in lookup.items():
        label_counts=[label_rows[(*key,label)] for label in ('True','False')]
        _close(sum(r['estimate']*r['n'] for r in label_counts)/row['n'],row['estimate'],'Label success marginals differ')
    for r in summary["dataset_summary"]:
        _require(r["n"] == counts[r["hop"]] and r["labels"] == {label:labels[r["hop"],label] for label in ("True","False")},"Dataset label counts differ")
    _require(len(summary['dataset_summary'])==3,'Dataset summary strata are incomplete')
    diagnostic_keys=[]
    for r in summary["diagnostics"]:
        diagnostic_keys.append(tuple(r[k] for k in ('stage','condition','template','hop')))
        n = 4320 if r["hop"] == "all" else counts[r["hop"]]
        _require(r["n"] == n and sum(r["answer_status_counts"].values()) == n and sum(r["finish_reason_counts"].values()) == n,"Diagnostic denominator differs")
        for k in ("answer_parse_failure_rate","format_compliance_rate","proof_parse_failure_rate","skip_rate","logic_error_rate","truncation_rate"):
            if r[k] is None:
                _require(r["condition"] == "direct" and k in ("proof_parse_failure_rate","skip_rate","logic_error_rate"),"Unexpected null diagnostic")
            else:
                _require(0 <= r[k] <= 1,"Diagnostic rate is out of range")
                _close(n*r[k],round(n*r[k]),"Diagnostic count is not integral")
    _require(len(diagnostic_keys)==len(set(diagnostic_keys)) and set(diagnostic_keys)=={(s,c,t,h) for s in STAGES for c in CONDITIONS for t in TEMPLATES for h in ('all',*HOPS)},'Diagnostic matrix is incomplete')
    return lookup


def _svg_series(path, gid, values, *, intervals=False):
    """Check actual SVG line coordinates against percentage-space values."""
    svg = ET.parse(path).getroot()
    ns = {"s":"http://www.w3.org/2000/svg"}
    groups = [g for g in svg.findall('.//s:g',ns) if g.get('id') == gid]
    _require(len(groups) == 1, f"Missing or duplicate plotted SVG series: {gid}")
    paths = groups[0].findall('s:path',ns)
    _require(len(paths) == (len(values) if intervals else 1), f"SVG series path count differs: {gid}")
    for index,p in enumerate(paths):
        clip_id = p.attrib['clip-path'][5:-1]
        clip = next(c for c in svg.findall('.//s:clipPath',ns) if c.get('id') == clip_id).find('s:rect',ns)
        top,height = float(clip.attrib['y']),float(clip.attrib['height'])
        numbers = [float(n) for n in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',p.attrib['d'])]
        actual_y = numbers[1::2]
        percentages = values[index] if intervals else values
        expected_y = [top + height*(102-v)/104 for v in percentages]
        _require(len(actual_y) == len(expected_y) and all(abs(a-b)<2e-5 for a,b in zip(actual_y,expected_y)), f"Rendered SVG coordinates differ: {gid}")


def check_figures(summary, root=ROOT):
    lookup = {(r['stage'],r['condition'],r['template'],r['hop'],r['metric']):r for r in summary['summaries']}
    directory = Path(root)/'reports/figures'
    plotted = json.loads((directory/'values.json').read_text())
    _require(len(plotted['stage_comparison']) == 3 and len(plotted['difficulty_curves']) == 8,"Plot series count differs")
    for name,series in plotted.items():
        for r in series:
            if name == 'stage_comparison':
                rows = [lookup[s,'cot','native','all',r['metric']] for s in STAGES]
                gid = 'stage_'+r['metric']
            else:
                rows = [lookup[r['stage'],r['condition'],'native',h,'answer_correct'] for h in HOPS]
                gid = 'difficulty_'+r['condition']+'_'+r['stage']
            expected = [row['estimate']*100 for row in rows]
            intervals = [[row['ci_low']*100,row['ci_high']*100] for row in rows]
            _require(r['y_percent'] == expected and r['ci_percent'] == intervals,"Plotted values differ from summary")
            _svg_series(directory/f'{name}.svg',gid,expected)
            _svg_series(directory/f'{name}.svg','ci_'+gid,intervals,intervals=True)


def check_links(paths):
    checked = 0
    for path in paths:
        text = path.read_text(encoding='utf-8')
        refs = re.findall(r'(?:href|src)=["\']([^"\']+)["\']',text) if path.suffix == '.html' else re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',text)
        for ref in refs:
            parsed = urlsplit(ref)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (path.parent/unquote(parsed.path)).resolve()
            _require(target.exists(), f"Broken local report link: {path.name}: {ref}")
            checked += 1
    return checked


def reference_files(root=ROOT):
    root = Path(root)
    result = []
    for path in root.rglob('*'):
        if not path.is_file() or any(part in ('__pycache__','.pytest_cache') for part in path.parts):
            continue
        relative = str(path.relative_to(root))
        _require(relative in REFERENCE_FILES or (relative.startswith('tests/test_') and path.suffix == '.py') or relative in ('tests/fixtures/original-items.jsonl', 'tests/fixtures/formal-items.jsonl', 'tests/fixtures/demonstrations.jsonl'),
                 f"File is outside the behavioral publication allowlist: {relative}")
        if relative not in ('provenance/publication.json', 'provenance/consolidation.json'):
            result.append(path)
    return result


def verify_publication(root=ROOT, *, check_hashes=True):
    root = Path(root)
    summary = json.loads((root/'reports/analysis.json').read_text())
    check_statistics(summary,published_corpus(root))
    for key,name in TABLE_NAMES.items():
        _verify_csv(root/'reports'/f'{name}.csv',summary[key])
    check_figures(summary,root)
    paths = [root/'reports/report.html',root/'README.md']
    if root == ROOT:
        paths += [REPO/'README.md',REPO/'README.zh-CN.md',REPO/'reports/report.zh-CN.md']
    links = check_links(paths)
    for path in list((REPO/'tulu_behavior').glob('*.py'))+list((root/'tests').glob('*.py')):
        _require(re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]',path.read_text()) is None,"Behavioral source or test contains Chinese text")
    for path in reference_files(root):
        if path.suffix in ('.html','.csv','.json','.jsonl','.svg'):
            _require(re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]',path.read_text(encoding='utf-8-sig')) is None,"Behavioral program output contains Chinese text")
    if check_hashes:
        ledger_path = root/'provenance/consolidation.json'
        if not ledger_path.exists():
            ledger_path = root/'provenance/publication.json'
        ledger = json.loads(ledger_path.read_text())
        _require(ledger['status']=='passed' and ledger['scope']=='published_aggregate_evidence_only',"Publication scope is missing")
        _require(fingerprint({k:v for k,v in ledger.items() if k!='identity'})==ledger['identity'],"Publication ledger identity differs")
        verify_files(REPO,ledger['files'])
        _require(ledger['code_files']==publication_code_hashes(),"Publication code identity changed")
    return {'status':'passed','scope':'published_aggregate_evidence_only','summary_cells':len(summary['summaries']),
            'transition_cells':len(summary['transitions']),'primary_tests':3,'local_links_checked':links,
            'svg_series_and_intervals_checked':11,'rescoring_performed':False,'bootstrap_recomputed':False}


def build_reference(source, reanalysis_directory):
    """Publish only a verified reanalysis; never copy raw generation artifacts."""
    source,reanalysis_directory = Path(source),Path(reanalysis_directory)
    record = json.loads((reanalysis_directory/'manifests/reanalysis.json').read_text())
    _require(record['status']=='passed' and record['comparison']['status']=='passed',"Full reanalysis must pass before publication")
    _require(record['publication_code_files']==publication_code_hashes(),"Reanalysis belongs to different publication code")
    original = json.loads((source/'reports/analysis.json').read_text())
    accepted = json.loads((source/'manifests/server_final_verification.json').read_text())
    _require(accepted['status']=='passed' and accepted['case_review']['status']=='passed' and not accepted['offline'],"Original server acceptance is incomplete")
    audit = json.loads((source/'reports/error_audit_sample.json').read_text())
    review = load_case_review(source,record['historical_protocol_digest'],audit)
    _require(review['status']=='passed',"Historical case-review evidence is incomplete")
    history = {
        'schema_version':1,'status':'accepted_original_experiment','protocol_digest':record['historical_protocol_digest'],
        'generation':original['provenance']['generation'],'statistics':original['statistics'],
        'historical_analysis_code_sha256':original['provenance']['analysis_code_sha256'],
        'historical_scoring_code_sha256':original['provenance']['scoring_code_sha256'],
        'source_config_sha256':original['provenance']['config_sha256'],
        'formal_items':accepted['formal_items'],'physical_generations':accepted['physical_generations'],
        'logical_records':accepted['logical_records'],'committed_batches':accepted['committed_batches'],
        'original_server_checks':{k:accepted[k] for k in ('verified_utc','environment_checked','model_assets_checked','output_token_decoding_checked','rescored_generations')},
        'case_review':{'status':'passed','reviewer':'Codex','sampled_entries':len(audit['cases']),
                       'reviewed_unique_generations':review['reviewed_unique_generations'],'findings_count':review['findings_count'],
                       'interpretation':'Proof parsing failures can induce downstream skip flags; answer and proof are scored independently.'},
        'source_manifest_hashes':{name:sha256_file(source/name) for name in ('manifests/frozen_protocol.json','manifests/analysis.json','manifests/server_final_verification.json','manifests/case_review.json')},
        'raw_batch_hashes_fingerprint':fingerprint(original['provenance']['raw_batches']),
        'presentation_policy':record['comparison']['presentation_policy'],
    }
    atomic_json(ROOT/'provenance/history.json',history)
    atomic_json(ROOT/'provenance/reanalysis.json',record)
    summary = json.loads((reanalysis_directory/'reports/analysis.json').read_text())
    summary['case_review']={'status':'passed','reviewer':'Codex','reviewed_generation_count':review['reviewed_unique_generations'],
                            'meaning':'Historical attestation; no raw case text is published'}
    provenance = {
        'schema_version':2,'generated_at_utc':utc_now(),'protocol':load_config()['protocol'],
        'protocol_digest':history['protocol_digest'],'generation':history['generation'],'statistics':summary['statistics'],
        'config_sha256':sha256_file(ROOT/'experiment.json'),'dataset_sha256':sha256_file(ROOT/'data/formal.jsonl'),
        'models':[{k:m[k] for k in ('stage','id','revision','wrapper')} for m in load_config()['models']],
        'historical_analysis_code_sha256':history['historical_analysis_code_sha256'],
        'publication_code_identity':record['publication_code_identity'],
        'raw_batch_hashes_fingerprint':history['raw_batch_hashes_fingerprint'],
        'scope':'Aggregate publication; complete raw generations, tokenized inputs and per-item scores are not distributed',
    }
    summary['provenance']=provenance
    reports = ROOT/'reports'; reports.mkdir(exist_ok=True)
    for key,name in TABLE_NAMES.items():
        _write_csv(reports/f'{name}.csv',summary[key])
    atomic_json(reports/'analysis.json',summary); atomic_json(reports/'provenance.json',provenance)
    _figures(summary,reports/'figures')
    (reports/'report.html').write_text(_report(summary,load_config(),{'cases':[]},published=True))
    return summary


def save_validation():
    if not (ROOT/'provenance/publication.json').exists():
        atomic_json(ROOT/'provenance/publication.json',{'status':'pending','scope':'published_aggregate_evidence_only'})
    checks = verify_publication(check_hashes=False)
    paths = reference_files()+list((REPO/'tulu_behavior').glob('*.py'))
    paths += [REPO/'README.md',REPO/'README.zh-CN.md',REPO/'reports/report.zh-CN.md',REPO/'.gitignore']
    ledger = {**checks,'schema_version':1,'verified_utc':utc_now(),
              'code_files':publication_code_hashes(),
              'files':{str(p.relative_to(REPO)):sha256_file(p) for p in sorted(set(paths))},
              'limitations':['Published counts permit McNemar/Holm checks, but not raw rescoring or paired-bootstrap recomputation.',
                             'Saved bootstrap intervals are compared against a separate full local reanalysis.']}
    ledger['identity']=fingerprint(ledger)
    atomic_json(ROOT/'provenance/publication.json',ledger)
    return verify_publication()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-reference',action='store_true',help='Explicitly rebuild published aggregate files from a verified reanalysis')
    parser.add_argument('--source-dir',type=Path,help='Original complete accepted experiment')
    parser.add_argument('--output-dir',type=Path,default=REPO/'runs/behavior-reanalysis',help='Previously verified independent reanalysis directory')
    parser.add_argument('--model-dir',type=Path,help='Unused for aggregate publication; models are never loaded')
    args=parser.parse_args()
    if args.build_reference:
        if args.source_dir is None:
            parser.error('--source-dir is required with --build-reference')
        build_reference(args.source_dir,args.output_dir)
        result=save_validation()
    else:
        result=verify_publication()
    print(json.dumps(result))


if __name__=='__main__':
    main()
