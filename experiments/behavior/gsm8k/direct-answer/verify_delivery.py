"""Verify delivered artifacts and immutable outputs without loading any model."""
import csv
from common import ROOT, digest_file, load_json, read_jsonl


def main():
    state = load_json(ROOT / 'status/workflow.json')
    assert state['state'] == 'complete' and state['formal_count'] == 9372
    report = load_json(ROOT / 'reports/report_manifest.json')
    for filename, expected in report['files'].items():
        assert digest_file(ROOT / 'reports' / filename) == expected, filename
    for filename, expected in report['reproduction_files'].items():
        assert digest_file(ROOT / filename) == expected, filename
    for manifest in ['manifests/frozen_protocol.json', 'manifests/baseline.json']:
        for filename, expected in load_json(ROOT / manifest)['files'].items():
            assert digest_file(ROOT / filename) == expected, filename
    for phase, count in [('formal', 9372), ('pilot', 128)]:
        rows = [r for p in (ROOT / 'outputs' / phase).rglob('batch_*.jsonl') for r in read_jsonl(p)]
        assert len(rows) == len({(r['stage'],r['id']) for r in rows}) == count
    with (ROOT / 'reports/comparison_per_question.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows) == len({(r['stage'],r['id']) for r in rows}) == 9372
    audit=load_json(ROOT / 'reports/independent_audit.json')
    assert audit['status']=='passed' and audit['protocol_contrasts']==8 and audit['baseline_checked_records']==9372
    resume=load_json(ROOT / 'manifests/resume_check.json')
    assert resume['new_jobs']==resume['regenerated_answers']==0
    print('Delivery verified: 9372 formal + 128 pilot; all report, reproduction, frozen and baseline hashes match')


if __name__=='__main__':
    main()
