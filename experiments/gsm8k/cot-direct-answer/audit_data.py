"""Independently check the exact transformation against retained historical hashes."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ref = json.loads((ROOT / 'reference/cot/data.json').read_text())
    direct = json.loads((ROOT / 'reference/direct-answer/experiment.json').read_text())
    cfg = json.loads((ROOT / 'experiment.json').read_text())
    base = ROOT / 'data/baseline'
    for name, expected in ref['prepared_files'].items():
        assert sha(base / name) == expected, name
    for name, info in ref['source_files'].items():
        assert sha(base / 'data/raw' / name) == info['sha256'], name
    instruction = direct['answer_instruction']
    assert cfg['answer_instruction'] == instruction
    prefix = (base / 'data/prepared/eight_shot_prefix.txt').read_text()
    assert (ROOT / 'data/prepared/eight_shot_prefix.txt').read_text() == instruction + '\n\n' + prefix
    ids = {}
    for phase, count in [('formal', 2343), ('pilot', 32)]:
        old = read(base / f'data/prepared/{phase}.jsonl')
        new = read(ROOT / f'data/prepared/{phase}.jsonl')
        assert len(old) == len(new) == count
        for a, b in zip(old, new):
            assert a['content'].startswith(prefix)
            assert b == dict(a, content=instruction + '\n\n' + a['content'])
        ids[phase] = {r['id'] for r in new}
        assert len(ids[phase]) == count
        if phase == 'formal':
            assert collections.Counter(r['split'] for r in new) == {'train': 1024, 'test': 1319}
    assert not ids['formal'] & ids['pilot']
    assert sha(ROOT / 'data/prepared/rlvr_gsm8k_matches.jsonl') == ref['prepared_files']['data/prepared/rlvr_gsm8k_matches.jsonl']
    result = {'status': 'passed', 'rlvr_records_checked': 7473, 'train_sample_checked': 1024,
              'test_checked': 1319, 'pilot_checked': 32,
              'data_manifest_sha256': sha(ROOT / 'manifests/data.json'),
              'implementation': 'No production imports; byte-verified historical panel and raw data; exact ordered field-by-field transformation'}
    path = ROOT / 'manifests/data_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(result, flush=True)


if __name__ == '__main__':
    main()
