"""Add the fixed answer instruction to a hash-verified original CoT panel."""
import shutil
import subprocess
import sys
from pathlib import Path
from common import runtime as runtime_config, ROOT, digest_file, fingerprint, load_json, read_jsonl, utc, write_json, write_jsonl


def derive(rows, prefix, instruction):
    result = []
    for row in rows:
        assert row['content'].startswith(prefix), row['id']
        result.append({**row, 'content': instruction + '\n\n' + row['content']})
    return result


def verify_baseline(root):
    reference = load_json(ROOT / 'reference/cot/data.json')
    for rel, expected in reference['prepared_files'].items():
        assert digest_file(root / rel) == expected, rel
    for name, info in reference['source_files'].items():
        assert digest_file(root / 'data/raw' / name) == info['sha256'], name


def baseline():
    destination = ROOT / 'data/baseline'
    if (destination / 'data/prepared/formal.jsonl').exists():
        verify_baseline(destination)
        return destination
    runtime = runtime_config()
    source = Path(runtime['cot_root']) if runtime.get('cot_root') else None
    reference = load_json(ROOT / 'reference/cot/data.json')
    required = list(reference['prepared_files']) + ['data/raw/' + n for n in reference['source_files']]
    if source and all((source / rel).is_file() for rel in required):
        verify_baseline(source)
        for rel in required:
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, target)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'reference/cot/experiment.json', destination / 'experiment.json')
        subprocess.run([sys.executable, '-B', str(ROOT / 'rebuild_cot.py'), str(destination)], check=True)
    verify_baseline(destination)
    return destination


def main():
    cfg = load_json(ROOT / 'experiment.json')
    original = load_json(ROOT / 'reference/cot/experiment.json')
    direct = load_json(ROOT / 'reference/direct-answer/experiment.json')
    for key in ['inference', 'train_size', 'test_size', 'pilot_size', 'sample_seed', 'pilot_seed',
                'bootstrap_seed', 'bootstrap_replicates', 'gsm_revision', 'rlvr_revision']:
        assert cfg[key] == original[key] == direct[key], key
    for model, old in zip(cfg['models'], original['models']):
        assert {k:v for k,v in model.items() if k != 'path'} == {k:v for k,v in old.items() if k != 'path'}
    assert len(cfg['models']) == len(original['models']) == 4
    assert cfg['answer_instruction'] == direct['answer_instruction']
    manifest = ROOT / 'manifests/data.json'
    if manifest.exists():
        saved = load_json(manifest)
        assert saved['config_hash'] == fingerprint(cfg)
        for rel, expected in saved['prepared_files'].items():
            assert digest_file(ROOT / rel) == expected, rel
        verify_baseline(ROOT / 'data/baseline')
        return
    base = baseline()
    prefix = (base / 'data/prepared/eight_shot_prefix.txt').read_text()
    assert prefix.count('Question:') == 9 and prefix.count('Answer:') == 8
    out = ROOT / 'data/prepared'
    out.mkdir(parents=True, exist_ok=True)
    for phase in ['formal', 'pilot']:
        rows = read_jsonl(base / f'data/prepared/{phase}.jsonl')
        write_jsonl(out / f'{phase}.jsonl', derive(rows, prefix, cfg['answer_instruction']))
    shutil.copy2(base / 'data/prepared/rlvr_gsm8k_matches.jsonl', out / 'rlvr_gsm8k_matches.jsonl')
    (out / 'eight_shot_prefix.txt').write_text(cfg['answer_instruction'] + '\n\n' + prefix)
    meta = load_json(ROOT / 'reference/cot/data.json')
    meta.update({'created_utc': utc(), 'config_hash': fingerprint(cfg),
                 'derivation': 'Prepend unchanged direct-answer instruction and two newlines to original CoT content; all other fields unchanged',
                 'prepared_files': {str(p.relative_to(ROOT)): digest_file(p) for p in out.iterdir() if p.is_file()},
                 'raw_root': str(base / 'data/raw'),
                 'baseline_manifest_sha256': digest_file(ROOT / 'reference/cot/data.json')})
    write_json(manifest, meta)
    print('Verified original CoT data; prepared 2343 formal and 32 pilot inputs', flush=True)


if __name__ == '__main__':
    main()
