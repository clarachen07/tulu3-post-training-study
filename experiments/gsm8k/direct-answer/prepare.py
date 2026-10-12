"""Derive answer-only inputs from a byte-verified, read-only baseline snapshot."""
import shutil
from common import ROOT, digest_file, fingerprint, load_json, read_jsonl, utc, write_json, write_jsonl


def main():
    cfg = load_json(ROOT / 'experiment.json')
    base = ROOT / 'baseline'
    oldcfg = load_json(base / 'experiment.json')
    for key in ['models', 'inference', 'train_size', 'test_size', 'pilot_size', 'sample_seed',
                'pilot_seed', 'bootstrap_seed', 'bootstrap_replicates', 'rlvr_revision', 'gsm_revision']:
        assert cfg[key] == oldcfg[key], key
    frozen = load_json(base / 'manifests/frozen_protocol.json')
    for rel, expected in frozen['files'].items():
        if (base / rel).exists():
            assert digest_file(base / rel) == expected, rel
    for rel, expected in load_json(base / 'reports/statistics.json')['sources'].items():
        assert digest_file(base / rel) == expected, rel
    manifest = ROOT / 'manifests/data.json'
    if manifest.exists():
        saved = load_json(manifest)
        assert saved['config_hash'] == fingerprint(cfg)
        for rel, expected in saved['prepared_files'].items():
            assert digest_file(ROOT / rel) == expected
        return
    oldprefix = (base / 'data/prepared/eight_shot_prefix.txt').read_text()
    pieces = oldprefix.split('\n\nQuestion: ')
    assert len(pieces) == 9 and pieces[-1] == ''
    demos = []
    for j, piece in enumerate(pieces[:-1]):
        question, answer = piece.split('\nAnswer:', 1)
        question = question.removeprefix('Question: ') if j == 0 else question
        assert answer.rstrip().endswith('So the answer is ' + cfg['demo_answers'][j] + '.')
        demos.append('Question: ' + question + '\nAnswer:' + cfg['demo_answers'][j])
    prefix = cfg['answer_instruction'] + '\n\n' + '\n\n'.join(demos) + '\n\nQuestion: '
    out = ROOT / 'data/prepared'
    out.mkdir(parents=True, exist_ok=True)
    for phase in ['formal', 'pilot']:
        rows = read_jsonl(base / f'data/prepared/{phase}.jsonl')
        for row in rows:
            assert row['content'].startswith(oldprefix)
            row['content'] = prefix + row['content'][len(oldprefix):]
        write_jsonl(out / f'{phase}.jsonl', rows)
    shutil.copy2(base / 'data/prepared/rlvr_gsm8k_matches.jsonl', out / 'rlvr_gsm8k_matches.jsonl')
    (out / 'eight_shot_prefix.txt').write_text(prefix)
    source = load_json(base / 'manifests/data.json')
    source.update({'created_utc': utc(), 'config_hash': fingerprint(cfg),
                   'derivation': 'Same target suffixes and panel as baseline; eight demonstration rationales replaced by final numbers; fixed direct-answer instruction prepended',
                   'baseline_data_manifest_sha256': digest_file(base / 'manifests/data.json'),
                   'prepared_files': {str(p.relative_to(ROOT)): digest_file(p) for p in out.iterdir() if p.is_file()}})
    write_json(manifest, source)
    write_json(ROOT / 'manifests/baseline.json', {'created_utc': utc(),
        'files': {str(p.relative_to(ROOT)): digest_file(p) for p in base.rglob('*') if p.is_file()}})
    assets = load_json(base / 'manifests/assets.json')
    assets['config_hash'] = fingerprint(cfg)
    assets['reuse_provenance'] = {'source_sha256': digest_file(base / 'manifests/assets.json'),
                                'policy': 'Full published weight hashes reused only with unchanged inode/size/mtime/ctime/resolved path; checked on server before inference'}
    write_json(ROOT / 'manifests/assets.json', assets)
    print('Derived 2343 formal and 32 pilot inputs; baseline preserved', flush=True)


if __name__ == '__main__':
    main()
