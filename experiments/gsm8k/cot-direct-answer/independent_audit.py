"""Independent reconstruction: intentionally does not import production scoring/I/O."""
import collections
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def numerical_value(text):
    cleaned = "".join(char for index, char in enumerate(text)
                      if not (char == "," and 0 < index < len(text) - 1
                              and text[index - 1].isdecimal() and text[index + 1].isdecimal()))
    matches = re.findall(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)", cleaned)
    return Decimal(matches[-1]) if matches else None


def format_valid(text):
    text = text.strip()
    prefix = re.match(r"(?:Answer\s*:|Final\s+answer\s*:|The\s+answer\s+is)\s*", text, re.I)
    if prefix:
        text = text[prefix.end():]
    if text.endswith('.'):
        text = text[:-1].rstrip()
    for left, right in [('$$', '$$'), ('$', '$'), (r'\(', r'\)'), (r'\[', r'\]')]:
        if text.startswith(left):
            if not text.endswith(right) or len(text) < len(left) + len(right):
                return False
            text = text[len(left):-len(right)]
            break
    if text.startswith(r'\boxed{') and text.endswith('}'):
        text = text[7:-1].strip()
    if text[:1] in ['+', '-']:
        text = text[1:]
    if not text:
        return False
    pieces = text.split('.')
    if len(pieces) > 2 or (len(pieces) == 2 and not pieces[1].isdecimal()):
        return False
    integer = pieces[0]
    if not integer:
        return len(pieces) == 2
    groups = integer.split(',')
    return all(g.isdecimal() for g in groups) and (len(groups) == 1 or
        (1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])))


def bounds(values, draws):
    counts = values[draws].sum(axis=1, dtype=np.int64)
    return counts, np.percentile(counts / len(values), [2.5, 97.5], axis=0)


def main():
    cfg = json.loads((ROOT / 'experiment.json').read_text())
    stats = json.loads((ROOT / 'reports/statistics.json').read_text())
    frozen = json.loads((ROOT / 'manifests/frozen_protocol.json').read_text())
    comparison = json.loads((ROOT / 'reports/comparison.json').read_text())
    for rel, expected in frozen['files'].items():
        assert sha256(ROOT / rel) == expected, rel
    formal_rows = read(ROOT / 'data/prepared/formal.jsonl')
    formal = {r['id']: r for r in formal_rows}
    assert len(formal) == len(formal_rows) == 2343
    assert collections.Counter(r['split'] for r in formal_rows) == {'train': 1024, 'test': 1319}
    pilot = read(ROOT / 'data/prepared/pilot.jsonl')
    assert len(pilot) == len({r['id'] for r in pilot}) == 32
    assert not set(formal) & {r['id'] for r in pilot}
    original_meta = json.loads((ROOT / 'reference/cot/data.json').read_text())
    for rel, expected in original_meta['prepared_files'].items():
        assert sha256(ROOT / 'data/baseline' / rel) == expected
    for phase in ['formal', 'pilot']:
        old = read(ROOT / f'data/baseline/data/prepared/{phase}.jsonl')
        new = read(ROOT / f'data/prepared/{phase}.jsonl')
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert b == dict(a, content=cfg['answer_instruction'] + '\n\n' + a['content'])
    assets = json.loads((ROOT / 'manifests/assets.json').read_text())
    published = json.loads((ROOT / 'manifests/published_identity.json').read_text())
    assert published['status'] == 'passed'
    shards = 0
    for model in cfg['models']:
        actual = next(m for m in assets['models'] if m['registry']['stage'] == model['stage'])
        identity = next(m for m in published['models'] if m['stage'] == model['stage'])
        assert actual['registry'] == model and identity['revision'] == model['revision']
        weights = [f['sha256'] for f in actual['files'] if f['path'].endswith('.safetensors')]
        assert sorted(weights) == sorted(identity['published_sha256'])
        shards += len(weights)
    assert shards == 16
    stages = [m['stage'] for m in cfg['models']]
    config_hash = object_sha256(cfg)
    panels, seen_sources = {}, {}
    native_tokens = None
    checked = 0
    for stage in stages:
        tokenized = read(ROOT / f'data/tokenized/{stage}/formal.jsonl')
        prompts = {r['id']: r for r in tokenized}
        assert len(prompts) == len(tokenized) and set(prompts) == set(formal)
        if stage != 'base':
            tokens = {i:p['prompt_token_hash'] for i,p in prompts.items()}
            if native_tokens is None:
                native_tokens = tokens
            else:
                assert tokens == native_tokens
        for ident, p in prompts.items():
            text = formal[ident]['content']
            manual = text + '\nAnswer:' if stage == 'base' else '<|user|>\n' + text + '\n<|assistant|>\n'
            assert p['prompt_text'] == manual
            assert p['prompt_text_hash'] == object_sha256(manual)
            assert p['prompt_token_hash'] == object_sha256(p['prompt_token_ids'])
            assert len(p['prompt_token_ids']) + cfg['inference']['max_tokens'] <= cfg['inference']['max_model_len']
        panel = {}
        for path in sorted((ROOT / f'outputs/formal/{stage}').glob('batch_*.jsonl')):
            rel = str(path.relative_to(ROOT))
            seen_sources[rel] = sha256(path)
            assert seen_sources[rel] == stats['sources'][rel]
            for r in read(path):
                ident = r['id']
                assert ident not in panel and ident in formal
                assert r['stage'] == stage and r['phase'] == 'formal'
                assert r['config_hash'] == config_hash
                assert r['record_hash'] == object_sha256({k:v for k,v in r.items() if k != 'record_hash'})
                for key in ['split', 'gold', 'question_hash']:
                    assert r[key] == formal[ident][key]
                for key in ['prompt_token_hash', 'prompt_text_hash']:
                    assert r[key] == prompts[ident][key]
                value = numerical_value(r['continuation'])
                correct = value is not None and value == Decimal(str(r['gold']).replace(',', ''))
                assert correct == r['correct']
                assert value == (None if r['extracted_answer'] is None else Decimal(r['extracted_answer']))
                assert format_valid(r['continuation']) == r['answer_only']
                assert r['output_tokens'] == len(r['output_token_ids']) <= cfg['inference']['max_tokens']
                assert r['input_tokens'] == len(prompts[ident]['prompt_token_ids'])
                assert r['truncated'] == (r['finish_reason'] == 'length')
                assert r['finish_reason'] in ['length', 'stop']
                panel[ident] = r
                checked += 1
        assert set(panel) == set(formal)
        panels[stage] = panel
    assert seen_sources == stats['sources'] == comparison['new_sources']
    assert checked == stats['formal_count'] == 9372
    max_difference = 0.0
    for si, split in enumerate(['train', 'test']):
        ids = sorted(i for i,r in formal.items() if r['split'] == split)
        values = np.array([[panels[s][i]['correct'] for s in stages] for i in ids], dtype=np.int8)
        rng = np.random.default_rng(cfg['bootstrap_seed'] + si)
        draws = rng.integers(0, len(ids), size=(cfg['bootstrap_replicates'], len(ids)))
        counts, intervals = bounds(values, draws)
        stage_intervals = np.percentile((counts[:,1:] - counts[:,:-1]) / len(ids), [2.5,97.5], axis=0)
        for j, stage in enumerate(stages):
            saved = next(r for r in stats['summary'] if r['split'] == split and r['stage'] == stage)
            rows = [panels[stage][i] for i in ids]
            assert saved['n'] == len(ids) and saved['correct'] == int(values[:,j].sum())
            assert saved['accuracy'] == float(values[:,j].mean())
            for key, total in [('answer_only', sum(r['answer_only'] for r in rows)),
                               ('correct_and_answer_only', sum(r['correct'] and r['answer_only'] for r in rows)),
                               ('no_answer', sum(r['extracted_answer'] is None for r in rows)),
                               ('truncated', sum(r['truncated'] for r in rows))]:
                assert saved[key] == total
                rate = 'truncation_rate' if key == 'truncated' else key + '_rate'
                assert saved[rate] == total / len(ids)
            assert saved['mean_output_tokens'] == np.mean([r['output_tokens'] for r in rows])
            assert saved['median_output_tokens'] == np.median([r['output_tokens'] for r in rows])
            max_difference = max(max_difference, abs(saved['ci_low']-intervals[0,j]), abs(saved['ci_high']-intervals[1,j]))
            for group in ['cot', 'direct-answer']:
                c = next(c for c in comparison['pairs'] if c['baseline'] == group and c['stage'] == stage and c['split'] == split)
                oldstats = json.loads((ROOT / f'reference/{group}/statistics.json').read_text())
                oldsummary = next(r for r in oldstats['summary'] if r['split'] == split and r['stage'] == stage)
                assert c['old_correct'] == oldsummary['correct'] and c['new_correct'] == saved['correct']
                assert c['n'] == len(ids)
                assert abs(c['delta'] - (saved['correct']-oldsummary['correct'])/len(ids)) < 1e-15
                evidence = comparison['evidence'][group]['stages'][stage]
                assert c['paired_available'] == evidence['paired_available']
                if not c['paired_available']:
                    assert c['ci_low'] is c['ci_high'] is None
                    assert all(c[k] is None for k in ['wrong_to_correct','correct_to_wrong','both_correct','both_wrong'])
                    assert evidence['reason']
                    continue
                old = {}
                for name, expected_hash in evidence['sources'].items():
                    path = Path(name)
                    assert sha256(path) == expected_hash
                    assert expected_hash in oldstats['sources'].values()
                    for r in read(path):
                        assert r['id'] not in old
                        assert r['gold'] == formal[r['id']]['gold']
                        val = numerical_value(r['continuation'])
                        assert r['correct'] == (val is not None and val == Decimal(str(r['gold']).replace(',','')))
                        old[r['id']] = r
                assert set(old) == set(formal)
                old_values = np.array([old[i]['correct'] for i in ids], dtype=np.int8)
                assert old_values.sum() == c['old_correct']
                delta_counts = (values[:,j] - old_values)[draws].sum(axis=1, dtype=np.int64)
                ci = np.percentile(delta_counts / len(ids), [2.5,97.5])
                max_difference = max(max_difference, abs(c['ci_low']-ci[0]), abs(c['ci_high']-ci[1]))
                for key, a, b in [('wrong_to_correct',0,1),('correct_to_wrong',1,0),('both_correct',1,1),('both_wrong',0,0)]:
                    assert c[key] == int(((old_values == a) & (values[:,j] == b)).sum())
        for j in range(3):
            p = next(p for p in stats['pairs'] if p['split'] == split and p['from_stage'] == stages[j])
            assert abs(p['delta'] - (values[:,j+1]-values[:,j]).mean()) < 1e-15
            for key, a, b in [('wrong_to_correct',0,1),('correct_to_wrong',1,0),('both_correct',1,1),('both_wrong',0,0)]:
                assert p[key] == int(((values[:,j] == a) & (values[:,j+1] == b)).sum())
            max_difference = max(max_difference, abs(p['ci_low']-stage_intervals[0,j]), abs(p['ci_high']-stage_intervals[1,j]))
    assert max_difference < 1e-12
    resume = json.loads((ROOT / 'manifests/resume_check.json').read_text())
    assert resume['status'] == 'passed' and resume['regenerated_answers'] == 0
    result = {'status':'passed', 'checked_records':checked, 'stage_split_cells':8,
              'published_weight_shards_verified':shards, 'frozen_files_verified':len(frozen['files']),
              'paired_historical_contrasts':sum(c['paired_available'] for c in comparison['pairs']),
              'aggregate_only_historical_contrasts':sum(not c['paired_available'] for c in comparison['pairs']),
              'statistics_sha256':sha256(ROOT / 'reports/statistics.json'),
              'comparison_sha256':sha256(ROOT / 'reports/comparison.json'),
              'bootstrap_replicates_per_split':cfg['bootstrap_replicates'],
              'maximum_interval_difference':float(max_difference),
              'scope':'Independent numerical and format parsers, ordered input transformation, hashes, prompts, completeness, counts, bootstrap intervals and verified available historical contrasts; no production imports'}
    (ROOT / 'reports/independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
