"""Compare only with existing historical evidence; never generate old answers."""
from pathlib import Path
import numpy as np
from analyze import STAGES, load_results, write_csv
from common import runtime as runtime_config, ROOT, digest_file, fingerprint, load_json, read_jsonl, score, utc, write_json


def historical_panel(root, stage, expected, reference):
    """Missing/incomplete evidence permits aggregates, but never unverified pairing."""
    if root is None:
        return None, {}, 'Historical evidence directory not configured'
    root = Path(root)
    paths = sorted((root / f'outputs/formal/{stage}').glob('batch_*.jsonl'))
    if not paths:
        return None, {}, 'Historical raw outputs missing'
    sources, panel = {}, {}
    try:
        for path in paths:
            rel = str(path.relative_to(root))
            actual = digest_file(path)
            if actual != reference['sources'].get(rel):
                raise ValueError('Unverified historical file: ' + rel)
            sources[str(path.resolve())] = actual
            for row in read_jsonl(path):
                ident = row['id']
                if ident not in expected or ident in panel:
                    raise ValueError('Unexpected or duplicate historical item: ' + ident)
                if row['phase'] != 'formal' or row['stage'] != stage:
                    raise ValueError('Historical phase/stage mismatch')
                if any(row[k] != expected[ident][k] for k in ['gold', 'split', 'question_hash']):
                    raise ValueError('Historical question identity mismatch')
                if row['record_hash'] != fingerprint({k:v for k,v in row.items() if k != 'record_hash'}):
                    raise ValueError('Historical record hash mismatch')
                if score(row['continuation'], row['gold']) != (row['extracted_answer'], row['correct']):
                    raise ValueError('Historical score mismatch')
                panel[ident] = row
        if set(panel) != set(expected):
            raise ValueError('Historical panel incomplete')
        for split in ['train', 'test']:
            rows = [r for r in panel.values() if r['split'] == split]
            saved = next(r for r in reference['summary'] if r['stage'] == stage and r['split'] == split)
            if len(rows) != saved['n'] or sum(r['correct'] for r in rows) != saved['correct']:
                raise ValueError('Historical aggregate mismatch')
    except (AssertionError, ValueError, KeyError, TypeError) as exc:
        return None, sources, str(exc)
    return panel, sources, None


def contrast(old, new, n, old_correct, replicates, seed):
    result = {'n': n, 'old_correct': old_correct, 'new_correct': int(new.sum()),
              'old_accuracy': old_correct / n, 'new_accuracy': float(new.mean()),
              'delta': float(new.mean() - old_correct / n), 'paired_available': old is not None,
              'ci_low': None, 'ci_high': None, 'wrong_to_correct': None, 'correct_to_wrong': None,
              'both_correct': None, 'both_wrong': None}
    if old is not None:
        old = np.asarray(old, dtype=np.int8)
        assert len(old) == n and int(old.sum()) == old_correct
        diffs = new.astype(np.int8) - old
        rng = np.random.default_rng(seed)
        draws = []
        for offset in range(0, replicates, 500):
            k = min(500, replicates - offset)
            draws.extend(diffs[rng.integers(0, n, size=(k, n))].mean(axis=1))
        bounds = np.quantile(draws, [.025, .975])
        result.update(ci_low=float(bounds[0]), ci_high=float(bounds[1]),
                      wrong_to_correct=int(((old == 0) & (new == 1)).sum()),
                      correct_to_wrong=int(((old == 1) & (new == 0)).sum()),
                      both_correct=int(((old == 1) & (new == 1)).sum()),
                      both_wrong=int(((old == 0) & (new == 0)).sum()))
    return result


def main():
    cfg = load_json(ROOT / 'experiment.json')
    runtime = runtime_config()
    expected, new, new_sources = load_results()
    comparisons, per_question, evidence = [], [], {}
    for group, key in [('cot', 'cot_root'), ('direct-answer', 'direct_answer_root')]:
        reference = load_json(ROOT / f'reference/{group}/statistics.json')
        evidence[group] = {'reference_statistics_sha256': digest_file(ROOT / f'reference/{group}/statistics.json'), 'stages': {}}
        for stage in STAGES:
            old, sources, reason = historical_panel(runtime.get(key), stage, expected, reference)
            evidence[group]['stages'][stage] = {'paired_available': old is not None, 'reason': reason, 'sources': sources}
            for si, split in enumerate(['train', 'test']):
                ids = sorted(i for i, r in expected.items() if r['split'] == split)
                saved = next(r for r in reference['summary'] if r['stage'] == stage and r['split'] == split)
                assert saved['n'] == len(ids)
                values = np.array([new[stage][i]['correct'] for i in ids], dtype=np.int8)
                old_values = None if old is None else [old[i]['correct'] for i in ids]
                row = contrast(old_values, values, len(ids), saved['correct'], cfg['bootstrap_replicates'], cfg['bootstrap_seed'] + si)
                comparisons.append(dict(row, baseline=group, stage=stage, split=split, unavailable_reason=reason))
                if old is not None:
                    per_question.extend({'baseline': group, 'stage': stage, 'split': split, 'id': i,
                                         'gold': expected[i]['gold'], 'old_correct': old[i]['correct'],
                                         'new_correct': new[stage][i]['correct']} for i in ids)
    write_json(ROOT / 'reports/comparison.json', {'created_utc': utc(), 'direction': 'cot-direct-answer minus historical baseline',
               'pairs': comparisons, 'evidence': evidence, 'new_sources': new_sources,
               'bootstrap_replicates': cfg['bootstrap_replicates'], 'bootstrap_seed': cfg['bootstrap_seed'],
               'historical_generation_policy': 'Never rerun either historical group'})
    write_csv(ROOT / 'reports/protocol_comparison.csv', comparisons)
    if per_question:
        write_csv(ROOT / 'reports/comparison_per_question.csv', per_question)
    print('Computed 16 historical contrasts; paired:', sum(r['paired_available'] for r in comparisons), flush=True)


if __name__ == '__main__':
    main()
