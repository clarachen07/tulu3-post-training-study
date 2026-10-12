"""Question-paired comparisons: answer-only minus the original protocol."""
import numpy as np
from analyze import STAGES, load_results, write_csv
from common import ROOT, read_jsonl, load_json, score, digest_file, write_json, utc
from format_policy import answer_only


def main():
    cfg = load_json(ROOT / 'experiment.json')
    expected, new, new_sources = load_results()
    old, old_sources = {}, {}
    for stage in STAGES:
        panel = {}
        for path in sorted((ROOT / f'baseline/outputs/formal/{stage}').glob('batch_*.jsonl')):
            old_sources[str(path.relative_to(ROOT))] = digest_file(path)
            for row in read_jsonl(path):
                assert row['id'] not in panel and row['gold'] == expected[row['id']]['gold']
                assert score(row['continuation'], row['gold']) == (row['extracted_answer'], row['correct'])
                panel[row['id']] = row
        assert set(panel) == set(expected)
        old[stage] = panel
    comparisons, examples, per_question = [], [], []
    for si, split in enumerate(['train', 'test']):
        ids = sorted(i for i, r in expected.items() if r['split'] == split)
        matrix = np.array([[old[s][i]['correct'] for s in STAGES] + [new[s][i]['correct'] for s in STAGES] for i in ids], dtype=np.int8)
        rng = np.random.default_rng(cfg['bootstrap_seed'] + si)
        estimates = []
        for offset in range(0, cfg['bootstrap_replicates'], 500):
            k = min(500, cfg['bootstrap_replicates'] - offset)
            estimates.append(matrix[rng.integers(0, len(ids), size=(k, len(ids)))].mean(axis=1))
        draws = np.concatenate(estimates)
        intervals = np.quantile(draws[:, 4:] - draws[:, :4], [.025, .975], axis=0)
        for j, stage in enumerate(STAGES):
            a, b = matrix[:, j], matrix[:, j + 4]
            row = {'split': split, 'stage': stage, 'n': len(ids),
                   'old_correct': int(a.sum()), 'new_correct': int(b.sum()),
                   'old_accuracy': float(a.mean()), 'new_accuracy': float(b.mean()),
                   'delta': float((b-a).mean()), 'ci_low': float(intervals[0, j]), 'ci_high': float(intervals[1, j]),
                   'wrong_to_correct': int(((a == 0) & (b == 1)).sum()),
                   'correct_to_wrong': int(((a == 1) & (b == 0)).sum()),
                   'both_correct': int(((a == 1) & (b == 1)).sum()),
                   'both_wrong': int(((a == 0) & (b == 0)).sum())}
            for label, data in [('old', old), ('new', new)]:
                panel = [data[stage][i] for i in ids]
                row.update({label + '_answer_only_rate': sum(answer_only(r['continuation']) for r in panel) / len(ids),
                            label + '_no_answer_rate': sum(r['extracted_answer'] is None for r in panel) / len(ids),
                            label + '_truncation_rate': sum(r['truncated'] for r in panel) / len(ids),
                            label + '_mean_output_tokens': float(np.mean([r['output_tokens'] for r in panel])),
                            label + '_median_output_tokens': float(np.median([r['output_tokens'] for r in panel]))})
            comparisons.append(row)
            for direction, mask in [('wrong_to_correct', (a == 0) & (b == 1)), ('correct_to_wrong', (a == 1) & (b == 0))]:
                for index in np.flatnonzero(mask)[:2]:
                    ident = ids[int(index)]
                    examples.append({'id': ident, 'stage': stage, 'split': split, 'direction': direction,
                                     'question': expected[ident]['question'], 'gold': expected[ident]['gold'],
                                     'old': old[stage][ident], 'new': new[stage][ident]})
            for ident in ids:
                arow, brow = old[stage][ident], new[stage][ident]
                per_question.append({'id': ident, 'split': split, 'stage': stage, 'gold': expected[ident]['gold'],
                                     'old_correct': arow['correct'], 'new_correct': brow['correct'],
                                     'old_answer': arow['extracted_answer'], 'new_answer': brow['extracted_answer'],
                                     'old_answer_only': answer_only(arow['continuation']), 'new_answer_only': answer_only(brow['continuation']),
                                     'old_output_tokens': arow['output_tokens'], 'new_output_tokens': brow['output_tokens']})
    old_timing = load_json(ROOT / 'baseline/reports/statistics.json')['timing']
    new_timing = load_json(ROOT / 'reports/statistics.json')['timing']
    timing = []
    for stage in STAGES:
        a = next(t for t in old_timing if t['stage'] == stage)
        b = next(t for t in new_timing if t['stage'] == stage)
        timing.append({'stage': stage, 'old_process_seconds': a['allocated_process_seconds'],
                       'new_process_seconds': b['allocated_process_seconds'],
                       'old_generation_seconds': a['committed_generation_seconds'], 'new_generation_seconds': b['committed_generation_seconds'],
                       'old_output_tokens': a['output_tokens'], 'new_output_tokens': b['output_tokens']})
    out = ROOT / 'reports'
    write_csv(out / 'protocol_comparison.csv', comparisons)
    write_csv(out / 'comparison_per_question.csv', per_question)
    write_csv(out / 'timing_comparison.csv', timing)
    write_json(out / 'comparison_examples.json', examples)
    write_json(out / 'comparison.json', {'created_utc': utc(), 'direction': 'answer_only minus original',
        'pairs': comparisons, 'timing': timing, 'old_sources': old_sources, 'new_sources': new_sources,
        'bootstrap_replicates': cfg['bootstrap_replicates'], 'bootstrap_seed': cfg['bootstrap_seed']})
    print('Eight paired protocol contrasts computed', flush=True)


if __name__ == '__main__':
    main()
