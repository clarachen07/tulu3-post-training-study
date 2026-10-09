"""Preflight and final scientific-artifact verification, independent of launch status."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .common import (ROOT, atomic_json, load_config, read_jsonl, sha256_file,
                     utc_now, validate_frozen, protocol_digest)
from .extract import verify_batch


def preflight(root):
    frozen = validate_frozen(root); cfg = load_config(root)
    items = read_jsonl(root / 'data/items.jsonl')
    n = cfg['data']['sample_count']; ids = [x['sample_id'] for x in items]
    if len(items) != n or len(set(ids)) != n or len({x['text'] for x in items}) != n:
        raise RuntimeError('Dataset count or uniqueness mismatch')
    if any(x['text'] != x['question'] + ' ' + x['query'] or x['hop'] != 3 for x in items):
        raise RuntimeError('Prompt construction or hop mismatch')
    stats = cfg['statistics']
    rng = np.random.RandomState(stats['subsample_seed'])
    expected = np.stack([rng.permutation(n)[:stats['subsample_size']] for _ in range(stats['subsample_repeats'])])
    if not np.array_equal(expected, np.load(root / 'data/subsamples.npy')):
        raise RuntimeError('Shared subsample schedule differs from fixed RNG')
    rng = np.random.RandomState(stats['scale_seed'])
    if not np.array_equal(np.stack([rng.permutation(n) for _ in stats['scale_k']]), np.load(root / 'data/scale_permutations.npy')):
        raise RuntimeError('Shared scale schedule differs from fixed RNG')
    for model in cfg['models']:
        rows = read_jsonl(root / f'data/tokenized/{model["stage"]}.jsonl')
        if [x['sample_id'] for x in rows] != ids:
            raise RuntimeError('Tokenized sample order differs')
        for row in rows:
            if (row['attention_mask'] != [1] * len(row['input_ids'])
                    or row['readout_index'] != len(row['input_ids']) - 1):
                raise RuntimeError('Unexpected padding, truncation or last-token readout')
    return dict(protocol_digest=frozen['protocol_digest'], samples=n)


def final_check(root):
    outcome = preflight(root); cfg = load_config(root); digest = protocol_digest(root)
    records = read_jsonl(root / 'data/items.jsonl'); n = len(records)
    for model in cfg['models']:
        stage = model['stage']; target = root / 'activations' / stage
        completion = json.loads((target / 'complete.json').read_text())
        if (completion['protocol_digest'], completion['completed_samples'], completion['stage'],
                completion['mode'], completion['fp32'], completion['tf32']) != (digest, n, stage, 'formal', True, False):
            raise RuntimeError(f'Incomplete or inconsistent extraction: {stage}')
        expected = []
        for start in range(0, n, cfg['inference']['write_batch_size']):
            ids = [r['sample_id'] for r in records[start:start+cfg['inference']['write_batch_size']]]
            path = target / f'batch_{start:06d}.npz'; expected.append(path.name)
            if not verify_batch(path, ids, digest, cfg['inference']['layers'], cfg['inference']['hidden_size'], stage=stage, mode='formal'):
                raise RuntimeError(f'Missing committed activation batch: {path}')
        if sorted(p.name for p in target.glob('batch_*.npz')) != sorted(expected):
            raise RuntimeError('Unexpected activation batch inventory')
    report = json.loads((root / 'reports/analysis_manifest.json').read_text())
    if report['protocol_digest'] != digest: raise RuntimeError('Analysis belongs to another protocol')
    for path, value in report['outputs'].items():
        if sha256_file(root / path) != value: raise RuntimeError(f'Report output changed: {path}')
    from .analyze import load_inputs, audit_stage
    inputs = load_inputs(root, cfg)
    for m in cfg['models']:
        audit = audit_stage(root, m['stage'], cfg, inputs)
        if report['source_digests'][m['stage']] != audit['digest']: raise RuntimeError('Analysis input provenance changed')
    def csv_rows(name):
        with (root / 'reports' / name).open() as f: return list(csv.DictReader(f))
    reps = csv_rows('id_replicates.csv'); summary = csv_rows('id_summary.csv'); scales = csv_rows('scale_checks.csv')
    if len(reps) != 4*32*20 or len(summary) != 4*32 or len(scales) != 4*3*20:
        raise RuntimeError('Incomplete statistic tables')
    for row in summary:
        group = [r for r in reps if r['stage'] == row['stage'] and r['layer'] == row['layer']]
        if sorted(int(r['replicate']) for r in group) != list(range(20)): raise RuntimeError('Replicate mismatch')
        if row['status'] == 'ok':
            values = np.array([float(r['id']) for r in group])
            np.testing.assert_allclose([float(row['id_mean']), float(row['id_sd'])],
                [values.mean(), values.std(ddof=0)], rtol=1e-12, atol=1e-12)
        elif any(r['status'] != 'ok' for r in group):
            if row['id_mean'] or row['id_sd']: raise RuntimeError('Invalid result was filled in')
        else: raise RuntimeError('Unexpected invalid summary')
    outcome.update(status='passed', completed_utc=utc_now(), verified_input_samples=4*n,
                   invalid_layer_count=report['invalid_layer_count'], outputs=report['outputs'])
    atomic_json(root / 'manifests/final_verification.json', outcome)
    return outcome


def main():
    p = argparse.ArgumentParser(); from .common import add_workspace_arguments, workspace
    add_workspace_arguments(p, analysis=True)
    p.add_argument('--preflight', action='store_true'); args = p.parse_args()
    root = workspace(args, analysis=True)
    result = preflight(root) if args.preflight else final_check(root)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
