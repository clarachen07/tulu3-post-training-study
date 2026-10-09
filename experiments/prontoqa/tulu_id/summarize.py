"""Derive paired peak and sample-size summaries from verified TwoNN tables."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .common import atomic_json, load_config, sha256_file

CURVE_FIELDS = ['stage','peak_layer','first_mean','first_sd','peak_mean','peak_sd',
                'last_mean','last_sd','peak_minus_first_mean','peak_minus_first_sd',
                'peak_minus_last_mean','peak_minus_last_sd']
SCALE_FIELDS = ['stage','layer','full_n','full_id','small_n','small_mean','small_sd',
                'relative_change_percent','middle_above_both_endpoints_all_k']


def summaries(replicates, layers, scales, config):
    """Select one mean peak, then pair endpoint differences by replicate ID."""
    curves, sizes = [], []
    for model in config['models']:
        stage = model['stage']
        rows = sorted([r for r in layers if r['stage'] == stage], key=lambda r:int(r['layer']))
        if len(rows) != config['inference']['layers'] or any(r['status'] != 'ok' for r in rows):
            raise ValueError('Curve summary requires every valid layer')
        first, peak, last = rows[0], max(rows,key=lambda r:float(r['id_mean'])), rows[-1]
        curve = dict(stage=stage,peak_layer=int(peak['layer']))
        paired = []
        for name, row in [('first',first),('peak',peak),('last',last)]:
            values = sorted([r for r in replicates if r['stage']==stage and int(r['layer'])==int(row['layer'])],
                            key=lambda r:int(r['replicate']))
            if ([int(r['replicate']) for r in values] != list(range(config['statistics']['subsample_repeats']))
                    or any(r['status'] != 'ok' for r in values)):
                raise ValueError('Paired curve coordinates are incomplete or invalid')
            array = np.array([float(r['id']) for r in values])
            curve[name+'_mean'],curve[name+'_sd'] = float(array.mean()),float(array.std(ddof=0))
            paired.append(array)
        for name, difference in [('peak_minus_first',paired[1]-paired[0]),('peak_minus_last',paired[1]-paired[2])]:
            curve[name+'_mean'],curve[name+'_sd'] = float(difference.mean()),float(difference.std(ddof=0))
        curves.append(curve)
        selected = config['statistics']['scale_layers']
        indexed = {(int(r['layer']),int(r['k'])):r for r in scales if r['stage']==stage}
        wanted = {(layer,k) for layer in selected for k in config['statistics']['scale_k']}
        if len(indexed) != len(wanted) or set(indexed) != wanted or any(r['status']!='ok' for r in indexed.values()):
            raise ValueError('Scale summary coordinates are incomplete or invalid')
        ordering = all(float(indexed[(selected[1],k)]['id_mean']) > max(
                       float(indexed[(selected[0],k)]['id_mean']),float(indexed[(selected[2],k)]['id_mean']))
                       for k in config['statistics']['scale_k'])
        for layer in selected:
            records = sorted([r for (l,k),r in indexed.items() if l==layer],key=lambda r:int(r['n_block']))
            small, full = records[0],records[-1]
            large_mean, small_mean = float(full['id_mean']),float(small['id_mean'])
            sizes.append(dict(stage=stage,layer=layer,full_n=int(full['n_block']),full_id=large_mean,
                              small_n=int(small['n_block']),small_mean=small_mean,small_sd=float(small['id_sd']),
                              relative_change_percent=100*(small_mean/large_mean-1),
                              middle_above_both_endpoints_all_k=ordering))
    return curves,sizes


def write_summaries(root):
    """Update the result manifest and independently recheck it before PCA freezes inputs."""
    root = Path(root)
    from .analyze import _write_csv
    def read(name):
        with (root/'reports'/name).open(newline='') as stream:
            return list(csv.DictReader(stream))
    curves,sizes = summaries(read('id_replicates.csv'),read('id_summary.csv'),read('scale_checks.csv'),load_config(root))
    _write_csv(root/'reports/curve_summary.csv',curves,CURVE_FIELDS)
    _write_csv(root/'reports/scale_summary.csv',sizes,SCALE_FIELDS)
    path = root/'reports/analysis_manifest.json'
    import json
    manifest = json.loads(path.read_text())
    for name in ['curve_summary.csv','scale_summary.csv']:
        relative = 'reports/'+name
        manifest['outputs'][relative] = sha256_file(root/relative)
    manifest['summary_implementation'] = {'tulu_id/summarize.py':sha256_file(Path(__file__))}
    atomic_json(path,manifest)
    from .verify import final_check
    final_check(root)
    return curves,sizes
