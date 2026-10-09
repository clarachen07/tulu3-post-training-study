"""Compare a complete reproduction with the unmodified published numbers."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .common import ROOT, atomic_json, sha256_file, utc_now

TABLES = [f'reports/{name}.csv' for name in
          ['id_replicates','id_summary','scale_checks','scale_blocks','curve_summary','scale_summary']]
TABLES += [f'pca/reports/{name}.csv' for name in
           ['full','replicates','summary','comparison_replicates','comparison_summary',
            'scale_blocks','scale_summary','curve_summary']]
COORDINATES = ['stage','layer','preprocessing','sample_set','replicate','k','block']
EXACT_COUNTS = {'pc_id90','legacy_index_gt90','n_samples','n_features','n_constant_features',
                'rank_cap','numerical_rank','n_estimates','n_valid_estimates','n_total',
                'n_valid','n_zero','n_tied','n_fit','n_block','n_dropped','peak_layer',
                'first_layer','last_layer','full_n','small_n'}


def compare_tables(reference, actual, *, rtol=1e-10, atol=1e-12):
    """Pair records by scientific coordinates, never by their row order."""
    with Path(reference).open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields, wanted = reader.fieldnames, list(reader)
    with Path(actual).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != set(fields or []):
            raise ValueError(f'CSV fields differ: {actual}')
        observed = list(reader)
    if not fields or not wanted or len(observed) != len(wanted):
        raise ValueError(f'CSV record count differs: {actual}')
    coordinates = [field for field in COORDINATES if field in fields]
    def index(rows):
        result = {tuple(row[k] for k in coordinates):row for row in rows}
        if len(result) != len(rows):
            raise ValueError(f'Duplicate comparison coordinates: {actual}')
        return result
    left, right = index(wanted),index(observed)
    if set(left) != set(right):
        raise ValueError(f'CSV coordinate coverage differs: {actual}')
    maximum_absolute_error, numeric_fields = 0., 0
    for key in left:
        for field in fields:
            expected, value = left[key][field],right[key][field]
            try:
                expected_float,value_float = float(expected),float(value)
            except (ValueError,TypeError):
                if value != expected:
                    raise ValueError(f'CSV value differs: {actual}/{key}/{field}')
                continue
            if not np.isfinite([expected_float,value_float]).all():
                raise ValueError(f'Nonfinite CSV value: {actual}/{key}/{field}')
            if field in EXACT_COUNTS:
                valid = value_float == expected_float and value_float.is_integer()
            else:
                valid = np.isclose(value_float,expected_float,rtol=rtol,atol=atol)
            if not valid:
                raise ValueError(f'Numeric reproduction differs: {actual}/{key}/{field}: {value} vs {expected}')
            numeric_fields += 1
            maximum_absolute_error = max(maximum_absolute_error,abs(value_float-expected_float))
    return dict(rows=len(wanted),numeric_values=numeric_fields,maximum_absolute_error=maximum_absolute_error,
                reference_sha256=sha256_file(reference),reproduction_sha256=sha256_file(actual))


def compare(reference=ROOT, output=ROOT/'runs/reproduction'):
    reference,output = Path(reference).resolve(),Path(output).resolve()
    if reference==output:
        raise ValueError('Reference and reproduction must be distinct directories')
    tables = {name:compare_tables(reference/name,output/name) for name in TABLES}
    manifest = json.loads((output/'pca/manifests/run.json').read_text())
    from pca.run import cache_paths
    spectrum_files,spectrum_values = 0,0
    for stage in manifest['config']['stages']:
        for layer in manifest['config']['layers']:
            actual,_ = cache_paths(output,manifest['identity'],stage,layer)
            expected = reference/f'pca/spectra/{stage}/layer_{layer:02d}.npz'
            with np.load(expected,allow_pickle=False) as left,np.load(actual,allow_pickle=False) as right:
                if set(left.files)!=set(right.files):
                    raise ValueError(f'Spectrum array coverage differs: {stage}/{layer}')
                for name in left.files:
                    if left[name].shape!=right[name].shape or right[name].dtype!=np.float64:
                        raise ValueError(f'Spectrum dimensions/precision differ: {stage}/{layer}/{name}')
                    np.testing.assert_allclose(right[name],left[name],rtol=1e-10,atol=1e-12,
                                               err_msg=f'{stage}/{layer}/{name}')
                    spectrum_values += left[name].size
            spectrum_files += 1
    result = dict(status='passed',verified_utc=utc_now(),rtol=1e-10,atol=1e-12,
                  component_counts='exact',tables=tables,spectrum_files=spectrum_files,
                  spectrum_values=spectrum_values,pca_identity=manifest['identity'])
    atomic_json(output/'pca/reference_comparison.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/reproduction')
    args=parser.parse_args()
    print(json.dumps(compare(args.reference_dir,args.output_dir)),flush=True)


if __name__=='__main__':
    main()
