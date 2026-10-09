"""One frozen FP32 backbone forward per prompt; no answer generation."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from .common import (ROOT, atomic_json, load_config, read_jsonl, sha256_file,
                     utc_now, validate_frozen, protocol_digest, add_workspace_arguments, workspace)
from .audit_assets import check_assets


def verify_batch(path, expected_ids, digest, layers=32, hidden_size=4096, stage=None, mode=None):
    path = Path(path); sidecar = path.with_suffix('.json')
    if not path.exists() or not sidecar.exists(): return False
    meta = json.loads(sidecar.read_text())
    if meta['protocol_digest'] != digest or meta['sample_ids'] != list(expected_ids):
        raise RuntimeError(f'Batch identity mismatch: {path}')
    if stage is not None:
        audit = meta.get('validation') or {}
        if (audit.get('stage') != stage or audit.get('mode') != mode or
            audit.get('protocol_digest') != digest or not audit.get('fp32') or audit.get('tf32') is not False):
            raise RuntimeError(f'Batch stage/mode/precision identity mismatch: {path}')
    if sha256_file(path) != meta['sha256']: raise RuntimeError(f'Batch digest mismatch: {path}')
    with np.load(path, allow_pickle=False) as data:
        if data['sample_ids'].tolist() != list(expected_ids): raise RuntimeError('Batch IDs differ')
        x = data['hidden']
        if x.dtype != np.float32 or x.shape != (len(expected_ids), layers, hidden_size) or not np.isfinite(x).all():
            raise RuntimeError(f'Invalid activations: {path}')
    return True


def commit_batch(path, ids, hidden, digest, validation=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, sample_ids=np.asarray(ids), hidden=np.asarray(hidden, dtype=np.float32))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    atomic_json(path.with_suffix('.json'), dict(protocol_digest=digest, sample_ids=list(ids),
        shape=list(hidden.shape), dtype='float32', sha256=sha256_file(path),
        validation=validation, committed_utc=utc_now()))


class LastTokenCapture:
    def __init__(self, model):
        self.model = model; self.values = {}; self.index = None
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(model.layers)]

    def _hook(self, index):
        def hook(module, args, output):
            import torch
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dtype != torch.float32: raise RuntimeError(f'Non-FP32 block {index+1}')
            self.values[index] = hidden[0, self.index, :].detach().to('cpu').clone()
        return hook

    def run(self, input_ids, attention_mask, check_norm=False):
        import torch
        self.values = {}
        valid = torch.nonzero(attention_mask[0], as_tuple=False).flatten()
        if not len(valid): raise RuntimeError('Empty prompt')
        self.index = int(valid[-1])
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=False, output_hidden_states=False, return_dict=True)
        if sorted(self.values) != list(range(len(self.model.layers))):
            raise RuntimeError('Missing decoder hook output')
        states = torch.stack([self.values[i] for i in range(len(self.model.layers))])
        error = None
        if check_norm:
            expected = out.last_hidden_state[0, self.index].detach().cpu()
            # Calls the actual norm through its offload hooks when its weight is on CPU/meta.
            actual = self.model.norm(states[-1][None, None, :].to(input_ids.device))[0, 0].detach().cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            error = float((actual - expected).abs().max())
        return states.numpy(), error

    def close(self):
        for h in self.handles: h.remove()


def main():
    p = argparse.ArgumentParser(); add_workspace_arguments(p)
    p.add_argument('--stage', required=True, choices=['base', 'sft', 'dpo', 'rlvr'])
    p.add_argument('--pilot', action='store_true'); args = p.parse_args()
    root = workspace(args); config = load_config(root); inf = config['inference']
    validate_frozen(root); check_assets(root); digest = protocol_digest(root)
    model_config = next(x for x in config['models'] if x['stage'] == args.stage)
    records = read_jsonl(root / f'data/tokenized/{args.stage}.jsonl')
    if args.pilot: records = records[:inf['pilot_size']]
    mode = 'pilot' if args.pilot else 'formal'
    target = root / ('pilot' if args.pilot else 'activations') / args.stage
    target.mkdir(parents=True, exist_ok=True)
    completion = target / 'complete.json'; batch_size = inf['write_batch_size']
    batches = [(start, records[start:start+batch_size]) for start in range(0, len(records), batch_size)]
    missing = []
    for start, rows in batches:
        file = target / f'batch_{start:06d}.npz'
        if not verify_batch(file, [r['sample_id'] for r in rows], digest, stage=args.stage, mode=mode): missing.append((start, rows))
    if not missing and completion.exists():
        previous = json.loads(completion.read_text())
        if (previous['protocol_digest'] != digest or previous['stage'] != args.stage or previous['mode'] != mode or
                previous['completed_samples'] != len(records)):
            raise RuntimeError('Completion identity differs')
        if args.pilot and (len(previous['norm_errors']) != len(records) or previous['repeat_max_abs'] != 0):
            raise RuntimeError('Incomplete pilot validation evidence')
        print(f'{args.stage} {"pilot" if args.pilot else "formal"}: verified, nothing missing', flush=True); return
    if not missing:
        last_start = batches[-1][0]
        recovered = json.loads((target / f'batch_{last_start:06d}.json').read_text()).get('validation')
        if not recovered or recovered['protocol_digest'] != digest:
            raise RuntimeError('Committed batch is missing its forward validation evidence')
        if args.pilot and (len(recovered['norm_errors']) != len(records) or recovered['repeat_max_abs'] != 0):
            raise RuntimeError('Incomplete pilot validation evidence')
        check_assets(root)
        recovered.update(completed_samples=len(records), completed_utc=utc_now(), recovered_finalization=True)
        atomic_json(completion, recovered)
        print('Recovered completion from committed batch validation', flush=True)
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import AutoModel
    torch.set_num_threads(4)
    torch.manual_seed(inf['seed']); torch.cuda.manual_seed_all(inf['seed'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this frozen protocol')
    started = time.monotonic()
    backbone = AutoModel.from_pretrained(model_config['path'], local_files_only=True, use_safetensors=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation=inf['attention_implementation'],
        device_map='auto', max_memory={0: f"{inf['gpu_weight_budget_gib']}GiB", 'cpu': f"{inf['cpu_weight_budget_gib']}GiB"},
        low_cpu_mem_usage=True)
    backbone.eval(); backbone.requires_grad_(False); backbone.config.use_cache = False
    if any(p.dtype != torch.float32 for p in backbone.parameters()): raise RuntimeError('Non-FP32 parameters')
    if len(backbone.layers) != inf['layers'] or backbone.config.hidden_size != inf['hidden_size']:
        raise RuntimeError('Architecture does not match frozen protocol')
    if 'disk' in backbone.hf_device_map.values(): raise RuntimeError('Unexpected disk weight offload')
    capture = LastTokenCapture(backbone)
    audit = dict(protocol_digest=digest, stage=args.stage, mode=mode, fp32=True, tf32=False,
       device_map={k: str(v) for k,v in backbone.hf_device_map.items()}, norm_errors=[], repeat_max_abs=None)
    try:
        with torch.inference_mode():
            for start, rows in missing:
                results = []
                for row in rows:
                    ids = torch.tensor([row['input_ids']], dtype=torch.long, device='cuda:0')
                    mask = torch.tensor([row['attention_mask']], dtype=torch.long, device='cuda:0')
                    if row['readout_index'] != int(torch.nonzero(mask[0], as_tuple=False)[-1]):
                        raise RuntimeError('Readout index mismatch')
                    states, err = capture.run(ids, mask, check_norm=args.pilot)
                    if not np.isfinite(states).all(): raise RuntimeError('Nonfinite activations')
                    if args.pilot:
                        audit['norm_errors'].append(err)
                        if audit['repeat_max_abs'] is None:
                            again, _ = capture.run(ids, mask, check_norm=True)
                            if not np.array_equal(states, again): raise RuntimeError('Repeated FP32 forward differs')
                            audit['repeat_max_abs'] = float(np.max(np.abs(states-again)))
                    results.append(states)
                    print(f'{args.stage}/{mode} {row["sample_id"]}', flush=True)
                array = np.stack(results)
                audit.update(elapsed_seconds=time.monotonic()-started,
                             max_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                commit_batch(target / f'batch_{start:06d}.npz', [r['sample_id'] for r in rows], array, digest, audit)
    finally:
        capture.close()
    check_assets(root)
    audit.update(completed_samples=len(records), elapsed_seconds=time.monotonic()-started,
                 completed_utc=utc_now(), max_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(completion, audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__': main()
