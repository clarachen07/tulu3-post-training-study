"""Freeze exact inputs, audit checkpoints and run the four-stage GPU evaluation."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gc
from importlib import metadata
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from protocol import (
    ROOT, STAGES, MODEL_NAMES, REGISTRY, SHARD_SIZE, TASKS, MAX_NEW_TOKENS, STOP, Decoder,
    atomic_json, encode, fingerprint, freeze_data, hops, label_for, prompt_parts,
    read_gzip, read_json, sha256_file, verify_files, write_gzip,
)
from tulu_behavior.assets import loader_candidates, stat_identity

BATCH_SIZES = (1, 4, 8, 16, 32, 64)
SHARED_COMPUTE_ENV = "TULU_COMPOSITIONALITY_SHARED_GPU"
SHARED_MEMORY_FRACTION = 0.78
PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "accelerate", "pyarrow", "jinja2")
SHARED_CODE = (
    ROOT.parent / "prontoqa/tulu_behavior/__init__.py",
    ROOT.parent / "prontoqa/tulu_behavior/common.py",
    ROOT.parent / "prontoqa/tulu_behavior/assets.py",
)


def environment(packages=PACKAGES):
    return {"python": platform.python_version(), **{name: metadata.version(name) for name in packages}}


def code_identity():
    paths = [ROOT / name for name in ("protocol.py", "runner.py", "experiment.py", "requirements.txt")]
    paths.extend(SHARED_CODE)
    return {str(path.relative_to(ROOT.parent)): sha256_file(path) for path in sorted(paths)}


def model_audit(root, model_dir):
    path = root / "models.json"
    registry = {model["stage"]: model for model in read_json(REGISTRY)["models"]}
    if path.exists():
        audit = read_json(path)
        if audit["registry_sha256"] != sha256_file(REGISTRY):
            raise ValueError("Checkpoint registry changed")
        for model in audit["models"]:
            directory = (model_dir / MODEL_NAMES[model["stage"]]).resolve()
            if str(directory) != model["directory"] or loader_candidates(directory) != list(model["files"]):
                raise ValueError(f"Checkpoint path or loader candidates changed: {directory}")
            for name, record in model["files"].items():
                if stat_identity(directory / name) != record["stat"]:
                    raise ValueError(f"Checkpoint changed after audit: {directory / name}")
        return audit
    models = []
    for stage in STAGES:
        reference = registry[stage]
        directory = (model_dir / MODEL_NAMES[stage]).resolve()
        files = loader_candidates(directory)
        expected = reference["files"]
        required = {"config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
                    "model.safetensors.index.json"}
        required.update(name for name in expected if name.endswith(".safetensors"))
        if not required <= set(files) <= expected.keys():
            raise ValueError(f"Checkpoint loader files do not match the registry: {directory}")
        proofs = {}
        for name in files:
            file = directory / name
            before = stat_identity(file)
            digest = sha256_file(file)
            if digest != expected[name] or before != stat_identity(file):
                raise ValueError(f"Checkpoint hash differs: {file}")
            proofs[name] = {"sha256": digest, "stat": before}
        models.append({"stage": stage, "id": reference["id"], "revision": reference["revision"],
                       "directory": str(directory), "files": proofs})
        print(f"Audited {stage}: {len(proofs)} checkpoint files", flush=True)
    audit = {"registry_sha256": sha256_file(REGISTRY), "models": models}
    atomic_json(path, audit)
    return audit


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                         use_fast=True)


def dataset_sources(data, spec, task):
    for index in spec["full"]:
        demos = [data[demo] for demo in spec["demos"][index]]
        for hop_index, hop in enumerate(hops(task)):
            yield (index, hop_index, prompt_parts(task, hop, data[index], demos))


def freeze_inputs(root, model_dir):
    dataset, queries, data = freeze_data(root)
    if model_dir is None:
        return data
    models = model_audit(root, model_dir)
    identity = {
        "code": code_identity(), "models": fingerprint(models), "data": fingerprint(data),
        "environment": environment(("transformers", "tokenizers", "pyarrow", "jinja2")),
    }
    manifest_path = root / "inputs.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["identity"] != identity:
            raise ValueError("Frozen input identity changed. Use a new run directory.")
        verify_files(root, old["files"])
        return old
    files = {}
    partition = {}
    for model in models["models"]:
        stage = model["stage"]
        tokenizer = load_tokenizer(model["directory"])
        partition[stage] = {}
        for task, rows in dataset.items():
            requests = list(dataset_sources(rows, queries[task], task))
            frozen = []
            for start in range(0, len(requests), 256):
                batch = requests[start:start + 256]
                ids = encode(tokenizer, stage, [parts for _, _, parts in batch])
                frozen.extend([[row, hop, tokens] for (row, hop, _), tokens in zip(batch, ids, strict=True)])
            # Similar lengths reduce masked prefill work. Query IDs preserve the original pairing.
            frozen.sort(key=lambda request: (len(request[2]), request[0], request[1]))
            shards = []
            for start in range(0, len(frozen), SHARD_SIZE):
                relative = f"inputs/{stage}/{task}/{start:06d}.json.gz"
                write_gzip(root / relative, frozen[start:start + SHARD_SIZE])
                files[relative] = sha256_file(root / relative)
                shards.append(relative)
            partition[stage][task] = shards
        print(f"Froze all prompts for {stage}", flush=True)
    manifest = {"identity": identity, "files": files, "partition": partition}
    atomic_json(manifest_path, manifest)
    return manifest


def smi(query):
    result = subprocess.run(["nvidia-smi", "-i", "0", *query, "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True)
    return result.stdout.strip()


def shared_compute():
    return os.environ.get(SHARED_COMPUTE_ENV) == "1"


@contextmanager
def workspace_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Run workspace is in use: {root}") from error
        yield


@contextmanager
def gpu_admission(lock_path):
    with Path(lock_path).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The BNU shared GPU lock is busy") from error
        apps = smi(["--query-compute-apps=pid"])
        free_mib = int(smi(["--query-gpu=memory.free"]))
        if free_mib < 23000 or (apps and not shared_compute()):
            raise RuntimeError(f"GPU has active compute processes or insufficient free memory ({free_mib} MiB)")
        if apps:
            print(f"Shared GPU admission: compute PIDs {apps.splitlines()}, {free_mib} MiB free", flush=True)
        yield


class Generator:
    def __init__(self, directory):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig, StopStringCriteria

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
            raise RuntimeError("Select one BF16-capable CUDA GPU")
        if shared_compute():
            torch.cuda.set_per_process_memory_fraction(SHARED_MEMORY_FRACTION, 0)
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.tokenizer = load_tokenizer(directory)
        self.decode = Decoder(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        ).eval()
        if any(parameter.dtype != torch.bfloat16 or parameter.device.type != "cuda"
               for parameter in self.model.parameters()):
            raise RuntimeError("Checkpoint was not loaded entirely in BF16 on CUDA")
        eos = self.model.generation_config.eos_token_id
        eos = eos if eos is not None else self.tokenizer.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        self.device = self.model.device
        self.eos = torch.tensor(eos, device=self.device)
        pad = self.tokenizer.pad_token_id
        self.pad = eos[0] if pad is None else pad
        self.settings = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
            eos_token_id=eos, pad_token_id=self.pad, bos_token_id=self.tokenizer.bos_token_id,
            repetition_penalty=1.0,
        )
        self.stop = StopStringCriteria(self.tokenizer, [STOP])

    def __call__(self, batch):
        from transformers import StoppingCriteria, StoppingCriteriaList
        torch = self.torch
        width = max(len(tokens) for _, _, tokens in batch)
        count = len(batch)
        input_ids = torch.tensor([[self.pad] * (width - len(tokens)) + tokens for _, _, tokens in batch],
                                 device=self.device)
        mask = torch.tensor([[0] * (width - len(tokens)) + [1] * len(tokens) for _, _, tokens in batch],
                            device=self.device)
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        reasons = torch.zeros_like(lengths)
        eos, stop = self.eos, self.stop

        class Tracker(StoppingCriteria):
            def __call__(self, ids, scores, **kwargs):
                suffix = ids[:, width:]
                eos_hit = (suffix[:, -1:] == eos).any(dim=-1)
                string_hit = stop(suffix, scores)
                hit = (eos_hit | string_hit) & (lengths == 0)
                lengths.masked_fill_(hit, suffix.shape[1])
                reasons.masked_fill_(hit & eos_hit, 1)
                reasons.masked_fill_(hit & ~eos_hit, 2)
                return eos_hit | string_hit

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=mask, generation_config=self.settings,
                stopping_criteria=StoppingCriteriaList([Tracker()]),
            )
        ids = output[:, width:].tolist()
        reason_names = ("length", "eos", "stop")
        return [(tokens[:length or MAX_NEW_TOKENS], reason_names[reason])
                for tokens, length, reason in zip(ids, lengths.tolist(), reasons.tolist(), strict=True)]


def get_runtime(root, inputs):
    return {"input_manifest_sha256": sha256_file(root / "inputs.json"), "source": inputs["identity"],
            "dtype": "bfloat16", "attention": "sdpa", "max_new_tokens": MAX_NEW_TOKENS, "stop": STOP,
            "shared_cuda_allocator_fraction": SHARED_MEMORY_FRACTION if shared_compute() else None,
            "environment": environment(), "hardware": smi(["--query-gpu=name,uuid,driver_version"])}


def pilot_sources(root, inputs, stage):
    chosen = []
    for task, (count, _, _) in TASKS.items():
        # Two frozen queries per task exercise every hop and both answer boundaries.
        picked = set(random.Random(0).sample(range(count), 2))
        for name in inputs["partition"][stage][task]:
            chosen.extend(row for row in read_gzip(root / name) if row[0] in picked)
    return sorted(chosen, key=lambda row: len(row[2]))


def pilot(root, inputs, models, runtime):
    import torch
    path = root / "pilot.json"
    if path.exists():
        candidate = read_json(path)
        if candidate["runtime"] != runtime:
            raise ValueError("Pilot runtime changed. Use a new run directory.")
        return candidate
    records = {}
    for model in models["models"]:
        stage = model["stage"]
        sources = pilot_sources(root, inputs, stage)
        generator = Generator(model["directory"])
        generator(sources[:1])
        records[stage] = {}
        for size in BATCH_SIZES:
            started = time.monotonic()
            try:
                generated = [row for start in range(0, len(sources), size)
                             for row in generator(sources[start:start + size])]
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                records[stage][str(size)] = {"status": "oom"}
                continue
            if size == 1:
                reference = generated
            mismatches = sum(left != right for left, right in zip(reference, generated, strict=True))
            trial = {"status": "ok", "seconds": time.monotonic() - started, "n": len(sources),
                     "token_or_stop_mismatches": mismatches, "generated_sha256": fingerprint(generated)}
            records[stage][str(size)] = trial
            print(f"Pilot {stage} batch={size}: {trial}", flush=True)
        del generator
        gc.collect()
        torch.cuda.empty_cache()
    eligible = [size for size in BATCH_SIZES if all(
        (trial := records[stage][str(size)])["status"] == "ok"
        and trial["token_or_stop_mismatches"] == 0 for stage in STAGES
    )]
    selected = min(eligible, key=lambda s: sum(records[stage][str(s)]["seconds"] for stage in STAGES))
    candidate = {"runtime": runtime, "stages": records, "batch_size": selected}
    atomic_json(path, candidate)
    print(f"Selected common batch {selected}", flush=True)
    return candidate


def prediction_path(root, input_name):
    return root / input_name.replace("inputs/", "predictions/", 1)


def load_shard(root, name, inputs, run):
    path = prediction_path(root, name)
    envelope = read_gzip(path)
    if (envelope["run"] != fingerprint(run) or envelope["input_sha256"] != inputs["files"][name]
            or envelope["rows_sha256"] != fingerprint(envelope["rows"])):
        raise ValueError(f"Committed prediction shard differs: {path}")
    return envelope["rows"]


def stage_run(root, inputs, model, dataset, run):
    stage = model["stage"]
    generator = None
    total = 0
    started = time.monotonic()
    size = run["batch_size"]
    for task, names in inputs["partition"][stage].items():
        for name in names:
            sources = read_gzip(root / name)
            if prediction_path(root, name).exists():
                existing = load_shard(root, name, inputs, run)
                if [(row[0], row[1]) for row in existing] != [(row[0], row[1]) for row in sources]:
                    raise ValueError(f"Committed predictions do not match the frozen requests: {name}")
            else:
                if generator is None:
                    generator = Generator(model["directory"])
                records = []
                for start in range(0, len(sources), size):
                    batch = sources[start:start + size]
                    for (index, hop_index, _), (ids, reason) in zip(batch, generator(batch), strict=True):
                        raw, continuation, tokenwise = generator.decode(ids)
                        label = label_for(task, hops(task)[hop_index], dataset[task][index])
                        records.append([index, hop_index, ids, reason, label, raw, continuation, tokenwise])
                write_gzip(prediction_path(root, name), {
                    "run": fingerprint(run), "input_sha256": inputs["files"][name],
                    "rows_sha256": fingerprint(records), "rows": records,
                })
            total += len(sources)
            atomic_json(root / "status" / f"{stage}.json",
                        {"stage": stage, "predictions": total, "task": task, "last_shard": name})
        print(f"{stage} {task}: {total} committed ({time.monotonic() - started:.1f} s)", flush=True)
    del generator


def execute(root, model_dir, lock_path, stage=None, pilot_only=False):
    import torch
    inputs = freeze_inputs(root, model_dir)
    models = model_audit(root, model_dir)
    dataset, _, _ = freeze_data(root)
    with gpu_admission(lock_path):
        runtime = get_runtime(root, inputs)
        run = pilot(root, inputs, models, runtime)
        if pilot_only:
            return
        for model in models["models"]:
            if stage is None or model["stage"] == stage:
                stage_run(root, inputs, model, dataset, run)
                gc.collect()
                torch.cuda.empty_cache()
