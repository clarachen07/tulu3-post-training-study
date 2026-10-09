import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str, record: dict):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_input_text(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"WARNING: chat template failed, falling back to raw prompt. Error: {e}")
            return prompt
    return prompt


def load_model_and_tokenizer(model_name_or_path: str, dtype_name: str, device: str):
    print(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = get_torch_dtype(dtype_name)

    print(f"Loading model: {model_name_or_path}")
    print(f"dtype={dtype_name}, device={device}")

    load_kwargs = {
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }

    if device == "auto":
        load_kwargs["device_map"] = "auto"
    elif device == "cuda":
        load_kwargs["device_map"] = {"": "cuda:0"}
    elif device == "cpu":
        load_kwargs["device_map"] = {"": "cpu"}
    else:
        raise ValueError(f"Unsupported device: {device}")

    if torch_dtype != "auto":
        # transformers 4.x commonly uses torch_dtype; newer versions may prefer dtype.
        load_kwargs["torch_dtype"] = torch_dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except TypeError as e:
        if "torch_dtype" in load_kwargs:
            load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        else:
            raise e

    model.eval()
    return tokenizer, model


def generate_one(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    use_chat_template: bool,
):
    input_text = build_input_text(tokenizer, prompt, use_chat_template)

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    torch.manual_seed(seed)
    random.seed(seed)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[0, input_len:]
    generation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "input_text": input_text,
        "generation": generation,
        "full_text": full_text,
        "num_prompt_tokens": input_len,
        "num_generated_tokens": int(generated_ids.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-stage", required=True)
    parser.add_argument("--input-path", default="data/math500_pilot_20_prompts.jsonl")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-samples-per-problem", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--use-chat-template", action="store_true")
    args = parser.parse_args()

    examples = list(read_jsonl(args.input_path))
    if args.limit is not None:
        examples = examples[: args.limit]

    print(f"Loaded {len(examples)} prompts from {args.input_path}")

    tokenizer, model = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        dtype_name=args.dtype,
        device=args.device,
    )

    output_path = Path(args.output_path)
    if output_path.exists():
        print(f"WARNING: output file already exists and will be appended: {output_path}")

    total = 0

    for ex_idx, ex in enumerate(examples):
        prompt = ex["prompt"]

        for sample_id in range(args.num_samples_per_problem):
            current_seed = args.seed + ex_idx * 1000 + sample_id

            print(
                f"Generating problem {ex_idx + 1}/{len(examples)}, "
                f"sample {sample_id + 1}/{args.num_samples_per_problem}, "
                f"seed={current_seed}"
            )

            result = generate_one(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=current_seed,
                use_chat_template=args.use_chat_template,
            )

            record = {
                "pilot_id": ex.get("pilot_id"),
                "problem_id": ex.get("problem_id"),
                "subject": ex.get("subject"),
                "level": ex.get("level"),
                "gold_answer": ex.get("answer"),
                "prompt": prompt,
                "model_name_or_path": args.model_name_or_path,
                "model_stage": args.model_stage,
                "sample_id": sample_id,
                "generation": result["generation"],
                "full_text": result["full_text"],
                "num_prompt_tokens": result["num_prompt_tokens"],
                "num_generated_tokens": result["num_generated_tokens"],
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": current_seed,
                "use_chat_template": args.use_chat_template,
            }

            append_jsonl(args.output_path, record)
            total += 1

    print(f"Done. Wrote {total} generations to {args.output_path}")


if __name__ == "__main__":
    main()
