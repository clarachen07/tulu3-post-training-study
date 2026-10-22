"""Freeze final token IDs and hashes for every model and every question."""
from collections import Counter
from transformers import AutoTokenizer
from common import local_path, ROOT, digest_file, fingerprint, load_json, read_jsonl, utc, write_json, write_jsonl


def main():
    cfg = load_json(ROOT / "experiment.json")
    destination = ROOT / "manifests/tokenization.json"
    if destination.exists():
        meta = load_json(destination)
        assert meta["config_hash"] == fingerprint(cfg)
        for name, h in meta["files"].items():
            assert digest_file(ROOT / name) == h
        print("Existing tokenized panel verified", flush=True)
        return
    files, details = {}, []
    for model in cfg["models"]:
        tokenizer = AutoTokenizer.from_pretrained(str(local_path(model["path"])), local_files_only=True, use_fast=True)
        lengths, bos_counts = [], Counter()
        for phase in ["pilot", "formal"]:
            rows = []
            for r in read_jsonl(ROOT / f"data/prepared/{phase}.jsonl"):
                if model["wrapper"] == "plain":
                    text = r["content"] + "\nAnswer:"
                    ids = tokenizer.encode(text, add_special_tokens=True)
                else:
                    messages = [{"role": "user", "content": r["content"]}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    assert ids == tokenizer.encode(text, add_special_tokens=False)
                assert len(ids) + cfg["inference"]["max_tokens"] <= cfg["inference"]["max_model_len"]
                assert not tokenizer.pad_token_id or tokenizer.pad_token_id not in ids
                lengths.append(len(ids))
                bos_counts[ids.count(tokenizer.bos_token_id)] += 1
                if model["wrapper"] == "plain":
                    assert ids.count(tokenizer.bos_token_id) == 1
                out = {k: r[k] for k in ["id", "split", "rlvr_row", "gold", "question_hash"]}
                out.update({"phase": phase, "stage": model["stage"], "prompt_text": text,
                            "prompt_token_ids": ids, "prompt_text_hash": fingerprint(text),
                            "prompt_token_hash": fingerprint(ids)})
                rows.append(out)
            path = ROOT / f"data/tokenized/{model['stage']}/{phase}.jsonl"
            write_jsonl(path, rows)
            files[str(path.relative_to(ROOT))] = digest_file(path)
        details.append({"stage": model["stage"], "min_input_tokens": min(lengths),
                        "max_input_tokens": max(lengths), "bos_counts": dict(bos_counts),
                        "chat_template": tokenizer.chat_template if model["wrapper"] == "native" else None,
                        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
                        "pad_token_id": tokenizer.pad_token_id})
    write_json(destination, {"created_utc": utc(), "config_hash": fingerprint(cfg),
                             "files": files, "models": details})
    print(details, flush=True)


if __name__ == "__main__":
    main()
