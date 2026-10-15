"""Meaningful controls for boundaries, padding, raw hooks, PCA, and interrupted writes."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from truth_pca.common import immutable_manifest
from truth_pca.engine import capture
from truth_pca.inputs import encode_statement, padded_batch
from truth_pca.pca import get_pca
from truth_pca.raw import save_chunk, validate_chunk


class BoundaryTokenizer:
    """Offsets deliberately include multicharacter punctuation and native suffixes."""

    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, native=False):
        self.chat_template = "native" if native else None
        self.tokens = {}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = "<|user|>\n" + messages[0]["content"] + "\n<|assistant|>\n"
        return self(text, add_special_tokens=False)["input_ids"] if tokenize else text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets, i = [], [], 0
        if add_special_tokens:
            ids.append(1)
            offsets.append((0, 0))
        while i < len(text):
            size = 2 if text[i : i + 2] in (".\n", "'.") else 1
            if text[i : i + 3] == "'.\n":
                size = 3
            token = text[i : i + size]
            index = (
                10 + list(self.tokens.values()).index(token)
                if token in self.tokens.values()
                else 10 + len(self.tokens)
            )
            self.tokens[index] = token
            ids.append(index)
            offsets.append((i, i + size))
            i += size
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids):
        return "".join(self.tokens.get(i, "<BOS>") for i in ids)


@pytest.mark.parametrize("statement", ["A city is here.", "The word means '猫'."])
@pytest.mark.parametrize("stage", ["base", "sft"])
def test_final_period_offsets_preserve_native_text(statement, stage):
    tok = BoundaryTokenizer(stage != "base")
    record = encode_statement(tok, stage, statement)
    end = record["statement_span"][1]
    a, b = record["punctuation_offset"]
    assert record["serialized"][a:b].startswith(
        "'." if statement.endswith("'.") else "."
    )
    assert a <= end - 1 < b
    assert record["punctuation_contains_template_suffix"] == (stage != "base")
    assert record["statement"] == statement


def test_left_padding_anchors_and_position_ids():
    records = [
        {"input_ids": [1, 3, 4], "punctuation_position": 2},
        {"input_ids": [1, 8, 9, 7, 6], "punctuation_position": 3},
    ]
    batch, positions = padded_batch(records, 2, "cpu")
    assert positions == [4, 3]
    assert batch["input_ids"].tolist() == [[2, 2, 1, 3, 4], [1, 8, 9, 7, 6]]
    assert batch["attention_mask"].tolist() == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
    assert batch["position_ids"][0].tolist() == [0, 0, 0, 1, 2]


def test_tiny_llama_hook_is_raw_block_not_final_norm():
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=40,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0,
        )
    ).eval()
    rows = [
        {"input_ids": [1, 3, 4], "punctuation_position": 2},
        {"input_ids": [1, 8, 9, 7, 6], "punctuation_position": 3},
    ]
    batch, positions = padded_batch(rows, 2, "cpu")
    packed, final = capture(model, batch, positions, True)
    assert packed.shape == (3, 2, 24)
    assert not torch.allclose(packed[-1], final)
    assert torch.allclose(model.model.norm(packed[-1]), final, atol=1e-6)
    for i, row in enumerate(rows):
        single, anchor = padded_batch([row], 2, "cpu")
        value, _ = capture(model, single, anchor)
        assert torch.allclose(packed[:, i], value[:, 0], atol=2e-6)
    wrong, _ = capture(model, batch, [p - 1 for p in positions])
    relative = (wrong[0] - packed[0]).norm(dim=-1) / packed[0].norm(dim=-1)
    assert bool((relative > 8 * torch.finfo(torch.bfloat16).eps).all())
    assert all(not b._forward_hooks for b in model.model.layers)


def test_pca_matches_pinned_author_and_sign_rule():
    root = Path(__file__).resolve().parents[1]
    path = root / "third_party/geometry-of-truth/utils.py"
    spec = importlib.util.spec_from_file_location("pinned_author_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(19)
    x = torch.randn(150, 7) @ torch.diag(
        torch.tensor([9.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.5])
    )
    x -= x.mean(0)
    basis, fit_mean, eigenvalues, checks = get_pca(x)
    reference = module.get_pcs(x, 2)
    assert torch.allclose(basis @ basis.t(), reference @ reference.t(), atol=2e-5)
    assert torch.allclose(basis.t() @ basis, torch.eye(2), atol=2e-5)
    assert all(basis[torch.argmax(basis[:, i].abs()), i] > 0 for i in range(2))
    assert eigenvalues[0] > eigenvalues[1]


def test_joint_pca_centers_each_dataset_before_concatenation():
    torch.manual_seed(2)
    first, second = torch.randn(30, 5), torch.randn(40, 5) + 100
    centered = torch.cat([first - first.mean(0), second - second.mean(0)])
    basis, _, _, _ = get_pca(centered)
    assert torch.allclose((centered @ basis).mean(0), torch.zeros(2), atol=2e-5)


def test_manifest_rejects_changed_protocol(tmp_path):
    path = tmp_path / "manifest.json"
    identity = {"model_revision": "abc", "batch": 25}
    assert immutable_manifest(path, identity) == immutable_manifest(path, identity)
    with pytest.raises(RuntimeError):
        immutable_manifest(path, {**identity, "batch": 1})


def test_raw_bf16_roundtrip_and_corruption_rejection(tmp_path):
    torch.manual_seed(3)
    value = torch.randn(32, 2, 4096).to(torch.bfloat16)
    item = save_chunk(tmp_path, 0, value, "test_signature")
    assert validate_chunk(tmp_path, 0, 2, "test_signature") == item
    bits = np.load(tmp_path / item["filename"], allow_pickle=False)
    assert torch.equal(torch.from_numpy(bits).view(torch.bfloat16), value)
    with pytest.raises(RuntimeError):
        validate_chunk(tmp_path, 0, 2, "other_signature")
    path = tmp_path / item["filename"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(RuntimeError):
        validate_chunk(tmp_path, 0, 2, "test_signature")
