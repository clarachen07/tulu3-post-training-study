from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer

from .io import read_jsonl


class SSPDataset(Dataset):
    def __init__(self, path: str | Path, condition: str):
        self.condition = condition
        self.records = list(read_jsonl(path))
        if not self.records:
            raise ValueError(f"No training records in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "input_ids": record["input_ids"],
            "labels": record["labels"],
            "triples": record["triples"][self.condition],
            "record_id": record["record_id"],
        }


class SSPCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            maximum = (
                (maximum + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of
        max_triples = max(1, max(len(feature["triples"]) for feature in features))
        input_ids, labels, attention_mask, triples, triple_mask = [], [], [], [], []
        record_ids = []
        for feature in features:
            length = len(feature["input_ids"])
            padding = maximum - length
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            attention_mask.append([1] * length + [0] * padding)
            feature_triples = list(feature["triples"])
            triples.append(feature_triples + [[0, 0, 0]] * (max_triples - len(feature_triples)))
            triple_mask.append([1] * len(feature_triples) + [0] * (max_triples - len(feature_triples)))
            record_ids.append(feature["record_id"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "triples": torch.tensor(triples, dtype=torch.long),
            "triple_mask": torch.tensor(triple_mask, dtype=torch.bool),
            "record_ids": record_ids,
        }


def unwrap_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


class SSPTrainer(Trainer):
    def __init__(self, *args: Any, ntp: bool, stp: bool, beta: float, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.use_ntp = bool(ntp)
        self.use_stp = bool(stp)
        self.beta = float(beta)
        self._component_sum = {"ntp_loss": 0.0, "stp_loss": 0.0}
        self._component_count = 0
        self._wall_start = time.time()

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        del num_items_in_batch
        record_ids = inputs.pop("record_ids", None)
        triples = inputs.pop("triples")
        triple_mask = inputs.pop("triple_mask")
        labels = inputs.pop("labels")
        causal_lm = unwrap_causal_lm(model)
        transformer = causal_lm.model
        outputs = transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        zero = hidden.sum() * 0.0

        ntp_loss = zero
        logits = None
        if self.use_ntp:
            logits = causal_lm.lm_head(hidden)
            ntp_loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().float().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        stp_loss = zero
        if self.use_stp:
            batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None, None]
            vectors = hidden[batch, triples]
            before = vectors[:, :, 1] - vectors[:, :, 0]
            after = vectors[:, :, 2] - vectors[:, :, 1]
            scores = 1.0 - F.cosine_similarity(before.float(), after.float(), dim=-1, eps=1e-8)
            if not torch.any(triple_mask):
                raise RuntimeError(f"No STP triples in batch: {record_ids}")
            stp_loss = scores[triple_mask].mean()

        total = ntp_loss + self.beta * stp_loss
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite loss: total={total}, ntp={ntp_loss}, stp={stp_loss}, records={record_ids}"
            )
        self._component_sum["ntp_loss"] += float(ntp_loss.detach())
        self._component_sum["stp_loss"] += float(stp_loss.detach())
        self._component_count += 1
        result = {"logits": logits, "last_hidden_state": hidden}
        return (total, result) if return_outputs else total

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._component_count and ("loss" in logs or "train_loss" in logs):
            logs.setdefault("global_step", int(self.state.global_step))
            logs.setdefault("max_steps", int(self.state.max_steps))
            logs.setdefault("epoch", float(self.state.epoch or 0.0))
            logs.setdefault("elapsed_wall_seconds", time.time() - self._wall_start)
            for key, value in self._component_sum.items():
                logs[key] = value / self._component_count
            self._component_sum = {"ntp_loss": 0.0, "stp_loss": 0.0}
            self._component_count = 0
            print("SSP_METRIC " + json.dumps(logs, sort_keys=True), flush=True)
        super().log(logs, *args, **kwargs)
