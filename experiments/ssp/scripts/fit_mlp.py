#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ssp_tulu.io import atomic_json, read_json, set_global_seed, stable_int
from ssp_tulu.mlp import ProbeExamples, ResidualPredictor, normalized_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--train-geometry", required=True)
    parser.add_argument("--eval-geometry", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-mode", choices=["problem", "pair"], default="problem")
    return parser.parse_args()


def load_examples(directory: Path, horizon: int, final_layer: int) -> ProbeExamples:
    manifest = read_json(directory / "manifest.json")
    previous, current, target, baseline = [], [], [], []
    problem_ids, pair_ids = [], []
    for shard_info in manifest["shards"]:
        records = torch.load(shard_info["path"], map_location="cpu", weights_only=False)
        for record in records:
            z = record["layer_states"][final_layer].float()
            for k in range(1, len(z) - horizon):
                previous.append(z[k - 1])
                current.append(z[k])
                target.append(z[k + horizon])
                baseline.append(z[k] + horizon * (z[k] - z[k - 1]))
                problem_ids.append(str(record["problem_id"]))
                pair_ids.append(str(record["pair_id"]))
    if not previous:
        raise RuntimeError(f"No probe examples in {directory} for horizon {horizon}")
    return ProbeExamples(
        previous=torch.stack(previous),
        current=torch.stack(current),
        target=torch.stack(target),
        baseline=torch.stack(baseline),
        problem_ids=problem_ids,
        pair_ids=pair_ids,
    )


def split_name(identifier: str, seed: int, mode: str) -> str:
    bucket = stable_int(f"{seed}:{mode}:{identifier}", modulo=100)
    if mode == "problem":
        if bucket < 80:
            return "train"
        if bucket < 90:
            return "dev"
        return "test"
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "dev"
    return "test"


def indices_for(examples: ProbeExamples, seed: int, mode: str, split: str) -> list[int]:
    identifiers = examples.problem_ids if mode == "problem" else examples.pair_ids
    return [
        index
        for index, identifier in enumerate(identifiers)
        if split_name(identifier, seed, mode) == split
    ]


def evaluate(model, examples: ProbeExamples, device: torch.device, batch_size: int) -> dict:
    dataset = TensorDataset(
        examples.previous, examples.current, examples.target, examples.baseline
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    linear_errors, mlp_errors = [], []
    model.eval()
    with torch.inference_mode():
        for previous, current, target, baseline in loader:
            previous = previous.to(device)
            current = current.to(device)
            target = target.to(device)
            baseline = baseline.to(device)
            prediction = baseline + model(current, previous)
            linear_errors.append(normalized_error(baseline, target).cpu())
            mlp_errors.append(normalized_error(prediction, target).cpu())
    linear = torch.cat(linear_errors).numpy()
    mlp = torch.cat(mlp_errors).numpy()
    return {
        "n": int(len(linear)),
        "linear_mse": float(np.mean(linear)),
        "mlp_mse": float(np.mean(mlp)),
        "mlp_over_linear": float(np.mean(mlp) / np.mean(linear)),
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty MLP output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    settings = config["mlp"]
    seed = int(settings["split_seed"])
    set_global_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    final_layer = max(map(int, config["geometry"]["layers"]))
    summaries = {}
    for horizon in map(int, config["geometry"]["horizons"]):
        train_all = load_examples(Path(args.train_geometry), horizon, final_layer)
        eval_all = (
            train_all
            if Path(args.train_geometry).resolve() == Path(args.eval_geometry).resolve()
            else load_examples(Path(args.eval_geometry), horizon, final_layer)
        )
        train_examples = train_all.subset(
            indices_for(train_all, seed, args.split_mode, "train")
        )
        dev_examples = train_all.subset(indices_for(train_all, seed, args.split_mode, "dev"))
        test_examples = eval_all.subset(indices_for(eval_all, seed, args.split_mode, "test"))
        dimension = train_examples.current.shape[-1]
        model = ResidualPredictor(
            dimension=dimension, hidden_width=int(settings["hidden_width"])
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        dataset = TensorDataset(
            train_examples.previous,
            train_examples.current,
            train_examples.target,
            train_examples.baseline,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + horizon),
        )
        best_state = copy.deepcopy(model.state_dict())
        best_dev = float("inf")
        stale = 0
        history = []
        for epoch in range(int(settings["epochs"])):
            model.train()
            losses = []
            for previous, current, target, baseline in loader:
                previous = previous.to(device)
                current = current.to(device)
                target = target.to(device)
                baseline = baseline.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = baseline + model(current, previous)
                loss = normalized_error(prediction, target).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            dev = evaluate(model, dev_examples, device, int(settings["batch_size"]))
            history.append(
                {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **dev}
            )
            print(
                json.dumps(
                    {
                        "event": "mlp_epoch",
                        "label": args.label,
                        "horizon": horizon,
                        **history[-1],
                    }
                ),
                flush=True,
            )
            if dev["mlp_mse"] < best_dev:
                best_dev = dev["mlp_mse"]
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= int(settings["patience"]):
                    break
        model.load_state_dict(best_state)
        test_metrics = evaluate(model, test_examples, device, int(settings["batch_size"]))
        torch.save(model.state_dict(), output / f"predictor-m{horizon}.pt")
        summaries[str(horizon)] = {
            "train_examples": len(train_examples.current),
            "dev_examples": len(dev_examples.current),
            "test_examples": len(test_examples.current),
            "best_dev_mse": best_dev,
            "test": test_metrics,
            "history": history,
        }
    atomic_json(
        output / "summary.json",
        {
            "label": args.label,
            "split_mode": args.split_mode,
            "train_geometry": str(Path(args.train_geometry).resolve()),
            "eval_geometry": str(Path(args.eval_geometry).resolve()),
            "settings": settings,
            "horizons": summaries,
        },
    )
    atomic_json(output / "complete.json", {"ok": True})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

