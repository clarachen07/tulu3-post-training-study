from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Any

from .io import stable_int
from .segmentation import semantic_clean_steps, split_math_steps


@dataclass
class EncodedTrajectory:
    input_ids: list[int]
    labels: list[int]
    boundary_positions: list[int]
    steps_kept: int


def _bos_ids(tokenizer: Any) -> list[int]:
    if tokenizer.bos_token_id is None:
        return []
    return [int(tokenizer.bos_token_id)]


def encode_training_trajectory(
    tokenizer: Any,
    question: str,
    steps: list[str],
    step_token_id: int,
    max_length: int,
) -> EncodedTrajectory | None:
    """Build [question] STEP [step1] STEP ... at complete step boundaries."""
    ids = _bos_ids(tokenizer)
    question_ids = tokenizer.encode(str(question).strip(), add_special_tokens=False)
    ids.extend(question_ids)
    if not question_ids:
        return None

    ids.append(step_token_id)
    boundaries = [len(ids) - 1]
    answer_start = len(ids)
    kept = 0
    eos = [] if tokenizer.eos_token_id is None else [int(tokenizer.eos_token_id)]

    for step in steps:
        step_ids = tokenizer.encode(str(step).strip(), add_special_tokens=False)
        if not step_ids:
            continue
        candidate = ids + step_ids + [step_token_id]
        if len(candidate) + len(eos) > max_length:
            break
        ids = candidate
        boundaries.append(len(ids) - 1)
        kept += 1

    if kept < 2 or len(boundaries) < 3:
        return None
    if eos and len(ids) < max_length:
        ids.extend(eos)
    labels = [-100] * answer_start + ids[answer_start:]
    return EncodedTrajectory(ids, labels, boundaries, kept)


def encode_geometry_trajectory(
    tokenizer: Any,
    question: str,
    steps: list[str],
    max_length: int,
    boundary_mode: str,
    step_token_id: int,
    literal_marker: str,
) -> tuple[list[int], list[int]] | None:
    """Encode plain trajectories and return exact boundary hidden-state indices."""
    ids = _bos_ids(tokenizer)
    ids.extend(tokenizer.encode(str(question).strip(), add_special_tokens=False))
    if not ids:
        return None

    if boundary_mode == "natural":
        boundaries = [len(ids) - 1]
        for step in steps:
            piece = tokenizer.encode("\n\n" + str(step).strip(), add_special_tokens=False)
            if not piece or len(ids) + len(piece) > max_length:
                break
            ids.extend(piece)
            boundaries.append(len(ids) - 1)
    elif boundary_mode in {"reserved", "literal"}:
        marker_ids = (
            [step_token_id]
            if boundary_mode == "reserved"
            else tokenizer.encode(literal_marker, add_special_tokens=False)
        )
        if not marker_ids:
            raise ValueError(f"Marker has no token ids: {boundary_mode}")
        if len(ids) + len(marker_ids) > max_length:
            return None
        ids.extend(marker_ids)
        boundaries = [len(ids) - 1]
        for step in steps:
            step_ids = tokenizer.encode(str(step).strip(), add_special_tokens=False)
            candidate = step_ids + marker_ids
            if not step_ids or len(ids) + len(candidate) > max_length:
                break
            ids.extend(candidate)
            boundaries.append(len(ids) - 1)
    else:
        raise ValueError(f"Unknown boundary mode: {boundary_mode}")

    if len(boundaries) < 3:
        return None
    return ids, boundaries


def generate_triples(
    condition: str,
    token_count: int,
    boundary_positions: list[int],
    record_id: str,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed + stable_int(record_id))
    consecutive = [
        [boundary_positions[index - 1], boundary_positions[index], boundary_positions[index + 1]]
        for index in range(1, len(boundary_positions) - 1)
    ]
    target_count = max(1, len(consecutive))

    if condition in {"B1", "B2"}:
        return []
    if condition in {"A", "A1"}:
        return consecutive

    if condition in {"A2", "A2-match"}:
        population = boundary_positions
    elif condition in {"C", "C-match"}:
        # Restrict STP to the answer region so question length cannot dominate.
        start = boundary_positions[0]
        population = range(start, token_count)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    count = target_count if condition.endswith("-match") else 1
    if len(population) < 3:
        raise ValueError(f"Fewer than three candidate positions for {record_id}")
    chosen: set[tuple[int, int, int]] = set()
    attempts = 0
    unique_target = min(count, math.comb(len(population), 3))
    while len(chosen) < unique_target and attempts < max(count * 20, 100):
        chosen.add(tuple(sorted(rng.sample(population, 3))))
        attempts += 1
    result = list(sorted(chosen))
    while len(result) < count:
        # Replacement is only reached for unusually short trajectories.
        result.append(tuple(sorted(rng.sample(population, 3))))
    return [list(map(int, triple)) for triple in result]


def segment_record_solution(record: dict[str, Any], clean: bool = False) -> tuple[list[str], str]:
    solution = record.get("solution", record.get("answer", record.get("response", "")))
    segmentation = semantic_clean_steps(solution) if clean else split_math_steps(solution)
    return segmentation.steps, segmentation.mode
