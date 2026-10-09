from __future__ import annotations

import re
from dataclasses import dataclass


LATEX_ENV_PATTERN = re.compile(
    r"(\\begin\{(?P<env>align\*?|aligned|array|cases|gather\*?|equation\*?)\}"
    r".*?\\end\{(?P=env)\}|\[asy\].*?\[/asy\])",
    flags=re.DOTALL,
)
SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=(?:[A-Z0-9]|\\[A-Za-z]+|\$|Therefore|Thus|Hence|So))"
)


@dataclass(frozen=True)
class Segmentation:
    steps: list[str]
    mode: str


def _protect_latex_blocks(text: str) -> tuple[str, dict[str, str]]:
    blocks: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        key = f"SSPLATEXBLOCK{len(blocks)}TOKEN"
        blocks[key] = match.group(0)
        return key

    return LATEX_ENV_PATTERN.sub(replace, text), blocks


def _restore(text: str, blocks: dict[str, str]) -> str:
    for key, value in blocks.items():
        text = text.replace(key, value)
    return text


def _clean(parts: list[str], blocks: dict[str, str]) -> list[str]:
    restored = [_restore(part, blocks).strip() for part in parts]
    return [part for part in restored if part]


def split_math_steps(text: str, minimum_steps: int = 2) -> Segmentation:
    """Paragraph-first segmentation with LaTeX-aware sentence fallback."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return Segmentation([], "empty")

    protected, blocks = _protect_latex_blocks(normalized)
    paragraphs = _clean(re.split(r"\n[ \t]*\n+", protected), blocks)
    if len(paragraphs) >= minimum_steps:
        return Segmentation(paragraphs, "paragraph")

    # A single display-math line or an asy block remains attached to the
    # neighboring explanatory sentence. This avoids treating LaTeX internals
    # as semantic reasoning steps.
    sentence_parts = _clean(SENTENCE_BOUNDARY.split(protected), blocks)
    if len(sentence_parts) >= minimum_steps:
        return Segmentation(sentence_parts, "sentence_fallback")

    return Segmentation([normalized], "unsplittable")


def raw_paragraph_steps(text: str) -> Segmentation:
    """Strict frozen-analysis boundary definition: blank-line paragraphs only."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return Segmentation([], "empty")
    protected, blocks = _protect_latex_blocks(normalized)
    paragraphs = _clean(re.split(r"\n[ \t]*\n+", protected), blocks)
    return Segmentation(paragraphs, "paragraph")


def semantic_clean_steps(text: str, minimum_steps: int = 2) -> Segmentation:
    """Sensitivity segmentation: merge short fragments into adjacent steps."""
    primary = split_math_steps(text, minimum_steps=minimum_steps)
    if len(primary.steps) < minimum_steps:
        return primary
    merged: list[str] = []
    for index, step in enumerate(primary.steps):
        token_like_words = step.split()
        remaining = len(primary.steps) - index - 1
        if merged and len(token_like_words) < 4 and len(merged) + remaining >= minimum_steps:
            merged[-1] = f"{merged[-1]}\n{step}"
        else:
            merged.append(step)
    if len(merged) < minimum_steps:
        return primary
    return Segmentation(merged, f"{primary.mode}_semantic_clean")
