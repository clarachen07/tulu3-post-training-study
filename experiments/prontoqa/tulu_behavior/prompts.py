"""Shared English content for the controlled proof and direct-answer conditions."""
from __future__ import annotations

from .scoring import ground_proof_steps


COMMON_INSTRUCTION = (
    "Solve each problem independently using only the facts and rules in that problem. "
    "Treat the invented category names as arbitrary symbols. "
    "A statement is True when it follows from the given facts and rules, and False when "
    "its explicit negation follows."
)
COT_INSTRUCTION = (
    "Write a short step-by-step proof in English. Write one factual statement per line, "
    "using the same names and predicates as the problem. Start from a given fact and "
    "derive each subsequent new fact by applying one given rule to facts already established. "
    "You do not need to repeat the rules. Do not skip intermediate facts or add commentary. "
    "Finish with exactly one separate line: Answer: True or Answer: False."
)
DIRECT_INSTRUCTION = (
    "Give only the answer, with no explanation. Write exactly one line: "
    "Answer: True or Answer: False."
)


def _question(item: dict) -> str:
    text = item["question"] + " " + item["query"]
    if "text" in item and item["text"] != text:
        raise ValueError("Item text contains unexpected content beyond the question and query.")
    return text


def demonstration_proof(item: dict) -> list[str]:
    """Keep atomic ground facts, omitting the intervening universally quantified rules."""
    steps = ground_proof_steps(item)
    if len(steps) != item["hop"] + 1:
        raise ValueError("Demonstration does not contain the expected atomic ground-fact chain.")
    return steps


def build_prompt(item: dict, demos: list[dict], condition: str) -> str:
    if condition not in ("cot", "direct"):
        raise ValueError(f"Unknown condition: {condition}")
    if len(demos) != 4 or [demo["answer"] for demo in demos] != ["True", "False", "True", "False"]:
        raise ValueError("Expected four fixed demonstrations in True, False, True, False order.")
    if any(demo["hop"] != 3 for demo in demos):
        raise ValueError("All demonstrations must use the fixed 3-hop condition.")
    questions = [_question(demo) for demo in demos]
    target = _question(item)
    if len(set(questions)) != len(questions) or target in questions:
        raise ValueError("Demonstrations must be distinct and disjoint from the target item.")
    instruction = COT_INSTRUCTION if condition == "cot" else DIRECT_INSTRUCTION
    blocks = [COMMON_INSTRUCTION + "\n" + instruction]
    for demo, question in zip(demos, questions):
        answer = f"Answer: {demo['answer']}"
        if condition == "cot":
            answer = "\n".join(demonstration_proof(demo) + [answer])
        blocks.append(f"Q: {question}\nA:\n{answer}")
    blocks.append(f"Q: {target}\nA:\n")
    return "\n\n".join(blocks)
