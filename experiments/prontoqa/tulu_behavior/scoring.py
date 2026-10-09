"""Deterministic answer and proof scoring for fictional Modus Ponens PrOntoQA.

The grammar is deliberately small and consumes each complete assertion. It
supports the fixed generator's singular/Each/Every/plural rule variants and
ground statements, including explicit negation. It never repairs a proof or
uses its reference answer to interpret model text.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable


_NOUNS = frozenset(("wumpus", "yumpus", "zumpus", "dumpus", "rompus",
                    "numpus", "tumpus", "vumpus", "impus", "jompus"))
_PLURALS = {word + "es": word for word in _NOUNS}
_WORD = r"[A-Za-z][A-Za-z-]*"
_LITERAL = re.compile(rf"(?:(not)\s+)?(?:(a|an)\s+)?({_WORD})", re.I)
_RULE_SINGULAR = re.compile(rf"(?:Every|Each)\s+({_WORD})\s+is\s+(.+)", re.I)
_RULE_PLURAL = re.compile(rf"(?:All\s+)?({_WORD})\s+are\s+(.+)", re.I)
_GROUND = re.compile(rf"({_WORD})\s+is\s+(.+)", re.I)
_ANSWER_PREFIX = re.compile(r"(?:Final\s+)?Answer\s*:", re.I)
_ANSWER_LINE = re.compile(r"(?:Final\s+)?Answer\s*:\s*(True|False)\s*[.!]?", re.I)
_BARE_ANSWER = re.compile(r"(True|False)\s*[.!]?", re.I)
_PROOF_HEADER = re.compile(r"(?:Reasoning|Proof)\s*:", re.I)
_LIST_PREFIX = re.compile(r"^(?:[-*]\s+|\d+[.)]\s+)")


@dataclass(frozen=True)
class Literal:
    predicate: str
    negated: bool = False

    def opposite(self) -> Literal:
        return Literal(self.predicate, not self.negated)


@dataclass(frozen=True)
class Atom:
    entity: str
    literal: Literal

    def opposite(self) -> Atom:
        return Atom(self.entity, self.literal.opposite())


@dataclass(frozen=True)
class Rule:
    antecedent: Literal
    consequent: Literal


Formula = Atom | Rule


def _literal(text: str, *, plural: bool = False) -> Literal:
    match = _LITERAL.fullmatch(text.strip())
    if match is None:
        raise ValueError("Expected one unary predicate, optionally explicitly negated")
    negation, article, word = match.groups()
    word = word.lower()
    if article:
        if word not in _NOUNS or plural:
            raise ValueError("An article must precede a singular ontology noun")
    elif word in _PLURALS:
        if not plural:
            raise ValueError("A ground individual requires a singular predicate")
        word = _PLURALS[word]
    elif word in _NOUNS:
        raise ValueError("A singular ontology noun requires an article")
    elif word in {"true", "false", "and", "or", "not"}:
        raise ValueError("A label or connective is not a predicate")
    return Literal(word, bool(negation))


def parse_statement(text: str) -> Formula:
    """Parse one complete assertion; trailing punctuation is optional."""
    text = text.strip()
    if text.endswith((".", ";")):
        text = text[:-1].rstrip()
    if not text:
        raise ValueError("Empty assertion")
    match = _RULE_SINGULAR.fullmatch(text)
    if match:
        noun, consequent = match.groups()
        if noun.lower() not in _NOUNS:
            raise ValueError("Rule subject is outside the fixed ontology grammar")
        return Rule(Literal(noun.lower()), _literal(consequent))
    match = _RULE_PLURAL.fullmatch(text)
    if match:
        noun, consequent = match.groups()
        if noun.lower() not in _PLURALS:
            raise ValueError("A plural rule needs an ontology plural subject")
        return Rule(Literal(_PLURALS[noun.lower()]), _literal(consequent, plural=True))
    match = _GROUND.fullmatch(text)
    if match:
        entity, consequent = match.groups()
        if entity.lower() in {"every", "each", "all", "answer", "reasoning", "proof"}:
            raise ValueError("Invalid individual name")
        return Atom(entity.lower(), _literal(consequent))
    raise ValueError("Assertion is outside the controlled PrOntoQA grammar")


def _assertions(text: str) -> list[str]:
    """Retain every nonempty fragment, including a tail without punctuation."""
    fragments = []
    for line in text.splitlines():
        line = _LIST_PREFIX.sub("", line.strip(), count=1)
        fragments.extend(part.strip() for part in re.split(r"[.;]", line) if part.strip())
    return fragments


def _context(item: dict) -> tuple[set[Atom], set[Rule], Atom, Atom, list[str]]:
    statements = _assertions(item["question"])
    if not statements:
        raise ValueError("Question has no axioms")
    formulas = [parse_statement(part) for part in statements]
    facts = {formula for formula in formulas if isinstance(formula, Atom)}
    rules = {formula for formula in formulas if isinstance(formula, Rule)}
    query_text = item["query"]
    prefix = "True or false: "
    if not query_text.startswith(prefix):
        raise ValueError("Expected the original True or false query")
    query = parse_statement(query_text[len(prefix):])
    if not isinstance(query, Atom):
        raise ValueError("Query must describe an individual")
    if item["answer"] not in {"True", "False"}:
        raise ValueError("Gold answer must be True or False")
    target = query if item["answer"] == "True" else query.opposite()
    return facts, rules, query, target, statements


def _closure(facts: Iterable[Atom], rules: Iterable[Rule]) -> dict[Atom, int]:
    """Minimum number of MP applications from the supplied input facts."""
    distances = {fact: 0 for fact in facts}
    rules = tuple(rules)
    while True:
        updated = False
        for fact, distance in list(distances.items()):
            for rule in rules:
                if fact.literal == rule.antecedent:
                    consequence = Atom(fact.entity, rule.consequent)
                    if consequence not in distances or distances[consequence] > distance + 1:
                        distances[consequence] = distance + 1
                        updated = True
        if not updated:
            return distances


def _rule_entailed(candidate: Rule, rules: set[Rule]) -> bool:
    synthetic = Atom("__rule_subject__", candidate.antecedent)
    return Atom(synthetic.entity, candidate.consequent) in _closure({synthetic}, rules)


def _check_proof(item: dict, statements: list[str]) -> dict:
    facts, rules, _, target, _ = _context(item)
    closure = _closure(facts, rules)
    known = set(facts)
    seen: set[Formula] = set()
    diagnostics = []
    reached_target = False
    for index, text in enumerate(statements, 1):
        step = {"index": index, "text": text}
        try:
            formula = parse_statement(text)
        except ValueError as error:
            step.update(status="parse_error", detail=str(error))
            diagnostics.append(step)
            continue
        if formula in seen:
            # Only already accepted assertions enter seen. Repeated unsupported
            # assertions remain errors and cannot become valid by repetition.
            step["status"] = "redundant"
        elif formula in facts or formula in rules:
            step["status"] = "axiom"
        elif isinstance(formula, Atom) and any(
            rule.consequent == formula.literal
            and Atom(formula.entity, rule.antecedent) in known
            for rule in rules
        ):
            step["status"] = "derived"
        elif ((isinstance(formula, Atom) and formula in closure)
              or (isinstance(formula, Rule) and _rule_entailed(formula, rules))):
            step["status"] = "skip"
            if isinstance(formula, Atom):
                step["minimum_hops_from_context"] = closure[formula]
        else:
            step["status"] = "logic_error"
        if step["status"] in {"axiom", "derived", "redundant"}:
            seen.add(formula)
            if isinstance(formula, Atom):
                known.add(formula)
            reached_target |= formula == target
        diagnostics.append(step)
    counts = Counter(step["status"] for step in diagnostics)
    parseable = bool(statements) and counts["parse_error"] == 0
    success = bool(statements) and parseable and reached_target and not (
        counts["skip"] or counts["logic_error"])
    return {
        "proof_success": success,
        "proof_parseable": parseable,
        "has_skip": bool(counts["skip"]),
        "has_logic_error": bool(counts["logic_error"]),
        "step_count": len(statements),
        "diagnostics": {
            "proof_steps": diagnostics,
            "parse_error_count": counts["parse_error"],
            "skip_count": counts["skip"],
            "logic_error_count": counts["logic_error"],
            "redundant_count": counts["redundant"],
            "reached_target": reached_target,
            "missing_conclusion": not reached_target,
            "empty_proof": not statements,
        },
    }


def _unbold(line: str) -> str:
    if line.startswith("**") and line.endswith("**") and len(line) > 4:
        return line[2:-2].strip()
    return line


def _extract_answer(text: str) -> dict:
    lines = [(index, _unbold(line.strip())) for index, line in enumerate(text.splitlines(), 1)
             if line.strip()]
    last_line = lines[-1][0] if lines else None
    candidates = []
    labels = set()
    invalid_marked = False
    proof_lines = []
    header_count = 0
    consumed_answer_lines = []
    for line_number, line in lines:
        marked = _ANSWER_PREFIX.match(line)
        valid = _ANSWER_LINE.fullmatch(line) if marked else _BARE_ANSWER.fullmatch(line)
        if marked or valid:
            label = valid.group(1).title() if valid else None
            candidates.append({"line": line_number, "text": line, "value": label,
                               "marked": bool(marked), "final": line_number == last_line})
            if label:
                labels.add(label)
                consumed_answer_lines.append(line_number)
            else:
                invalid_marked = True
                labels.update(value.title() for value in re.findall(r"\b(True|False)\b", line, re.I))
                # Never discard unconsumed prose on a malformed answer line.
                proof_lines.append(line)
        elif _PROOF_HEADER.fullmatch(line):
            header_count += 1
        else:
            proof_lines.append(line)
    acceptable = [candidate for candidate in candidates
                  if candidate["value"] and (candidate["marked"] or candidate["final"])]
    if len(labels) > 1:
        status, answer = "conflicting", None
    elif invalid_marked:
        status, answer = "invalid", None
    elif not acceptable:
        status, answer = "missing", None
    else:
        status, answer = "ok", acceptable[-1]["value"]
    # A bare label away from the last line is not a permitted answer delimiter.
    # Keep it in the proof so that it cannot make malformed output disappear.
    for candidate in candidates:
        if candidate["value"] and not candidate["marked"] and not candidate["final"]:
            proof_lines.append(candidate["text"])
    final_answer = bool(acceptable and acceptable[-1]["line"] == last_line)
    return {"extracted_answer": answer, "answer_status": status,
            "proof_text": "\n".join(proof_lines),
            "answer_candidates": candidates, "header_count": header_count,
            "single_final_answer": len(candidates) == 1 and final_answer and status == "ok",
            "trailing_after_answer": bool(consumed_answer_lines and
                                           max(consumed_answer_lines) != last_line)}


def score_response(item: dict, text: str, condition: str) -> dict:
    """Score one full model response without repairing or truncating its text."""
    if condition not in {"cot", "direct"}:
        raise ValueError("condition must be cot or direct")
    if not isinstance(text, str):
        raise TypeError("Model response must be text")
    answer = _extract_answer(text)
    correct = answer["answer_status"] == "ok" and answer["extracted_answer"] == item["answer"]
    result = {"extracted_answer": answer["extracted_answer"], "answer_correct": correct,
              "answer_status": answer["answer_status"]}
    if condition == "cot":
        result.update(_check_proof(item, _assertions(answer["proof_text"])))
        result["joint_success"] = correct and result["proof_success"]
        result["format_compliant"] = bool(answer["single_final_answer"]
                                           and answer["header_count"] <= 1
                                           and result["step_count"]
                                           and result["proof_parseable"])
    else:
        result.update(proof_success=None, joint_success=None, proof_parseable=None,
                      has_skip=None, has_logic_error=None, step_count=0, diagnostics={})
        result["format_compliant"] = bool(answer["single_final_answer"]
                                           and not answer["proof_text"].strip()
                                           and not answer["header_count"])
    result["diagnostics"].update(answer_candidates=answer["answer_candidates"],
                                  trailing_after_answer=answer["trailing_after_answer"])
    return result


def ground_proof_steps(item: dict) -> list[str]:
    """Return gold individual assertions in source order, preserving wording."""
    return [sentence for sentence in item["chain_of_thought"]
            if isinstance(parse_statement(sentence), Atom)]


def validate_item(item: dict) -> dict:
    """Validate a generated item; return JSON statistics or raise ValueError."""
    facts, rules, query, target, statements = _context(item)
    if not facts or not rules:
        raise ValueError("Expected both facts and rules")
    if not isinstance(item.get("hop"), int) or isinstance(item["hop"], bool) or item["hop"] < 1:
        raise ValueError("hop must be a positive integer")
    closure = _closure(facts, rules)
    if target not in closure or target.opposite() in closure:
        raise ValueError("Gold label must have exactly one provable query polarity")
    if closure[target] != item["hop"]:
        raise ValueError(f"Minimum proof depth {closure[target]} does not equal hop {item['hop']}")
    gold = item.get("chain_of_thought")
    if not isinstance(gold, list) or not gold or any(not isinstance(step, str) or not step.strip() for step in gold):
        raise ValueError("Gold proof must be a nonempty list of assertions")
    if parse_statement(gold[-1]) != target:
        raise ValueError("Gold proof must end with the required query polarity")
    scored = _check_proof(item, gold)
    if not scored["proof_success"]:
        raise ValueError(f"Gold proof is not a complete strict proof: {scored['diagnostics']}")
    if "text" in item and item["text"] != item["question"] + " " + item["query"]:
        raise ValueError("Stored text differs from question plus query")
    return {"rule_count": len(rules), "fact_count": len(facts),
            "sentence_count": len(statements), "gold_step_count": len(gold),
            "gold_ground_step_count": len(ground_proof_steps(item)),
            "min_hops": closure[target], "query_negated": query.literal.negated,
            "target_negated": target.literal.negated, "query_entity": query.entity,
            "entity_count": len({fact.entity for fact in facts}),
            "lexical_token_count": len(re.findall(r"\b[\w-]+\b", item["question"] + " " + item["query"]))}
