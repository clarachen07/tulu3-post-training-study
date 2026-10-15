"""Exact cities strings from the author's pinned patching.py."""

from __future__ import annotations

FALSE_PROMPT = (
    "The city of Tokyo is in Japan. This statement is: TRUE\n"
    "The city of Hanoi is in Poland. This statement is: FALSE\n"
    "The city of Chicago is in Canada. This statement is:"
)
TRUE_PROMPT = FALSE_PROMPT.replace("Chicago", "Toronto")


def build_prompt(tokenizer, stage):
    native = stage != "base"
    texts, sequences = [], []
    for raw in (FALSE_PROMPT, TRUE_PROMPT):
        if native:
            messages = [{"role": "user", "content": raw}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            if tokenizer(text, add_special_tokens=False).input_ids != ids:
                raise ValueError("Rendered template does not reproduce its token IDs")
        else:
            if tokenizer.chat_template:
                raise ValueError("Frozen Base unexpectedly has a chat template")
            text = raw
            ids = tokenizer(text, add_special_tokens=True).input_ids
            if ids.count(tokenizer.bos_token_id) != 1:
                raise ValueError("Base must contain exactly one BOS")
        texts.append(text)
        sequences.append(list(ids))
    false_ids, true_ids = sequences
    if len(false_ids) != len(true_ids):
        raise ValueError("True/false prompts have unequal token counts")
    differences = [i for i, (a, b) in enumerate(zip(false_ids, true_ids)) if a != b]
    expected = (51, 37) if native else (42, 33)
    if (len(false_ids), differences) != (expected[0], [expected[1]]):
        raise ValueError(
            f"Frozen tokenization differs: {len(false_ids)}, {differences}"
        )
    tokens = [
        {"position": i, "id": token, "text": tokenizer.decode([token])}
        for i, token in enumerate(false_ids)
    ]
    city = differences[0]
    country, period = city + 3, city + 4
    if (
        tokens[city]["text"] != " Chicago"
        or tokens[country]["text"] != " Canada"
        or tokens[period]["text"] != "."
    ):
        raise ValueError("Semantic anchors do not match the frozen strings")
    forms = {
        "spaced": {"true": " TRUE", "false": " FALSE"},
        "bare": {"true": "TRUE", "false": "FALSE"},
    }
    candidates = {}
    for form, values in forms.items():
        candidates[form] = {}
        for label, value in values.items():
            ids = tokenizer(value, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise ValueError("Candidate is not a single token")
            for text, seq in zip(texts, sequences):
                if (
                    tokenizer(text + value, add_special_tokens=not native).input_ids
                    != seq + ids
                ):
                    raise ValueError("Candidate changes the frozen prefix")
            candidates[form][label] = ids[0]
    if candidates != {
        "spaced": {"true": 8378, "false": 7989},
        "bare": {"true": 21260, "false": 31451},
    }:
        raise ValueError("Unexpected candidate token IDs")
    return dict(
        false_text=texts[0],
        true_text=texts[1],
        false_ids=false_ids,
        true_ids=true_ids,
        tokens=tokens,
        positions=list(range(city, len(false_ids))),
        anchors=dict(
            city=city, country=country, period=period, readout=len(false_ids) - 1
        ),
        primary_form="bare" if native else "spaced",
        candidate_ids=candidates,
        chat_template=tokenizer.chat_template if native else None,
        chat_wrapper_positions=list(range(city + 9, len(false_ids))) if native else [],
        chat_boundary_positions=[city + 8] if native else [],
        differences=differences,
        interface="native" if native else "plain",
    )
