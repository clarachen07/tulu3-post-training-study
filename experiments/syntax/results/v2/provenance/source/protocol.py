"""Canonical task content, independent of a checkpoint's input wrapper."""
import hashlib
import random


UPOS = "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split()
DEPREL = "acl advcl advmod amod appos aux case cc ccomp clf compound conj cop csubj dep det discourse dislocated expl fixed flat goeswith iobj list mark nmod nsubj nummod obj obl orphan parataxis punct reparandum root vocative xcomp".split()
END_PARSE = "<END_PARSE>"
# These delimit a new input/role, never an ordinary row or blank line.
STOP_STRINGS = [
    END_PARSE,
    "\n### Input",
    "\n### Example",
    "\n### Instruction",
    "\nInput:\n",
    "\nheader: user",
    "\nheader: system",
    "<|user|>",
    "<|system|>",
    "<|start_header_id|>user<|end_header_id|>",
    "<|start_header_id|>system<|end_header_id|>",
]
HEAD_RULE = (
    "HEAD is the index of this token's immediate syntactic governor under "
    "Universal Dependencies, not the indexes of its dependents. Use 0 only "
    "for the sentence root. A token cannot be its own head."
)


def stable_rng(seed, *parts):
    """Separate deterministic streams without depending on Python's hash seed."""
    key = "\0".join([str(seed), *map(str, parts)]).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))


def input_only(sentence):
    """Never pass evaluation labels, legacy messages, or gold_output to prompts."""
    tokens = sentence.get("tokens", sentence.get("gold"))
    if tokens is None:
        raise ValueError("sentence must contain input tokens")
    clean = [{"id": t["id"], "form": t["form"]} for t in tokens]
    if [t["id"] for t in clean] != list(range(1, len(clean) + 1)):
        raise ValueError("input token IDs must be contiguous and one-based")
    return {"text": sentence["text"], "tokens": clean}


def _input_text(sentence):
    source = input_only(sentence)
    words = "\n".join(f"{t['id']}\t{t['form']}" for t in source["tokens"])
    return f"Sentence: {source['text']}\nTokens (ID, FORM):\n{words}"


def full_answer(gold):
    """One five-column table, base UD relations only, explicit end marker."""
    rows = [
        f"{t['id']}\t{t['form']}\t{t['upos']}\t{t['head']}\t{t['deprel'].split(':')[0]}"
        for t in gold
    ]
    return "\n".join([*rows, END_PARSE])


def full_prompt(sentence, example):
    return (
        "### Instruction\n"
        "Parse the supplied English sentence according to Universal Dependencies. "
        "Use exactly the supplied tokenization and token IDs.\n"
        "Return one TSV table with five fields per row: ID, FORM, UPOS, HEAD, DEPREL. "
        "Copy each ID and FORM exactly once, in order. Separate fields with literal tabs. "
        "Do not add column headers, code fences, Markdown tables, or explanations.\n"
        f"{HEAD_RULE} The complete tree must have exactly one root.\n"
        f"Allowed UPOS tags: {', '.join(UPOS)}.\n"
        f"Allowed DEPREL base labels: {', '.join(DEPREL)}. "
        "Use base labels only, without colon subtypes.\n"
        f"After the last row, write {END_PARSE} on a new line and stop.\n\n"
        f"### Example\n{_input_text(example)}\n### Parse\n"
        f"{full_answer(example['gold'])}\n\n"
        f"### Input\n{_input_text(sentence)}\n### Parse\n"
    )


def candidate_options(sentence, target_id, task):
    """Semantic options derived only from input IDs/forms and the tag inventory."""
    tokens = input_only(sentence)["tokens"]
    if target_id not in {t["id"] for t in tokens}:
        raise ValueError("target token is absent")
    if task == "upos":
        return [{"value": tag, "display": tag} for tag in UPOS]
    if task != "head":
        raise ValueError(f"unknown diagnostic task: {task}")
    return [{"value": 0, "display": "HEAD=0: ROOT"}] + [
        {"value": t["id"], "display": f"HEAD={t['id']}: {t['form']}"}
        for t in tokens if t["id"] != target_id
    ]


def code_candidates(options, order):
    """Order 1 shifts every option to a different code/position when K > 1."""
    if order not in (0, 1):
        raise ValueError("candidate order must be 0 or 1")
    if not options or len(options) > 82:
        raise ValueError("diagnostics support 1..82 options, using codes 100..181")
    shift = len(options) // 2 if order else 0
    ordered = options[shift:] + options[:shift]
    return [{"code": str(100 + i), **option} for i, option in enumerate(ordered)]


def choice_codes(request):
    codes = [c["code"] for c in request["candidates"]]
    if len(codes) != len(set(codes)):
        raise ValueError("candidate codes must be unique")
    return codes


def _candidate_table(candidates):
    return "\n".join(f"{c['code']}\t{c['display']}" for c in candidates)


def _grammar_question(sentence, target_id, task, candidates):
    source = input_only(sentence)
    target = next(t for t in source["tokens"] if t["id"] == target_id)
    question = (
        "Select this token's Universal Dependencies UPOS tag."
        if task == "upos" else "Select this token's HEAD. " + HEAD_RULE
    )
    return (
        f"{_input_text(source)}\nTarget: ID={target_id}, FORM={target['form']}\n"
        f"{question}\nCandidates (CODE, OPTION):\n{_candidate_table(candidates)}\n"
        "Answer:\n"
    )


def diagnostic_prompt(request, example):
    """Mapping controls never use the sentence, training example, or grammar gold."""
    codes = choice_codes(request)
    if not codes:
        raise ValueError("a diagnostic request needs candidates")
    if request["kind"] == "mapping":
        selected = next(c for c in request["candidates"] if c["value"] == request["mapping_target"])
        return (
            "Find the explicitly specified option in the table and return its CODE. "
            "This is a code lookup, not a grammar question. "
            "Output exactly one code, without explanation.\n"
            f"Candidates (CODE, OPTION):\n{_candidate_table(request['candidates'])}\n"
            f"Specified option: {selected['display']}\nAnswer:\n"
        )
    if request["kind"] != "grammar":
        raise ValueError("unknown diagnostic kind")
    if example.get("split") != "train":
        raise ValueError("the diagnostic example must come from the training split")
    task = request["task"]
    example_target = 3
    options = candidate_options(example, example_target, task)
    # The seed key excludes kind/order: both orders use the same demonstration
    # options, with the same pre-specified cyclic shift as the actual question.
    stable_rng(
        request.get("selection_seed", 42), "example-options", request["sent_id"],
        request["target_id"], task,
    ).shuffle(options)
    example_candidates = code_candidates(options, request["order"])
    gold = next(t for t in example["gold"] if t["id"] == example_target)
    answer = next(c["code"] for c in example_candidates if c["value"] == gold[task])
    return (
        "Choose the best option for the specified token. Return its CODE, not "
        "its label, token index, or word. Output exactly one code, without explanation.\n\n"
        f"Training example:\n{_grammar_question(example, example_target, task, example_candidates)}"
        f"{answer}\n\nQuestion:\n"
        f"{_grammar_question(request['sentence'], request['target_id'], task, request['candidates'])}"
    )
