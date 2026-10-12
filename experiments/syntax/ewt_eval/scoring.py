"""No label repairs; whole-dataset denominators and separately audited author scores."""
import collections
import io
import re

from .common import author_path

UPOS = set("ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split())
DEPREL = set("acl advcl advmod amod appos aux case cc ccomp clf compound conj cop csubj dep det discourse dislocated expl fixed flat goeswith iobj list mark nmod nsubj nummod obj obl orphan parataxis punct reparandum root vocative xcomp".split())
MARKER = re.compile(r"^\s*-\s*Task\s+([123])\s*$")


def blocks(text):
    result, block = [], []
    for line in text.splitlines():
        if "\t" in line:
            block.append(line.split("\t"))
        elif block:
            result.append(block)
            block = []
    if block:
        result.append(block)
    return result


def parse_tasks(text):
    marked = collections.defaultdict(list)
    active, section = None, []
    any_marker = False
    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            any_marker = True
            if active is not None:
                marked[active].append("\n".join(section))
            active, section = int(m.group(1)), []
        elif active is not None:
            section.append(line)
    if active is not None:
        marked[active].append("\n".join(section))
    result = {}
    for task in [1, 2, 3]:
        width = task + 2
        if any_marker:
            sections = marked.get(task, [])
            if len(sections) != 1:
                result[task] = {"rows": [], "failure": "missing_task" if not sections else "ambiguous_task"}
                continue
            candidates = blocks(sections[0])
        else:
            candidates = [b for b in blocks(text) if all(len(row) == width for row in b)]
        if len(candidates) != 1:
            result[task] = {"rows": [], "failure": "missing_table" if not candidates else "ambiguous_table"}
        else:
            result[task] = {"rows": candidates[0], "failure": None}
    return result


def matches_form(pred, gold):
    return pred == gold["form"] or (gold["space_after"] and pred == gold["form"] + " ")


def aligned_rows(task, gold, width):
    ids = collections.Counter()
    indexed = {}
    issues = collections.Counter()
    for row in task["rows"]:
        if len(row) != width:
            issues["wrong_field_count"] += 1
        if not row or not re.fullmatch(r"[1-9][0-9]*", row[0]):
            issues["invalid_id"] += 1
            continue
        idx = int(row[0])
        ids[idx] += 1
        if len(row) != width:
            continue
        if not 1 <= idx <= len(gold):
            issues["out_of_range_id"] += 1
            continue
        if not matches_form(row[1], gold[idx - 1]):
            issues["form_mismatch"] += 1
            continue
        indexed[idx] = row
    for idx, count in ids.items():
        if count > 1:
            indexed.pop(idx, None)
            issues["duplicate_id"] += count
    complete = not task["failure"] and not issues and len(task["rows"]) == len(gold) and len(indexed) == len(gold)
    return indexed, complete, dict(issues)


def head_value(row, n):
    if len(row) < 4 or not re.fullmatch(r"[0-9]+", row[3]):
        return None
    value = int(row[3])
    return value if 0 <= value <= n else None


def tree_diagnostics(rows, n):
    raw = {}
    for idx, row in rows.items():
        if re.fullmatch(r"-?[0-9]+", row[3]):
            raw[idx] = int(row[3])
    invalid = sum(h < 0 or h > n for h in raw.values())
    selfloops = sum(i == h for i, h in raw.items())
    complete = len(raw) == n and invalid == 0
    roots = sum(h == 0 for h in raw.values())
    cycle = False
    for start in raw:
        seen, cur = set(), start
        while cur in raw and cur != 0:
            if cur in seen:
                cycle = True
                break
            seen.add(cur)
            cur = raw[cur]
    return {"head_out_of_range": invalid, "self_loop_tokens": selfloops, "head_table_complete": complete, "roots": roots, "multiple_roots": roots > 1, "zero_roots": complete and roots == 0, "cycle": cycle, "valid_tree": complete and roots == 1 and not cycle}


def strict_sentence(gold, output, finish_reason=None, ignore_punct=False):
    tasks = parse_tasks(output)
    indexed, complete, issues = {}, {}, {}
    for task in [1, 2, 3]:
        indexed[task], complete[task], issues[task] = aligned_rows(tasks[task], gold, task + 2)
    count = collections.Counter()
    for g in gold:
        if ignore_punct and g["upos"] == "PUNCT":
            continue
        count["gold"] += 1
        for task in [1, 2, 3]:
            row = indexed[task].get(g["id"])
            if row is None:
                continue
            count[f"task{task}_aligned"] += 1
            count[f"task{task}_valid_upos"] += row[2] in UPOS
            count[f"task{task}_upos"] += row[2] in UPOS and row[2] == g["upos"]
            if task >= 2:
                h = head_value(row, len(gold))
                head_ok = h is not None and h != g["id"] and h == g["head"]
                count[f"task{task}_uas"] += head_ok
                if task == 3:
                    valid = row[4].split(":")[0] in DEPREL
                    count["task3_valid_deprel"] += valid
                    count["task3_las"] += head_ok and valid and row[4].split(":")[0] == g["deprel"].split(":")[0]
                    count["task3_las_subtypes"] += head_ok and valid and row[4] == g["deprel"]
        r2, r3 = indexed[2].get(g["id"]), indexed[3].get(g["id"])
        if r2 and r3:
            h2, h3 = head_value(r2, len(gold)), head_value(r3, len(gold))
            if h2 is not None and h3 is not None:
                count["head_consistency_eligible"] += 1
                count["head_consistent"] += h2 == h3
    diagnostics = {"task_failures": {str(t): tasks[t]["failure"] for t in tasks}, "table_complete": {str(t): complete[t] for t in complete}, "table_issues": {str(t): issues[t] for t in issues}, "truncated": finish_reason == "length", "tree": tree_diagnostics(indexed[3], len(gold))}
    return {"counts": dict(count), "diagnostics": diagnostics}


def author_task1(gold, output, ignore_punct=False):
    # Explicit extension: same row-count gate and positional ID/FORM recovery as author.
    author_path()
    from llmpp.eval import is_punctuation
    task = parse_tasks(output)[1]
    rows = []
    for row in task["rows"]:
        try:
            if len(row) != 3:
                break
            int(row[0])
            rows.append(row)
        except ValueError:
            break
    accepted = len(rows) == len(gold)
    if not accepted:
        rows = []
    g = sum(not ignore_punct or t["upos"] != "PUNCT" for t in gold)
    c = sum(not ignore_punct or not is_punctuation(row[2]) for row in rows)
    m = sum((not ignore_punct or t["upos"] != "PUNCT") and row[2] == t["upos"] for t, row in zip(gold, rows))
    return {"gold": g, "content": c, "correct": m, "accepted": accepted}


def author_messages(sentence, output):
    return sentence["messages"] + [{"role": "assistant", "gold": sentence["gold_output"], "content": output}]


def author_scores(sentences, outputs):
    author_path()
    from llmpp.eval import eval as evaluate, parse_records
    from llmpp.utils import select_last_tsv_part
    records = [author_messages(s, o) for s, o in zip(sentences, outputs)]
    result = {}
    for ignore in [False, True]:
        for subtypes in [False, True]:
            key = ("no_punct" if ignore else "with_punct") + ("_subtypes" if subtypes else "")
            # Vendor may crash on certain malformed outputs; preserve output and
            # count that sentence as empty under its own row-count gate, recording it.
            safe, exceptions = [], []
            for s, messages in zip(sentences, records):
                try:
                    evaluate([messages], 0, subtypes, None, ignore_punct=ignore)
                    safe.append(messages)
                except Exception as e:
                    empty = author_messages(s, "")
                    safe.append(empty)
                    exceptions.append({"sent_id": s["sent_id"], "error": repr(e)})
            scored = evaluate(safe, 0, subtypes, None, ignore_punct=ignore)
            scored["adapter_exceptions"] = exceptions
            tokens = scored["token"]
            denom = tokens["gold"] + tokens["content"]
            scored["metrics"] = {metric: (2 * tokens[field] / denom if denom else 0) for metric, field in [("UPOS", "correct_upos"), ("UAS", "correct_head"), ("LAS", "correct_head_deprel")]}
            result[key] = scored
    audit = collections.Counter()
    for s, output in zip(sentences, outputs):
        raw = select_last_tsv_part(output)
        try:
            parsed = parse_records(output, 0)
        except Exception:
            parsed = []
        audit["row_count_discard_sentences"] += len(parsed) != len(s["gold"])
        # Mirror width interpretation when counting author HEAD repairs.
        f2_isdigit = bool(raw and len(raw[0]) > 2 and raw[0][2].isdigit())
        for row in raw[:len(parsed)]:
            if len(row) not in [4, 5, 6]:
                continue
            hfield = 2 if len(row) == 4 or f2_isdigit else 3
            dfield = 3 if len(row) == 4 else (5 if len(row) == 6 and not f2_isdigit else 4)
            try:
                h = int(row[hfield])
            except ValueError:
                continue
            audit["negative_head_repairs"] += h < 0
            audit["head_upper_bound_repairs"] += h > len(parsed)
            audit["root_label_repairs"] += h == 0 and row[dfield] != "root"
    result["repair_audit"] = dict(audit)
    return result


def summarize_strict(rows):
    totals = collections.Counter()
    diag = collections.Counter()
    failures = collections.Counter()
    for row in rows:
        totals.update(row["counts"])
        d = row["diagnostics"]
        diag["sentences"] += 1
        diag["truncated"] += d["truncated"]
        for task in [1, 2, 3]:
            diag[f"task{task}_table_complete"] += d["table_complete"][str(task)]
            reason = d["task_failures"][str(task)]
            if reason:
                failures[f"task{task}:{reason}"] += 1
        for key, value in d["tree"].items():
            diag[key] += value
    denom = totals["gold"]
    metrics = {k: totals[k] / denom if denom else 0 for k in ["task1_upos", "task2_upos", "task2_uas", "task3_upos", "task3_uas", "task3_las", "task3_las_subtypes", "task1_aligned", "task2_aligned", "task3_aligned", "task1_valid_upos", "task3_valid_deprel"]}
    metrics["head_consistency"] = totals["head_consistent"] / totals["head_consistency_eligible"] if totals["head_consistency_eligible"] else None
    return {"counts": dict(totals), "metrics": metrics, "diagnostic_counts": dict(diag), "failure_counts": dict(failures)}
