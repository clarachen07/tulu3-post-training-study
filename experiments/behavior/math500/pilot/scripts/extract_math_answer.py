import argparse
import csv
import json
import re
from pathlib import Path


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_last_boxed(text: str) -> str | None:
    """
    Conservative parser for the last \\boxed{...}.
    Supports simple nested braces.
    """
    marker = r"\boxed{"
    starts = [m.start() for m in re.finditer(re.escape(marker), text)]
    if not starts:
        return None

    start = starts[-1]
    i = start + len(marker)
    depth = 1
    chars = []

    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
        i += 1

    return None


def extract_answer(generation: str) -> tuple[str, str]:
    """
    Return (answer, method).
    The extractor is intentionally conservative for pilot-stage evaluation.
    """
    if not generation:
        return "", "empty_generation"

    boxed = find_last_boxed(generation)
    if boxed:
        return boxed, "boxed"

    patterns = [
        r"final answer is\s*[:：]?\s*([^\n\.]+)",
        r"the answer is\s*[:：]?\s*([^\n\.]+)",
        r"answer\s*[:：]\s*([^\n\.]+)",
        r"答案是\s*[:：]?\s*([^\n。]+)",
    ]

    lower_text = generation.lower()
    for pat in patterns:
        m = re.search(pat, lower_text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(), "text_pattern"

    # Fallback: search last non-empty line for a simple number/fraction.
    lines = [ln.strip() for ln in generation.splitlines() if ln.strip()]
    for line in reversed(lines[-5:]):
        candidates = re.findall(
            r"[-+]?\d+(?:\.\d+)?|\\frac\{[-+]?\d+\}\{[-+]?\d+\}|[-+]?\d+/\d+",
            line,
        )
        if candidates:
            return candidates[-1].strip(), "last_line_number"

    return "", "not_found"


def strip_outer_boxed(s: str) -> str:
    s = s.strip()
    if s.startswith(r"\boxed{") and s.endswith("}"):
        return s[len(r"\boxed{") : -1].strip()
    return s


def normalize_answer(s: str) -> str:
    if s is None:
        return ""

    s = str(s).strip()
    s = strip_outer_boxed(s)

    # Remove common formatting noise.
    replacements = {
        "\n": "",
        "\t": "",
        " ": "",
        "$": "",
        ".": "",
        ",": "",
        r"\left": "",
        r"\right": "",
        r"\!": "",
        r"\,": "",
        r"\;": "",
        r"\:": "",
        r"\dfrac": r"\frac",
        r"\tfrac": r"\frac",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    # Remove trailing punctuation / sentence fragments.
    s = s.strip()
    s = re.sub(r"(squarecentimeters|cm\^2|cm2|units|unit)$", "", s, flags=re.IGNORECASE)

    # Normalize simple LaTeX fractions: \frac{a}{b}
    m = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den != 0 and num % den == 0:
            return str(num // den)
        return f"{num}/{den}"

    # Normalize simple a/b fractions.
    m = re.fullmatch(r"([-+]?\d+)/([-+]?\d+)", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den != 0 and num % den == 0:
            return str(num // den)
        return f"{num}/{den}"

    # Normalize integers like 054 -> 54.
    if re.fullmatch(r"[-+]?\d+", s):
        return str(int(s))

    # Normalize floats conservatively.
    if re.fullmatch(r"[-+]?\d+\.\d+", s):
        try:
            val = float(s)
            if val.is_integer():
                return str(int(val))
            return str(val)
        except Exception:
            pass

    return s


def evaluate_record(record: dict) -> dict:
    generation = record.get("generation", "")
    gold = record.get("gold_answer", record.get("answer", ""))

    extracted, method = extract_answer(generation)

    norm_extracted = normalize_answer(extracted)
    norm_gold = normalize_answer(gold)

    is_correct = bool(norm_extracted) and norm_extracted == norm_gold

    # For pilot: any mismatch or failed extraction should be manually reviewed.
    needs_manual_check = not is_correct

    out = dict(record)
    out.update(
        {
            "extracted_answer": extracted,
            "normalized_extracted_answer": norm_extracted,
            "normalized_gold_answer": norm_gold,
            "is_correct_auto": is_correct,
            "needs_manual_check": needs_manual_check,
            "extraction_method": method,
        }
    )
    return out


def write_summary_csv(path: str, records: list[dict]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "pilot_id",
        "problem_id",
        "subject",
        "level",
        "sample_id",
        "model_stage",
        "gold_answer",
        "extracted_answer",
        "normalized_extracted_answer",
        "normalized_gold_answer",
        "is_correct_auto",
        "needs_manual_check",
        "extraction_method",
        "num_generated_tokens",
        "generation_preview",
    ]

    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            row["generation_preview"] = str(r.get("generation", ""))[:200].replace("\n", " ")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    records = [evaluate_record(r) for r in read_jsonl(args.input_path)]

    write_jsonl(args.output_jsonl, records)
    write_summary_csv(args.output_csv, records)

    n = len(records)
    n_correct = sum(1 for r in records if r["is_correct_auto"])
    n_manual = sum(1 for r in records if r["needs_manual_check"])

    print(f"Loaded {n} generations from {args.input_path}")
    print(f"Auto-correct: {n_correct}/{n}")
    print(f"Needs manual check: {n_manual}/{n}")
    print(f"Wrote eval JSONL to {args.output_jsonl}")
    print(f"Wrote summary CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
