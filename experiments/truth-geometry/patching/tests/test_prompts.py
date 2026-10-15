import ast
from pathlib import Path

from truth_patching.prompts import FALSE_PROMPT, TRUE_PROMPT


def test_city_prompts_match_author_pinned_source_verbatim():
    source = (
        Path(__file__).resolve().parents[1]
        / "third_party/geometry-of-truth/patching.py"
    ).read_text()
    assignments = {}
    for line in source.splitlines():
        text = line.strip()
        for name in ("false_prompt", "true_prompt"):
            prefix = "# " + name + " = "
            if text.startswith(prefix) and name not in assignments:
                assignments[name] = ast.literal_eval(text[len(prefix) :])
    assert assignments == {"false_prompt": FALSE_PROMPT, "true_prompt": TRUE_PROMPT}
    assert TRUE_PROMPT == FALSE_PROMPT.replace("Chicago", "Toronto")
    assert not FALSE_PROMPT.endswith((" ", "\n"))
