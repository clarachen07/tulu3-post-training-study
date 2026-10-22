"""Output-format classification, separate from the unchanged numerical scorer."""
import re

NUMBER = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)'
VALUE = rf'(?:{NUMBER}|\\boxed\{{\s*{NUMBER}\s*\}})'
WRAPPED = rf'(?:{VALUE}|\${VALUE}\$|\$\${VALUE}\$\$|\\\({VALUE}\\\)|\\\[{VALUE}\\\])'
PATTERN = re.compile(rf'(?:Answer\s*:|Final\s+answer\s*:|The\s+answer\s+is)?\s*{WRAPPED}\s*\.?', re.IGNORECASE)


def answer_only(text):
    return PATTERN.fullmatch(text.strip()) is not None
