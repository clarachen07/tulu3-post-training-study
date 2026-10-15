"""Allow the standalone experiment packages to be tested from the experiment root."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "patching"),
    str(ROOT / "pca"),
    str(ROOT / "scripts"),
]
