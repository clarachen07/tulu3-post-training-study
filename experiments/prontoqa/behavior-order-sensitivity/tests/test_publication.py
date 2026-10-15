"""Aggregate checks detect tampered transitions, tests, plots and raw-data leaks."""

import json
import re
import shutil

import pytest

from tulu_behavior.common import REPO
from tulu_behavior.order_publication import (
    ROOT,
    check_order_figures,
    check_order_statistics,
    reference_files,
)


def inputs():
    read = lambda p: json.loads(p.read_text())
    return (
        read(ROOT / "reports/order_comparison.json"),
        read(ROOT / "reports/analysis.json"),
        read(REPO / "behavior/reports/analysis.json"),
    )


def test_published_order_statistics_and_svg_coordinates():
    comparison, summary, baseline = inputs()
    assert check_order_statistics(comparison, summary, baseline) == 128
    check_order_figures(summary, comparison)


@pytest.mark.parametrize(
    "kind", ["counts", "holm", "base_alias", "missing", "interaction", "ordering"]
)
def test_semantic_changes_cannot_hide_behind_new_hashes(kind):
    comparison, summary, baseline = inputs()
    if kind == "counts":
        comparison["summaries"][0]["gained_correct"] += 1
    elif kind == "holm":
        comparison["primary_tests"][0]["p_holm"] = 0.99
    elif kind == "base_alias":
        comparison["summaries"][0]["ci_low_pp"] -= 1
    elif kind == "missing":
        comparison["summaries"].pop()
    elif kind == "ordering":
        comparison["ordering_summary"]["formal"]["1"]["unchanged_questions"] += 1
    else:
        comparison["stage_gain_changes"][0]["stage_gain_change_pp"] += 1
    with pytest.raises(RuntimeError):
        check_order_statistics(comparison, summary, baseline)


def test_actual_svg_change_is_rejected(tmp_path):
    comparison, summary, _ = inputs()
    shutil.copytree(ROOT / "reports/figures", tmp_path / "reports/figures")
    path = tmp_path / "reports/figures/order_effect.svg"
    text = path.read_text()
    offset = text.index('<g id="order_effect">')
    match = re.search(r'(<use [^>]* y=")([0-9.]+)', text[offset:])
    assert match is not None
    start, end = offset + match.start(2), offset + match.end(2)
    text = text[:start] + str(float(match.group(2)) + 5) + text[end:]
    path.write_text(text)
    with pytest.raises(RuntimeError, match="SVG coordinates differ"):
        check_order_figures(summary, comparison, tmp_path)


def test_publication_rejects_raw_runtime_and_delivery_files(tmp_path):
    path = tmp_path / "outputs/formal/batch.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    with pytest.raises(RuntimeError, match="allowlist"):
        reference_files(tmp_path)
