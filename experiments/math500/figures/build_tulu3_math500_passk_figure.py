from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_JSON = ROOT / "results/frozen/four-stage-seed1-3/four-stage-pass12.json"
PASS16_JSON = ROOT / "results/frozen/three-stage-pass16.json"
OUT = ROOT / "figures/tulu3_math500_passk_four_comparisons.svg"

W, H = 1950, 620
LEFT, RIGHT, TOP, BOTTOM = 145, 45, 92, 150
GAP = 45
PANEL_COUNT = 4
PANEL_W = (W - LEFT - RIGHT - (PANEL_COUNT - 1) * GAP) / PANEL_COUNT
PANEL_H = H - TOP - BOTTOM

COLORS = {
    "base": "#0A8F86",
    "sft": "#D9952F",
    "dpo": "#3E6FB4",
    "rlvr": "#F06B6B",
}
LABELS = {"base": "Base", "sft": "SFT", "dpo": "DPO", "rlvr": "RLVR"}
MARKERS = {"base": "triangle", "sft": "square", "dpo": "circle", "rlvr": "diamond"}


def esc(value: object) -> str:
    return html.escape(str(value))


def x_pos(panel_x: float, k: int) -> float:
    return panel_x + math.log2(k) / 4.0 * PANEL_W


def y_pos(value: float) -> float:
    return TOP + (1.0 - value) * PANEL_H


def marker(x: float, y: float, kind: str, color: str, size: float = 5.8) -> str:
    if kind == "triangle":
        points = f"{x},{y-size} {x-size},{y+size*0.82} {x+size},{y+size*0.82}"
        return f'<polygon points="{points}" fill="{color}" stroke="white" stroke-width="1.1"/>'
    if kind == "square":
        return (
            f'<rect x="{x-size}" y="{y-size}" width="{2*size}" height="{2*size}" '
            f'fill="{color}" stroke="white" stroke-width="1.1"/>'
        )
    if kind == "diamond":
        points = f"{x},{y-size} {x+size},{y} {x},{y+size} {x-size},{y}"
        return f'<polygon points="{points}" fill="{color}" stroke="white" stroke-width="1.1"/>'
    return f'<circle cx="{x}" cy="{y}" r="{size}" fill="{color}" stroke="white" stroke-width="1.1"/>'


def path_for(panel_x: float, values: dict[int, float]) -> str:
    ordered = sorted(values.items())
    points = [(x_pos(panel_x, k), y_pos(v)) for k, v in ordered]
    d = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))
    return d


with BASE_JSON.open() as f:
    pass12 = json.load(f)
with PASS16_JSON.open() as f:
    pass16 = json.load(f)

series: dict[str, dict[int, float]] = {}
series["base"] = {
    int(k): v / 100.0 for k, v in pass12["pooled"]["base"]["pass_at"].items()
}
for name in ("sft", "dpo", "rlvr"):
    series[name] = {
        int(k): v / 100.0 for k, v in pass16["pooled"][name]["pass_at"].items()
    }

comparisons = [
    ("Base → RLVR (cumulative)", "base", "rlvr"),
    ("Base → SFT", "base", "sft"),
    ("SFT → DPO", "sft", "dpo"),
    ("DPO → RLVR", "dpo", "rlvr"),
]

parts: list[str] = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block" role="img" aria-labelledby="title desc">',
    "<title id=\"title\">Tülu-3 8B post-training stage comparisons on MATH500</title>",
    (
        "<desc id=\"desc\">Four aligned pass at k panels compare Base directly to RLVR, "
        "then show Base to SFT, SFT to DPO, and DPO to RLVR. The direct Base to RLVR "
        "contrast is large, while DPO and RLVR nearly overlap.</desc>"
    ),
    f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>',
    (
        '<style>'
        '.title{font:700 28px "Times New Roman",Times,serif;fill:#20262D}'
        '.axis{font:20px "Times New Roman",Times,serif;fill:#222A31}'
        '.axis-label{font:24px "Times New Roman",Times,serif;fill:#222A31}'
        '.row-label{font:700 31px "Times New Roman",Times,serif;fill:#20262D}'
        '.legend{font:18px Arial,sans-serif;fill:#222A31}'
        '.note{font:16px Arial,sans-serif;fill:#5F6872}'
        '.endpoint{font:600 16px Arial,sans-serif}'
        '</style>'
    ),
]

# Shared row / axis labels.
parts.extend(
    [
        f'<text class="row-label" x="33" y="{TOP + PANEL_H/2}" text-anchor="middle" transform="rotate(-90 33 {TOP + PANEL_H/2})">MATH500</text>',
        f'<text class="axis-label" x="78" y="{TOP + PANEL_H/2}" text-anchor="middle" transform="rotate(-90 78 {TOP + PANEL_H/2})">Coverage (pass@k)</text>',
        f'<text class="axis-label" x="{LEFT + (PANEL_COUNT*PANEL_W + (PANEL_COUNT-1)*GAP)/2}" y="{H-80}" text-anchor="middle">Number of Samples k</text>',
    ]
)

for panel_idx, (title, first, second) in enumerate(comparisons):
    px = LEFT + panel_idx * (PANEL_W + GAP)
    parts.append(
        f'<text class="title" x="{px + PANEL_W/2:.2f}" y="47" text-anchor="middle">{esc(title)}</text>'
    )

    # Grid and y ticks.
    for tick_i in range(6):
        value = tick_i * 0.2
        y = y_pos(value)
        parts.append(
            f'<line x1="{px:.2f}" x2="{px+PANEL_W:.2f}" y1="{y:.2f}" y2="{y:.2f}" stroke="#DDE2E6" stroke-width="1"/>'
        )
        if panel_idx == 0:
            label = "0" if value == 0 else ("1" if value == 1 else f"{value:.1f}")
            parts.append(
                f'<text class="axis" x="{px-12:.2f}" y="{y+7:.2f}" text-anchor="end">{label}</text>'
            )

    # X ticks and vertical grid.
    for k in (1, 2, 4, 8, 16):
        x = x_pos(px, k)
        parts.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{TOP}" y2="{TOP+PANEL_H}" stroke="#E7EAED" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="axis" x="{x:.2f}" y="{TOP+PANEL_H+30}" text-anchor="middle">{k}</text>'
        )

    # Frame.
    parts.append(
        f'<rect x="{px:.2f}" y="{TOP}" width="{PANEL_W:.2f}" height="{PANEL_H}" fill="none" stroke="#9DA5AB" stroke-width="1.4"/>'
    )

    # Curves and markers.
    for name in (first, second):
        values = series[name]
        parts.append(
            f'<path d="{path_for(px, values)}" fill="none" stroke="{COLORS[name]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for k, value in sorted(values.items()):
            x, y = x_pos(px, k), y_pos(value)
            parts.append(marker(x, y, MARKERS[name], COLORS[name]))

    # Compact legend.
    lx, ly = px + 18, TOP + 22
    for j, name in enumerate((first, second)):
        yy = ly + j * 29
        parts.extend(
            [
                f'<line x1="{lx}" x2="{lx+38}" y1="{yy}" y2="{yy}" stroke="{COLORS[name]}" stroke-width="3"/>',
                marker(lx + 19, yy, MARKERS[name], COLORS[name], 5.2),
                f'<text class="legend" x="{lx+48}" y="{yy+6}">{LABELS[name]}</text>',
            ]
        )

    # Direct endpoint labels to make the near tie visible in the final panel.
    if panel_idx == 0:
        common_x = x_pos(px, 12)
        base_y = y_pos(series["base"][12])
        rlvr_y = y_pos(series["rlvr"][12])
        parts.extend(
            [
                f'<text class="endpoint" x="{common_x-8}" y="{base_y+24}" text-anchor="end" fill="{COLORS["base"]}">27.6%</text>',
                f'<text class="endpoint" x="{common_x-8}" y="{rlvr_y-11}" text-anchor="end" fill="{COLORS["rlvr"]}">77.9%</text>',
                f'<text class="note" x="{px+PANEL_W-15}" y="{TOP+PANEL_H-27}" text-anchor="end">matched-k Δ@12 = +50.3 pp</text>',
            ]
        )

    if panel_idx == PANEL_COUNT - 1:
        end_x = x_pos(px, 16)
        dpo_y = y_pos(series["dpo"][16])
        rlvr_y = y_pos(series["rlvr"][16])
        parts.extend(
            [
                f'<text class="endpoint" x="{end_x-10}" y="{dpo_y+25}" text-anchor="end" fill="{COLORS["dpo"]}">79.2%</text>',
                f'<text class="endpoint" x="{end_x-10}" y="{rlvr_y-12}" text-anchor="end" fill="{COLORS["rlvr"]}">80.6%</text>',
                f'<text class="note" x="{px+PANEL_W-16}" y="{TOP+PANEL_H-49}" text-anchor="end">Δ = +1.4 pp</text>',
                f'<text class="note" x="{px+PANEL_W-16}" y="{TOP+PANEL_H-25}" text-anchor="end">95% CI [−1.4, +4.4]</text>',
            ]
        )

parts.extend(
    [
        f'<text class="note" x="{W/2}" y="{H-43}" text-anchor="middle">'
        "Unbiased pooled pass@k · Base: Seeds 1–3 (12 samples/question) · SFT/DPO/RLVR: Seeds 1–4 (16 samples/question) · n=500"
        "</text>",
        f'<text class="note" x="{W/2}" y="{H-18}" text-anchor="middle">'
        "Base uses the shared SFT chat template; interpret Base→RLVR as a cumulative, interface-sensitive contrast."
        "</text>",
        "</svg>",
    ]
)

svg_text = "\n".join(parts)

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=OUT)
parser.add_argument("--html-out", type=Path)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(svg_text, encoding="utf-8")
if args.html_out:
    theme_svg = svg_text
    for old, new in (
        ('fill="#FFFFFF"', 'fill="var(--background,#FFFFFF)"'),
        ("fill:#20262D", "fill:var(--foreground,#20262D)"),
        ("fill:#222A31", "fill:var(--foreground,#222A31)"),
        ("fill:#5F6872", "fill:var(--muted-foreground,#5F6872)"),
        ('stroke="#DDE2E6"', 'stroke="var(--border,#DDE2E6)"'),
        ('stroke="#E7EAED"', 'stroke="var(--border,#E7EAED)"'),
        ('stroke="#9DA5AB"', 'stroke="var(--muted-foreground,#9DA5AB)"'),
        ('stroke="white"', 'stroke="var(--background,#FFFFFF)"'),
        ("#0A8F86", "var(--viz-series-1,#0A8F86)"),
        ("#D9952F", "var(--viz-series-2,#D9952F)"),
        ("#3E6FB4", "var(--viz-series-3,#3E6FB4)"),
        ("#F06B6B", "var(--viz-series-4,#F06B6B)"),
    ):
        theme_svg = theme_svg.replace(old, new)
    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(theme_svg, encoding="utf-8")

print(args.output)
