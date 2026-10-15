"""Validate patching results and render descriptive, four-stage reports.

Usage: python -m truth_patching.report --root /path/to/experiment
The report never infers completion from the presence of a heatmap.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


STAGES = ("base", "sft", "dpo", "rlvr")
N_LAYERS = 32
ANCHORS = ("city", "country", "period", "readout")
ANCHOR_LABELS = {
    "city": "City (changed entity)",
    "country": "Country (statement-final word)",
    "period": "Statement-final punctuation",
    "readout": "Readout (final input token)",
}
SCORE_FIELDS = (
    "difference",
    "true_logit",
    "false_logit",
    "true_logprob",
    "false_logprob",
    "true_prob",
    "false_prob",
    "binary_true_prob",
)


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_score(score: Any, name: str) -> None:
    if not isinstance(score, dict):
        raise ValueError(f"{name}: score is not an object")
    for form in ("spaced", "bare"):
        values = score.get(form)
        if not isinstance(values, dict):
            raise ValueError(f"{name}: missing {form} candidate scores")
        for key in SCORE_FIELDS:
            if not _finite(values.get(key)):
                raise ValueError(f"{name}: {form}.{key} is missing or nonfinite")
        for key in ("true_prob", "false_prob", "binary_true_prob"):
            if not 0 <= values[key] <= 1:
                raise ValueError(f"{name}: {form}.{key} lies outside [0, 1]")


def _optional_signature(value: Any, expected: Any, name: str) -> None:
    if (
        isinstance(value, dict)
        and "signature" in value
        and value["signature"] != expected
    ):
        raise ValueError(f"{name}: signature mismatch")


def _token_map(prompt: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for token in prompt.get("tokens", []):
        position = token.get("position")
        if isinstance(position, int) and not isinstance(position, bool):
            result[position] = token
    return result


def _chat_positions(prompt: dict[str, Any], stage: str) -> set[int]:
    explicit = prompt.get("chat_wrapper_positions")
    if explicit is not None:
        return set(explicit)
    if stage == "base":
        return set()
    tokens = sorted(_token_map(prompt).items())
    # Decode the supplied token labels, without invoking a different tokenizer.
    combined = "".join(str(token.get("text", "")) for _, token in tokens)
    spans = [
        match.span()
        for match in re.finditer(r"<\|(?:user|assistant|system)\|>\n?", combined)
    ]
    output: set[int] = set()
    offset = 0
    for position, token in tokens:
        end = offset + len(str(token.get("text", "")))
        if any(offset < right and end > left for left, right in spans):
            output.add(position)
        offset = end
    # In the specified single-user native template, the final input token is
    # part of the assistant generation header, even with partial token labels.
    if "<|assistant|>" in str(prompt.get("false_text", "")):
        readout = prompt.get("anchors", {}).get("readout")
        if isinstance(readout, int):
            output.add(readout)
    return output


def _label(
    token: dict[str, Any], position: int, chat: set[int], boundary: set[int]
) -> str:
    raw = str(token.get("text", "?"))
    text = raw.replace("\n", r"\n").replace("\t", r"\t").replace("\r", r"\r")
    text = text.replace(" ", "\u2420")
    suffix = (
        " [boundary]"
        if position in boundary
        else (" [chat]" if position in chat else "")
    )
    return f"{position}: {text}" + suffix


def _extrema(values: list[float | None]) -> dict[str, Any]:
    present = [
        (index, value) for index, value in enumerate(values) if value is not None
    ]
    if not present:
        return {
            "maximum": None,
            "maximum_layer": None,
            "minimum": None,
            "minimum_layer": None,
        }
    maximum = max(present, key=lambda item: item[1])
    minimum = min(present, key=lambda item: item[1])
    return {
        "maximum": maximum[1],
        "maximum_layer": maximum[0],
        "minimum": minimum[1],
        "minimum_layer": minimum[0],
    }


def _read_stage(root: Path, stage: str) -> dict[str, Any]:
    directory = root / "results" / stage
    result: dict[str, Any] = {
        "stage": stage,
        "available": False,
        "complete": False,
        "errors": [],
        "warnings": [],
        "completed": 0,
        "expected": None,
    }
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        result["warnings"].append("manifest.json 不存在，阶段尚无可报告结果。")
        return result
    try:
        manifest = _load(manifest_path)
        if not manifest.get("signature"):
            raise ValueError("manifest.json: signature is missing")
        if str(manifest.get("stage", "")).lower() != stage:
            raise ValueError("manifest.json: stage does not match directory")
        prompt = manifest["prompt"]
        positions = prompt["positions"]
        if not isinstance(positions, list) or not positions:
            raise ValueError("manifest.prompt.positions must be a nonempty list")
        if any(
            not isinstance(p, int) or isinstance(p, bool) or p < 0 for p in positions
        ):
            raise ValueError("manifest.prompt.positions contains invalid indices")
        if len(set(positions)) != len(positions):
            raise ValueError("manifest.prompt.positions contains duplicates")
        positions = sorted(positions)
        if prompt["primary_form"] not in ("spaced", "bare"):
            raise ValueError("manifest.prompt.primary_form must be spaced or bare")
        tokens = _token_map(prompt)
        for position in positions:
            if position not in tokens:
                raise ValueError(f"missing token metadata at position {position}")
        for anchor in ANCHORS:
            if prompt["anchors"].get(anchor) not in positions:
                raise ValueError(f"anchor {anchor} is not a patched position")
        signature = manifest["signature"]
        result.update(
            available=True,
            signature=signature,
            model=manifest.get("model"),
            prompt=prompt,
            positions=positions,
            primary_form=prompt["primary_form"],
            expected=N_LAYERS * len(positions),
        )
        chat = _chat_positions(prompt, stage)
        boundary = set(prompt.get("chat_boundary_positions", []))
        chat -= boundary
        result["chat_wrapper_positions"] = sorted(chat)
        result["chat_boundary_positions"] = sorted(boundary)
        result["token_labels"] = [
            _label(tokens[p], p, chat, boundary) for p in positions
        ]
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["errors"].append(str(error))
        return result

    baseline: dict[str, Any] | None = None
    try:
        baseline = _load(directory / "baseline.json")
        _optional_signature(baseline, signature, "baseline.json")
        for label in ("false", "true"):
            _validate_score(baseline[label], f"baseline.{label}")
        result["baseline"] = baseline
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["errors"].append(f"baseline: {error}")
        baseline = None

    raw: list[list[float | None]] = [[None] * len(positions) for _ in range(N_LAYERS)]
    delta: list[list[float | None]] = [[None] * len(positions) for _ in range(N_LAYERS)]
    position_columns = {position: column for column, position in enumerate(positions)}
    seen: set[tuple[int, int]] = set()
    point_files = sorted((directory / "points").glob("*.json"))
    for path in point_files:
        try:
            point = _load(path)
            if point.get("signature") != signature:
                raise ValueError("signature mismatch")
            if str(point.get("stage", "")).lower() != stage:
                raise ValueError("stage mismatch")
            layer, position = point["layer"], point["position"]
            if (
                not isinstance(layer, int)
                or isinstance(layer, bool)
                or not 0 <= layer < N_LAYERS
            ):
                raise ValueError(f"invalid layer {layer}")
            if (
                not isinstance(position, int)
                or isinstance(position, bool)
                or position not in position_columns
            ):
                raise ValueError(f"unexpected position {position}")
            filename = re.fullmatch(r"l(\d+)_p(\d+)\.json", path.name)
            if filename is None or (int(filename[1]), int(filename[2])) != (
                layer,
                position,
            ):
                raise ValueError("filename coordinates do not match point metadata")
            coordinate = (layer, position)
            if coordinate in seen:
                raise ValueError("duplicate layer/position")
            _validate_score(point["scores"], path.name)
            seen.add(coordinate)
            column = position_columns[position]
            value = point["scores"][prompt["primary_form"]]["difference"]
            raw[layer][column] = value
            if baseline is not None:
                delta[layer][column] = (
                    value - baseline["false"][prompt["primary_form"]]["difference"]
                )
        except (OSError, ValueError, TypeError, KeyError) as error:
            result["errors"].append(f"{path.name}: {error}")
    missing = [
        {"layer": layer, "position": position}
        for layer in range(N_LAYERS)
        for position in positions
        if (layer, position) not in seen
    ]
    result.update(completed=len(seen), missing=missing, raw=raw, delta=delta)
    if missing:
        result["warnings"].append(
            f"缺少 {len(missing)} / {result['expected']} 个有效干预点。"
        )

    controls_ok = False
    try:
        controls = _load(directory / "controls.json")
        _optional_signature(controls, signature, "controls.json")
        checks = controls.get("checks")
        controls_ok = (
            controls.get("passed") is True
            and isinstance(checks, list)
            and bool(checks)
            and all(
                isinstance(check, dict) and check.get("passed") is True
                for check in checks
            )
        )
        result["controls"] = controls
        if not controls_ok:
            result["errors"].append("controls.json: controls have not all passed")
    except FileNotFoundError:
        result["warnings"].append("controls.json 尚不存在。")
    except (OSError, ValueError, TypeError) as error:
        result["errors"].append(f"controls: {error}")

    status_complete = False
    try:
        status = _load(directory / "status.json")
        _optional_signature(status, signature, "status.json")
        result["status"] = status
        status_complete = status.get("complete") is True
        if status.get("expected") != result["expected"]:
            result["errors"].append(
                "status.json: expected count does not match 32 × positions"
            )
        if status.get("completed") != len(seen):
            message = "status.json 的 completed 与实读有效文件数不同；报告采用后者。"
            result["errors" if status_complete else "warnings"].append(message)
        if status_complete and missing:
            result["errors"].append(
                "status.json claims completion but valid points are missing"
            )
    except FileNotFoundError:
        result["warnings"].append("status.json 尚不存在。")
    except (OSError, ValueError, TypeError) as error:
        result["errors"].append(f"status: {error}")

    anchors: dict[str, Any] = {}
    for name in ANCHORS:
        position = prompt["anchors"][name]
        column = position_columns[position]
        values = [row[column] for row in delta]
        # The final block/final token is an algebraic identity check, not a
        # location discovered by the experiment; exclude it from peak claims.
        descriptive = values[:-1] if name == "readout" else values
        anchors[name] = {
            "position": position,
            "token": tokens[position],
            "delta": values,
            "raw": [row[column] for row in raw],
            "chat_wrapper": position in chat,
            "chat_boundary": position in boundary,
            **_extrema(descriptive),
        }
    result["anchors"] = anchors
    eligible = [
        {
            "layer": layer,
            "position": position,
            "delta": delta[layer][column],
            "token": tokens[position],
            "chat_wrapper": position in chat,
            "chat_boundary": position in boundary,
        }
        for layer in range(N_LAYERS)
        for column, position in enumerate(positions)
        if delta[layer][column] is not None
        and (layer, position) != (N_LAYERS - 1, prompt["anchors"]["readout"])
    ]
    result["global_extrema_excluding_identity"] = {
        "maximum": max(eligible, key=lambda point: point["delta"])
        if eligible
        else None,
        "minimum": min(eligible, key=lambda point: point["delta"])
        if eligible
        else None,
    }
    if baseline is not None:
        form = prompt["primary_form"]
        expected_identity = baseline["true"][form]["difference"]
        observed_identity = raw[-1][position_columns[prompt["anchors"]["readout"]]]
        result["last_layer_readout_identity"] = {
            "expected_true_baseline": expected_identity,
            "observed": observed_identity,
            "absolute_error": None
            if observed_identity is None
            else abs(observed_identity - expected_identity),
            "interpretation": "identity check, not evidence for a truth representation",
        }
    result["complete"] = (
        not missing
        and baseline is not None
        and controls_ok
        and status_complete
        and not result["errors"]
    )
    return result


def collect(root: Path) -> dict[str, Any]:
    stages = {stage: _read_stage(root, stage) for stage in STAGES}
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "complete": all(stage["complete"] for stage in stages.values()),
        "has_integrity_errors": any(stage["errors"] for stage in stages.values()),
        "metric": "log P(TRUE) - log P(FALSE), equal to the candidate logit difference",
        "delta": "patched difference minus the unpatched false-prompt difference",
        "layers": {
            "count": N_LAYERS,
            "numbering": "zero-based",
            "site": "decoder block output, before final RMSNorm",
        },
        "stages": stages,
    }


def _scales(summary: dict[str, Any]) -> tuple[tuple[float, float], float]:
    raw = [
        value
        for stage in summary["stages"].values()
        for row in stage.get("raw", [])
        for value in row
        if value is not None
    ]
    delta = [
        value
        for stage in summary["stages"].values()
        for row in stage.get("delta", [])
        for value in row
        if value is not None
    ]
    minimum, maximum = (min(raw), max(raw)) if raw else (-1.0, 1.0)
    if minimum == maximum:
        minimum, maximum = minimum - 0.5, maximum + 0.5
    return (minimum, maximum), max((abs(value) for value in delta), default=1.0) or 1.0


def render_figures(
    summary: dict[str, Any], root: Path, output: Path | None = None
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    import numpy as np

    destination = output if output is not None else root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}
    )
    raw_limits, delta_limit = _scales(summary)
    summary["figure_scales"] = {
        "raw": list(raw_limits),
        "delta": [-delta_limit, delta_limit],
    }
    outputs: list[str] = []

    def save(figure: Any, name: str) -> None:
        for extension in ("png", "pdf"):
            path = destination / f"{name}.{extension}"
            figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
            outputs.append(str(path.relative_to(destination)))
        plt.close(figure)

    for kind, cmap_name in (("raw", "Blues"), ("delta", "RdBu_r")):
        norm = (
            Normalize(*raw_limits)
            if kind == "raw"
            else TwoSlopeNorm(0, -delta_limit, delta_limit)
        )
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#dddddd")
        figure, axes = plt.subplots(2, 2, figsize=(22, 15), layout="constrained")
        for axis, stage_name in zip(axes.flat, STAGES):
            stage = summary["stages"][stage_name]
            if not stage.get(kind):
                axis.axis("off")
                axis.text(
                    0.5,
                    0.5,
                    f"{stage_name.upper()}: no valid result matrix",
                    ha="center",
                    va="center",
                )
                continue
            values = np.array(stage[kind], dtype=float)
            axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            axis.set_xticks(
                range(len(stage["positions"])),
                stage["token_labels"],
                rotation=65,
                ha="right",
                fontsize=8,
            )
            axis.set_yticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
            axis.set_ylabel("Decoder block (zero-based)")
            axis.set_xlabel(
                "Absolute input position: token; [chat] = new; [boundary] = text + template"
            )
            state = "complete" if stage["complete"] else "PARTIAL / UNVERIFIED"
            axis.set_title(
                f"{stage_name.upper()} | {stage['primary_form']} labels | {stage['completed']}/{stage['expected']} | {state}"
            )
            for anchor, detail in stage.get("anchors", {}).items():
                column = stage["positions"].index(detail["position"])
                axis.axvline(
                    column, color="#555555", linewidth=0.55, linestyle=":", alpha=0.65
                )
            readout_column = stage["positions"].index(
                stage["prompt"]["anchors"]["readout"]
            )
            axis.plot(
                readout_column,
                31,
                marker="s",
                fillstyle="none",
                color="#ff9900",
                markersize=9,
                markeredgewidth=1.5,
            )
        label = (
            "Patched log P(TRUE) - log P(FALSE)"
            if kind == "raw"
            else "Change from false-prompt baseline (delta)"
        )
        figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=list(axes.flat),
            shrink=0.77,
            label=label,
        )
        figure.suptitle(
            "Tulu 3 8B lineage: residual-stream activation replacement\n"
            + (
                "Raw candidate log-probability difference"
                if kind == "raw"
                else "Baseline-adjusted patch effect"
            )
            + "; common color scale across stages\nOrange outline: final block/readout identity check; gray: missing data",
            fontsize=16,
        )
        save(figure, f"{kind}_heatmaps")

    figure, axes = plt.subplots(
        2, 2, figsize=(15, 10), layout="constrained", sharex=True, sharey=True
    )
    colors = {"base": "#3b3b3b", "sft": "#0072b2", "dpo": "#d55e00", "rlvr": "#009e73"}
    for axis, anchor in zip(axes.flat, ANCHORS):
        axis.axhline(0, color="#777777", linewidth=0.8)
        for stage_name in STAGES:
            stage = summary["stages"][stage_name]
            detail = stage.get("anchors", {}).get(anchor)
            if detail is None:
                continue
            values = np.array(detail["delta"], dtype=float)
            label = f"{stage_name.upper()} (p={detail['position']})" + (
                " [chat]" if detail["chat_wrapper"] else ""
            )
            if anchor == "readout":
                axis.plot(
                    range(31),
                    values[:31],
                    label=label,
                    color=colors[stage_name],
                    linewidth=1.8,
                )
                if np.isfinite(values[31]):
                    axis.scatter(
                        [31],
                        [values[31]],
                        marker="x",
                        color=colors[stage_name],
                        s=55,
                        zorder=4,
                    )
            else:
                axis.plot(
                    range(32),
                    values,
                    label=label,
                    color=colors[stage_name],
                    linewidth=1.8,
                )
        axis.set_title(ANCHOR_LABELS[anchor])
        axis.set_xlabel("Decoder block (zero-based)")
        axis.set_ylabel("Delta from false-prompt baseline")
        axis.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
        axis.grid(alpha=0.18)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
        if anchor == "readout":
            axis.text(
                0.02,
                0.02,
                "x at block 31: identity check, not representation evidence",
                transform=axis.transAxes,
                fontsize=8,
            )
    figure.suptitle(
        "Four predeclared token anchors: layerwise causal effect\n"
        "Base uses completion; SFT/DPO/RLVR use native chat (a format confound)",
        fontsize=15,
    )
    save(figure, "anchor_curves")
    return outputs


def _number(value: Any, digits: int = 5) -> str:
    return f"{value:.{digits}f}" if _finite(value) else "缺失"


def _escape(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", r"\n").replace("`", "'")


def _model_text(model: Any) -> str:
    if isinstance(model, dict):
        identifier = model.get(
            "id", model.get("model_id", model.get("repo_id", model.get("name", "")))
        )
        revision = model.get("revision", model.get("commit", ""))
        return (
            f"{identifier} @ {revision}"
            if identifier
            else json.dumps(model, ensure_ascii=False)
        )
    return str(model)


def render_markdown(summary: dict[str, Any]) -> str:
    complete = summary["complete"]
    lines = [
        "# Tulu 3 8B 四阶段：真假判断的激活替换实验",
        "",
        "**四阶段实验已完成，且完整性与控制检查通过。**"
        if complete
        else "**当前是部分或尚未验证的结果，不能称为四阶段实验已完成。**",
        "",
        "本报告复现原论文第 3 节的残差流替换定位方法；仅报告实际保存的前向计算结果。"
        "固定真句作为供体、假句作为受体，在每个 decoder block 输出处逐个替换 token 的完整残差向量。"
        "层号为 0–31；该状态位于最终 RMSNorm 之前。",
        "",
        "主指标为 `log P(TRUE) − log P(FALSE)`，等于两候选 token 的 logit 差。"
        "raw 图显示替换后的主指标；delta 图与锚点曲线显示该值减去未替换假句基线。"
        "正 delta 表示干预使输出相对更偏向 TRUE，不表示输出已经翻转，也不自动证明存在抽象的真假表征。",
        "",
        "## 数据完整性与冻结模型",
        "",
        "| 阶段 | 有效点 / 应有点 | 状态 | 主候选形式 | 模型与 revision |",
        "|---|---:|---|---|---|",
    ]
    for stage_name, stage in summary["stages"].items():
        state = (
            "完整且验证通过"
            if stage["complete"]
            else ("存在完整性错误" if stage["errors"] else "部分 / 待验证")
        )
        lines.append(
            f"| {stage_name.upper()} | {stage['completed']} / {stage['expected'] if stage['expected'] is not None else '未知'} "
            f"| {state} | {stage.get('primary_form', '缺失')} | {_escape(_model_text(stage.get('model', '缺失')))} |"
        )
    lines += [
        "",
        "每阶段要求 `32 × manifest.prompt.positions` 个互不重复的干预点；逐个验证坐标、阶段和 signature。"
        "缺失点保持为空并在图中显示为灰色，不以零补齐。完成状态还要求 baseline、控制检查和运行状态一致。",
        "",
    ]
    for stage_name, stage in summary["stages"].items():
        for message in stage["errors"]:
            lines.append(f"- **{stage_name.upper()} 错误**：{_escape(message)}")
        for message in stage["warnings"]:
            lines.append(f"- {stage_name.upper()}：{_escape(message)}")
    lines += [
        "",
        "## 未干预的真假基线",
        "",
        "| 阶段 | 候选形式 | 假句差值 | 真句差值 | 假句 P(TRUE) | 假句 P(FALSE) | 真句 P(TRUE) | 真句 P(FALSE) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage_name, stage in summary["stages"].items():
        baseline = stage.get("baseline")
        if baseline is None:
            continue
        for form in ("spaced", "bare"):
            false, true = baseline["false"][form], baseline["true"][form]
            form_label = form + (
                "（主指标）" if form == stage["primary_form"] else "（辅助诊断）"
            )
            values = [
                false["difference"],
                true["difference"],
                false["true_prob"],
                false["false_prob"],
                true["true_prob"],
                true["false_prob"],
            ]
            lines.append(
                f"| {stage_name.upper()} | {form_label} | "
                + " | ".join(_number(value) for value in values)
                + " |"
            )
    lines += [
        "",
        "概率均为全词表 softmax 下的候选概率；两者之和不必为 1。"
        "不能把 `binary_true_prob` 误读成模型生成 TRUE 的全词表概率。",
        "",
    ]
    for stage_name, stage in summary["stages"].items():
        if "baseline" not in stage:
            continue
        form = stage["primary_form"]
        false_d = stage["baseline"]["false"][form]["difference"]
        true_d = stage["baseline"]["true"][form]["difference"]
        describe = lambda value: (
            "TRUE" if value > 0 else ("FALSE" if value < 0 else "两者相同")
        )
        lines.append(
            f"- {stage_name.upper()} 主候选下：假句更偏向 {describe(false_d)}（{_number(false_d)}），"
            f"真句更偏向 {describe(true_d)}（{_number(true_d)}）；真假基线差为 {_number(true_d - false_d)}。"
        )
    lines += [
        "",
        "## 预先固定的四个位置",
        "",
        "锚点在实验前固定为被替换的城市、句末国家词、句号、读出位置。"
        "下表只描述各锚点实际观测的 delta 最大值与最小值，不设置人为的“复现通过”阈值。"
        "读出位置的极值统计排除最后一层恒等检查。部分结果的极值仅针对已完成点。",
        "",
        "| 阶段 | 锚点 | 绝对位置与 token | 最大 delta（层） | 最小 delta（层） |",
        "|---|---|---|---:|---:|",
    ]
    for stage_name, stage in summary["stages"].items():
        for anchor, detail in stage.get("anchors", {}).items():
            token = _escape(detail["token"].get("text", "?"))
            if detail["chat_wrapper"]:
                token += " [chat：非原文位置]"
            elif detail.get("chat_boundary"):
                token += " [boundary：原文与模板的合并 token]"
            lines.append(
                f"| {stage_name.upper()} | {anchor} | {detail['position']}: {token} "
                f"| {_number(detail['maximum'])}（{detail['maximum_layer']}） "
                f"| {_number(detail['minimum'])}（{detail['minimum_layer']}） |"
            )
    lines += [""]
    for stage_name, stage in summary["stages"].items():
        anchors = stage.get("anchors", {})
        if not anchors:
            continue
        country, period = anchors["country"], anchors["period"]
        lines.append(
            f"- {stage_name.upper()} 句末描述：国家词的最大 delta 为 {_number(country['maximum'])}"
            f"（第 {country['maximum_layer']} 层），句号的最大 delta 为 {_number(period['maximum'])}"
            f"（第 {period['maximum_layer']} 层）。应结合完整热图检查效应是否集中在句末及其层分布；"
            "不把单个最大点直接命名为“真值神经元”或跨数据集的真值方向。"
        )
        maximum = stage.get("global_extrema_excluding_identity", {}).get("maximum")
        if maximum is not None:
            token = _escape(maximum["token"].get("text", "?"))
            category = (
                "原文与模板合并的边界 token"
                if maximum.get("chat_boundary")
                else (
                    "新增 chat 模板位置" if maximum["chat_wrapper"] else "原始提示位置"
                )
            )
            lines.append(
                f"  排除最末层读出恒等点后，全图最大的 delta 为 {_number(maximum['delta'])}，"
                f"位于第 {maximum['layer']} 层、位置 {maximum['position']}（{token}；{category}）。"
            )
    lines += [
        "",
        "## 图与解释边界",
        "",
        "![四阶段 raw 热图](../figures/raw_heatmaps.png)",
        "",
        "[raw 热图 PDF](../figures/raw_heatmaps.pdf)",
        "",
        "![四阶段 delta 热图](../figures/delta_heatmaps.png)",
        "",
        "[delta 热图 PDF](../figures/delta_heatmaps.pdf)",
        "",
        "![四个预固定锚点曲线](../figures/anchor_curves.png)",
        "",
        "[锚点曲线 PDF](../figures/anchor_curves.pdf)",
        "",
        "raw 热图统一使用 Blues 色阶；delta 热图统一使用以 0 为中心的对称色阶。"
        "这些尺度由本次四阶段全部有效值确定，不对个别阶段分别缩放。",
        "",
        "Base 使用原生 completion；SFT、DPO、RLVR 使用各自冻结 tokenizer 的原生 Tulu chat 模板。"
        "因此 Base→SFT 同时包含训练阶段、模板包装与主候选 token 形式的变化，不能把差异全部归因于 SFT。"
        "`spaced` 表示带前导空格的 ` TRUE` / ` FALSE`，`bare` 表示无前导空格的 `TRUE` / `FALSE`。"
        "两种形式均保留作诊断，主指标取 manifest 声明的形式。",
        "",
        "图中 `[chat]` 仅表示纯新增的 assistant header token；chat 的最终读出位置也位于新增模板中。"
        "`[boundary]` 表示原文冒号与模板新增换行被 tokenizer 合并为 `:\\n` 的位置，"
        "它同时包含原文和包装内容，不能视作完全新增 token。"
        "这些位置与原文句子位置必须分开解释。token 横轴使用各阶段真实输入的绝对位置，不能按列号直接作跨模板一一对应。",
        "",
        "**最后一层 × 最后输入位置仅是恒等检查。** 在该位置替换完整残差向量后，"
        "后续相同的最终归一化和输出头会读出供体结果；它接近真句基线是实现应满足的关系，"
        "不能作为句末真值表征存在的证据。热图以橙色空框标记，readout 曲线以单独叉号显示。",
        "",
        "| 阶段 | 最末位置替换值 | 真句基线 | 绝对误差 |",
        "|---|---:|---:|---:|",
    ]
    for stage_name, stage in summary["stages"].items():
        check = stage.get("last_layer_readout_identity")
        if check:
            lines.append(
                f"| {stage_name.upper()} | {_number(check['observed'])} | {_number(check['expected_true_baseline'])} | {_number(check['absolute_error'], 8)} |"
            )
    lines += [
        "",
        "本次是指定原论文案例的确定性定位实验，不提供置信区间，也不将单个城市真假对的结果外推为所有事实、"
        "所有提示或所有数据集上的规律。若未出现作者的句末效应，应如实记录；图形可见热点本身不构成预设的通过判据。",
        "",
        "## 控制检查",
        "",
        "| 阶段 | 检查数量 | 最大绝对误差 | 是否全部通过 | 完整记录 |",
        "|---|---:|---:|---|---|",
    ]
    for stage_name, stage in summary["stages"].items():
        controls = stage.get("controls", {})
        checks = controls.get("checks", [])
        errors = [
            check["max_abs_error"]
            for check in checks
            if _finite(check.get("max_abs_error"))
        ]
        maximum_error = max(errors) if errors else None
        passed = (
            controls.get("passed") is True
            and bool(checks)
            and all(check.get("passed") is True for check in checks)
        )
        record = (
            f"[controls.json](../results/{stage_name}/controls.json)"
            if controls
            else "尚无记录"
        )
        lines.append(
            f"| {stage_name.upper()} | {len(checks)} | {_number(maximum_error, 8)} | {passed} | {record} |"
        )
    lines += [
        "",
        "机器可读结果见 [summary.json](../artifacts/summary.json)。所有数值来自保存的 baseline 与干预点，"
        "未把缺失点、模拟数据或生成式解释当作实验观测。",
        "",
        "方法来源：[The Geometry of Truth，第 3 节](https://arxiv.org/html/2310.06824v3#S3)。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Experiment directory containing results/<stage>/",
    )
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiment_utils.runtime import protect_published

    protect_published(root)
    summary = collect(root)
    summary["figure_files"] = render_figures(summary, root)
    _write_json(root / "artifacts" / "summary.json", summary)
    report = root / "reports" / "REPORT.zh-CN.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown(summary), encoding="utf-8")
    counts = ", ".join(
        f"{name}={value['completed']}/{value['expected']}"
        for name, value in summary["stages"].items()
    )
    print(f"Report: {report}\nComplete: {summary['complete']}\n{counts}")
    if summary["has_integrity_errors"]:
        print(
            "Integrity errors are recorded in summary.json and the report.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
