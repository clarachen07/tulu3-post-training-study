"""Analyze paired prompt activations and render the five TwoNN overlays."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path

import numpy as np
from tulu_id.analyze import _summary
from tulu_id.twonn import estimate_distances, euclidean_distances

from direct_id import (
    CONFIG, FIG2_TASKS, PER_TASK, REPLICATES, STAGES, SUBSET, SUBSET_SEED,
    atomic_json, fingerprint, read_json, require, sha256_file, stage_batches,
    verify_completion, verify_files,
)

PANELS = (*FIG2_TASKS, "mixed")
LABELS = ("Base", "SFT", "DPO", "RLVR")
COLORS = ("#174A7E", "#CB7725", "#33846B", "#A14966")
TITLES = dict(zip(PANELS, ("Vocabulary · Antonym → Spanish", "Arithmetic · Add 100 → Double",
                         "Geography · Park → Country → Capital", "Book facts · Book → Author → Birth year",
                         "Balanced mixture · Four direct composition tasks"), strict=True))
ZH_TITLES = dict(zip(PANELS, ("词汇", "算术", "地理", "书籍事实", "四任务等量混合"), strict=True))
OUTPUT_FILES = ("id_summary.csv", "id_replicates.csv", "sample_ids.csv", "subsamples.json", "report.zh-CN.md",
                *(f"figures/{panel}.{ext}" for panel in PANELS for ext in ("png", "svg")))
LAYERS = 32


def schedule():
    rng = np.random.RandomState(SUBSET_SEED)
    draws = {panel: [] for panel in PANELS}
    for _ in range(REPLICATES):
        for offset, task in enumerate(FIG2_TASKS):
            draws[task].append(offset * PER_TASK + rng.permutation(PER_TASK)[:SUBSET])
        draws["mixed"].append(np.concatenate([draws[task][-1] for task in FIG2_TASKS]))
    return draws


def saved_schedule(draws, bundle):
    ids = [row["sample_id"] for row in bundle["rows"]]
    return dict(seed=SUBSET_SEED, sample_ids=ids,
                panels={p: [x.tolist() for x in draws[p]] for p in PANELS})


def capture_inventory(root, bundle):
    receipt = read_json(root / "capture.json")
    token_path = root / "tokenization.json"
    models = read_json(root / "models.json")
    require(receipt["identity"] == fingerprint({k: v for k, v in receipt.items() if k != "identity"})
            and receipt["input"] == bundle["identity"] and receipt["models"] == fingerprint(models)
            and receipt["inference"] == read_json(CONFIG)["inference"]
            and receipt["tokenization"] == sha256_file(token_path), "Invalid capture identity")
    inventory, batches_by_stage = {}, {}
    for stage in STAGES:
        for mode in ("pilot", "formal"):
            batches, _ = stage_batches(root, bundle, receipt, stage, mode)
            verify_completion(root, bundle, receipt, stage, mode, batches)
            path = batches[0][0].parent / "complete.json"
            inventory[path.relative_to(root).as_posix()] = sha256_file(path)
            if mode == "formal":
                batches_by_stage[stage] = batches
    return receipt["identity"], inventory, batches_by_stage


def load_stage(batches, n):
    activations = np.empty((n, LAYERS, 4096), dtype=np.float32)
    for path, _, positions in batches:
        with np.load(path, allow_pickle=False) as content:
            activations[positions] = content["hidden"]
    return activations


def analyze_layer(hidden, stage, layer, draws, fraction):
    full_distances = euclidean_distances(hidden)
    summaries, estimates = [], []
    for panel in PANELS:
        subset_indices = draws[panel]
        n_pool = PER_TASK * len(FIG2_TASKS) if panel == "mixed" else PER_TASK
        fit = []
        for index, subset in enumerate(subset_indices, 1):
            est = estimate_distances(full_distances[np.ix_(subset, subset)], fraction)
            fit.append(dict(panel=panel, stage=stage, layer=layer, replicate=index, **est))
        summaries.append(dict(panel=panel, stage=stage, layer=layer, n_pool=n_pool,
                              n_subsample=len(subset_indices[0]), **_summary(fit)))
        estimates.extend(fit)
    return summaries, estimates


def write_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_plots(root, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    ranges = [(r["id_mean"] - r["id_sd"], r["id_mean"] + r["id_sd"]) for r in summaries if r["status"] == "ok"]
    if ranges:
        low, high = min(x[0] for x in ranges), max(x[1] for x in ranges)
        pad = max((high - low) * 0.055, 0.05)
        bounds = max(0, low - pad), high + pad
    else:
        bounds = (0, 1)
    target = root / "figures"
    target.mkdir(exist_ok=True)
    for panel in PANELS:
        selected = [r for r in summaries if r["panel"] == panel]
        fig, ax = plt.subplots(figsize=(10, 5.7), layout="constrained")
        for stage, color, label in zip(STAGES, COLORS, LABELS, strict=True):
            line = [r for r in selected if r["stage"] == stage]
            x = [r["layer"] for r in line]
            y = [r["id_mean"] if r["status"] == "ok" else np.nan for r in line]
            sd = [r["id_sd"] if r["status"] == "ok" else np.nan for r in line]
            artist = ax.errorbar(x, y, yerr=sd, fmt="o-", markersize=3.5, linewidth=1.6, elinewidth=0.8,
                                 capsize=1.7, color=color, label=label)
            artist.lines[0].set_gid(stage)
        n_pool, n_subset = selected[0]["n_pool"], selected[0]["n_subsample"]
        ax.set_title(f"{TITLES[panel]}\nN = {n_pool:,} · {REPLICATES} overlapping subsamples of {n_subset:,}"
                     " · Mean ± population SD", loc="left", pad=15, fontsize=12)
        ax.set(xlabel="Raw decoder block", ylabel="TwoNN intrinsic dimension",
               xticks=[1, *range(4, LAYERS + 1, 4)], xlim=(0.5, 32.5), ylim=bounds)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        for ext in ("png", "svg"):
            path = target / f"{panel}.{ext}"
            tmp = path.with_suffix(".tmp")
            fig.savefig(tmp, format=ext, dpi=180, facecolor="white")
            tmp.replace(path)
        plt.close(fig)
    return bounds


def shape_row(summaries, panel, stage):
    curve = [r for r in summaries if r["panel"] == panel and r["stage"] == stage]
    if any(r["status"] != "ok" for r in curve):
        return f"| {ZH_TITLES[panel]} | {dict(zip(STAGES, LABELS))[stage]} | 有缺点 | — | — | — | — | — |"
    y = np.asarray([row["id_mean"] for row in curve])
    peak = int(np.argmax(y))
    rebounds = int(np.sum(np.diff(y[peak:]) > 0))
    return (f"| {ZH_TITLES[panel]} | {dict(zip(STAGES, LABELS))[stage]} | L{peak+1} | {y[0]:.2f} | "
            f"{y[peak]:.2f} | {y[-1]:.2f} | {y[peak]-y[-1]:.2f} | {rebounds} |")


def write_report(root, summaries):
    lines = ["# 直接组合题的逐层 TwoNN", "",
             "四阶段回放同一批 1,440 道直接组合题的原始 10-shot token ID。"
             "每题用 FP32 前向读取最后一个有效输入 token 在 32 个 decoder block 的原始输出。", "",
             "每个点为 20 个重叠子样本的 TwoNN 均值，误差条为总体标准差。"
             "混合图在每次抽样中合并四类各 324 道后直接计算近邻。"
             "单任务和混合的估计样本数分别为 324 和 1,296，跨图以逐层形状为主。"
             "Base 使用原生续写，后三阶段使用原生聊天模板且输入 token 逐题相同。", ""]
    for panel in ("mixed", *FIG2_TASKS):
        lines += [f"## {ZH_TITLES[panel]}", "", f"![{ZH_TITLES[panel]}逐层 TwoNN](figures/{panel}.png)", "",
                  f"[SVG](figures/{panel}.svg)", ""]
    lines += ["## 峰位与末层", "",
              "峰值取未平滑均值的最大点，并列时取最早层。峰后上升步数统计相邻层均值回升的次数。"
              "PrOntoQA 的四阶段均值峰位为 L14–L15，峰后整体回落，末层仍高于首层。"
              "以下逐阶段数值供形状对照。"
              "[PrOntoQA 报告第 3 节](../../../prontoqa/reports/report.zh-CN.md#3-twonn共同中部峰形阶段差异随深度变化)",
              "", "| 题集 | 阶段 | 峰层 | L1 | 峰值 | L32 | 峰值 − L32 | 峰后上升步数 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    lines.extend(shape_row(summaries, panel, stage) for panel in ("mixed", *FIG2_TASKS) for stage in STAGES)
    lines += ["", "[逐层汇总](id_summary.csv) · [逐次估计](id_replicates.csv) · [题号](sample_ids.csv)"
               " · [子样索引](subsamples.json) · [验证清单](manifest.json)", ""]
    tmp = root / "report.tmp"
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(root / "report.zh-CN.md")


def analyze(root, bundle):
    capture, inventory, batches = capture_inventory(root, bundle)
    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)
    draws = schedule()
    fraction = read_json(CONFIG)["statistics"]["fit_fraction"]
    summaries, estimates = [], []
    for stage in STAGES:
        activations = load_stage(batches[stage], len(bundle["rows"]))
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(analyze_layer, activations[:, index, :], stage, index + 1, draws, fraction)
                    for index in range(LAYERS)]
            for i, job in enumerate(jobs, 1):
                summary, fit = job.result()
                summaries.extend(summary)
                estimates.extend(fit)
                print(f"{stage}/TwoNN: {i}/{LAYERS} layers", flush=True)
        del activations

    sample_ids = [{key: row[key] for key in ("sample_id", "task", "row", "hop", "demos")}
                  for row in bundle["rows"]]
    write_csv(report_dir / "sample_ids.csv", sample_ids)
    write_csv(report_dir / "id_replicates.csv", estimates)
    write_csv(report_dir / "id_summary.csv", summaries)
    atomic_json(report_dir / "subsamples.json", saved_schedule(draws, bundle))
    bounds = write_plots(report_dir, summaries)
    write_report(report_dir, summaries)
    atomic_json(report_dir / "manifest.json", dict(
        input=bundle["identity"], capture=capture, capture_inventory=inventory,
        source=dict(report=sha256_file(Path(__file__)), estimator=sha256_file(Path(estimate_distances.__code__.co_filename))),
        counts=dict(panels=len(PANELS), layers=LAYERS, stages=len(STAGES),
                    estimates=len(estimates), summaries=len(summaries),
                    invalid_estimates=sum(x["status"] != "ok" for x in estimates)),
        y_bounds=bounds, files={name: sha256_file(report_dir / name) for name in OUTPUT_FILES},
    ))


def verify_reports(report_dir, bundle):
    import ast
    import math
    import xml.etree.ElementTree as ET

    manifest = read_json(report_dir / "manifest.json")
    require(set(manifest["files"]) == set(OUTPUT_FILES) and manifest["input"] == bundle["identity"],
            "Report input identity or deliverables differ")
    verify_files(report_dir, manifest["files"])
    receipt = read_json(report_dir / "subsamples.json")
    require(receipt == saved_schedule(schedule(), bundle), "Subsample identity changed")
    for i in range(REPLICATES):
        mixed = receipt["panels"]["mixed"][i]
        require(len(set(mixed)) == SUBSET * len(FIG2_TASKS), "Mixed subsample repeats or drops questions")
        for t, task in enumerate(FIG2_TASKS):
            indices = receipt["panels"][task][i]
            require(len(set(indices)) == SUBSET and all(t * PER_TASK <= n < (t + 1) * PER_TASK for n in indices)
                    and indices == mixed[t * SUBSET:(t + 1) * SUBSET], f"Wrong balanced allocation for {task}")

    def records(name):
        with (report_dir / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    for got, row in zip(records("sample_ids.csv"), bundle["rows"], strict=True):
        require(all(got[key] == str(row[key]) for key in ("sample_id", "task", "row", "hop"))
                and ast.literal_eval(got["demos"]) == row["demos"], "Report sample IDs differ")
    estimates, summaries = records("id_replicates.csv"), records("id_summary.csv")
    key = lambda row: (row["panel"], row["stage"], int(row["layer"]))
    wanted = {(panel, stage, layer) for panel in PANELS for stage in STAGES for layer in range(1, LAYERS + 1)}
    require(len(summaries) == len(wanted) == manifest["counts"]["summaries"] and
            {key(row) for row in summaries} == wanted, "Wrong summary grid")
    groups = {item: [] for item in wanted}
    for fit in estimates:
        groups[key(fit)].append(fit)
    require(len(estimates) == len(wanted) * REPLICATES == manifest["counts"]["estimates"]
            and manifest["counts"]["invalid_estimates"] == sum(f["status"] != "ok" for f in estimates),
            "Wrong replicate count")
    for summary in summaries:
        fits = groups[key(summary)]
        n = SUBSET * (len(FIG2_TASKS) if summary["panel"] == "mixed" else 1)
        n_pool = n * PER_TASK // SUBSET
        require({int(fit["replicate"]) for fit in fits} == set(range(1, REPLICATES + 1)) and
                int(summary["n_pool"]) == n_pool and int(summary["n_subsample"]) == n and
                all(int(fit["n_total"]) == n and fit["status"] in {"ok", "invalid"} for fit in fits),
                "Replicate identity, sample size or fit status differs")
        good = [float(fit["id"]) for fit in fits if fit["status"] == "ok"]
        require(all(math.isfinite(x) and x > 0 for x in good), "Invalid successful estimates")
        if len(good) == REPLICATES:
            mean = math.fsum(good) / REPLICATES
            sd = math.sqrt(math.fsum((x - mean) ** 2 for x in good) / REPLICATES)
            require(summary["status"] == "ok" and math.isclose(float(summary["id_mean"]), mean, abs_tol=1e-12)
                    and math.isclose(float(summary["id_sd"]), sd, abs_tol=1e-12), "Summary differs from replicates")
        else:
            require(summary["status"] == "invalid" and not summary["id_mean"] and not summary["id_sd"],
                    "Invalid layer was plotted")
    for panel in PANELS:
        svg = ET.parse(report_dir / "figures" / f"{panel}.svg")
        require(set(STAGES) <= {element.get("id") for element in svg.iter()}, f"Missing stage artists for {panel}")
        with (report_dir / "figures" / f"{panel}.png").open("rb") as stream:
            require(stream.read(8) == b"\x89PNG\r\n\x1a\n", f"Bad PNG for {panel}")
    return manifest


def verify(root, bundle):
    manifest = verify_reports(root / "reports", bundle)
    capture, inventory, _ = capture_inventory(root, bundle)
    require(manifest["capture"] == capture and manifest["capture_inventory"] == inventory, "Report source changed")
    print(f"Verified all activations and reports: {manifest['counts']}", flush=True)
