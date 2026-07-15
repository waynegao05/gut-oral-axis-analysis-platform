from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


INK = "#17212B"
MUTED = "#647180"
GRID = "#E3E8EE"
BLUE = "#356C9A"
BLUE_LIGHT = "#BFD3E1"
GOLD = "#C18A28"
CHANCE = "#8995A2"
BACKGROUND = "#FBFCFE"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_horizon(report: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(row["horizon"]): row for row in report["horizons"]}


def _interpolate_roc(row: dict[str, Any], grid: np.ndarray) -> np.ndarray:
    false_positive_rate = np.asarray(row["false_positive_rate"], dtype=float)
    true_positive_rate = np.asarray(row["true_positive_rate"], dtype=float)
    unique_false_positive_rate = np.unique(false_positive_rate)
    maximum_true_positive_rate = np.asarray(
        [
            float(true_positive_rate[false_positive_rate == value].max())
            for value in unique_false_positive_rate
        ],
        dtype=float,
    )
    return np.interp(grid, unique_false_positive_rate, maximum_true_positive_rate)


def build_roc_figure(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> list[str]:
    source_dir = Path(input_dir)
    destination = Path(output_dir) if output_dir is not None else source_dir
    destination.mkdir(parents=True, exist_ok=True)

    strict42 = _rows_by_horizon(_read_json(source_dir / "split42_three_seed_roc.json"))
    strict43 = _rows_by_horizon(_read_json(source_dir / "split43_three_seed_roc.json"))
    champion = _rows_by_horizon(_read_json(source_dir / "split42_champion_roc.json"))
    horizons = sorted(strict42)
    if set(horizons) != set(strict43) or set(horizons) != set(champion):
        raise ValueError("ROC reports do not contain the same horizons.")

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "Arial", "sans-serif"],
            "axes.unicode_minus": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(1, len(horizons), figsize=(14.2, 5.7), sharex=True, sharey=True)
    fig.patch.set_facecolor(BACKGROUND)
    grid = np.linspace(0.0, 1.0, 401)
    for axis, horizon in zip(np.atleast_1d(axes), horizons):
        row42 = strict42[horizon]
        row43 = strict43[horizon]
        champion_row = champion[horizon]
        tpr42 = _interpolate_roc(row42, grid)
        tpr43 = _interpolate_roc(row43, grid)
        strict_mean = np.mean(np.vstack([tpr42, tpr43]), axis=0)
        strict_low = np.minimum(tpr42, tpr43)
        strict_high = np.maximum(tpr42, tpr43)
        champion_fpr = np.asarray(champion_row["false_positive_rate"], dtype=float)
        champion_tpr = np.asarray(champion_row["true_positive_rate"], dtype=float)
        strict_mean_auc = float(np.mean([row42["auc"], row43["auc"]]))

        axis.set_facecolor(BACKGROUND)
        axis.fill_between(
            grid,
            strict_low,
            strict_high,
            color=BLUE_LIGHT,
            alpha=0.5,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(grid, strict_mean, color=BLUE, linewidth=2.5, zorder=3)
        axis.step(
            champion_fpr,
            champion_tpr,
            where="post",
            color=GOLD,
            linewidth=2.0,
            linestyle=(0, (5, 3)),
            zorder=4,
        )
        axis.plot([0.0, 1.0], [0.0, 1.0], color=CHANCE, linewidth=1.2, linestyle=":", zorder=2)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=0)
        axis.set_title(
            f"时间点 {horizon:g}\n严格均值 {strict_mean_auc:.3f} · 探索性最高 {float(champion_row['auc']):.3f}",
            fontsize=12.3,
            fontweight="bold",
            pad=12,
        )
        axis.text(
            0.97,
            0.05,
            f"严格 cases {int(row42['num_cases'])}/{int(row43['num_cases'])}\n"
            f"controls {int(row42['num_controls'])}/{int(row43['num_controls'])}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            color=MUTED,
        )

    axes_array = np.atleast_1d(axes)
    axes_array[0].set_ylabel("真阳性率 TPR（敏感度）", labelpad=10)
    for axis in axes_array:
        axis.set_xlabel("假阳性率 FPR（1 - 特异度）", labelpad=9)

    legend_handles = [
        Patch(facecolor=BLUE_LIGHT, edgecolor="none", alpha=0.5, label="严格 split42–43 范围"),
        Line2D([0], [0], color=BLUE, linewidth=2.5, label="严格跨划分平均 ROC"),
        Line2D(
            [0],
            [0],
            color=GOLD,
            linewidth=2.0,
            linestyle=(0, (5, 3)),
            label="单划分最高模型（探索性）",
        ),
        Line2D([0], [0], color=CHANCE, linewidth=1.2, linestyle=":", label="随机水平"),
    ]
    fig.suptitle(
        "当前模型的时间依赖 ROC 曲线",
        x=0.065,
        y=0.975,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.905,
        "Cumulative/dynamic ROC（IPCW）· topology_v6 · test n=720/split",
        ha="left",
        fontsize=10.5,
        color=MUTED,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.952),
        frameon=False,
        ncol=2,
        fontsize=9.2,
        columnspacing=1.5,
        handlelength=2.7,
    )
    fig.text(
        0.065,
        0.028,
        "定义：病例为该时间点前已观察到事件者，对照为该时间点后仍无事件者；此前删失样本排除，"
        "病例使用训练集估计的 IPCW 权重。浅蓝带仅表示两个划分的范围，不是置信区间。",
        ha="left",
        fontsize=9.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.79, bottom=0.16, wspace=0.20)

    written: list[str] = []
    for suffix in ("png", "svg"):
        path = destination / f"survival_roc_v2.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=BACKGROUND)
        written.append(str(path.as_posix()))
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="outputs/current_mainline_v2/survival_auc_v2",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            {"figures": build_roc_figure(args.input_dir, output_dir=args.output_dir)},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
