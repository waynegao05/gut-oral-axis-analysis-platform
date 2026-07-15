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


INK = "#17212B"
MUTED = "#647180"
GRID = "#E3E8EE"
BLUE = "#356C9A"
BLUE_LIGHT = "#A9C4D8"
GOLD = "#C18A28"
RANGE = "#A4AFBA"
BACKGROUND = "#FBFCFE"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auc_series(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = report["horizons"]
    return (
        np.asarray([float(row["horizon"]) for row in rows], dtype=float),
        np.asarray([float(row["auc"]) for row in rows], dtype=float),
    )


def _assert_same_horizons(reference: np.ndarray, candidate: np.ndarray, label: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(reference, candidate):
        raise ValueError(f"AUC horizons do not match for {label}.")


def build_auc_figure(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> list[str]:
    source_dir = Path(input_dir)
    destination = Path(output_dir) if output_dir is not None else source_dir
    destination.mkdir(parents=True, exist_ok=True)

    strict42 = _read_json(source_dir / "split42_three_seed_auc.json")
    strict43 = _read_json(source_dir / "split43_three_seed_auc.json")
    champion = _read_json(source_dir / "split42_champion_auc.json")
    champion_reference = _read_json(source_dir / "split42_champion_reference_auc.json")
    strict43_reference = _read_json(source_dir / "split43_reference_auc.json")

    horizons, auc42 = _auc_series(strict42)
    horizons43, auc43 = _auc_series(strict43)
    champion_horizons, champion_auc = _auc_series(champion)
    champion_reference_horizons, champion_reference_auc = _auc_series(champion_reference)
    strict43_reference_horizons, strict43_reference_auc = _auc_series(strict43_reference)
    for values, label in [
        (horizons43, "strict split43"),
        (champion_horizons, "split42 champion"),
        (champion_reference_horizons, "split42 champion reference"),
        (strict43_reference_horizons, "strict split43 reference"),
    ]:
        _assert_same_horizons(horizons, values, label)

    strict_mean = np.mean(np.vstack([auc42, auc43]), axis=0)
    strict_low = np.minimum(auc42, auc43)
    strict_high = np.maximum(auc42, auc43)
    champion_delta = (champion_auc - champion_reference_auc) * 1000.0
    strict43_delta = (auc43 - strict43_reference_auc) * 1000.0

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

    fig, (axis_auc, axis_delta) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.6),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28},
    )
    fig.patch.set_facecolor(BACKGROUND)
    for axis in (axis_auc, axis_delta):
        axis.set_facecolor(BACKGROUND)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=0)

    x = np.arange(len(horizons), dtype=float)
    cap_width = 0.08
    axis_auc.vlines(x, strict_low, strict_high, color=RANGE, linewidth=3.2, zorder=1)
    axis_auc.hlines(strict_low, x - cap_width, x + cap_width, color=RANGE, linewidth=2.0, zorder=1)
    axis_auc.hlines(strict_high, x - cap_width, x + cap_width, color=RANGE, linewidth=2.0, zorder=1)
    axis_auc.scatter(
        x,
        strict_mean,
        s=95,
        color=BLUE,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    axis_auc.scatter(
        x + 0.16,
        champion_auc,
        s=82,
        marker="D",
        facecolor=GOLD,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    for position, value in zip(x, strict_mean):
        axis_auc.annotate(
            f"{value:.3f}",
            (position, value),
            xytext=(-8, -17),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=9.5,
            color=BLUE,
            fontweight="bold",
        )
    for position, value in zip(x + 0.16, champion_auc):
        axis_auc.annotate(
            f"{value:.3f}",
            (position, value),
            xytext=(6, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9.3,
            color=GOLD,
            fontweight="bold",
        )

    axis_auc.set_xticks(x, [f"{value:g}" for value in horizons])
    axis_auc.set_xlim(-0.35, len(horizons) - 0.55)
    axis_auc.set_ylim(0.785, 0.872)
    axis_auc.set_yticks(np.arange(0.79, 0.871, 0.02))
    axis_auc.grid(axis="y", color=GRID, linewidth=0.9)
    axis_auc.set_axisbelow(True)
    axis_auc.set_xlabel("预设随访时间点（原始 time 单位）", labelpad=10)
    axis_auc.set_ylabel("Cumulative/dynamic AUC（IPCW）")
    axis_auc.set_title("A  时间依赖生存 AUC", loc="left", fontsize=14, fontweight="bold", pad=14)
    auc_legend = [
        Line2D([0], [0], color=RANGE, linewidth=3, label="严格 split42–43 范围"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BLUE,
            markeredgecolor="white",
            markersize=9,
            label="严格跨划分均值",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=GOLD,
            markeredgecolor="white",
            markersize=8,
            label="单划分最高模型（探索性）",
        ),
    ]
    axis_auc.legend(handles=auc_legend, frameon=False, loc="lower left", fontsize=9.5)
    y = np.arange(len(horizons), dtype=float)
    bar_height = 0.28
    axis_delta.barh(
        y - 0.17,
        champion_delta,
        height=bar_height,
        color=GOLD,
        label="split42 最高模型 vs 其基线",
        zorder=3,
    )
    axis_delta.barh(
        y + 0.17,
        strict43_delta,
        height=bar_height,
        color=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=1.0,
        label="split43 严格模型 vs 其基线",
        zorder=3,
    )
    axis_delta.axvline(0.0, color=INK, linewidth=1.1, zorder=2)
    axis_delta.grid(axis="x", color=GRID, linewidth=0.9)
    axis_delta.set_axisbelow(True)
    axis_delta.set_yticks(y, [f"时间点 {value:g}" for value in horizons])
    axis_delta.invert_yaxis()
    axis_delta.set_xlim(-0.30, 1.82)
    axis_delta.set_xlabel("相对基线的 ΔAUC（×10⁻³）", labelpad=10)
    axis_delta.set_title("B  改进模块带来的 AUC 变化", loc="left", fontsize=14, fontweight="bold", pad=14)
    axis_delta.legend(frameon=False, loc="lower right", fontsize=9.2)
    for positions, values, color in [
        (y - 0.17, champion_delta, GOLD),
        (y + 0.17, strict43_delta, BLUE),
    ]:
        for position, value in zip(positions, values):
            offset = 0.05 if value >= 0.0 else -0.05
            axis_delta.text(
                value + offset,
                position,
                f"{value:+.2f}",
                va="center",
                ha="left" if value >= 0.0 else "right",
                fontsize=9.2,
                color=color,
                fontweight="bold",
            )

    fig.suptitle(
        "当前模型的生存 AUC 结果",
        x=0.065,
        y=0.975,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.065,
        0.900,
        "topology_v6 · test n=720/split · 指标考虑右删失；三个时间点为离散预设评估点",
        ha="left",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.065,
        0.025,
        "注：AUC 轴为局部放大，严格均值仅基于 split42/43；普通 event AUC 约为 0.699，"
        "但其忽略随访时间与删失，不与图中的生存 AUC 直接比较。",
        ha="left",
        fontsize=9.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.16)

    written: list[str] = []
    for suffix in ("png", "svg"):
        path = destination / f"survival_auc_v2.{suffix}"
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
            {"figures": build_auc_figure(args.input_dir, output_dir=args.output_dir)},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
