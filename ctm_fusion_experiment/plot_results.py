from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, plots_dir: Path, filename: str) -> str:
    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def build_plots(output_dir: str | Path) -> list[str]:
    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fold_dirs = sorted(path.parent for path in output_path.glob("fold_*/fold_summary.json"))
    if not fold_dirs:
        raise ValueError(f"No fold summaries found under {output_path.as_posix()}.")

    summaries = [_read_json(fold_dir / "fold_summary.json") for fold_dir in fold_dirs]
    folds = [int(summary["fold"]) for summary in summaries]
    baseline_scores = [float(summary["baseline"]["test"]["c_index"]) for summary in summaries]
    ctm_scores = [float(summary["ctm"]["test"]["c_index"]) for summary in summaries]
    written = []

    x = np.arange(len(folds))
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(x - 0.18, baseline_scores, width=0.36, label="Concat-Cox")
    axis.bar(x + 0.18, ctm_scores, width=0.36, label="CTM-Cox")
    axis.set_xticks(x, [str(fold) for fold in folds])
    axis.set_xlabel("CV fold")
    axis.set_ylabel("Test c-index")
    axis.set_title("Fold-level c-index comparison")
    axis.legend()
    written.append(_save(fig, plots_dir, "fold_cindex_comparison.png"))

    baseline_params = [int(summary["baseline"].get("parameters", 0)) for summary in summaries]
    ctm_params = [int(summary["ctm"].get("parameters", 0)) for summary in summaries]
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Concat-Cox", "CTM-Cox"], [np.mean(baseline_params), np.mean(ctm_params)])
    axis.set_ylabel("Trainable parameters")
    axis.set_title("Fusion-model parameter count")
    written.append(_save(fig, plots_dir, "parameter_comparison.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for fold_dir in fold_dirs:
        fold = fold_dir.name
        for history_name, label, color in [
            ("baseline_history.json", "Concat-Cox", "tab:blue"),
            ("ctm_history.json", "CTM-Cox", "tab:orange"),
        ]:
            history = _read_json(fold_dir / history_name)
            epochs = [row["epoch"] for row in history]
            axes[0].plot(epochs, [row["train_loss"] for row in history], color=color, alpha=0.55, label=f"{fold} {label}")
            axes[1].plot(epochs, [row["c_index"] for row in history], color=color, alpha=0.55, label=f"{fold} {label}")
    axes[0].set_title("Fusion training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation c-index")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("c-index")
    axes[1].legend(fontsize=7)
    written.append(_save(fig, plots_dir, "training_curves.png"))

    tick_counts: Counter[int] = Counter()
    attention = []
    for fold_dir in fold_dirs:
        analysis = _read_json(fold_dir / "ctm_analysis.json")
        tick_counts.update({int(tick): int(count) for tick, count in analysis["stable_tick_histogram"].items()})
        attention.append(np.asarray(analysis["mean_attention_by_tick_and_modality"], dtype=float))

    fig, axis = plt.subplots(figsize=(6, 4))
    ticks = sorted(tick_counts)
    axis.bar([str(tick) for tick in ticks], [tick_counts[tick] for tick in ticks])
    axis.set_xlabel("Selected stable tick")
    axis.set_ylabel("Test samples")
    axis.set_title("CTM stable-tick distribution")
    written.append(_save(fig, plots_dir, "ctm_stable_tick_histogram.png"))

    mean_attention = np.mean(np.stack(attention), axis=0)
    fig, axis = plt.subplots(figsize=(7, 4))
    image = axis.imshow(mean_attention.T, aspect="auto", cmap="viridis")
    axis.set_yticks([0, 1, 2], ["graph", "clinical", "metabolomics"])
    axis.set_xticks(np.arange(mean_attention.shape[0]), np.arange(mean_attention.shape[0]))
    axis.set_xlabel("Internal CTM tick")
    axis.set_title("Mean modality attention by CTM tick")
    fig.colorbar(image, ax=axis, label="attention weight")
    written.append(_save(fig, plots_dir, "ctm_modality_attention.png"))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/formal")
    args = parser.parse_args()
    print(json.dumps({"plots": build_plots(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
