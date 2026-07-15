from __future__ import annotations

from collections import Counter
import json

import numpy as np
import pytest
import yaml

from ctm_fusion_experiment.evaluate_baseline_v9_oof import build_baseline_v9_oof_summary
from ctm_fusion_experiment.evaluate_baseline_v9_fixed_policy import build_baseline_v9_fixed_policy_summary
from ctm_fusion_experiment.evaluate_baseline_v10_repeated_cv import build_repeated_cv_summary
from ctm_fusion_experiment.train_baseline_v9_oof import _make_inner_splits, _subset_arrays
from ctm_fusion_experiment.utils.data_loader import FoldSplit, FusionArraySet


def _toy_arrays(sample_count: int = 12) -> FusionArraySet:
    sample_ids = tuple(f"s{index:02d}" for index in range(sample_count))
    return FusionArraySet(
        sample_ids=sample_ids,
        graph=np.arange(sample_count * 2, dtype=np.float32).reshape(sample_count, 2),
        clinical=np.arange(sample_count * 3, dtype=np.float32).reshape(sample_count, 3),
        metabolite=np.arange(sample_count * 4, dtype=np.float32).reshape(sample_count, 4),
        time=np.arange(1, sample_count + 1, dtype=np.float32),
        event=np.asarray([index % 2 for index in range(sample_count)], dtype=np.float32),
    )


def test_subset_arrays_preserves_requested_order_and_values() -> None:
    arrays = _toy_arrays()
    split = FoldSplit(
        fold=1,
        train_ids=("s03", "s01", "s09"),
        val_ids=("s02", "s00"),
        test_ids=("s11", "s04"),
    )

    subset = _subset_arrays(arrays, split)

    assert subset.train.sample_ids == ("s03", "s01", "s09")
    assert subset.val.sample_ids == ("s02", "s00")
    assert subset.test.sample_ids == ("s11", "s04")
    np.testing.assert_array_equal(subset.train.graph[:, 0], np.asarray([6.0, 2.0, 18.0], dtype=np.float32))
    np.testing.assert_array_equal(subset.test.time, np.asarray([12.0, 5.0], dtype=np.float32))


def test_make_inner_splits_produces_oof_test_coverage() -> None:
    arrays = _toy_arrays()

    splits = _make_inner_splits(arrays, folds=3, val_ratio=0.25, seed=17)

    assert len(splits) == 3
    test_counts = Counter(sample_id for split in splits for sample_id in split.test_ids)
    assert test_counts == Counter(arrays.sample_ids)
    for split in splits:
        assert set(split.train_ids).isdisjoint(split.val_ids)
        assert set(split.train_ids).isdisjoint(split.test_ids)
        assert set(split.val_ids).isdisjoint(split.test_ids)


def test_baseline_v9_configs_are_independent() -> None:
    formal = yaml.safe_load(open("ctm_fusion_experiment/configs/baseline_v9_oof.yaml", encoding="utf-8"))
    smoke = yaml.safe_load(open("ctm_fusion_experiment/configs/baseline_v9_oof_smoke.yaml", encoding="utf-8"))
    replication_by_seed = {
        seed: yaml.safe_load(
            open(f"ctm_fusion_experiment/configs/baseline_v10_oof_replication_seed{seed}.yaml", encoding="utf-8")
        )
        for seed in (7, 21, 123, 2026)
    }

    assert formal["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/baseline_v9_oof_formal"
    assert smoke["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/baseline_v9_oof_smoke"
    for seed, replication in replication_by_seed.items():
        assert replication["paths"]["output_dir"] == f"outputs/ctm_fusion_experiment/baseline_v10_oof_replication_seed{seed}"
        assert replication["seed"] == seed
    assert formal["baseline_v9"]["inner_folds"] == 3
    assert smoke["baseline_v9"]["inner_folds"] == 2
    assert formal["baseline_v5"]["selection_policy"] == "ensemble_only_or_reference"


def test_build_baseline_v9_oof_summary_writes_v9_named_outputs(tmp_path) -> None:
    fold_dir = tmp_path / "fold_01"
    fold_dir.mkdir()
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "graph_encoders": [{"training_seconds": 1.0}],
                "reference_baseline": {"test": {"c_index": 0.5}},
                "candidate_models": [
                    {"training_seconds": 1.0, "test": {"c_index": 0.5}},
                    {"training_seconds": 1.0, "test": {"c_index": 0.6}},
                ],
                "baseline_v5_selected": {
                    "candidate_name": "top2_mean",
                    "weights": [0.5, 0.5],
                    "validation_reference_c_index": 0.5,
                    "validation_c_index": 0.55,
                    "test": {"c_index": 0.6},
                    "pair_diagnostics": {
                        "improved_pairs": 4.0,
                        "regressed_pairs": 1.0,
                        "net_improved_pairs": 3.0,
                        "pair_credit_delta": 3.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_baseline_v9_oof_summary(tmp_path)

    assert summary["selected_paired_comparison"]["mean_delta"] == pytest.approx(0.1)
    assert "out-of-fold head predictions" in summary["interpretation"]
    assert (tmp_path / "baseline_v9_oof_summary.json").exists()
    assert (tmp_path / "baseline_v9_oof_fold_comparison.csv").exists()


def test_build_baseline_v9_fixed_policy_summary_uses_saved_predictions(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    fold_dir = source / "fold_01"
    fold_dir.mkdir(parents=True)
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "candidate_models": [
                    {"name": "reference"},
                    {"name": "g0_b11"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (fold_dir / "baseline_v9_oof_selection.json").write_text(
        json.dumps(
            {
                "risk_means": [0.0, 0.0],
                "risk_stds": [1.0, 1.0],
                "oof_reference_c_index": 0.0,
                "candidates": [
                    {"candidate_name": "reference", "weights": [1.0, 0.0], "c_index": 0.0, "c_index_delta": 0.0},
                    {"candidate_name": "mean_all", "weights": [0.0, 1.0], "c_index": 1.0, "c_index_delta": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    pd_rows = [
        "sample_id,time,event,reference_risk,g0_b11_risk",
        "s1,1.0,1.0,1.0,4.0",
        "s2,2.0,1.0,2.0,3.0",
        "s3,3.0,1.0,3.0,2.0",
        "s4,4.0,1.0,4.0,1.0",
    ]
    (fold_dir / "test_predictions.csv").write_text("\n".join(pd_rows), encoding="utf-8")

    summary = build_baseline_v9_fixed_policy_summary(source, output, policy_name="mean_all")

    assert summary["selected_paired_comparison"]["mean_delta"] == pytest.approx(1.0)
    assert summary["folds"][0]["policy_name"] == "mean_all"
    assert (output / "baseline_v9_fixed_mean_all_summary.json").exists()
    assert (output / "baseline_v9_fixed_mean_all_fold_comparison.csv").exists()

    shrunk = build_baseline_v9_fixed_policy_summary(
        source,
        output,
        policy_name="mean_all",
        shrinkage_alpha=0.0,
    )

    assert shrunk["selected_paired_comparison"]["mean_delta"] == pytest.approx(0.0)
    assert shrunk["folds"][0]["shrinkage_alpha"] == pytest.approx(0.0)
    assert (output / "baseline_v9_fixed_mean_all_alpha_0p0_summary.json").exists()


def test_build_repeated_cv_summary_combines_runs(tmp_path) -> None:
    summaries = []
    for run_name, selected in (("seed_a", 0.6), ("seed_b", 0.7)):
        path = tmp_path / f"{run_name}.json"
        path.write_text(
            json.dumps(
                {
                    "policy_name": "softmax_all",
                    "shrinkage_alpha": 0.1,
                    "folds": [
                        {
                            "fold": 1,
                            "reference_c_index": 0.5,
                            "selected_c_index": selected,
                            "selected_delta": selected - 0.5,
                            "policy_name": "softmax_all",
                            "shrinkage_alpha": 0.1,
                            "net_improved_pairs": 1.0,
                            "pair_credit_delta": 1.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        summaries.append((run_name, path))

    output = tmp_path / "out"
    result = build_repeated_cv_summary(summaries, output, output_stem="combined")

    assert result["num_repeated_cv_folds"] == 2
    assert result["selected_paired_comparison"]["mean_delta"] == pytest.approx(0.15)
    assert result["runs"] == ["seed_a", "seed_b"]
    assert (output / "combined_summary.json").exists()
    assert (output / "combined_folds.csv").exists()
