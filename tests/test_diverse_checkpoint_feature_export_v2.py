from __future__ import annotations

import json

import numpy as np
import pytest

from research.diverse_checkpoint_feature_export_v2 import (
    _extend_with_main_model_risk,
    _parse_paths,
    _parse_optional_strings,
    collect_checkpoint_specs,
)


def test_collect_checkpoint_specs_reads_baseline_and_diversity_rows(tmp_path) -> None:
    baseline_dir = tmp_path / "baseline" / "research_seed7"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "best_model.pt").write_bytes(b"")
    (baseline_dir / "config_snapshot.yaml").write_text("seed: 7\n", encoding="utf-8")

    diverse_dir = tmp_path / "diverse" / "ranking_w0p02" / "seed_21"
    diverse_dir.mkdir(parents=True)
    (diverse_dir / "best_model.pt").write_bytes(b"")
    (diverse_dir / "config_snapshot.yaml").write_text("seed: 21\n", encoding="utf-8")
    config_path = tmp_path / "generated.yaml"
    config_path.write_text("seed: 21\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "variant": "ranking_w0p02",
                        "seed": 21,
                        "output_dir": str(diverse_dir),
                        "config_path": str(config_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = collect_checkpoint_specs(
        diversity_summary_paths=[summary_path],
        baseline_checkpoint_glob=str(tmp_path / "baseline" / "*" / "best_model.pt"),
        base_config_path="research_config_v2.yaml",
        include_baseline_checkpoints=True,
    )

    assert [spec.variant for spec in specs] == ["baseline", "ranking_w0p02"]
    assert specs[0].seed == 7
    assert specs[1].seed == 21
    assert specs[1].member_name == "ranking_w0p02:seed21"


def test_collect_checkpoint_specs_deduplicates_duplicate_checkpoints(tmp_path) -> None:
    run_dir = tmp_path / "diverse" / "topk8" / "seed_7"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pt").write_bytes(b"")
    (run_dir / "config_snapshot.yaml").write_text("seed: 7\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    row = {"variant": "topk8", "seed": 7, "output_dir": str(run_dir), "config_path": str(run_dir / "config_snapshot.yaml")}
    summary_path.write_text(json.dumps({"runs": [row, row]}), encoding="utf-8")

    specs = collect_checkpoint_specs(
        diversity_summary_paths=[summary_path],
        baseline_checkpoint_glob=str(run_dir / "best_model.pt"),
        base_config_path="research_config_v2.yaml",
        include_baseline_checkpoints=False,
    )

    assert len(specs) == 1


def test_collect_checkpoint_specs_filters_variants(tmp_path) -> None:
    keep_dir = tmp_path / "diverse" / "ranking_w0p02" / "seed_7"
    drop_dir = tmp_path / "diverse" / "topk8" / "seed_7"
    keep_dir.mkdir(parents=True)
    drop_dir.mkdir(parents=True)
    (keep_dir / "best_model.pt").write_bytes(b"")
    (drop_dir / "best_model.pt").write_bytes(b"")
    (keep_dir / "config_snapshot.yaml").write_text("seed: 7\n", encoding="utf-8")
    (drop_dir / "config_snapshot.yaml").write_text("seed: 7\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "runs": [
                    {"variant": "ranking_w0p02", "seed": 7, "output_dir": str(keep_dir), "config_path": str(keep_dir / "config_snapshot.yaml")},
                    {"variant": "topk8", "seed": 7, "output_dir": str(drop_dir), "config_path": str(drop_dir / "config_snapshot.yaml")},
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = collect_checkpoint_specs(
        diversity_summary_paths=[summary_path],
        baseline_checkpoint_glob=str(keep_dir / "best_model.pt"),
        base_config_path="research_config_v2.yaml",
        include_baseline_checkpoints=False,
        include_variants=["ranking_w0p02"],
    )

    assert len(specs) == 1
    assert specs[0].variant == "ranking_w0p02"


def test_extend_with_main_model_risk_appends_raw_and_standardized_member() -> None:
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    standardized = np.asarray([[0.0, 1.0], [2.0, 3.0]])

    extended_raw, extended_standardized = _extend_with_main_model_risk(
        raw_matrix=raw,
        standardized_matrix=standardized,
        main_risk=np.asarray([10.0, 14.0]),
        main_val_mean=8.0,
        main_val_std=2.0,
    )

    np.testing.assert_allclose(extended_raw[-1], [10.0, 14.0])
    np.testing.assert_allclose(extended_standardized[-1], [1.0, 3.0])
    assert extended_raw.shape == (3, 2)


def test_parse_paths_ignores_empty_parts() -> None:
    assert _parse_paths("a.json, b.json,,") == ["a.json", "b.json"]
    assert _parse_optional_strings("") is None
    assert _parse_optional_strings("ranking_w0p02, aux_light") == ["ranking_w0p02", "aux_light"]


def test_collect_checkpoint_specs_rejects_missing_baseline(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_checkpoint_specs(
            diversity_summary_paths=[],
            baseline_checkpoint_glob=str(tmp_path / "missing" / "*.pt"),
            base_config_path="research_config_v2.yaml",
            include_baseline_checkpoints=True,
        )
