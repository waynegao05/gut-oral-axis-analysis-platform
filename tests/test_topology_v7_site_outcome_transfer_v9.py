from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.topology_v7_site_outcome_transfer_v9.development_screen import (
    _signed_survival_label,
)
from experiments.topology_v7_site_outcome_transfer_v9.public_prior import (
    PANEL_TAXA,
    panel_clr,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = (
    ROOT
    / "experiments/topology_v7_site_outcome_transfer_v9/protocol_lock.json"
)


def test_signed_survival_label_marks_censoring_negative() -> None:
    labels = _signed_survival_label(
        np.array([10.0, 20.0, 30.0]),
        np.array([1.0, 0.0, 1.0]),
    )
    assert labels.tolist() == [10.0, -20.0, 30.0]


def test_panel_clr_is_centered_per_sample() -> None:
    values = np.arange(1, len(PANEL_TAXA) * 2 + 1, dtype=float).reshape(
        2, len(PANEL_TAXA)
    )
    transformed = panel_clr(values)
    assert transformed.shape == values.shape
    assert np.allclose(transformed.mean(axis=1), 0.0)


def test_protocol_excludes_precomputed_edge_weights_and_formal_tuning() -> None:
    protocol = json.loads(LOCK.read_text(encoding="utf-8"))
    assert protocol["candidate_model"]["precomputed_edge_weights_used"] is False
    assert protocol["dataset"]["formal"] == "data/research/topology_v7_generator_v3"
    assert protocol["formal_data_may_be_used_for_tuning"] is False
    assert protocol["public_outcome_prior"]["source_url"].startswith(
        "https://zenodo.org/"
    )
