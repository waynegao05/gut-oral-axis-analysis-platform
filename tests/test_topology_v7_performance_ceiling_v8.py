from __future__ import annotations

import json
from pathlib import Path

from experiments.topology_v7_performance_ceiling_v8 import ceiling_audit


ROOT = Path(__file__).resolve().parents[1]
LOCK = (
    ROOT
    / "experiments/topology_v7_performance_ceiling_v8/protocol_lock.json"
)


def test_performance_ceiling_protocol_is_audit_only() -> None:
    protocol = json.loads(LOCK.read_text(encoding="utf-8"))
    assert protocol["status"] == "audit_only"
    assert protocol["target_c_index"] == 0.761
    assert "deployment" in protocol["prohibited_uses"]


def test_latent_target_is_not_written_to_oof_export() -> None:
    source = Path(ceiling_audit.__file__).read_text(encoding="utf-8")
    assert "audit_only_oof_predictions_without_latent_target.csv" in source
    assert "oof.drop(columns=[PROVENANCE_TARGET])" in source


def test_decision_does_not_treat_latent_score_as_mathematical_ceiling() -> None:
    source = Path(ceiling_audit.__file__).read_text(encoding="utf-8")
    assert "not a mathematical upper bound" in source
