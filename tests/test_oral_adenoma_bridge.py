from __future__ import annotations

import joblib
import numpy as np
import pytest

import enhanced_app
from config.settings import (
    ORAL_ADENOMA_ARTIFACT_PATH,
    ORAL_ADENOMA_ARTIFACT_SHA256,
    PROJECT_ROOT,
)
from experiments.oral_adenoma_internal_v3.benchmark import load_inputs
from experiments.oral_adenoma_internal_v3.predict import predict_frame
from src.oral_adenoma_bridge import OralAdenomaBridge


def _bridge() -> OralAdenomaBridge:
    return OralAdenomaBridge(
        ORAL_ADENOMA_ARTIFACT_PATH,
        ORAL_ADENOMA_ARTIFACT_SHA256,
    )


def _oral_abundances(bridge: OralAdenomaBridge, row) -> dict[str, float]:
    return {
        taxonomy: float(row[feature_id])
        for feature_id, taxonomy in zip(
            bridge.feature_ids,
            bridge.taxonomies,
            strict=True,
        )
    }


def test_json_release_matches_locked_sklearn_bundle() -> None:
    model_path = (
        PROJECT_ROOT
        / "outputs"
        / "oral_adenoma_internal_v3"
        / "oral_adenoma_internal_model.joblib"
    )
    if not model_path.exists():
        pytest.skip("The source sklearn bundle is a local research output.")

    bridge = _bridge()
    frame, _ = load_inputs()
    bundle = joblib.load(model_path)
    expected = predict_frame(bundle, frame)["adenoma_probability"].to_numpy()
    actual = np.asarray(
        [
            bridge.score(
                _oral_abundances(bridge, row),
                sample_type="oral_swab",
            ).probability
            for _, row in frame.iterrows()
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(
        actual >= bridge.threshold,
        expected >= float(bundle["threshold"]),
    )


def test_bridge_requires_complete_oral_percentages_and_rejects_stool() -> None:
    bridge = _bridge()
    frame, _ = load_inputs()
    abundances = _oral_abundances(bridge, frame.iloc[0])

    prediction = bridge.score(abundances, sample_type="saliva")
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.as_dict()["research_only"] is True

    incomplete = dict(abundances)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="完整的 381 个菌属"):
        bridge.score(incomplete, sample_type="oral_swab")

    with pytest.raises(ValueError, match="禁止粪便"):
        bridge.score(abundances, sample_type="stool")


def test_internal_endpoint_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(enhanced_app, "ENABLE_INTERNAL_ORAL_ADENOMA", False)
    response = enhanced_app.app.test_client().get(
        "/internal/oral-adenoma/schema"
    )
    assert response.status_code == 404
    assert response.get_json()["ok"] is False


def test_internal_endpoint_runs_when_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.setattr(enhanced_app, "ENABLE_INTERNAL_ORAL_ADENOMA", True)
    bridge = _bridge()
    frame, _ = load_inputs()
    response = enhanced_app.app.test_client().post(
        "/internal/oral-adenoma/analyze",
        json={
            "sample_type": "oral_swab",
            "oral_abundances": _oral_abundances(bridge, frame.iloc[0]),
        },
    )

    assert response.status_code == 200
    result = response.get_json()["oral_adenoma_result"]
    assert result["model_release"] == "oral_adenoma_internal_v3"
    assert result["research_only"] is True
    assert result["verified_diminutive_adenoma_le_5mm"] is False
    assert result["formal_internal_metrics"]["adenoma_sensitivity"][
        "value"
    ] == pytest.approx(22 / 34)
    assert result["formal_internal_metrics"]["false_positive_rate"][
        "value"
    ] == pytest.approx(3 / 58)


def test_index_uses_typescript_bundle_and_hides_internal_panel_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(enhanced_app, "ENABLE_INTERNAL_ORAL_ADENOMA", False)
    html = enhanced_app.app.test_client().get("/").get_data(as_text=True)
    assert 'static/generated/app.js' in html
    assert 'id="oral-adenoma-panel"' not in html


def test_index_shows_internal_panel_only_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(enhanced_app, "ENABLE_INTERNAL_ORAL_ADENOMA", True)
    html = enhanced_app.app.test_client().get("/").get_data(as_text=True)
    assert 'id="oral-adenoma-panel"' in html
    assert "不接受粪便、血液或组织数据" in html
