from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.data import apply_tabular_standardizer, fit_tabular_standardizer, validate_research_feature_tables


def _valid_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    graph = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S2", "S2"],
            "node_name": ["A", "B", "A", "B"],
            "abundance": [0.2, 0.3, 0.4, 0.1],
            "function_score": [0.4, 0.5, 0.6, 0.2],
            "edge_weight": [0.5, 0.5, 0.3, 0.3],
        }
    )
    clinical = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "age": [50.0, 60.0],
            "bmi": [23.0, 27.0],
            "smoking": [0.0, 1.0],
            "family_history": [1.0, 0.0],
        }
    )
    metabolites = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "bile_acids": [0.3, 0.4],
            "scfa": [0.2, 0.5],
        }
    )
    return graph, clinical, metabolites


def test_research_feature_tables_report_validated_ranges() -> None:
    graph, clinical, metabolites = _valid_tables()

    summary = validate_research_feature_tables(graph, clinical, metabolites)

    assert summary["validated"] is True
    assert summary["observed_ranges"]["clinical table"]["age"] == {"min": 50.0, "max": 60.0}


def test_research_feature_tables_reject_negative_abundance() -> None:
    graph, clinical, metabolites = _valid_tables()
    graph.loc[0, "abundance"] = -0.1

    with pytest.raises(ValueError, match="graph table.abundance"):
        validate_research_feature_tables(graph, clinical, metabolites)


def test_research_feature_tables_reject_non_finite_age() -> None:
    graph, clinical, metabolites = _valid_tables()
    clinical.loc[0, "age"] = np.inf

    with pytest.raises(ValueError, match="clinical table.age"):
        validate_research_feature_tables(graph, clinical, metabolites)


def test_research_feature_tables_reject_invalid_binary_value() -> None:
    graph, clinical, metabolites = _valid_tables()
    clinical.loc[0, "smoking"] = 0.5

    with pytest.raises(ValueError, match="clinical table.smoking"):
        validate_research_feature_tables(graph, clinical, metabolites)


def test_tabular_standardizer_uses_train_statistics_for_all_splits() -> None:
    train = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "age": [40.0, 60.0],
            "smoking": [0.0, 1.0],
            "bile_acids": [0.2, 0.4],
        }
    )
    validation = pd.DataFrame(
        {
            "sample_id": ["V1"],
            "age": [80.0],
            "smoking": [1.0],
            "bile_acids": [0.5],
        }
    )

    standardizer = fit_tabular_standardizer(
        train,
        clinical_columns=["age", "smoking"],
        metabolite_columns=["bile_acids"],
    )
    train_scaled = apply_tabular_standardizer(train, standardizer)
    validation_scaled = apply_tabular_standardizer(validation, standardizer)

    np.testing.assert_allclose(
        train_scaled[["age", "smoking", "bile_acids"]].mean().to_numpy(),
        0.0,
        atol=1e-12,
    )
    assert standardizer["fit_split"] == "train"
    assert standardizer["features"]["age"]["mean"] == pytest.approx(50.0)
    assert standardizer["features"]["age"]["scale"] == pytest.approx(10.0)
    assert validation_scaled.loc[0, "age"] == pytest.approx(3.0)


def test_tabular_standardizer_handles_zero_variance_feature() -> None:
    train = pd.DataFrame({"age": [50.0, 50.0], "bile_acids": [0.2, 0.4]})

    standardizer = fit_tabular_standardizer(
        train,
        clinical_columns=["age"],
        metabolite_columns=["bile_acids"],
    )
    transformed = apply_tabular_standardizer(train, standardizer)

    assert standardizer["features"]["age"]["zero_variance"] is True
    assert standardizer["features"]["age"]["scale"] == pytest.approx(1.0)
    np.testing.assert_allclose(transformed["age"].to_numpy(), 0.0)
