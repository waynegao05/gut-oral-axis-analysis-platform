from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.ac_icam_real_outcome_v8.benchmark import _fast_c_index
from experiments.ac_icam_real_outcome_v8.data import (
    PANEL_PATH,
    load_v8_cohort,
)
from experiments.ac_icam_real_outcome_v8.modeling import (
    ClinicalTransformer,
    MicrobiomeTransformer,
)
from research.metrics import concordance_index


def test_v8_real_cohort_alignment_and_endpoint_counts() -> None:
    cohort = load_v8_cohort()
    assert len(cohort.patients) == 246
    assert len(cohort.genera) == 517
    assert cohort.tumor.shape == (246, 517)
    assert cohort.normal.shape == (246, 517)
    assert np.isfinite(cohort.tumor).all()
    assert np.isfinite(cohort.normal).all()

    all_stage_pfs = cohort.subset(endpoint="PFS", scope="all_stage")
    stage_i_iii_pfs = cohort.subset(
        endpoint="PFS",
        scope="stage_i_iii",
    )
    assert len(all_stage_pfs.patients) == 246
    assert int(all_stage_pfs.patients["pfs_event"].sum()) == 71
    assert len(stage_i_iii_pfs.patients) == 209
    assert int(stage_i_iii_pfs.patients["pfs_event"].sum()) == 37


def test_published_panels_are_complete_and_reference_only() -> None:
    cohort = load_v8_cohort()
    report = cohort.quality_report["published_panels"]
    assert report["mbr_tumor"]["matched_features"] == 41
    assert report["mrs_tumor"]["matched_features"] == 14
    assert report["mrs_normal"]["matched_features"] == 12
    assert not report["candidate_use_allowed"]

    panels = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    assert panels["mbr_2023"]["reference_only"]
    assert panels["mrs_16s_2025"]["reference_only"]


def test_clinical_transformer_handles_unseen_categories() -> None:
    train = pd.DataFrame(
        {
            "age": [50.0, 60.0, 70.0],
            "stage": [1.0, 2.0, 3.0],
            "path_t": [2.0, 3.0, 4.0],
            "path_n": [0.0, 1.0, 2.0],
            "path_m": [0.0, np.nan, 0.0],
            "sex": ["Female", "Male", "Female"],
            "tumor_location": ["Left", "Right", "Left"],
            "tumor_morphology": ["A", "B", "A"],
        }
    )
    transformer = ClinicalTransformer.fit(
        train,
        include_icr=False,
        include_treatment=False,
    )
    test = train.iloc[[0]].copy()
    test.loc[:, "tumor_location"] = "Unseen"
    transformed = transformer.transform(test)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] == len(transformer.feature_names)
    assert np.isfinite(transformed).all()


def test_microbiome_filter_and_selection_are_train_local() -> None:
    tumor = np.asarray(
        [
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
            [0.6, 0.4, 0.0],
            [0.5, 0.5, 0.0],
            [0.4, 0.6, 0.0],
            [0.3, 0.7, 0.0],
        ],
        dtype=float,
    )
    normal = tumor[:, ::-1].copy()
    normal[:, 0] = 0.2
    normal[:, 1] = 0.8
    normal[:, 2] = 0.0
    transformer = MicrobiomeTransformer.fit(
        tumor,
        normal,
        ("A", "B", "test_only"),
        np.asarray([1, 2, 3, 4, 5, 6], dtype=float),
        np.asarray([1, 1, 0, 1, 0, 1], dtype=float),
        prevalence_threshold=0.20,
        top_k=2,
    )
    assert transformer.prevalence_mask.tolist() == [True, True, False]
    transformed = transformer.transform(tumor, normal)
    assert transformed.shape == (6, 10)
    assert np.isfinite(transformed).all()


def test_fast_c_index_matches_project_metric() -> None:
    rng = np.random.default_rng(42)
    time = rng.uniform(1.0, 100.0, size=40)
    event = rng.binomial(1, 0.45, size=40)
    risk = rng.normal(size=40)
    assert _fast_c_index(time, event, risk) == concordance_index(
        time,
        event,
        risk,
    )
