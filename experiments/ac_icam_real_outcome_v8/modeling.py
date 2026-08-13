from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def cox_loss_gradient(
    coefficients: np.ndarray,
    values: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    margin = values @ coefficients
    event_times = np.unique(time[event > 0.5])
    num_events = float(event.sum())
    if num_events <= 0.0:
        raise ValueError("Cox training requires at least one observed event.")
    loss = 0.0
    gradient = np.zeros_like(coefficients)
    for current_time in event_times:
        observed = (time == current_time) & (event > 0.5)
        risk_set = time >= current_time
        risk_margin = margin[risk_set]
        shift = float(np.max(risk_margin))
        weights = np.exp(risk_margin - shift)
        denominator = float(weights.sum())
        observed_count = int(observed.sum())
        loss -= float(margin[observed].sum())
        loss += observed_count * (shift + np.log(denominator))
        weighted_mean = (
            values[risk_set] * weights[:, None]
        ).sum(axis=0) / denominator
        gradient -= values[observed].sum(axis=0)
        gradient += observed_count * weighted_mean
    loss = loss / num_events + 0.5 * float(l2) * float(
        coefficients @ coefficients
    )
    gradient = gradient / num_events + float(l2) * coefficients
    return float(loss), np.asarray(gradient, dtype=float)


@dataclass(frozen=True)
class RidgeCoxModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    l2: float
    optimization_success: bool

    def predict(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float).copy()
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = np.broadcast_to(
                self.mean, matrix.shape
            )[missing]
        standardized = (matrix - self.mean) / self.scale
        return np.asarray(
            standardized @ self.coefficients,
            dtype=float,
        )


def fit_ridge_cox(
    values: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    l2: float,
) -> RidgeCoxModel:
    matrix = np.asarray(values, dtype=float).copy()
    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=float)
    mean = np.nanmean(
        np.where(np.isfinite(matrix), matrix, np.nan),
        axis=0,
    )
    mean = np.where(np.isfinite(mean), mean, 0.0)
    missing = ~np.isfinite(matrix)
    if missing.any():
        matrix[missing] = np.broadcast_to(mean, matrix.shape)[missing]
    scale = np.std(matrix, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    standardized = (matrix - mean) / scale

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return cox_loss_gradient(
            beta,
            standardized,
            time_values,
            event_values,
            float(l2),
        )

    result = minimize(
        objective,
        np.zeros(standardized.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-7},
    )
    return RidgeCoxModel(
        mean=np.asarray(mean, dtype=float),
        scale=np.asarray(scale, dtype=float),
        coefficients=np.asarray(result.x, dtype=float),
        l2=float(l2),
        optimization_success=bool(result.success),
    )


@dataclass(frozen=True)
class ClinicalTransformer:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_medians: np.ndarray
    category_levels: tuple[tuple[str, ...], ...]
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        include_icr: bool,
        include_treatment: bool = False,
    ) -> "ClinicalTransformer":
        numeric = ["age", "stage", "path_t", "path_n", "path_m"]
        if include_icr:
            numeric.append("icr_score")
        if include_treatment:
            numeric.append("adjuvant_any")
        categorical = ["sex", "tumor_location", "tumor_morphology"]
        numeric_values = (
            frame[numeric]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
        medians = np.nanmedian(numeric_values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        category_levels: list[tuple[str, ...]] = []
        feature_names = list(numeric)
        for column in categorical:
            levels = tuple(
                sorted(
                    frame[column]
                    .fillna("__missing__")
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )
            category_levels.append(levels)
            feature_names.extend(
                f"{column}=={level}" for level in levels
            )
        return cls(
            numeric_columns=tuple(numeric),
            categorical_columns=tuple(categorical),
            numeric_medians=np.asarray(medians, dtype=float),
            category_levels=tuple(category_levels),
            feature_names=tuple(feature_names),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = (
            frame[list(self.numeric_columns)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
        missing = ~np.isfinite(numeric)
        if missing.any():
            numeric[missing] = np.broadcast_to(
                self.numeric_medians,
                numeric.shape,
            )[missing]
        blocks = [numeric]
        for column, levels in zip(
            self.categorical_columns,
            self.category_levels,
        ):
            values = frame[column].fillna("__missing__").astype(str).to_numpy()
            blocks.append(
                np.column_stack(
                    [(values == level).astype(float) for level in levels]
                )
            )
        return np.column_stack(blocks).astype(float)


def _clr(values: np.ndarray, pseudocount: float) -> np.ndarray:
    logged = np.log(np.asarray(values, dtype=float) + float(pseudocount))
    return logged - logged.mean(axis=1, keepdims=True)


def _composition(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    total = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(total, 1e-12, None)


def _shannon(values: np.ndarray) -> np.ndarray:
    composition = _composition(values)
    return -np.sum(
        np.where(
            composition > 0.0,
            composition * np.log(np.clip(composition, 1e-15, None)),
            0.0,
        ),
        axis=1,
    )


def _paired_summaries(
    tumor: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    tumor_composition = _composition(tumor)
    normal_composition = _composition(normal)
    midpoint = 0.5 * (tumor_composition + normal_composition)
    tumor_kl = np.sum(
        np.where(
            tumor_composition > 0.0,
            tumor_composition
            * np.log(
                np.clip(tumor_composition, 1e-15, None)
                / np.clip(midpoint, 1e-15, None)
            ),
            0.0,
        ),
        axis=1,
    )
    normal_kl = np.sum(
        np.where(
            normal_composition > 0.0,
            normal_composition
            * np.log(
                np.clip(normal_composition, 1e-15, None)
                / np.clip(midpoint, 1e-15, None)
            ),
            0.0,
        ),
        axis=1,
    )
    tumor_shannon = _shannon(tumor)
    normal_shannon = _shannon(normal)
    tumor_richness = np.mean(tumor > 0.0, axis=1)
    normal_richness = np.mean(normal > 0.0, axis=1)
    bray_curtis = np.sum(np.abs(tumor - normal), axis=1) / np.clip(
        np.sum(tumor + normal, axis=1),
        1e-12,
        None,
    )
    return np.column_stack(
        [
            tumor_shannon,
            normal_shannon,
            tumor_shannon - normal_shannon,
            tumor_richness,
            normal_richness,
            tumor_richness - normal_richness,
            bray_curtis,
            0.5 * (tumor_kl + normal_kl),
        ]
    )


SUMMARY_FEATURE_NAMES = (
    "tumor_shannon",
    "normal_shannon",
    "shannon_delta",
    "tumor_richness_fraction",
    "normal_richness_fraction",
    "richness_delta",
    "paired_bray_curtis",
    "paired_jensen_shannon",
)


@dataclass(frozen=True)
class MicrobiomeTransformer:
    prevalence_mask: np.ndarray
    pseudocount: float
    raw_mean: np.ndarray
    raw_scale: np.ndarray
    selected_indices: np.ndarray
    feature_names: tuple[str, ...]
    prevalence_threshold: float
    top_k: int

    @staticmethod
    def _raw_features(
        tumor: np.ndarray,
        normal: np.ndarray,
        *,
        mask: np.ndarray,
        pseudocount: float,
    ) -> np.ndarray:
        tumor_selected = np.asarray(tumor, dtype=float)[:, mask]
        normal_selected = np.asarray(normal, dtype=float)[:, mask]
        tumor_clr = _clr(tumor_selected, pseudocount)
        normal_clr = _clr(normal_selected, pseudocount)
        mean_clr = 0.5 * (tumor_clr + normal_clr)
        delta_clr = tumor_clr - normal_clr
        summaries = _paired_summaries(tumor_selected, normal_selected)
        return np.column_stack([mean_clr, delta_clr, summaries])

    @classmethod
    def fit(
        cls,
        tumor: np.ndarray,
        normal: np.ndarray,
        genera: Sequence[str],
        time: np.ndarray,
        event: np.ndarray,
        *,
        prevalence_threshold: float,
        top_k: int,
    ) -> "MicrobiomeTransformer":
        tumor_values = np.asarray(tumor, dtype=float)
        normal_values = np.asarray(normal, dtype=float)
        prevalence = np.mean(
            (tumor_values > 0.0) | (normal_values > 0.0),
            axis=0,
        )
        mask = prevalence >= float(prevalence_threshold)
        if int(mask.sum()) < 2:
            raise ValueError("Microbiome prevalence filtering retained <2 genera.")
        positive = np.concatenate(
            [
                tumor_values[:, mask][tumor_values[:, mask] > 0.0],
                normal_values[:, mask][normal_values[:, mask] > 0.0],
            ]
        )
        if positive.size == 0:
            raise ValueError("No positive abundance remained after filtering.")
        pseudocount = 0.5 * float(np.min(positive))
        raw = cls._raw_features(
            tumor_values,
            normal_values,
            mask=mask,
            pseudocount=pseudocount,
        )
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (raw - mean) / scale

        retained_genera = np.asarray(genera, dtype=object)[mask]
        taxon_feature_count = 2 * len(retained_genera)
        zero = np.zeros(taxon_feature_count, dtype=float)
        _, gradient = cox_loss_gradient(
            zero,
            standardized[:, :taxon_feature_count],
            np.asarray(time, dtype=float),
            np.asarray(event, dtype=float),
            0.0,
        )
        selected_taxa = np.argsort(-np.abs(gradient))[
            : min(int(top_k), taxon_feature_count)
        ]
        summary_indices = np.arange(
            taxon_feature_count,
            standardized.shape[1],
            dtype=int,
        )
        selected = np.concatenate([selected_taxa, summary_indices])

        all_names = [
            *(f"paired_mean_clr::{genus}" for genus in retained_genera),
            *(f"tumor_minus_normal_clr::{genus}" for genus in retained_genera),
            *SUMMARY_FEATURE_NAMES,
        ]
        return cls(
            prevalence_mask=np.asarray(mask, dtype=bool),
            pseudocount=float(pseudocount),
            raw_mean=np.asarray(mean, dtype=float),
            raw_scale=np.asarray(scale, dtype=float),
            selected_indices=np.asarray(selected, dtype=int),
            feature_names=tuple(all_names[index] for index in selected),
            prevalence_threshold=float(prevalence_threshold),
            top_k=int(top_k),
        )

    def transform(
        self,
        tumor: np.ndarray,
        normal: np.ndarray,
    ) -> np.ndarray:
        raw = self._raw_features(
            tumor,
            normal,
            mask=self.prevalence_mask,
            pseudocount=self.pseudocount,
        )
        standardized = (raw - self.raw_mean) / self.raw_scale
        return standardized[:, self.selected_indices]


@dataclass(frozen=True)
class FittedClinicalCox:
    transformer: ClinicalTransformer
    model: RidgeCoxModel

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.transformer.transform(frame))


def fit_clinical_cox(
    frame: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    *,
    include_icr: bool,
    include_treatment: bool,
    l2: float,
) -> FittedClinicalCox:
    transformer = ClinicalTransformer.fit(
        frame,
        include_icr=include_icr,
        include_treatment=include_treatment,
    )
    model = fit_ridge_cox(
        transformer.transform(frame),
        time,
        event,
        l2=float(l2),
    )
    return FittedClinicalCox(transformer=transformer, model=model)


@dataclass(frozen=True)
class FittedMicrobiomeCox:
    transformer: MicrobiomeTransformer
    model: RidgeCoxModel

    def predict(
        self,
        tumor: np.ndarray,
        normal: np.ndarray,
    ) -> np.ndarray:
        return self.model.predict(self.transformer.transform(tumor, normal))


def fit_microbiome_cox(
    tumor: np.ndarray,
    normal: np.ndarray,
    genera: Sequence[str],
    time: np.ndarray,
    event: np.ndarray,
    *,
    prevalence_threshold: float,
    top_k: int,
    l2: float,
) -> FittedMicrobiomeCox:
    transformer = MicrobiomeTransformer.fit(
        tumor,
        normal,
        genera,
        time,
        event,
        prevalence_threshold=float(prevalence_threshold),
        top_k=int(top_k),
    )
    model = fit_ridge_cox(
        transformer.transform(tumor, normal),
        time,
        event,
        l2=float(l2),
    )
    return FittedMicrobiomeCox(transformer=transformer, model=model)


@dataclass(frozen=True)
class RiskCalibration:
    mean: float
    scale: float

    @classmethod
    def fit(cls, risk: np.ndarray) -> "RiskCalibration":
        values = np.asarray(risk, dtype=float)
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        return cls(mean=float(np.mean(values)), scale=scale)

    def transform(self, risk: np.ndarray) -> np.ndarray:
        return (
            np.asarray(risk, dtype=float) - float(self.mean)
        ) / float(self.scale)
