from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


PSEUDOCOUNT_PERCENT = 1e-4


class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered log-ratio transform for oral relative-abundance percentages."""

    def __init__(self, pseudocount_percent: float = PSEUDOCOUNT_PERCENT) -> None:
        self.pseudocount_percent = pseudocount_percent

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "CLRTransformer":
        self._validate(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = self._validate(x)
        logged = np.log(values + self.pseudocount_percent)
        output = logged - logged.mean(axis=1, keepdims=True)
        if not np.isfinite(output).all():
            raise ValueError("CLR transformation produced a non-finite value.")
        return output

    @staticmethod
    def _validate(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim != 2:
            raise ValueError("Oral abundance input must be a two-dimensional matrix.")
        if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
            raise ValueError("Oral abundances must be finite percentages in [0, 100].")
        return values
