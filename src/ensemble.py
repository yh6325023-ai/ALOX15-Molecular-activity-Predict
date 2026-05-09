# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np


class _Predictor(Protocol):
    def predict(self, X):  # noqa: ANN001
        ...


@dataclass
class MeanEnsembleRegressor:
    """Averaging ensemble over fold models (supports optional weights)."""

    models: List[_Predictor]
    use_predict_proba: bool = False
    weights: List[float] | None = None

    @staticmethod
    def _sigmoid(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        x = np.clip(x, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, X):  # noqa: ANN001
        preds = []
        for m in self.models:
            if self.use_predict_proba and hasattr(m, "predict_proba"):
                score = m.predict_proba(X)[:, 1]
            elif self.use_predict_proba and hasattr(m, "decision_function"):
                score = self._sigmoid(m.decision_function(X))
            else:
                score = m.predict(X)
            preds.append(np.asarray(score, dtype=float))
        mat = np.vstack(preds)
        if self.weights is None or len(self.weights) != mat.shape[0]:
            return np.mean(mat, axis=0)
        w = np.asarray(self.weights, dtype=float).reshape(-1)
        w = np.where(np.isfinite(w), w, 0.0)
        if np.sum(w) <= 1e-12:
            return np.mean(mat, axis=0)
        w = w / np.sum(w)
        return np.average(mat, axis=0, weights=w)

