from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class HybridDistanceRegressor(BaseEstimator, RegressorMixin):
    """Random-Forest + Ridge residual-correction hybrid (Eqs. 8-11 in the paper)."""

    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    rf_params: dict[str, Any] = field(default_factory=dict)
    ridge_alpha: float = 1.0
    random_state: int = 42

    preprocessor: ColumnTransformer | None = field(default=None, init=False, repr=False)
    rf_model: RandomForestRegressor | None = field(default=None, init=False, repr=False)
    linear_model: Pipeline | None = field(default=None, init=False, repr=False)
    feature_names_in_: list[str] | None = field(default=None, init=False, repr=False)

    def _build_preprocessor(self) -> ColumnTransformer:
        transformers = []
        if self.numeric_features:
            transformers.append(("num", "passthrough", self.numeric_features))
        if self.categorical_features:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                self.categorical_features,
            ))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "HybridDistanceRegressor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HybridDistanceRegressor requires a pandas DataFrame as input.")
        self.feature_names_in_ = list(X.columns)

        self.preprocessor = self._build_preprocessor()
        X_enc = self.preprocessor.fit_transform(X)

        rf_kwargs = {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "random_state": self.random_state,
        }
        rf_kwargs.update(self.rf_params or {})
        self.rf_model = RandomForestRegressor(**rf_kwargs)
        self.rf_model.fit(X_enc, np.asarray(y))

        residuals = np.asarray(y) - self.rf_model.predict(X_enc)
        self.linear_model = Pipeline([
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=self.ridge_alpha, random_state=self.random_state)),
        ])
        self.linear_model.fit(X_enc, residuals)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.rf_model is None or self.linear_model is None:
            raise RuntimeError("HybridDistanceRegressor is not fitted.")
        X_enc = self.preprocessor.transform(X)
        return self.rf_model.predict(X_enc) + self.linear_model.predict(X_enc)

    def predict_components(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_enc = self.preprocessor.transform(X)
        rf_pred = self.rf_model.predict(X_enc)
        ridge_pred = self.linear_model.predict(X_enc)
        return rf_pred, ridge_pred
