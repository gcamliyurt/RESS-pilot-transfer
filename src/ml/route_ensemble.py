"""Route-stratified outbound ensemble.

Outbound meeting points cluster into a small number of bearing-based departure
routes from the breakwater. We fit a KMeans clusterer on observed bearings,
train one HybridDistanceRegressor per cluster, and at inference predict the
route from pre-meeting features via a Random Forest classifier so the pipeline
is end-to-end usable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from .model import HybridDistanceRegressor


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


@dataclass
class OutboundRouteEnsemble:
    """KMeans on observed meeting bearings → per-cluster HybridDistanceRegressor.
    A separate RF classifier maps pre-meeting features → cluster id."""

    n_clusters: int = 2
    breakwater_lat: float = 35.0700
    breakwater_lon: float = 128.81
    rf_params: dict[str, Any] = field(default_factory=dict)
    ridge_alpha: float = 1.0
    random_state: int = 42

    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)

    clusterer: KMeans | None = field(default=None, init=False, repr=False)
    classifier: RandomForestClassifier | None = field(default=None, init=False, repr=False)
    regressors: dict[int, HybridDistanceRegressor] = field(default_factory=dict, init=False, repr=False)
    cluster_centers_deg_: np.ndarray | None = field(default=None, init=False, repr=False)
    classifier_train_acc_: float | None = field(default=None, init=False, repr=False)

    def _bearings(self, meeting_lat: pd.Series, meeting_lon: pd.Series) -> np.ndarray:
        return np.array([
            bearing_deg(self.breakwater_lat, self.breakwater_lon, la, lo)
            for la, lo in zip(meeting_lat, meeting_lon)
        ])

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        meeting_lat: pd.Series,
        meeting_lon: pd.Series,
    ) -> "OutboundRouteEnsemble":
        bearings = self._bearings(meeting_lat, meeting_lon)

        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_ids = self.clusterer.fit_predict(bearings.reshape(-1, 1))
        self.cluster_centers_deg_ = np.sort(self.clusterer.cluster_centers_.flatten())

        self.classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=self.random_state,
        )
        # The classifier is trained on the same encoded feature representation we use for regression.
        # We piggy-back on a fresh HybridDistanceRegressor's preprocessor to get encodings.
        proxy = HybridDistanceRegressor(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            rf_params=self.rf_params,
            ridge_alpha=self.ridge_alpha,
            random_state=self.random_state,
        )
        proxy.preprocessor = proxy._build_preprocessor()
        X_enc = proxy.preprocessor.fit_transform(X)
        self._classifier_preprocessor = proxy.preprocessor
        self.classifier.fit(X_enc, cluster_ids)
        self.classifier_train_acc_ = float(self.classifier.score(X_enc, cluster_ids))

        # Train one regressor per cluster
        self.regressors = {}
        y_arr = np.asarray(y)
        for cid in range(self.n_clusters):
            mask = cluster_ids == cid
            if mask.sum() < 30:
                continue
            reg = HybridDistanceRegressor(
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
                rf_params=self.rf_params,
                ridge_alpha=self.ridge_alpha,
                random_state=self.random_state,
            )
            reg.fit(X.iloc[mask], y_arr[mask])
            self.regressors[cid] = reg
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.classifier is None or not self.regressors:
            raise RuntimeError("OutboundRouteEnsemble is not fitted.")
        X_enc = self._classifier_preprocessor.transform(X)
        cluster_ids = self.classifier.predict(X_enc)
        out = np.full(len(X), np.nan, dtype=float)
        for cid, reg in self.regressors.items():
            mask = cluster_ids == cid
            if mask.any():
                out[mask] = reg.predict(X.iloc[mask])
        # Fallback: any rows whose predicted cluster has no regressor (shouldn't happen) → use first regressor
        if np.isnan(out).any():
            fallback = next(iter(self.regressors.values()))
            out[np.isnan(out)] = fallback.predict(X.iloc[np.isnan(out)])
        return out

    def predict_route_bearing(self, X: pd.DataFrame) -> np.ndarray:
        """Centre bearing (deg from the breakwater) of the route the classifier predicts."""
        cluster_ids = self.classifier.predict(self._classifier_preprocessor.transform(X))
        return self.clusterer.cluster_centers_.flatten()[cluster_ids]

    def predict_oracle(self, X: pd.DataFrame, true_cluster_ids: np.ndarray) -> np.ndarray:
        """Predict using the TRUE cluster assignments (computed from observed bearings).

        Reports the upper bound on ensemble performance: it isolates the
        per-cluster regression quality from the route-classifier accuracy.
        """
        out = np.full(len(X), np.nan, dtype=float)
        for cid, reg in self.regressors.items():
            mask = true_cluster_ids == cid
            if mask.any():
                out[mask] = reg.predict(X.iloc[mask])
        if np.isnan(out).any():
            fallback = next(iter(self.regressors.values()))
            out[np.isnan(out)] = fallback.predict(X.iloc[np.isnan(out)])
        return out

    def assign_clusters(
        self, meeting_lat: pd.Series, meeting_lon: pd.Series
    ) -> np.ndarray:
        bearings = self._bearings(meeting_lat, meeting_lon)
        return self.clusterer.predict(bearings.reshape(-1, 1))
