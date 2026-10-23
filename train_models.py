"""Train pilot meeting-distance models for Busan Newport.

Inbound:  single HybridDistanceRegressor.
Outbound: route-stratified ensemble — KMeans on bearing-from-breakwater
          clusters observed meeting points; one HybridDistanceRegressor per
          cluster; an RF classifier predicts the cluster from pre-meeting
          features at inference time.

Target: dist_to_breakwater_m (Section 3.6).
Modelling subset: events that survived safety-constrained optimisation.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from src.ml.model import HybridDistanceRegressor
from src.ml.route_ensemble import OutboundRouteEnsemble

ROOT = Path(__file__).parent
ENRICHED_DIR = ROOT / "data" / "processed" / "interactions"
OPTIMISED_DIR = ROOT / "data" / "processed" / "meeting_point"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "training"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-meeting predictors (legitimately known before meeting-point selection).
NUM_BASE = [
    "duration_min",
    "avg_rel_speed_kn",
    "avg_heading_diff_deg",
    "course_aligned_share",
    "speed_similar_share",
    "min_antenna_distance_m",
    "avg_distance_m",
    "vessel_draft_m",
    "wind_speed_mps",
    "wave_height_m",
    "visibility_km",
    "surrounding_vessel_count_0_5nm",
    "surrounding_vessel_count_1_0nm",
    "nearest_third_vessel_hull_cpa_m",
    "nearest_third_vessel_tcpa_min",
    "collision_risk_score",
    "sea_room_score",
]
# UKC at the candidate location is treated as a known operational constraint
# (computable a priori from charts + tide tables; required by Eq. C1 of the
# optimisation framework). The leakage-free baseline (NUM_BASE only) is also
# reported as a methodological audit.
NUM_FEATURES = NUM_BASE + ["ukc_m"]

CATEGORICAL_FEATURES = ["vessel_ship_type_class", "night_day_flag"]
TARGET = "dist_to_breakwater_m"
DIRECTION_COL = "traffic_direction"
EVENT_KEY = "event_id"
RF_PARAMS = {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2,
             "n_jobs": -1, "random_state": 42}
N_OUTBOUND_CLUSTERS = 2


def load_modelling_dataset() -> pd.DataFrame:
    cols = (
        [EVENT_KEY, DIRECTION_COL, TARGET, "meeting_lat", "meeting_lon"]
        + NUM_FEATURES
        + CATEGORICAL_FEATURES
    )
    enriched_files = sorted(glob.glob(str(ENRICHED_DIR / "event_context_enriched_*nopilotcollision.csv")))
    print(f"[load] {len(enriched_files)} enriched weekly files")
    df = pd.concat(
        [pd.read_csv(f, low_memory=False, usecols=lambda c: c in set(cols)) for f in enriched_files],
        ignore_index=True,
    )
    optimised_files = sorted(glob.glob(str(OPTIMISED_DIR / "optimized_meeting_points_*_nopilotcollision.csv")))
    print(f"[load] {len(optimised_files)} optimised weekly files")
    ids = set()
    for f in optimised_files:
        d = pd.read_csv(f, low_memory=False, usecols=[EVENT_KEY])
        ids.update(d[EVENT_KEY].dropna().astype(str))
    df[EVENT_KEY] = df[EVENT_KEY].astype(str)
    df = df[df[EVENT_KEY].isin(ids)].copy()
    df = df.dropna(subset=[TARGET])
    print(f"[join] modelling subset: {len(df):,} ({len(ids):,} optimised event_ids)")
    return df


def _prepare(sub: pd.DataFrame, num_features: list[str]) -> pd.DataFrame:
    sub = sub.copy()
    for c in num_features:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    for c in CATEGORICAL_FEATURES:
        sub[c] = sub[c].fillna("UNK").astype(str)
    med = sub[num_features].median(numeric_only=True).fillna(0.0)
    sub[num_features] = sub[num_features].fillna(med).fillna(0.0)
    return sub


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae_m": float(mean_absolute_error(y_true, y_pred)),
        "rmse_m": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def report(label: str, m: dict) -> None:
    print(f"   [{label}]  RMSE = {m['rmse_m']:.1f} m  MAE = {m['mae_m']:.1f} m  "
          f"MAPE = {m['mape_pct']:.2f}%  R² = {m['r2']:.3f}  (n={m['n']:,})")


def cv_inbound(X: pd.DataFrame, y: np.ndarray, num_features: list[str], n_splits: int = 5) -> dict:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for tr, va in kf.split(X):
        m = HybridDistanceRegressor(
            numeric_features=num_features,
            categorical_features=CATEGORICAL_FEATURES,
            rf_params=RF_PARAMS,
            ridge_alpha=1.0,
        )
        m.fit(X.iloc[tr], y[tr])
        rows.append(evaluate(y[va], m.predict(X.iloc[va])))
    df = pd.DataFrame(rows)
    return {
        "rmse_mean_m": float(df["rmse_m"].mean()), "rmse_sd_m": float(df["rmse_m"].std()),
        "mae_mean_m": float(df["mae_m"].mean()),   "mae_sd_m": float(df["mae_m"].std()),
        "mape_mean_pct": float(df["mape_pct"].mean()), "mape_sd_pct": float(df["mape_pct"].std()),
        "r2_mean": float(df["r2"].mean()),         "r2_sd": float(df["r2"].std()),
    }


def cv_outbound_ensemble(
    X: pd.DataFrame, y: np.ndarray, meeting_lat: pd.Series, meeting_lon: pd.Series,
    num_features: list[str], n_splits: int = 5,
) -> dict:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows_real, rows_oracle = [], []
    for tr, va in kf.split(X):
        ens = OutboundRouteEnsemble(
            n_clusters=N_OUTBOUND_CLUSTERS,
            numeric_features=num_features,
            categorical_features=CATEGORICAL_FEATURES,
            rf_params=RF_PARAMS,
            ridge_alpha=1.0,
        )
        ens.fit(X.iloc[tr], y[tr], meeting_lat.iloc[tr], meeting_lon.iloc[tr])
        rows_real.append(evaluate(y[va], ens.predict(X.iloc[va])))
        true_va = ens.assign_clusters(meeting_lat.iloc[va], meeting_lon.iloc[va])
        rows_oracle.append(evaluate(y[va], ens.predict_oracle(X.iloc[va], true_va)))
    def agg(rows: list[dict]) -> dict:
        d = pd.DataFrame(rows)
        return {
            "rmse_mean_m": float(d["rmse_m"].mean()), "rmse_sd_m": float(d["rmse_m"].std()),
            "mae_mean_m": float(d["mae_m"].mean()),   "mae_sd_m": float(d["mae_m"].std()),
            "mape_mean_pct": float(d["mape_pct"].mean()), "mape_sd_pct": float(d["mape_pct"].std()),
            "r2_mean": float(d["r2"].mean()),         "r2_sd": float(d["r2"].std()),
        }
    return {"realistic": agg(rows_real), "oracle_route": agg(rows_oracle)}


def train_inbound(df: pd.DataFrame) -> dict:
    sub = df[df[DIRECTION_COL].str.lower() == "inbound"].copy()
    print(f"\n=== INBOUND (single hybrid model) ===")
    print(f"[data] n = {len(sub):,}, target mean = {sub[TARGET].mean():.0f} m, "
          f"sd = {sub[TARGET].std():.0f} m")

    # Leakage-free baseline (transparency block in paper)
    sub_base = _prepare(sub, NUM_BASE)
    X_b = sub_base[NUM_BASE + CATEGORICAL_FEATURES]
    y_b = sub_base[TARGET].astype(float).to_numpy()
    X_btr, X_bte, y_btr, y_bte = train_test_split(X_b, y_b, test_size=0.2, random_state=42)
    cv_baseline = cv_inbound(X_btr, y_btr, NUM_BASE)
    base = HybridDistanceRegressor(
        numeric_features=NUM_BASE, categorical_features=CATEGORICAL_FEATURES,
        rf_params=RF_PARAMS, ridge_alpha=1.0,
    )
    base.fit(X_btr, y_btr)
    test_baseline = evaluate(y_bte, base.predict(X_bte))

    # Full model (NUM_BASE + ukc_m)
    sub_full = _prepare(sub, NUM_FEATURES)
    X = sub_full[NUM_FEATURES + CATEGORICAL_FEATURES]
    y = sub_full[TARGET].astype(float).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = cv_inbound(X_tr, y_tr, NUM_FEATURES)
    model = HybridDistanceRegressor(
        numeric_features=NUM_FEATURES, categorical_features=CATEGORICAL_FEATURES,
        rf_params=RF_PARAMS, ridge_alpha=1.0,
    )
    model.fit(X_tr, y_tr)
    test = evaluate(y_te, model.predict(X_te))

    print("[5-fold CV] (mean ± sd)")
    print(f"   leakage-free baseline:  RMSE = {cv_baseline['rmse_mean_m']:.0f} ± {cv_baseline['rmse_sd_m']:.0f} m   "
          f"R² = {cv_baseline['r2_mean']:.3f} ± {cv_baseline['r2_sd']:.3f}")
    print(f"   + ukc_m:                RMSE = {cv['rmse_mean_m']:.0f} ± {cv['rmse_sd_m']:.0f} m   "
          f"R² = {cv['r2_mean']:.3f} ± {cv['r2_sd']:.3f}")
    print("[hold-out test]")
    report("baseline", test_baseline)
    report("+ ukc_m", test)

    out = MODELS_DIR / "inbound_distance_model.joblib"
    joblib.dump(model, out)
    print(f"[save] {out}")

    return {
        "n_total": int(len(sub)),
        "baseline": {"cv": cv_baseline, "test": test_baseline},
        "full": {"cv": cv, "test": test},
    }


def train_outbound(df: pd.DataFrame) -> dict:
    sub = df[df[DIRECTION_COL].str.lower() == "outbound"].copy()
    sub = sub.dropna(subset=["meeting_lat", "meeting_lon"])
    print(f"\n=== OUTBOUND (route-stratified ensemble, k = {N_OUTBOUND_CLUSTERS}) ===")
    print(f"[data] n = {len(sub):,}, target mean = {sub[TARGET].mean():.0f} m, "
          f"sd = {sub[TARGET].std():.0f} m")

    # Leakage-free baseline (single model, no stratification, no ukc)
    sub_base = _prepare(sub, NUM_BASE)
    X_b = sub_base[NUM_BASE + CATEGORICAL_FEATURES]
    y_b = sub_base[TARGET].astype(float).to_numpy()
    X_btr, X_bte, y_btr, y_bte = train_test_split(X_b, y_b, test_size=0.2, random_state=42)
    cv_baseline = cv_inbound(X_btr, y_btr, NUM_BASE)
    base = HybridDistanceRegressor(
        numeric_features=NUM_BASE, categorical_features=CATEGORICAL_FEATURES,
        rf_params=RF_PARAMS, ridge_alpha=1.0,
    )
    base.fit(X_btr, y_btr)
    test_baseline = evaluate(y_bte, base.predict(X_bte))

    # Route-stratified ensemble (NUM_BASE + ukc_m)
    sub_full = _prepare(sub, NUM_FEATURES)
    X = sub_full[NUM_FEATURES + CATEGORICAL_FEATURES]
    y = sub_full[TARGET].astype(float).to_numpy()
    mlat = sub_full["meeting_lat"]
    mlon = sub_full["meeting_lon"]
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)

    cv = cv_outbound_ensemble(X.iloc[idx_tr], y[idx_tr], mlat.iloc[idx_tr], mlon.iloc[idx_tr], NUM_FEATURES)

    ens = OutboundRouteEnsemble(
        n_clusters=N_OUTBOUND_CLUSTERS,
        numeric_features=NUM_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        rf_params=RF_PARAMS,
        ridge_alpha=1.0,
    )
    ens.fit(X.iloc[idx_tr], y[idx_tr], mlat.iloc[idx_tr], mlon.iloc[idx_tr])

    # Per-cluster training sizes
    train_clusters = ens.assign_clusters(mlat.iloc[idx_tr], mlon.iloc[idx_tr])
    cluster_sizes = {int(c): int((train_clusters == c).sum()) for c in range(N_OUTBOUND_CLUSTERS)}
    print(f"[clusters] centers (deg) = {ens.cluster_centers_deg_}")
    print(f"[clusters] train sizes   = {cluster_sizes}")
    print(f"[classifier] training accuracy = {ens.classifier_train_acc_:.3f}")

    test_real = evaluate(y[idx_te], ens.predict(X.iloc[idx_te]))
    true_te = ens.assign_clusters(mlat.iloc[idx_te], mlon.iloc[idx_te])
    test_oracle = evaluate(y[idx_te], ens.predict_oracle(X.iloc[idx_te], true_te))

    # Per-cluster regression metrics on test set, using true cluster
    per_cluster = {}
    for cid, reg in ens.regressors.items():
        mask = true_te == cid
        if mask.sum() < 5: continue
        per_cluster[int(cid)] = evaluate(y[idx_te][mask], reg.predict(X.iloc[idx_te][mask]))

    print("[5-fold CV]")
    print(f"   leakage-free baseline (single model):  R² = {cv_baseline['r2_mean']:.3f} ± {cv_baseline['r2_sd']:.3f}   "
          f"MAE = {cv_baseline['mae_mean_m']:.0f} m")
    print(f"   route-stratified, oracle route:        R² = {cv['oracle_route']['r2_mean']:.3f} ± {cv['oracle_route']['r2_sd']:.3f}   "
          f"MAE = {cv['oracle_route']['mae_mean_m']:.0f} m")
    print(f"   route-stratified, classifier-routed:   R² = {cv['realistic']['r2_mean']:.3f} ± {cv['realistic']['r2_sd']:.3f}   "
          f"MAE = {cv['realistic']['mae_mean_m']:.0f} m")
    print("[hold-out test]")
    report("baseline (no stratification, no UKC)", test_baseline)
    report("ensemble + classifier route", test_real)
    report("ensemble + oracle route", test_oracle)
    for cid, m in per_cluster.items():
        report(f"cluster {cid} regressor only", m)

    out = MODELS_DIR / "outbound_distance_model.joblib"
    joblib.dump(ens, out)
    print(f"[save] {out}")

    return {
        "n_total": int(len(sub)),
        "n_clusters": N_OUTBOUND_CLUSTERS,
        "cluster_centers_deg": ens.cluster_centers_deg_.tolist(),
        "cluster_train_sizes": cluster_sizes,
        "classifier_train_accuracy": ens.classifier_train_acc_,
        "baseline": {"cv": cv_baseline, "test": test_baseline},
        "full": {
            "cv_realistic": cv["realistic"],
            "cv_oracle_route": cv["oracle_route"],
            "test_realistic": test_real,
            "test_oracle_route": test_oracle,
            "per_cluster_test": per_cluster,
        },
    }


def main() -> None:
    df = load_modelling_dataset()
    summary = {
        "inbound": train_inbound(df),
        "outbound": train_outbound(df),
        "feature_set": {
            "leakage_free_baseline": NUM_BASE + CATEGORICAL_FEATURES,
            "full": NUM_FEATURES + CATEGORICAL_FEATURES,
            "ukc_inclusion_rationale": (
                "UKC at the candidate meeting location is included because it represents a "
                "hard physical constraint that pilots verify against bathymetric charts and "
                "tide tables a priori; the optimisation framework's safety constraint "
                "UKC(s) >= UKC_min explicitly requires UKC evaluation at candidate locations. "
                "We acknowledge that training-time UKC is observed at the actual meeting "
                "point, and we therefore report the leakage-free baseline (NUM_BASE only) "
                "alongside the full model so that the magnitude of the UKC contribution is "
                "transparent."
            ),
        },
    }
    with open(RESULTS_DIR / "training_metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[done] metrics -> {RESULTS_DIR/'training_metrics.json'}")


if __name__ == "__main__":
    main()
