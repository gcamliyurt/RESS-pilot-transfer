"""RESS R2 revision: leakage-controlled validation, baselines and k-sweep.

Answers Reviewer #5 C3 (grouped / temporal validation), R5 C4 (baselines)
and Reviewer #4 C2 (k in {2,3,4} sweep). Every preprocessing step
(median imputation, encoders, KMeans route clustering, route classifier)
is fitted inside each training fold only.

Schemes
  random   : 5-fold KFold on events (reproduces the R1 setting)
  grouped  : 5-fold GroupKFold by vessel_mmsi (no ship in both train and test)
  temporal : train on meetings before TEMPORAL_CUTOFF, test on/after it

Output: results/revision_r2/validation_metrics.{json,csv}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline

import os

import train_models as tm

# FEATURE_SET=r1      : predictor set of the submitted manuscript (includes quantities
#                       measured during the transfer and UKC at the meeting point)
# FEATURE_SET=apriori : decision-time predictors only (build_r2_apriori_features.py)
# FEATURE_SET=adaptive: same predictors at the adaptive decision time t_dec, plus lead_min
#                       (DECISION_TIME=adaptive build_r2_apriori_features.py); also writes the
#                       grouped-CV out-of-fold predictions the optimiser anchors on
# FEATURE_SET=adaptive_ukc: adaptive + decision-time UKC features (build_r2_ukc_features.py)
# FEATURE_SET=causal  : inputs at the online trigger (first fix within 13 km of the breakwater;
#                       build_r2_causal_features.py), no lead time (the transfer time is unknown online)
FEATURE_SET = os.environ.get("FEATURE_SET", "r1")
APRIORI_NUM = ["ref_dist_to_breakwater_m", "ref_bearing_from_breakwater_sin", "ref_bearing_from_breakwater_cos",
               "ref_sog_kn", "ref_cog_sin", "ref_cog_cos", "traffic_0_5nm", "traffic_1nm", "traffic_2nm",
               "nearest_third_m", "vessel_draft_m", "wind_speed_mps", "wave_height_m", "hour_kst", "month"]
# FEATURE_SET=dispatch: inputs at pilot-boat dispatch (build_r2_dispatch_features.py)
DISPATCH_NUM = ["ship_visible", "ref_dist_to_breakwater_m", "ref_bearing_from_breakwater_sin", "ref_bearing_from_breakwater_cos",
                "ref_sog_kn", "ref_cog_sin", "ref_cog_cos", "pilot_dist_to_breakwater_m", "pilot_bearing_sin",
                "pilot_bearing_cos", "pilot_sog_kn", "pilot_cog_sin", "pilot_cog_cos", "pilot_ship_range_m",
                "pilot_to_ship_bearing_sin", "pilot_to_ship_bearing_cos", "intercept_dist_to_breakwater_m",
                "intercept_time_min", "traffic_0_5nm", "traffic_1nm", "traffic_2nm", "nearest_third_m",
                "vessel_draft_m", "wind_speed_mps", "wave_height_m", "hour_kst", "month"]
UKC_NUM = ["ukc_ref_m", "ukc2_boundary_m", "ukc4_boundary_m", "tide_dec_m"]
if FEATURE_SET in ("dispatch", "dispatch_ukc", "dispatch_prevwin"):
    tm.NUM_FEATURES = DISPATCH_NUM + (UKC_NUM if FEATURE_SET == "dispatch_ukc" else [])
    tm.CATEGORICAL_FEATURES = ["vessel_ship_type_class"]
elif FEATURE_SET in ("apriori", "adaptive", "adaptive_ukc", "causal"):
    tm.NUM_FEATURES = (APRIORI_NUM + (["lead_min"] if FEATURE_SET in ("adaptive", "adaptive_ukc") else [])
                       + (UKC_NUM if FEATURE_SET == "adaptive_ukc" else []))
    tm.CATEGORICAL_FEATURES = ["vessel_ship_type_class"]
from src.ml.model import HybridDistanceRegressor
from src.ml.route_ensemble import OutboundRouteEnsemble

OUT_DIR = tm.ROOT / "results" / f"revision_r2_logconfirmed_{FEATURE_SET}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_CUTOFF = pd.Timestamp("2024-07-01", tz="UTC")
N_BOOT = 1000
RNG = np.random.default_rng(42)
FEATS = tm.NUM_FEATURES + tm.CATEGORICAL_FEATURES
# Verified Busan pilot-boat fleet (13 MMSIs). BusanPB.xlsx `pilotBoat` also listed
# 44 harbour tugs; see 6.3 Queueing/corrections/CORRECTION_MEMO_6.0_meetingPoint.md.
VERIFIED_PILOTS = {440008040, 440105210, 440110850, 440111140, 440111150, 440111160, 440111190,
                   440111210, 440111230, 440121630, 440127410, 440135830, 440155330}
VERIFIED_ONLY = True


R2_DATASET = tm.ROOT / "data" / "processed" / "r2" / "modelling_r2.csv"


def sanitise_intercept(df):
    """Intercept features are undefined when boat and ship speeds nearly match (division by ~0):
    infinite values, intercept times over 3 h and intercept points over 50 km away become missing
    (imputed within the training folds)."""
    df = df.replace([np.inf, -np.inf], np.nan)
    for c, hi in (("intercept_time_min", 180.0), ("intercept_dist_to_breakwater_m", 50000.0), ("intercept_dist_to_base_m", 50000.0)):
        if c in df:
            df.loc[(df[c] > hi) | (df[c] < 0), c] = np.nan
    return df


def load() -> pd.DataFrame:
    if R2_DATASET.exists():
        # Log-confirmed transfers from the rebuilt pipeline (build_r2_dataset.py);
        # all confirmed events are modelled, not only those the optimiser kept.
        df = pd.read_csv(R2_DATASET, low_memory=False)
        if FEATURE_SET in ("apriori", "adaptive", "adaptive_ukc", "causal", "dispatch", "dispatch_ukc", "dispatch_prevwin"):
            ap = pd.read_csv(R2_DATASET.parent / {"apriori": "apriori_features_r2.csv",
                                                   "causal": "causal_features_r2.csv",
                                                   "dispatch": "dispatch_features_r2.csv",
                                                   "dispatch_ukc": "dispatch_features_r2.csv",
                                                   "dispatch_prevwin": "dispatch_features_r2_prevwin.csv"}.get(FEATURE_SET, "apriori_features_r2_adaptive.csv"))
            if FEATURE_SET == "adaptive_ukc":
                ap = ap.merge(pd.read_csv(R2_DATASET.parent / "ukc_features_r2_adaptive.csv"), on="event_id")
            if FEATURE_SET == "dispatch_ukc":
                ap = ap.merge(pd.read_csv(R2_DATASET.parent / "ukc_features_r2_dispatch.csv"), on="event_id", how="left")
            df = df.drop(columns=[c for c in ap.columns if c != "event_id" and c in df.columns]).merge(ap, on="event_id")
        df = sanitise_intercept(df).dropna(subset=[tm.TARGET]).copy()
        df["meeting_time"] = pd.to_datetime(df["meeting_time"].astype(str).str[:19]).dt.tz_localize("UTC")
        for c in tm.NUM_FEATURES:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in tm.CATEGORICAL_FEATURES:
            df[c] = df[c].fillna("UNK").astype(str)
        return df.reset_index(drop=True)
    # Same modelling subset as train_models.py, plus grouping / time keys.
    extra = ["vessel_mmsi", "meeting_time", "pilot_mmsi"]
    cols = set([tm.EVENT_KEY, tm.DIRECTION_COL, tm.TARGET, "meeting_lat", "meeting_lon"]
               + tm.NUM_FEATURES + tm.CATEGORICAL_FEATURES + extra)
    files = sorted(tm.ENRICHED_DIR.glob("event_context_enriched_*nopilotcollision.csv"))
    df = pd.concat([pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols) for f in files],
                   ignore_index=True)
    ids = set()
    for f in sorted(tm.OPTIMISED_DIR.glob("optimized_meeting_points_*_nopilotcollision.csv")):
        ids.update(pd.read_csv(f, low_memory=False, usecols=[tm.EVENT_KEY])[tm.EVENT_KEY].dropna().astype(str))
    df[tm.EVENT_KEY] = df[tm.EVENT_KEY].astype(str)
    df = df[df[tm.EVENT_KEY].isin(ids)].dropna(subset=[tm.TARGET]).copy()
    if VERIFIED_ONLY:
        df = df[pd.to_numeric(df["pilot_mmsi"], errors="coerce").isin(VERIFIED_PILOTS)].copy()
    df["meeting_time"] = pd.to_datetime(df["meeting_time"], errors="coerce", utc=True)
    for c in tm.NUM_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in tm.CATEGORICAL_FEATURES:
        df[c] = df[c].fillna("UNK").astype(str)
    return df.reset_index(drop=True)


def impute(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = tr[tm.NUM_FEATURES].median().fillna(0.0)
    tr, te = tr.copy(), te.copy()
    tr[tm.NUM_FEATURES] = tr[tm.NUM_FEATURES].fillna(med)
    te[tm.NUM_FEATURES] = te[tm.NUM_FEATURES].fillna(med)
    return tr, te


def splits(df: pd.DataFrame, scheme: str):
    idx = np.arange(len(df))
    if scheme == "random":
        yield from KFold(5, shuffle=True, random_state=42).split(idx)
    elif scheme == "grouped":
        yield from GroupKFold(5).split(idx, groups=df["vessel_mmsi"].astype(str))
    elif scheme == "temporal":
        t = df["meeting_time"]
        yield idx[t < TEMPORAL_CUTOFF], idx[t >= TEMPORAL_CUTOFF]


def _sk_pipe(est):
    proxy = HybridDistanceRegressor(numeric_features=tm.NUM_FEATURES,
                                    categorical_features=tm.CATEGORICAL_FEATURES)
    return Pipeline([("pre", proxy._build_preprocessor()), ("est", est)])


def fit_predict(name: str, tr: pd.DataFrame, te: pd.DataFrame, direction: str, k: int = 2) -> np.ndarray:
    ytr = tr[tm.TARGET].to_numpy(float)
    if name == "historical_mean":
        return np.full(len(te), ytr.mean())
    if name == "historical_mean_by_route" and direction == "outbound":
        ens = OutboundRouteEnsemble(n_clusters=k)
        b_tr = ens._bearings(tr.meeting_lat, tr.meeting_lon)
        from sklearn.cluster import KMeans
        km = KMeans(k, random_state=42, n_init=10).fit(b_tr.reshape(-1, 1))
        means = pd.Series(ytr).groupby(km.labels_).mean()
        lab_te = km.predict(ens._bearings(te.meeting_lat, te.meeting_lon).reshape(-1, 1))
        return means.reindex(lab_te).fillna(ytr.mean()).to_numpy()  # oracle-route lookup
    if name == "linear":
        return _sk_pipe(LinearRegression()).fit(tr[FEATS], ytr).predict(te[FEATS])
    if name == "random_forest":
        return _sk_pipe(RandomForestRegressor(**tm.RF_PARAMS)).fit(tr[FEATS], ytr).predict(te[FEATS])
    if name == "xgboost":
        # Fixed, untuned settings like the other models. On macOS the OpenMP runtime can be
        # taken from scikit-learn: DYLD_FALLBACK_LIBRARY_PATH=<sklearn>/.dylibs
        from xgboost import XGBRegressor
        est = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                           colsample_bytree=0.8, random_state=42, n_jobs=-1)
        return _sk_pipe(est).fit(tr[FEATS], ytr).predict(te[FEATS])
    if name == "grad_boost":
        return _sk_pipe(HistGradientBoostingRegressor(random_state=42)).fit(tr[FEATS], ytr).predict(te[FEATS])
    # Proposed model = RF-Ridge hybrid in both directions. On the verified data the outbound
    # bearings are unimodal (k = 2 splits 816 / 13), so the route ensemble is reported only
    # as an alternative ("route_ensemble", "route_ensemble_oracle").
    if name in ("hybrid", "proposed"):
        m = HybridDistanceRegressor(numeric_features=tm.NUM_FEATURES, categorical_features=tm.CATEGORICAL_FEATURES,
                                    rf_params=tm.RF_PARAMS, ridge_alpha=1.0)
        return m.fit(tr[FEATS], ytr).predict(te[FEATS])
    if name in ("route_ensemble", "route_ensemble_oracle") and direction == "outbound":
        ens = OutboundRouteEnsemble(n_clusters=k, numeric_features=tm.NUM_FEATURES,
                                    categorical_features=tm.CATEGORICAL_FEATURES,
                                    rf_params=tm.RF_PARAMS, ridge_alpha=1.0)
        ens.fit(tr[FEATS], ytr, tr.meeting_lat, tr.meeting_lon)
        if name == "route_ensemble":
            return ens.predict(te[FEATS])
        return ens.predict_oracle(te[FEATS], ens.assign_clusters(te.meeting_lat, te.meeting_lon))
    raise ValueError(name)


def metrics(y, p, groups=None) -> dict:
    m = tm.evaluate(np.asarray(y), np.asarray(p))
    # 95% bootstrap CI on pooled out-of-fold predictions, resampling whole ships
    # (repeated calls of one ship are not independent); conditional on the fitted folds.
    n = len(y); y, p = np.asarray(y), np.asarray(p)
    g = np.asarray(groups) if groups is not None else np.arange(n)
    ug, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    bs = {"mae_m": [], "r2": []}
    for _ in range(N_BOOT):
        i = np.concatenate([members[k] for k in RNG.integers(0, len(ug), len(ug))])
        e = tm.evaluate(y[i], p[i]); bs["mae_m"].append(e["mae_m"]); bs["r2"].append(e["r2"])
    for key, v in bs.items():
        m[f"{key}_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return m


def run_direction(df: pd.DataFrame, direction: str) -> list[dict]:
    sub = df[df[tm.DIRECTION_COL].str.lower() == direction].reset_index(drop=True)
    if direction == "outbound":
        sub = sub.dropna(subset=["meeting_lat", "meeting_lon"]).reset_index(drop=True)
    models = ["historical_mean", "linear", "random_forest", "grad_boost", "xgboost", "hybrid", "proposed"]
    if direction == "outbound":
        models += ["historical_mean_by_route", "route_ensemble", "route_ensemble_oracle"]
    rows = []
    for scheme in ["random", "grouped", "temporal"]:
        folds = list(splits(sub, scheme))
        for name in models:
            t0 = time.perf_counter()
            ys, ps, fold_r2, sizes, gs = [], [], [], [], []
            for tr_i, te_i in folds:
                tr, te = impute(sub.iloc[tr_i], sub.iloc[te_i])
                p = fit_predict(name, tr, te, direction)
                y = te[tm.TARGET].to_numpy(float)
                ys.append(y); ps.append(p); fold_r2.append(tm.evaluate(y, p)["r2"])
                gs.append(te["vessel_mmsi"].astype(str).to_numpy())
                sizes.append((len(tr_i), len(te_i)))
            m = metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(gs))
            m.update(n_train=int(np.mean([a for a, _ in sizes])), n_test_total=int(sum(b for _, b in sizes)), direction=direction, scheme=scheme, model=name,
                     n_train_test=sizes, r2_fold_sd=float(np.std(fold_r2)),
                     n_vessels=int(sub["vessel_mmsi"].nunique()),
                     wall_s=round(time.perf_counter() - t0, 1))
            print(f"[{direction:8s}|{scheme:8s}] {name:24s} R²={m['r2']:.3f} "
                  f"[{m['r2_ci95'][0]:.3f},{m['r2_ci95'][1]:.3f}] MAE={m['mae_m']:.0f} m  ({m['wall_s']}s)", flush=True)
            rows.append(m)
    return rows


def _gate_bearing(lat, lon) -> np.ndarray:
    """Bearing (deg) from the New Port breakwater gate (mean of the breakwater lights)."""
    from src.core.enrich_event_context import BREAKWATER_LIGHTS
    k = np.cos(np.radians(35.05))
    x = (np.asarray(lon, float) - BREAKWATER_LIGHTS["lon"].mean()) * k
    y = np.asarray(lat, float) - BREAKWATER_LIGHTS["lat"].mean()
    return np.degrees(np.arctan2(x, y)) % 360


def _circ_mean(deg) -> float:
    r = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def oof_predictions(df: pd.DataFrame, k_nn: int = 25) -> pd.DataFrame:
    """Grouped-CV out-of-fold predictions of the proposed model (no ship in its own
    training fold): predicted meeting distance and the anchor bearing from the gate,
    the circular mean of the meeting bearings of the k_nn training ships whose
    decision-time bearing from the gate is closest (training fold only)."""
    parts = []
    for direction in ("inbound", "outbound"):
        sub = df[df[tm.DIRECTION_COL].str.lower() == direction].dropna(subset=["meeting_lat", "meeting_lon"]).reset_index(drop=True)
        sub["meet_brg"] = _gate_bearing(sub.meeting_lat, sub.meeting_lon)
        sub["ship_brg"] = np.degrees(np.arctan2(sub.ref_bearing_from_breakwater_sin, sub.ref_bearing_from_breakwater_cos)) % 360
        for fold, (tr_i, te_i) in enumerate(splits(sub, "grouped")):
            tr, te = impute(sub.iloc[tr_i], sub.iloc[te_i])
            diff = np.abs(((te.ship_brg.to_numpy()[:, None] - tr.ship_brg.to_numpy()[None, :]) + 180) % 360 - 180)
            nn = np.argsort(diff, axis=1)[:, :k_nn]
            parts.append(te[[tm.EVENT_KEY]].assign(
                traffic_direction=direction, fold=fold,
                pred_dist_m=fit_predict("proposed", tr, te, direction),
                anchor_bearing_deg=[_circ_mean(tr.meet_brg.to_numpy()[r]) for r in nn]))
    return pd.concat(parts, ignore_index=True)


def k_sweep(df: pd.DataFrame) -> list[dict]:
    """Reviewer #4 C2: k in {2,3,4}, silhouette + held-out ensemble R² (grouped CV)."""
    sub = df[df[tm.DIRECTION_COL].str.lower() == "outbound"].dropna(subset=["meeting_lat", "meeting_lon"]).reset_index(drop=True)
    rows = []
    for k in (2, 3, 4):
        sil, ys, pr, po = [], [], [], []
        for tr_i, te_i in splits(sub, "grouped"):
            tr, te = impute(sub.iloc[tr_i], sub.iloc[te_i])
            ens = OutboundRouteEnsemble(n_clusters=k, numeric_features=tm.NUM_FEATURES,
                                        categorical_features=tm.CATEGORICAL_FEATURES,
                                        rf_params=tm.RF_PARAMS, ridge_alpha=1.0)
            ytr = tr[tm.TARGET].to_numpy(float)
            ens.fit(tr[FEATS], ytr, tr.meeting_lat, tr.meeting_lon)
            b = ens._bearings(tr.meeting_lat, tr.meeting_lon).reshape(-1, 1)
            sil.append(silhouette_score(b, ens.clusterer.labels_, sample_size=min(3000, len(b)), random_state=42))
            ys.append(te[tm.TARGET].to_numpy(float))
            pr.append(ens.predict(te[FEATS]))
            po.append(ens.predict_oracle(te[FEATS], ens.assign_clusters(te.meeting_lat, te.meeting_lon)))
        y = np.concatenate(ys)
        r = dict(k=k, silhouette=float(np.mean(sil)),
                 r2_classifier=tm.evaluate(y, np.concatenate(pr))["r2"],
                 r2_oracle=tm.evaluate(y, np.concatenate(po))["r2"])
        print(f"[k-sweep] k={k} silhouette={r['silhouette']:.3f} R²(clf)={r['r2_classifier']:.3f} R²(oracle)={r['r2_oracle']:.3f}", flush=True)
        rows.append(r)
    return rows


def main() -> None:
    df = load()
    print(f"[data] {len(df):,} events, {df.vessel_mmsi.nunique():,} vessels, "
          f"{df.meeting_time.min()} → {df.meeting_time.max()}; "
          f"temporal test share = {(df.meeting_time >= TEMPORAL_CUTOFF).mean():.1%}", flush=True)
    rows = run_direction(df, "inbound") + run_direction(df, "outbound")
    ks = k_sweep(df)
    with open(OUT_DIR / "validation_metrics.json", "w") as fh:
        json.dump({"validation": rows, "k_sweep": ks, "temporal_cutoff": str(TEMPORAL_CUTOFF)}, fh, indent=2, default=str)
    pd.DataFrame(rows).drop(columns=["n_train_test"]).to_csv(OUT_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame(ks).to_csv(OUT_DIR / "k_sweep.csv", index=False)
    if FEATURE_SET in ("adaptive", "adaptive_ukc", "causal", "dispatch"):
        oof_predictions(df).to_csv(R2_DATASET.parent / f"oof_predictions_r2_{FEATURE_SET}.csv", index=False)
    print(f"[done] → {OUT_DIR}")


if __name__ == "__main__":
    main()
