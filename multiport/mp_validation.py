"""Multi-port: transferability of the decision-time meeting-distance model (RESS R2, Section 6.2).

For every port and direction, the same models and partitions as Busan (revision_r2_validation.py):
  ship-grouped 5-fold CV (GroupKFold by ship MMSI) and a forward temporal hold-out
  (train January-October 2023, test November-December 2023), with
  B1 official boarding area (US: NOAA pilot boarding area position; distance to the base),
  B2 historical mean, B3 linear, B4 random forest, B5 XGBoost, B6 RF-Ridge hybrid (proposed).
Leave-one-port-out (LOPO): the hybrid trained on all other ports of the same direction
predicts the held-out port, to test transfer without local retraining.
Output: results/multiport/validation_metrics.csv, lopo.csv
On macOS set DYLD_FALLBACK_LIBRARY_PATH=<sklearn>/.dylibs for XGBoost.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ml.model import HybridDistanceRegressor  # noqa: E402

FEAT_DIR = ROOT / "data" / "processed" / "multiport"
OUT = ROOT / "results" / "multiport"
LEE = ROOT.parents[1] / "6.8 leeSide" / "data"
NUM = ["ref_dist_to_base_m", "ref_bearing_sin", "ref_bearing_cos", "ref_sog_kn", "ref_cog_sin", "ref_cog_cos",
       "traffic_0_5nm", "traffic_1nm", "traffic_2nm", "nearest_third_m", "vessel_draft_m",
       "wind_speed_mps", "wave_height_m", "hour_local", "month", "lead_min"]
CAT = ["vessel_ship_type_class"]
# MP_SET=dispatch: inputs at pilot-boat dispatch (build_mp_dispatch_features.py), no lead time
MP_SET = os.environ.get("MP_SET", "trigger")
if MP_SET == "dispatch":
    NUM = ["ship_visible", "ref_dist_to_base_m", "ref_bearing_sin", "ref_bearing_cos", "ref_sog_kn", "ref_cog_sin",
           "ref_cog_cos", "pilot_dist_to_base_m", "pilot_bearing_sin", "pilot_bearing_cos", "pilot_sog_kn",
           "pilot_cog_sin", "pilot_cog_cos", "pilot_ship_range_m", "pilot_to_ship_bearing_sin",
           "pilot_to_ship_bearing_cos", "intercept_dist_to_base_m", "intercept_time_min", "traffic_0_5nm",
           "traffic_1nm", "traffic_2nm", "nearest_third_m", "vessel_draft_m", "wind_speed_mps", "wave_height_m",
           "hour_local", "month"]
    OUT = ROOT / "results" / "multiport_dispatch"
TARGET = "dist_to_base_m"
RF = {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "n_jobs": -1, "random_state": 42}
CUTOFF = pd.Timestamp("2023-11-01", tz="UTC")
US_PBA = {"us_galveston": "galveston_bay", "us_longbeach": "long_beach", "us_port_arthur": "port_arthur",
          "us_capehenry": "cape_henry", "us_swpass": "mississippi_river_southwest_pass",
          "us_port_everglades": "port_everglades"}


def hav_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def impute(tr, te):
    med = tr[NUM].median().fillna(0.0)
    tr, te = tr.copy(), te.copy()
    tr[NUM], te[NUM] = tr[NUM].fillna(med), te[NUM].fillna(med)
    return tr, te


def pipe(est):
    pre = HybridDistanceRegressor(numeric_features=NUM, categorical_features=CAT)._build_preprocessor()
    return Pipeline([("pre", pre), ("est", est)])


def fit_predict(name, tr, te, b1=None):
    y = tr[TARGET].to_numpy(float)
    X, Xt = tr[NUM + CAT], te[NUM + CAT]
    if name == "B1_official_zone":
        return np.full(len(te), b1)
    if name == "B2_historical_mean":
        return np.full(len(te), y.mean())
    if name == "B3_linear":
        return pipe(LinearRegression()).fit(X, y).predict(Xt)
    if name == "B4_random_forest":
        return pipe(RandomForestRegressor(**RF)).fit(X, y).predict(Xt)
    if name == "B5_xgboost":
        from xgboost import XGBRegressor
        return pipe(XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, random_state=42, n_jobs=-1)).fit(X, y).predict(Xt)
    if name == "B6_hybrid":
        return HybridDistanceRegressor(numeric_features=NUM, categorical_features=CAT, rf_params=RF,
                                       ridge_alpha=1.0).fit(X, y).predict(Xt)
    raise ValueError(name)


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
    pattern = "dispatch_*.csv" if MP_SET == "dispatch" else "features_*.csv"
    d = pd.concat([pd.read_csv(f) for f in sorted(FEAT_DIR.glob(pattern))], ignore_index=True)
    for c in NUM:
        if c not in d:
            d[c] = np.nan
    d = sanitise_intercept(d)
    d = d[d.traffic_direction.isin(["inbound", "outbound"])].dropna(subset=[TARGET]).copy()
    d["meeting_time"] = pd.to_datetime(d.meeting_time, utc=True)
    d["vessel_ship_type_class"] = d.vessel_ship_type_class.fillna("UNK").astype(str)
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.reset_index(drop=True)


def b1_distance(port: str) -> float | None:
    if port not in US_PBA:
        return None
    pba = json.loads((LEE / "raw/multiport/us_pba_groups.json").read_text())[US_PBA[port]]
    base = pd.read_csv(LEE / "processed/multiport/pilot_bases.csv").set_index("port").loc[port]
    return float(hav_m(base.base_lat, base.base_lon, pba["lat"], pba["lon"]))


def metrics(y, p, groups=None, n_boot=0) -> dict:
    out = {"r2": float(r2_score(y, p)), "mae_m": float(mean_absolute_error(y, p)),
           "rmse_m": float(np.sqrt(np.mean((y - p) ** 2)))}
    if n_boot and groups is not None:  # ship-clustered bootstrap CI for R²
        rng = np.random.default_rng(42)
        ug, inv = np.unique(groups, return_inverse=True)
        mem = [np.where(inv == k)[0] for k in range(len(ug))]
        bs = [r2_score(y[i], p[i]) for i in (np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(n_boot))]
        out["r2_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def main() -> None:
    d = load()
    rows = []
    models = ["B1_official_zone", "B2_historical_mean", "B3_linear", "B4_random_forest", "B5_xgboost", "B6_hybrid"]
    for (port, direc), sub in d.groupby(["port", "traffic_direction"]):
        sub = sub.reset_index(drop=True)
        if len(sub) < 200 or sub.vessel_mmsi.nunique() < 20:
            continue
        b1 = b1_distance(port)
        schemes = {"grouped": list(GroupKFold(5).split(sub, groups=sub.vessel_mmsi.astype(str))),
                   "temporal": [(np.where(sub.meeting_time < CUTOFF)[0], np.where(sub.meeting_time >= CUTOFF)[0])]}
        for scheme, folds in schemes.items():
            for m in models:
                if m == "B1_official_zone" and b1 is None:
                    continue
                ys, ps, gs = [], [], []
                for tr_i, te_i in folds:
                    tr, te = impute(sub.iloc[tr_i], sub.iloc[te_i])
                    ys.append(te[TARGET].to_numpy(float)); ps.append(fit_predict(m, tr, te, b1))
                    gs.append(te.vessel_mmsi.astype(str).to_numpy())
                r = metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(gs), 500 if m == "B6_hybrid" else 0)
                rows.append({"port": port, "direction": direc, "scheme": scheme, "model": m, "n": len(sub),
                             "n_ships": int(sub.vessel_mmsi.nunique()), "target_sd_m": float(sub[TARGET].std()), **r})
                print(f"{port:20s} {direc:8s} {scheme:8s} {m:20s} R²={r['r2']:.3f} MAE={r['mae_m']:.0f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "validation_metrics.csv", index=False)

    # Leave-one-port-out: hybrid trained on the other ports of the same direction.
    lo = []
    for direc, dd in d.groupby("traffic_direction"):
        ports = [p for p, g in dd.groupby("port") if len(g) >= 200]
        for p in ports:
            tr, te = impute(dd[dd.port.isin(ports) & (dd.port != p)], dd[dd.port == p])
            pred = fit_predict("B6_hybrid", tr, te)
            lo.append({"port": p, "direction": direc, "n": len(te), **metrics(te[TARGET].to_numpy(float), pred)})
            print(f"LOPO {p:20s} {direc:8s} R²={lo[-1]['r2']:.3f} MAE={lo[-1]['mae_m']:.0f}", flush=True)
    pd.DataFrame(lo).to_csv(OUT / "lopo.csv", index=False)


if __name__ == "__main__":
    main()
