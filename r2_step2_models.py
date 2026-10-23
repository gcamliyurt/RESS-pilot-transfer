"""RESS R2 step 2 (models): numbers for the response letter on the adaptive
decision-time data (FEATURE_SET=adaptive).

  B1 official fixed zone baseline, XGBoost, baselines table restated, held-out
  route-classifier accuracy, OOF prediction intervals, zone sizes and pilot-boat
  speed (R4 C4), computational cost (R4 C10), k-sweep (R4 C2).
Output: results/revision_r2_step2/models/
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time

os.environ["FEATURE_SET"] = "adaptive"

import numpy as np
import pandas as pd
import sklearn

import revision_r2_validation as v
import train_models as tm
from src.core.detect_interceptions import load_zone_lookup
from src.core.enrich_event_context import BREAKWATER_LIGHTS, _nearest_buoy
from src.ml.model import HybridDistanceRegressor
from src.ml.route_ensemble import OutboundRouteEnsemble

OUT = tm.ROOT / "results" / "revision_r2_step2" / "models"
OUT.mkdir(parents=True, exist_ok=True)


def _xgb():
    try:
        from xgboost import XGBRegressor
        return "xgboost", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                       colsample_bytree=0.8, random_state=42, n_jobs=-1)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return "hist_gradient_boosting (xgboost not installed)", HistGradientBoostingRegressor(random_state=42)


def baselines(df: pd.DataFrame) -> list[dict]:
    zones = load_zone_lookup()
    zd = {}
    for direction, key in (("inbound", "embark"), ("outbound", "disembark")):
        c = zones[key].centroid
        zd[direction] = float(_nearest_buoy(float(c.y), float(c.x), BREAKWATER_LIGHTS)[1])
    xgb_name, xgb = _xgb()
    rows = []
    for direction in ("inbound", "outbound"):
        sub = df[df[tm.DIRECTION_COL].str.lower() == direction].dropna(subset=["meeting_lat", "meeting_lon"]).reset_index(drop=True)
        for scheme in ("grouped", "temporal"):
            folds = list(v.splits(sub, scheme))
            for name in ("B1_official_zone", "xgboost"):
                ys, ps, fr2 = [], [], []
                for tr_i, te_i in folds:
                    tr, te = v.impute(sub.iloc[tr_i], sub.iloc[te_i])
                    y = te[tm.TARGET].to_numpy(float)
                    if name == "B1_official_zone":
                        p = np.full(len(te), zd[direction])
                    else:
                        p = v._sk_pipe(xgb).fit(tr[v.FEATS], tr[tm.TARGET].to_numpy(float)).predict(te[v.FEATS])
                    ys.append(y); ps.append(p); fr2.append(tm.evaluate(y, p)["r2"])
                m = v.metrics(np.concatenate(ys), np.concatenate(ps))
                m.update(direction=direction, scheme=scheme, model=name if name != "xgboost" else xgb_name,
                         r2_fold_sd=float(np.std(fr2)), zone_centroid_dist_m=zd[direction] if name == "B1_official_zone" else None)
                rows.append(m)
    return rows


def route_classifier_accuracy(df: pd.DataFrame) -> dict:
    sub = df[df[tm.DIRECTION_COL].str.lower() == "outbound"].dropna(subset=["meeting_lat", "meeting_lon"]).reset_index(drop=True)
    tr_acc, te_acc, n = [], [], []
    for tr_i, te_i in v.splits(sub, "grouped"):
        tr, te = v.impute(sub.iloc[tr_i], sub.iloc[te_i])
        ens = OutboundRouteEnsemble(n_clusters=2, numeric_features=tm.NUM_FEATURES, categorical_features=tm.CATEGORICAL_FEATURES,
                                    rf_params=tm.RF_PARAMS, ridge_alpha=1.0)
        ens.fit(tr[v.FEATS], tr[tm.TARGET].to_numpy(float), tr.meeting_lat, tr.meeting_lon)
        true = ens.assign_clusters(te.meeting_lat, te.meeting_lon)
        pred = ens.classifier.predict(ens._classifier_preprocessor.transform(te[v.FEATS]))
        tr_acc.append(ens.classifier_train_acc_); te_acc.append(float((pred == true).mean())); n.append(len(te))
    return {"training_accuracy_mean": float(np.mean(tr_acc)), "heldout_accuracy_pooled": float(np.average(te_acc, weights=n)),
            "heldout_accuracy_fold_sd": float(np.std(te_acc)), "n_outbound": int(sum(n))}


def intervals(df: pd.DataFrame) -> dict:
    o = pd.read_csv(tm.ROOT / "data/processed/r2/oof_predictions_r2_adaptive.csv")
    o = o.merge(df[["event_id", tm.TARGET, "vessel_mmsi"]], on="event_id")
    o["err"] = o[tm.TARGET] - o.pred_dist_m
    out = {}
    for direction, g in o.groupby("traffic_direction"):
        r = {"n": int(len(g)), "mae_m": float(g.err.abs().mean()),
             "share_abs_err_lt_500m": float((g.err.abs() < 500).mean())}
        for lvl in (0.8, 0.9):
            lo, hi = np.quantile(g.err, [(1 - lvl) / 2, 1 - (1 - lvl) / 2])
            r[f"pi{int(lvl*100)}_m"] = [float(lo), float(hi)]
            # conformal-style check: interval from the other folds' residuals, coverage on the held-out fold
            cov, w = [], []
            gg = g.reset_index(drop=True)
            for tr_i, te_i in v.splits(gg.assign(vessel_mmsi=gg.vessel_mmsi), "grouped"):
                a, b = np.quantile(gg.err.iloc[tr_i], [(1 - lvl) / 2, 1 - (1 - lvl) / 2])
                e = gg.err.iloc[te_i]
                cov.append(float(((e >= a) & (e <= b)).mean())); w.append(len(te_i))
            r[f"pi{int(lvl*100)}_heldout_coverage"] = float(np.average(cov, weights=w))
        out[direction] = r
    return out


def zone_and_speed() -> dict:
    zones = load_zone_lookup()
    k = np.cos(np.radians(35.05)) * 111320.0
    res = {}
    for key in ("embark", "disembark"):
        z = zones[key]
        xs, ys = z.exterior.coords.xy
        pts = np.c_[np.asarray(xs) * k, np.asarray(ys) * 111320.0]
        ext = float(max(np.hypot(*(pts[i] - pts[j])) for i in range(len(pts)) for j in range(i + 1, len(pts))))
        res[key] = {"area_km2": float(z.area * k * 111320.0 / 1e6), "max_extent_km": ext / 1000}
    sog = []
    for f in sorted((tm.ROOT / "data/processed/ais_master/weeks").glob("*.csv"))[::6]:
        for c in pd.read_csv(f, usecols=["mmsi", "spd"], chunksize=2_000_000):
            c = c[c.mmsi.isin(v.VERIFIED_PILOTS)]
            sog.append(pd.to_numeric(c.spd, errors="coerce"))
    s = pd.concat(sog)
    s = s[(s > 5) & (s < 40)]
    res["pilot_boat_sog_kn"] = {"median_when_gt5kn": float(s.median()), "n_fixes": int(len(s)), "weeks_sampled": "every 6th week"}
    return res


def cost(df: pd.DataFrame) -> dict:
    hw = {"machine": subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout.strip(),
          "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
          "ram_gb": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30,
          "python": platform.python_version(), "sklearn": sklearn.__version__}
    res = {"hardware": hw}
    for direction in ("inbound", "outbound"):
        sub = df[df[tm.DIRECTION_COL].str.lower() == direction].dropna(subset=["meeting_lat", "meeting_lon"]).reset_index(drop=True)
        tr, te = v.impute(sub, sub)
        t0 = time.perf_counter()
        if direction == "inbound":
            m = HybridDistanceRegressor(numeric_features=tm.NUM_FEATURES, categorical_features=tm.CATEGORICAL_FEATURES,
                                        rf_params=tm.RF_PARAMS, ridge_alpha=1.0).fit(tr[v.FEATS], tr[tm.TARGET].to_numpy(float))
        else:
            m = OutboundRouteEnsemble(n_clusters=2, numeric_features=tm.NUM_FEATURES, categorical_features=tm.CATEGORICAL_FEATURES,
                                      rf_params=tm.RF_PARAMS, ridge_alpha=1.0)
            m.fit(tr[v.FEATS], tr[tm.TARGET].to_numpy(float), tr.meeting_lat, tr.meeting_lon)
        t_fit = time.perf_counter() - t0
        one = te[v.FEATS].iloc[[0]]
        t0 = time.perf_counter()
        for _ in range(50):
            m.predict(one)
        res[direction] = {"n_train": int(len(tr)), "train_s": t_fit, "inference_ms_per_event": (time.perf_counter() - t0) / 50 * 1000}
    tf = OUT / "optimiser_timing.json"
    if tf.exists():
        res["optimiser"] = json.loads(tf.read_text())
    return res


def main() -> None:
    df = v.load()
    b = baselines(df)
    vm = pd.read_csv(tm.ROOT / "results/revision_r2_logconfirmed_adaptive/validation_metrics.csv")
    vm = vm[vm.scheme.isin(["grouped", "temporal"])]
    table = pd.concat([vm[["direction", "scheme", "model", "r2", "r2_ci95", "r2_fold_sd", "mae_m", "mae_m_ci95", "rmse_m", "n"]],
                       pd.DataFrame(b)[["direction", "scheme", "model", "r2", "r2_ci95", "r2_fold_sd", "mae_m", "mae_m_ci95", "rmse_m", "n"]]],
                      ignore_index=True).sort_values(["direction", "scheme", "r2"])
    table.to_csv(OUT / "baselines_table.csv", index=False)
    rep = {"baselines_new": b, "route_classifier": route_classifier_accuracy(df), "prediction_intervals": intervals(df),
           "zones_and_pilot_speed": zone_and_speed(), "cost": cost(df),
           "k_sweep": pd.read_csv(tm.ROOT / "results/revision_r2_logconfirmed_adaptive/k_sweep.csv").to_dict("records")}
    (OUT / "step2_models.json").write_text(json.dumps(rep, indent=2, default=str))
    print(table.round(3).to_string())
    print(json.dumps({k: rep[k] for k in rep if k != "baselines_new"}, indent=1, default=str))


if __name__ == "__main__":
    main()
