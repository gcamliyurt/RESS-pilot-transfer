"""RESS R2 step 2 (group D): thresholds, data flow, reproducibility, Figure 4 and 9.

  R4 C1  empirical operational thresholds: binned response curves and a single
         hinge (piecewise-linear) change-point with a 1000-sample bootstrap CI
  R4 C8  data-flow funnel with the cause of every drop
  R5 C5  reproducibility parameters read from the code, recall restricted to
         log acts whose ship is visible inside the AIS extent
  R4 C6  Figure 4 statistics of the decision-time optimiser
  R5 C6  Figure 9 case: congestion-triggered relocation, before/after indicators
Output: results/revision_r2_step2/reporting/
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
R2 = ROOT / "data" / "processed" / "r2"
MP = ROOT / "data" / "processed" / "meeting_point"
OUT = ROOT / "results" / "revision_r2_step2" / "reporting"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
OFFSHORE_M = 3000.0


def hav_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def load_dataset() -> pd.DataFrame:
    d = pd.read_csv(R2 / "modelling_r2.csv", low_memory=False)
    ap = pd.read_csv(R2 / "apriori_features_r2_adaptive.csv", usecols=["event_id", "traffic_0_5nm"])
    return d.merge(ap, on="event_id", how="left")


def load_anchor_run() -> pd.DataFrame:
    o = pd.concat(pd.read_csv(f, low_memory=False) for f in glob.glob(str(MP / "optimized_meeting_points_*_r2verified_anchor.csv")))
    o["recommended_gate_pass"] = o.recommended_gate_pass.astype(str).str.lower().eq("true")
    return o


# ---------------------------------------------------------------- R4 C1 thresholds
THRESH_VARS = {
    # name: (column / transform, bin width, Table 1 range)
    "Hs_m": (lambda d: d.wave_height_m, 0.25, "1.8-2.2 m"),
    "wind_kt": (lambda d: d.wind_speed_mps * 1.943844, 2.0, "20-22 kt"),
    "visibility_nm": (lambda d: d.visibility_km / 1.852, 0.5, "~1 NM"),
    # unique third-party vessels within 0.5 NM of the ship at t_dec (apriori features). The event-level
    # surrounding_vessel_count_* columns count AIS rows in a +/-5 min window, not vessels (collision_risk.py).
    "traffic_0_5nm": (lambda d: d.traffic_0_5nm, 1.0, "4-6 vessels"),
    "ukc_m": (lambda d: d.ukc_m, 1.0, "UKC < 2 m margin"),
}


def _hinge_fit(x, y, taus):
    """Least-squares y = a + b x + c max(0, x - tau) over a tau grid; returns (tau, sse, c)."""
    best = (np.nan, np.inf, np.nan)
    X0 = np.column_stack([np.ones_like(x), x])
    for t in taus:
        X = np.column_stack([X0, np.maximum(0.0, x - t)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y - X @ coef) ** 2).sum())
        if sse < best[1]:
            best = (float(t), sse, float(coef[2]))
    return best


def thresholds(d: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    rows, curves = [], []
    for name, (fx, width, table1) in THRESH_VARS.items():
        for direction, g in d.groupby("traffic_direction"):
            x = pd.to_numeric(fx(g), errors="coerce").to_numpy(float)
            y = pd.to_numeric(g.dist_to_breakwater_m, errors="coerce").to_numpy(float) / 1000.0
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if len(x) < 50 or np.unique(x).size < 5:
                rows.append({"variable": name, "direction": direction, "n": int(len(x)), "note": "too few / no variation"})
                continue
            b = np.floor(x / width) * width
            cur = pd.DataFrame({"bin": b, "y": y}).groupby("bin").agg(
                n=("y", "size"), median_dist_km=("y", "median"), offshore_share=("y", lambda s: (s > OFFSHORE_M / 1000).mean()))
            curves.append(cur.reset_index().assign(variable=name, direction=direction, bin_width=width))
            taus = np.unique(np.quantile(x, np.linspace(0.05, 0.95, 46)))
            tau, sse_h, slope_change = _hinge_fit(x, y, taus)
            lin = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)
            sse_l = float(((y - np.column_stack([np.ones_like(x), x]) @ lin[0]) ** 2).sum())
            n = len(x)
            # F-test hinge vs linear (one extra slope + tau searched, so p is optimistic)
            f = ((sse_l - sse_h) / 2) / (sse_h / (n - 4))
            from scipy.stats import f as fdist
            p = float(1 - fdist.cdf(f, 2, n - 4))
            boot = []
            for _ in range(1000):
                i = RNG.integers(0, n, n)
                boot.append(_hinge_fit(x[i], y[i], taus)[0])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            rows.append({"variable": name, "direction": direction, "n": int(n), "bin_width": width,
                         "x_range": [round(float(np.nanmin(x)), 2), round(float(np.nanmax(x)), 2)],
                         "x_p95": round(float(np.quantile(x, 0.95)), 2),
                         "change_point": round(tau, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
                         "slope_change_km_per_unit": round(slope_change, 3),
                         "r2_gain_vs_linear": round((sse_l - sse_h) / max(((y - y.mean()) ** 2).sum(), 1e-9), 4),
                         "F_p_value": p, "table1_range": table1,
                         "supported": bool(p < 0.01 and (hi - lo) <= 0.5 * (np.quantile(x, .95) - np.quantile(x, .05)))})
    return rows, pd.concat(curves, ignore_index=True)


# ---------------------------------------------------------------- R4 C8 funnel
def funnel(d: pd.DataFrame, o: pd.DataFrame) -> dict:
    det = json.loads((R2 / "detection_validation_r2.json").read_text())
    dt = pd.read_csv(R2 / "decision_time_r2.csv")
    ap = pd.read_csv(R2 / "apriori_features_r2_adaptive.csv", usecols=["event_id"])
    dt["cause"] = np.where(dt.t_dec.isna(),
                           np.where(dt.first_vessel_fix.isna() & dt.first_pilot_fix.isna(), "no_fix_either_in_T-20..T",
                                    np.where(dt.first_vessel_fix.isna(), "no_ship_fix_in_T-20..T", "no_pilot_fix_in_T-20..T")),
                           np.where(dt.optimisable, "optimisable", "lead_lt_5min"))
    steps = {}
    for direction in ("inbound", "outbound"):
        dd = d[d.traffic_direction == direction]
        ddt = dt[dt.event_id.isin(dd.event_id)]
        opt_ids = set(ddt[ddt.optimisable].event_id)
        feat_ids = set(ap.event_id) & opt_ids
        oo = o[o.event_id.isin(dd.event_id)]
        steps[direction] = {
            "log_confirmed": int(len(dd)),
            "decision_time_causes": ddt.cause.value_counts().to_dict(),
            "optimisable_lead_ge_5min": len(opt_ids),
            "with_decision_time_features": len(feat_ids),
            "dropped_no_ship_fix_within_3min_of_t_dec": len(opt_ids - feat_ids),
            "optimised": int(len(oo)),
            "dropped_by_optimiser_no_fix_or_empty": len(feat_ids - set(oo.event_id)),
            "gate_passing": int(oo.recommended_gate_pass.sum()),
            "gate_fail_reasons": oo.loc[~oo.recommended_gate_pass, "recommended_gate_reason"].str.split(";").explode().value_counts().to_dict(),
        }
    nan = {c: int(d[c].isna().sum()) for c in ["wind_speed_mps", "wave_height_m", "visibility_km", "tide_height_m",
                                               "depth_m", "ukc_m", "vessel_draft_m", "current_speed_kn"] if c in d.columns}
    return {"ais_sessions_detected": det["ais_sessions_detected"], "log_acts_newport": det["log_acts_newport"],
            "log_confirmed": det["log_confirmed_sessions"], "by_direction": steps,
            "missing_values_among_log_confirmed": nan, "n_log_confirmed": int(len(d))}


# ---------------------------------------------------------------- R5 C5 reproducibility
def visible_recall(d: pd.DataFrame) -> dict:
    """Recall over log acts whose ship has any AIS fix within +/-2 h inside the AIS extent."""
    from build_r2_dataset import load_log
    L = load_log()
    L["week"] = L.t_log.map(lambda t: ROOT / "data/processed/ais_master/weeks" / f"{t:%Y-%m}-week{min((t.day - 1) // 7 + 1, 5)}.csv")
    vis = np.zeros(len(L), bool)
    for wf, g in L.groupby("week"):
        if not Path(wf).exists():
            continue
        ids = set(g.vessel_mmsi.astype(str))
        a = pd.concat(c[c.mmsi.astype(str).isin(ids)] for c in pd.read_csv(wf, usecols=["mmsi", "datetime"], chunksize=2_000_000))
        by = {k: np.sort(pd.to_datetime(v.datetime).values) for k, v in a.groupby(a.mmsi.astype(str))}
        for idx, r in g.iterrows():
            ts = by.get(str(r.vessel_mmsi))
            if ts is None:
                continue
            i = np.searchsorted(ts, np.datetime64(r.t_log - pd.Timedelta(hours=2)))
            vis[L.index.get_loc(idx)] = i < len(ts) and ts[i] <= np.datetime64(r.t_log + pd.Timedelta(hours=2))
    L["visible"] = vis
    L["matched"] = L.log_act.isin(set(d.log_act))
    out = {}
    for direction, g in L.groupby("dir"):
        out[direction] = {"log_acts_mapped_to_mmsi": int(len(g)), "ship_visible_in_ais_pm2h": int(g.visible.sum()),
                          "matched": int(g.matched.sum()),
                          "recall_all_mapped": round(float(g.matched.mean()), 4),
                          "recall_visible_only": round(float(g.matched[g.visible].mean()), 4)}
    return out


def reproducibility(o: pd.DataFrame, d: pd.DataFrame) -> dict:
    import src.core.detect_interceptions as di
    import src.core.optimize_meeting_point as om
    import train_models as tm
    out_b = d[d.traffic_direction == "outbound"].dropna(subset=["meeting_lat", "meeting_lon"])
    brg = bearing(35.07, 128.81, out_b.meeting_lat.to_numpy(float), out_b.meeting_lon.to_numpy(float))
    gates = json.loads((R2 / "gate_thresholds.json").read_text())
    return {
        "detection": {k: getattr(di, k) for k in ["COURSE_ALIGNMENT_THRESHOLD_DEG", "SPEED_MIN_KN", "SPEED_MAX_KN",
                                                   "SPEED_DIFF_MAX_KN", "BOARDING_MAX_SPEED_KN", "BOARDING_MIN_MOVEMENT_KN",
                                                   "MIN_TARGET_LOA_M", "MIN_TARGET_BEAM_M"] if hasattr(di, k)}
                     | {"max_gap_min_within_event (CLI default)": 15.0},
        "dbscan": "not used anywhere in the R2 pipeline (no DBSCAN in src/ or top-level scripts)",
        "rf_params": tm.RF_PARAMS, "ridge_alpha": 1.0, "hyperparameter_grid": "none (fixed parameters, no grid search)",
        "route_classifier": {"model": "RandomForestClassifier", "n_estimators": 300, "max_depth": None, "min_samples_leaf": 3},
        "route_clustering": {"model": "KMeans on raw bearing degrees (1-D, no sin/cos)", "k": 2, "n_init": 10, "random_state": 42,
                             "origin": [35.07, 128.81], "min_cluster_size_for_regressor": 30,
                             "outbound_bearing_range_deg": [round(float(brg.min()), 1), round(float(brg.max()), 1)],
                             "bearings_within_20deg_of_north": int(((brg < 20) | (brg > 340)).sum()),
                             "wraparound_risk": "none in practice" if not ((brg < 20) | (brg > 340)).any() else "present"},
        "candidate_generation": {"zone_grid_step_m": 300.0, "zone_grid_max_points": 36,
                                 "anchor_ring_m": list(om.ANCHOR_RING_M), "anchor_ring_bearings_deg": list(range(0, 360, 45)),
                                 "proximity_lambda_per_km": om.PROXIMITY_LAMBDA,
                                 "fairway_centerline": "port/starboard buoy-pair midpoints within 0.03 deg lat of the anchor, max 12",
                                 "forward_corridor": "along x {0.75,0.9,1.0,1.1,1.25,1.4} of ship->anchor, lateral {-1,-.5,0,.5,1} x min(350,max(120,step)) m",
                                 "decision_time": "t_dec = max(T-20 min, first joint ship+pilot fix), lead >= 5 min",
                                 "historical_hotspot": "median of same-direction meeting points before t_dec (same week file)"},
        "gate_thresholds_calibrated": gates.get("thresholds", {}), "gate_pass_rate_target": gates.get("pass_rate"),
    }


# ---------------------------------------------------------------- R4 C6 Figure 4
SOURCE_DEF = {
    "ml_anchor": "decision-time ML anchor: gate + out-of-fold predicted breakwater distance along the fold-learnt bearing",
    "anchor_ring": "points 150/300/500 m from the anchor at 45 deg steps (small relocation)",
    "fairway_centerline": "midpoint of a port/starboard lateral-buoy pair near the anchor",
    "zone_grid": "regular 300 m grid inside the official embarkation/disembarkation polygon (max 36)",
    "forward_corridor": "points along / beside the ship-to-anchor line, inside the boarding polygon",
    "rewind_midpoint": "midpoint of the pilot-boat and ship positions at the decision time",
    "historical_hotspot": "median of earlier same-direction meeting points",
    "zone_centroid": "centroid of the official boarding polygon",
}


def figure4(o: pd.DataFrame) -> dict:
    return {
        "n": int(len(o)),
        "source_counts": o.recommended_source.value_counts().to_dict(), "source_definitions": SOURCE_DEF,
        "waypoint_mode": o.recommended_waypoint_source.value_counts().to_dict(),
        "waypoint_note": ("all 'direct': route hazard / clearance metrics are computed on the craft's observed AIS track legs "
                          "(_track_leg), which ignore the waypoint, so every waypoint option ties with 'direct' on hazard and "
                          "clearance, and the next sort key (max turn angle, ascending) always picks 'direct' (0 deg). "
                          "Not a numerical bug, but the waypoint options are inert since the track-based route gate; "
                          "Figure 4b should be dropped or replaced."),
        "gate_pass": int(o.recommended_gate_pass.sum()),
        "collision_class": o.recommended_collision_risk_class.astype(str).value_counts().to_dict(),
        "action": o.recommendation_action.value_counts().to_dict(),
    }


# ---------------------------------------------------------------- R5 C6 Figure 9
def figure9(o: pd.DataFrame, d: pd.DataFrame) -> list[dict]:
    files = glob.glob(str(MP / "candidates_*_r2verified_anchor.csv"))
    cols = ["event_id", "candidate_source", "candidate_lat", "candidate_lon", "gate_pass", "gate_reason", "ukc_m",
            "wave_height_m", "dist_to_breakwater_m", "surrounding_vessel_count_0_5nm", "surrounding_vessel_count_1_0nm",
            "nearest_third_vessel_cpa_m", "nearest_third_vessel_hull_cpa_m", "collision_risk_class"]
    anc = pd.concat(pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols).query("candidate_source == 'ml_anchor'")
                    for f in files).drop_duplicates("event_id")
    x = o[(o.traffic_direction_ais if "traffic_direction_ais" in o else o.traffic_direction).notna()].merge(
        anc.add_prefix("a_").rename(columns={"a_event_id": "event_id"}), on="event_id")
    x["move_m"] = hav_m(x.a_candidate_lat, x.a_candidate_lon, x.recommended_lat, x.recommended_lon)
    x["dir_conf"] = x.event_id.map(d.set_index("event_id").traffic_direction)
    congested = x.a_gate_reason.astype(str).str.contains("collision") | (x.a_surrounding_vessel_count_0_5nm >= 4)
    c = x[(x.dir_conf == "inbound") & congested & (x.move_m > 500) & x.recommended_gate_pass].copy()
    # ranked on the (row-count) traffic proxy the optimiser actually used; unique counts are recomputed below
    c["traffic_drop"] = c.a_surrounding_vessel_count_0_5nm - c.recommended_surrounding_vessel_count_0_5nm
    st = pd.read_csv(ROOT.parent / "Static_merged_bbox_result_gokhan.csv", encoding="cp949")
    st.columns = ["mmsi", "name", "type", "imo", "call", "A", "B", "C", "D", "draft", "gt"]
    st = st.drop_duplicates("mmsi").set_index("mmsi")
    meta = d.set_index("event_id")
    c = c.sort_values(["traffic_drop", "move_m"], ascending=[False, True]).head(4)
    def snapshot_counts(r, lat, lon):
        wf = ROOT / "data/processed/ais_master/weeks" / f"{pd.Timestamp(str(r.rewind_time)[:19]):%Y-%m}-week{min((pd.Timestamp(str(r.rewind_time)[:19]).day - 1) // 7 + 1, 5)}.csv"
        t = pd.Timestamp(str(r.rewind_time)[:19])
        a = pd.concat(ch[(pd.to_datetime(ch.datetime) - t).abs() <= pd.Timedelta(minutes=1)]
                      for ch in pd.read_csv(wf, usecols=["mmsi", "datetime", "lat", "lon"], chunksize=2_000_000))
        a = a[~a.mmsi.astype(str).isin({str(r.vessel_mmsi), str(r.pilot_mmsi)})].drop_duplicates("mmsi")
        dist = hav_m(lat, lon, a.lat.to_numpy(float), a.lon.to_numpy(float))
        return {"unique_vessels_0_5nm": int((dist <= 926).sum()), "unique_vessels_1nm": int((dist <= 1852).sum()),
                "unique_vessels_2nm": int((dist <= 3704).sum()), "nearest_vessel_m": round(float(dist.min()), 0) if len(dist) else None}

    out = []
    for r in c.itertuples():
        s = st.loc[r.vessel_mmsi] if r.vessel_mmsi in st.index else None
        rec_cpa = r.recommended_collision_risk_class
        out.append({
            "event_id": r.event_id, "meeting_time_kst": str(r.meeting_time)[:19], "decision_time_kst": str(r.rewind_time)[:19],
            "lead_min": round(float(r.lead_min), 1), "direction": "inbound",
            "vessel": None if s is None else {"name": str(s["name"]), "type": str(meta.loc[r.event_id, "vessel_ship_type_class"]),
                                              "loa_m": float(s.A + s.B) if pd.notna(s.A) else None, "gt": float(s["gt"])},
            "before_anchor": {"vessels_0_5nm": r.a_surrounding_vessel_count_0_5nm, "vessels_1nm": r.a_surrounding_vessel_count_1_0nm,
                              "min_cpa_m": r.a_nearest_third_vessel_cpa_m, "hull_cpa_m": r.a_nearest_third_vessel_hull_cpa_m,
                              "collision_class": r.a_collision_risk_class, "ukc_m": r.a_ukc_m, "hs_m": r.a_wave_height_m,
                              "dist_to_breakwater_m": r.a_dist_to_breakwater_m, "gate": r.a_gate_reason},
            "after_recommended": {"vessels_0_5nm": r.recommended_surrounding_vessel_count_0_5nm, "collision_class": rec_cpa,
                                  "ukc_m": r.recommended_ukc_m, "hs_m": r.recommended_wave_height_m,
                                  "dist_to_breakwater_m": r.recommended_dist_to_breakwater_m, "source": r.recommended_source},
            "snapshot_before_anchor_t_dec_pm1min": snapshot_counts(r, r.a_candidate_lat, r.a_candidate_lon),
            "snapshot_after_recommended_t_dec_pm1min": snapshot_counts(r, r.recommended_lat, r.recommended_lon),
            "relocation_m": round(float(r.move_m), 0),
            "relocation_bearing_deg": round(float(bearing(r.a_candidate_lat, r.a_candidate_lon, r.recommended_lat, r.recommended_lon)), 0),
            "actual_meeting_offset_from_recommended_m": round(float(hav_m(r.actual_meeting_lat, r.actual_meeting_lon, r.recommended_lat, r.recommended_lon)), 0),
        })
    return out


def main() -> None:
    d, o = load_dataset(), load_anchor_run()
    th, curves = thresholds(d)
    curves.to_csv(OUT / "threshold_response_curves.csv", index=False)
    rep = {"thresholds": th, "funnel": funnel(d, o), "reproducibility": reproducibility(o, d),
           "recall_visible": visible_recall(d), "figure4": figure4(o), "figure9_cases": figure9(o, d)}
    (OUT / "step2_reporting.json").write_text(json.dumps(rep, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    print(json.dumps(rep, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


if __name__ == "__main__":
    main()
