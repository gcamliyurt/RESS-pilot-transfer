"""RESS R2 step 2, environment (Reviewer #3 C2/C4/C5, Reviewer #5 C6(ii)).

  1. KMA station inventory per network, data actually available, and distance from
     each log-confirmed meeting point to the station used / nearest station.
  2. Leave-one-station-out CV: nearest-station vs IDW (p = 2) RMSE per variable.
  3. Sensitivity of the adaptive decision-time model to the assignment scheme.
  4. Dynamic squat (Barrass) and its effect on the UKC >= 2 m rate.
  5. UKC sign check.
Output : results/revision_r2_step2/environment/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MET = ROOT / "data" / "raw" / "metocean"
R2 = ROOT / "data" / "processed" / "r2"
OUT = ROOT / "results" / "revision_r2_step2" / "environment"
OUT.mkdir(parents=True, exist_ok=True)
USED_STATION = 22104  # the R2 enrichment (_read_weather_hourly) uses the Geoje marine buoy only
NETWORK = {"aws": "AWS", "등표기상관측": "Lighthouse", "파고부이": "Wave buoy", "해양기상부이": "Marine buoy"}
CB = {"container/cargo": 0.65, "tanker": 0.80, "other": 0.70}


def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def stations() -> pd.DataFrame:
    s = pd.read_excel(MET / "관측소위치정보.xlsx")
    s.columns = ["name_ko", "station_id", "kind", "lat", "lon"]
    s["network"] = s.kind.map(NETWORK)
    return s


def observations() -> pd.DataFrame:
    """Hourly wind / wave per station, long format (station_id, t, wind_mps, wave_m)."""
    rd = lambda f: pd.read_csv(MET / f, encoding="cp949")
    parts = []
    a = rd("OBS_AWS_TIM_20250731120149.csv")
    parts.append(pd.DataFrame({"station_id": a["지점"], "t": a["일시"], "wind_mps": a["풍속(m/s)"], "wave_m": np.nan}))
    for f in ("OBS_BUOY_TIM_20250731121556.csv",):
        b = rd(f)
        parts.append(pd.DataFrame({"station_id": b["지점"], "t": b["일시"], "wind_mps": b["풍속(m/s)"], "wave_m": b["유의파고(m)"]}))
    c = rd("OBS_CWBUOY_TIM_20250731121902.csv")
    parts.append(pd.DataFrame({"station_id": c["지점"], "t": c["일시"], "wind_mps": np.nan, "wave_m": c["유의파고(m)"]}))
    lb = rd("OBS_LB_TIM_20250731121353.csv")
    parts.append(pd.DataFrame({"station_id": lb["지점"], "t": lb["일시"], "wind_mps": lb["풍속(m/s)"], "wave_m": lb["유의파고(m)"]}))
    g = pd.read_csv(MET / "wind&wave.csv")  # Geoje buoy 2023 (the R2 enrichment source)
    parts.append(pd.DataFrame({"station_id": g["Station_ID"], "t": g["Date_Time"], "wind_mps": g["Wind_Speed_mps"],
                               "wave_m": g["Significant_Wave_Height_m"]}))
    o = pd.concat(parts, ignore_index=True)
    o["t"] = pd.to_datetime(o.t, errors="coerce").dt.floor("h")
    for c_ in ("wind_mps", "wave_m"):
        o[c_] = pd.to_numeric(o[c_], errors="coerce")
    return o.dropna(subset=["t"]).drop_duplicates(["station_id", "t"], keep="last")


def inventory(s, o, d) -> dict:
    avail = o.groupby("station_id").agg(wind_hours=("wind_mps", "count"), wave_hours=("wave_m", "count"),
                                        first=("t", "min"), last=("t", "max")).reset_index()
    s = s.merge(avail, on="station_id", how="left")
    s.to_csv(OUT / "kma_stations.csv", index=False)
    dist = km(d.meeting_lat.to_numpy()[:, None], d.meeting_lon.to_numpy()[:, None], s.lat.to_numpy()[None, :], s.lon.to_numpy()[None, :])
    rep = {"stations_per_network": s.network.value_counts().to_dict(),
           "stations_with_wind": int((s.wind_hours > 0).sum()), "stations_with_wave": int((s.wave_hours > 0).sum()),
           "visibility_stations": 0,
           "multi_station_period": f"{o[o.station_id != USED_STATION].t.min():%Y-%m-%d} to {o[o.station_id != USED_STATION].t.max():%Y-%m-%d}",
           "n_meetings": int(len(d))}
    used = s.station_id.eq(USED_STATION).to_numpy()
    du = dist[:, used].ravel()
    rep["dist_to_used_station_km"] = {"station": "Geoje marine buoy 22104", "median": round(float(np.median(du)), 1),
                                      "p95": round(float(np.percentile(du, 95)), 1)}
    for var in ("wind", "wave"):
        m = (s[f"{var}_hours"] > 0).to_numpy()
        dn = dist[:, m].min(axis=1)
        rep[f"dist_to_nearest_{var}_station_km"] = {"station": str(s.station_id[m].to_numpy()[dist[:, m].argmin(axis=1)][0]),
                                                    "median": round(float(np.median(dn)), 1), "p95": round(float(np.percentile(dn, 95)), 1)}
    for net, g in s.groupby("network"):
        dn = dist[:, s.network.eq(net).to_numpy()].min(axis=1)
        rep.setdefault("dist_to_nearest_by_network_km", {})[net] = {"median": round(float(np.median(dn)), 1), "p95": round(float(np.percentile(dn, 95)), 1)}
    return rep


def _idw(vals: np.ndarray, dkm: np.ndarray, p: float = 2.0) -> np.ndarray:
    w = 1.0 / np.maximum(dkm, 0.1) ** p
    w = np.where(np.isnan(vals), 0.0, w)
    return np.where(w.sum(axis=1) > 0, np.nansum(vals * w, axis=1) / np.maximum(w.sum(axis=1), 1e-12), np.nan)


def _nearest(vals: np.ndarray, dkm: np.ndarray) -> np.ndarray:
    dd = np.where(np.isnan(vals), np.inf, dkm)
    i = dd.argmin(axis=1)
    out = vals[np.arange(len(vals)), i]
    return np.where(np.isfinite(dd.min(axis=1)), out, np.nan)


def loso(s, o) -> dict:
    o = o[o.station_id != USED_STATION | (o.t >= "2024-01-01")]
    o = o[o.t >= "2024-01-01"]
    rep = {}
    for var in ("wind_mps", "wave_m"):
        w = o.pivot_table(index="t", columns="station_id", values=var)
        w = w.loc[:, w.count() > 1000]
        ids = w.columns.to_numpy()
        S = s.set_index("station_id").loc[ids]
        D = km(S.lat.to_numpy()[:, None], S.lon.to_numpy()[:, None], S.lat.to_numpy()[None, :], S.lon.to_numpy()[None, :])
        V = w.to_numpy()
        rows = []
        for j, sid in enumerate(ids):
            others = np.arange(len(ids)) != j
            dk = np.broadcast_to(D[j, others], (len(V), others.sum()))
            y = V[:, j]
            pn, pi = _nearest(V[:, others], dk), _idw(V[:, others], dk)
            ok = ~np.isnan(y) & ~np.isnan(pn) & ~np.isnan(pi)
            rows.append({"station_id": int(sid), "network": S.network.iloc[j], "n_hours": int(ok.sum()),
                         "nearest_km": round(float(D[j, others].min()), 1),
                         "rmse_nearest": float(np.sqrt(np.mean((pn[ok] - y[ok]) ** 2))),
                         "rmse_idw": float(np.sqrt(np.mean((pi[ok] - y[ok]) ** 2)))})
        r = pd.DataFrame(rows)
        r.to_csv(OUT / f"loso_{var}.csv", index=False)
        g = r[r.station_id == USED_STATION]
        rep[var] = {"n_stations": int(len(r)), "rmse_nearest_mean": round(r.rmse_nearest.mean(), 3),
                    "rmse_idw_mean": round(r.rmse_idw.mean(), 3),
                    "idw_better_share": round(float((r.rmse_idw < r.rmse_nearest).mean()), 2),
                    "geoje_22104": g[["rmse_nearest", "rmse_idw", "nearest_km"]].round(3).to_dict("records")}
    rep["visibility"] = "no visibility observations in any of the four KMA files; visibility_km is empty in the R2 data (0/3,243)"
    return rep


def sensitivity(s, o) -> dict:
    """Adaptive decision-time model on Jan-Aug 2024 events (multi-station period):
    current Geoje-buoy features vs nearest-station vs IDW at the meeting point."""
    os.environ["FEATURE_SET"] = "adaptive"
    sys.argv = sys.argv[:1]
    import revision_r2_validation as v
    df = v.load()
    t = df.meeting_time.dt.tz_localize(None).dt.floor("h")
    df = df[(t >= "2024-01-01") & (t < "2024-09-01")].reset_index(drop=True)
    t = df.meeting_time.dt.tz_localize(None).dt.floor("h")
    o = o[o.t >= "2024-01-01"]
    variants = {"geoje_buoy_used": df.copy()}
    for name, fn in (("nearest_station", _nearest), ("idw_p2", _idw)):
        x = df.copy()
        for var, col in (("wind_mps", "wind_speed_mps"), ("wave_m", "wave_height_m")):
            w = o.pivot_table(index="t", columns="station_id", values=var)
            w = w.loc[:, w.count() > 1000].reindex(t.values)
            S = s.set_index("station_id").loc[w.columns]
            dk = km(df.meeting_lat.to_numpy()[:, None], df.meeting_lon.to_numpy()[:, None], S.lat.to_numpy()[None, :], S.lon.to_numpy()[None, :])
            x[col] = fn(w.to_numpy(), dk)
        variants[name] = x
    rep = {}
    for name, x in variants.items():
        for direction in ("inbound", "outbound"):
            sub = x[x[v.tm.DIRECTION_COL].str.lower() == direction].dropna(subset=["meeting_lat", "meeting_lon"]).reset_index(drop=True)
            ys, ps = [], []
            for tr_i, te_i in v.splits(sub, "grouped"):
                tr, te = v.impute(sub.iloc[tr_i], sub.iloc[te_i])
                ps.append(v.fit_predict("proposed", tr, te, direction)); ys.append(te[v.tm.TARGET].to_numpy(float))
            m = v.tm.evaluate(np.concatenate(ys), np.concatenate(ps))
            rep.setdefault(direction, {})[name] = {"n": int(len(sub)), "r2": round(m["r2"], 3), "mae_m": round(m["mae_m"], 1)}
    for direction, r in rep.items():
        base = r["geoje_buoy_used"]
        for name in ("nearest_station", "idw_p2"):
            r[name]["dR2"] = round(r[name]["r2"] - base["r2"], 3)
            r[name]["dMAE_m"] = round(r[name]["mae_m"] - base["mae_m"], 1)
    return rep


def meeting_sog(d: pd.DataFrame) -> pd.Series:
    """Ship SOG (kn) at the meeting time from the weekly AIS master (nearest fix within 2 min)."""
    W = ROOT / "data" / "processed" / "ais_master" / "weeks"
    T = pd.to_datetime(d.meeting_time.astype(str).str[:19])
    wf = T.map(lambda x: W / f"{x:%Y-%m}-week{min((x.day - 1) // 7 + 1, 5)}.csv")
    out = pd.Series(np.nan, index=d.index)
    for f, idx in d.groupby(wf).groups.items():
        if not Path(f).exists():
            continue
        ids = set(d.loc[idx, "vessel_mmsi"].astype(str))
        a = pd.concat(c[c.mmsi.astype(str).isin(ids)] for c in pd.read_csv(f, usecols=["mmsi", "datetime", "spd"], chunksize=2_000_000))
        a["mmsi"] = a.mmsi.astype(str); a["datetime"] = pd.to_datetime(a.datetime)
        by = {k: g.sort_values("datetime") for k, g in a.groupby("mmsi")}
        for i in idx:
            g = by.get(str(d.at[i, "vessel_mmsi"]))
            if g is None:
                continue
            dt = (g.datetime - T[i]).abs()
            j = dt.idxmin()
            if dt[j] <= pd.Timedelta(minutes=2):
                out[i] = g.at[j, "spd"]
    return out


def squat(d: pd.DataFrame) -> dict:
    st = pd.read_csv(ROOT.parent / "Static_merged_bbox_result_gokhan.csv", encoding="cp949")
    code = st.drop_duplicates("MMSI").set_index("MMSI")["선종코드"]
    c = pd.to_numeric(d.vessel_mmsi, errors="coerce").map(code)
    d = d.assign(cb_class=np.where(c.between(80, 89), "tanker", np.where(c.between(70, 79), "container/cargo", "other")))
    d["cb"] = d.cb_class.map(CB)
    d["sog_meet"] = meeting_sog(d)
    d.loc[(d.sog_meet < 0) | (d.sog_meet > 30), "sog_meet"] = np.nan
    d["squat_open_m"] = d.cb * d.sog_meet ** 2 / 100.0
    d["squat_conf_m"] = 2 * d.squat_open_m
    ukc = pd.to_numeric(d.ukc_m, errors="coerce")
    share = lambda m: {"n": int(m.sum()), "of": int(m.notna().sum() if m.dtype != bool else len(m)), "pct": round(100 * float(m.mean()), 1)}
    rep = {"cb_assumption": CB | {"note": "AIS type 70-79 = container/cargo (New Port is a container port; 0.80 bulk value is the upper bound), 80-89 = tanker"},
           "cb_class_counts": d.cb_class.value_counts().to_dict(),
           "sog_at_meeting_kn": {k: {"n": int(g.sog_meet.notna().sum()), "median": round(g.sog_meet.median(), 1),
                                     "q25": round(g.sog_meet.quantile(.25), 1), "q75": round(g.sog_meet.quantile(.75), 1)}
                                 for k, g in [("all", d)] + list(d.groupby("cb_class"))},
           "squat_open_m": {"median": round(d.squat_open_m.median(), 2), "p95": round(d.squat_open_m.quantile(.95), 2)},
           "squat_confined_m": {"median": round(d.squat_conf_m.median(), 2), "p95": round(d.squat_conf_m.quantile(.95), 2)}}
    ok = ukc.notna() & d.squat_open_m.notna()
    rep["real_transfers_ukc_ge_2m"] = {"n_with_sog": int(ok.sum()), "no_squat": share((ukc[ok] >= 2)),
                                       "open_squat": share((ukc[ok] - d.squat_open_m[ok]) >= 2),
                                       "confined_squat": share((ukc[ok] - d.squat_conf_m[ok]) >= 2)}
    d["cb_upper"] = np.where(d.cb_class == "container/cargo", 0.80, d.cb)
    rep["real_transfers_ukc_ge_2m"]["confined_squat_cb0.80"] = share((ukc[ok] - 2 * d.cb_upper[ok] * d.sog_meet[ok] ** 2 / 100) >= 2)

    import glob
    o = pd.concat(pd.read_csv(f, usecols=["event_id", "recommended_ukc_m", "recommendation_status"])
                  for f in glob.glob(str(ROOT / "data/processed/meeting_point/optimized_meeting_points_*_r2verified_causal.csv")))
    o = o[o.recommendation_status == "feasible"] if "recommendation_status" in o else o  # infeasible: diagnostics only
    o = o.merge(d[["event_id", "cb", "sog_meet", "squat_open_m", "squat_conf_m"]], on="event_id")
    ap = pd.read_csv(R2 / "dispatch_features_r2.csv", usecols=["event_id", "ref_sog_kn"])
    o = o.merge(ap, on="event_id", how="left")
    o["squat_dec_conf_m"] = 2 * o.cb * o.ref_sog_kn ** 2 / 100
    u = pd.to_numeric(o.recommended_ukc_m, errors="coerce")
    k = u.notna() & o.squat_conf_m.notna()
    kd = u.notna() & o.squat_dec_conf_m.notna()
    rep["recommendations_ukc_ge_2m"] = {"n": int(len(o)), "no_squat": share(u >= 2),
                                        "open_squat_meeting_sog": share((u[k] - o.squat_open_m[k]) >= 2),
                                        "confined_squat_meeting_sog": share((u[k] - o.squat_conf_m[k]) >= 2),
                                        "confined_squat_decision_sog": share((u[kd] - o.squat_dec_conf_m[kd]) >= 2),
                                        "decision_sog_median_kn": round(o.ref_sog_kn.median(), 1)}
    d[["event_id", "cb_class", "cb", "sog_meet", "squat_open_m", "squat_conf_m", "ukc_m"]].to_csv(OUT / "squat_by_event.csv", index=False)
    return rep


def ukc_sign(d: pd.DataFrame) -> dict:
    u = pd.to_numeric(d.ukc_m, errors="coerce")
    return {"compute_ukc": "src/core/ukc_risk.py: UKC = depth - draft - squat, depth positive (ENC DEPARE, depth_source=enc_geojson for 3,243/3,243); tide added before",
            "depth_m_min_median": [float(pd.to_numeric(d.depth_m, errors='coerce').min()), float(pd.to_numeric(d.depth_m, errors='coerce').median())],
            "real_transfers_ukc_m": {"n": int(u.notna().sum()), "median": round(u.median(), 1), "q25": round(u.quantile(.25), 1),
                                     "q75": round(u.quantile(.75), 1), "min": round(u.min(), 1), "lt_0": int((u < 0).sum())},
            "old_figure8_bug": "enrich_sessions_with_bathymetry_weather.py at HEAD computed ukc_start_m = GEBCO elevation (negative) - draft, "
                               "giving UKC of about -10 m (old Fig. 8); working-tree version takes abs(elevation) + tide - draft. "
                               "The R2 pipeline does not use this script."}


def main() -> None:
    d = pd.read_csv(R2 / "modelling_r2.csv", low_memory=False)
    d = d.dropna(subset=["meeting_lat", "meeting_lon"])
    s, o = stations(), observations()
    rep = {"inventory": inventory(s, o, d), "loso_cv": loso(s, o), "assignment_sensitivity_2024JanAug": sensitivity(s, o),
           "squat": squat(d), "ukc_sign": ukc_sign(d)}
    (OUT / "environment.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
