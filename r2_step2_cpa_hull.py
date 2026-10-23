"""RESS R2 step 2 (Reviewer #5 C1/C2): signed vs truncated CPA-SD and hull-geometry sensitivity.

For each log-confirmed transfer, the subject ship and its nearest third-party vessel
(nearest_third_vessel_mmsi from enrichment) are taken from the weekly AIS master at the
meeting time, moved linearly to TCPA (TCPA clipped at 0: a past CPA is not a future risk),
and compared under four hull models:
  (a) length-only      r = L/2                      -> signed Delta_SD = D_CPA - (L_O + L_T)/2
  (b) beam-only        r = B/2
  (c) heading-weighted r = (L|cos t| + B|sin t|)/2, t = ship heading vs line of sight
  (d) polygon          oriented LOA x beam rectangles (heading, AIS antenna offsets A/B/C/D)
CPA-SD = max(0, Delta_SD). Heading 511/missing -> COG; missing dims -> 50 x 10 m (as the pipeline).
Encounter type from relative heading (COLREG-style): head-on |dpsi| >= 165 deg; overtaking
|dpsi| <= 67.5 deg with one ship > 22.5 deg abaft the other's beam; else crossing;
'stationary' when either SOG < 1 kn.
The decision-time recommendations (optimized_meeting_points_*_r2verified_causal.csv) are
re-classified under each model with the pipeline thresholds (<100 m high, <300 m medium),
alongside the pipeline's own domain-clearance class.
Output: results/revision_r2_step2/cpa_hull/
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "revision_r2_step2" / "cpa_hull"
OUT.mkdir(parents=True, exist_ok=True)
WEEKS = ROOT / "data" / "processed" / "ais_master" / "weeks"
TOL = pd.Timedelta(minutes=3)
KN = 0.514444
VARIANTS = ["length", "beam", "heading_weighted", "polygon"]


def week_file(t: pd.Timestamp) -> Path:
    return WEEKS / f"{t:%Y-%m}-week{min((t.day - 1) // 7 + 1, 5)}.csv"


def load_static() -> pd.DataFrame:
    st = pd.read_csv(ROOT.parent / "Static_merged_bbox_result_gokhan.csv", encoding="cp949")
    st.columns = ["mmsi", "name", "type", "imo", "call", "A", "B", "C", "D", "draft", "gt"]
    st["mmsi"] = st.mmsi.astype(str)
    return st.drop_duplicates("mmsi").set_index("mmsi")[["A", "B", "C", "D"]]


def states(queries: pd.DataFrame) -> pd.DataFrame:
    """queries: key, mmsi, t -> nearest AIS fix within TOL (lat, lon, spd, crs, hdg)."""
    out = []
    for wf, g in queries.groupby(queries.t.map(week_file)):
        if not Path(wf).exists():
            continue
        ids = set(g.mmsi)
        t0, t1 = g.t.min() - TOL, g.t.max() + TOL
        a = pd.concat(c[c.mmsi.astype(str).isin(ids)] for c in pd.read_csv(wf, chunksize=2_000_000,
                                                                           usecols=["mmsi", "datetime", "lat", "lon", "spd", "crs", "hdg"]))
        a["mmsi"] = a.mmsi.astype(str)
        a["datetime"] = pd.to_datetime(a.datetime)
        a = a[(a.datetime >= t0) & (a.datetime <= t1)].sort_values("datetime")
        by = {k: v.reset_index(drop=True) for k, v in a.groupby("mmsi")}
        for q in g.itertuples():
            s = by.get(q.mmsi)
            if s is None:
                continue
            dt = (s.datetime - q.t).abs()
            i = dt.idxmin()
            if dt[i] <= TOL:
                r = s.loc[i]
                out.append({"key": q.key, "mmsi": q.mmsi, "lat": r.lat, "lon": r.lon, "spd": r.spd, "crs": r.crs, "hdg": r.hdg})
    return pd.DataFrame(out)


def dims(static: pd.DataFrame, mmsi: str) -> tuple[float, float, float, float, bool]:
    if mmsi in static.index:
        a, b, c, d = (float(x) if pd.notna(x) else 0.0 for x in static.loc[mmsi])
        if a + b > 0 and c + d > 0:
            return a, b, c, d, True
    return 25.0, 25.0, 5.0, 5.0, False  # 50 x 10 m centred, as estimate_loa_beam_m defaults


def heading(r) -> tuple[float, bool]:
    h = r.hdg
    if pd.notna(h) and 0 <= float(h) < 360:
        return float(h), False
    return float(r.crs) if pd.notna(r.crs) else 0.0, True


def rect(cx, cy, hdg, a, b, c, d) -> Polygon:
    """Hull rectangle; (cx, cy) = antenna position, A fwd / B aft / C port / D starboard."""
    th = math.radians(hdg)
    f = (math.sin(th), math.cos(th))
    s = (math.cos(th), -math.sin(th))
    pts = [(fw, st) for fw, st in ((a, d), (a, -c), (-b, -c), (-b, d))]
    return Polygon([(cx + fw * f[0] + st * s[0], cy + fw * f[1] + st * s[1]) for fw, st in pts])


def _z(x) -> float:
    return 0.0 if pd.isna(x) else float(x)


def pair(own, tgt, so, stt) -> dict:
    lat0 = (own.lat + tgt.lat) / 2
    kx = 111320.0 * math.cos(math.radians(lat0))
    po = np.array([(own.lon - tgt.lon) * kx / 2, (own.lat - tgt.lat) * 111320.0 / 2])
    pt = -po
    ho, fo = heading(own)
    ht, ft = heading(tgt)
    vo = np.array([math.sin(math.radians(_z(own.crs))), math.cos(math.radians(_z(own.crs)))]) * _z(own.spd) * KN
    vt = np.array([math.sin(math.radians(_z(tgt.crs))), math.cos(math.radians(_z(tgt.crs)))]) * _z(tgt.spd) * KN
    dp, dv = pt - po, vt - vo
    tcpa_raw = -float(dp @ dv) / float(dv @ dv) if float(dv @ dv) > 0 else 0.0
    tcpa = max(0.0, tcpa_raw)
    po_c, pt_c = po + vo * tcpa, pt + vt * tcpa  # antenna positions at TCPA
    ao, bo, co, do_, ok_o = so
    at, bt, ct, dt_, ok_t = stt
    Po, Pt = rect(*po_c, ho, ao, bo, co, do_), rect(*pt_c, ht, at, bt, ct, dt_)
    co_xy, ct_xy = np.array(Po.centroid.coords[0]), np.array(Pt.centroid.coords[0])
    D = float(np.hypot(*(ct_xy - co_xy)))  # centre-to-centre distance at TCPA
    los = math.degrees(math.atan2(*(ct_xy - co_xy))) % 360
    Lo, Bo, Lt, Bt = ao + bo, co + do_, at + bt, ct + dt_
    t_o, t_t = math.radians(los - ho), math.radians(los + 180 - ht)
    dpsi = (ht - ho + 180) % 360 - 180
    rel_t = (los - ho) % 360            # bearing of tgt relative to own bow
    rel_o = (los + 180 - ht) % 360      # bearing of own relative to tgt bow
    if _z(own.spd) < 1 or _z(tgt.spd) < 1:
        enc = "stationary"
    elif abs(dpsi) >= 165:
        enc = "head_on"
    elif abs(dpsi) <= 67.5 and (112.5 <= rel_t <= 247.5 or 112.5 <= rel_o <= 247.5):
        enc = "overtaking"
    else:
        enc = "crossing"
    return {
        "tcpa_raw_min": tcpa_raw / 60, "tcpa_min": tcpa / 60, "d_cpa_centre_m": D,
        "clr_length": D - (Lo + Lt) / 2,
        "clr_beam": D - (Bo + Bt) / 2,
        "clr_heading_weighted": D - 0.5 * (Lo * abs(math.cos(t_o)) + Bo * abs(math.sin(t_o)))
                                  - 0.5 * (Lt * abs(math.cos(t_t)) + Bt * abs(math.sin(t_t))),
        "clr_polygon": -1.0 if Po.intersects(Pt) else float(Po.distance(Pt)),
        "polygon_intersects": bool(Po.intersects(Pt)),
        "encounter": enc, "dpsi_deg": dpsi,
        "own_hdg_fallback": fo, "tgt_hdg_fallback": ft, "own_dims_ok": ok_o, "tgt_dims_ok": ok_t,
    }


def share(m) -> dict:
    m = pd.Series(m).dropna().astype(bool)
    return {"n": int(m.sum()), "of": int(len(m)), "pct": round(100 * float(m.mean()), 1) if len(m) else None}


def run_pairs(q: pd.DataFrame, static) -> pd.DataFrame:
    """q: key, own_mmsi, tgt_mmsi, t."""
    s = states(pd.concat([q[["key", "t"]].assign(mmsi=q.own_mmsi, key=q.key + "|o"),
                          q[["key", "t"]].assign(mmsi=q.tgt_mmsi, key=q.key + "|t")]))
    s = s.set_index("key")
    rows = []
    for r in q.itertuples():
        if r.key + "|o" not in s.index or r.key + "|t" not in s.index:
            continue
        o, t = s.loc[r.key + "|o"], s.loc[r.key + "|t"]
        rows.append({"key": r.key, **pair(o, t, dims(static, r.own_mmsi), dims(static, r.tgt_mmsi))})
    return pd.DataFrame(rows)


def cls(x: pd.Series) -> pd.Series:
    return pd.cut(x, [-np.inf, 100, 300, np.inf], right=False, labels=["high", "medium", "low"]).astype(str)


def main() -> None:
    static = load_static()
    d = pd.read_csv(ROOT / "data/processed/r2/modelling_r2.csv", low_memory=False)
    d["t"] = pd.to_datetime(d.meeting_time.astype(str).str[:19])
    has = d.nearest_third_vessel_mmsi.notna()
    q = pd.DataFrame({"key": d.event_id.astype(str), "own_mmsi": d.vessel_mmsi.astype("Int64").astype(str),
                      "tgt_mmsi": pd.to_numeric(d.nearest_third_vessel_mmsi, errors="coerce").astype("Int64").astype(str),
                      "t": d.t})[has]
    ev = run_pairs(q, static).rename(columns={"key": "event_id"})
    ev.to_csv(OUT / "event_hull_cpa.csv", index=False)

    # Existing pipeline columns, for verification of the headline figures.
    centre = pd.to_numeric(d.nearest_third_vessel_center_cpa_m, errors="coerce")
    hull = pd.to_numeric(d.nearest_third_vessel_hull_cpa_m, errors="coerce")
    n_all = len(d)
    rep = {
        "events": n_all, "with_third_vessel": int(has.sum()), "with_both_ais_fixes_at_meeting": int(len(ev)),
        "pipeline_columns_check": {
            "note": "nearest_third_vessel_hull_cpa_m is the CURRENT (meeting-time) rectangle clearance, not at TCPA; "
                    "centre - hull is therefore not a length correction",
            "median_centre_minus_hull_m": round(float((centre - hull).median()), 1),
            "hull_le_0": share(hull <= 0), "hull_lt_100": share(hull < 100)},
        "data_quality": {"own_heading_fallback_cog": share(ev.own_hdg_fallback), "tgt_heading_fallback_cog": share(ev.tgt_hdg_fallback),
                         "own_dims_missing": share(~ev.own_dims_ok), "tgt_dims_missing": share(~ev.tgt_dims_ok),
                         "tcpa_in_past_clipped_to_0": share(ev.tcpa_raw_min < 0)},
        "signed_delta_sd": {
            "median_m": round(float(ev.clr_length.median()), 1),
            "p05_p95_m": [round(float(ev.clr_length.quantile(.05)), 1), round(float(ev.clr_length.quantile(.95)), 1)],
            "lt_0": share(ev.clr_length < 0), "lt_100": share(ev.clr_length < 100),
            "median_length_correction_m": round(float((ev.d_cpa_centre_m - ev.clr_length).median()), 1)},
        "truncated_cpa_sd": {"median_m": round(float(ev.clr_length.clip(lower=0).median()), 1),
                             "eq_0": share(ev.clr_length <= 0), "lt_100": share(ev.clr_length.clip(lower=0) < 100)},
        "delta_sd_negative_polygon_check": {
            "events_delta_sd_lt_0": int((ev.clr_length < 0).sum()),
            "polygon_intersection": share(ev.polygon_intersects[ev.clr_length < 0]),
            "polygon_clearance_median_m": round(float(ev.clr_polygon[ev.clr_length < 0].median()), 1)},
        "variants_events": {}, "by_encounter": {}, "recommendations": {},
    }
    for v in VARIANTS:
        x = ev[f"clr_{v}"]
        rep["variants_events"][v] = {"median_m": round(float(x.median()), 1), "lt_0": share(x < 0), "lt_100": share(x < 100),
                                     "class": cls(x).value_counts().to_dict()}
    for enc, g in ev.groupby("encounter"):
        rep["by_encounter"][enc] = {"n": len(g), **{v: {"median_m": round(float(g[f"clr_{v}"].median()), 1),
                                                         "lt_100_pct": round(100 * float((g[f"clr_{v}"] < 100).mean()), 1)}
                                                     for v in VARIANTS},
                                     "polygon_intersection_pct": round(100 * float(g.polygon_intersects.mean()), 1)}

    # Decision-time recommendations: own ship at the decision time vs the nearest third vessel to the
    # recommended point (as summarize_nearby_traffic in the optimiser), re-classified per hull model.
    o = pd.concat(pd.read_csv(f, low_memory=False) for f in
                  glob.glob(str(ROOT / "data/processed/meeting_point/optimized_meeting_points_*_r2verified_causal.csv")))
    o = o[o.recommendation_status == "feasible"] if "recommendation_status" in o else o  # infeasible: diagnostics only
    c = pd.concat(pd.read_csv(f, low_memory=False, usecols=["event_id", "candidate_lat", "candidate_lon", "nearest_third_vessel_mmsi"])
                  for f in glob.glob(str(ROOT / "data/processed/meeting_point/candidates_*_r2verified_causal.csv")))
    rec = o.merge(c, left_on=["event_id", "recommended_lat", "recommended_lon"],
                  right_on=["event_id", "candidate_lat", "candidate_lon"], how="left").drop_duplicates("event_id")
    rec["t"] = pd.to_datetime(rec.rewind_time.astype(str).str[:19])
    hasr = rec.nearest_third_vessel_mmsi.notna()
    qr = pd.DataFrame({"key": rec.event_id.astype(str), "own_mmsi": rec.vessel_mmsi.astype("Int64").astype(str),
                       "tgt_mmsi": pd.to_numeric(rec.nearest_third_vessel_mmsi, errors="coerce").astype("Int64").astype(str),
                       "t": rec.t})[hasr]
    rr = run_pairs(qr, static).rename(columns={"key": "event_id"})
    rr.to_csv(OUT / "recommendation_hull_cpa.csv", index=False)
    n_rec = len(rec)
    base = rec.recommended_collision_risk_class.astype(str)
    rep["recommendations"] = {
        "n": n_rec, "with_third_vessel": int(hasr.sum()), "recomputed": int(len(rr)),
        "pipeline_domain_class_low": share(base == "low"),
        "pipeline_domain_class": base.value_counts().to_dict(),
        "definition": "pipeline class = domain clearance (centre CPA - two ship-domain radii, radius = half-diagonal + "
                      "max(0.25 L, 1.5 B)); <100 m high, <300 m medium, else low; no third vessel -> unknown",
    }
    for v in VARIANTS:
        k = cls(rr[f"clr_{v}"])
        # events with no third vessel / no fix keep the pipeline class (low when nothing is near)
        low = (k == "low").sum() + int((base[~rec.event_id.astype(str).isin(rr.event_id.astype(str))] == "low").sum())
        rep["recommendations"][f"low_{v}"] = {"n": int(low), "of": n_rec, "pct": round(100 * low / n_rec, 1)}
    (OUT / "cpa_hull_summary.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
