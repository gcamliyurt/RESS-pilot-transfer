"""RESS R2: diagnose the meeting-point optimiser on log-confirmed events.

Two questions:
  1. Coverage: why do only ~53% of confirmed events get a recommendation?
     Each missing event is traced to the first optimiser stage that drops it
     (same checks, in the same order, as optimize_meeting_points()).
  2. Offset: why is the recommended point a median ~1.3 km from the actual one?
     Offsets are broken down by candidate source, direction and gate status.
  3. After the fix (run_r2_opt_anchor.sh): coverage, offset of the ML anchor and
     of the recommendation, and the proximity-penalty (lambda) sensitivity.
Output : results/revision_r2_optimiser_diagnosis/
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
import os
TAG = os.environ.get("OPT_TAG", "anchor")  # anchor: adaptive run; causal: dispatch run
OUT = ROOT / "results" / ("revision_r2_optimiser_diagnosis" if TAG == "anchor" else f"revision_r2_optimiser_diagnosis_{TAG}")
OUT.mkdir(parents=True, exist_ok=True)
REWIND = pd.Timedelta(minutes=20)
WIN = pd.Timedelta(minutes=5)


def hav_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def week_of(path: str) -> str:
    return re.search(r"(\d{4}_\d{2}_week\d)", path).group(1)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(ROOT / "data/processed/r2/modelling_r2.csv", low_memory=False)
    wk = pd.concat([pd.read_csv(f, usecols=["event_id"]).assign(week=week_of(f))
                    for f in glob.glob(str(ROOT / "data/processed/interactions/event_context_enriched_*_r2verified.csv"))])
    d = d.merge(wk.drop_duplicates("event_id"), on="event_id", how="left")
    o = pd.concat([pd.read_csv(f, low_memory=False).assign(week=week_of(f))
                   for f in glob.glob(str(ROOT / "data/processed/meeting_point/optimized_meeting_points_*_r2verified.csv"))])
    o = o.merge(d[["event_id", "traffic_direction"]], on="event_id", suffixes=("_ais", ""))
    return d, o


def coverage(d: pd.DataFrame, done: set) -> pd.DataFrame:
    miss = d[~d.event_id.isin(done)].copy()
    miss["drop_stage"] = "other"
    miss.loc[miss.meeting_time.isna(), "drop_stage"] = "no_meeting_time"
    for week, g in miss[miss.drop_stage == "other"].groupby("week"):
        f = ROOT / "data/processed/ais_master/weeks" / (week.replace("_week", "-week").replace("_", "-", 1) + ".csv")
        if not f.exists():
            miss.loc[g.index, "drop_stage"] = "no_master_file"
            continue
        ids = set(g.pilot_mmsi.astype(str)) | set(g.vessel_mmsi.astype(str))
        m = pd.concat(c[c.mmsi.astype(str).isin(ids)] for c in pd.read_csv(f, usecols=["mmsi", "datetime"], chunksize=2_000_000))
        m["mmsi"] = m.mmsi.astype(str)
        m["datetime"] = pd.to_datetime(m.datetime)
        by = {k: v.datetime.values for k, v in m.groupby("mmsi")}
        for i, r in g.iterrows():
            t = pd.Timestamp(str(r.meeting_time)[:19]) - REWIND
            def has(mmsi):
                ts = by.get(str(mmsi))
                return ts is not None and bool(((ts >= np.datetime64(t - WIN)) & (ts <= np.datetime64(t + WIN))).any())
            p, v = has(r.pilot_mmsi), has(r.vessel_mmsi)
            miss.at[i, "drop_stage"] = ("no_fix_both" if not (p or v) else "no_pilot_fix_at_T-20" if not p
                                        else "no_vessel_fix_at_T-20" if not v else "empty_candidates_or_other")
    return miss


def main() -> None:
    d, o = load()
    if "--after" in __import__("sys").argv:
        rep = after_fix(d)
        (OUT / "diagnosis_after_fix.json").write_text(json.dumps(rep, indent=2, default=float))
        print(json.dumps(rep, indent=2, default=float))
        print(pd.read_csv(OUT / "lambda_sensitivity.csv").round(3).to_string())
        return
    o["recommended_gate_pass"] = o.recommended_gate_pass.astype(str).str.lower().eq("true")
    o["offset_m"] = hav_m(o.actual_meeting_lat, o.actual_meeting_lon, o.recommended_lat, o.recommended_lon)
    o["baseline_offset_m"] = hav_m(o.actual_meeting_lat, o.actual_meeting_lon, o.baseline_lat, o.baseline_lon)

    miss = coverage(d, set(o.event_id))
    cov = {
        "confirmed": int(len(d)), "optimised": int(d.event_id.isin(o.event_id).sum()),
        "drop_stage": miss.drop_stage.value_counts().to_dict(),
        "drop_stage_by_direction": miss.groupby("traffic_direction").drop_stage.value_counts().unstack(fill_value=0).to_dict("index"),
    }
    miss[["event_id", "week", "traffic_direction", "pilot_mmsi", "vessel_mmsi", "meeting_time", "drop_stage"]].to_csv(OUT / "missing_events.csv", index=False)

    def summ(g):
        return pd.Series({"n": len(g), "median_offset_m": g.offset_m.median(), "share_le_500m": (g.offset_m <= 500).mean(),
                          "gate_pass": g.recommended_gate_pass.mean()})
    off = {
        "overall": summ(o).to_dict(),
        "by_direction": o.groupby("traffic_direction").apply(summ).to_dict("index"),
        "by_source": o.groupby("recommended_source").apply(summ).sort_values("n", ascending=False).to_dict("index"),
        "by_gate_pass": o.assign(g=o.recommended_gate_pass).groupby("g").apply(summ).to_dict("index"),
        "baseline_median_offset_m": float(o.baseline_offset_m.median()),
        "top_gate_fail_reasons": o.loc[~o.recommended_gate_pass, "recommended_gate_reason"].str.split(";").explode().value_counts().head(10).to_dict(),
    }
    o.to_csv(OUT / "optimised_with_offsets.csv", index=False)
    rep = {"coverage": cov, "offset": off}
    (OUT / "diagnosis.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float))


def after_fix(d: pd.DataFrame) -> dict:
    files = glob.glob(str(ROOT / f"data/processed/meeting_point/optimized_meeting_points_*_r2verified_{TAG}.csv"))
    o = pd.concat(pd.read_csv(f, low_memory=False) for f in files)
    o = o.merge(d[["event_id", "traffic_direction"]], on="event_id", suffixes=("_ais", ""))
    o["recommended_gate_pass"] = o.recommended_gate_pass.astype(str).str.lower().eq("true")
    if "recommendation_status" in o:  # causal run: feasible recommendations only; infeasible are diagnostics
        o = o[o.recommendation_status == "feasible"]
    o["offset_m"] = hav_m(o.actual_meeting_lat, o.actual_meeting_lon, o.recommended_lat, o.recommended_lon)
    o["anchor_offset_m"] = hav_m(o.actual_meeting_lat, o.actual_meeting_lon, o.anchor_lat, o.anchor_lon)

    def summ(g):
        return pd.Series({"n": len(g), "coverage_of_confirmed": np.nan,
                          "anchor_median_offset_m": g.anchor_offset_m.median(), "anchor_le_500m": (g.anchor_offset_m <= 500).mean(),
                          "median_offset_m": g.offset_m.median(), "share_le_500m": (g.offset_m <= 500).mean(),
                          "median_move_from_anchor_m": g.recommended_dist_to_anchor_m.median(),
                          "kept_anchor": (g.recommended_source == "ml_anchor").mean(),
                          "gate_pass": g.recommended_gate_pass.mean(), "lead_min_median": g.lead_min.median()})
    by_dir = o.groupby("traffic_direction").apply(summ)
    by_dir["coverage_of_confirmed"] = by_dir.n / d.traffic_direction.value_counts()
    rep = {"overall": summ(o).to_dict() | {"coverage_of_confirmed": len(o) / len(d)},
           "by_direction": by_dir.to_dict("index"),
           "by_source": o.recommended_source.value_counts().to_dict(),
           "top_gate_fail_reasons": o.loc[~o.recommended_gate_pass, "recommended_gate_reason"].str.split(";").explode().value_counts().head(10).to_dict()}

    # Lambda sensitivity from the saved candidates (same feasible set, re-ranked offline).
    rows = []
    for f in glob.glob(str(ROOT / f"data/processed/meeting_point/candidates_*_r2verified_{TAG}.csv")):
        c = pd.read_csv(f, low_memory=False)
        if "safety_score" not in c.columns:  # dumps written before the penalty columns were logged
            c["safety_score"] = c["candidate_score"]
            c = c.merge(o[["event_id", "anchor_lat", "anchor_lon"]], on="event_id")
            c["dist_to_anchor_m"] = hav_m(c.anchor_lat, c.anchor_lon, c.candidate_lat, c.candidate_lon)
        c = c[["event_id", "candidate_source", "candidate_lat", "candidate_lon", "gate_pass", "safety_score", "dist_to_anchor_m"]]
        rows.append(c[c.gate_pass.astype(str).str.lower().eq("true")])
    c = pd.concat(rows).merge(o[["event_id", "actual_meeting_lat", "actual_meeting_lon", "traffic_direction"]], on="event_id")
    c["d_actual"] = hav_m(c.actual_meeting_lat, c.actual_meeting_lon, c.candidate_lat, c.candidate_lon)
    sens = []
    for lam in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        b = c.assign(obj=c.safety_score - lam * c.dist_to_anchor_m / 1000).sort_values("obj", ascending=False).groupby("event_id").head(1)
        for direc, g in [("all", b)] + list(b.groupby("traffic_direction")):
            sens.append({"lambda": lam, "direction": direc, "n": len(g), "median_offset_m": g.d_actual.median(),
                         "share_le_500m": (g.d_actual <= 500).mean(), "median_move_from_anchor_m": g.dist_to_anchor_m.median(),
                         "mean_safety_score": g.safety_score.mean(), "kept_anchor": (g.candidate_source == "ml_anchor").mean()})
    pd.DataFrame(sens).to_csv(OUT / "lambda_sensitivity.csv", index=False)
    o.to_csv(OUT / "optimised_anchor_with_offsets.csv", index=False)
    return rep


if __name__ == "__main__":
    main()
