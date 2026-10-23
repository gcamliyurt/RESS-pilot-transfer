"""RESS R2: calibrate the optimiser's safety gates on real, completed transfers.

A gate is relaxed only where its default rejects more than (1 - PASS_RATE) of the
log-confirmed pilot transfers (data/processed/r2/modelling_r2.csv): a "minimum
clearance" gate then moves down to the (1 - PASS_RATE) quantile of observed values,
a "maximum distance" gate up to the PASS_RATE quantile. Defaults are never tightened. Route metrics are measured along the real pilot-boat and ship
tracks up to the meeting. UKC < 2 m stays a fixed physical rule, not calibrated.
This answers Reviewer #4 C1 (how operational thresholds were derived).

Output: data/processed/r2/gate_thresholds.json (read by optimize_meeting_point.py)
--folds OOF.csv: also data/processed/r2/gate_thresholds_folds.json, one threshold set per grouped-CV
fold calibrated only on transfers outside that fold (so no evaluated event informs its own gates),
and gate_calibration_metrics.csv (per-transfer observed values).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.optimize_meeting_point import (
    GATE_DEFAULTS,
    _load_breakwater_geometries,
    _load_water_polygons,
    _read_buoys,
    _route_option_metrics,
)

ROOT = Path(__file__).parent
R2 = ROOT / "data" / "processed" / "r2"
PASS_RATE = 0.95


def _tracks() -> dict[str, dict]:
    out = {}
    for f in glob.glob(str(ROOT / "data/processed/interactions/event_trajectories_*_r2verified.json")):
        out.update(json.load(open(f)))
    return out


def _until(track: list[dict], t_end: pd.Timestamp) -> list[tuple[float, float]]:
    pts = [(p["lat"], p["lon"]) for p in track if pd.Timestamp(p["timestamp"]) <= t_end]
    return [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]


def main() -> None:
    d = pd.read_csv(R2 / "modelling_r2.csv", low_memory=False)
    tr = _tracks()
    buoys, breakwaters, water = _read_buoys(), _load_breakwater_geometries(), _load_water_polygons()

    rows = []
    for r in d.itertuples():
        t = tr.get(str(r.event_id))
        if not t:
            continue
        t_meet = pd.Timestamp(r.meeting_time)
        ves, pil = _until(t["vessel_track"], t_meet), _until(t["pilot_track"], t_meet)
        if len(ves) < 2 or len(pil) < 2:
            continue
        m = _route_option_metrics(
            pilot_lat=pil[0][0], pilot_lon=pil[0][1], vessel_lat=ves[0][0], vessel_lon=ves[0][1],
            cand_lat=float(r.meeting_lat), cand_lon=float(r.meeting_lon), waypoint=None,
            buoys=buoys, breakwaters=breakwaters, water_polygons=water,
            vessel_track=ves, pilot_track=pil,
        )
        rows.append({"event_id": r.event_id, **{k: m[k] for k in (
            "route_min_buoy_clearance_m", "route_min_breakwater_clearance_m", "route_water_coverage_ratio")}})
    R = d.merge(pd.DataFrame(rows), on="event_id", how="left")
    R.to_csv(R2 / "gate_calibration_metrics.csv", index=False)
    out = _calibrate(R)
    (R2 / "gate_thresholds.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["thresholds"], indent=2))
    if "--folds" in sys.argv:
        oof = pd.read_csv(sys.argv[sys.argv.index("--folds") + 1])
        fold_of = oof.set_index("event_id").fold
        R["fold"] = R.event_id.map(fold_of)
        folds = {str(int(k)): _calibrate(R[R.fold != k])["thresholds"] for k in sorted(R.fold.dropna().unique())}
        (R2 / "gate_thresholds_folds.json").write_text(json.dumps({"pass_rate": PASS_RATE, "folds": folds}, indent=2))
        print(json.dumps(folds, indent=1))


def _calibrate(R: pd.DataFrame) -> dict:
    q_lo, q_hi = 1.0 - PASS_RATE, PASS_RATE
    in_fw = R["in_fairway"].astype(str).str.lower().isin(["true", "1", "1.0"])
    obs = {
        "collision_domain_cpa_min_m": ("min", R["nearest_third_vessel_cpa_m"]),
        "buoy_min_m": ("min", R["dist_to_nearest_buoy_m"]),
        "breakwater_min_m": ("min", R["dist_to_breakwater_m"]),
        "fairway_edge_max_m": ("max", R.loc[~in_fw, "dist_to_fairway_edge_m"]),
        "route_buoy_clearance_min_m": ("min", R["route_min_buoy_clearance_m"]),
        "route_breakwater_clearance_min_m": ("min", R["route_min_breakwater_clearance_m"]),
        "route_water_coverage_min": ("min", R["route_water_coverage_ratio"]),
    }
    thresholds, report = {}, {}
    for k, (kind, s) in obs.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 50:
            report[k] = {"n": int(len(s)), "kept_default": GATE_DEFAULTS[k]}
            continue
        q = float(s.quantile(q_lo if kind == "min" else q_hi))
        v = min(q, GATE_DEFAULTS[k]) if kind == "min" else max(q, GATE_DEFAULTS[k])
        thresholds[k] = round(v, 3)
        default_pass = float((s >= GATE_DEFAULTS[k]).mean() if kind == "min" else (s <= GATE_DEFAULTS[k]).mean())
        new_pass = float((s >= v).mean() if kind == "min" else (s <= v).mean())
        report[k] = {"n": int(len(s)), "default": GATE_DEFAULTS[k], "calibrated": round(v, 3),
                     "real_transfers_passing_default": round(default_pass, 3),
                     "real_transfers_passing_calibrated": round(new_pass, 3)}
    ukc = pd.to_numeric(R["ukc_m"], errors="coerce")
    report["ukc_min_m (fixed rule 2.0)"] = {"real_transfers_passing": round(float((ukc >= 2.0).mean()), 3)}

    return {"pass_rate": PASS_RATE, "n_transfers": int(len(R)), "thresholds": thresholds, "report": report}


if __name__ == "__main__":
    main()
