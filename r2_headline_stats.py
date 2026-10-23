"""RESS R2: headline statistics for the revised manuscript and response letter.

Reads the log-confirmed dataset, the decision-time (leakage-free) re-optimised
meeting points (run_r2_opt_anchor.sh) and the adaptive-decision-time model metrics, and writes
results/revision_r2_headline.json. Every count states its denominator
(Reviewer #4 C3); CPA-SD is reported both signed and truncated (Reviewer #5 C1).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
R2 = ROOT / "data" / "processed" / "r2"
K = np.cos(np.radians(35.05)) * 111000.0


def _share(mask: pd.Series) -> dict:
    mask = mask.dropna().astype(bool)
    return {"n": int(mask.sum()), "of": int(len(mask)), "pct": round(100.0 * float(mask.mean()), 1)}


def main() -> None:
    d = pd.read_csv(R2 / "modelling_r2.csv", low_memory=False)
    det = json.loads((R2 / "detection_validation_r2.json").read_text())
    O = pd.concat([pd.read_csv(f, low_memory=False)
                   for f in glob.glob(str(ROOT / "data/processed/meeting_point/optimized_meeting_points_*_r2verified_causal.csv"))])
    O = O[O.recommendation_status == "feasible"] if "recommendation_status" in O else O  # infeasible: diagnostics only
    o = O.merge(d[["event_id", "traffic_direction"]], on="event_id", suffixes=("_ais", ""))

    centre = pd.to_numeric(d.nearest_third_vessel_center_cpa_m, errors="coerce")
    hull = pd.to_numeric(d.nearest_third_vessel_hull_cpa_m, errors="coerce")
    rec_vs_act = np.hypot((o.recommended_lat - o.actual_meeting_lat) * 111000.0,
                          (o.recommended_lon - o.actual_meeting_lon) * K)
    ukc_rec = pd.to_numeric(o.recommended_ukc_m, errors="coerce")
    ukc_obs = pd.to_numeric(d.ukc_m, errors="coerce")

    out = {
        "detection": det,
        "dataset": {
            "log_confirmed_transfers": int(len(d)),
            "vessels": int(d.vessel_mmsi.nunique()),
            "by_direction": d.traffic_direction.value_counts().to_dict(),
            "median_meeting_dist_to_breakwater_m": d.groupby("traffic_direction").dist_to_breakwater_m.median().round(0).to_dict(),
        },
        "observed_conditions_at_real_transfers": {
            "ukc_ge_2m": _share(ukc_obs >= 2.0),
            "ukc_median_m": round(float(ukc_obs.median()), 1),
            "wind_median_mps": round(float(d.wind_speed_mps.median()), 1),
            "wave_median_m": round(float(d.wave_height_m.median()), 2),
        },
        "cpa_sd": {
            "n": int(hull.notna().sum()),
            "median_length_correction_m": round(float((centre - hull).median()), 0),
            "signed_hull_cpa_le_0": _share(hull <= 0),
            "hull_cpa_lt_100m": _share(hull < 100),
            "centre_cpa_lt_100m": _share(centre < 100),
        },
        "optimisation": {
            "optimised_of_confirmed": _share(d.event_id.isin(o.event_id)),
            "recommended_low_collision_risk": _share(o.recommended_collision_risk_class.astype(str) == "low"),
            "recommended_ukc_ge_2m": _share(ukc_rec >= 2.0),
            "recommendation_within_500m_of_actual": _share(rec_vs_act <= 500),
            "recommendation_median_offset_from_actual_m": round(float(np.nanmedian(rec_vs_act)), 0),
            "recommended_source": o.recommended_source.value_counts().to_dict(),
            "action": o.recommendation_action.value_counts().to_dict(),
            "recommended_gate_pass": _share(o.recommended_gate_pass.astype(str).str.lower() == "true"),
            "waypoint_mode": o.recommended_waypoint_source.value_counts().to_dict(),
            "lead_min_median_by_direction": o.groupby("traffic_direction").lead_min.median().round(1).to_dict(),
            "anchor_median_offset_from_actual_m": o.assign(a=np.hypot((o.anchor_lat - o.actual_meeting_lat) * 111000.0,
                                                                      (o.anchor_lon - o.actual_meeting_lon) * K))
                                                   .groupby("traffic_direction").a.median().round(0).to_dict(),
            "median_move_from_anchor_m": o.groupby("traffic_direction").recommended_dist_to_anchor_m.median().round(0).to_dict(),
            "optimised_by_direction": {k: _share(d[d.traffic_direction == k].event_id.isin(o.event_id)) for k in ("inbound", "outbound")},
            "by_direction_median_offset_m": pd.Series(rec_vs_act.values, index=o.traffic_direction.values)
                                              .groupby(level=0).median().round(0).to_dict(),
        },
    }
    vm = ROOT / "results" / "revision_r2_logconfirmed_adaptive" / "validation_metrics.csv"
    if vm.exists():
        m = pd.read_csv(vm)
        out["models_adaptive"] = m[m.model.isin(["historical_mean", "linear", "random_forest", "grad_boost", "hybrid", "proposed",
                                                 "historical_mean_by_route", "proposed_oracle"])][
            ["direction", "scheme", "model", "r2", "mae_m", "rmse_m"]].round(3).to_dict("records")
    gates = R2 / "gate_thresholds.json"
    if gates.exists():
        out["gate_calibration"] = json.loads(gates.read_text())
    p = ROOT / "results" / "revision_r2_headline.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
