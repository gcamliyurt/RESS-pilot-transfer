"""RESS R2: decision-time UKC features (no meeting-point information).

UKC at the observed meeting point leaks the target (it is only known once the
meeting position is known). These features use only what is known at t_dec:
the ship's declared draught, the tide at t_dec and ENC depth along the ship's
approach line (from the breakwater gate through the ship's t_dec position).

  ukc_ref_m          UKC at the ship's t_dec position
  ukc2_boundary_m    distance from the gate, along the approach line, beyond which
  ukc4_boundary_m    UKC stays >= 2 m / >= 4 m (the depth-feasible boarding limit)
  tide_dec_m         predicted tide at t_dec

Output: data/processed/r2/ukc_features_r2_adaptive.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.enrich_event_context import (
    BREAKWATER_LIGHTS,
    _load_bathymetry_sampler,
    _read_weather_hourly,
    _sample_bathymetry,
)

ROOT = Path(__file__).parent
R2 = ROOT / "data" / "processed" / "r2"
STEP_M, MAX_M = 100.0, 15000.0
GATE = (float(BREAKWATER_LIGHTS["lat"].mean()), float(BREAKWATER_LIGHTS["lon"].mean()))
K = np.cos(np.radians(35.05)) * 111320.0


def _point(brg_deg: float, dist_m: float) -> tuple[float, float]:
    b = np.radians(brg_deg)
    return GATE[0] + dist_m * np.cos(b) / 111320.0, GATE[1] + dist_m * np.sin(b) / K


def _boundary(depth: np.ndarray, dist: np.ndarray, tide: float, draft: float, ukc_min: float) -> float:
    """Smallest distance beyond which UKC >= ukc_min at every sampled point."""
    ok = (depth + (0.0 if np.isnan(tide) else tide) - draft) >= ukc_min
    ok[np.isnan(depth)] = True  # unsampled cells do not block
    bad = np.where(~ok)[0]
    return float(dist[0]) if len(bad) == 0 else float(dist[min(bad[-1] + 1, len(dist) - 1)])


def main() -> None:
    import os
    if os.environ.get("DECISION", "adaptive") == "dispatch":  # pilot-boat dispatch decision time
        f = pd.read_csv(R2 / "dispatch_features_r2.csv", parse_dates=["t_dispatch"]).rename(columns={"t_dispatch": "t_dec"})
        f = f[f.ship_visible == 1]
    else:
        f = pd.read_csv(R2 / "apriori_features_r2_adaptive.csv", parse_dates=["t_dec"])
    sampler, source = _load_bathymetry_sampler()
    w = _read_weather_hourly()
    tide = w.set_index("meeting_hour")["tide_height_m"] if "tide_height_m" in w.columns else pd.Series(dtype=float)
    tide = tide[~tide.index.duplicated()]

    f["ship_brg"] = np.degrees(np.arctan2(f.ref_bearing_from_breakwater_sin, f.ref_bearing_from_breakwater_cos)) % 360
    dist = np.arange(STEP_M, MAX_M + STEP_M, STEP_M)
    profiles: dict[int, np.ndarray] = {}  # depth profile per whole-degree bearing, sampled once
    for b in sorted(f.ship_brg.round().astype(int).unique()):
        profiles[b] = np.array([_sample_bathymetry(*_point(b, d), sampler) for d in dist])

    rows = []
    for e in f.itertuples():
        t = pd.Timestamp(e.t_dec).tz_localize("UTC").floor("h")  # KST wall-clock labelled UTC, as the weather table
        td = float(tide.get(t, np.nan))
        draft = float(e.vessel_draft_m) if pd.notna(e.vessel_draft_m) else np.nan
        x, y = e.ref_dist_to_breakwater_m * e.ref_bearing_from_breakwater_sin, e.ref_dist_to_breakwater_m * e.ref_bearing_from_breakwater_cos
        d_ref = _sample_bathymetry(GATE[0] + y / 111320.0, GATE[1] + x / K, sampler)
        prof = profiles[int(round(e.ship_brg)) % 360] if int(round(e.ship_brg)) in profiles else profiles[int(round(e.ship_brg))]
        rows.append({
            "event_id": e.event_id,
            "tide_dec_m": td,
            "ukc_ref_m": (d_ref + (0 if np.isnan(td) else td) - draft) if pd.notna(draft) else np.nan,
            "ukc2_boundary_m": _boundary(prof, dist, td, draft, 2.0) if pd.notna(draft) else np.nan,
            "ukc4_boundary_m": _boundary(prof, dist, td, draft, 4.0) if pd.notna(draft) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(R2 / ("ukc_features_r2_dispatch.csv" if os.environ.get("DECISION", "adaptive") == "dispatch" else "ukc_features_r2_adaptive.csv"), index=False)
    print(f"depth source: {source}; {len(out)} events")
    print(out.describe().round(2).to_string())


if __name__ == "__main__":
    main()
