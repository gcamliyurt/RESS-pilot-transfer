"""RESS R2 (causal): decision trigger observable online, and inputs taken only at or before it.

Trigger  t_trig = first AIS fix of the ship within RING_M of the New Port breakwater lights in the
         visit that ends with the transfer (a visit starts after a gap of >= VISIT_GAP in the ship's
         AIS). Inbound ships cross the ring on approach; outbound ships appear inside it when they
         leave the berths (which lie outside the AIS extent). The observed transfer time T is used
         only to select the event and as the outcome, never as an input.
Inputs   ship state = the trigger fix itself; traffic = each third vessel's latest fix in
         [t_trig - TRAFFIC_WIN, t_trig] (service craft excluded); wind/wave = hourly series at the hour
         containing t_trig; hour and month of t_trig (KST); draught and type from the ship's static data.
         Pilot-boat position = its latest fix in [t_trig - PILOT_WIN, t_trig], else none (the optimiser
         then starts the pilot leg at the breakwater gate).
Every timestamp used is <= t_trig (asserted).
Output   data/processed/r2/causal_features_r2.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.detect_interceptions import service_craft_mmsi
from src.core.enrich_event_context import BREAKWATER_LIGHTS, _read_weather_hourly

ROOT = Path(__file__).parent
R2 = ROOT / "data" / "processed" / "r2"
WEEKS = ROOT / "data" / "processed" / "ais_master" / "weeks"
RING_M = 13000.0
VISIT_GAP = pd.Timedelta(minutes=60)
LOOKBACK = pd.Timedelta(hours=3)
TRAFFIC_WIN = pd.Timedelta(minutes=3)
PILOT_WIN = pd.Timedelta(minutes=30)
NM = 1852.0
K = np.cos(np.radians(35.05)) * 111320.0


def _bw_dist(lat, lon) -> np.ndarray:
    """Distance (m) to the nearest breakwater light, the target definition."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    d = [np.hypot((lon - lo) * K, (lat - la) * 111320.0) for la, lo in zip(BREAKWATER_LIGHTS.lat, BREAKWATER_LIGHTS.lon)]
    return np.min(d, axis=0)


def _gate_xy(lat, lon):
    return (np.asarray(lon, float) - BREAKWATER_LIGHTS.lon.mean()) * K, (np.asarray(lat, float) - BREAKWATER_LIGHTS.lat.mean()) * 111320.0


def _week_file(t: pd.Timestamp) -> Path:
    return WEEKS / f"{t:%Y-%m}-week{min((t.day - 1) // 7 + 1, 5)}.csv"


def _load(files: list[Path], t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for f in files:
        if not f.exists():
            continue
        for ch in pd.read_csv(f, chunksize=2_000_000, usecols=["mmsi", "datetime", "lat", "lon", "spd", "crs"]):
            ch["datetime"] = pd.to_datetime(ch["datetime"], errors="coerce")
            ch = ch[(ch.datetime >= t0) & (ch.datetime <= t1)]
            if len(ch):
                parts.append(ch)
    a = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["mmsi", "datetime", "lat", "lon", "spd", "crs"])
    a["mmsi"] = a["mmsi"].astype("int64").astype(str)
    return a


def main() -> None:
    d = pd.read_csv(R2 / "modelling_r2.csv", low_memory=False)
    d["T"] = pd.to_datetime(d.meeting_time.astype(str).str[:19])  # KST wall-clock
    d["wk"] = d["T"].map(_week_file)
    service = set(map(str, service_craft_mmsi()))
    w = _read_weather_hourly()
    w = w.assign(h=pd.to_datetime(w.meeting_hour).dt.tz_localize(None)).drop_duplicates("h").set_index("h")

    rows = []
    for wk, ev in d.groupby("wk"):
        t0, t1 = ev["T"].min() - LOOKBACK, ev["T"].max()
        files = sorted({wk, _week_file(t0)})
        a = _load(files, t0, t1)
        by = {m: g.sort_values("datetime") for m, g in a.groupby("mmsi")}
        third_all = a[~a.mmsi.isin(service)]
        for e in ev.itertuples():
            s = by.get(str(e.vessel_mmsi))
            if s is None:
                continue
            s = s[(s.datetime >= e.T - LOOKBACK) & (s.datetime <= e.T)]
            if s.empty:
                continue
            gaps = s.datetime.diff() > VISIT_GAP
            visit = s[gaps.cumsum() == gaps.sum()]                      # last visit before T
            inside = visit[_bw_dist(visit.lat, visit.lon) <= RING_M]
            if inside.empty:
                continue
            fix = inside.iloc[0]
            t = fix.datetime
            x, y = _gate_xy(fix.lat, fix.lon)
            tw = third_all[(third_all.datetime >= t - TRAFFIC_WIN) & (third_all.datetime <= t) & (third_all.mmsi != str(e.vessel_mmsi))]
            tw = tw.sort_values("datetime").drop_duplicates("mmsi", keep="last")
            ox, oy = _gate_xy(tw.lat, tw.lon)
            dist = np.hypot(ox - x, oy - y)
            p = by.get(str(e.pilot_mmsi))
            p = p[(p.datetime >= t - PILOT_WIN) & (p.datetime <= t)] if p is not None else None
            pfix = p.iloc[-1] if p is not None and len(p) else None
            wx = w.loc[t.floor("h")] if t.floor("h") in w.index else None
            used = [fix.datetime] + list(tw.datetime) + ([pfix.datetime] if pfix is not None else [])
            assert max(used) <= t, "input observed after the trigger"
            rows.append({
                "event_id": e.event_id, "t_trig": t, "trigger_kind": "ring_crossing" if len(inside) < len(visit) else "first_appearance",
                "trigger_to_meeting_min": (e.T - t).total_seconds() / 60,   # outcome-side diagnostic, never a predictor
                "ref_dist_to_breakwater_m": float(np.hypot(x, y)),
                "ref_bearing_from_breakwater_sin": float(np.sin(np.arctan2(x, y))),
                "ref_bearing_from_breakwater_cos": float(np.cos(np.arctan2(x, y))),
                "ref_sog_kn": float(fix.spd) if pd.notna(fix.spd) else np.nan,
                "ref_cog_sin": float(np.sin(np.radians(fix.crs))) if pd.notna(fix.crs) and fix.crs < 360 else np.nan,
                "ref_cog_cos": float(np.cos(np.radians(fix.crs))) if pd.notna(fix.crs) and fix.crs < 360 else np.nan,
                "traffic_0_5nm": int((dist <= 0.5 * NM).sum()), "traffic_1nm": int((dist <= NM).sum()),
                "traffic_2nm": int((dist <= 2 * NM).sum()), "nearest_third_m": float(dist.min()) if len(dist) else np.nan,
                "wind_speed_mps": float(wx.wind_speed_mps) if wx is not None else np.nan,
                "wave_height_m": float(wx.wave_height_m) if wx is not None else np.nan,
                "tide_trig_m": float(wx.tide_height_m) if wx is not None and "tide_height_m" in wx else np.nan,
                "hour_kst": t.hour, "month": t.month,
                "vessel_draft_m": e.vessel_draft_m, "vessel_ship_type_class": e.vessel_ship_type_class,
                "pilot_lat": float(pfix.lat) if pfix is not None else np.nan,
                "pilot_lon": float(pfix.lon) if pfix is not None else np.nan,
                "ship_lat": float(fix.lat), "ship_lon": float(fix.lon),
            })
        print(f"{Path(wk).name}: {sum(1 for r in rows if r['t_trig'] >= t0)} / {len(ev)}", flush=True)
    out = pd.DataFrame(rows)
    out = out[out.trigger_to_meeting_min > 0]
    out.to_csv(R2 / "causal_features_r2.csv", index=False)
    dd = out.merge(d[["event_id", "traffic_direction"]], on="event_id")
    print(dd.groupby("traffic_direction").agg(n=("event_id", "size"), lead_median=("trigger_to_meeting_min", "median"),
                                             ring=("trigger_kind", lambda s: (s == "ring_crossing").mean())).round(2))


if __name__ == "__main__":
    main()
