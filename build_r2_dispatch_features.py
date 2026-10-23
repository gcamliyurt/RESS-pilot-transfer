"""RESS R2 (causal): decision at pilot-boat dispatch, inputs only at or before it.

Dispatch t_d (v2) = within the window from the end of the same pilot boat's previous transfer
session (any of the 4,599 detected sessions; at most LOOKBACK before the transfer) to the start of
this session, the first fix above DISPATCH_KN after the boat's last waiting fix (SOG <= WAIT_KN)
that heads for this ship: course within MAX_HEADING_OFF of the bearing to the ship if the ship is
visible, otherwise heading seaward (away from the breakwater gate). Pilot boats run back-to-back
jobs; without the previous-session bound the first movement in the window was often the end of an
earlier job (v1, kept as dispatch_features_r2_v1.csv). The pilot station dispatches the boat to a known ship, so this is an
operational decision point; the transfer time T only selects the event and is the outcome.
Inputs at t_d (latest observation at or before t_d, at most MAX_AGE old): ship and pilot-boat
position, speed and course; range and bearing between them; the kinematic intercept (where and
when the boat meets the ship if the ship keeps course and speed and the boat runs at its speed,
at least MIN_PILOT_KN); unique third vessels around the ship (service craft excluded); wind and
wave of the hour containing t_d; hour and month of t_d; draught and ship type.
Every timestamp used is <= t_d (asserted).
Output: data/processed/r2/dispatch_features_r2.csv
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
LOOKBACK = pd.Timedelta(minutes=90)
WAIT_KN, DISPATCH_KN, MIN_PILOT_KN = 2.0, 5.0, 12.0
MAX_AGE = pd.Timedelta(minutes=3)
MAX_HEADING_OFF = 60.0
import os as _os
DISPATCH_RULE = _os.environ.get("DISPATCH_RULE", "v3")
# DISPATCH_WINDOW=prev: the search starts at the end of the boat's previous session only (no bound
# derived from the transfer time T); events whose previous session ended more than MAX_IDLE before
# T are left out of this sensitivity sample, the trigger itself never uses T.
DISPATCH_WINDOW = _os.environ.get("DISPATCH_WINDOW", "T90")
MAX_IDLE = pd.Timedelta(hours=12)
TRAFFIC_WIN = pd.Timedelta(minutes=3)
NM = 1852.0
KN = 0.514444
K = np.cos(np.radians(35.05)) * 111320.0
GX, GY = float(BREAKWATER_LIGHTS.lon.mean()), float(BREAKWATER_LIGHTS.lat.mean())


def _xy(lat, lon):
    return (np.asarray(lon, float) - GX) * K, (np.asarray(lat, float) - GY) * 111320.0


def _bw_dist_xy(x, y) -> float:
    lx, ly = _xy(BREAKWATER_LIGHTS.lat, BREAKWATER_LIGHTS.lon)
    return float(np.min(np.hypot(lx - x, ly - y)))


def _intercept(sx, sy, svx, svy, px, py, vp):
    """Time (s) and point at which a boat at P with speed vp meets a ship at S moving with V."""
    dx, dy = sx - px, sy - py
    a = svx * svx + svy * svy - vp * vp
    b = 2 * (svx * dx + svy * dy)
    c = dx * dx + dy * dy
    if abs(a) < 1e-9:
        ts = [-c / b] if b != 0 else []
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return np.nan, np.nan, np.nan
        r = np.sqrt(disc)
        ts = [(-b - r) / (2 * a), (-b + r) / (2 * a)]
    ts = [t for t in ts if t > 0]
    if not ts:
        return np.nan, np.nan, np.nan
    t = min(ts)
    return t, sx + svx * t, sy + svy * t


def _week_file(t: pd.Timestamp) -> Path:
    return WEEKS / f"{t:%Y-%m}-week{min((t.day - 1) // 7 + 1, 5)}.csv"


def _load(files, t0, t1) -> pd.DataFrame:
    parts = []
    for f in files:
        if not Path(f).exists():
            continue
        for ch in pd.read_csv(f, chunksize=2_000_000, usecols=["mmsi", "datetime", "lat", "lon", "spd", "crs"]):
            ch["datetime"] = pd.to_datetime(ch["datetime"], errors="coerce")
            ch = ch[(ch.datetime >= t0) & (ch.datetime <= t1)]
            if len(ch):
                parts.append(ch)
    a = pd.concat(parts, ignore_index=True)
    a["mmsi"] = a["mmsi"].astype("int64").astype(str)
    return a


def _heads_for(pf: pd.Series, s: pd.DataFrame | None) -> bool:
    """Pilot boat heading towards the ship (if visible) or seaward from the breakwater gate."""
    if pd.isna(pf.crs) or pf.crs >= 360:
        return True
    c = np.radians(pf.crs)
    px, py = _xy(pf.lat, pf.lon)
    sf = _asof(s, pf.datetime) if s is not None else None
    if sf is not None:
        sx, sy = _xy(sf.lat, sf.lon)
        brg = np.degrees(np.arctan2(sx - px, sy - py)) % 360
        return abs((pf.crs - brg + 180) % 360 - 180) <= MAX_HEADING_OFF
    return float(np.sin(c) * px + np.cos(c) * py) > 0  # moving away from the gate


def _dispatch(p: pd.DataFrame, s: pd.DataFrame | None) -> pd.Series | None:
    """v3 (online): the FIRST fix above DISPATCH_KN heading for the ship after the window start.
    v2 took the first such fix after the boat's LAST waiting fix before the session, which needs
    hindsight (a live system cannot know which wait is the last)."""
    if DISPATCH_RULE == "v2":
        waiting = p[p.spd <= WAIT_KN]
        after = p[p.datetime > waiting.datetime.max()] if len(waiting) else p
    else:
        after = p
    for _, pf in after[after.spd > DISPATCH_KN].iterrows():
        if _heads_for(pf, s):
            return pf
    return None


def _asof(g: pd.DataFrame, t) -> pd.Series | None:
    h = g[(g.datetime <= t) & (g.datetime >= t - MAX_AGE)]
    return h.iloc[-1] if len(h) else None


def main() -> None:
    d = pd.read_csv(R2 / "modelling_r2.csv", low_memory=False)
    d["T"] = pd.to_datetime(d.meeting_time.astype(str).str[:19])  # KST wall-clock
    d["wk"] = d["T"].map(_week_file)
    service = set(map(str, service_craft_mmsi()))
    w = _read_weather_hourly()
    w = w.assign(h=pd.to_datetime(w.meeting_hour).dt.tz_localize(None)).drop_duplicates("h").set_index("h")
    import glob
    S = pd.concat([pd.read_csv(f, usecols=["event_id", "pilot_mmsi", "start_time", "end_time"])
                   for f in glob.glob(str(ROOT / "data/processed/interactions/event_context_enriched_*_r2verified.csv"))])
    for c in ("start_time", "end_time"):
        S[c] = pd.to_datetime(S[c].astype(str).str[:19])
    S["pilot_mmsi"] = S.pilot_mmsi.astype("int64").astype(str)
    sess = {m: g.sort_values("end_time") for m, g in S.groupby("pilot_mmsi")}
    own_start = S.set_index("event_id").start_time

    rows, n_no_dispatch, n_no_ship = [], 0, 0
    for wk, ev in d.groupby("wk"):
        t0, t1 = ev["T"].min() - (MAX_IDLE if DISPATCH_WINDOW == "prev" else LOOKBACK), ev["T"].max()
        a = _load(sorted({wk, _week_file(t0)}), t0, t1)
        by = {m: g.sort_values("datetime") for m, g in a.groupby("mmsi")}
        third_all = a[~a.mmsi.isin(service)]
        for e in ev.itertuples():
            p = by.get(str(e.pilot_mmsi))
            s = by.get(str(e.vessel_mmsi))
            if p is None or s is None:
                n_no_dispatch += 1
                continue
            t_end = min(own_start.get(e.event_id, e.T), e.T)
            prev = sess.get(str(e.pilot_mmsi))
            prev = prev[prev.end_time < t_end].end_time.max() if prev is not None else pd.NaT
            if DISPATCH_WINDOW == "prev":
                if pd.isna(prev) or prev < e.T - MAX_IDLE:
                    n_no_dispatch += 1
                    continue
                t_start = prev
            else:
                t_start = max(e.T - LOOKBACK, prev) if pd.notna(prev) else e.T - LOOKBACK
            p = p[(p.datetime > t_start) & (p.datetime <= t_end)]
            pf = _dispatch(p, s) if len(p) else None
            if pf is None:
                n_no_dispatch += 1
                continue
            t = pf.datetime
            sf = _asof(s, t)
            px, py = _xy(pf.lat, pf.lon)
            if sf is None:
                # Ship not yet in AIS range at dispatch (still beyond the AIS extent / at the berth):
                # known online and informative; ship features left empty and imputed in training folds.
                n_no_ship += 1
                rows.append({"event_id": e.event_id, "t_dispatch": t, "ship_visible": 0,
                             "dispatch_to_meeting_min": (e.T - t).total_seconds() / 60,
                             "pilot_dist_to_breakwater_m": float(np.hypot(px, py)),
                             "pilot_bearing_sin": float(np.sin(np.arctan2(px, py))), "pilot_bearing_cos": float(np.cos(np.arctan2(px, py))),
                             "pilot_sog_kn": float(pf.spd),
                             "pilot_cog_sin": float(np.sin(np.radians(pf.crs))) if pd.notna(pf.crs) and pf.crs < 360 else np.nan,
                             "pilot_cog_cos": float(np.cos(np.radians(pf.crs))) if pd.notna(pf.crs) and pf.crs < 360 else np.nan,
                             "wind_speed_mps": float(w.loc[t.floor("h")].wind_speed_mps) if t.floor("h") in w.index else np.nan,
                             "wave_height_m": float(w.loc[t.floor("h")].wave_height_m) if t.floor("h") in w.index else np.nan,
                             "hour_kst": t.hour, "month": t.month,
                             "vessel_draft_m": e.vessel_draft_m, "vessel_ship_type_class": e.vessel_ship_type_class,
                             "pilot_lat": float(pf.lat), "pilot_lon": float(pf.lon)})
                continue
            sx, sy = _xy(sf.lat, sf.lon)
            sog = float(sf.spd) if pd.notna(sf.spd) else 0.0
            cog = float(sf.crs) if pd.notna(sf.crs) and sf.crs < 360 else np.degrees(np.arctan2(-sx, -sy))
            svx, svy = sog * KN * np.sin(np.radians(cog)), sog * KN * np.cos(np.radians(cog))
            vp = max(float(pf.spd), MIN_PILOT_KN) * KN
            ti, ix, iy = _intercept(sx, sy, svx, svy, px, py, vp)
            tw = third_all[(third_all.datetime >= t - TRAFFIC_WIN) & (third_all.datetime <= t) & (third_all.mmsi != str(e.vessel_mmsi))]
            tw = tw.sort_values("datetime").drop_duplicates("mmsi", keep="last")
            ox, oy = _xy(tw.lat, tw.lon)
            dist = np.hypot(ox - sx, oy - sy)
            wx = w.loc[t.floor("h")] if t.floor("h") in w.index else None
            assert max([pf.datetime, sf.datetime] + list(tw.datetime)) <= t, "input observed after dispatch"
            rng = float(np.hypot(sx - px, sy - py))
            pcog = float(pf.crs) if pd.notna(pf.crs) and pf.crs < 360 else np.nan
            rows.append({
                "event_id": e.event_id, "t_dispatch": t, "ship_visible": 1,
                "dispatch_to_meeting_min": (e.T - t).total_seconds() / 60,  # outcome-side diagnostic, never a predictor
                "ref_dist_to_breakwater_m": float(np.hypot(sx, sy)),
                "ref_bearing_from_breakwater_sin": float(np.sin(np.arctan2(sx, sy))),
                "ref_bearing_from_breakwater_cos": float(np.cos(np.arctan2(sx, sy))),
                "ref_sog_kn": sog, "ref_cog_sin": float(np.sin(np.radians(cog))), "ref_cog_cos": float(np.cos(np.radians(cog))),
                "pilot_dist_to_breakwater_m": float(np.hypot(px, py)),
                "pilot_bearing_sin": float(np.sin(np.arctan2(px, py))), "pilot_bearing_cos": float(np.cos(np.arctan2(px, py))),
                "pilot_sog_kn": float(pf.spd),
                "pilot_cog_sin": float(np.sin(np.radians(pcog))) if pd.notna(pcog) else np.nan,
                "pilot_cog_cos": float(np.cos(np.radians(pcog))) if pd.notna(pcog) else np.nan,
                "pilot_ship_range_m": rng,
                "pilot_to_ship_bearing_sin": float((sx - px) / rng) if rng > 0 else 0.0,
                "pilot_to_ship_bearing_cos": float((sy - py) / rng) if rng > 0 else 0.0,
                "intercept_dist_to_breakwater_m": _bw_dist_xy(ix, iy) if pd.notna(ix) else np.nan,
                "intercept_time_min": ti / 60 if pd.notna(ti) else np.nan,
                "traffic_0_5nm": int((dist <= 0.5 * NM).sum()), "traffic_1nm": int((dist <= NM).sum()),
                "traffic_2nm": int((dist <= 2 * NM).sum()), "nearest_third_m": float(dist.min()) if len(dist) else np.nan,
                "wind_speed_mps": float(wx.wind_speed_mps) if wx is not None else np.nan,
                "wave_height_m": float(wx.wave_height_m) if wx is not None else np.nan,
                "hour_kst": t.hour, "month": t.month,
                "vessel_draft_m": e.vessel_draft_m, "vessel_ship_type_class": e.vessel_ship_type_class,
                "ship_lat": float(sf.lat), "ship_lon": float(sf.lon), "pilot_lat": float(pf.lat), "pilot_lon": float(pf.lon),
            })
        print(f"{Path(wk).name}: {len(rows)} so far", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(R2 / ("dispatch_features_r2_prevwin.csv" if DISPATCH_WINDOW == "prev" else "dispatch_features_r2.csv"), index=False)
    dd = out.merge(d[["event_id", "traffic_direction"]], on="event_id")
    print(f"no dispatch found: {n_no_dispatch}; ship not yet visible at dispatch (kept, flagged): {n_no_ship}")
    print(dd.groupby("traffic_direction").ship_visible.agg(["size", "mean"]).round(3))
    print(dd.groupby("traffic_direction").dispatch_to_meeting_min.describe(percentiles=[.1, .25, .5, .75, .9]).round(1))


if __name__ == "__main__":
    main()
