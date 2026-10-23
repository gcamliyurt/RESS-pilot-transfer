"""Multi-port (causal): decision at pilot-boat dispatch, inputs only at or before it.

Same rule as Busan (build_r2_dispatch_features.py): within the window from the end of the same
pilot boat's previous session (or LOOKBACK before the transfer) to the start of this session, the
first fix above DISPATCH_KN that heads for the ship (online: no knowledge of later waits) (course within
MAX_HEADING_OFF of the bearing to it if visible, otherwise away from the pilot base). Inputs at t_d
are backward as-of (<= t_d, at most MAX_AGE old): ship and pilot-boat states, range/bearing, kinematic
intercept, unique third vessels (pilot fleet and service craft excluded), ERA5 wind/wave of the hour
containing t_d (a reanalysis: retrospective, not an online forecast), local hour and month.
Target: distance of the transfer from the pilot-boat base. Every timestamp used is <= t_d (asserted).
Output: data/processed/multiport/dispatch_<port>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mp_features import LEE, OUT, PORTS, SERVICE_TYPES, hav_m, load_day, type_class  # noqa: E402

LOOKBACK = pd.Timedelta(minutes=90)
WAIT_KN, DISPATCH_KN, MIN_PILOT_KN = 2.0, 5.0, 12.0
MAX_AGE = pd.Timedelta(minutes=3)
MAX_HEADING_OFF = 60.0
NM, KN = 1852.0, 0.514444
ERA5 = {"us_galveston": "era5_us_galveston", "us_longbeach": "era5_us_longbeach", "us_port_arthur": "era5_us_port_arthur",
        "us_capehenry": "era5_us_capehenry", "us_swpass": "era5_us_swpass", "us_port_everglades": "era5_us_port_everglades",
        "dk_skagen": "era5_dk_skagen", "dk_gedser": "era5_dk_gedser", "dk_drogden": "era5_dk_drogden",
        "dk_helsingor": "era5_dk_helsingor", "dk_greatbelt": "era5_dk_greatbelt"}


def build(port: str) -> pd.DataFrame:
    ev_name, ais_dir, utc_off = PORTS[port]
    base = pd.read_csv(LEE / "processed/multiport/pilot_bases.csv").set_index("port").loc[port]
    blat, blon = float(base.base_lat), float(base.base_lon)
    k = np.cos(np.radians(blat)) * 111320.0
    xy = lambda la, lo: ((np.asarray(lo, float) - blon) * k, (np.asarray(la, float) - blat) * 111320.0)
    e = pd.read_csv(LEE / f"processed/multiport/events_{ev_name}.csv", low_memory=False)
    e = e.drop_duplicates(["pilot_mmsi", "ship_mmsi", "start"]).copy()
    for c in ("start", "end", "meeting_time"):
        e[c] = pd.to_datetime(e[c], utc=True, errors="coerce")
    sess = {m: g.sort_values("end") for m, g in e.groupby("pilot_mmsi")}           # all detected sessions
    e = e.dropna(subset=["meeting_time", "lat", "lon"])
    e = e[e.traffic_direction.isin(["inbound", "outbound"]) & (e.meeting_time.dt.year == 2023)]
    fleet = set(pd.to_numeric(e.pilot_mmsi, errors="coerce").dropna().astype(int))
    wx = pd.read_csv(LEE / f"raw/multiport/weather/{ERA5[port]}.csv")
    wx["time"] = pd.to_datetime(wx.time, utc=True)
    wx = wx.set_index("time")
    folder = LEE / "raw/multiport" / ais_dir

    def asof(g, t):
        h = g[(g.t <= t) & (g.t >= t - MAX_AGE)]
        return h.iloc[-1] if len(h) else None

    rows = []
    for day, ev in e.groupby(e.meeting_time.dt.floor("D")):
        ais = pd.concat([load_day(folder, day - pd.Timedelta(days=1)), load_day(folder, day)], ignore_index=True)
        if ais.empty:
            continue
        by = {m: g.sort_values("t") for m, g in ais.groupby("MMSI")}
        third = ais[~ais.MMSI.astype(int).isin(fleet) & ~ais.vtype.isin(SERVICE_TYPES)]
        for r in ev.itertuples():
            p, s = by.get(float(r.pilot_mmsi)), by.get(float(r.ship_mmsi))
            if p is None:
                continue
            prev = sess[r.pilot_mmsi]
            prev = prev[prev.end < r.start].end.max()
            t_start = max(r.meeting_time - LOOKBACK, prev) if pd.notna(prev) else r.meeting_time - LOOKBACK
            pw = p[(p.t > t_start) & (p.t <= min(r.start, r.meeting_time))]
            # v3 (online): first move heading for the ship after the previous session (no "last wait" hindsight)
            after = pw
            pf = None
            for _, c in after[after.SOG > DISPATCH_KN].iterrows():
                px, py = xy(c.LAT, c.LON)
                sf_c = asof(s, c.t) if s is not None else None
                if pd.isna(c.COG) or c.COG >= 360:
                    pf = c; break
                if sf_c is not None:
                    sx, sy = xy(sf_c.LAT, sf_c.LON)
                    brg = np.degrees(np.arctan2(sx - px, sy - py)) % 360
                    if abs((c.COG - brg + 180) % 360 - 180) <= MAX_HEADING_OFF:
                        pf = c; break
                elif np.sin(np.radians(c.COG)) * px + np.cos(np.radians(c.COG)) * py > 0:
                    pf = c; break
            if pf is None:
                continue
            t = pf.t
            sf = asof(s, t) if s is not None else None
            px, py = xy(pf.LAT, pf.LON)
            w = wx.loc[t.floor("h")] if t.floor("h") in wx.index else None
            loc = t + pd.Timedelta(hours=utc_off)
            row = {"port": port, "event_id": f"{port}_{int(r.pilot_mmsi)}_{int(r.ship_mmsi)}_{r.meeting_time:%Y%m%d%H%M}",
                   "vessel_mmsi": int(r.ship_mmsi), "traffic_direction": r.traffic_direction, "meeting_time": r.meeting_time,
                   "t_dispatch": t, "dispatch_to_meeting_min": (r.meeting_time - t).total_seconds() / 60,
                   "meeting_lat": r.lat, "meeting_lon": r.lon, "dist_to_base_m": float(hav_m(blat, blon, r.lat, r.lon)),
                   "pilot_dist_to_base_m": float(np.hypot(px, py)),
                   "pilot_bearing_sin": float(np.sin(np.arctan2(px, py))), "pilot_bearing_cos": float(np.cos(np.arctan2(px, py))),
                   "pilot_sog_kn": float(pf.SOG),
                   "pilot_cog_sin": float(np.sin(np.radians(pf.COG))) if pd.notna(pf.COG) and pf.COG < 360 else np.nan,
                   "pilot_cog_cos": float(np.cos(np.radians(pf.COG))) if pd.notna(pf.COG) and pf.COG < 360 else np.nan,
                   "wind_speed_mps": float(w.wind_speed_10m) if w is not None else np.nan,
                   "wave_height_m": float(w.wave_height) if w is not None else np.nan,
                   "hour_local": loc.hour, "month": loc.month, "vessel_draft_m": r.draft_m,
                   "vessel_ship_type_class": type_class(r.ship_type), "ship_visible": int(sf is not None)}
            used = [pf.t]
            if sf is not None:
                sx, sy = xy(sf.LAT, sf.LON)
                sog = float(sf.SOG) if pd.notna(sf.SOG) else 0.0
                cog = float(sf.COG) if pd.notna(sf.COG) and sf.COG < 360 else np.degrees(np.arctan2(-sx, -sy))
                svx, svy = sog * KN * np.sin(np.radians(cog)), sog * KN * np.cos(np.radians(cog))
                vp = max(float(pf.SOG), MIN_PILOT_KN) * KN
                dx, dy = sx - px, sy - py
                a, b, cq = svx * svx + svy * svy - vp * vp, 2 * (svx * dx + svy * dy), dx * dx + dy * dy
                disc = b * b - 4 * a * cq
                ts = [q for q in ((-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)) if q > 0] if disc >= 0 and a != 0 else []
                ti = min(ts) if ts else np.nan
                ix, iy = (sx + svx * ti, sy + svy * ti) if ts else (np.nan, np.nan)
                tw = third[(third.t >= t - MAX_AGE) & (third.t <= t) & (third.MMSI != r.ship_mmsi)].sort_values("t").drop_duplicates("MMSI", keep="last")
                ox, oy = xy(tw.LAT, tw.LON)
                dist = np.hypot(ox - sx, oy - sy)
                rng = float(np.hypot(dx, dy))
                used += [sf.t] + list(tw.t)
                row.update({"ref_dist_to_base_m": float(np.hypot(sx, sy)),
                            "ref_bearing_sin": float(np.sin(np.arctan2(sx, sy))), "ref_bearing_cos": float(np.cos(np.arctan2(sx, sy))),
                            "ref_sog_kn": sog, "ref_cog_sin": float(np.sin(np.radians(cog))), "ref_cog_cos": float(np.cos(np.radians(cog))),
                            "pilot_ship_range_m": rng,
                            "pilot_to_ship_bearing_sin": float(dx / rng) if rng > 0 else 0.0,
                            "pilot_to_ship_bearing_cos": float(dy / rng) if rng > 0 else 0.0,
                            "intercept_dist_to_base_m": float(np.hypot(ix, iy)) if ts else np.nan,
                            "intercept_time_min": ti / 60 if ts else np.nan,
                            "traffic_0_5nm": int((dist <= 0.5 * NM).sum()), "traffic_1nm": int((dist <= NM).sum()),
                            "traffic_2nm": int((dist <= 2 * NM).sum()), "nearest_third_m": float(dist.min()) if len(dist) else np.nan})
            assert max(used) <= t, "input observed after dispatch"
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"dispatch_{port}.csv", index=False)
    print(f"{port}: {len(e)} transfers -> {len(out)} with dispatch "
          f"(ship visible {out.ship_visible.mean():.2f}; median lead in/out "
          f"{out[out.traffic_direction=='inbound'].dispatch_to_meeting_min.median():.1f}/"
          f"{out[out.traffic_direction=='outbound'].dispatch_to_meeting_min.median():.1f} min)", flush=True)
    return out


if __name__ == "__main__":
    for port in (sys.argv[1:] or PORTS):
        build(port)
