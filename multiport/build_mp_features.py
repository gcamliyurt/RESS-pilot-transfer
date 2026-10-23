"""Multi-port: decision-time features for pilot transfers outside Busan (RESS R2, Section 6.2).

Same rules as the Busan R2 pipeline (build_r2_decision_time.py, build_r2_apriori_features.py):
  t_dec   = max(T - 20 min, first time both the ship and the pilot boat have an AIS fix after T - 20 min),
            events with lead T - t_dec < 5 min are dropped
  target  = distance of the transfer position from the port's pilot-boat base (the Busan analogue of
            the breakwater: how far out the pilot boat goes); base positions from pilot_bases.csv
  features at t_dec: ship distance/bearing from the base, SOG, COG (sin, cos), third-party vessels within
            0.5 / 1 / 2 NM (unique vessels, service craft and the pilot fleet excluded), nearest third vessel,
            draught, ship-type class, ERA5 wind and wave, hour (local), month, lead time
Inputs (read-only): ../../6.8 leeSide/data/processed/multiport/events_<port>.csv and the daily AIS
parquet files in ../../6.8 leeSide/data/raw/multiport/. All times UTC.
Output: data/processed/multiport/features_<port>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEE = ROOT.parents[1] / "6.8 leeSide" / "data"
OUT = ROOT / "data" / "processed" / "multiport"
NM = 1852.0
MAX_LEAD, MIN_LEAD = pd.Timedelta(minutes=20), pd.Timedelta(minutes=5)

# port -> (events file, AIS folder, UTC offset of local time for the hour feature)
PORTS = {
    "us_galveston": ("us_galveston", "ais/galveston", -6),
    "us_longbeach": ("us_longbeach", "ais/longbeach", -8),
    "us_port_arthur": ("us_port_arthur", "ais/port_arthur", -6),
    "us_capehenry": ("us_capehenry", "ais/capehenry", -5),
    "us_swpass": ("us_swpass", "ais/swpass", -6),
    "us_port_everglades": ("us_port_everglades", "ais/port_everglades", -5),
    "dk_skagen": ("dk_skagen", "ais_dk/skagen", 1),
    "dk_gedser": ("dk_gedser", "ais_dk/gedser", 1),
    "dk_drogden": ("dk_drogden", "ais_dk/drogden", 1),
    "dk_helsingor": ("dk_helsingor", "ais_dk/helsingor", 1),
    "dk_greatbelt": ("dk_greatbelt", "ais_dk/greatbelt", 1),
}
SERVICE_TYPES = {31, 32, 50, 52, 53}
# Danish AIS ship-type text -> AIS type code (as 6.8 code/multiport/detect_ports.py)
DK_TYPE = {"Pilot": 50, "SAR": 51, "Tug": 52, "Port tender": 53, "Law enforcement": 55, "Towing": 31,
           "Towing long/wide": 32}  # towing, pilot, tug, port tender (as in Busan)


def hav_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    k = np.cos(np.radians(lat1))
    return np.degrees(np.arctan2((np.asarray(lon2) - lon1) * k, np.asarray(lat2) - lat1)) % 360


def type_class(t) -> str:
    t = pd.to_numeric(t, errors="coerce")
    if pd.isna(t):
        return "UNK"
    t = int(t)
    return {7: "Cargo", 8: "Tanker", 6: "Passenger"}.get(t // 10, "Other")


def load_day(folder: Path, day: pd.Timestamp) -> pd.DataFrame:
    f = folder / f"{day:%Y-%m-%d}.parquet"
    if not f.exists():
        return pd.DataFrame()
    x = pd.read_parquet(f)
    tcol = "VesselType" if "VesselType" in x else ("ShipType" if "ShipType" in x else None)
    x = x.rename(columns={tcol: "vtype"}) if tcol else x.assign(vtype=np.nan)
    x = x[["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "vtype"]].copy()
    x["t"] = pd.to_datetime(x.BaseDateTime, utc=True, errors="coerce")
    x["MMSI"] = pd.to_numeric(x.MMSI, errors="coerce")
    if x.vtype.dtype == object or str(x.vtype.dtype).startswith("string"):
        x["vtype"] = x.vtype.map(DK_TYPE)  # Danish AIS gives the ship type as text
    x["vtype"] = pd.to_numeric(x.vtype, errors="coerce")
    return x.dropna(subset=["t", "MMSI", "LAT", "LON"])


def build(port: str) -> pd.DataFrame:
    ev_name, ais_dir, utc_off = PORTS[port]
    base = pd.read_csv(LEE / "processed/multiport/pilot_bases.csv").set_index("port").loc[port]
    b_lat, b_lon = float(base.base_lat), float(base.base_lon)
    e = pd.read_csv(LEE / f"processed/multiport/events_{ev_name}.csv", low_memory=False)
    e = e.dropna(subset=["meeting_time", "lat", "lon", "traffic_direction"])
    e = e.drop_duplicates(["pilot_mmsi", "ship_mmsi", "start"]).copy()
    e["T"] = pd.to_datetime(e.meeting_time, utc=True)
    e = e[e["T"].dt.year == 2023]
    fleet = set(pd.to_numeric(e.pilot_mmsi, errors="coerce").dropna().astype(int))
    folder = LEE / "raw/multiport" / ais_dir

    rows = []
    for day, ev in e.groupby(e["T"].dt.floor("D")):
        ais = pd.concat([load_day(folder, day - pd.Timedelta(days=1)), load_day(folder, day)], ignore_index=True)
        if ais.empty:
            continue
        ais = ais[(ais.t >= ev["T"].min() - MAX_LEAD - pd.Timedelta(minutes=3)) & (ais.t <= ev["T"].max())]
        by = {m: g.sort_values("t") for m, g in ais.groupby("MMSI")}
        third_ok = ~ais.MMSI.astype(int).isin(fleet) & ~ais.vtype.isin(SERVICE_TYPES)
        third = ais[third_ok]
        for r in ev.itertuples():
            s, p = by.get(float(r.ship_mmsi)), by.get(float(r.pilot_mmsi))
            if s is None or p is None:
                continue
            t0 = r.T - MAX_LEAD
            fs, fp = s[(s.t >= t0) & (s.t <= r.T)].t, p[(p.t >= t0) & (p.t <= r.T)].t
            if fs.empty or fp.empty:
                continue
            t_dec = max(fs.min(), fp.min(), t0)
            lead = (r.T - t_dec).total_seconds() / 60
            if lead < MIN_LEAD.total_seconds() / 60:
                continue
            i = (s.t - t_dec).abs().idxmin()
            if abs((s.t[i] - t_dec).total_seconds()) > 180:
                continue
            sp = s.loc[i]
            w = third[(third.t >= t_dec - pd.Timedelta(minutes=1)) & (third.t <= t_dec + pd.Timedelta(minutes=1))
                      & (third.MMSI != r.ship_mmsi)].sort_values("t").drop_duplicates("MMSI", keep="last")
            d = hav_m(sp.LAT, sp.LON, w.LAT.to_numpy(), w.LON.to_numpy()) if len(w) else np.array([])
            brg = float(bearing(b_lat, b_lon, sp.LAT, sp.LON))
            cog = float(sp.COG) if pd.notna(sp.COG) and sp.COG < 360 else np.nan
            loc = t_dec + pd.Timedelta(hours=utc_off)
            rows.append({
                "port": port, "event_id": f"{port}_{int(r.pilot_mmsi)}_{int(r.ship_mmsi)}_{r.T:%Y%m%d%H%M}",
                "vessel_mmsi": int(r.ship_mmsi), "traffic_direction": r.traffic_direction,
                "meeting_time": r.T, "t_dec": t_dec, "lead_min": lead,
                "meeting_lat": r.lat, "meeting_lon": r.lon,
                "dist_to_base_m": float(hav_m(b_lat, b_lon, r.lat, r.lon)),
                "ref_dist_to_base_m": float(hav_m(b_lat, b_lon, sp.LAT, sp.LON)),
                "ref_bearing_sin": np.sin(np.radians(brg)), "ref_bearing_cos": np.cos(np.radians(brg)),
                "ref_sog_kn": float(sp.SOG) if pd.notna(sp.SOG) else np.nan,
                "ref_cog_sin": np.sin(np.radians(cog)), "ref_cog_cos": np.cos(np.radians(cog)),
                "traffic_0_5nm": int((d <= 0.5 * NM).sum()), "traffic_1nm": int((d <= NM).sum()),
                "traffic_2nm": int((d <= 2 * NM).sum()), "nearest_third_m": float(d.min()) if len(d) else np.nan,
                "vessel_draft_m": r.draft_m, "vessel_loa_m": r.loa_m, "vessel_ship_type_class": type_class(r.ship_type),
                "wind_speed_mps": r.era5_wind_mps, "wave_height_m": r.era5_wave_m,
                "hour_local": loc.hour, "month": loc.month,
            })
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"features_{port}.csv", index=False)
    print(f"{port}: {len(e)} events -> {len(out)} with decision-time features "
          f"({out.traffic_direction.value_counts().to_dict() if len(out) else {}})", flush=True)
    return out


if __name__ == "__main__":
    for port in (sys.argv[1:] or PORTS):
        build(port)
