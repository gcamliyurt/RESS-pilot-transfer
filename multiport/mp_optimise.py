"""Multi-port: run the Busan decision-time optimiser (src/core/optimize_meeting_point.py, unchanged)
at US ports by injecting port geometry and data into its loaders for this process only:

  charts      NOAA ENC layers (enc_convert.py) -> fairways, depth areas/contours/soundings, land,
              lateral/cardinal/special buoys and beacons (CATLAM 1 = port, 2 = starboard)
  depth       fast ENC sampler (STR-tree), highest usage band first; MLLW chart datum
  tide        NOAA CO-OPS hourly predictions (MLLW), added to charted depth for UKC
  weather     ERA5 hourly wind and wave (6.8 leeSide)
  reference   the pilot-boat base replaces the Busan breakwater lights (target definition)
  zones       NOAA PILBOP polygons in the ENC; where absent (Galveston, Port Arthur) a circle around
              the NOAA pilot boarding area position with radius = 75th percentile of the observed
              transfers' distance to it (1-5 km); study area = the port's AIS bbox
  AIS         daily parquet files (6.8 leeSide), unique vessels; service craft and the pilot fleet
              excluded from third-party traffic
  gates       calibrated per port on its observed transfers with the Busan rule (95 %, relax-only)

Usage:  python3 mp_optimise.py calibrate <port>
        python3 mp_optimise.py run <port> <month 1-12>
        python3 mp_optimise.py summary
Outputs: data/processed/multiport/opt/<port>/..., results/multiport/optimiser_summary.csv
"""

from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.core.enrich_event_context as eec  # noqa: E402
import src.core.optimize_meeting_point as om  # noqa: E402
from src.core.collision_risk import summarize_nearby_traffic  # noqa: E402

LEE = ROOT.parents[1] / "6.8 leeSide" / "data"
MP = ROOT / "data" / "processed" / "multiport"
RES = ROOT / "results" / "multiport"
AIS_DIR = {"us_galveston": "galveston", "us_longbeach": "longbeach", "us_port_arthur": "port_arthur",
           "us_capehenry": "capehenry", "us_swpass": "swpass", "us_port_everglades": "port_everglades"}
PBA = {"us_galveston": "galveston_bay", "us_longbeach": "long_beach", "us_port_arthur": "port_arthur",
       "us_capehenry": "cape_henry", "us_swpass": "mississippi_river_southwest_pass",
       "us_port_everglades": "port_everglades"}
TIDE = {"us_galveston": "8771341", "us_longbeach": "9410660", "us_port_arthur": "8770570",
        "us_capehenry": "8638863", "us_swpass": "8760922", "us_port_everglades": "8722956"}
SERVICE_TYPES = {31, 32, 50, 52, 53}


def hav_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- port data
@functools.lru_cache(maxsize=None)
def _day(port: str, day: str) -> pd.DataFrame:
    f = LEE / "raw/multiport/ais" / AIS_DIR[port] / f"{day}.parquet"
    if not f.exists():
        return pd.DataFrame()
    x = pd.read_parquet(f, columns=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading",
                                    "VesselName", "VesselType", "Length", "Width", "Draft"])
    half_l = pd.to_numeric(x.Length, errors="coerce") / 2
    half_w = pd.to_numeric(x.Width, errors="coerce") / 2
    return pd.DataFrame({
        "mmsi": pd.to_numeric(x.MMSI, errors="coerce").astype("Int64").astype(str),
        "datetime": pd.to_datetime(x.BaseDateTime, utc=True, errors="coerce"),
        "lat": x.LAT, "lon": x.LON, "spd": x.SOG, "crs": x.COG,
        "hdg": pd.to_numeric(x.Heading, errors="coerce").where(lambda h: h < 360),
        "ship_name": x.VesselName, "vtype": pd.to_numeric(x.VesselType, errors="coerce"),
        "draft_m": x.Draft, "dim_a": half_l, "dim_b": half_l, "dim_c": half_w, "dim_d": half_w,
    }).dropna(subset=["datetime", "lat", "lon"])


def master_subset(port: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(t0.floor("D"), t1.floor("D"), freq="D")
    x = pd.concat([_day(port, f"{d:%Y-%m-%d}") for d in days], ignore_index=True)
    return x[(x.datetime >= t0) & (x.datetime <= t1)] if len(x) else x


@functools.lru_cache(maxsize=None)
def base(port: str) -> tuple[float, float]:
    b = pd.read_csv(LEE / "processed/multiport/pilot_bases.csv").set_index("port").loc[port]
    return float(b.base_lat), float(b.base_lon)


def enc_dir(port: str) -> Path:
    return MP / "enc" / port


def _gj(port: str, layer: str) -> list[dict]:
    f = enc_dir(port) / f"{port}_{layer}.geojson"
    return json.loads(f.read_text())["features"] if f.exists() else []


@functools.lru_cache(maxsize=None)
def buoys(port: str) -> pd.DataFrame:
    rows = []
    for layer in ("BOYLAT", "BCNLAT", "BOYCAR", "BCNCAR", "BOYSPP", "BCNSPP", "BOYSAW", "BOYISD", "BCNISD"):
        for f in _gj(port, layer):
            g = shape(f["geometry"]).centroid
            cat = str(f["properties"].get("CATLAM", ""))
            side = "port" if cat.startswith("1") else ("starboard" if cat.startswith("2") else pd.NA)
            rows.append({"name": f["properties"].get("OBJNAM") or layer, "lat": g.y, "lon": g.x,
                         "type": "buoy", "side": side})
    return pd.DataFrame(rows).drop_duplicates(["lat", "lon"]).reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def fairways(port: str) -> list:
    out = []
    for layer in ("FAIRWY", "TSSBND", "TSELNE", "RECTRC", "NAVLNE"):
        out += [(f"{port}_{layer}", shape(f["geometry"])) for f in _gj(port, layer) if f.get("geometry")]
    return out


@functools.lru_cache(maxsize=None)
def depth_index(port: str):
    """Depth areas (minimum of DRVAL1/2) and soundings, highest usage band first."""
    areas = []
    # Dredged areas (DRGARE, maintained channel depth) take priority over depth areas: US
    # channels are charted as DRGARE, and DEPARE alone gives the shallower surroundings.
    feats = [(f, 1) for f in _gj(port, "DRGARE")] + [(f, 0) for f in _gj(port, "DEPARE")]
    for f, drg in sorted(feats, key=lambda t: (-t[1], -int(t[0]["properties"].get("band", 0)))):
        p = f["properties"]
        vals = [v for v in (p.get("DRVAL1"), p.get("DRVAL2")) if v not in (None, "None", "nan")]
        vals = [float(v) for v in vals if pd.notna(pd.to_numeric(v, errors="coerce"))]
        if vals and f.get("geometry"):
            areas.append((shape(f["geometry"]), max(min(vals), 0.0), 10 * drg + int(p.get("band", 0))))
    snd = []
    for f in _gj(port, "SOUNDG"):
        for c in f["geometry"]["coordinates"]:
            if len(c) >= 3:
                snd.append((c[1], c[0], abs(float(c[2]))))
    snd = np.array(snd) if snd else np.zeros((0, 3))
    tree = STRtree([a[0] for a in areas]) if areas else None
    return areas, tree, snd


@functools.lru_cache(maxsize=200000)
def _depth_cached(port: str, lat5: float, lon5: float) -> float:
    areas, tree, snd = depth_index(port)
    lat, lon = lat5, lon5
    if tree is not None:
        pt = Point(lon, lat)
        hits = [areas[i] for i in tree.query(pt) if areas[i][0].contains(pt)]
        if hits:
            return float(max(hits, key=lambda h: h[2])[1])  # dredged area, then highest band
    if len(snd):  # outside charted areas only: nearest sounding within 300 m
        d = hav_m(lat, lon, snd[:, 0], snd[:, 1])
        i = int(np.argmin(d))
        if d[i] <= 300.0:
            return float(snd[i, 2])
    return np.nan


def sample_depth(port: str, lat: float, lon: float) -> float:
    return _depth_cached(port, round(float(lat), 5), round(float(lon), 5))


@functools.lru_cache(maxsize=None)
def water(port: str) -> dict:
    polys = [shape(f["geometry"]) for f in _gj(port, "DEPARE") if f.get("geometry")]
    land = [shape(f["geometry"]) for f in _gj(port, "LNDARE") if f.get("geometry")
            and f["geometry"]["type"] in ("Polygon", "MultiPolygon")]
    from shapely.prepared import prep
    # Simplified (~10 m) for speed: US ENCs have far more vertices than the Busan cells; the
    # boundary only feeds the route edge-clearance score, so the effect is below 10 m.
    u = unary_union(polys).simplify(0.0001, preserve_topology=True)
    lnd = unary_union(land).simplify(0.0001, preserve_topology=True) if land else None
    return {"land_prepared": prep(lnd) if lnd is not None else None, "polygons": polys,
            "union": u, "prepared": prep(u), "boundary": u.boundary}


@functools.lru_cache(maxsize=None)
def weather(port: str) -> pd.DataFrame:
    w = pd.read_csv(LEE / f"raw/multiport/weather/era5_{port}.csv")
    w["meeting_hour"] = pd.to_datetime(w.time, utc=True).dt.floor("h")
    t = pd.read_csv(MP / "tide" / f"{port}_{TIDE[port]}.csv")
    t["meeting_hour"] = pd.to_datetime(t.time_utc, utc=True).dt.floor("h")
    w = w.merge(t[["meeting_hour", "tide_mllw_m"]], on="meeting_hour", how="left")
    return pd.DataFrame({"meeting_hour": w.meeting_hour, "wind_speed_mps": w.wind_speed_10m,
                         "wind_dir_deg": w.wind_direction_10m, "wave_height_m": w.wave_height,
                         "tide_height_m": w.tide_mllw_m, "light_condition": pd.NA, "visibility_km": np.nan,
                         "current_speed_kn": np.nan, "current_dir_deg": np.nan})


DISPATCH = os.environ.get("MP_SET", "trigger") == "dispatch"
FEAT = "dispatch_{}.csv" if DISPATCH else "features_{}.csv"
OOF = "oof_dispatch_{}.csv" if DISPATCH else "oof_{}.csv"
OPT = "opt_dispatch" if DISPATCH else "opt"
ZONE_RADIUS_M = 5000.0  # fixed (not derived from evaluated transfers) where the ENC has no PILBOP


@functools.lru_cache(maxsize=None)
def zones(port: str) -> dict:
    b = json.loads((LEE / "raw/multiport/us_pba_groups.json").read_text())[PBA[port]]
    cells = pd.read_csv(ROOT / "data/raw/multiport_enc/cells.csv")
    bb = json.loads(cells[cells.port == port].bbox.iloc[0])  # AIS extent [lon_min, lat_min, lon_max, lat_max]
    study = box(*bb)
    pil = [shape(f["geometry"]) for f in _gj(port, "PILBOP") if f["geometry"]["type"] in ("Polygon", "MultiPolygon")]
    if pil:
        zone, rule = unary_union(pil), "ENC PILBOP"
    else:
        if DISPATCH:
            r = ZONE_RADIUS_M
        else:
            f = pd.read_csv(MP / f"features_{port}.csv")
            d = hav_m(b["lat"], b["lon"], f.meeting_lat, f.meeting_lon)
            r = float(np.clip(np.percentile(d, 75), 1000, 5000))
        k = np.cos(np.radians(b["lat"]))
        zone = Point(b["lon"], b["lat"]).buffer(r / 111320.0)
        zone = shape({"type": "Polygon", "coordinates": [[(b["lon"] + (x - b["lon"]) / k, y) for x, y in zone.exterior.coords]]})
        rule = f"circle r={r:.0f} m around NOAA PBA position"
    zone = zone.intersection(study) if zone.intersects(study) else zone
    return {"study_area": study, "embark": zone, "disembark": zone, "_rule": rule}


@functools.lru_cache(maxsize=None)
def exclusions(port: str) -> frozenset:
    f = pd.read_csv(MP / FEAT.format(port), usecols=["event_id"])
    fleet = {e.split("_")[-3] for e in f.event_id}
    svc = set()
    for d in ("2023-02-15", "2023-05-15", "2023-08-15", "2023-11-15"):
        x = _day(port, d)
        if len(x):
            svc |= set(x.loc[x.vtype.isin(SERVICE_TYPES), "mmsi"])
    return frozenset(fleet | svc)


def install(port: str) -> None:
    """Point the optimiser's loaders at this port (this process only)."""
    b_lat, b_lon = base(port)
    om.BREAKWATER_LIGHTS = pd.DataFrame([{"name": "pilot base", "lat": b_lat, "lon": b_lon,
                                          "type": "breakwater_light", "side": pd.NA}])
    om._read_buoys = lambda: buoys(port)
    om._load_fairway_geometries = lambda: fairways(port)
    om._load_breakwater_geometries = lambda: []
    om._load_water_polygons = lambda: water(port)
    om._load_bathymetry_sampler = lambda: ("multiport", "enc_noaa")
    om._sample_bathymetry = lambda lat, lon, sampler: sample_depth(port, lat, lon)
    om._read_weather_hourly = lambda: weather(port)
    om.load_zone_lookup = lambda: zones(port)
    om._collision_exclusion_mmsi = lambda: set(exclusions(port))
    om._load_master_subset = lambda path, t0, t1: master_subset(port, t0, t1)
    om.GATE_FILE = MP / OPT / port / "gate_thresholds.json"
    om._file_gates.cache_clear()
    # UKC: Busan keeps a fixed 2 m rule. US channel ports routinely operate below it and the ENC
    # depth-area lower bound is conservative, so the UKC minimum is calibrated like the other
    # gates (relax-only, >= 95 % of observed transfers pass, never above 2 m).
    ukc_min = 2.0
    if DISPATCH:
        # Charted US channel depths and AIS draughts do not support a UKC rule (5 of 6 ports would need a
        # negative threshold): the UKC gate is switched off and UKC is reported descriptively only.
        ukc_min = -np.inf
    elif om.GATE_FILE.exists():
        ukc_min = float(json.loads(om.GATE_FILE.read_text()).get("ukc_min_m", 2.0))
    om.classify_ukc_risk = lambda u, m=ukc_min: ("unknown" if pd.isna(u) else "critical" if float(u) < m
                                                  else "caution" if float(u) < 5.0 else "safe")


# ---------------------------------------------------------------- calibration
def calibrate(port: str) -> None:
    """Busan rule (calibrate_r2_gates.py): relax a gate only where its default rejects more
    than 5 % of the port's observed transfers; route metrics along the real tracks."""
    install(port)
    f = pd.read_csv(MP / FEAT.format(port))
    f = f[f.traffic_direction.isin(["inbound", "outbound"])]
    if DISPATCH:
        f["t_dec"] = f.t_dispatch
        fold_of = pd.read_csv(MP / OOF.format(port)).set_index("event_id").fold
    f = f.sample(min(len(f), 1500 if DISPATCH else 800), random_state=42)  # calibration sample per port
    fw, wat, bu = fairways(port), water(port), buoys(port)
    excl = exclusions(port)
    rows = []
    for r in f.itertuples():
        T, td = pd.Timestamp(r.meeting_time), pd.Timestamp(r.t_dec)
        pilot = r.event_id.split("_")[-3]
        m = master_subset(port, td - pd.Timedelta(minutes=2), T + pd.Timedelta(minutes=2))
        ves = m[m.mmsi == str(r.vessel_mmsi)].sort_values("datetime")
        pil = m[m.mmsi == pilot].sort_values("datetime")
        if len(ves) < 2 or len(pil) < 2:
            continue
        snap = m[(m.datetime >= T - pd.Timedelta(minutes=5)) & (m.datetime <= T + pd.Timedelta(minutes=5))]
        own = ves.iloc[(ves.datetime - T).abs().argmin()].copy()
        own["loa_est_m"], own["beam_est_m"] = own.dim_a * 2, own.dim_c * 2
        third = snap[~snap.mmsi.isin(excl | {str(r.vessel_mmsi)})].copy()
        third["loa_est_m"], third["beam_est_m"] = third.dim_a * 2, third.dim_c * 2
        tr = summarize_nearby_traffic(r.meeting_lat, r.meeting_lon, third, own)
        _, buoy_d, _ = eec._nearest_buoy(r.meeting_lat, r.meeting_lon, bu)
        in_fw, _, fw_edge, _ = om._fairway_position_metrics(r.meeting_lat, r.meeting_lon, fw)
        if not in_fw:
            in_fw, _, fw_edge = eec._fairway_context(r.meeting_lat, r.meeting_lon, fw)
        vt = [(a, b) for a, b in zip(ves[ves.datetime <= T].lat, ves[ves.datetime <= T].lon)]
        pt = [(a, b) for a, b in zip(pil[pil.datetime <= T].lat, pil[pil.datetime <= T].lon)]
        if len(vt) < 2 or len(pt) < 2:
            continue
        rm = om._route_option_metrics(pilot_lat=pt[0][0], pilot_lon=pt[0][1], vessel_lat=vt[0][0], vessel_lon=vt[0][1],
                                      cand_lat=r.meeting_lat, cand_lon=r.meeting_lon, waypoint=None, buoys=bu,
                                      breakwaters=[], water_polygons=wat, vessel_track=vt, pilot_track=pt)
        depth = sample_depth(port, r.meeting_lat, r.meeting_lon)
        w = weather(port).set_index("meeting_hour").tide_height_m
        tide = w.get(T.floor("h"), 0.0)
        rows.append({"event_id": r.event_id, "nearest_third_vessel_cpa_m": tr.get("nearest_third_vessel_cpa_m"), "dist_to_nearest_buoy_m": buoy_d,
                     "dist_to_breakwater_m": r.dist_to_base_m, "in_fairway": bool(in_fw), "dist_to_fairway_edge_m": fw_edge,
                     "route_min_buoy_clearance_m": rm["route_min_buoy_clearance_m"],
                     "route_water_coverage_ratio": rm["route_water_coverage_ratio"],
                     "ukc_m": (depth + (0 if pd.isna(tide) else tide) - r.vessel_draft_m) if pd.notna(depth) else np.nan})
    R = pd.DataFrame(rows)
    out = MP / OPT / port
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "calibration_metrics.csv", index=False)
    if DISPATCH:  # one threshold set per fold, from transfers of ships outside that fold
        # fold k excludes every transfer of a ship held out in fold k (either direction)
        ship_of = lambda ids: ids.str.split("_").str[-2]
        oof = fold_of.reset_index()
        held = oof.assign(ship=ship_of(oof.event_id)).groupby("fold").ship.agg(set)
        folds = {str(int(k)): _thresholds(R[~ship_of(R.event_id).isin(held[k])])[0] for k in sorted(held.index)}
        (out / "gate_thresholds_folds.json").write_text(json.dumps({"pass_rate": 0.95, "folds": folds}, indent=2))
    th, rep = _thresholds(R)
    ukc = pd.to_numeric(R.ukc_m, errors="coerce")
    ukc_min = round(float(min(2.0, ukc.dropna().quantile(0.05))), 2) if ukc.notna().sum() >= 50 else 2.0
    rep["ukc_ge_2m_real_transfers"] = {"n": int(ukc.notna().sum()), "share": round(float((ukc >= 2).mean()), 3),
                                       "ukc_min_calibrated_m": ukc_min, "ukc_gate": "off" if DISPATCH else "calibrated"}
    rep["zone_rule"] = zones(port)["_rule"]
    (out / "gate_thresholds.json").write_text(json.dumps({"pass_rate": 0.95, "n_transfers": len(R), "ukc_min_m": ukc_min,
                                                          "thresholds": th, "report": rep}, indent=2))
    print(port, json.dumps(th), json.dumps(rep["ukc_ge_2m_real_transfers"]), rep["zone_rule"], flush=True)


def _thresholds(R: pd.DataFrame) -> tuple[dict, dict]:
    obs = {"collision_domain_cpa_min_m": ("min", R.nearest_third_vessel_cpa_m),
           "buoy_min_m": ("min", R.dist_to_nearest_buoy_m), "breakwater_min_m": ("min", R.dist_to_breakwater_m),
           "fairway_edge_max_m": ("max", R.loc[~R.in_fairway, "dist_to_fairway_edge_m"]),
           "route_buoy_clearance_min_m": ("min", R.route_min_buoy_clearance_m),
           "route_water_coverage_min": ("min", R.route_water_coverage_ratio)}
    th, rep = {}, {}
    for k, (kind, s) in obs.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 50:
            rep[k] = {"n": int(len(s)), "kept_default": om.GATE_DEFAULTS[k]}
            continue
        q = float(s.quantile(0.05 if kind == "min" else 0.95))
        v = min(q, om.GATE_DEFAULTS[k]) if kind == "min" else max(q, om.GATE_DEFAULTS[k])
        th[k] = round(v, 3)
        rep[k] = {"n": int(len(s)), "default": om.GATE_DEFAULTS[k], "calibrated": round(v, 3)}
    return th, rep


# ---------------------------------------------------------------- optimisation
def run(port: str, month: int) -> None:
    install(port)
    f = pd.read_csv(MP / FEAT.format(port))
    a = pd.read_csv(MP / OOF.format(port))
    if DISPATCH:
        a = a[a.ship_visible == 1]  # the optimiser needs the ship's position at the decision time
    f["meeting_time"] = pd.to_datetime(f.meeting_time, utc=True)
    f = f[(f.meeting_time.dt.month == month) & f.event_id.isin(a.event_id)]
    ev = pd.DataFrame({"event_id": f.event_id, "pilot_mmsi": f.event_id.str.split("_").str[-3],
                       "vessel_mmsi": f.vessel_mmsi.astype(str), "traffic_direction": f.traffic_direction,
                       "meeting_time": f.meeting_time, "meeting_lat": f.meeting_lat, "meeting_lon": f.meeting_lon})
    a["t_dec"] = pd.to_datetime(a.t_dec, utc=True)
    log: list = []
    fg = MP / OPT / port / "gate_thresholds_folds.json"
    out = om.optimize_meeting_points(ev, master_path=Path("."), anchors=a[a.event_id.isin(ev.event_id)], candidate_log=log,
                                     fold_gates=json.loads(fg.read_text())["folds"] if DISPATCH and fg.exists() else None)
    d = MP / OPT / port
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"optimized_{month:02d}.csv", index=False)
    if log:  # keep the anchor rows (UKC veto statistics) and all candidates of the first events (maps)
        c = pd.concat(log, ignore_index=True)
        c[c.candidate_source == "ml_anchor"].to_csv(d / f"candidates_{month:02d}.csv", index=False)
        c[c.event_id.isin(c.event_id.unique()[:15])].to_csv(d / f"candidates_sample_{month:02d}.csv", index=False)
    print(port, month, len(ev), "->", len(out), flush=True)


def summary() -> None:
    rows = []
    for port in AIS_DIR:
        files = sorted((MP / OPT / port).glob("optimized_*.csv"))
        if not files:
            continue
        o = pd.concat(pd.read_csv(x) for x in files)
        f = pd.read_csv(MP / FEAT.format(port))
        n_conf = int(f.traffic_direction.isin(["inbound", "outbound"]).sum())
        o["gate"] = o.recommended_gate_pass.astype(str).str.lower().eq("true")
        n_infeasible = int((o.get("recommendation_status", pd.Series("feasible", index=o.index)) == "no_feasible").sum())
        o = o[o.get("recommendation_status", pd.Series("feasible", index=o.index)) != "no_feasible"]  # feasible recommendations only
        o["off_rec"] = hav_m(o.actual_meeting_lat, o.actual_meeting_lon, o.recommended_lat, o.recommended_lon)
        o["off_anc"] = hav_m(o.actual_meeting_lat, o.actual_meeting_lon, o.anchor_lat, o.anchor_lon)
        cls = o.recommended_collision_risk_class.value_counts(normalize=True)
        anc = pd.concat(pd.read_csv(x, usecols=["gate_reason", "ukc_m"]) for x in sorted((MP / OPT / port).glob("candidates_[0-9]*.csv")))
        row = {"port": port, "events_with_features": n_conf, "optimised": len(o) + n_infeasible,
               "feasible_recommendations": len(o), "no_feasible": n_infeasible,
               "coverage": round(len(o) / n_conf, 3), "gate_pass": round(o.gate.mean(), 3),
               "ukc_ge_2m": round((pd.to_numeric(o.recommended_ukc_m, errors="coerce") >= 2).mean(), 3),
               "ukc_min_port_m": json.loads((MP / "opt" / port / "gate_thresholds.json").read_text()).get("ukc_min_m"),
               "ukc_ge_port_min": round((pd.to_numeric(o.recommended_ukc_m, errors="coerce") >= json.loads((MP / "opt" / port / "gate_thresholds.json").read_text()).get("ukc_min_m", 2.0)).mean(), 3),
               "low": round(cls.get("low", 0), 3), "medium": round(cls.get("medium", 0), 3), "high": round(cls.get("high", 0), 3),
               "kept_anchor": round((o.recommended_source == "ml_anchor").mean(), 3),
               "anchors_ukc_lt_2m": int((pd.to_numeric(anc.ukc_m, errors="coerce") < 2).sum()),
               "anchors_pass_all_gates": round(float((anc.gate_reason == "passes_gates").mean()), 3)}
        for direc in ("inbound", "outbound"):
            g = o[o.traffic_direction == direc]
            row[f"move_from_anchor_m_{direc}"] = round(g.recommended_dist_to_anchor_m.median(), 0)
            row[f"anchor_to_obs_m_{direc}"] = round(g.off_anc.median(), 0)
            row[f"rec_to_obs_m_{direc}"] = round(g.off_rec.median(), 0)
        rows.append(row)
    s = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    s.to_csv(RES / ("optimiser_summary_dispatch.csv" if DISPATCH else "optimiser_summary.csv"), index=False)
    print(s.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calibrate":
        calibrate(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]))
    else:
        summary()
