from __future__ import annotations

import argparse
import math
from pathlib import Path

import functools
import json

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from src.core.collision_risk import summarize_nearby_traffic
from src.core.detect_interceptions import service_craft_mmsi
from src.core.detect_interceptions import (
    _pick_master_path,
    LEGACY_TUG_BOATS,
    haversine_distance_m,
    load_pilot_mmsi,
    load_zone_lookup,
)
from src.core.enrich_event_context import (
    BREAKWATER_LIGHTS,
    _fairway_context,
    _load_bathymetry_sampler,
    _load_breakwater_geometries,
    _load_fairway_geometries,
    _load_master_subset,
    _nearest_buoy,
    _nearest_polygon_distance_m,
    _read_buoys,
    _read_weather_hourly,
    _sample_bathymetry,
)
from src.core.ukc_risk import classify_ukc_risk, compute_ukc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "meeting_point"
ENC_GEOJSON_DIR = PROJECT_ROOT / "data" / "processed" / "charts_enc_geojson"


def _collision_exclusion_mmsi() -> set[str]:
    return set(service_craft_mmsi())


def load_events(events_path: Path) -> pd.DataFrame:
    try:
        events = pd.read_csv(events_path)
    except EmptyDataError:
        events = pd.DataFrame()
    for col in ["start_time", "meeting_time", "end_time"]:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce", utc=True)
    return events


GATE_DEFAULTS = {
    "breakwater_min_m": 250.0,
    "buoy_min_m": 120.0,
    "fairway_edge_max_m": 800.0,
    "route_buoy_clearance_min_m": 80.0,
    "route_breakwater_clearance_min_m": 150.0,
    "route_water_coverage_min": 0.98,
    "collision_domain_cpa_min_m": 100.0,
}
GATE_FILE = PROJECT_ROOT / "data" / "processed" / "r2" / "gate_thresholds.json"


_GATE_OVERRIDE: dict[str, float] | None = None  # set per event in decision-time mode (fold-calibrated gates)


@functools.lru_cache(maxsize=1)
def _file_gates() -> dict[str, float]:
    g = dict(GATE_DEFAULTS)
    if GATE_FILE.exists():
        g.update(json.loads(GATE_FILE.read_text(encoding="utf-8")).get("thresholds", {}))
    return g


def _gates() -> dict[str, float]:
    """Gate thresholds; calibrated on log-confirmed real transfers when
    data/processed/r2/gate_thresholds.json exists (calibrate_r2_gates.py)."""
    if _GATE_OVERRIDE is not None:
        return {**GATE_DEFAULTS, **_GATE_OVERRIDE}
    return _file_gates()


def _candidate_gate_status(row: pd.Series) -> tuple[bool, str]:
    G = _gates()
    reasons: list[str] = []

    if str(row.get("ukc_risk_class")) == "critical":
        reasons.append("ukc_critical")
    domain_cpa = pd.to_numeric(pd.Series([row.get("nearest_third_vessel_cpa_m")]), errors="coerce").iloc[0]
    if pd.notna(domain_cpa):
        if domain_cpa < G["collision_domain_cpa_min_m"]:
            reasons.append("collision_risk_high")
    elif str(row.get("collision_risk_class")) == "high":
        reasons.append("collision_risk_high")

    breakwater_dist = pd.to_numeric(pd.Series([row.get("dist_to_breakwater_m")]), errors="coerce").iloc[0]
    if pd.notna(breakwater_dist) and breakwater_dist < G["breakwater_min_m"]:
        reasons.append("too_close_breakwater")

    buoy_dist = pd.to_numeric(pd.Series([row.get("dist_to_nearest_buoy_m")]), errors="coerce").iloc[0]
    if pd.notna(buoy_dist) and buoy_dist < G["buoy_min_m"]:
        reasons.append("too_close_buoy")

    fairway_edge_m = pd.to_numeric(pd.Series([row.get("dist_to_fairway_edge_m")]), errors="coerce").iloc[0]
    in_fairway = bool(row.get("in_fairway")) if pd.notna(row.get("in_fairway")) else False
    if not in_fairway and pd.notna(fairway_edge_m) and fairway_edge_m > G["fairway_edge_max_m"]:
        reasons.append("poor_fairway_access")

    route_buoy_clearance = pd.to_numeric(pd.Series([row.get("route_min_buoy_clearance_m")]), errors="coerce").iloc[0]
    if pd.notna(route_buoy_clearance) and route_buoy_clearance < G["route_buoy_clearance_min_m"]:
        reasons.append("route_hits_buoy_corridor")

    route_breakwater_clearance = pd.to_numeric(pd.Series([row.get("route_min_breakwater_clearance_m")]), errors="coerce").iloc[0]
    if pd.notna(route_breakwater_clearance) and route_breakwater_clearance < G["route_breakwater_clearance_min_m"]:
        reasons.append("route_hits_breakwater_corridor")

    route_water_coverage = pd.to_numeric(pd.Series([row.get("route_water_coverage_ratio")]), errors="coerce").iloc[0]
    if pd.notna(route_water_coverage) and route_water_coverage < G["route_water_coverage_min"]:
        reasons.append("route_leaves_water_area")

    forward_proj_m = pd.to_numeric(pd.Series([row.get("forward_progress_proj_m")]), errors="coerce").iloc[0]
    traffic_direction = str(row.get("traffic_direction", "")).lower()
    if traffic_direction == "inbound" and pd.notna(forward_proj_m):
        local_collision = str(row.get("collision_risk_class", "")).lower()
        nearby_count = pd.to_numeric(pd.Series([row.get("surrounding_vessel_count_0_5nm")]), errors="coerce").iloc[0]
        severe_blocking = (local_collision == "high") or (pd.notna(nearby_count) and nearby_count >= 6.0)
        if forward_proj_m < 0.0:
            reasons.append("candidate_astern_of_vessel")
        elif forward_proj_m < 200.0 and not severe_blocking:
            reasons.append("candidate_not_sufficiently_ahead")

    return (len(reasons) == 0), ";".join(reasons) if reasons else "passes_gates"


def _nearest_row(df: pd.DataFrame, ts: pd.Timestamp, backward: bool = False) -> pd.Series | None:
    if df.empty:
        return None
    if backward:  # decision-time mode: latest observation at or before ts only
        df = df[df["datetime"] <= ts]
        return df.sort_values("datetime").iloc[-1] if len(df) else None
    order = (df["datetime"] - ts).abs().sort_values().index
    if len(order) == 0:
        return None
    return df.loc[order[0]]


@functools.lru_cache(maxsize=1)  # called once per event; polygons never change
def _load_water_polygons() -> dict[str, object]:
    polys: list[object] = []
    if not ENC_GEOJSON_DIR.exists():
        return {"polygons": [], "union": None, "prepared": None, "boundary": None}
    for fp in ENC_GEOJSON_DIR.glob("*DEPARE.geojson"):
        try:
            gj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for feat in gj.get("features", []):
            geom = feat.get("geometry")
            if not geom:
                continue
            try:
                polys.append(shape(geom))
            except Exception:
                continue
    if not polys:
        return {"polygons": [], "union": None, "prepared": None, "boundary": None}
    water_union = unary_union(polys)
    land_polys = []
    for fp in ENC_GEOJSON_DIR.glob("*LNDARE.geojson"):
        try:
            land_polys += [shape(f["geometry"]) for f in json.loads(fp.read_text(encoding="utf-8")).get("features", []) if f.get("geometry")]
        except Exception:
            continue
    return {
        "land_prepared": prep(unary_union(land_polys)) if land_polys else None,
        "land_union": unary_union(land_polys) if land_polys else None,
        "polygons": polys,
        "union": water_union,
        "prepared": prep(water_union),
        "boundary": water_union.boundary,
    }


def _sample_points_in_polygon(poly, step_m: float, max_points: int = 36) -> list[tuple[float, float]]:
    minx, miny, maxx, maxy = poly.bounds
    lat_step = float(step_m) / 111320.0
    lon_step = float(step_m) / (111320.0 * max(np.cos(np.radians((miny + maxy) / 2.0)), 1e-6))
    pts: list[tuple[float, float]] = []
    lat = miny
    while lat <= maxy and len(pts) < max_points:
        lon = minx
        while lon <= maxx and len(pts) < max_points:
            if poly.contains(Point(lon, lat)):
                pts.append((float(lat), float(lon)))
            lon += lon_step
        lat += lat_step
    return pts


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    dlon = lon2_r - lon1_r
    y = math.sin(dlon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _turn_angle_deg(in_bearing: float, out_bearing: float) -> float:
    diff = abs(in_bearing - out_bearing) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _normalize_waypoints(waypoint_spec) -> list[tuple[float, float]]:
    if waypoint_spec is None:
        return []
    if isinstance(waypoint_spec, tuple) and len(waypoint_spec) == 2:
        return [(float(waypoint_spec[0]), float(waypoint_spec[1]))]
    out: list[tuple[float, float]] = []
    if isinstance(waypoint_spec, list):
        for wp in waypoint_spec:
            if isinstance(wp, tuple) and len(wp) == 2:
                out.append((float(wp[0]), float(wp[1])))
    return out


def _historical_hotspot(events: pd.DataFrame, event: pd.Series) -> tuple[float, float] | None:
    if events.empty or "meeting_lat" not in events.columns or "meeting_lon" not in events.columns:
        return None
    if pd.notna(event.get("decision_time")) and "meeting_time" in events.columns:
        events = events[events["meeting_time"] < event.get("decision_time")]  # anchor mode: past events only
    if events.empty or "meeting_lat" not in events.columns or "meeting_lon" not in events.columns:
        return None
    same_dir = events.copy()
    if "traffic_direction" in same_dir.columns and pd.notna(event.get("traffic_direction")):
        same_dir = same_dir[same_dir["traffic_direction"] == event.get("traffic_direction")]
    if "event_id" in same_dir.columns and pd.notna(event.get("event_id")):
        same_dir = same_dir[same_dir["event_id"] != event.get("event_id")]
    same_dir = same_dir.dropna(subset=["meeting_lat", "meeting_lon"])
    if same_dir.empty:
        return None
    return float(same_dir["meeting_lat"].median()), float(same_dir["meeting_lon"].median())


def _preferred_zone(direction: object, zones: dict[str, object]):
    d = str(direction).lower() if pd.notna(direction) else ""
    if d == "inbound":
        return zones["embark"]
    if d == "outbound":
        return zones["disembark"]
    return zones["embark"].union(zones["disembark"])


def _fairway_position_metrics(lat: float, lon: float, fairways: list[tuple[str, object]]) -> tuple[bool, object, float, float]:
    if not fairways:
        return False, pd.NA, np.nan, np.nan
    pt = Point(lon, lat)
    inside_hits = []
    for name, geom in fairways:
        try:
            if geom.contains(pt):
                inside_hits.append((name, float(geom.boundary.distance(pt)) * 111320.0))
        except Exception:
            continue
    if inside_hits:
        best_name, edge_m = max(inside_hits, key=lambda x: x[1])
        return True, best_name, 0.0, edge_m
    distances = []
    for name, geom in fairways:
        try:
            distances.append((name, float(geom.distance(pt)) * 111320.0))
        except Exception:
            continue
    if not distances:
        return False, pd.NA, np.nan, np.nan
    nearest_name, nearest_m = min(distances, key=lambda x: x[1])
    return False, nearest_name, nearest_m, 0.0


def _forward_progress_metrics(
    vessel_lat: float,
    vessel_lon: float,
    actual_lat: float | None,
    actual_lon: float | None,
    cand_lat: float,
    cand_lon: float,
) -> tuple[float, float, float]:
    if actual_lat is None or actual_lon is None or pd.isna(actual_lat) or pd.isna(actual_lon):
        return 0.5, np.nan, np.nan
    lat_ref = float(np.mean([vessel_lat, float(actual_lat), cand_lat]))
    cos_ref = max(np.cos(np.radians(lat_ref)), 1e-6)
    vx = (float(actual_lon) - vessel_lon) * 111320.0 * cos_ref
    vy = (float(actual_lat) - vessel_lat) * 111320.0
    cx = (cand_lon - vessel_lon) * 111320.0 * cos_ref
    cy = (cand_lat - vessel_lat) * 111320.0
    vnorm = math.hypot(vx, vy)
    if vnorm < 1.0:
        return 0.5, np.nan, np.nan
    proj_actual = vnorm
    proj_cand = (cx * vx + cy * vy) / vnorm
    # Prefer candidates that are at least as far ahead as the actual meeting,
    # but do not reward excessive forward displacement.
    ratio = proj_cand / max(proj_actual, 1.0)
    if ratio <= 0:
        return 0.0, float(ratio), float(proj_cand)
    if ratio <= 1.0:
        return max(0.0, min(1.0, 0.35 + 0.65 * ratio)), float(ratio), float(proj_cand)
    return max(0.0, min(1.0, 1.0 - 0.6 * min(ratio - 1.0, 1.0))), float(ratio), float(proj_cand)


def _offset_point_m(lat: float, lon: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    lat_ref = max(math.cos(math.radians(lat)), 1e-6)
    out_lat = lat + (dy_m / 111320.0)
    out_lon = lon + (dx_m / (111320.0 * lat_ref))
    return float(out_lat), float(out_lon)


def _forward_corridor_candidates(
    vessel_lat: float,
    vessel_lon: float,
    actual_lat: float | None,
    actual_lon: float | None,
    preferred_zone,
    step_m: float,
) -> list[tuple[float, float]]:
    if actual_lat is None or actual_lon is None or pd.isna(actual_lat) or pd.isna(actual_lon):
        return []
    lat_ref = float(np.mean([vessel_lat, float(actual_lat)]))
    cos_ref = max(np.cos(np.radians(lat_ref)), 1e-6)
    vx = (float(actual_lon) - vessel_lon) * 111320.0 * cos_ref
    vy = (float(actual_lat) - vessel_lat) * 111320.0
    vnorm = math.hypot(vx, vy)
    if vnorm < 50.0:
        return []

    ux = vx / vnorm
    uy = vy / vnorm
    # Left/right normal to the approach axis.
    nx = -uy
    ny = ux

    along_steps = [0.75, 0.9, 1.0, 1.1, 1.25, 1.4]
    lateral_offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
    lateral_span_m = max(120.0, min(350.0, float(step_m)))
    pts: list[tuple[float, float]] = []
    for along in along_steps:
        for lateral in lateral_offsets:
            dx = ux * (vnorm * along) + nx * (lateral * lateral_span_m)
            dy = uy * (vnorm * along) + ny * (lateral * lateral_span_m)
            cand_lat, cand_lon = _offset_point_m(vessel_lat, vessel_lon, dx, dy)
            try:
                if preferred_zone.contains(Point(cand_lon, cand_lat)):
                    pts.append((cand_lat, cand_lon))
            except Exception:
                continue
    return pts


def _buoy_centerline_candidates(
    preferred_zone,
    buoys: pd.DataFrame | None,
    vessel_lat: float,
    actual_lat: float | None,
) -> list[tuple[float, float]]:
    if buoys is None or buoys.empty or "side" not in buoys.columns:
        return []
    work = buoys.copy()
    work["side_norm"] = work["side"].astype("string").str.strip().str.lower()
    port = work[work["side_norm"].str.contains("port", na=False)].copy()
    star = work[work["side_norm"].str.contains("starboard", na=False)].copy()
    if port.empty or star.empty:
        return []

    target_lat = float(actual_lat) if actual_lat is not None and not pd.isna(actual_lat) else float(vessel_lat)
    pts: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for _, p in port.iterrows():
        sdist = (star["lat"] - float(p["lat"])).abs()
        if sdist.empty:
            continue
        s = star.loc[sdist.idxmin()]
        mid_lat = (float(p["lat"]) + float(s["lat"])) / 2.0
        mid_lon = (float(p["lon"]) + float(s["lon"])) / 2.0
        if abs(mid_lat - target_lat) > 0.03:
            continue
        key = (round(mid_lat, 6), round(mid_lon, 6))
        if key in seen:
            continue
        seen.add(key)
        pts.append((mid_lat, mid_lon))
    pts.sort(key=lambda x: abs(x[0] - target_lat))
    return pts[:12]


def _route_option_metrics(
    pilot_lat: float,
    pilot_lon: float,
    vessel_lat: float,
    vessel_lon: float,
    cand_lat: float,
    cand_lon: float,
    waypoint,
    buoys: pd.DataFrame | None = None,
    breakwaters: list[tuple[str, object]] | None = None,
    water_polygons: dict[str, object] | None = None,
    vessel_track: list[tuple[float, float]] | None = None,
    pilot_track: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    def _track_leg(track, start_lat: float, start_lon: float) -> list[tuple[float, float]]:
        # Each craft stays on its own observed (channel-following) track up to the
        # track point nearest the candidate, then makes the short offset to it. A
        # straight line from the rewind position cuts across islands/breakwaters.
        if callable(track):  # decision-time mode: shortest path through charted water
            return track(cand_lat, cand_lon)
        if not track or len(track) < 2:
            return [(start_lat, start_lon), (cand_lat, cand_lon)]
        k = math.cos(math.radians(cand_lat))
        d2 = [((la - cand_lat) ** 2 + ((lo - cand_lon) * k) ** 2) for la, lo in track]
        i = int(np.argmin(d2))
        return list(track[: i + 1]) + [(cand_lat, cand_lon)]

    def _ship_leg() -> list[tuple[float, float]]:
        return _track_leg(vessel_track, vessel_lat, vessel_lon)

    def _pilot_leg() -> list[tuple[float, float]]:
        return _track_leg(pilot_track, pilot_lat, pilot_lon)

    def _xy(lat: float, lon: float, lat_ref: float) -> tuple[float, float]:
        x = lon * 111320.0 * math.cos(math.radians(lat_ref))
        y = lat * 111320.0
        return (x, y)

    def _route_hazard_metrics(points_latlon: list[tuple[float, float]]) -> tuple[float, float]:
        if len(points_latlon) < 2:
            return (np.nan, np.nan)
        lat_ref = float(np.mean([p[0] for p in points_latlon]))
        line = LineString([_xy(lat, lon, lat_ref) for lat, lon in points_latlon])
        min_buoy = np.nan
        if buoys is not None and not buoys.empty:
            vals = []
            for r in buoys.itertuples(index=False):
                if pd.isna(getattr(r, "lat", np.nan)) or pd.isna(getattr(r, "lon", np.nan)):
                    continue
                vals.append(line.distance(Point(*_xy(float(r.lat), float(r.lon), lat_ref))))
            if vals:
                min_buoy = float(min(vals))
        min_breakwater = np.nan
        if breakwaters:
            vals = []
            for _nm, poly in breakwaters:
                try:
                    coords = [(_xy(lat, lon, lat_ref)) for lon, lat in poly.exterior.coords]
                    vals.append(line.distance(LineString(coords)))
                except Exception:
                    continue
            if vals:
                min_breakwater = float(min(vals))
        return (min_buoy, min_breakwater)

    def _route_water_metrics(points_latlon: list[tuple[float, float]]) -> tuple[float, float]:
        if len(points_latlon) < 2 or not water_polygons or water_polygons.get("union") is None:
            return (1.0, np.nan)
        water_union = water_polygons["union"]
        water_prepared = water_polygons["prepared"]
        water_boundary = water_polygons["boundary"]
        line_geo = LineString([(lon, lat) for lat, lon in points_latlon])
        sample_points = [line_geo.interpolate(t, normalized=True) for t in np.linspace(0.0, 1.0, 21)]
        # A route fails only where it crosses charted LAND (ENC LNDARE). DEPARE has
        # small gaps that real, completed transits pass through, so "not inside a
        # depth area" is not evidence of a hazard.
        land = water_polygons.get("land_prepared")
        inside = 0
        edge_dists = []
        for pt in sample_points:
            try:
                on_land = land.contains(pt) if land is not None else not (
                    water_prepared.contains(pt) or water_union.distance(pt) < 1e-9)
                if not on_land:
                    inside += 1
                    edge_dists.append(pt.distance(water_boundary))
            except Exception:
                continue
        coverage_ratio = inside / max(len(sample_points), 1)
        min_edge_deg = min(edge_dists) if edge_dists else np.nan
        # rough degree-to-metre conversion near Busan
        min_edge_m = float(min_edge_deg * 111320.0) if pd.notna(min_edge_deg) else np.nan
        return (coverage_ratio, min_edge_m)

    waypoints = _normalize_waypoints(waypoint)

    def _path_distance(points_latlon: list[tuple[float, float]]) -> float:
        total = 0.0
        for (lat1, lon1), (lat2, lon2) in zip(points_latlon[:-1], points_latlon[1:]):
            total += float(haversine_distance_m(np.array([lat1]), np.array([lon1]), np.array([lat2]), np.array([lon2]))[0])
        return total

    def _path_max_turn(points_latlon: list[tuple[float, float]]) -> float:
        if len(points_latlon) < 3:
            return 0.0
        turns = []
        for p0, p1, p2 in zip(points_latlon[:-2], points_latlon[1:-1], points_latlon[2:]):
            turns.append(
                _turn_angle_deg(
                    _bearing_deg(p0[0], p0[1], p1[0], p1[1]),
                    _bearing_deg(p1[0], p1[1], p2[0], p2[1]),
                )
            )
        return float(max(turns)) if turns else 0.0

    direct_pilot = float(haversine_distance_m(np.array([pilot_lat]), np.array([pilot_lon]), np.array([cand_lat]), np.array([cand_lon]))[0])
    direct_vessel = float(haversine_distance_m(np.array([vessel_lat]), np.array([vessel_lon]), np.array([cand_lat]), np.array([cand_lon]))[0])
    direct_avg = (direct_pilot + direct_vessel) / 2.0

    if not waypoints:
        pilot_buoy_clear, pilot_break_clear = _route_hazard_metrics(_pilot_leg())
        vessel_buoy_clear, vessel_break_clear = _route_hazard_metrics(_ship_leg())
        pilot_water_cov, pilot_water_edge = _route_water_metrics(_pilot_leg())
        vessel_water_cov, vessel_water_edge = _route_water_metrics(_ship_leg())
        min_buoy_clear = np.nanmin([pilot_buoy_clear, vessel_buoy_clear]) if not (pd.isna(pilot_buoy_clear) and pd.isna(vessel_buoy_clear)) else np.nan
        min_break_clear = np.nanmin([pilot_break_clear, vessel_break_clear]) if not (pd.isna(pilot_break_clear) and pd.isna(vessel_break_clear)) else np.nan
        water_cov = min(pilot_water_cov, vessel_water_cov)
        min_water_edge = np.nanmin([pilot_water_edge, vessel_water_edge]) if not (pd.isna(pilot_water_edge) and pd.isna(vessel_water_edge)) else np.nan
        route_hazard_score = 1.0
        if pd.notna(min_buoy_clear):
            route_hazard_score = min(route_hazard_score, max(0.0, min(1.0, float(min_buoy_clear) / 1200.0)))
        if pd.notna(min_break_clear):
            route_hazard_score = min(route_hazard_score, max(0.0, min(1.0, float(min_break_clear) / 1000.0)))
        route_hazard_score = min(route_hazard_score, water_cov)
        if pd.notna(min_water_edge):
            route_hazard_score = min(route_hazard_score, max(0.0, min(1.0, float(min_water_edge) / 250.0)))
        return {
            "waypoint_source": "direct",
            "waypoint_lat": np.nan,
            "waypoint_lon": np.nan,
            "avg_route_distance_m": direct_avg,
            "route_detour_ratio": 1.0,
            "max_turn_angle_deg": 0.0,
            "route_path_score": max(0.0, min(1.0, 1.0 - direct_avg / 8000.0)),
            "route_min_buoy_clearance_m": min_buoy_clear,
            "route_min_breakwater_clearance_m": min_break_clear,
            "route_water_coverage_ratio": water_cov,
            "route_min_water_edge_clearance_m": min_water_edge,
            "route_hazard_score": route_hazard_score,
            "route_waypoints_json": "[]",
            "route_waypoint_count": 0,
        }

    pilot_points = [(pilot_lat, pilot_lon)] + waypoints + [(cand_lat, cand_lon)]
    vessel_points = [(vessel_lat, vessel_lon)] + waypoints + [(cand_lat, cand_lon)]
    avg_route = (_path_distance(pilot_points) + _path_distance(vessel_points)) / 2.0
    detour_ratio = avg_route / max(direct_avg, 1.0)
    max_turn = max(_path_max_turn(pilot_points), _path_max_turn(vessel_points))

    detour_penalty = min(max(detour_ratio - 1.0, 0.0), 0.6)
    turn_penalty = min(max_turn / 150.0, 0.8)
    pilot_buoy_clear, pilot_break_clear = _route_hazard_metrics(_pilot_leg())
    vessel_buoy_clear, vessel_break_clear = _route_hazard_metrics(_ship_leg())
    pilot_water_cov, pilot_water_edge = _route_water_metrics(_pilot_leg())
    vessel_water_cov, vessel_water_edge = _route_water_metrics(_ship_leg())
    min_buoy_clear = np.nanmin([pilot_buoy_clear, vessel_buoy_clear]) if not (pd.isna(pilot_buoy_clear) and pd.isna(vessel_buoy_clear)) else np.nan
    min_break_clear = np.nanmin([pilot_break_clear, vessel_break_clear]) if not (pd.isna(pilot_break_clear) and pd.isna(vessel_break_clear)) else np.nan
    water_cov = min(pilot_water_cov, vessel_water_cov)
    min_water_edge = np.nanmin([pilot_water_edge, vessel_water_edge]) if not (pd.isna(pilot_water_edge) and pd.isna(vessel_water_edge)) else np.nan
    route_hazard_score = 1.0
    if pd.notna(min_buoy_clear):
        route_hazard_score = min(route_hazard_score, max(0.0, min(1.0, float(min_buoy_clear) / 1200.0)))
    if pd.notna(min_break_clear):
        route_hazard_score = min(route_hazard_score, max(0.0, min(1.0, float(min_break_clear) / 1000.0)))
    route_hazard_score = min(route_hazard_score, water_cov)
    if pd.notna(min_water_edge):
        route_hazard_score = min(route_hazard_score, max(0.0, min(1.0, float(min_water_edge) / 250.0)))
    # Extra legs are only worthwhile if they remain operationally reasonable.
    if detour_ratio > 1.20:
        route_hazard_score *= 0.55
    if max_turn > 120.0:
        route_hazard_score *= 0.35
    elif max_turn > 90.0:
        route_hazard_score *= 0.65
    route_score = max(0.0, min(1.0, (1.0 - 0.50 * detour_penalty - 0.50 * turn_penalty) * route_hazard_score))
    wp_lat, wp_lon = waypoints[0]
    return {
        "waypoint_source": "staged",
        "waypoint_lat": wp_lat,
        "waypoint_lon": wp_lon,
        "avg_route_distance_m": avg_route,
        "route_detour_ratio": detour_ratio,
        "max_turn_angle_deg": max_turn,
        "route_path_score": route_score,
        "route_min_buoy_clearance_m": min_buoy_clear,
        "route_min_breakwater_clearance_m": min_break_clear,
        "route_water_coverage_ratio": water_cov,
        "route_min_water_edge_clearance_m": min_water_edge,
        "route_hazard_score": route_hazard_score,
        "route_waypoints_json": json.dumps([[float(lat), float(lon)] for lat, lon in waypoints]),
        "route_waypoint_count": len(waypoints),
    }


def generate_candidate_points(
    event: pd.Series,
    rewind_pilot: pd.Series,
    rewind_vessel: pd.Series,
    historical_events: pd.DataFrame,
    buoys: pd.DataFrame | None = None,
    step_m: float = 300.0,
) -> pd.DataFrame:
    zones = load_zone_lookup()
    preferred_zone = _preferred_zone(event.get("traffic_direction"), zones)
    candidates: list[dict[str, object]] = []

    hotspot = _historical_hotspot(historical_events, event)
    if hotspot is not None:
        candidates.append({"candidate_source": "historical_hotspot", "lat": hotspot[0], "lon": hotspot[1]})

    centroid = preferred_zone.centroid
    candidates.append({"candidate_source": "zone_centroid", "lat": float(centroid.y), "lon": float(centroid.x)})

    for lat, lon in _sample_points_in_polygon(preferred_zone, step_m=step_m):
        candidates.append({"candidate_source": "zone_grid", "lat": lat, "lon": lon})

    for lat, lon in _buoy_centerline_candidates(
        preferred_zone=preferred_zone,
        buoys=buoys,
        vessel_lat=float(rewind_vessel["lat"]),
        actual_lat=event.get("meeting_lat"),
    ):
        candidates.append({"candidate_source": "fairway_centerline", "lat": lat, "lon": lon})

    for lat, lon in _forward_corridor_candidates(
        vessel_lat=float(rewind_vessel["lat"]),
        vessel_lon=float(rewind_vessel["lon"]),
        actual_lat=event.get("meeting_lat"),
        actual_lon=event.get("meeting_lon"),
        preferred_zone=preferred_zone,
        step_m=step_m,
    ):
        candidates.append({"candidate_source": "forward_corridor", "lat": lat, "lon": lon})

    mid_lat = float(rewind_pilot["lat"] + rewind_vessel["lat"]) / 2.0
    mid_lon = float(rewind_pilot["lon"] + rewind_vessel["lon"]) / 2.0
    candidates.append({"candidate_source": "rewind_midpoint", "lat": mid_lat, "lon": mid_lon})

    if pd.notna(event.get("anchor_lat")):
        a_lat, a_lon = float(event["anchor_lat"]), float(event["anchor_lon"])
        candidates.append({"candidate_source": "ml_anchor", "lat": a_lat, "lon": a_lon})
        # Local ring so a small relocation is possible when the anchor itself fails a gate.
        for r_m in ANCHOR_RING_M:
            for brg in range(0, 360, 45):
                la, lo = _walk(a_lat, a_lon, brg, r_m)
                candidates.append({"candidate_source": "anchor_ring", "lat": la, "lon": lo})
    elif pd.notna(event.get("meeting_lat")) and pd.notna(event.get("meeting_lon")):
        candidates.append({"candidate_source": "actual_meeting_point", "lat": float(event["meeting_lat"]), "lon": float(event["meeting_lon"])})

    cand = pd.DataFrame(candidates).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    if cand.empty:
        return cand
    cand["inside_study_area"] = [zones["study_area"].contains(Point(lon, lat)) for lat, lon in zip(cand["lat"], cand["lon"])]
    cand = cand[cand["inside_study_area"]].reset_index(drop=True)
    return cand


def _weather_at_ts(weather: pd.DataFrame, ts: pd.Timestamp) -> dict[str, float]:
    if weather.empty:
        return {
            "wind_speed_mps": np.nan,
            "wind_dir_deg": np.nan,
            "wave_height_m": np.nan,
            "tide_height_m": np.nan,
            "visibility_km": np.nan,
            "current_speed_kn": np.nan,
            "current_dir_deg": np.nan,
        }
    hour = ts.floor("h")
    hit = weather[weather["meeting_hour"] == hour]
    if hit.empty:
        return {
            "wind_speed_mps": np.nan,
            "wind_dir_deg": np.nan,
            "wave_height_m": np.nan,
            "tide_height_m": np.nan,
            "visibility_km": np.nan,
            "current_speed_kn": np.nan,
            "current_dir_deg": np.nan,
        }
    row = hit.iloc[0]
    return {
        "wind_speed_mps": float(row.get("wind_speed_mps", np.nan)),
        "wind_dir_deg": float(row.get("wind_dir_deg", np.nan)),
        "wave_height_m": float(row.get("wave_height_m", np.nan)),
        "tide_height_m": float(row.get("tide_height_m", np.nan)),
        "visibility_km": float(row.get("visibility_km", np.nan)),
        "current_speed_kn": float(row.get("current_speed_kn", np.nan)),
        "current_dir_deg": float(row.get("current_dir_deg", np.nan)),
    }


def score_candidates(
    candidates: pd.DataFrame,
    rewind_time: pd.Timestamp,
    rewind_pilot: pd.Series,
    rewind_vessel: pd.Series,
    context_window: pd.DataFrame,
    route_waypoint_seed: tuple[float, float] | None = None,
    vessel_track: list[tuple[float, float]] | None = None,
    pilot_track: list[tuple[float, float]] | None = None,
    actual_meeting_lat: float | None = None,
    actual_meeting_lon: float | None = None,
    source_bonus_on: bool = True,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    buoys = _read_buoys()
    fairways = _load_fairway_geometries()
    breakwaters = _load_breakwater_geometries()
    water_polygons = _load_water_polygons()
    bathy_sampler, bathy_source = _load_bathymetry_sampler()
    weather = _read_weather_hourly()
    collision_exclusions = _collision_exclusion_mmsi()

    vessel_draft = pd.to_numeric(pd.Series([rewind_vessel.get("draft_m")]), errors="coerce").iloc[0]
    vessel_loa = pd.to_numeric(pd.Series([rewind_vessel.get("loa_est_m")]), errors="coerce").iloc[0]
    vessel_beam = pd.to_numeric(pd.Series([rewind_vessel.get("beam_est_m")]), errors="coerce").iloc[0]
    weather_now = _weather_at_ts(weather, rewind_time)
    local_hour_kst = rewind_time.hour  # AIS/KMA timestamps are KST wall-clock labelled UTC; keep tables on that same clock
    night_flag = 1 if (local_hour_kst < 6 or local_hour_kst >= 18) else 0

    out_rows: list[dict[str, object]] = []
    third = context_window[~context_window["mmsi"].isin([rewind_pilot["mmsi"], rewind_vessel["mmsi"]])].copy()
    third = third[~third["mmsi"].astype(str).isin(collision_exclusions)].copy()
    if not third.empty:
        third["loa_est_m"] = third[[c for c in ["dim_a", "dim_b"] if c in third.columns]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        third["beam_est_m"] = third[[c for c in ["dim_c", "dim_d"] if c in third.columns]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)

    for row in candidates.itertuples(index=False):
        chart_depth_m = _sample_bathymetry(float(row.lat), float(row.lon), bathy_sampler)
        tide_height_m = weather_now["tide_height_m"]
        effective_depth_m = chart_depth_m if pd.isna(tide_height_m) else chart_depth_m + float(tide_height_m)
        ukc_m = compute_ukc(effective_depth_m, vessel_draft)
        ukc_risk = classify_ukc_risk(ukc_m)

        traffic = summarize_nearby_traffic(float(row.lat), float(row.lon), third, rewind_vessel)
        in_fairway, fairway_name, fairway_edge_m = _fairway_context(float(row.lat), float(row.lon), fairways)
        in_fairway_pos, fairway_name_pos, fairway_dist_m_pos, fairway_center_dist_m = _fairway_position_metrics(float(row.lat), float(row.lon), fairways)
        if pd.notna(fairway_name_pos):
            fairway_name = fairway_name_pos
        if not in_fairway and in_fairway_pos:
            in_fairway = True
            fairway_edge_m = fairway_dist_m_pos
        buoy_name, buoy_dist_m, buoy_type = _nearest_buoy(float(row.lat), float(row.lon), buoys)
        breakwater_name, breakwater_dist_m = _breakwater_distance_m(float(row.lat), float(row.lon), breakwaters)

        dist_pilot_m = float(haversine_distance_m(
            np.array([float(rewind_pilot["lat"])]),
            np.array([float(rewind_pilot["lon"])]),
            np.array([float(row.lat)]),
            np.array([float(row.lon)]),
        )[0])
        dist_vessel_m = float(haversine_distance_m(
            np.array([float(rewind_vessel["lat"])]),
            np.array([float(rewind_vessel["lon"])]),
            np.array([float(row.lat)]),
            np.array([float(row.lon)]),
        )[0])

        waypoint_options: list[tuple[str, object]] = [("direct", None)]
        midpoint_lat = float(rewind_pilot["lat"] + rewind_vessel["lat"]) / 2.0
        midpoint_lon = float(rewind_pilot["lon"] + rewind_vessel["lon"]) / 2.0
        midpoint_leg = ((midpoint_lat + float(row.lat)) / 2.0, (midpoint_lon + float(row.lon)) / 2.0)
        waypoint_options.append(("rewind_midpoint_leg", midpoint_leg))
        if route_waypoint_seed is not None:
            waypoint_options.append(("zone_seed", route_waypoint_seed))
            seed_mid = (
                (float(route_waypoint_seed[0]) + float(row.lat)) / 2.0,
                (float(route_waypoint_seed[1]) + float(row.lon)) / 2.0,
            )
            midpoint_seed = (
                (midpoint_lat + float(route_waypoint_seed[0])) / 2.0,
                (midpoint_lon + float(route_waypoint_seed[1])) / 2.0,
            )
            waypoint_options.append(("zone_seed_then_leg", [route_waypoint_seed, seed_mid]))
            waypoint_options.append(("mid_then_zone_seed", [midpoint_seed, route_waypoint_seed]))
            waypoint_options.append(("mid_seed_then_leg", [midpoint_seed, seed_mid]))

        route_eval_rows = []
        for wp_source, wp in waypoint_options:
            metrics = _route_option_metrics(
                pilot_lat=float(rewind_pilot["lat"]),
                pilot_lon=float(rewind_pilot["lon"]),
                vessel_lat=float(rewind_vessel["lat"]),
                vessel_lon=float(rewind_vessel["lon"]),
                cand_lat=float(row.lat),
                cand_lon=float(row.lon),
                waypoint=wp,
                buoys=buoys,
                breakwaters=breakwaters,
                water_polygons=water_polygons,
                vessel_track=vessel_track,
                pilot_track=pilot_track,
            )
            metrics["waypoint_option_source"] = wp_source
            route_eval_rows.append(metrics)
        route_eval = pd.DataFrame(route_eval_rows).sort_values(
            ["route_hazard_score", "route_min_buoy_clearance_m", "route_min_water_edge_clearance_m", "max_turn_angle_deg", "route_path_score", "avg_route_distance_m"],
            ascending=[False, False, False, True, False, True],
        ).reset_index(drop=True)
        best_route = route_eval.iloc[0]

        ukc_score = 0.0 if pd.isna(ukc_m) else max(0.0, min(1.0, float(ukc_m) / 8.0))
        traffic_score = max(0.0, min(1.0, 1.0 - float(traffic["surrounding_vessel_count_0_5nm"]) / 20.0))
        collision_score = {"low": 1.0, "medium": 0.5, "high": 0.1}.get(str(traffic["collision_risk_class"]), 0.25)
        fairway_score = 1.0 if in_fairway else max(0.0, min(1.0, 1.0 - float(fairway_edge_m) / 500.0 if pd.notna(fairway_edge_m) else 0.25))
        fairway_center_score = max(0.0, min(1.0, float(fairway_center_dist_m) / 250.0)) if (in_fairway and pd.notna(fairway_center_dist_m)) else 0.0
        buoy_clearance_score = 0.4 if pd.isna(buoy_dist_m) else max(0.0, min(1.0, float(buoy_dist_m) / 500.0))
        breakwater_clearance_score = 0.5 if pd.isna(breakwater_dist_m) else max(0.0, min(1.0, float(breakwater_dist_m) / 900.0))
        forward_progress_score, forward_progress_ratio, forward_progress_proj_m = _forward_progress_metrics(
            vessel_lat=float(rewind_vessel["lat"]),
            vessel_lon=float(rewind_vessel["lon"]),
            actual_lat=actual_meeting_lat,
            actual_lon=actual_meeting_lon,
            cand_lat=float(row.lat),
            cand_lon=float(row.lon),
        )
        nav_score = min(
            max(0.0, min(1.0, 1.0 - float(breakwater_dist_m) / 3000.0 if pd.notna(breakwater_dist_m) else 0.6)),
            max(0.0, min(1.0, float(buoy_dist_m) / 800.0 if pd.notna(buoy_dist_m) else 0.7)),
        )
        route_score = float(best_route["route_path_score"])
        route_hazard_score = float(best_route["route_hazard_score"])
        route_water_score = float(best_route["route_water_coverage_ratio"]) if pd.notna(best_route["route_water_coverage_ratio"]) else 1.0
        route_clearance_score = 1.0
        if pd.notna(best_route["route_min_buoy_clearance_m"]):
            route_clearance_score = min(route_clearance_score, max(0.0, min(1.0, float(best_route["route_min_buoy_clearance_m"]) / 1500.0)))
        if pd.notna(best_route["route_min_breakwater_clearance_m"]):
            route_clearance_score = min(route_clearance_score, max(0.0, min(1.0, float(best_route["route_min_breakwater_clearance_m"]) / 1200.0)))
        if pd.notna(best_route["route_min_water_edge_clearance_m"]):
            route_clearance_score = min(route_clearance_score, max(0.0, min(1.0, float(best_route["route_min_water_edge_clearance_m"]) / 300.0)))
        wind = weather_now["wind_speed_mps"]
        wave = weather_now["wave_height_m"]
        weather_score = 1.0
        if pd.notna(wind):
            weather_score -= min(float(wind) / 25.0, 0.6)
        if pd.notna(wave):
            weather_score -= min(float(wave) / 5.0, 0.4)
        visibility = weather_now["visibility_km"]
        if pd.notna(visibility) and visibility > 0:
            weather_score -= min(max(0.0, (5.0 - float(visibility)) / 5.0), 0.25)
        if night_flag:
            weather_score -= 0.08
        weather_score = max(0.0, min(1.0, weather_score))
        source_bonus = 0.0
        if not source_bonus_on:
            pass
        elif str(row.candidate_source) == "fairway_centerline":
            source_bonus = 0.04
        elif str(row.candidate_source) == "forward_corridor":
            source_bonus = 0.02

        overall = (
            0.24 * ukc_score
            + 0.14 * traffic_score
            + 0.15 * collision_score
            + 0.09 * fairway_score
            + 0.10 * fairway_center_score
            + 0.08 * forward_progress_score
            + 0.06 * buoy_clearance_score
            + 0.05 * breakwater_clearance_score
            + 0.04 * nav_score
            + 0.04 * route_score
            + 0.05 * route_hazard_score
            + 0.03 * route_clearance_score
            + 0.02 * route_water_score
            + 0.05 * weather_score
            + source_bonus
        )
        out_rows.append(
            {
                "candidate_source": row.candidate_source,
                "candidate_lat": float(row.lat),
                "candidate_lon": float(row.lon),
                "depth_m": effective_depth_m,
                "depth_source": bathy_source,
                "chart_depth_m": chart_depth_m,
                "tide_height_m": tide_height_m,
                "ukc_m": ukc_m,
                "ukc_risk_class": ukc_risk,
                "in_fairway": bool(in_fairway),
                "fairway_name": fairway_name,
                "dist_to_fairway_edge_m": fairway_edge_m,
                "nearest_buoy_name": buoy_name,
                "nearest_buoy_type": buoy_type,
                "dist_to_nearest_buoy_m": buoy_dist_m,
                "nearest_breakwater_name": breakwater_name,
                "dist_to_breakwater_m": breakwater_dist_m,
                "dist_pilot_to_candidate_m": dist_pilot_m,
                "dist_vessel_to_candidate_m": dist_vessel_m,
                "selected_waypoint_source": best_route["waypoint_option_source"],
                "selected_waypoint_lat": best_route["waypoint_lat"],
                "selected_waypoint_lon": best_route["waypoint_lon"],
                "route_waypoint_count": best_route["route_waypoint_count"],
                "route_waypoints_json": best_route["route_waypoints_json"],
                "selected_route_distance_m": best_route["avg_route_distance_m"],
                "selected_route_detour_ratio": best_route["route_detour_ratio"],
                "selected_route_max_turn_angle_deg": best_route["max_turn_angle_deg"],
                "route_min_buoy_clearance_m": best_route["route_min_buoy_clearance_m"],
                "route_min_breakwater_clearance_m": best_route["route_min_breakwater_clearance_m"],
                "route_water_coverage_ratio": best_route["route_water_coverage_ratio"],
                "route_min_water_edge_clearance_m": best_route["route_min_water_edge_clearance_m"],
                "route_hazard_score": route_hazard_score,
                "route_clearance_score": route_clearance_score,
                "route_water_score": route_water_score,
                "wind_speed_mps": weather_now["wind_speed_mps"],
                "wind_dir_deg": weather_now["wind_dir_deg"],
                "wave_height_m": weather_now["wave_height_m"],
                "visibility_km": weather_now["visibility_km"],
                "current_speed_kn": weather_now["current_speed_kn"],
                "current_dir_deg": weather_now["current_dir_deg"],
                "night_flag": night_flag,
                "local_hour_kst": local_hour_kst,
                "ukc_score": ukc_score,
                "traffic_score": traffic_score,
                "collision_score": collision_score,
                "fairway_score": fairway_score,
                "fairway_center_score": fairway_center_score,
                "buoy_clearance_score": buoy_clearance_score,
                "breakwater_clearance_score": breakwater_clearance_score,
                "forward_progress_score": forward_progress_score,
                "forward_progress_ratio": forward_progress_ratio,
                "forward_progress_proj_m": forward_progress_proj_m,
                "navigation_score": nav_score,
                "route_score": route_score,
                "weather_score": weather_score,
                "source_bonus": source_bonus,
                "candidate_score": overall,
                **traffic,
            }
        )
    return pd.DataFrame(out_rows).sort_values(["candidate_score", "ukc_m"], ascending=[False, False]).reset_index(drop=True)


def _breakwater_distance_m(lat: float, lon: float, breakwaters) -> tuple[object, float]:
    """Breakwater polygons when supplied, else the nearest New Port breakwater light:
    the same definition as the model target and the gate calibration (enrich_event_context)."""
    name, dist = _nearest_polygon_distance_m(lat, lon, breakwaters)
    if pd.isna(dist):
        name, dist, _ = _nearest_buoy(lat, lon, BREAKWATER_LIGHTS)
    return name, dist


WATER_GRID_M = 100.0
WATER_PAD_M = 2000.0


def water_router(starts: list[tuple[float, float]], targets: list[tuple[float, float]], water_polygons: dict):
    """Shortest paths through charted water (8-connected grid, WATER_GRID_M) from each start to any
    target, built once per event. Returns one function per start: (lat, lon) -> list of (lat, lon).
    Used in decision-time mode, where the craft's future AIS track may not be used."""
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    land = (water_polygons or {}).get("land_union")
    if land is None and (water_polygons or {}).get("land_prepared") is not None:
        land = water_polygons["land_prepared"].context
    pts = starts + targets
    lat0 = float(np.mean([p[0] for p in pts]))
    ky = 111320.0
    kx = 111320.0 * math.cos(math.radians(lat0))
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    la0, la1 = min(lats) - WATER_PAD_M / ky, max(lats) + WATER_PAD_M / ky
    lo0, lo1 = min(lons) - WATER_PAD_M / kx, max(lons) + WATER_PAD_M / kx
    step = WATER_GRID_M
    while ((la1 - la0) * ky / step) * ((lo1 - lo0) * kx / step) > 40000:  # long legs: coarsen to <= ~40k cells
        step *= 2
    ny = int(math.ceil((la1 - la0) * ky / step)) + 1
    nx = int(math.ceil((lo1 - lo0) * kx / step)) + 1
    glat = la0 + np.arange(ny) * step / ky
    glon = lo0 + np.arange(nx) * step / kx
    LON, LAT = np.meshgrid(glon, glat)
    flat_lat, flat_lon = LAT.ravel(), LON.ravel()
    if land is not None:
        shapely.prepare(land)  # in place, once per geometry; makes the point-in-land test fast
    wet = ~shapely.contains_xy(land, flat_lon, flat_lat) if land is not None else np.ones(flat_lat.size, bool)
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, wts = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[0:ny - dy, max(0, -dx):nx - max(0, dx)].ravel()
        b = idx[dy:ny, max(0, dx):nx + min(0, dx)].ravel()
        ok = wet[a] & wet[b]
        w = step * math.hypot(dx, dy)
        rows += [a[ok], b[ok]]; cols += [b[ok], a[ok]]; wts.append(np.full(2 * int(ok.sum()), w))
    G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nx, ny * nx)).tocsr()
    wet_idx = np.where(wet)[0]

    def cell(lat, lon):
        i = int(np.clip(round((lat - la0) * ky / step), 0, ny - 1))
        j = int(np.clip(round((lon - lo0) * kx / step), 0, nx - 1))
        c = int(idx[i, j])
        if not wet[c] and len(wet_idx):  # a start/target on charted land snaps to the nearest water cell
            c = int(wet_idx[np.argmin((flat_lat[wet_idx] - lat) ** 2 + ((flat_lon[wet_idx] - lon) * kx / ky) ** 2)])
        return c

    routers = []
    for slat, slon in starts:
        src = cell(slat, slon)
        dist, pred = dijkstra(G, indices=src, return_predecessors=True)

        def route(clat, clon, src=src, dist=dist, pred=pred, slat=slat, slon=slon):
            tgt = cell(clat, clon)
            if not np.isfinite(dist[tgt]):
                return [(slat, slon), (clat, clon)]  # unreachable through water: straight line (fails the water gate)
            path = [tgt]
            while path[-1] != src and pred[path[-1]] >= 0:
                path.append(int(pred[path[-1]]))
            path = path[::-1]
            return [(slat, slon)] + [(float(flat_lat[c]), float(flat_lon[c])) for c in path[1:-1]] + [(clat, clon)]
        routers.append(route)
    return routers


ANCHOR_RING_M = (150.0, 300.0, 500.0)
PROXIMITY_LAMBDA = 0.1  # score units per km moved from the anchor; sensitivity in diagnose_r2_optimiser.py


def _walk(lat: float, lon: float, bearing: float, dist_m: float) -> tuple[float, float]:
    b = math.radians(bearing)
    return _offset_point_m(lat, lon, dist_m * math.sin(b), dist_m * math.cos(b))


def ml_anchor_point(pred_dist_m: float, anchor_bearing_deg: float, breakwaters) -> tuple[float, float]:
    """Decision-time anchor: walk from the breakwater gate (mean of the breakwater
    lights) along the anchor bearing to the predicted breakwater distance (the model
    target). The bearing is learnt on the training fold (revision_r2_validation.py)."""
    gate = (float(BREAKWATER_LIGHTS["lat"].mean()), float(BREAKWATER_LIGHTS["lon"].mean()))
    la, lo = gate
    for k in range(1, 1200):
        la, lo = _walk(*gate, anchor_bearing_deg, k * 25.0)
        if _breakwater_distance_m(la, lo, breakwaters)[1] >= pred_dist_m:
            break
    return la, lo


def optimize_meeting_points(
    events: pd.DataFrame,
    master_path: Path,
    rewind_min: int = 20,
    candidate_step_m: float = 300.0,
    anchors: pd.DataFrame | None = None,
    candidate_log: list | None = None,
    proximity_lambda: float = PROXIMITY_LAMBDA,
    fold_gates: dict | None = None,
) -> pd.DataFrame:
    """anchors (event_id -> t_dec, lead_min, pred_dist_m, anchor_bearing_deg, fold) switches on
    decision-time mode: evaluate at the decision time t_dec (pilot-boat dispatch) with past-only AIS,
    water-routed legs and fold-calibrated gates, and replace every use of the observed meeting point by
    the ML anchor (the observed point is kept only for evaluation; lead_min is reported, never used)."""
    if events.empty:
        return pd.DataFrame()
    anchor_by_id = anchors.set_index("event_id").to_dict("index") if anchors is not None else None
    breakwaters_geom = _load_breakwater_geometries() if anchors is not None else None

    out_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        if pd.isna(getattr(event, "meeting_time", pd.NaT)):
            continue
        anchor = None
        if anchor_by_id is not None:
            anchor = anchor_by_id.get(getattr(event, "event_id"))
            if anchor is None:
                continue
            rewind_time = pd.Timestamp(anchor["t_dec"])
        else:
            rewind_time = pd.Timestamp(getattr(event, "meeting_time")) - pd.Timedelta(minutes=rewind_min)
        causal = anchor is not None
        time_min = rewind_time - pd.Timedelta(minutes=5)
        # Decision-time mode: nothing after the decision time is loaded (traffic, tracks, positions).
        time_max = rewind_time if causal else rewind_time + pd.Timedelta(minutes=5)
        meeting_ts = pd.Timestamp(getattr(event, "meeting_time"))
        track_end = rewind_time if causal else meeting_ts + pd.Timedelta(minutes=5)
        track_start = rewind_time - pd.Timedelta(minutes=15) if causal else time_min
        wide = _load_master_subset(master_path, min(time_min, track_start), max(time_max, track_end))
        if wide.empty:
            continue
        master = wide[(wide["datetime"] >= time_min) & (wide["datetime"] <= time_max)].copy()
        if master.empty:
            continue
        # Ship's observed AIS track from the rewind time to the meeting: its real,
        # channel-following route, used for the ship leg of the route gate.
        track = wide[wide["mmsi"] == str(getattr(event, "vessel_mmsi"))].sort_values("datetime")
        track = track[(track["datetime"] >= track_start) & (track["datetime"] <= track_end)]
        vessel_track = list(zip(pd.to_numeric(track["lat"], errors="coerce"), pd.to_numeric(track["lon"], errors="coerce")))
        vessel_track = [(la, lo) for la, lo in vessel_track if pd.notna(la) and pd.notna(lo)]
        ptr = wide[wide["mmsi"] == str(getattr(event, "pilot_mmsi"))].sort_values("datetime")
        ptr = ptr[(ptr["datetime"] >= track_start) & (ptr["datetime"] <= track_end)]
        pilot_track = [(float(la), float(lo)) for la, lo in zip(pd.to_numeric(ptr["lat"], errors="coerce"),
                       pd.to_numeric(ptr["lon"], errors="coerce")) if pd.notna(la) and pd.notna(lo)]
        for col in ["draft_m", "dim_a", "dim_b", "dim_c", "dim_d", "spd", "crs", "hdg"]:
            if col in master.columns:
                master[col] = pd.to_numeric(master[col], errors="coerce")
        master["loa_est_m"] = master[[c for c in ["dim_a", "dim_b"] if c in master.columns]].sum(axis=1, min_count=1)
        master["beam_est_m"] = master[[c for c in ["dim_c", "dim_d"] if c in master.columns]].sum(axis=1, min_count=1)

        pilot_rows = master[master["mmsi"] == str(getattr(event, "pilot_mmsi"))].copy()
        vessel_rows = master[master["mmsi"] == str(getattr(event, "vessel_mmsi"))].copy()
        rewind_pilot = _nearest_row(pilot_rows, rewind_time, backward=causal)
        rewind_vessel = _nearest_row(vessel_rows, rewind_time, backward=causal)
        if rewind_pilot is None or rewind_vessel is None:
            continue

        event_series = pd.Series(event._asdict())
        if anchor is not None:
            a_lat, a_lon = ml_anchor_point(float(anchor["pred_dist_m"]), float(anchor["anchor_bearing_deg"]), breakwaters_geom)
            event_series["anchor_lat"], event_series["anchor_lon"] = a_lat, a_lon
            event_series["decision_time"] = rewind_time
            # Candidate generation sees only the anchor, never the observed point.
            ref_lat, ref_lon = a_lat, a_lon
            # Forward progress is measured along the ship's own course at t_dec: an axis
            # ship -> anchor flips backwards once the ship has passed the anchor.
            fwd_lat, fwd_lon = ref_lat, ref_lon
            sog = pd.to_numeric(pd.Series([rewind_vessel.get("spd")]), errors="coerce").iloc[0]
            cog = pd.to_numeric(pd.Series([rewind_vessel.get("crs")]), errors="coerce").iloc[0]
            if pd.notna(sog) and pd.notna(cog) and sog >= 1.0 and cog < 360:
                reach = max(float(haversine_distance_m(np.array([float(rewind_vessel["lat"])]), np.array([float(rewind_vessel["lon"])]),
                                                       np.array([a_lat]), np.array([a_lon]))[0]), 500.0)
                fwd_lat, fwd_lon = _walk(float(rewind_vessel["lat"]), float(rewind_vessel["lon"]), float(cog), reach)
            if pd.notna(anchor.get("traffic_direction")):
                event_series["traffic_direction"] = anchor["traffic_direction"]  # logged direction
        else:
            ref_lat, ref_lon = event_series.get("meeting_lat"), event_series.get("meeting_lon")
            fwd_lat, fwd_lon = ref_lat, ref_lon
        global _GATE_OVERRIDE
        _GATE_OVERRIDE = (fold_gates or {}).get(str(int(anchor["fold"]))) if anchor is not None and "fold" in anchor and pd.notna(anchor["fold"]) else None
        gen_event = event_series.copy()
        gen_event["meeting_lat"], gen_event["meeting_lon"] = ref_lat, ref_lon
        buoys = _read_buoys()
        candidates = generate_candidate_points(
            gen_event,
            rewind_pilot,
            rewind_vessel,
            events,
            buoys=buoys,
            step_m=candidate_step_m,
        )
        zones = load_zone_lookup()
        preferred_zone = _preferred_zone(event_series.get("traffic_direction"), zones)
        route_waypoint_seed = (float(preferred_zone.centroid.y), float(preferred_zone.centroid.x))
        if causal and not candidates.empty:
            vessel_track, pilot_track = water_router(
                [(float(rewind_vessel["lat"]), float(rewind_vessel["lon"])), (float(rewind_pilot["lat"]), float(rewind_pilot["lon"]))],
                list(zip(candidates["lat"].astype(float), candidates["lon"].astype(float))), _load_water_polygons())
        scored = score_candidates(
            candidates,
            rewind_time,
            rewind_pilot,
            rewind_vessel,
            master,
            route_waypoint_seed=route_waypoint_seed,
            vessel_track=vessel_track,
            pilot_track=pilot_track,
            actual_meeting_lat=fwd_lat,
            actual_meeting_lon=fwd_lon,
            source_bonus_on=anchor is None,
        )
        if scored.empty:
            continue

        scored["traffic_direction"] = event_series.get("traffic_direction")  # read by the inbound astern gates
        gate_eval = scored.apply(lambda r: _candidate_gate_status(r), axis=1, result_type="expand")
        scored = scored.copy()
        scored["gate_pass"] = gate_eval[0].astype(bool)
        scored["gate_reason"] = gate_eval[1].astype("string")

        if anchor is not None:
            # Decision-time objective: safety score minus a penalty per km of relocation
            # away from the ML anchor (the predicted operational meeting point).
            scored["dist_to_anchor_m"] = haversine_distance_m(
                np.full(len(scored), float(event_series["anchor_lat"])), np.full(len(scored), float(event_series["anchor_lon"])),
                scored["candidate_lat"].to_numpy(float), scored["candidate_lon"].to_numpy(float))
            scored["safety_score"] = scored["candidate_score"]
            scored["candidate_score"] = scored["safety_score"] - proximity_lambda * scored["dist_to_anchor_m"] / 1000.0
            scored = scored.sort_values(["candidate_score", "ukc_m"], ascending=[False, False]).reset_index(drop=True)
        if candidate_log is not None:
            candidate_log.append(scored.assign(event_id=getattr(event, "event_id")))
        feasible = scored[scored["gate_pass"]].copy()
        best = feasible.iloc[0] if not feasible.empty else scored.iloc[0]
        status = "feasible" if not feasible.empty else "no_feasible"  # infeasible: diagnostic point only
        baseline = scored[scored["candidate_source"].isin(["historical_hotspot", "zone_centroid"])].head(1)
        baseline = baseline.iloc[0] if not baseline.empty else best
        recommended_safe, recommended_gate_reason = bool(best["gate_pass"]), str(best["gate_reason"])
        baseline_safe, baseline_gate_reason = _candidate_gate_status(baseline)
        recommendation_action = "keep_baseline" if (
            baseline["candidate_source"] == best["candidate_source"]
            and float(baseline["candidate_lat"]) == float(best["candidate_lat"])
            and float(baseline["candidate_lon"]) == float(best["candidate_lon"])
        ) else "replace_baseline"

        out_rows.append(
            {
                "event_id": getattr(event, "event_id"),
                "pilot_mmsi": getattr(event, "pilot_mmsi"),
                "vessel_mmsi": getattr(event, "vessel_mmsi"),
                "traffic_direction": getattr(event, "traffic_direction", pd.NA),
                "meeting_time": getattr(event, "meeting_time"),
                "rewind_time": rewind_time,
                "actual_meeting_lat": getattr(event, "meeting_lat", np.nan),
                "actual_meeting_lon": getattr(event, "meeting_lon", np.nan),
                "recommended_lat": best["candidate_lat"],
                "recommended_lon": best["candidate_lon"],
                "recommended_source": best["candidate_source"],
                "recommended_score": best["candidate_score"],
                "recommended_gate_pass": recommended_safe,
                "recommended_gate_reason": recommended_gate_reason,
                "recommended_ukc_m": best["ukc_m"],
                "recommended_ukc_risk_class": best["ukc_risk_class"],
                "recommended_chart_depth_m": best["chart_depth_m"],
                "recommended_tide_height_m": best["tide_height_m"],
                "recommended_collision_risk_class": best["collision_risk_class"],
                "recommended_surrounding_vessel_count_0_5nm": best["surrounding_vessel_count_0_5nm"],
                "recommended_in_fairway": best["in_fairway"],
                "recommended_dist_to_fairway_edge_m": best["dist_to_fairway_edge_m"],
                "recommended_dist_to_nearest_buoy_m": best["dist_to_nearest_buoy_m"],
                "recommended_dist_to_breakwater_m": best["dist_to_breakwater_m"],
                "recommended_wind_speed_mps": best["wind_speed_mps"],
                "recommended_wave_height_m": best["wave_height_m"],
                "recommended_visibility_km": best["visibility_km"],
                "recommended_current_speed_kn": best["current_speed_kn"],
                "recommended_night_flag": best["night_flag"],
                "recommended_waypoint_source": best["selected_waypoint_source"],
                "recommended_waypoint_lat": best["selected_waypoint_lat"],
                "recommended_waypoint_lon": best["selected_waypoint_lon"],
                "recommended_route_waypoint_count": best["route_waypoint_count"],
                "recommended_route_waypoints_json": best["route_waypoints_json"],
                "recommended_route_distance_m": best["selected_route_distance_m"],
                "recommended_route_detour_ratio": best["selected_route_detour_ratio"],
                "recommended_route_max_turn_angle_deg": best["selected_route_max_turn_angle_deg"],
                "recommended_route_min_buoy_clearance_m": best["route_min_buoy_clearance_m"],
                "recommended_route_min_breakwater_clearance_m": best["route_min_breakwater_clearance_m"],
                "recommended_route_hazard_score": best["route_hazard_score"],
                "baseline_lat": baseline["candidate_lat"],
                "baseline_lon": baseline["candidate_lon"],
                "baseline_source": baseline["candidate_source"],
                "baseline_score": baseline["candidate_score"],
                "baseline_gate_pass": baseline_safe,
                "baseline_gate_reason": baseline_gate_reason,
                "baseline_ukc_m": baseline["ukc_m"],
                "baseline_chart_depth_m": baseline["chart_depth_m"],
                "baseline_tide_height_m": baseline["tide_height_m"],
                "baseline_collision_risk_class": baseline["collision_risk_class"],
                "baseline_surrounding_vessel_count_0_5nm": baseline["surrounding_vessel_count_0_5nm"],
                "baseline_waypoint_source": baseline["selected_waypoint_source"],
                "baseline_waypoint_lat": baseline["selected_waypoint_lat"],
                "baseline_waypoint_lon": baseline["selected_waypoint_lon"],
                "baseline_route_waypoint_count": baseline["route_waypoint_count"],
                "baseline_route_waypoints_json": baseline["route_waypoints_json"],
                "baseline_route_min_buoy_clearance_m": baseline["route_min_buoy_clearance_m"],
                "baseline_route_min_breakwater_clearance_m": baseline["route_min_breakwater_clearance_m"],
                "baseline_route_hazard_score": baseline["route_hazard_score"],
                "recommendation_action": recommendation_action,
                "pilot_rewind_lat": float(rewind_pilot["lat"]),
                "pilot_rewind_lon": float(rewind_pilot["lon"]),
                "vessel_rewind_lat": float(rewind_vessel["lat"]),
                "vessel_rewind_lon": float(rewind_vessel["lon"]),
                "candidate_count": len(scored),
                "recommendation_status": status,
                "anchor_lat": event_series.get("anchor_lat", np.nan),
                "anchor_lon": event_series.get("anchor_lon", np.nan),
                "lead_min": anchor["lead_min"] if anchor is not None else rewind_min,
                "recommended_dist_to_anchor_m": best.get("dist_to_anchor_m", np.nan),
                "recommended_safety_score": best.get("safety_score", np.nan),
            }
        )
    return pd.DataFrame(out_rows)


def _read_anchors(path: str) -> pd.DataFrame:
    a = pd.read_csv(path)
    a["t_dec"] = pd.to_datetime(a["t_dec"]).dt.tz_localize("UTC")  # KST wall-clock labelled UTC, as the AIS
    return a


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize dynamic pilot-transfer meeting points before detected meeting time")
    parser.add_argument("--events", type=str, required=True, help="Enriched event context CSV path")
    parser.add_argument("--master", type=str, default=None, help="AIS master path")
    parser.add_argument("--rewind-min", type=int, default=20)
    parser.add_argument("--candidate-step-m", type=float, default=300.0)
    parser.add_argument("--day", type=str, default=None, help="Optional YYYY-MM-DD filter on meeting_time")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of events after filtering")
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument("--proximity-lambda", type=float, default=PROXIMITY_LAMBDA)
    parser.add_argument("--fold-gates", type=str, default=None,
                        help="JSON with gate thresholds calibrated on each training fold (calibrate_r2_gates.py --folds)")
    parser.add_argument("--dump-candidates", action="store_true", help="Also save every scored candidate")
    parser.add_argument("--anchors", type=str, default=None,
                        help="Decision-time mode: CSV with event_id, t_dec, lead_min, pred_dist_m, anchor_bearing_deg")
    args = parser.parse_args()

    events_path = Path(args.events)
    events = load_events(events_path)
    if args.day and "meeting_time" in events.columns:
        day_start = pd.Timestamp(args.day, tz="UTC")
        day_end = day_start + pd.Timedelta(days=1)
        events = events[(events["meeting_time"] >= day_start) & (events["meeting_time"] < day_end)].copy()
    if args.limit is not None and args.limit >= 0:
        events = events.head(int(args.limit)).copy()
    start_date = events["meeting_time"].min().isoformat() if "meeting_time" in events.columns and events["meeting_time"].notna().any() else None
    end_date = events["meeting_time"].max().isoformat() if "meeting_time" in events.columns and events["meeting_time"].notna().any() else None
    master_path = Path(args.master) if args.master else _pick_master_path(None, start_date=start_date, end_date=end_date)
    out = optimize_meeting_points(
        events,
        master_path=master_path,
        rewind_min=args.rewind_min,
        candidate_step_m=args.candidate_step_m,
        anchors=_read_anchors(args.anchors) if args.anchors else None,
        candidate_log=(log := []) if args.dump_candidates else None,
        proximity_lambda=args.proximity_lambda,
        fold_gates=json.loads(Path(args.fold_gates).read_text())["folds"] if args.fold_gates else None,
    )
    if args.dump_candidates and log:
        pd.concat(log, ignore_index=True).to_csv(OUT_DIR / f"candidates_{args.tag}.csv", index=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"optimized_meeting_points_{args.tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved optimized meeting points: {out_path}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
