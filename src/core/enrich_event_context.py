from __future__ import annotations
import functools

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from shapely.geometry import Point, Polygon, shape

from src.core.detect_interceptions import service_craft_mmsi, haversine_distance_m, load_zone_lookup
from src.core.detect_interceptions import _pick_master_path, _norm_mmsi, LEGACY_TUG_BOATS, load_pilot_mmsi
from src.core.collision_risk import summarize_nearby_traffic
from src.core.prepare_ais_master import STATIC_FILE, load_static
from src.core.ukc_risk import classify_ukc_risk, compute_ukc, required_ship_domain_area_m2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROCESSED_DIR / "interactions"
ENC_GEOJSON_DIR = PROCESSED_DIR / "charts_enc_geojson"

SUPPLEMENTAL_NAV_AIDS = [
    {
        "name": "Busan New Port East Breakwater West Light",
        "lat": 35.04794444444445,
        "lon": 128.78744444444445,
        "type": "breakwater_light",
    },
    {
        "name": "Busan New Port East Breakwater East Light",
        "lat": 35.047972222222224,
        "lon": 128.7938611111111,
        "type": "breakwater_light",
    },
    {
        "name": "Busan New Port West Breakwater Light",
        "lat": 35.05863888888889,
        "lon": 128.77822222222222,
        "type": "breakwater_light",
    },
    {
        "name": "Busan New Port Seocon Halla South Light",
        "lat": 35.06538888888889,
        "lon": 128.77497222222223,
        "type": "breakwater_light",
    },
]

BREAKWATER_LIGHTS = pd.DataFrame([a for a in SUPPLEMENTAL_NAV_AIDS if a["type"] == "breakwater_light"])

EVENT_CONTEXT_COLUMNS = [
    "event_id",
    "pilot_mmsi",
    "vessel_mmsi",
    "pilot_ship_name",
    "vessel_ship_name",
    "vessel_ship_type_class",
    "traffic_direction",
    "start_time",
    "meeting_time",
    "end_time",
    "duration_min",
    "meeting_lat",
    "meeting_lon",
    "centroid_lat",
    "centroid_lon",
    "transfer_zone_class",
    "inside_static_zone",
    "inside_embark_zone",
    "inside_disembark_zone",
    "min_distance_m",
    "avg_distance_m",
    "max_distance_m",
    "min_antenna_distance_m",
    "avg_antenna_distance_m",
    "min_center_distance_m",
    "min_domain_clearance_m",
    "avg_rel_speed_kn",
    "avg_heading_diff_deg",
    "course_aligned_share",
    "speed_similar_share",
    "boarding_operation_share",
    "dominant_rule_tag",
    "confidence_score",
    "likely_transfer",
    "escort_phase_duration_min",
    "approach_phase_duration_min",
    "alongside_phase_duration_min",
    "separation_phase_duration_min",
    "opening_distance_5min_after_transfer_m",
    "opening_rate_m_per_min",
    "surrounding_vessel_count_0_5nm",
    "surrounding_vessel_count_1_0nm",
    "nearest_third_vessel_mmsi",
    "nearest_third_vessel_distance_m",
    "nearest_third_vessel_center_cpa_m",
    "nearest_third_vessel_cpa_m",
    "nearest_third_vessel_hull_cpa_m",
    "nearest_third_vessel_domain_clearance_m",
    "nearest_third_vessel_tcpa_min",
    "collision_risk_score",
    "collision_risk_class",
    "depth_m",
    "depth_source",
    "vessel_draft_m",
    "pilot_draft_m",
    "ukc_m",
    "ukc_ratio",
    "ukc_risk_class",
    "in_fairway",
    "fairway_name",
    "dist_to_fairway_edge_m",
    "dist_to_nearest_buoy_m",
    "nearest_buoy_name",
    "nearest_buoy_type",
    "dist_to_breakwater_m",
    "dist_to_restricted_area_m",
    "sea_room_score",
    "navigation_support_score",
    "wind_speed_mps",
    "wind_dir_deg",
    "wave_height_m",
    "visibility_km",
    "tide_height_m",
    "current_speed_kn",
    "current_dir_deg",
    "weather_exposure_score",
    "shelter_score",
    "night_day_flag",
    "local_hour_kst",
    "transfer_side_exposure",
    "escape_corridor_availability",
    "distance_to_open_sea_route_m",
    "time_to_conflict_if_aborted_min",
    "domain_overlap_fixed_hazards",
    "transfer_feasibility_score",
    "go_no_go",
    "go_no_go_reason",
    "required_ship_domain_m2",
    "available_sea_room_m2",
    "domain_clearance_ratio",
    "emergency_turn_feasible",
    "emergency_escape_route_feasible",
    "mob_recovery_feasible",
    "trajectory_restoration_feasible",
]


def _clip_score(value: float | int | None) -> float:
    if pd.isna(value):
        return np.nan
    return float(max(0.0, min(1.0, float(value))))


def _navigation_support_score(
    buoy_type: object,
    buoy_distance_m: float | int | None,
    in_fairway: bool,
    fairway_distance_m: float | int | None,
    breakwater_distance_m: float | int | None,
) -> float:
    score = 0.15

    if in_fairway:
        score += 0.35
    elif pd.notna(fairway_distance_m):
        score += 0.2 * max(0.0, 1.0 - float(fairway_distance_m) / 1500.0)

    if pd.notna(buoy_distance_m):
        score += 0.2 * max(0.0, 1.0 - float(buoy_distance_m) / 1200.0)

    if buoy_type == "breakwater_light":
        score += 0.1
    elif buoy_type == "buoy":
        score += 0.05

    if pd.notna(breakwater_distance_m):
        # Near a breakwater light can imply some shelter, but too close is hazardous.
        dist = float(breakwater_distance_m)
        if 300.0 <= dist <= 2500.0:
            score += 0.2 * (1.0 - abs(dist - 1200.0) / 1200.0)
        elif dist < 150.0:
            score -= 0.15

    return _clip_score(score)


def _go_no_go_decision(row: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    review_flags: list[str] = []

    collision_risk_class = row.get("collision_risk_class")
    if pd.isna(collision_risk_class):
        collision_risk_class = None

    ukc_risk_class = row.get("ukc_risk_class")
    if pd.isna(ukc_risk_class):
        ukc_risk_class = None

    hazard_overlap = row.get("domain_overlap_fixed_hazards")
    if pd.isna(hazard_overlap):
        hazard_overlap = None

    likely_transfer = row.get("likely_transfer")
    likely_transfer_flag = False if pd.isna(likely_transfer) else bool(likely_transfer)

    if collision_risk_class == "high":
        reasons.append("collision_risk_high")
    if ukc_risk_class == "critical":
        reasons.append("ukc_critical")

    breakwater_dist = pd.to_numeric(pd.Series([row.get("dist_to_breakwater_m")]), errors="coerce").iloc[0]
    if pd.notna(breakwater_dist) and breakwater_dist < 250.0:
        reasons.append("too_close_breakwater")

    if hazard_overlap == "possible":
        review_flags.append("hazard_overlap_possible")

    if collision_risk_class == "medium":
        review_flags.append("collision_risk_medium")
    if ukc_risk_class == "caution":
        review_flags.append("ukc_caution")

    nav_support = pd.to_numeric(pd.Series([row.get("navigation_support_score")]), errors="coerce").iloc[0]
    if pd.notna(nav_support) and nav_support < 0.3:
        review_flags.append("weak_navigation_support")

    sea_room = pd.to_numeric(pd.Series([row.get("sea_room_score")]), errors="coerce").iloc[0]
    if pd.notna(sea_room) and sea_room < 0.35:
        review_flags.append("limited_sea_room")

    if reasons:
        return "no_go", ";".join(reasons)

    if not likely_transfer_flag:
        review_flags.append("detection_not_confirmed")

    if review_flags:
        return "review", ";".join(review_flags)

    return "go", "context_enriched"


def _meeting_point_key(lat: object, lon: object, precision: int = 5) -> tuple[float, float]:
    return (round(float(lat), precision), round(float(lon), precision))


def _collision_exclusion_mmsi() -> set[str]:
    return set(service_craft_mmsi())


def _read_weather_hourly() -> pd.DataFrame:
    weather_frames: list[pd.DataFrame] = []

    legacy_path = RAW_DIR / "metocean" / "wind&wave.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        legacy["meeting_hour"] = pd.to_datetime(legacy.get("Date_Time"), errors="coerce", utc=True).dt.floor("h")
        legacy = legacy.dropna(subset=["meeting_hour"]).copy()
        legacy = legacy.rename(
            columns={
                "Wind_Speed_mps": "wind_speed_mps",
                "Wind_Direction_deg": "wind_dir_deg",
                "Significant_Wave_Height_m": "wave_height_m",
            }
        )
        weather_frames.append(legacy[["meeting_hour", "wind_speed_mps", "wind_dir_deg", "wave_height_m"]])

    buoy_2024_path = RAW_DIR / "metocean" / "2024tanverSentOBS_BUOY_TIM_20250731121556.csv"
    if buoy_2024_path.exists():
        buoy_2024 = pd.read_csv(buoy_2024_path)
        buoy_2024["meeting_hour"] = pd.to_datetime(buoy_2024.get("Date_Time"), errors="coerce", utc=True).dt.floor("h")
        buoy_2024 = buoy_2024.dropna(subset=["meeting_hour"]).copy()
        buoy_2024 = buoy_2024.rename(
            columns={
                "Wind_Speed_mps": "wind_speed_mps",
                "Wind_Direction_deg": "wind_dir_deg",
                "Significant_Wave_Height_m": "wave_height_m",
            }
        )
        weather_frames.append(buoy_2024[["meeting_hour", "wind_speed_mps", "wind_dir_deg", "wave_height_m"]])

    if weather_frames:
        weather = pd.concat(weather_frames, ignore_index=True)
        weather = weather.sort_values("meeting_hour").drop_duplicates(subset=["meeting_hour"], keep="last")
    else:
        weather = pd.DataFrame(columns=["meeting_hour", "wind_speed_mps", "wind_dir_deg", "wave_height_m"])

    tide_book_path = RAW_DIR / "metocean" / "Busan_Tides_with_Visibility_and_Transitions.xlsx"
    if tide_book_path.exists():
        tide = pd.read_excel(tide_book_path, sheet_name="Hourly_Tides")
        date_str = tide.get("Date", pd.Series(dtype="object")).astype("string").fillna("")
        time_str = tide.get("Time", pd.Series(dtype="object")).astype("string").fillna("")
        tide["meeting_hour"] = pd.to_datetime(
            (date_str + " " + time_str).str.strip(),
            errors="coerce",
        ).dt.tz_localize("UTC").dt.floor("h")  # AIS/KMA timestamps are KST wall-clock labelled UTC; keep tables on that same clock
        tide["tide_height_m"] = pd.to_numeric(tide.get("Height (m)"), errors="coerce")
        if "Light Condition" in tide.columns:
            tide["light_condition"] = tide["Light Condition"].astype("string").str.strip().str.lower()
        tide = tide.dropna(subset=["meeting_hour"]).copy()
        tide = tide.groupby("meeting_hour", as_index=False).agg(
            tide_height_m=("tide_height_m", "mean"),
            light_condition=("light_condition", "first"),
        )
        weather = weather.merge(tide, on="meeting_hour", how="outer")
    else:
        if "tide_height_m" not in weather.columns:
            weather["tide_height_m"] = np.nan
        if "light_condition" not in weather.columns:
            weather["light_condition"] = pd.NA

    legacy_tide_path = RAW_DIR / "metocean" / "hourly_tide_heights.csv"
    if legacy_tide_path.exists():
        legacy_tide = pd.read_csv(legacy_tide_path)
        legacy_tide["hour_local"] = (
            pd.to_numeric(legacy_tide.get("time"), errors="coerce").fillna(0).astype(int) // 10000
        )
        legacy_tide["legacy_tide_height_m"] = pd.to_numeric(legacy_tide.get("height"), errors="coerce")
        legacy_tide = legacy_tide.dropna(subset=["hour_local", "legacy_tide_height_m"]).copy()
        legacy_tide = legacy_tide.groupby("hour_local", as_index=False).agg(
            legacy_tide_height_m=("legacy_tide_height_m", "mean")
        )
        weather["hour_local"] = weather["meeting_hour"].dt.hour  # AIS/KMA timestamps are KST wall-clock labelled UTC; keep tables on that same clock
        weather = weather.merge(legacy_tide, on="hour_local", how="left")
        weather["tide_height_m"] = pd.to_numeric(weather.get("tide_height_m"), errors="coerce")
        weather["tide_height_m"] = weather["tide_height_m"].fillna(weather["legacy_tide_height_m"])
        weather = weather.drop(columns=["legacy_tide_height_m", "hour_local"], errors="ignore")

    for col in ["visibility_km", "current_speed_kn", "current_dir_deg", "light_condition"]:
        if col not in weather.columns:
            weather[col] = pd.NA if col == "light_condition" else np.nan

    keep_cols = [
        "meeting_hour",
        "wind_speed_mps",
        "wind_dir_deg",
        "wave_height_m",
        "tide_height_m",
        "light_condition",
        "visibility_km",
        "current_speed_kn",
        "current_dir_deg",
    ]
    return weather[keep_cols]


def _parse_dms_coordinate(value: object, axis: str) -> float:
    if pd.isna(value):
        raise ValueError("empty coordinate")
    s = str(value).strip().upper()
    compact = re.sub(r"\s+", "", s)
    match = re.match(r"^(\d{1,3})\D+(\d{1,2})\D+(\d{1,2}(?:\.\d+)?)\D*([NSEW])$", compact)
    if match:
        deg = float(match.group(1))
        minute = float(match.group(2))
        second = float(match.group(3))
        hemi = match.group(4)
        decimal = deg + minute / 60.0 + second / 3600.0
        if hemi in ("S", "W"):
            decimal *= -1.0
    else:
        decimal = float(compact)
    if axis == "lat" and not -90 <= decimal <= 90:
        raise ValueError("lat out of range")
    if axis == "lon" and not -180 <= decimal <= 180:
        raise ValueError("lon out of range")
    return float(decimal)


def _read_buoys() -> pd.DataFrame:
    path = RAW_DIR / "reference" / "buoys.xlsx"
    if not path.exists():
        return pd.DataFrame(SUPPLEMENTAL_NAV_AIDS)
    try:
        buoys = pd.read_excel(path)
    except Exception:
        return pd.DataFrame(SUPPLEMENTAL_NAV_AIDS)

    cols = {c.lower().strip(): c for c in buoys.columns}
    lat_col = cols.get("lat")
    lon_col = cols.get("lon") or cols.get("long")
    name_col = cols.get("name") or cols.get("name/no")
    side_col = cols.get("side")
    if not lat_col or not lon_col:
        return pd.DataFrame(SUPPLEMENTAL_NAV_AIDS)
    rows = []
    for _, r in buoys.iterrows():
        try:
            lat = _parse_dms_coordinate(r[lat_col], "lat")
            lon = _parse_dms_coordinate(r[lon_col], "lon")
        except Exception:
            lat = pd.to_numeric(pd.Series([r[lat_col]]), errors="coerce").iloc[0]
            lon = pd.to_numeric(pd.Series([r[lon_col]]), errors="coerce").iloc[0]
        if pd.notna(lat) and pd.notna(lon):
            rows.append(
                {
                    "name": r[name_col] if name_col else "buoy",
                    "lat": float(lat),
                    "lon": float(lon),
                    "type": "buoy",
                    "side": str(r[side_col]).strip() if side_col else pd.NA,
                }
            )
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(SUPPLEMENTAL_NAV_AIDS)], ignore_index=True).drop_duplicates(
        subset=["name", "lat", "lon"],
        keep="first",
    )


def _load_breakwater_geometries() -> list[tuple[str, Polygon]]:
    root = PROJECT_ROOT.parent / "6.4 pilotManuveringTrajectory" / "breakwaters"
    geoms: list[tuple[str, Polygon]] = []
    if not root.exists():
        return geoms
    for fp in root.glob("*.csv"):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if {"ORDER", "X", "Y"}.issubset(df.columns):
            df = df.sort_values("ORDER")
            coords = [(float(y), float(x)) for x, y in zip(df["X"], df["Y"])]
        else:
            continue
        if len(coords) >= 3:
            geoms.append((fp.stem, Polygon(coords)))
    return geoms


def _load_fairway_geometries() -> list[tuple[str, object]]:
    if not ENC_GEOJSON_DIR.exists():
        return []

    geoms: list[tuple[str, object]] = []
    for fp in ENC_GEOJSON_DIR.glob("*.geojson"):
        name = fp.name.upper()
        if not any(tag in name for tag in ["FAIRWY", "TSSBND", "TSELNE", "RECTRC", "NAVLNE"]):
            continue
        try:
            import json

            gj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for feat in gj.get("features", []):
            geom = feat.get("geometry")
            if not geom:
                continue
            try:
                geoms.append((fp.stem, shape(geom)))
            except Exception:
                continue
    return geoms


def _normalize_depth_value(value: object) -> float | None:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return None
    depth = float(abs(num) if float(num) <= 0 else num)
    return depth if depth > 0 else None


def _load_enc_depth_sampler():
    soundings: list[tuple[float, float, float]] = []
    depth_areas: list[tuple[object, float]] = []
    depth_contours: list[tuple[object, float]] = []

    if not ENC_GEOJSON_DIR.exists():
        return None, "enc_missing"

    import json

    for fp in ENC_GEOJSON_DIR.glob("*SOUNDG.geojson"):
        try:
            gj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for feat in gj.get("features", []):
            geom = feat.get("geometry") or {}
            if geom.get("type") != "MultiPoint":
                continue
            for coord in geom.get("coordinates", []):
                if len(coord) < 3:
                    continue
                depth = _normalize_depth_value(coord[2])
                if depth is None:
                    continue
                soundings.append((float(coord[1]), float(coord[0]), depth))

    for fp in ENC_GEOJSON_DIR.glob("*DEPARE.geojson"):
        try:
            gj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for feat in gj.get("features", []):
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties", {})
            d1 = _normalize_depth_value(props.get("DRVAL1"))
            d2 = _normalize_depth_value(props.get("DRVAL2"))
            depth_candidates = [d for d in [d1, d2] if d is not None]
            if not depth_candidates:
                continue
            try:
                depth_areas.append((shape(geom), min(depth_candidates)))
            except Exception:
                continue

    for fp in ENC_GEOJSON_DIR.glob("*DEPCNT.geojson"):
        try:
            gj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for feat in gj.get("features", []):
            geom = feat.get("geometry")
            if not geom:
                continue
            depth = _normalize_depth_value((feat.get("properties") or {}).get("VALDCO"))
            if depth is None:
                continue
            try:
                depth_contours.append((shape(geom), depth))
            except Exception:
                continue

    if not soundings and not depth_areas and not depth_contours:
        return None, "enc_invalid"

    return {"soundings": soundings, "areas": depth_areas, "contours": depth_contours}, "enc_geojson"


@functools.lru_cache(maxsize=1)  # the optimiser calls this per event; reopening the .nc per call fails
def _load_bathymetry_sampler():
    # ENC soundings/depth areas first: GEBCO's ~450 m grid puts real transfer
    # positions in 5 m of water where 11-12 m draught ships routinely pass.
    enc = _load_enc_depth_sampler()
    if enc[0] is not None:
        return enc
    candidates = sorted((RAW_DIR / "bathymetry").glob("*.nc"))
    if not candidates:
        return _load_enc_depth_sampler()
    try:
        import xarray as xr
    except Exception:
        return _load_enc_depth_sampler()
    ds = xr.open_dataset(candidates[0])
    lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    z_name = "elevation" if "elevation" in ds.data_vars else None
    if not lat_name or not lon_name or not z_name:
        try:
            ds.close()
        except Exception:
            pass
        return _load_enc_depth_sampler()
    return (ds, lat_name, lon_name, z_name), candidates[0].name


def _sample_bathymetry(lat: float, lon: float, sampler) -> float:
    if sampler is None:
        return np.nan
    if isinstance(sampler, dict):
        point = Point(lon, lat)

        soundings = sampler.get("soundings", [])
        if soundings:
            sounding_df = pd.DataFrame(soundings, columns=["lat", "lon", "depth_m"])
            d = haversine_distance_m(
                np.full(len(sounding_df), lat, dtype="float64"),
                np.full(len(sounding_df), lon, dtype="float64"),
                sounding_df["lat"].to_numpy(dtype="float64"),
                sounding_df["lon"].to_numpy(dtype="float64"),
            )
            idx = int(np.argmin(d))
            if float(d[idx]) <= 1000.0:
                return float(sounding_df.iloc[idx]["depth_m"])

        for geom, depth in sampler.get("areas", []):
            try:
                if geom.contains(point):
                    return float(depth)
            except Exception:
                continue

        contour_hits: list[tuple[float, float]] = []
        for geom, depth in sampler.get("contours", []):
            try:
                deg_dist = geom.distance(point)
            except Exception:
                continue
            dist_m = float(deg_dist) * 111320.0
            contour_hits.append((dist_m, float(depth)))
        if contour_hits:
            contour_hits.sort(key=lambda x: x[0])
            if contour_hits[0][0] <= 1000.0:
                return contour_hits[0][1]
        return np.nan

    ds, lat_name, lon_name, z_name = sampler
    try:
        val = ds[z_name].sel({lat_name: lat, lon_name: lon}, method="nearest").item()
    except Exception:
        return np.nan
    if pd.isna(val):
        return np.nan
    return float(-val if val <= 0 else val)


def _nearest_buoy(lat: float, lon: float, buoys: pd.DataFrame) -> tuple[object, float, object]:
    if buoys.empty:
        return pd.NA, np.nan, pd.NA
    d = haversine_distance_m(
        np.full(len(buoys), lat, dtype="float64"),
        np.full(len(buoys), lon, dtype="float64"),
        buoys["lat"].to_numpy(dtype="float64"),
        buoys["lon"].to_numpy(dtype="float64"),
    )
    idx = int(np.argmin(d))
    return buoys.iloc[idx]["name"], float(d[idx]), buoys.iloc[idx].get("type", pd.NA)


def _nearest_polygon_distance_m(lat: float, lon: float, polygons: list[tuple[str, Polygon]]) -> tuple[object, float]:
    if not polygons:
        return pd.NA, np.nan
    pt = Point(lon, lat)
    distances = []
    for name, geom in polygons:
        try:
            distances.append((name, float(geom.distance(pt))))
        except Exception:
            continue
    if not distances:
        return pd.NA, np.nan
    nearest_name, nearest_deg = min(distances, key=lambda x: x[1])
    return nearest_name, nearest_deg * 111320.0


def _fairway_context(lat: float, lon: float, fairways: list[tuple[str, object]]) -> tuple[bool, object, float]:
    if not fairways:
        return False, pd.NA, np.nan
    pt = Point(lon, lat)
    for name, geom in fairways:
        try:
            if geom.contains(pt):
                return True, name, 0.0
        except Exception:
            continue
    distances = []
    for name, geom in fairways:
        try:
            distances.append((name, float(geom.distance(pt))))
        except Exception:
            continue
    if not distances:
        return False, pd.NA, np.nan
    nearest_name, nearest_deg = min(distances, key=lambda x: x[1])
    return False, nearest_name, nearest_deg * 111320.0


def _load_master_subset(master_path: Path, time_min: pd.Timestamp, time_max: pd.Timestamp) -> pd.DataFrame:
    usecols = [
        "mmsi", "datetime", "lat", "lon", "spd", "crs", "hdg", "ship_name",
        "draft_m", "dim_a", "dim_b", "dim_c", "dim_d",
    ]
    if master_path.suffix == ".parquet":
        try:
            master = pd.read_parquet(
                master_path,
                columns=[c for c in usecols if c],
                filters=[("datetime", ">=", time_min), ("datetime", "<=", time_max)],
            )
        except Exception:
            csv_fallback = master_path.with_suffix(".csv")
            if csv_fallback.exists():
                return _load_master_subset(csv_fallback, time_min, time_max)
            master = pd.read_parquet(master_path, columns=[c for c in usecols if c])
            if "datetime" in master.columns:
                master["datetime"] = pd.to_datetime(master["datetime"], errors="coerce", utc=True)
                master = master[(master["datetime"] >= time_min) & (master["datetime"] <= time_max)]
    else:
        chunks = []
        for chunk in pd.read_csv(master_path, usecols=lambda c: c in usecols, chunksize=250000):
            if "datetime" not in chunk.columns:
                continue
            chunk["datetime"] = pd.to_datetime(chunk["datetime"], errors="coerce", utc=True)
            chunk = chunk[(chunk["datetime"] >= time_min) & (chunk["datetime"] <= time_max)]
            if not chunk.empty:
                chunks.append(chunk)
        master = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    if not master.empty:
        master["datetime"] = pd.to_datetime(master["datetime"], errors="coerce", utc=True)
        master["mmsi"] = _norm_mmsi(master["mmsi"]).astype("string")
        static_cols = {"ship_name", "draft_m", "dim_a", "dim_b", "dim_c", "dim_d"}
        if static_cols.isdisjoint(master.columns) and STATIC_FILE.exists():
            static = load_static(STATIC_FILE)
            static["mmsi"] = static["mmsi"].astype("string")
            master = master.merge(static, on="mmsi", how="left", validate="many_to_one")
    return master


def enrich_events(events: pd.DataFrame, master: pd.DataFrame | None = None) -> pd.DataFrame:
    out = events.copy()
    if out.empty:
        for col in EVENT_CONTEXT_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out[EVENT_CONTEXT_COLUMNS]

    zones = load_zone_lookup()
    weather = _read_weather_hourly()
    buoys = _read_buoys()
    fairways = _load_fairway_geometries()
    breakwaters = _load_breakwater_geometries()
    bathy_sampler, bathy_source = _load_bathymetry_sampler()
    collision_exclusions = _collision_exclusion_mmsi()

    for col in ["start_time", "meeting_time", "end_time"]:
        out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)

    out["meeting_hour"] = out["meeting_time"].dt.floor("h")
    if not weather.empty:
        weather_cols = [
            "meeting_hour",
            "wind_speed_mps",
            "wind_dir_deg",
            "wave_height_m",
            "tide_height_m",
            "light_condition",
            "visibility_km",
            "current_speed_kn",
            "current_dir_deg",
        ]
        out = out.merge(weather[weather_cols], on="meeting_hour", how="left")
    else:
        out["wind_speed_mps"] = np.nan
        out["wind_dir_deg"] = np.nan
        out["wave_height_m"] = np.nan
        out["tide_height_m"] = np.nan
        out["light_condition"] = pd.NA
        out["visibility_km"] = np.nan
        out["current_speed_kn"] = np.nan
        out["current_dir_deg"] = np.nan

    kst = out["meeting_time"]  # AIS/KMA timestamps are KST wall-clock labelled UTC; keep tables on that same clock
    out["local_hour_kst"] = kst.dt.hour
    out["night_day_flag"] = np.where((out["local_hour_kst"] < 6) | (out["local_hour_kst"] >= 18), "night", "day")
    if "light_condition" in out.columns:
        light_mask = out["light_condition"].notna()
        out.loc[light_mask, "night_day_flag"] = out.loc[light_mask, "light_condition"].replace(
            {
                "night": "night",
                "day": "day",
                "dawn": "day",
                "dusk": "night",
            }
        )

    static_zone = zones["embark"].union(zones["disembark"])
    spatial_cache: dict[tuple[float, float], dict[str, object]] = {}

    for lat, lon in zip(out["meeting_lat"], out["meeting_lon"]):
        key = _meeting_point_key(lat, lon)
        if key in spatial_cache:
            continue
        pt = Point(float(lon), float(lat))
        buoy_name, buoy_dist, buoy_type = _nearest_buoy(float(lat), float(lon), buoys)
        fairway_hit, fairway_nm, fairway_dist = _fairway_context(float(lat), float(lon), fairways)
        breakwater_name, breakwater_dist = _nearest_polygon_distance_m(float(lat), float(lon), breakwaters)
        if pd.isna(breakwater_dist):
            # Target definition (paper Sec. 3.6): distance to the nearest New Port
            # breakwater light. Computed explicitly so it no longer depends on which
            # ENC buoy layers happen to load.
            breakwater_name, breakwater_dist, _ = _nearest_buoy(
                float(lat), float(lon), BREAKWATER_LIGHTS
            )
        spatial_cache[key] = {
            "inside_static_zone": static_zone.contains(pt),
            "inside_embark_zone": zones["embark"].contains(pt),
            "inside_disembark_zone": zones["disembark"].contains(pt),
            "nearest_buoy_name": buoy_name,
            "nearest_buoy_type": buoy_type,
            "dist_to_nearest_buoy_m": buoy_dist,
            "in_fairway": fairway_hit,
            "fairway_name": fairway_nm,
            "dist_to_fairway_edge_m": fairway_dist,
            "nearest_breakwater_name": breakwater_name,
            "dist_to_breakwater_m": breakwater_dist,
            "depth_m": _sample_bathymetry(float(lat), float(lon), bathy_sampler),
        }

    spatial_rows = [
        spatial_cache[_meeting_point_key(lat, lon)]
        for lat, lon in zip(out["meeting_lat"], out["meeting_lon"])
    ]
    spatial_df = pd.DataFrame(spatial_rows, index=out.index)
    for col in spatial_df.columns:
        out[col] = spatial_df[col]
    out["depth_source"] = bathy_source
    out["navigation_support_score"] = [
        _navigation_support_score(btype, bdist, bool(fair), fdist, breakdist)
        for btype, bdist, fair, fdist, breakdist in zip(
            out["nearest_buoy_type"],
            out["dist_to_nearest_buoy_m"],
            out["in_fairway"],
            out["dist_to_fairway_edge_m"],
            out["dist_to_breakwater_m"],
        )
    ]

    out["escort_phase_duration_min"] = np.where(
        out["traffic_direction"].eq("outbound"),
        (out["duration_min"] * 0.55).round(2),
        (out["duration_min"] * 0.20).round(2),
    )
    out["approach_phase_duration_min"] = np.where(
        out["traffic_direction"].eq("inbound"),
        (out["duration_min"] * 0.45).round(2),
        (out["duration_min"] * 0.20).round(2),
    )
    out["alongside_phase_duration_min"] = (out["duration_min"] * 0.15).round(2)
    out["separation_phase_duration_min"] = (
        out["duration_min"]
        - out["escort_phase_duration_min"]
        - out["approach_phase_duration_min"]
        - out["alongside_phase_duration_min"]
    ).round(2)

    out["opening_distance_5min_after_transfer_m"] = np.nan
    out["opening_rate_m_per_min"] = np.nan
    out["surrounding_vessel_count_0_5nm"] = np.nan
    out["surrounding_vessel_count_1_0nm"] = np.nan
    out["nearest_third_vessel_mmsi"] = pd.NA
    out["nearest_third_vessel_distance_m"] = np.nan
    out["nearest_third_vessel_center_cpa_m"] = np.nan
    out["nearest_third_vessel_cpa_m"] = np.nan
    out["nearest_third_vessel_hull_cpa_m"] = np.nan
    out["nearest_third_vessel_domain_clearance_m"] = np.nan
    out["nearest_third_vessel_tcpa_min"] = np.nan
    out["collision_risk_score"] = np.nan
    out["collision_risk_class"] = pd.NA
    out["vessel_draft_m"] = np.nan
    out["pilot_draft_m"] = np.nan
    out["ukc_m"] = np.nan
    out["ukc_ratio"] = np.nan
    out["ukc_risk_class"] = pd.NA
    out["dist_to_restricted_area_m"] = np.nan
    out["sea_room_score"] = np.nan
    if "visibility_km" not in out.columns:
        out["visibility_km"] = np.nan
    if "tide_height_m" not in out.columns:
        out["tide_height_m"] = np.nan
    if "current_speed_kn" not in out.columns:
        out["current_speed_kn"] = np.nan
    if "current_dir_deg" not in out.columns:
        out["current_dir_deg"] = np.nan
    out["weather_exposure_score"] = np.nan
    out["shelter_score"] = np.nan
    out["transfer_side_exposure"] = "unknown"
    out["escape_corridor_availability"] = pd.NA
    out["distance_to_open_sea_route_m"] = np.nan
    out["time_to_conflict_if_aborted_min"] = np.nan
    out["domain_overlap_fixed_hazards"] = pd.NA
    out["go_no_go"] = pd.NA
    out["go_no_go_reason"] = pd.NA
    out["required_ship_domain_m2"] = np.nan
    out["available_sea_room_m2"] = np.nan
    out["domain_clearance_ratio"] = np.nan
    out["emergency_turn_feasible"] = pd.NA
    out["emergency_escape_route_feasible"] = pd.NA
    out["mob_recovery_feasible"] = pd.NA
    out["trajectory_restoration_feasible"] = pd.NA

    if master is not None and not master.empty:
        master = master.copy()
        master["mmsi"] = master["mmsi"].astype("string")
        if "draft_m" in master.columns:
            master["draft_m"] = pd.to_numeric(master["draft_m"], errors="coerce")
        master["loa_est_m"] = master[[c for c in ["dim_a", "dim_b"] if c in master.columns]].apply(
            pd.to_numeric, errors="coerce"
        ).sum(axis=1, min_count=1)
        master["beam_est_m"] = master[[c for c in ["dim_c", "dim_d"] if c in master.columns]].apply(
            pd.to_numeric, errors="coerce"
        ).sum(axis=1, min_count=1)

        for idx, event in out.iterrows():
            t0 = event["meeting_time"] - pd.Timedelta(minutes=5)
            t1 = event["meeting_time"] + pd.Timedelta(minutes=5)
            window = master[(master["datetime"] >= t0) & (master["datetime"] <= t1)].copy()
            if window.empty:
                continue

            pilot_rows = window[window["mmsi"] == str(event["pilot_mmsi"])].copy()
            vessel_rows = window[window["mmsi"] == str(event["vessel_mmsi"])].copy()
            if pilot_rows.empty or vessel_rows.empty:
                continue

            def nearest_row(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
                order = (df["datetime"] - ts).abs().sort_values().index
                return df.loc[order[0]]

            pilot_row = nearest_row(pilot_rows, event["meeting_time"])
            vessel_row = nearest_row(vessel_rows, event["meeting_time"])

            out.at[idx, "pilot_draft_m"] = pilot_row.get("draft_m", np.nan)
            out.at[idx, "vessel_draft_m"] = vessel_row.get("draft_m", np.nan)

            draft = pd.to_numeric(pd.Series([vessel_row.get("draft_m")]), errors="coerce").iloc[0]
            effective_depth = pd.to_numeric(pd.Series([event.get("depth_m")]), errors="coerce").iloc[0]
            tide_height = pd.to_numeric(pd.Series([event.get("tide_height_m")]), errors="coerce").iloc[0]
            if pd.notna(effective_depth) and pd.notna(tide_height):
                effective_depth = float(effective_depth) + float(tide_height)
            if pd.notna(effective_depth) and pd.notna(draft):
                ukc = compute_ukc(effective_depth, draft)
                out.at[idx, "ukc_m"] = ukc
                out.at[idx, "ukc_ratio"] = ukc / float(draft) if draft else np.nan
                out.at[idx, "ukc_risk_class"] = classify_ukc_risk(ukc)

            third = window[~window["mmsi"].isin([str(event["pilot_mmsi"]), str(event["vessel_mmsi"])])].copy()
            third = third[~third["mmsi"].astype(str).isin(collision_exclusions)].copy()
            if "ship_name" in third.columns:
                third = third[~third.get("ship_type_class", pd.Series(index=third.index)).eq("Service_Tug_Pilot")].copy()
            traffic_summary = summarize_nearby_traffic(
                float(event["meeting_lat"]),
                float(event["meeting_lon"]),
                third,
                vessel_row,
            )
            for key, value in traffic_summary.items():
                out.at[idx, key] = value

            surrounding_1nm = pd.to_numeric(
                pd.Series([out.at[idx, "surrounding_vessel_count_1_0nm"]]),
                errors="coerce",
            ).fillna(0).iloc[0]
            out.at[idx, "sea_room_score"] = _clip_score(1.0 - (surrounding_1nm / 10.0))
            out.at[idx, "navigation_support_score"] = _navigation_support_score(
                event.get("nearest_buoy_type"),
                event.get("dist_to_nearest_buoy_m"),
                bool(event.get("in_fairway")),
                event.get("dist_to_fairway_edge_m"),
                event.get("dist_to_breakwater_m"),
            )
            out.at[idx, "weather_exposure_score"] = max(
                0.0,
                min(
                    1.0,
                    0.6 * (1.0 - min(pd.to_numeric(pd.Series([event.get("wind_speed_mps")]), errors="coerce").fillna(0).iloc[0] / 20.0, 1.0))
                    + 0.4 * (1.0 - min(pd.to_numeric(pd.Series([event.get("wave_height_m")]), errors="coerce").fillna(0).iloc[0] / 4.0, 1.0)),
                ),
            )
            out.at[idx, "shelter_score"] = (
                np.nan
                if pd.isna(event["dist_to_breakwater_m"])
                else _clip_score(1.0 - float(event["dist_to_breakwater_m"]) / 5000.0)
            )
            out.at[idx, "distance_to_open_sea_route_m"] = np.nan
            out.at[idx, "time_to_conflict_if_aborted_min"] = out.at[idx, "nearest_third_vessel_tcpa_min"]
            out.at[idx, "domain_overlap_fixed_hazards"] = (
                "possible" if (pd.notna(event["dist_to_breakwater_m"]) and event["dist_to_breakwater_m"] < 500) else "none"
            )
            loa_est = pd.to_numeric(pd.Series([vessel_row.get("loa_est_m")]), errors="coerce").iloc[0]
            out.at[idx, "required_ship_domain_m2"] = required_ship_domain_area_m2(loa_est)
            if pd.notna(out.at[idx, "required_ship_domain_m2"]):
                # Proxy for now: available sea room from a 1 nm circle reduced by nearby traffic occupancy.
                occupied_ratio = min(float(surrounding_1nm) / 12.0, 0.85)
                out.at[idx, "available_sea_room_m2"] = math.pi * (1852.0**2) * (1.0 - occupied_ratio)
            if pd.notna(out.at[idx, "required_ship_domain_m2"]) and pd.notna(out.at[idx, "available_sea_room_m2"]):
                out.at[idx, "domain_clearance_ratio"] = out.at[idx, "available_sea_room_m2"] / out.at[idx, "required_ship_domain_m2"]
            domain_ratio = pd.to_numeric(pd.Series([out.at[idx, "domain_clearance_ratio"]]), errors="coerce").iloc[0]
            out.at[idx, "emergency_turn_feasible"] = pd.notna(domain_ratio) and domain_ratio >= 1.0
            out.at[idx, "emergency_escape_route_feasible"] = (
                out.at[idx, "collision_risk_class"] in {"low", "unknown"}
                and out.at[idx, "ukc_risk_class"] in {"safe", "unknown"}
            )
            out.at[idx, "mob_recovery_feasible"] = (
                out.at[idx, "sea_room_score"] >= 0.4
                and out.at[idx, "collision_risk_class"] != "high"
            )
            out.at[idx, "trajectory_restoration_feasible"] = (
                bool(out.at[idx, "emergency_escape_route_feasible"])
                and bool(out.at[idx, "emergency_turn_feasible"])
            )

            decision, decision_reason = _go_no_go_decision(out.loc[idx])
            out.at[idx, "go_no_go"] = decision
            out.at[idx, "go_no_go_reason"] = decision_reason

    out["transfer_feasibility_score"] = np.where(out["likely_transfer"], out["confidence_score"], out["confidence_score"] * 0.5)
    unresolved = out["go_no_go"].isna()
    out.loc[unresolved & out["likely_transfer"], "go_no_go"] = "review"
    out.loc[unresolved & out["likely_transfer"], "go_no_go_reason"] = "requires_context_enrichment"
    out.loc[unresolved & ~out["likely_transfer"], "go_no_go"] = "no_go"
    out.loc[unresolved & ~out["likely_transfer"], "go_no_go_reason"] = "detection_not_strong_enough"

    for col in EVENT_CONTEXT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[EVENT_CONTEXT_COLUMNS]


def enrich_events_chunked(
    events: pd.DataFrame,
    master_path: Path | None,
    chunk_freq: str = "7D",
    chunk_margin_min: int = 10,
) -> pd.DataFrame:
    if master_path is None or events.empty:
        return enrich_events(events, master=None)

    events = events.copy()
    events["meeting_time"] = pd.to_datetime(events["meeting_time"], errors="coerce", utc=True)
    valid = events.dropna(subset=["meeting_time"]).copy()
    invalid = events[events["meeting_time"].isna()].copy()

    if valid.empty:
        return enrich_events(events, master=None)

    valid["chunk_start"] = valid["meeting_time"].dt.floor(chunk_freq)
    enriched_chunks: list[pd.DataFrame] = []
    for _, chunk in valid.groupby("chunk_start", sort=True):
        time_min = chunk["meeting_time"].min() - pd.Timedelta(minutes=chunk_margin_min)
        time_max = chunk["meeting_time"].max() + pd.Timedelta(minutes=chunk_margin_min)
        master = _load_master_subset(master_path, time_min, time_max)
        enriched_chunk = enrich_events(chunk.drop(columns=["chunk_start"]), master=master)
        enriched_chunks.append(enriched_chunk)

    if not invalid.empty:
        enriched_chunks.append(enrich_events(invalid, master=None))

    enriched = pd.concat(enriched_chunks, ignore_index=True) if enriched_chunks else enrich_events(events, master=None)
    order = pd.Categorical(enriched["event_id"], categories=events["event_id"].astype("string"), ordered=True)
    enriched = enriched.assign(_event_order=order).sort_values("_event_order").drop(columns="_event_order")
    return enriched.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich detected meeting events with operational context")
    parser.add_argument("--events", type=str, required=True, help="Detected events CSV path")
    parser.add_argument("--master", type=str, default=None, help="AIS master parquet/csv path")
    parser.add_argument("--tag", type=str, default="latest", help="Output suffix tag")
    parser.add_argument("--chunk-freq", type=str, default="7D", help="Time chunk size for loading AIS master")
    parser.add_argument("--chunk-margin-min", type=int, default=10, help="Minutes added around each chunk window")
    args = parser.parse_args()

    try:
        events = pd.read_csv(args.events)
    except EmptyDataError:
        events = pd.DataFrame(columns=["event_id", "meeting_time"])
    if "meeting_time" in events.columns:
        events["meeting_time"] = pd.to_datetime(events["meeting_time"], errors="coerce", utc=True)
    start_date = events["meeting_time"].min().isoformat() if "meeting_time" in events.columns and events["meeting_time"].notna().any() else None
    end_date = events["meeting_time"].max().isoformat() if "meeting_time" in events.columns and events["meeting_time"].notna().any() else None
    master_path = Path(args.master) if args.master else _pick_master_path(None, start_date=start_date, end_date=end_date)
    enriched = enrich_events_chunked(
        events,
        master_path=master_path,
        chunk_freq=args.chunk_freq,
        chunk_margin_min=args.chunk_margin_min,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"event_context_enriched_{args.tag}.csv"
    enriched.to_csv(out_path, index=False)
    print(f"Saved enriched event context: {out_path}")


if __name__ == "__main__":
    main()
