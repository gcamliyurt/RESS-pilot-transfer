from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from src.core.build_operational_features import map_ship_type_class
from src.core.classify_traffic_direction import classify_event_direction
from src.core.prepare_ais_master import STATIC_FILE, load_static

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MASTER_DIR = PROJECT_ROOT / "data" / "processed" / "ais_master"
WEEKLY_MASTER_DIR = MASTER_DIR / "weeks"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "interactions"

DEFAULT_MASTER_PARQUET = MASTER_DIR / "ais_master_latest.parquet"
DEFAULT_MASTER_CSV = MASTER_DIR / "ais_master_latest.csv"
PILOT_REFERENCE_JSON = RAW_DIR / "reference" / "pilot_boat_trajectories.json"
PILOT_REFERENCE_XLSX = RAW_DIR / "ais_static" / "BusanPB.xlsx"

LEGACY_TUG_BOATS = {
    "440412320", "440155260", "352003392", "440139260", "440030820", "514446000",
    "440700180", "440051510", "440702880", "440301460", "440112210", "440148520",
    "440165910", "440151610", "440110910", "440075240", "39036", "440313260",
    "440713100", "440152850", "440100580", "440118970", "440018500", "440113610",
    "440127670", "440112730", "441925000", "440083810", "440132580", "440116550",
    "440300350", "440107360", "440184720", "440118750", "373493000", "440200060",
    "354714000", "440313250", "440108150", "440909000", "440016540", "440335980",
    "440005000", "440104800",
}

# Additional pilot boats observed in validated historical outputs that may be
# absent from the JSON reference and unavailable from Excel when openpyxl is
# not installed in the active environment.
ADDITIONAL_KNOWN_PILOT_MMSI = {
    "440155330",
}

# Busan Harbor Pilots' Association fleet (13 boats, busanpilot.co.kr/company/status).
# The legacy list above mixes in harbour tugs; use --pilot-fleet verified for the R2 revision.
VERIFIED_PILOT_MMSI = {
    "440008040", "440105210", "440110850", "440111140", "440111150", "440111160", "440111190",
    "440111210", "440111230", "440121630", "440127410", "440135830", "440155330",
}

COURSE_ALIGNMENT_THRESHOLD_DEG = 20.0
SPEED_MIN_KN = 4.0
SPEED_MAX_KN = 12.0
SPEED_DIFF_MAX_KN = 4.0
BOARDING_MAX_SPEED_KN = 2.0
BOARDING_MIN_MOVEMENT_KN = 0.3
MIN_TARGET_LOA_M = 60.0
MIN_TARGET_BEAM_M = 2.0
MIN_TARGET_TONNAGE = 500.0

EVENT_OUTPUT_COLUMNS = [
    "event_id",
    "pilot_mmsi",
    "vessel_mmsi",
    "pilot_ship_name",
    "vessel_ship_name",
    "vessel_ship_type_class",
    "start_time",
    "end_time",
    "duration_min",
    "n_points",
    "meeting_time",
    "meeting_lat",
    "meeting_lon",
    "meeting_ref_lat",
    "meeting_ref_lon",
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
    "passes_course_rule",
    "passes_speed_rule",
    "passes_boarding_rule",
    "passes_kinematic_rule",
    "dominant_rule_tag",
    "pilot_avg_speed_kn",
    "vessel_avg_speed_kn",
    "estimated_tonnage",
    "loa_est_m",
    "beam_est_m",
    "centroid_lat",
    "centroid_lon",
    "transfer_zone_class",
    "traffic_direction",
    "confidence_score",
    "likely_transfer",
]


def _norm_mmsi(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )
    return s.where(s.str.len().between(7, 9), pd.NA)


def _parse_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _week_number_for_timestamp(ts: pd.Timestamp) -> int:
    return int(((ts.day - 1) // 7) + 1)


def _pick_master_path(
    master_path: str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    if master_path:
        return Path(master_path)

    start_ts = _parse_utc_timestamp(start_date)
    end_ts = _parse_utc_timestamp(end_date)

    if start_ts is not None:
        effective_end = start_ts if end_ts is None else max(start_ts, end_ts - pd.Timedelta(seconds=1))
        same_month = (start_ts.year, start_ts.month) == (effective_end.year, effective_end.month)
        same_week = _week_number_for_timestamp(start_ts) == _week_number_for_timestamp(effective_end)

        if same_month and same_week:
            month_tag = start_ts.strftime("%Y-%m")
            weekly_csv = WEEKLY_MASTER_DIR / f"{month_tag}-week{_week_number_for_timestamp(start_ts)}.csv"
            if weekly_csv.exists():
                return weekly_csv

        monthly_csv = MASTER_DIR / f"{start_ts.strftime('%Y-%m')}.csv"
        if monthly_csv.exists():
            return monthly_csv

    if DEFAULT_MASTER_CSV.exists():
        return DEFAULT_MASTER_CSV
    if DEFAULT_MASTER_PARQUET.exists():
        return DEFAULT_MASTER_PARQUET
    raise FileNotFoundError("No AIS master file found. Build it first with prepare_ais_master.py or ensure monthly CSVs exist.")


def load_master(master_file: Path, sample_rows: int | None = None) -> pd.DataFrame:
    if master_file.suffix == ".parquet":
        try:
            df = pd.read_parquet(master_file)
        except Exception:
            csv_fallback = master_file.with_suffix(".csv")
            if not csv_fallback.exists():
                raise
            df = pd.read_csv(csv_fallback)
    else:
        df = pd.read_csv(master_file)

    if sample_rows and sample_rows > 0 and len(df) > sample_rows:
        df = df.iloc[:sample_rows].copy()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["mmsi"] = _norm_mmsi(df["mmsi"])
    df = df.dropna(subset=["datetime", "mmsi", "lat", "lon"]).copy()

    # Weekly split files may contain only dynamic AIS fields. Merge static vessel
    # attributes on demand so downstream mother-ship screening still works.
    static_cols = {"ship_name", "ship_type_code", "estimated_tonnage", "dim_a", "dim_b", "dim_c", "dim_d", "draft_m"}
    if static_cols.isdisjoint(df.columns) and STATIC_FILE.exists():
        static = load_static(STATIC_FILE)
        df = df.merge(static, on="mmsi", how="left", validate="many_to_one")

    return df


def _extract_mmsi_candidates(frame: pd.DataFrame) -> set[str]:
    candidates: set[str] = set()
    for col in frame.columns:
        series = _norm_mmsi(frame[col])
        vals = series.dropna().astype(str).tolist()
        candidates.update(vals)
    return candidates


SERVICE_CRAFT_SHIP_TYPES = {31, 32, 50, 52, 53}  # towing, pilot, tug, port tender


@functools.lru_cache(maxsize=1)
def service_craft_mmsi() -> frozenset[str]:
    """Vessels that are part of a transfer operation (pilot boats, attending tugs),
    excluded from third-party traffic / collision metrics."""
    ids = set(VERIFIED_PILOT_MMSI) | set(LEGACY_TUG_BOATS)
    try:
        ids |= set(load_pilot_mmsi())
    except Exception:
        pass
    static = PROJECT_ROOT.parent / "Static_merged_bbox_result_gokhan.csv"
    if static.exists():
        st = pd.read_csv(static, encoding="cp949", usecols=[0, 2])
        st.columns = ["mmsi", "ship_type"]
        st = st[pd.to_numeric(st["ship_type"], errors="coerce").isin(SERVICE_CRAFT_SHIP_TYPES)]
        ids |= set(_norm_mmsi(st["mmsi"]).astype(str))
    return frozenset(ids)


def load_pilot_mmsi() -> set[str]:
    pilot_ids: set[str] = set()

    if PILOT_REFERENCE_JSON.exists():
        payload = json.loads(PILOT_REFERENCE_JSON.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for item in payload.values():
                if isinstance(item, dict) and item.get("pilot_mmsi") is not None:
                    pilot_ids.add(str(int(float(item["pilot_mmsi"]))))

    if PILOT_REFERENCE_XLSX.exists():
        try:
            xls = pd.read_excel(PILOT_REFERENCE_XLSX)
            pilot_ids.update(_extract_mmsi_candidates(xls))
        except Exception:
            pass

    # Fallback: many known pilot craft are already tracked in the legacy tug/pilot
    # reference list used to exclude service vessels from mother-ship candidates.
    # Treat them as pilot-side candidates as well when richer references are absent.
    pilot_ids.update(LEGACY_TUG_BOATS)
    pilot_ids.update(ADDITIONAL_KNOWN_PILOT_MMSI)

    if not pilot_ids:
        raise FileNotFoundError("No pilot boat reference MMSI could be loaded from raw references")

    return pilot_ids


def load_zone_polygon(path: Path) -> Polygon:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    if {"x", "y"}.issubset(cols):
        lat = pd.to_numeric(df[cols["x"]], errors="coerce")
        lon = pd.to_numeric(df[cols["y"]], errors="coerce")
    else:
        df = pd.read_csv(path, header=None)
        lat = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        lon = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    coords = [(x, y) for x, y in zip(lon, lat) if pd.notna(x) and pd.notna(y)]
    if len(coords) < 3:
        raise ValueError(f"Could not build polygon from {path}")
    return Polygon(coords)


def load_zone_lookup() -> dict[str, Polygon]:
    zone_dir = RAW_DIR / "zones"
    return {
        "study_area": load_zone_polygon(zone_dir / "studyArea.csv"),
        "embark": load_zone_polygon(zone_dir / "busanNewPortEmbark.csv"),
        "disembark": load_zone_polygon(zone_dir / "busanNewPort - disEmbarkation.csv"),
    }


def prepare_detection_frame(
    df: pd.DataFrame,
    pilot_mmsi: set[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["mmsi"] = out["mmsi"].astype(str)

    if start_date:
        out = out[out["datetime"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        out = out[out["datetime"] < pd.Timestamp(end_date, tz="UTC")]

    zones = load_zone_lookup()
    study_area = zones["study_area"]
    study_bounds = study_area.bounds
    out = out[
        out["lon"].between(study_bounds[0], study_bounds[2])
        & out["lat"].between(study_bounds[1], study_bounds[3])
    ].copy()

    pts = [study_area.contains(Point(lon, lat)) for lon, lat in zip(out["lon"], out["lat"])]
    out = out.loc[pts].copy()

    if "ship_type_class" not in out.columns:
        out["ship_type_class"] = map_ship_type_class(out.get("ship_type_code", pd.Series(index=out.index)))

    out["spd"] = pd.to_numeric(out.get("spd"), errors="coerce")
    out["crs"] = pd.to_numeric(out.get("crs"), errors="coerce")
    out["time_bin"] = out["datetime"].dt.floor("5min")
    out["is_pilot_boat"] = out["mmsi"].isin(pilot_mmsi)
    out["is_legacy_tug"] = out["mmsi"].isin(LEGACY_TUG_BOATS)

    for col in ["estimated_tonnage", "dim_a", "dim_b", "dim_c", "dim_d"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "draft_m" in out.columns:
        out["draft_m"] = pd.to_numeric(out["draft_m"], errors="coerce")

    out["loa_est_m"] = out[[c for c in ["dim_a", "dim_b"] if c in out.columns]].sum(axis=1, min_count=1)
    out["beam_est_m"] = out[[c for c in ["dim_c", "dim_d"] if c in out.columns]].sum(axis=1, min_count=1)
    out["is_mother_ship_candidate"] = (
        (~out["is_pilot_boat"])
        & (~out["is_legacy_tug"])
        & (~out["ship_type_class"].eq("Service_Tug_Pilot"))
        & (
            out.get("estimated_tonnage", pd.Series(index=out.index)).fillna(0).gt(MIN_TARGET_TONNAGE)
            | out["loa_est_m"].fillna(0).ge(MIN_TARGET_LOA_M)
            | out["beam_est_m"].fillna(0).gt(MIN_TARGET_BEAM_M)
            | out["ship_type_class"].isin(["Cargo", "Tanker", "Passenger"])
        )
    )
    return out


def haversine_distance_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    r = 6371000.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def angular_difference_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    diff = (a - b).abs() % 360.0
    return diff.where(diff <= 180.0, 360.0 - diff)


def evaluate_speed_similarity(pilot_sog: pd.Series, vessel_sog: pd.Series) -> tuple[pd.Series, pd.Series]:
    pilot = pd.to_numeric(pilot_sog, errors="coerce")
    vessel = pd.to_numeric(vessel_sog, errors="coerce")
    diff = (pilot - vessel).abs()
    in_range = pilot.between(SPEED_MIN_KN, SPEED_MAX_KN) & vessel.between(SPEED_MIN_KN, SPEED_MAX_KN)
    return in_range & diff.le(SPEED_DIFF_MAX_KN), diff


def _shift_reference_to_center(
    lat: np.ndarray,
    lon: np.ndarray,
    crs: np.ndarray,
    dim_a: np.ndarray,
    dim_b: np.ndarray,
    dim_c: np.ndarray,
    dim_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    heading = np.radians(np.mod(np.nan_to_num(crs, nan=0.0), 360.0))
    forward_x = np.sin(heading)
    forward_y = np.cos(heading)
    starboard_x = np.cos(heading)
    starboard_y = -np.sin(heading)
    offset_forward = (np.nan_to_num(dim_a, nan=0.0) - np.nan_to_num(dim_b, nan=0.0)) / 2.0
    offset_starboard = (np.nan_to_num(dim_d, nan=0.0) - np.nan_to_num(dim_c, nan=0.0)) / 2.0
    east_m = forward_x * offset_forward + starboard_x * offset_starboard
    north_m = forward_y * offset_forward + starboard_y * offset_starboard
    center_lat = lat + (north_m / 111320.0)
    center_lon = lon + (east_m / np.maximum(111320.0 * np.cos(np.radians(lat)), 1e-6))
    return center_lat, center_lon


def _half_diag_and_domain_radius(
    loa_est: np.ndarray,
    beam_est: np.ndarray,
    dim_a: np.ndarray,
    dim_b: np.ndarray,
    dim_c: np.ndarray,
    dim_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    loa = np.where(
        np.isfinite(loa_est) & (loa_est > 0),
        loa_est,
        np.nan_to_num(dim_a, nan=0.0) + np.nan_to_num(dim_b, nan=0.0),
    )
    beam = np.where(
        np.isfinite(beam_est) & (beam_est > 0),
        beam_est,
        np.nan_to_num(dim_c, nan=0.0) + np.nan_to_num(dim_d, nan=0.0),
    )
    loa = np.where(loa > 0, loa, 50.0)
    beam = np.where(beam > 0, beam, np.maximum(10.0, 0.18 * loa))
    half_diag = np.hypot(loa / 2.0, beam / 2.0)
    domain_radius = half_diag + np.maximum(0.25 * loa, 1.5 * beam)
    return half_diag, domain_radius


def evaluate_boarding_signature(pilot_sog: pd.Series, vessel_sog: pd.Series) -> tuple[pd.Series, pd.Series]:
    pilot = pd.to_numeric(pilot_sog, errors="coerce")
    vessel = pd.to_numeric(vessel_sog, errors="coerce")
    max_speed = pd.concat([pilot, vessel], axis=1).max(axis=1)
    has_movement = pilot.gt(BOARDING_MIN_MOVEMENT_KN) | vessel.gt(BOARDING_MIN_MOVEMENT_KN)
    boarding = pilot.le(BOARDING_MAX_SPEED_KN) & vessel.le(BOARDING_MAX_SPEED_KN) & has_movement
    return boarding, max_speed


def build_candidate_pairs(df: pd.DataFrame, max_distance_m: float) -> pd.DataFrame:
    pilot = df[df["is_pilot_boat"]].copy()
    vessels = df[df["is_mother_ship_candidate"]].copy()

    if pilot.empty or vessels.empty:
        return pd.DataFrame()

    lat_thresh = max_distance_m / 111320.0
    all_pairs: list[pd.DataFrame] = []

    for time_bin, pilot_grp in pilot.groupby("time_bin", sort=True):
        vessel_grp = vessels[vessels["time_bin"] == time_bin]
        if vessel_grp.empty:
            continue

        for _, p in pilot_grp.iterrows():
            lon_thresh = max_distance_m / max(111320.0 * np.cos(np.radians(float(p["lat"]))), 1.0)
            subset = vessel_grp[
                vessel_grp["lat"].between(float(p["lat"]) - lat_thresh, float(p["lat"]) + lat_thresh)
                & vessel_grp["lon"].between(float(p["lon"]) - lon_thresh, float(p["lon"]) + lon_thresh)
            ].copy()
            if subset.empty:
                continue

            subset["pilot_mmsi"] = p["mmsi"]
            subset["pilot_ship_name"] = p.get("ship_name")
            subset["pilot_datetime"] = p["datetime"]
            subset["pilot_lat"] = float(p["lat"])
            subset["pilot_lon"] = float(p["lon"])
            subset["pilot_spd_kn"] = p.get("spd")
            subset["pilot_crs"] = p.get("crs")
            subset["pilot_dim_a"] = p.get("dim_a")
            subset["pilot_dim_b"] = p.get("dim_b")
            subset["pilot_dim_c"] = p.get("dim_c")
            subset["pilot_dim_d"] = p.get("dim_d")
            subset["pilot_loa_est_m"] = p.get("loa_est_m")
            subset["pilot_beam_est_m"] = p.get("beam_est_m")
            subset["pilot_draft_m"] = p.get("draft_m")

            subset["antenna_distance_m"] = haversine_distance_m(
                np.full(len(subset), float(p["lat"]), dtype="float64"),
                np.full(len(subset), float(p["lon"]), dtype="float64"),
                subset["lat"].to_numpy(dtype="float64"),
                subset["lon"].to_numpy(dtype="float64"),
            )
            subset["distance_m"] = subset["antenna_distance_m"]
            subset = subset[subset["antenna_distance_m"] <= (max_distance_m + 500.0)].copy()
            if subset.empty:
                continue

            pilot_center_lat, pilot_center_lon = _shift_reference_to_center(
                np.array([float(p["lat"])], dtype="float64"),
                np.array([float(p["lon"])], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("crs")]), errors="coerce").iloc[0]], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("dim_a")]), errors="coerce").iloc[0]], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("dim_b")]), errors="coerce").iloc[0]], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("dim_c")]), errors="coerce").iloc[0]], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("dim_d")]), errors="coerce").iloc[0]], dtype="float64"),
            )
            vessel_center_lat, vessel_center_lon = _shift_reference_to_center(
                subset["lat"].to_numpy(dtype="float64"),
                subset["lon"].to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("crs"), errors="coerce").to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("dim_a"), errors="coerce").to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("dim_b"), errors="coerce").to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("dim_c"), errors="coerce").to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("dim_d"), errors="coerce").to_numpy(dtype="float64"),
            )
            subset["own_center_lat"] = float(pilot_center_lat[0])
            subset["own_center_lon"] = float(pilot_center_lon[0])
            subset["tgt_center_lat"] = vessel_center_lat
            subset["tgt_center_lon"] = vessel_center_lon
            subset["center_distance_m"] = haversine_distance_m(
                np.full(len(subset), float(pilot_center_lat[0]), dtype="float64"),
                np.full(len(subset), float(pilot_center_lon[0]), dtype="float64"),
                vessel_center_lat,
                vessel_center_lon,
            )

            own_half_diag, own_domain_radius = _half_diag_and_domain_radius(
                np.array([pd.to_numeric(pd.Series([p.get("loa_est_m")]), errors="coerce").iloc[0]], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("beam_est_m")]), errors="coerce").iloc[0]], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("dim_a")]), errors="coerce").iloc[0]], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("dim_b")]), errors="coerce").iloc[0]], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("dim_c")]), errors="coerce").iloc[0]], dtype="float64"),
                np.array([pd.to_numeric(pd.Series([p.get("dim_d")]), errors="coerce").iloc[0]], dtype="float64"),
            )
            tgt_half_diag, tgt_domain_radius = _half_diag_and_domain_radius(
                pd.to_numeric(subset.get("loa_est_m"), errors="coerce").to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("beam_est_m"), errors="coerce").to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("dim_a"), errors="coerce").to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("dim_b"), errors="coerce").to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("dim_c"), errors="coerce").to_numpy(dtype="float64"),
                pd.to_numeric(subset.get("dim_d"), errors="coerce").to_numpy(dtype="float64"),
            )
            subset["hull_clearance_m"] = np.maximum(0.0, subset["center_distance_m"] - float(own_half_diag[0]) - tgt_half_diag)
            subset["hull_penetration_m"] = subset["center_distance_m"] - float(own_half_diag[0]) - tgt_half_diag
            subset["domain_clearance_m"] = subset["center_distance_m"] - float(own_domain_radius[0]) - tgt_domain_radius
            subset["interaction_mid_lat"] = (float(pilot_center_lat[0]) + vessel_center_lat) / 2.0
            subset["interaction_mid_lon"] = (float(pilot_center_lon[0]) + vessel_center_lon) / 2.0

            subset["distance_m"] = subset["hull_clearance_m"]
            subset = subset[subset["distance_m"] <= max_distance_m].copy()
            if subset.empty:
                continue

            subset["vessel_mmsi"] = subset["mmsi"]
            subset["vessel_ship_name"] = subset.get("ship_name")
            subset["vessel_datetime"] = subset["datetime"]
            subset["vessel_lat"] = subset["lat"]
            subset["vessel_lon"] = subset["lon"]
            subset["vessel_dim_a"] = subset["dim_a"]
            subset["vessel_dim_b"] = subset["dim_b"]
            subset["vessel_dim_c"] = subset["dim_c"]
            subset["vessel_dim_d"] = subset["dim_d"]
            subset["vessel_draft_m"] = subset["draft_m"] if "draft_m" in subset.columns else np.nan
            subset["vessel_spd_kn"] = subset["spd"]
            subset["vessel_crs"] = subset["crs"]
            subset["rel_speed_kn"] = (subset["pilot_spd_kn"] - subset["vessel_spd_kn"]).abs()
            subset["heading_diff_deg"] = angular_difference_deg(subset["pilot_crs"], subset["vessel_crs"])
            subset["course_aligned"] = subset["heading_diff_deg"].le(COURSE_ALIGNMENT_THRESHOLD_DEG)
            subset["speed_similar"], subset["speed_diff_kn"] = evaluate_speed_similarity(
                subset["pilot_spd_kn"],
                subset["vessel_spd_kn"],
            )
            subset["boarding_operation"], subset["boarding_max_speed_kn"] = evaluate_boarding_signature(
                subset["pilot_spd_kn"],
                subset["vessel_spd_kn"],
            )
            subset["kinematic_valid"] = subset["course_aligned"] | subset["speed_similar"] | subset["boarding_operation"]
            subset["rule_tag"] = np.select(
                [
                    subset["course_aligned"] & subset["speed_similar"],
                    subset["boarding_operation"],
                    subset["course_aligned"],
                    subset["speed_similar"],
                ],
                [
                    "course_aligned_speed_similar",
                    "boarding_operation",
                    "course_aligned_only",
                    "speed_similar_only",
                ],
                default="proximity_only",
            )
            subset = subset[subset["kinematic_valid"]].copy()
            if subset.empty:
                continue

            keep_cols = [
                "time_bin",
                "pilot_datetime",
                "pilot_mmsi",
                "pilot_ship_name",
                "pilot_lat",
                "pilot_lon",
                "pilot_spd_kn",
                "pilot_crs",
                "pilot_dim_a",
                "pilot_dim_b",
                "pilot_dim_c",
                "pilot_dim_d",
                "pilot_loa_est_m",
                "pilot_beam_est_m",
                "pilot_draft_m",
                "vessel_datetime",
                "vessel_mmsi",
                "vessel_ship_name",
                "vessel_lat",
                "vessel_lon",
                "vessel_spd_kn",
                "vessel_crs",
                "vessel_dim_a",
                "vessel_dim_b",
                "vessel_dim_c",
                "vessel_dim_d",
                "vessel_draft_m",
                "ship_type_class",
                "estimated_tonnage",
                "loa_est_m",
                "beam_est_m",
                "distance_m",
                "antenna_distance_m",
                "center_distance_m",
                "domain_clearance_m",
                "interaction_mid_lat",
                "interaction_mid_lon",
                "rel_speed_kn",
                "heading_diff_deg",
                "course_aligned",
                "speed_similar",
                "speed_diff_kn",
                "boarding_operation",
                "boarding_max_speed_kn",
                "kinematic_valid",
                "rule_tag",
            ]
            all_pairs.append(subset[keep_cols])

    if not all_pairs:
        return pd.DataFrame()

    pairs = pd.concat(all_pairs, ignore_index=True)
    pairs = pairs.sort_values(["pilot_mmsi", "vessel_mmsi", "time_bin"], kind="mergesort").reset_index(drop=True)
    return pairs


def classify_transfer_zone(lat: float, lon: float, zones: dict[str, Polygon]) -> str:
    point = Point(lon, lat)
    if zones["embark"].contains(point):
        return "embark"
    if zones["disembark"].contains(point):
        return "disembark"
    if zones["study_area"].contains(point):
        return "study_area_other"
    return "outside"


def score_event(
    min_distance_m: float,
    duration_min: float,
    mean_rel_speed_kn: float,
    mean_heading_diff_deg: float,
    course_aligned_share: float,
    speed_similar_share: float,
    boarding_share: float,
    max_distance_m: float,
    min_duration_min: float,
) -> float:
    proximity_score = 1.0 - min(min_distance_m / max_distance_m, 1.0)
    duration_score = min(duration_min / max(min_duration_min * 2.0, 1.0), 1.0)
    speed_score = 1.0 - min(mean_rel_speed_kn / 8.0, 1.0)
    heading_score = 1.0 - min(mean_heading_diff_deg / 90.0, 1.0)
    kinematic_score = 0.45 * course_aligned_share + 0.35 * speed_similar_share + 0.20 * boarding_share
    score = (
        0.3 * proximity_score
        + 0.2 * duration_score
        + 0.15 * speed_score
        + 0.1 * heading_score
        + 0.25 * kinematic_score
    )
    return float(np.clip(score, 0.0, 1.0))


def segment_events(
    pairs: pd.DataFrame,
    max_distance_m: float,
    min_duration_min: float,
    max_gap_min: float,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()

    zones = load_zone_lookup()
    events: list[dict] = []

    for (pilot_mmsi, vessel_mmsi), grp in pairs.groupby(["pilot_mmsi", "vessel_mmsi"], sort=False):
        grp = grp.sort_values("time_bin", kind="mergesort").reset_index(drop=True)
        gap_min = grp["time_bin"].diff().dt.total_seconds().div(60.0)
        event_break = gap_min.isna() | gap_min.gt(max_gap_min)
        grp["event_seq"] = event_break.cumsum()

        for _, event_grp in grp.groupby("event_seq", sort=False):
            start_time = event_grp["pilot_datetime"].min()
            end_time = event_grp["pilot_datetime"].max()
            duration_min = float((end_time - start_time).total_seconds() / 60.0)
            if duration_min < min_duration_min:
                continue

            min_distance_m = float(event_grp["distance_m"].min())
            min_antenna_distance_m = float(event_grp["antenna_distance_m"].min())
            avg_antenna_distance_m = float(event_grp["antenna_distance_m"].mean())
            min_center_distance_m = float(event_grp["center_distance_m"].min())
            min_domain_clearance_m = float(event_grp["domain_clearance_m"].min())
            mean_rel_speed_kn = float(event_grp["rel_speed_kn"].mean(skipna=True))
            mean_heading_diff_deg = float(event_grp["heading_diff_deg"].mean(skipna=True))
            course_aligned_share = float(event_grp["course_aligned"].mean())
            speed_similar_share = float(event_grp["speed_similar"].mean())
            boarding_share = float(event_grp["boarding_operation"].mean())
            centroid_lat = float(event_grp["interaction_mid_lat"].mean())
            centroid_lon = float(event_grp["interaction_mid_lon"].mean())
            meeting_idx = event_grp["distance_m"].astype(float).idxmin()
            meeting_row = event_grp.loc[meeting_idx]
            meeting_ref_lat = float((meeting_row["pilot_lat"] + meeting_row["vessel_lat"]) / 2.0)
            meeting_ref_lon = float((meeting_row["pilot_lon"] + meeting_row["vessel_lon"]) / 2.0)
            meeting_lat = float(meeting_row["interaction_mid_lat"])
            meeting_lon = float(meeting_row["interaction_mid_lon"])
            zone_class = classify_transfer_zone(centroid_lat, centroid_lon, zones)
            passes_course_rule = bool(course_aligned_share >= 0.5)
            passes_speed_rule = bool(speed_similar_share >= 0.5)
            passes_boarding_rule = bool(boarding_share >= 0.3)
            passes_kinematic_rule = passes_course_rule or passes_speed_rule or passes_boarding_rule
            if not passes_kinematic_rule:
                continue
            traffic_direction = classify_event_direction(
                start_lat=float(event_grp["vessel_lat"].iloc[0]),
                start_lon=float(event_grp["vessel_lon"].iloc[0]),
                end_lat=float(event_grp["vessel_lat"].iloc[-1]),
                end_lon=float(event_grp["vessel_lon"].iloc[-1]),
                avg_heading_deg=float(event_grp["vessel_crs"].mean(skipna=True)),
            )
            confidence_score = score_event(
                min_distance_m=min_distance_m,
                duration_min=duration_min,
                mean_rel_speed_kn=mean_rel_speed_kn,
                mean_heading_diff_deg=mean_heading_diff_deg,
                course_aligned_share=course_aligned_share,
                speed_similar_share=speed_similar_share,
                boarding_share=boarding_share,
                max_distance_m=max_distance_m,
                min_duration_min=min_duration_min,
            )

            start_stamp = pd.Timestamp(start_time).strftime("%Y%m%d_%H%M")
            event_id = f"INT_{pilot_mmsi}_{vessel_mmsi}_{start_stamp}"

            events.append(
                {
                    "event_id": event_id,
                    "pilot_mmsi": pilot_mmsi,
                    "vessel_mmsi": vessel_mmsi,
                    "pilot_ship_name": event_grp["pilot_ship_name"].dropna().iloc[0]
                    if event_grp["pilot_ship_name"].notna().any()
                    else None,
                    "vessel_ship_name": event_grp["vessel_ship_name"].dropna().iloc[0]
                    if event_grp["vessel_ship_name"].notna().any()
                    else None,
                    "vessel_ship_type_class": event_grp["ship_type_class"].dropna().iloc[0]
                    if event_grp["ship_type_class"].notna().any()
                    else None,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_min": duration_min,
                    "n_points": int(len(event_grp)),
                    "meeting_time": meeting_row["pilot_datetime"],
                    "meeting_lat": meeting_lat,
                    "meeting_lon": meeting_lon,
                    "meeting_ref_lat": meeting_ref_lat,
                    "meeting_ref_lon": meeting_ref_lon,
                    "min_distance_m": min_distance_m,
                    "avg_distance_m": float(event_grp["distance_m"].mean()),
                    "max_distance_m": float(event_grp["distance_m"].max()),
                    "min_antenna_distance_m": min_antenna_distance_m,
                    "avg_antenna_distance_m": avg_antenna_distance_m,
                    "min_center_distance_m": min_center_distance_m,
                    "min_domain_clearance_m": min_domain_clearance_m,
                    "avg_rel_speed_kn": mean_rel_speed_kn,
                    "avg_heading_diff_deg": mean_heading_diff_deg,
                    "course_aligned_share": course_aligned_share,
                    "speed_similar_share": speed_similar_share,
                    "boarding_operation_share": boarding_share,
                    "passes_course_rule": passes_course_rule,
                    "passes_speed_rule": passes_speed_rule,
                    "passes_boarding_rule": passes_boarding_rule,
                    "passes_kinematic_rule": passes_kinematic_rule,
                    "dominant_rule_tag": event_grp["rule_tag"].mode().iloc[0],
                    "pilot_avg_speed_kn": float(event_grp["pilot_spd_kn"].mean(skipna=True)),
                    "vessel_avg_speed_kn": float(event_grp["vessel_spd_kn"].mean(skipna=True)),
                    "estimated_tonnage": float(event_grp["estimated_tonnage"].median(skipna=True))
                    if event_grp["estimated_tonnage"].notna().any()
                    else np.nan,
                    "loa_est_m": float(event_grp["loa_est_m"].median(skipna=True))
                    if event_grp["loa_est_m"].notna().any()
                    else np.nan,
                    "beam_est_m": float(event_grp["beam_est_m"].median(skipna=True))
                    if event_grp["beam_est_m"].notna().any()
                    else np.nan,
                    "centroid_lat": centroid_lat,
                    "centroid_lon": centroid_lon,
                    "transfer_zone_class": zone_class,
                    "traffic_direction": traffic_direction,
                    "confidence_score": confidence_score,
                    "likely_transfer": bool(
                        (confidence_score >= 0.55)
                        and (min_distance_m <= max_distance_m * 0.75)
                        and passes_kinematic_rule
                        and (zone_class in {"embark", "disembark", "study_area_other"})
                    ),
                }
            )

    if not events:
        return pd.DataFrame()

    return pd.DataFrame(events).sort_values(["start_time", "pilot_mmsi", "vessel_mmsi"]).reset_index(drop=True)


def save_outputs(
    events: pd.DataFrame,
    out_dir: Path,
    report_name: str,
    csv_name: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / csv_name
    report_path = out_dir / report_name
    events_to_save = events.copy()
    if events_to_save.empty:
        events_to_save = pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS)
    else:
        for col in EVENT_OUTPUT_COLUMNS:
            if col not in events_to_save.columns:
                events_to_save[col] = pd.NA
        events_to_save = events_to_save[EVENT_OUTPUT_COLUMNS]

    events_to_save.to_csv(csv_path, index=False)

    report = {
        "rows": int(len(events_to_save)),
        "pilot_mmsi_count": int(events_to_save["pilot_mmsi"].nunique()) if len(events_to_save) else 0,
        "vessel_mmsi_count": int(events_to_save["vessel_mmsi"].nunique()) if len(events_to_save) else 0,
        "likely_transfer_count": int(events_to_save["likely_transfer"].sum()) if len(events_to_save) else 0,
        "zone_counts": events_to_save["transfer_zone_class"].value_counts(dropna=False).to_dict() if len(events_to_save) else {},
        "traffic_direction_counts": events_to_save["traffic_direction"].value_counts(dropna=False).to_dict() if len(events_to_save) else {},
        "csv_path": str(csv_path),
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect pilot boat / mother ship interception events")
    parser.add_argument("--master", type=str, default=None, help="Path to AIS master parquet/csv")
    parser.add_argument("--sample-rows", type=int, default=0, help="Load first N rows for quick tests")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive UTC start date, e.g. 2024-08-01")
    parser.add_argument("--end-date", type=str, default=None, help="Exclusive UTC end date, e.g. 2024-09-01")
    parser.add_argument("--max-distance-m", type=float, default=500.0, help="Pilot-vessel max separation")
    parser.add_argument("--min-duration-min", type=float, default=10.0, help="Minimum sustained duration")
    parser.add_argument("--max-gap-min", type=float, default=15.0, help="Maximum allowed gap within one event")
    parser.add_argument("--tag", type=str, default="latest", help="Output suffix tag")
    parser.add_argument("--pilot-fleet", choices=["legacy", "verified"], default="legacy",
                        help="legacy = reference files + tug fallback; verified = 13 confirmed pilot boats")
    args = parser.parse_args()

    master_file = _pick_master_path(
        args.master,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    sample_rows = args.sample_rows if args.sample_rows > 0 else None
    pilot_mmsi = set(VERIFIED_PILOT_MMSI) if args.pilot_fleet == "verified" else load_pilot_mmsi()

    print(f"Loading AIS master: {master_file}")
    master = load_master(master_file, sample_rows=sample_rows)
    prepared = prepare_detection_frame(
        master,
        pilot_mmsi=pilot_mmsi,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"Prepared rows in study area: {len(prepared):,}")
    pairs = build_candidate_pairs(prepared, max_distance_m=args.max_distance_m)
    print(f"Candidate pair rows: {len(pairs):,}")

    events = segment_events(
        pairs,
        max_distance_m=args.max_distance_m,
        min_duration_min=args.min_duration_min,
        max_gap_min=args.max_gap_min,
    )
    report = save_outputs(
        events,
        OUT_DIR,
        report_name=f"interceptions_{args.tag}_report.json",
        csv_name=f"interceptions_{args.tag}.csv",
    )

    print("Interception detection complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
