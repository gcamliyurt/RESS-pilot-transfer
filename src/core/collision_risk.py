from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.core.detect_interceptions import haversine_distance_m
from src.core.vessel_geometry import (
    ais_reference_to_center,
    hull_half_diagonal_m,
    pair_geometry_metrics,
    ship_domain_radius_m,
)


def course_to_velocity_ms(speed_kn: float | int | None, course_deg: float | int | None) -> tuple[float, float]:
    if speed_kn is None or course_deg is None or pd.isna(speed_kn) or pd.isna(course_deg):
        return 0.0, 0.0
    speed_ms = float(speed_kn) * 0.514444
    theta = math.radians(float(course_deg))
    return speed_ms * math.sin(theta), speed_ms * math.cos(theta)


def relative_cpa_tcpa(
    own_lat: float,
    own_lon: float,
    own_sog: float | int | None,
    own_crs: float | int | None,
    tgt_lat: float,
    tgt_lon: float,
    tgt_sog: float | int | None,
    tgt_crs: float | int | None,
) -> tuple[float, float]:
    mean_lat = math.radians((own_lat + tgt_lat) / 2.0)
    dx = (tgt_lon - own_lon) * 111320.0 * math.cos(mean_lat)
    dy = (tgt_lat - own_lat) * 111320.0
    rvx = course_to_velocity_ms(tgt_sog, tgt_crs)[0] - course_to_velocity_ms(own_sog, own_crs)[0]
    rvy = course_to_velocity_ms(tgt_sog, tgt_crs)[1] - course_to_velocity_ms(own_sog, own_crs)[1]
    rv2 = rvx * rvx + rvy * rvy
    if rv2 == 0:
        cpa_m = float(math.hypot(dx, dy))
        return cpa_m, np.nan
    tcpa_s = -((dx * rvx) + (dy * rvy)) / rv2
    cpa_x = dx + rvx * tcpa_s
    cpa_y = dy + rvy * tcpa_s
    return float(math.hypot(cpa_x, cpa_y)), float(tcpa_s / 60.0)


def relative_cpa_tcpa_geometry(
    own_lat: float,
    own_lon: float,
    own_sog: float | int | None,
    own_crs: float | int | None,
    own_dim_a: float | int | None,
    own_dim_b: float | int | None,
    own_dim_c: float | int | None,
    own_dim_d: float | int | None,
    tgt_lat: float,
    tgt_lon: float,
    tgt_sog: float | int | None,
    tgt_crs: float | int | None,
    tgt_dim_a: float | int | None,
    tgt_dim_b: float | int | None,
    tgt_dim_c: float | int | None,
    tgt_dim_d: float | int | None,
    own_loa: float | int | None = None,
    own_beam: float | int | None = None,
    tgt_loa: float | int | None = None,
    tgt_beam: float | int | None = None,
) -> dict[str, float]:
    own_center_lat, own_center_lon = ais_reference_to_center(
        own_lat, own_lon, own_crs, own_dim_a, own_dim_b, own_dim_c, own_dim_d
    )
    tgt_center_lat, tgt_center_lon = ais_reference_to_center(
        tgt_lat, tgt_lon, tgt_crs, tgt_dim_a, tgt_dim_b, tgt_dim_c, tgt_dim_d
    )

    center_cpa_m, tcpa_min = relative_cpa_tcpa(
        own_center_lat,
        own_center_lon,
        own_sog,
        own_crs,
        tgt_center_lat,
        tgt_center_lon,
        tgt_sog,
        tgt_crs,
    )
    own_domain_radius = ship_domain_radius_m(
        own_dim_a, own_dim_b, own_dim_c, own_dim_d, loa_fallback=own_loa, beam_fallback=own_beam
    )
    tgt_domain_radius = ship_domain_radius_m(
        tgt_dim_a, tgt_dim_b, tgt_dim_c, tgt_dim_d, loa_fallback=tgt_loa, beam_fallback=tgt_beam
    )
    own_hull_radius = hull_half_diagonal_m(
        own_dim_a, own_dim_b, own_dim_c, own_dim_d, loa_fallback=own_loa, beam_fallback=own_beam
    )
    tgt_hull_radius = hull_half_diagonal_m(
        tgt_dim_a, tgt_dim_b, tgt_dim_c, tgt_dim_d, loa_fallback=tgt_loa, beam_fallback=tgt_beam
    )
    geometry_now = pair_geometry_metrics(
        own_lat,
        own_lon,
        own_crs,
        own_dim_a,
        own_dim_b,
        own_dim_c,
        own_dim_d,
        tgt_lat,
        tgt_lon,
        tgt_crs,
        tgt_dim_a,
        tgt_dim_b,
        tgt_dim_c,
        tgt_dim_d,
        own_loa_fallback=own_loa,
        own_beam_fallback=own_beam,
        tgt_loa_fallback=tgt_loa,
        tgt_beam_fallback=tgt_beam,
    )
    return {
        "center_cpa_m": float(center_cpa_m),
        "hull_cpa_m": float(center_cpa_m - (own_hull_radius + tgt_hull_radius)),
        "domain_cpa_m": float(center_cpa_m - (own_domain_radius + tgt_domain_radius)),
        "tcpa_min": float(tcpa_min) if pd.notna(tcpa_min) else np.nan,
        "current_hull_clearance_m": float(geometry_now["hull_clearance_m"]),
        "current_domain_clearance_m": float(geometry_now["domain_clearance_m"]),
    }


def park_risk_proxy(
    own_loa: float | int | None,
    own_beam: float | int | None,
    tgt_loa: float | int | None,
    distance_m: float,
    cpa_m: float | int | None,
    tcpa_min: float | int | None,
    relative_bearing_deg: float | int | None,
) -> float:
    # Research-side proxy inspired by the Park material. Standard CPA/TCPA remain primary.
    loa_term = np.nanmean([own_loa, tgt_loa]) if any(pd.notna(x) for x in [own_loa, tgt_loa]) else 100.0
    beam_term = own_beam if pd.notna(own_beam) else 20.0
    cpa_nm = float(cpa_m) / 1852.0 if pd.notna(cpa_m) else 1.0
    tcpa_min = float(tcpa_min) if pd.notna(tcpa_min) else 30.0
    distance_nm = float(distance_m) / 1852.0
    rel_b = float(relative_bearing_deg) if pd.notna(relative_bearing_deg) else 180.0
    cross_factor = 0.5 if 45 <= rel_b <= 135 or 225 <= rel_b <= 315 else 0.2
    risk = (
        5.081905
        + 0.002517 * float(loa_term)
        + 0.01 * float(beam_term)
        + cross_factor
        - 0.430710 * distance_nm
        - 0.35 * cpa_nm
        - 0.03 * max(tcpa_min, 0.0)
    )
    return float(risk)


def classify_collision_risk(cpa_m: float | int | None) -> str:
    if pd.isna(cpa_m):
        return "unknown"
    if float(cpa_m) < 100:
        return "high"
    if float(cpa_m) < 300:
        return "medium"
    return "low"


def summarize_nearby_traffic(
    meeting_lat: float,
    meeting_lon: float,
    third_party_rows: pd.DataFrame,
    vessel_row: pd.Series,
) -> dict[str, object]:
    if third_party_rows.empty:
        return {
            "surrounding_vessel_count_0_5nm": 0,
            "surrounding_vessel_count_1_0nm": 0,
            "nearest_third_vessel_mmsi": pd.NA,
            "nearest_third_vessel_distance_m": np.nan,
            "nearest_third_vessel_cpa_m": np.nan,
            "nearest_third_vessel_tcpa_min": np.nan,
            "collision_risk_score": np.nan,
            "collision_risk_class": "unknown",
        }

    third = third_party_rows.copy()
    # One row per vessel (its latest fix in the window); counting AIS rows inflated
    # the vessel counts roughly by the number of position reports per vessel.
    if "datetime" in third.columns:
        third = third.sort_values("datetime")
    third = third.drop_duplicates("mmsi", keep="last")
    third["dist_to_meeting_m"] = haversine_distance_m(
        np.full(len(third), float(meeting_lat), dtype="float64"),
        np.full(len(third), float(meeting_lon), dtype="float64"),
        third["lat"].to_numpy(dtype="float64"),
        third["lon"].to_numpy(dtype="float64"),
    )
    nearest = third.sort_values("dist_to_meeting_m").iloc[0]
    collision_metrics = relative_cpa_tcpa_geometry(
        float(vessel_row["lat"]),
        float(vessel_row["lon"]),
        vessel_row.get("spd"),
        vessel_row.get("crs"),
        vessel_row.get("dim_a"),
        vessel_row.get("dim_b"),
        vessel_row.get("dim_c"),
        vessel_row.get("dim_d"),
        float(nearest["lat"]),
        float(nearest["lon"]),
        nearest.get("spd"),
        nearest.get("crs"),
        nearest.get("dim_a"),
        nearest.get("dim_b"),
        nearest.get("dim_c"),
        nearest.get("dim_d"),
        own_loa=vessel_row.get("loa_est_m"),
        own_beam=vessel_row.get("beam_est_m"),
        tgt_loa=nearest.get("loa_est_m"),
        tgt_beam=nearest.get("beam_est_m"),
    )
    rel_bearing = None
    try:
        rel_bearing = (
            math.degrees(
                math.atan2(
                    (float(nearest["lon"]) - float(vessel_row["lon"])) * math.cos(math.radians(float(vessel_row["lat"]))),
                    float(nearest["lat"]) - float(vessel_row["lat"]),
                )
            )
            + 360.0
        ) % 360.0
    except Exception:
        pass
    return {
        "surrounding_vessel_count_0_5nm": int((third["dist_to_meeting_m"] <= 926).sum()),
        "surrounding_vessel_count_1_0nm": int((third["dist_to_meeting_m"] <= 1852).sum()),
        "nearest_third_vessel_mmsi": nearest["mmsi"],
        "nearest_third_vessel_distance_m": float(nearest["dist_to_meeting_m"]),
        "nearest_third_vessel_center_cpa_m": collision_metrics["center_cpa_m"],
        "nearest_third_vessel_cpa_m": collision_metrics["domain_cpa_m"],
        "nearest_third_vessel_hull_cpa_m": collision_metrics["current_hull_clearance_m"],
        "nearest_third_vessel_domain_clearance_m": collision_metrics["current_domain_clearance_m"],
        "nearest_third_vessel_tcpa_min": collision_metrics["tcpa_min"],
        "collision_risk_score": park_risk_proxy(
            vessel_row.get("loa_est_m"),
            vessel_row.get("beam_est_m"),
            nearest.get("loa_est_m"),
            float(nearest["dist_to_meeting_m"]),
            collision_metrics["domain_cpa_m"],
            collision_metrics["tcpa_min"],
            rel_bearing,
        ),
        "collision_risk_class": classify_collision_risk(collision_metrics["domain_cpa_m"]),
    }
