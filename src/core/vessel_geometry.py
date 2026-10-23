from __future__ import annotations

import math

import pandas as pd


EARTH_RADIUS_M = 6371000.0
DEG_LAT_M = 111320.0
DEFAULT_LOA_M = 50.0
DEFAULT_BEAM_M = 10.0


def _safe_float(value: object) -> float | None:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return None
    return float(num)


def normalize_heading_deg(course_deg: object) -> float:
    course = _safe_float(course_deg)
    if course is None:
        return 0.0
    return float(course % 360.0)


def estimate_loa_beam_m(
    dim_a: object,
    dim_b: object,
    dim_c: object,
    dim_d: object,
    loa_fallback: object = None,
    beam_fallback: object = None,
) -> tuple[float, float]:
    a = _safe_float(dim_a)
    b = _safe_float(dim_b)
    c = _safe_float(dim_c)
    d = _safe_float(dim_d)
    loa = _safe_float(loa_fallback)
    beam = _safe_float(beam_fallback)

    if a is not None or b is not None:
        loa = (a or 0.0) + (b or 0.0)
    if c is not None or d is not None:
        beam = (c or 0.0) + (d or 0.0)

    loa = loa if loa is not None and loa > 0 else DEFAULT_LOA_M
    beam = beam if beam is not None and beam > 0 else max(DEFAULT_BEAM_M, 0.18 * loa)
    return float(loa), float(beam)


def _meters_per_degree_lon(lat_deg: float) -> float:
    return DEG_LAT_M * max(math.cos(math.radians(float(lat_deg))), 1e-6)


def latlon_to_local_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    x = (float(lon) - float(ref_lon)) * _meters_per_degree_lon(ref_lat)
    y = (float(lat) - float(ref_lat)) * DEG_LAT_M
    return float(x), float(y)


def local_xy_to_latlon(x_m: float, y_m: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    lat = float(ref_lat) + float(y_m) / DEG_LAT_M
    lon = float(ref_lon) + float(x_m) / _meters_per_degree_lon(ref_lat)
    return float(lat), float(lon)


def _forward_starboard_vectors(heading_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    theta = math.radians(normalize_heading_deg(heading_deg))
    forward = (math.sin(theta), math.cos(theta))
    starboard = (math.cos(theta), -math.sin(theta))
    return forward, starboard


def antenna_to_center_offset_m(dim_a: object, dim_b: object, dim_c: object, dim_d: object) -> tuple[float, float]:
    a = _safe_float(dim_a) or 0.0
    b = _safe_float(dim_b) or 0.0
    c = _safe_float(dim_c) or 0.0
    d = _safe_float(dim_d) or 0.0
    return (d - c) / 2.0, (a - b) / 2.0


def shift_position_m(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    out_lat = float(lat) + float(north_m) / DEG_LAT_M
    out_lon = float(lon) + float(east_m) / _meters_per_degree_lon(lat)
    return float(out_lat), float(out_lon)


def ais_reference_to_center(
    lat: float,
    lon: float,
    course_deg: object,
    dim_a: object,
    dim_b: object,
    dim_c: object,
    dim_d: object,
) -> tuple[float, float]:
    heading = normalize_heading_deg(course_deg)
    forward, starboard = _forward_starboard_vectors(heading)
    offset_starboard_m, offset_forward_m = antenna_to_center_offset_m(dim_a, dim_b, dim_c, dim_d)
    east_m = forward[0] * offset_forward_m + starboard[0] * offset_starboard_m
    north_m = forward[1] * offset_forward_m + starboard[1] * offset_starboard_m
    return shift_position_m(lat, lon, east_m, north_m)


def hull_half_diagonal_m(
    dim_a: object,
    dim_b: object,
    dim_c: object,
    dim_d: object,
    loa_fallback: object = None,
    beam_fallback: object = None,
) -> float:
    loa_m, beam_m = estimate_loa_beam_m(dim_a, dim_b, dim_c, dim_d, loa_fallback=loa_fallback, beam_fallback=beam_fallback)
    return float(math.hypot(loa_m / 2.0, beam_m / 2.0))


def ship_domain_radius_m(
    dim_a: object,
    dim_b: object,
    dim_c: object,
    dim_d: object,
    loa_fallback: object = None,
    beam_fallback: object = None,
) -> float:
    loa_m, beam_m = estimate_loa_beam_m(dim_a, dim_b, dim_c, dim_d, loa_fallback=loa_fallback, beam_fallback=beam_fallback)
    half_diag = math.hypot(loa_m / 2.0, beam_m / 2.0)
    margin = max(0.25 * loa_m, 1.5 * beam_m)
    return float(half_diag + margin)


def pair_geometry_metrics(
    own_lat: float,
    own_lon: float,
    own_course_deg: object,
    own_dim_a: object,
    own_dim_b: object,
    own_dim_c: object,
    own_dim_d: object,
    tgt_lat: float,
    tgt_lon: float,
    tgt_course_deg: object,
    tgt_dim_a: object,
    tgt_dim_b: object,
    tgt_dim_c: object,
    tgt_dim_d: object,
    own_loa_fallback: object = None,
    own_beam_fallback: object = None,
    tgt_loa_fallback: object = None,
    tgt_beam_fallback: object = None,
) -> dict[str, float]:
    ref_lat = (float(own_lat) + float(tgt_lat)) / 2.0
    ref_lon = (float(own_lon) + float(tgt_lon)) / 2.0

    own_center_lat, own_center_lon = ais_reference_to_center(
        float(own_lat), float(own_lon), own_course_deg, own_dim_a, own_dim_b, own_dim_c, own_dim_d,
    )
    tgt_center_lat, tgt_center_lon = ais_reference_to_center(
        float(tgt_lat), float(tgt_lon), tgt_course_deg, tgt_dim_a, tgt_dim_b, tgt_dim_c, tgt_dim_d,
    )
    own_center_x, own_center_y = latlon_to_local_xy(own_center_lat, own_center_lon, ref_lat, ref_lon)
    tgt_center_x, tgt_center_y = latlon_to_local_xy(tgt_center_lat, tgt_center_lon, ref_lat, ref_lon)
    center_distance_m = float(math.hypot(tgt_center_x - own_center_x, tgt_center_y - own_center_y))
    half_diag_sum_m = hull_half_diagonal_m(
        own_dim_a, own_dim_b, own_dim_c, own_dim_d, loa_fallback=own_loa_fallback, beam_fallback=own_beam_fallback,
    ) + hull_half_diagonal_m(
        tgt_dim_a, tgt_dim_b, tgt_dim_c, tgt_dim_d, loa_fallback=tgt_loa_fallback, beam_fallback=tgt_beam_fallback,
    )
    domain_radius_sum_m = ship_domain_radius_m(
        own_dim_a, own_dim_b, own_dim_c, own_dim_d, loa_fallback=own_loa_fallback, beam_fallback=own_beam_fallback,
    ) + ship_domain_radius_m(
        tgt_dim_a, tgt_dim_b, tgt_dim_c, tgt_dim_d, loa_fallback=tgt_loa_fallback, beam_fallback=tgt_beam_fallback,
    )
    hull_clearance_m = max(0.0, center_distance_m - half_diag_sum_m)
    interaction_mid_x = (own_center_x + tgt_center_x) / 2.0
    interaction_mid_y = (own_center_y + tgt_center_y) / 2.0
    interaction_mid_lat, interaction_mid_lon = local_xy_to_latlon(interaction_mid_x, interaction_mid_y, ref_lat, ref_lon)

    return {
        "own_center_lat": float(own_center_lat),
        "own_center_lon": float(own_center_lon),
        "tgt_center_lat": float(tgt_center_lat),
        "tgt_center_lon": float(tgt_center_lon),
        "center_distance_m": center_distance_m,
        "hull_clearance_m": hull_clearance_m,
        "hull_penetration_m": float(center_distance_m - half_diag_sum_m),
        "domain_clearance_m": float(center_distance_m - domain_radius_sum_m),
        "interaction_mid_lat": float(interaction_mid_lat),
        "interaction_mid_lon": float(interaction_mid_lon),
    }
