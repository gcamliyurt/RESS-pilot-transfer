from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_ukc(depth_m: float | int | None, draft_m: float | int | None, squat_factor: float = 0.0) -> float:
    if pd.isna(depth_m) or pd.isna(draft_m):
        return np.nan
    return float(depth_m) - float(draft_m) - float(squat_factor)


def classify_ukc_risk(ukc_m: float | int | None) -> str:
    if pd.isna(ukc_m):
        return "unknown"
    if float(ukc_m) < 2.0:
        return "critical"
    if float(ukc_m) < 5.0:
        return "caution"
    return "safe"


def required_ship_domain_area_m2(loa_m: float | int | None) -> float:
    if pd.isna(loa_m):
        return np.nan
    loa = float(loa_m)
    return (loa * 4.0) * (loa * 2.0) * math.pi / 4.0
