"""RESS R2 figures, group A: maps, optimisation summary, model fit and new figures.

All data are R2 (13 verified pilot boats, log-confirmed transfers, pilot-boat dispatch
decision time, leakage-free optimiser). Full width = 17 cm, fonts >= 9 pt,
Okabe-Ito colours with distinct markers / line styles (grayscale-readable).
Output: RESS_JRESS-D-26-03347/figures_R2/*.png (300 dpi)
"""

from __future__ import annotations

import geopandas as gpd
from enc_basemap import draw_enc
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from shapely.geometry import box, shape

from src.core.detect_interceptions import load_zone_lookup
from src.core.enrich_event_context import BREAKWATER_LIGHTS

ROOT = Path(__file__).parent
OUT = ROOT / "RESS_JRESS-D-26-03347" / "figures_R2"
OUT.mkdir(parents=True, exist_ok=True)
ENC = ROOT / "data" / "processed" / "charts_enc_geojson"
R2 = ROOT / "data" / "processed" / "r2"
W = 17 / 2.54  # full print width, inches
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
DIR_STYLE = {"inbound": dict(color=OI["blue"], marker="o"), "outbound": dict(color=OI["verm"], marker="^")}
VERIFIED_PILOTS = {440008040, 440105210, 440110850, 440111140, 440111150, 440111160, 440111190,
                   440111210, 440111230, 440121630, 440127410, 440135830, 440155330}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "font.family": "DejaVu Sans",
                     "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- map helpers
def _geoms(tag: str, extent):
    clip = box(*extent)
    out = []
    for f in ENC.glob(f"*_{tag}.geojson"):
        for ft in json.loads(f.read_text())["features"]:
            try:
                g = shape(ft["geometry"])
            except Exception:
                continue
            if g.intersects(clip):
                out.append(g.intersection(clip))
    return out


def _draw(ax, g, **kw):
    gs = getattr(g, "geoms", [g])
    for p in gs:
        if p.geom_type == "Polygon":
            ax.fill(*p.exterior.xy, **kw)
        elif p.geom_type in ("LineString", "LinearRing"):
            ax.plot(*p.xy, **{k: v for k, v in kw.items() if k in ("color", "lw", "ls", "alpha", "zorder", "label")})
        elif hasattr(p, "geoms"):
            _draw(ax, p, **kw)


def base_map(ax, extent, land=True, fairway=True, zones=True, lights=True, step=None, regional=False):
    """Official KHOA ENC background (enc_basemap: land, depth areas, contours, structures,
    aids to navigation). regional=True: maps far beyond the ENC cells get the Natural Earth
    1:50m coastline outside the ENC coverage (and the caption says so)."""
    lon0, lat0, lon1, lat1 = extent
    if land and regional:
        ne = gpd.read_file(ROOT.parents[1] / "6.8 leeSide/data/raw/naturalearth/ne_50m_land.geojson")
        from shapely.ops import unary_union
        from enc_basemap import layer
        enc_water = unary_union(layer("DEPARE").geometry.values)  # Natural Earth land only where the ENC shows no water
        ne.clip(box(*extent).difference(enc_water)).plot(ax=ax, color="#EFE6CC", edgecolor="#8A7A55", linewidth=0.4, zorder=0.2)
        draw_enc(ax, (lon0, lon1, lat0, lat1), depth=False, contours=False, aids=False, pilot=False, label_outside=False,
                 hatch_outside=False)
    elif land:
        draw_enc(ax, (lon0, lon1, lat0, lat1), aids=True, pilot=False)
    if fairway:
        first = True
        for tag in ("RECTRC", "NAVLNE"):
            for g in _geoms(tag, extent):
                _draw(ax, g, color=OI["grey"], lw=0.7, ls="--", zorder=2, label="Recommended track / nav. line" if first else None)
                first = False
    if zones:
        z = load_zone_lookup()
        for key, col, ls, lab in (("embark", OI["blue"], "-", "Embarkation zone"), ("disembark", OI["verm"], "-.", "Disembarkation zone")):
            ax.plot(*z[key].exterior.xy, color=col, lw=1.3, ls=ls, zorder=4, label=lab)
    if lights:
        ax.scatter(BREAKWATER_LIGHTS.lon, BREAKWATER_LIGHTS.lat, marker="*", s=70, color=OI["black"],
                   edgecolor="white", lw=0.5, zorder=6, label="Breakwater lights")
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / np.cos(np.radians((lat0 + lat1) / 2)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}°N"))
    if step:
        ax.set_xticks(np.arange(np.ceil(lon0 / step) * step, lon1, step))
        ax.set_yticks(np.arange(np.ceil(lat0 / step) * step, lat1, step))
    ax.grid(color="#bbbbbb", lw=0.4, ls=":", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    _scale_bar(ax, extent); _north(ax)


def _scale_bar(ax, extent):
    lon0, lat0, lon1, lat1 = extent
    km_per_deg = 111.32 * np.cos(np.radians((lat0 + lat1) / 2))
    span = (lon1 - lon0) * km_per_deg
    L = [l for l in (1, 2, 5, 10, 20, 50) if l <= span / 4][-1]
    x0, y0 = lon0 + 0.05 * (lon1 - lon0), lat0 + 0.05 * (lat1 - lat0)
    dx = L / km_per_deg
    ax.plot([x0, x0 + dx], [y0, y0], color="k", lw=2.5, solid_capstyle="butt", zorder=7)
    ax.plot([x0, x0 + dx / 2], [y0, y0], color="white", lw=1.2, solid_capstyle="butt", zorder=8)
    ax.text(x0 + dx / 2, y0 + 0.02 * (lat1 - lat0), f"{L} km", ha="center", va="bottom", fontsize=8.5, zorder=8,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))


def _north(ax):
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.85), xycoords="axes fraction", ha="center", va="center",
                fontsize=10, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2), zorder=9)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- data
def load_r2():
    d = pd.read_csv(R2 / "modelling_r2.csv", low_memory=False)
    o = pd.concat(pd.read_csv(f, low_memory=False) for f in
                  glob.glob(str(ROOT / "data/processed/meeting_point/optimized_meeting_points_*_r2verified_causal.csv")))
    o = o.merge(d[["event_id", "traffic_direction"]], on="event_id", suffixes=("_ais", ""))
    o = o[o.recommendation_status == "feasible"]  # infeasible events are reported separately, not as recommendations
    return d, o


STUDY = (128.67, 34.86, 128.88, 35.07)  # AIS extent supplied for the study


# ---------------------------------------------------------------- figures
def fig01_traffic():
    week = ROOT / "data/processed/ais_master/weeks/2024-03-week2.csv"
    a = pd.read_csv(week, usecols=["mmsi", "datetime", "lat", "lon", "spd"])
    a = a.iloc[::3]
    pil = a[a.mmsi.isin(VERIFIED_PILOTS)].sort_values(["mmsi", "datetime"])
    oth = a[~a.mmsi.isin(VERIFIED_PILOTS)]
    fig, ax = plt.subplots(figsize=(W, W * 0.82))
    hb = ax.hexbin(oth.lon, oth.lat, gridsize=140, bins="log", cmap="Greys", mincnt=1, linewidths=0, zorder=1.5,
                   extent=(STUDY[0], STUDY[2], STUDY[1], STUDY[3]))
    base_map(ax, STUDY, step=0.05)
    pil["datetime"] = pd.to_datetime(pil.datetime)
    first = True
    for _, g in pil.groupby("mmsi"):
        brk = (g.datetime.diff() > pd.Timedelta(minutes=10)).cumsum()
        for _, s in g.groupby(brk):
            if len(s) > 3:
                ax.plot(s.lon, s.lat, color=OI["orange"], lw=0.6, alpha=0.8, zorder=5,
                        label="Verified pilot-boat tracks (1 week)" if first else None)
                first = False
    cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("AIS reports per cell (log), 2024-03 week 2")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Study area: AIS traffic, boarding zones and pilot-boat operations, Busan New Port")
    save(fig, "figure_01_traffic_context.png")


def fig02_dispersion(d):
    ext = (128.75, 34.92, 128.86, 35.07)  # transfers lie east of 128.75 E, the western edge of ENC coverage
    fig, ax = plt.subplots(figsize=(W * 0.75, W * 0.75))
    base_map(ax, ext, step=0.03)
    for k, st in DIR_STYLE.items():
        g = d[d.traffic_direction == k]
        ax.scatter(g.meeting_lon, g.meeting_lat, s=6, alpha=0.45, lw=0, zorder=5, **st)
        ax.scatter([], [], s=30, label=f"{k.capitalize()} transfers (n = {len(g):,})", **st)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08), framealpha=0.9)
    ax.set_title("Log-confirmed pilot transfer positions (n = 3,243), Sep 2023 – Aug 2024")
    save(fig, "figure_02_transfer_dispersion.png")


def fig04_optimisation(o):
    fig, axs = plt.subplots(2, 2, figsize=(W, W * 0.72))
    n = len(o)
    # (a) source
    lab = {"anchor_ring": "Anchor ring (150–500 m)", "ml_anchor": "ML anchor (kept)", "fairway_centerline": "Fairway centreline",
           "forward_corridor": "Approach line", "zone_grid": "Official zone grid", "rewind_midpoint": "Boat–ship midpoint",
           "historical_hotspot": "Historical median", "zone_centroid": "Zone centroid"}
    s = o.recommended_source.value_counts()
    ax = axs[0, 0]
    ax.barh([lab.get(i, i) for i in s.index][::-1], s.values[::-1], color=OI["blue"], hatch=None)
    for y, v in enumerate(s.values[::-1]):
        ax.text(v + n * 0.01, y, f"{v:,}", va="center", fontsize=8.5)
    ax.set_xlim(0, s.max() * 1.22); ax.set_xlabel("Recommendations")
    ax.set_title(f"(a) Source of point (n = {n:,})", loc="left")
    # (b) relocation ECDF
    ax = axs[0, 1]
    for k, st in DIR_STYLE.items():
        x = np.sort(o.loc[o.traffic_direction == k, "recommended_dist_to_anchor_m"].dropna().to_numpy() / 1000)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=st["color"],
                ls="-" if k == "inbound" else "--", lw=1.5, label=f"{k.capitalize()} (median {np.median(x)*1000:.0f} m)")
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.01); ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3]); ax.set_xlabel("Relocation from ML anchor (km)"); ax.set_ylabel("Cumulative share")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02)); ax.grid(ls=":", lw=0.4)
    ax.set_title("(b) Relocation from ML anchor", loc="left")
    # (c) gates
    ax = axs[1, 0]
    passed = o.recommended_gate_pass.astype(str).str.lower().eq("true")
    reasons = o.loc[~passed, "recommended_gate_reason"].str.split(";").explode().value_counts()
    rl = {"route_hits_buoy_corridor": "Route: buoy corridor", "route_leaves_water_area": "Route: leaves water",
          "poor_fairway_access": "Fairway access", "collision_risk_high": "Collision risk", "too_close_buoy": "Buoy clearance"}
    names = ["Pass all gates"] + [f"Fail: {rl.get(r, r)}" for r in reasons.index]
    vals = [int(passed.sum())] + reasons.tolist()
    cols = [OI["green"]] + [OI["verm"]] * len(reasons)
    hatches = [""] + ["///"] * len(reasons)
    bars = ax.barh(names[::-1], vals[::-1], color=cols[::-1], log=True)
    for b, h in zip(bars, hatches[::-1]):
        b.set_hatch(h); b.set_edgecolor("white")
    for y, v in enumerate(vals[::-1]):
        ax.text(v * 1.15, y, f"{v:,}", va="center", fontsize=8.5)
    ax.set_xlim(1, max(vals) * 6); ax.set_xlabel("Recommendations (log scale)")
    ax.set_title(f"(c) Safety gates: {passed.mean()*100:.1f}% pass", loc="left")
    # (d) collision class
    ax = axs[1, 1]
    cc = o.recommended_collision_risk_class.astype(str).value_counts().reindex(["low", "medium", "high"]).fillna(0)
    b = ax.bar(["Low", "Medium", "High"], cc.values, color=[OI["green"], OI["orange"], OI["verm"]])
    for bar, h in zip(b, ["", "..", "///"]):
        bar.set_hatch(h); bar.set_edgecolor("white")
    for x, v in enumerate(cc.values):
        ax.text(x, v + n * 0.01, f"{int(v):,}\n({v/n*100:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, cc.max() * 1.3); ax.set_ylabel("Recommendations")
    ax.set_title("(d) Collision-risk class (domain clearance)", loc="left")
    fig.tight_layout()
    save(fig, "figure_04_optimisation_summary.png")


def fig_pred_vs_actual(d):
    p = pd.read_csv(R2 / "oof_predictions_r2_dispatch.csv").merge(d[["event_id", "dist_to_breakwater_m"]], on="event_id")
    fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.48))
    for ax, (k, st) in zip(axs, DIR_STYLE.items()):
        g = p[p.traffic_direction == k]
        y, yh = g.dist_to_breakwater_m / 1000, g.pred_dist_m / 1000
        r = y - yh
        lo, hi = np.quantile(r, [0.10, 0.90])
        lim = (0, np.ceil(max(np.quantile(y, 0.995), np.quantile(yh, 0.995))))
        xx = np.linspace(*lim, 50)
        ax.fill_between(xx, xx + lo, xx + hi, color=OI["grey"], alpha=0.25, lw=0, label="80% prediction interval")
        ax.plot(xx, xx, color="k", lw=1, ls="--", label="1:1")
        ax.scatter(yh, y, s=6, alpha=0.4, lw=0, color=st["color"], marker=st["marker"])
        r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ax.text(0.04, 0.96, f"{k.capitalize()}, n = {len(g):,}\nR² = {r2:.3f}\nMAE = {np.abs(r).mean():.3f} km",
                transform=ax.transAxes, va="top", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel("Predicted meeting distance (km)"); ax.set_ylabel("Observed meeting distance (km)")
        ax.grid(ls=":", lw=0.4)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("Out-of-fold predictions at pilot-boat dispatch, ship-grouped 5-fold CV", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "figure_pred_vs_actual.png")


def fig_stations(d):
    s = pd.read_csv(ROOT / "results/revision_r2_step2/environment/kma_stations.csv")
    ext = (128.6, 34.7, 129.9, 35.65)
    fig, ax = plt.subplots(figsize=(W, W * 0.62))
    base_map(ax, ext, fairway=False, lights=False, step=0.3, regional=True)
    mk = {"AWS": ("s", OI["blue"]), "Lighthouse": ("D", OI["purple"]), "Wave buoy": ("^", OI["green"]),
          "Marine buoy": ("o", OI["orange"])}
    ax.scatter(d.meeting_lon, d.meeting_lat, s=2, color=OI["grey"], alpha=0.4, lw=0, zorder=3, label="Transfer positions")
    for net, (m, c) in mk.items():
        g = s[s.network == net]
        has = (g.wind_hours.fillna(0) + g.wave_hours.fillna(0)) > 0
        ax.scatter(g.lon[has], g.lat[has], marker=m, s=45, color=c, edgecolor="k", lw=0.5, zorder=6,
                   label=f"{net} ({has.sum()} with data)")
        ax.scatter(g.lon[~has], g.lat[~has], marker=m, s=45, facecolor="white", edgecolor=c, lw=1.2, zorder=6,
                   label=f"{net} ({(~has).sum()} no data)" if (~has).any() else None)
    used = s[s.station_id == 22104].iloc[0]
    ax.scatter(used.lon, used.lat, s=220, facecolor="none", edgecolor="k", lw=1.5, zorder=7)
    ax.annotate("Geoje buoy 22104\n(series used, full year)", (used.lon, used.lat), xytext=(used.lon + 0.08, used.lat - 0.02),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.8))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
    ax.set_title("KMA stations by network (records Jan–Aug 2024; Geoje buoy full year)")
    save(fig, "figure_stations_map.png")


def fig_thresholds():
    t = pd.read_csv(ROOT / "results/revision_r2_step2/reporting/threshold_response_curves.csv")
    spec = [("Hs_m", "Hs (m)", (1.8, 2.2)), ("wind_kt", "Wind (kt)", (20, 22)),
            ("traffic_0_5nm", "Vessels ≤ 0.5 NM", (4, 6)), ("ukc_m", "UKC (m)", (0, 2))]
    fig, axs = plt.subplots(2, 4, figsize=(W, W * 0.5), sharex="col")
    for j, (v, xl, rng) in enumerate(spec):
        for i, (col, yl) in enumerate((("offshore_share", "Offshore share (> 3 km)"), ("median_dist_km", "Median distance (km)"))):
            ax = axs[i, j]
            ax.axvspan(*rng, color=OI["yellow"], alpha=0.45, lw=0, label="Table 1 range" if (i == 0 and j == 0) else None)
            for k, st in DIR_STYLE.items():
                g = t[(t.variable == v) & (t.direction == k) & (t.n >= 10)]
                ax.plot(g.bin, g[col], color=st["color"], marker=st["marker"], ms=3.5, lw=1.2,
                        ls="-" if k == "inbound" else "--", label=k.capitalize() if (i == 0 and j == 0) else None)
            if j == 0:
                ax.set_ylabel(yl)
            if i == 1:
                ax.set_xlabel(xl)
            ax.grid(ls=":", lw=0.4)
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout()
    save(fig, "figure_S_thresholds.png")


def fig_lambda():
    l = pd.read_csv(ROOT / "results/revision_r2_optimiser_diagnosis_causal/lambda_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(W * 0.6, W * 0.4))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    for k, ls in (("inbound", "-"), ("outbound", "--")):
        g = l[l.direction == k]
        x = g["lambda"].replace(0, 0.01)
        ax.plot(x, g.median_offset_m / 1000, color=OI["blue"], marker="o", ls=ls, lw=1.3, label=f"Median offset, {k}")
        ax2.plot(x, g.mean_safety_score, color=OI["verm"], marker="^", ls=ls, lw=1.3, label=f"Safety score, {k}")
    ax.axvline(0.1, color="k", lw=0.8, ls=":")
    ax.text(0.105, ax.get_ylim()[1] * 0.97, "λ = 0.1 (adopted)", fontsize=8.5, va="top")
    ax.set_xscale("log"); ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax.set_xticklabels(["0", "0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.set_xlabel("Relocation penalty λ (score per km)")
    ax.set_ylabel("Median offset from actual point (km)", color=OI["blue"])
    ax2.set_ylabel("Mean safety score", color=OI["verm"])
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    ax.grid(ls=":", lw=0.4)
    save(fig, "figure_lambda_sensitivity.png")


def main() -> None:
    d, o = load_r2()
    fig01_traffic()
    fig02_dispersion(d)
    fig04_optimisation(o)
    fig_pred_vs_actual(d)
    fig_stations(d)
    fig_thresholds()
    fig_lambda()


if __name__ == "__main__":
    main()
