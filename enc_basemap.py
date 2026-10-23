"""Chart-style background from the official KHOA ENC cells (S-57), for all map figures.

Same cells as ../meeting_point_safety.py, read directly with pyogrio's S-57 driver:
band 6 (berthing: KR65886D, KR65887D, KR65896D, KR658D97) where it covers, band 5
(harbour: KR5G3A44, KR5G3E22) elsewhere. ENC coverage is 128.75-129.00 E, 34.75-35.25 N;
area outside it is hatched and labelled, never filled from another source.

    from enc_basemap import draw_enc
    draw_enc(ax, extent=(lon0, lon1, lat0, lat1))
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
from shapely.geometry import box
from shapely.ops import unary_union

CHARTS = Path(__file__).resolve().parents[2] / "6.6 pilotTransferDevice" / "CHARTS"
BAND6 = ["KR65886D", "KR65887D", "KR65896D", "KR658D97"]
BAND5 = ["KR5G3A44", "KR5G3E22"]
os.environ.setdefault("OGR_S57_OPTIONS", "SPLIT_MULTIPOINT=ON,ADD_SOUNDG_DEPTH=ON,RETURN_PRIMITIVES=OFF")

# Paper-chart palette (INT 1 style): land buff, shallow blue, deep white.
LAND = "#F2E2B8"
LAND_EDGE = "#7A6A45"
DEPTH_BANDS = [(0, 5, "#9FC9E8"), (5, 10, "#C3DDF1"), (10, 20, "#E2EEF8"), (20, 1e9, "#FFFFFF")]


def _read(cell: str, layer: str) -> gpd.GeoDataFrame:
    path = CHARTS / f"{cell}.000"
    if layer not in {l[0] for l in pyogrio.list_layers(path)}:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return pyogrio.read_dataframe(path, layer=layer)


@functools.lru_cache(maxsize=None)
def _coverage(cell: str):
    return unary_union(_read(cell, "M_COVR").geometry.values)


@functools.lru_cache(maxsize=None)
def layer(name: str) -> gpd.GeoDataFrame:
    """One layer from all cells, band 6 preferred: band-5 features are clipped to
    the area that no band-6 cell covers."""
    b6_cov = unary_union([_coverage(c) for c in BAND6])
    parts = [_read(c, name) for c in BAND6]
    for c in BAND5:
        g = _read(c, name)
        if len(g):
            g = g.copy()
            g["geometry"] = g.geometry.difference(b6_cov) if g.geom_type.iloc[0] != "Point" else g.geometry
            if g.geom_type.iloc[0] == "Point":
                g = g[~g.geometry.within(b6_cov)]
            parts.append(g[~g.geometry.is_empty])
    parts = [p for p in parts if len(p)]
    return gpd.GeoDataFrame(gpd.pd.concat(parts, ignore_index=True), crs="EPSG:4326") if parts else gpd.GeoDataFrame(geometry=[])


def enc_coverage():
    return unary_union([_coverage(c) for c in BAND6 + BAND5])


def draw_enc(ax, extent, depth=True, contours=True, structures=True, aids=True, pilot=True, label_outside=True,
             hatch_outside=True):
    """Draw the ENC background into ax (lon/lat axes) and set the extent."""
    lon0, lon1, lat0, lat1 = extent
    view = box(lon0, lat0, lon1, lat1)
    ax.set_facecolor("white")
    if depth:
        dep = layer("DEPARE")
        dep = dep[dep.intersects(view)]
        for lo, hi, col in DEPTH_BANDS:
            sel = dep[(dep.DRVAL1.fillna(0) >= lo) & (dep.DRVAL1.fillna(0) < hi)]
            if len(sel):
                sel.plot(ax=ax, color=col, linewidth=0, zorder=0)
    if contours:
        dc = layer("DEPCNT")
        dc = dc[dc.intersects(view)]
        if len(dc):
            dc.plot(ax=ax, color="#6B9AC4", linewidth=0.35, zorder=1)
    lnd = layer("LNDARE")
    lnd = lnd[lnd.intersects(view) & lnd.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(lnd):
        lnd.plot(ax=ax, color=LAND, linewidth=0, zorder=2)
    # Coastline from COALNE (polygon edges would show the cell-boundary cuts)
    co = layer("COALNE")
    co = co[co.intersects(view)]
    if len(co):
        co.plot(ax=ax, color=LAND_EDGE, linewidth=0.6, zorder=2.5)
    if structures:
        sl = layer("SLCONS")
        sl = sl[sl.intersects(view)]
        if len(sl):
            sl.plot(ax=ax, color="#5A5A5A", linewidth=1.0, zorder=3)
    if aids:
        for name, mk in (("BOYLAT", "o"), ("BCNLAT", "^"), ("BOYSPP", "s"), ("BOYCAR", "D")):
            a = layer(name)
            a = a[a.geom_type == "Point"]
            a = a[a.intersects(view)]
            if len(a):
                # S-57 COLOUR: 3 = red, 4 = green (lists in pyogrio); others black-yellow
                cs = a["COLOUR"].astype(str) if "COLOUR" in a else gpd.pd.Series("", index=a.index)
                col = np.where(cs.str.contains("3"), "#D55E00", np.where(cs.str.contains("4"), "#009E73", "#E6C200"))
                ax.scatter(a.geometry.x, a.geometry.y, s=9, marker=mk, c=col, edgecolors="black",
                           linewidths=0.3, zorder=4)
    if pilot:
        pb = layer("PILBOP")
        pb = pb[pb.intersects(view)]
        if len(pb):
            poly = pb[pb.geom_type.isin(["Polygon", "MultiPolygon"])]
            if len(poly):
                poly.plot(ax=ax, facecolor="none", edgecolor="#CC79A7", linewidth=1.0, linestyle="--", zorder=4)
    outside = view.difference(enc_coverage())
    if hatch_outside and not outside.is_empty and outside.area > 1e-6:
        gpd.GeoSeries([outside], crs="EPSG:4326").plot(ax=ax, facecolor="#F4F4F4", edgecolor="#BBBBBB",
                                                       hatch="////", linewidth=0.3, zorder=0.5)
        if label_outside:
            c = outside.representative_point()
            ax.text(c.x, c.y, "outside ENC\ncoverage", fontsize=7, color="#777777", ha="center", va="center", zorder=5)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / np.cos(np.radians((lat0 + lat1) / 2)))
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_enc(ax, (128.70, 128.92, 34.88, 35.10))
    out = Path(__file__).parent / "RESS_JRESS-D-26-03347" / "figures_R2" / "_enc_basemap_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)
