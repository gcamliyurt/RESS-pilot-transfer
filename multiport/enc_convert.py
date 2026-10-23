"""Convert the NOAA ENC cells selected by enc_select.py to per-port GeoJSON layers in the
format the Busan optimiser reads (one file per S-57 object class, <port>_<OBJL>.geojson),
clipped to the port's AIS bbox (+0.05 deg). Read with GDAL's S57 driver via pyogrio,
updates applied. A 'band' property (usage band 3-5) is kept for priority (5 = harbour)."""
from __future__ import annotations
import json, os, warnings
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box

os.environ["OGR_S57_OPTIONS"] = "RETURN_PRIMITIVES=OFF,RETURN_LINKAGES=OFF,LNAM_REFS=OFF,UPDATES=APPLY,SPLIT_MULTIPOINT=OFF,ADD_SOUNDG_DEPTH=OFF"
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
ENC = ROOT / "data" / "raw" / "multiport_enc"
OUT = ROOT / "data" / "processed" / "multiport" / "enc"
LAYERS = ["DEPARE", "DRGARE", "DEPCNT", "SOUNDG", "LNDARE", "BOYLAT", "BOYCAR", "BOYSPP", "BOYSAW", "BOYISD",
          "BCNLAT", "BCNCAR", "BCNSPP", "BCNISD", "FAIRWY", "TSSBND", "TSELNE", "TSSLPT", "RECTRC", "NAVLNE",
          "DWRTCL", "DWRTPT", "OBSTRN", "UWTROC", "WRECKS", "PILBOP", "SLCONS", "M_COVR"]


def cell_file(cell: str) -> Path | None:
    hits = list((ENC / "cells").rglob(f"{cell}.000"))
    return hits[0] if hits else None


def main() -> None:
    cells = pd.read_csv(ENC / "cells.csv")
    for port, g in cells.groupby("port"):
        bb = eval(g.bbox.iloc[0]); clip = box(bb[0] - 0.05, bb[1] - 0.05, bb[2] + 0.05, bb[3] + 0.05)
        out = OUT / port; out.mkdir(parents=True, exist_ok=True)
        counts = {}
        for layer in LAYERS:
            parts = []
            for r in g.sort_values("band", ascending=False).itertuples():
                f = cell_file(r.cell)
                if f is None or layer not in [l[0] for l in pyogrio.list_layers(f)]:
                    continue
                try:
                    df = pyogrio.read_dataframe(f, layer=layer)
                except Exception:
                    continue
                if df.empty:
                    continue
                df = df[df.geometry.notna()]
                df = df[df.intersects(clip)]
                if df.empty:
                    continue
                df["band"], df["cell"] = r.band, r.cell
                parts.append(df)
            if not parts:
                continue
            gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
            keep = [c for c in gdf.columns if c in ("geometry", "band", "cell", "OBJNAM", "DRVAL1", "DRVAL2", "VALDCO",
                                                    "CATLAM", "CATCAM", "COLOUR", "BOYSHP", "VALSOU", "CATOBS", "WATLEV",
                                                    "CATSLC", "ORIENT", "TRAFIC", "CATTSS")]
            gdf = gdf[keep]
            for c in gdf.columns:
                if c != "geometry" and gdf[c].dtype == object:
                    gdf[c] = gdf[c].astype(str)
            gdf.to_file(out / f"{port}_{layer}.geojson", driver="GeoJSON")
            counts[layer] = len(gdf)
        print(port, counts, flush=True)
        (out / "layer_counts.json").write_text(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
