"""Select NOAA ENC cells (usage bands 3-5) intersecting each US port's AIS bbox from the NOAA ENC
product catalogue (ISO 19115 XML). Writes data/raw/multiport_enc/cells.csv."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
ENC = ROOT / "data" / "raw" / "multiport_enc"
AIS = ROOT.parents[1] / "6.8 leeSide" / "data" / "raw" / "multiport" / "ais"
PORTS = {"us_galveston": "galveston", "us_longbeach": "longbeach", "us_port_arthur": "port_arthur",
         "us_capehenry": "capehenry", "us_swpass": "swpass", "us_port_everglades": "port_everglades"}


def records():
    s = (ENC / "ENCProdCat_19115.xml").read_text()
    for blk in re.split(r"<MD_Metadata[ >]", s)[1:]:
        t = re.search(r"<title>\s*<gco:CharacterString>([^<]+)<", blk)
        if not t:
            continue
        ed = re.search(r"<edition>\s*<gco:CharacterString>([^<]+)<", blk)
        url = re.search(r"<URL>([^<]+\.zip)</URL>", blk)
        pts = [tuple(map(float, p.split())) for p in re.findall(r"<gml:pos>([^<]+)</gml:pos>", blk)]
        if len(pts) < 3 or not url:
            continue
        # gml:pos order is lat lon
        poly = Polygon([(lo, la) for la, lo in pts]).buffer(0)
        yield t.group(1).strip(), ed.group(1).strip() if ed else "", url.group(1), poly


def port_bbox(folder: str):
    import pyarrow.parquet as pq
    f = sorted((AIS / folder).glob("*.parquet"))[180]
    x = pq.read_table(f, columns=["LAT", "LON"]).to_pandas()
    return box(x.LON.min(), x.LAT.min(), x.LON.max(), x.LAT.max())


def main():
    recs = list(records())
    rows = []
    for port, folder in PORTS.items():
        bb = port_bbox(folder)
        for name, ed, url, poly in recs:
            band = name[2] if len(name) > 2 and name[2].isdigit() else ""
            if band in ("3", "4", "5") and poly.intersects(bb):
                rows.append({"port": port, "cell": name, "band": int(band), "edition": ed, "url": url,
                             "overlap_share": round(poly.intersection(bb).area / bb.area, 3),
                             "bbox": [round(v, 4) for v in bb.bounds]})
    d = pd.DataFrame(rows).sort_values(["port", "band", "cell"])
    d.to_csv(ENC / "cells.csv", index=False)
    print(d.groupby(["port", "band"]).cell.apply(list).to_string())


if __name__ == "__main__":
    main()
