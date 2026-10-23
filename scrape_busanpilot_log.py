"""Download the Busan Harbor Pilots' Association public pilotage log.

Source: http://www.busanpilot.co.kr/information/pilotage (도선 모니터링 현황),
one row per pilotage act: time, ship, GT/LOA, in/out + side, pilot, tugs,
from/to (berth / pilot station), agent. Used as ground truth for the
AIS-detected transfer counts (RESS R2 revision).

Usage: python3 scrape_busanpilot_log.py 2023-09-01 2024-08-31
Output: data/external/busanpilot_log_<start>_<end>.csv (resumable per day)
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

URL = "http://www.busanpilot.co.kr/information/pilotage?p={p}&DT_SHIPS={d}&DT_SHIPE={d}"
OUT_DIR = Path(__file__).parent / "data" / "external" / "busanpilot_days"
DELAY_S = 0.5
COLS = ["no", "status", "datetime", "ship", "gt_loa", "dir_side", "pilot", "tugs", "from_to", "agent", "linehandler"]


def fetch(url: str) -> str:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; LJMU)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(url)


def clean(cell: str) -> str:
    cell = re.sub(r"<br\s*/?\s*>", " | ", cell)
    cell = re.sub(r"<[^>]+>", " ", cell).replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", cell).strip(" |")


def parse(html: str) -> list[list[str]]:
    body = html.split("<tbody>", 1)[-1].split("</tbody>", 1)[0]
    cells = [clean(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)]
    return [cells[i:i + len(COLS)] for i in range(0, len(cells) - len(COLS) + 1, len(COLS))]


def day(d: date) -> pd.DataFrame:
    rows, p = [], 1
    while True:
        r = parse(fetch(URL.format(p=p, d=d.isoformat())))
        rows += r
        if len(r) < 20:
            break
        p += 1
        time.sleep(DELAY_S)
    return pd.DataFrame(rows, columns=COLS).assign(query_date=d.isoformat())


def main() -> None:
    start, end = (date.fromisoformat(a) for a in sys.argv[1:3])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = start
    while d <= end:
        f = OUT_DIR / f"{d.isoformat()}.csv"
        if not f.exists():
            df = day(d)
            df.to_csv(f, index=False)
            print(f"{d} {len(df)} acts", flush=True)
            time.sleep(DELAY_S)
        d += timedelta(days=1)
    allf = sorted(OUT_DIR.glob("*.csv"))
    out = OUT_DIR.parent / f"busanpilot_log_{start}_{end}.csv"
    pd.concat([pd.read_csv(f, dtype=str) for f in allf
               if start.isoformat() <= f.stem <= end.isoformat()]).to_csv(out, index=False)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
