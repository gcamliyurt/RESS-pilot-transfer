"""RESS R2: build the log-confirmed modelling dataset.

Inputs : *_r2verified enriched events + optimised meeting points (13-boat fleet,
         KST-consistent tide/night flag), Busan Harbor Pilots' Association log (PS-3).
Rule   : an AIS session is kept when the same ship (call sign -> MMSI, else name)
         has a logged New Port pilotage act within +/-2 h of the meeting time.
         Direction is taken from the log. AIS times are KST wall-clock.
Output : data/processed/r2/modelling_r2.csv, detection_validation_r2.json
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "processed" / "r2"
OUT.mkdir(parents=True, exist_ok=True)
TOL = pd.Timedelta(hours=2)


def norm(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def load_log() -> pd.DataFrame:
    L = pd.read_csv(ROOT / "data" / "external" / "busanpilot_log_newport_PS3.csv")
    L["t_log"] = pd.to_datetime(L["t"])
    st = pd.read_csv(ROOT.parent / "Static_merged_bbox_result_gokhan.csv", encoding="cp949")
    st.columns = ["mmsi", "name", "type", "imo", "call", "A", "B", "C", "D", "draft", "gt"]
    by_cs = st.assign(k=st.call.map(norm)).drop_duplicates("k").set_index("k").mmsi
    by_nm = st.assign(k=st.name.map(norm)).drop_duplicates("k").set_index("k").mmsi
    cs = L.ship.str.split("|").str[1].map(norm)
    L["vessel_mmsi"] = cs.map(by_cs).fillna(L["name"].map(by_nm))
    L["log_act"] = range(len(L))
    return L.dropna(subset=["vessel_mmsi"]).astype({"vessel_mmsi": "int64"})


def main() -> None:
    E = pd.concat([pd.read_csv(f, low_memory=False)
                   for f in sorted(glob.glob(str(ROOT / "data/processed/interactions/event_context_enriched_*_r2verified.csv")))],
                  ignore_index=True)
    O = pd.concat([pd.read_csv(f, low_memory=False, usecols=["event_id"])
                   for f in sorted(glob.glob(str(ROOT / "data/processed/meeting_point/optimized_meeting_points_*_r2verified.csv")))],
                  ignore_index=True)
    E["vessel_mmsi"] = pd.to_numeric(E["vessel_mmsi"], errors="coerce").astype("Int64")
    E["t_ais"] = pd.to_datetime(E["meeting_time"].astype(str).str[:19])  # KST wall-clock
    L = load_log()

    m = E[["event_id", "vessel_mmsi", "t_ais", "traffic_direction"]].merge(
        L[["vessel_mmsi", "t_log", "dir", "log_act"]], on="vessel_mmsi")
    m = m[(m.t_ais - m.t_log).abs() <= TOL]
    m["dt"] = (m.t_ais - m.t_log).abs()
    m = m.sort_values("dt").drop_duplicates("event_id").drop_duplicates("log_act")

    keep = E.merge(m[["event_id", "dir", "log_act"]], on="event_id")
    keep["ais_direction"] = keep["traffic_direction"]
    keep["traffic_direction"] = keep["dir"]
    keep["optimised"] = keep.event_id.isin(set(O.event_id))
    keep.drop(columns=["dir"]).to_csv(OUT / "modelling_r2.csv", index=False)

    rep = {
        "ais_sessions_detected": int(len(E)),
        "log_acts_newport": int(len(L)),
        "log_confirmed_sessions": int(len(keep)),
        "precision": round(len(keep) / len(E), 4),
        "recall_inbound": round((m.dir == "inbound").sum() / (L.dir == "inbound").sum(), 4),
        "recall_outbound": round((m.dir == "outbound").sum() / (L.dir == "outbound").sum(), 4),
        "ais_direction_agrees_with_log": round(float((keep.ais_direction == keep.traffic_direction).mean()), 4),
        "confirmed_by_direction": keep.traffic_direction.value_counts().to_dict(),
        "confirmed_and_optimised": int(keep.optimised.sum()),
    }
    (OUT / "detection_validation_r2.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
