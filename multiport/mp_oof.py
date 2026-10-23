"""Multi-port: ship-grouped out-of-fold predictions of the hybrid model, the anchors of the
multi-port optimiser. Same rule as Busan (revision_r2_validation.oof_predictions): predicted
distance from the pilot base, and an anchor bearing equal to the circular mean of the meeting
bearings (from the base) of the 25 training transfers whose ships had the most similar bearing
from the base at decision time, training fold only.
Output: data/processed/multiport/oof_<port>.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import mp_validation as mv

K_NN = 25


def circ_mean(deg) -> float:
    r = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def base_of(port: str) -> tuple[float, float]:
    b = pd.read_csv(mv.LEE / "processed/multiport/pilot_bases.csv").set_index("port").loc[port]
    return float(b.base_lat), float(b.base_lon)


def main(ports: list[str]) -> None:
    d = mv.load()
    for port in ports:
        b_lat, b_lon = base_of(port)
        parts = []
        for direc, sub in d[d.port == port].groupby("traffic_direction"):
            sub = sub.reset_index(drop=True)
            k = np.cos(np.radians(b_lat))
            sub["meet_brg"] = np.degrees(np.arctan2((sub.meeting_lon - b_lon) * k, sub.meeting_lat - b_lat)) % 360
            sub["ship_brg"] = np.degrees(np.arctan2(sub.ref_bearing_sin, sub.ref_bearing_cos)) % 360
            for fold, (tr_i, te_i) in enumerate(GroupKFold(5).split(sub, groups=sub.vessel_mmsi.astype(str))):
                tr, te = mv.impute(sub.iloc[tr_i], sub.iloc[te_i])
                diff = np.abs(((te.ship_brg.to_numpy()[:, None] - tr.ship_brg.to_numpy()[None, :]) + 180) % 360 - 180)
                nn = np.argsort(diff, axis=1)[:, :K_NN]
                tcol = "t_dispatch" if mv.MP_SET == "dispatch" else "t_dec"
                lcol = "dispatch_to_meeting_min" if mv.MP_SET == "dispatch" else "lead_min"
                keep = ["event_id", "traffic_direction", tcol, lcol] + (["ship_visible"] if mv.MP_SET == "dispatch" else [])
                parts.append(te[keep].rename(columns={tcol: "t_dec", lcol: "lead_min"}).assign(fold=fold,
                    pred_dist_m=mv.fit_predict("B6_hybrid", tr, te),
                    anchor_bearing_deg=[circ_mean(tr.meet_brg.to_numpy()[r]) for r in nn]))
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(mv.FEAT_DIR / (f"oof_dispatch_{port}.csv" if mv.MP_SET == "dispatch" else f"oof_{port}.csv"), index=False)
        print(port, len(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(mv.US_PBA))
