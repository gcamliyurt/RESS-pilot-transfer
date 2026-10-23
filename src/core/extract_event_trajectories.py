from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from src.core.detect_interceptions import _pick_master_path, load_master


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "interactions"


def load_events(events_path: Path) -> pd.DataFrame:
    try:
        events = pd.read_csv(events_path)
    except EmptyDataError:
        events = pd.DataFrame(columns=["event_id", "pilot_mmsi", "vessel_mmsi", "start_time", "end_time"])
    for col in ["start_time", "end_time"]:
        if col not in events.columns:
            events[col] = pd.Series(dtype="datetime64[ns, UTC]")
        events[col] = pd.to_datetime(events[col], errors="coerce", utc=True)
    return events


def _serialize_track(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for rec in df.sort_values("datetime", kind="mergesort").to_dict(orient="records"):
        rows.append(
            {
                "timestamp": pd.Timestamp(rec["datetime"]).isoformat() if pd.notna(rec["datetime"]) else None,
                "lat": float(rec["lat"]) if pd.notna(rec.get("lat")) else None,
                "lon": float(rec["lon"]) if pd.notna(rec.get("lon")) else None,
                "spd": float(rec["spd"]) if pd.notna(rec.get("spd")) else None,
                "crs": float(rec["crs"]) if pd.notna(rec.get("crs")) else None,
                "hdg": float(rec["hdg"]) if pd.notna(rec.get("hdg")) else None,
            }
        )
    return rows


def extract_event_trajectories(
    master: pd.DataFrame,
    events: pd.DataFrame,
    pre_buffer_min: int = 15,
    post_buffer_min: int = 15,
) -> dict[str, dict]:
    master = master.copy()
    master["mmsi"] = master["mmsi"].astype(str)

    payload: dict[str, dict] = {}
    for _, event in events.iterrows():
        start = event["start_time"] - pd.Timedelta(minutes=pre_buffer_min)
        end = event["end_time"] + pd.Timedelta(minutes=post_buffer_min)

        pilot_track = master[
            (master["mmsi"] == str(event["pilot_mmsi"]))
            & (master["datetime"] >= start)
            & (master["datetime"] <= end)
        ].copy()
        vessel_track = master[
            (master["mmsi"] == str(event["vessel_mmsi"]))
            & (master["datetime"] >= start)
            & (master["datetime"] <= end)
        ].copy()

        transfer_start = event["start_time"]
        transfer_end = event["end_time"]
        midpoint = transfer_start + (transfer_end - transfer_start) / 2

        payload[str(event["event_id"])] = {
            "event_id": str(event["event_id"]),
            "pilot_mmsi": str(event["pilot_mmsi"]),
            "vessel_mmsi": str(event["vessel_mmsi"]),
            "start_time": pd.Timestamp(event["start_time"]).isoformat(),
            "end_time": pd.Timestamp(event["end_time"]).isoformat(),
            "pre_buffer_min": pre_buffer_min,
            "post_buffer_min": post_buffer_min,
            "pilot_track": _serialize_track(pilot_track),
            "vessel_track": _serialize_track(vessel_track),
            "approach_phase": {
                "start_time": start.isoformat(),
                "end_time": midpoint.isoformat(),
            },
            "transfer_phase": {
                "start_time": transfer_start.isoformat(),
                "end_time": transfer_end.isoformat(),
            },
            "separation_phase": {
                "start_time": midpoint.isoformat(),
                "end_time": end.isoformat(),
            },
        }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract buffered pilot/vessel tracks for detected events")
    parser.add_argument("--events", type=str, required=True, help="Detected interception CSV path")
    parser.add_argument("--master", type=str, default=None, help="AIS master parquet/csv")
    parser.add_argument("--pre-buffer-min", type=int, default=15)
    parser.add_argument("--post-buffer-min", type=int, default=15)
    parser.add_argument("--tag", type=str, default="latest")
    args = parser.parse_args()

    events_path = Path(args.events)
    events = load_events(events_path)
    start_date = events["start_time"].min().isoformat() if events["start_time"].notna().any() else None
    end_date = events["end_time"].max().isoformat() if events["end_time"].notna().any() else None
    master_file = _pick_master_path(args.master, start_date=start_date, end_date=end_date)
    payload: dict[str, dict]
    if events.empty:
        payload = {}
    else:
        master = load_master(master_file)
        payload = extract_event_trajectories(
            master,
            events,
            pre_buffer_min=args.pre_buffer_min,
            post_buffer_min=args.post_buffer_min,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"event_trajectories_{args.tag}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved event trajectories: {out_path}")


if __name__ == "__main__":
    main()
