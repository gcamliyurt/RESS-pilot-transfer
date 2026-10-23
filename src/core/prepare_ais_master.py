from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "ais_master"

DYNAMIC_DIR = RAW_DIR / "ais_dynamic"
STATIC_FILE = RAW_DIR / "ais_static" / "Static_merged_bbox_result_gokhan.csv"

MASTER_PARQUET = OUT_DIR / "ais_master_latest.parquet"
MASTER_CSV = OUT_DIR / "ais_master_latest.csv"
MASTER_REPORT = OUT_DIR / "ais_master_latest_report.json"


def _norm_mmsi(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )
    return s.where(s.str.len().between(7, 9), pd.NA)


def _to_float32(series: pd.Series, decimals: int | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if decimals is not None:
        out = out.round(decimals)
    return out.astype("float32")


def _to_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int16")


def load_static(static_file: Path) -> pd.DataFrame:
    static = None
    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]:
        try:
            static = pd.read_csv(static_file, encoding=enc)
            break
        except Exception:
            continue
    if static is None:
        raise ValueError(f"Unable to read static AIS file: {static_file}")

    static = static.rename(
        columns={
            "MMSI": "mmsi",
            "mmsi": "mmsi",
            "선박명": "ship_name",
            "ship_name": "ship_name",
            "선종코드": "ship_type_code",
            "ship_type_code": "ship_type_code",
            "IMO": "imo",
            "imo": "imo",
            "호출부호": "call_sign",
            "call_sign": "call_sign",
            "DimA": "dim_a",
            "dim_a": "dim_a",
            "DimB": "dim_b",
            "dim_b": "dim_b",
            "DimC": "dim_c",
            "dim_c": "dim_c",
            "DimD": "dim_d",
            "dim_d": "dim_d",
            "흘수": "draft_m",
            "draft_m": "draft_m",
            "추정톤수": "estimated_tonnage",
            "estimated_tonnage": "estimated_tonnage",
        }
    )

    static["mmsi"] = _norm_mmsi(static["mmsi"])
    for col in ["dim_a", "dim_b", "dim_c", "dim_d", "draft_m"]:
        if col in static.columns:
            static[col] = _to_float32(static[col], decimals=2)
    if "estimated_tonnage" in static.columns:
        static["estimated_tonnage"] = pd.to_numeric(static["estimated_tonnage"], errors="coerce").round().astype(
            "Int32"
        )
    static = static.dropna(subset=["mmsi"]).drop_duplicates(subset=["mmsi"])
    return static


def load_dynamic_files(files: list[Path]) -> pd.DataFrame:
    if not files:
        raise ValueError("No dynamic AIS files to load")

    chunks = []
    cols = ["mmsi", "datetime", "lat", "lon", "spd", "crs", "hdg"]

    for f in files:
        df = pd.read_csv(f, usecols=cols)
        df["source_month"] = f.stem
        chunks.append(df)

    dynamic = pd.concat(chunks, ignore_index=True)
    dynamic["mmsi"] = _norm_mmsi(dynamic["mmsi"])
    dynamic = dynamic.dropna(subset=["mmsi"])

    dynamic["datetime"] = pd.to_datetime(dynamic["datetime"], errors="coerce", utc=True)
    dynamic = dynamic.dropna(subset=["datetime", "lat", "lon"])

    lat = pd.to_numeric(dynamic["lat"], errors="coerce")
    lon = pd.to_numeric(dynamic["lon"], errors="coerce")

    # Keep coordinate precision for spatial detection, but still store compact rounded index helpers.
    dynamic["lat"] = lat.astype("float32")
    dynamic["lon"] = lon.astype("float32")
    dynamic["lat_r4"] = lat.round(4).astype("float32")
    dynamic["lon_r4"] = lon.round(4).astype("float32")
    dynamic["lat_r3"] = lat.round(3).astype("float32")
    dynamic["lon_r3"] = lon.round(3).astype("float32")

    # Motion fields do not need decimal precision for downstream screening.
    for col in ["spd", "crs", "hdg"]:
        dynamic[col] = _to_nullable_int(dynamic[col])

    dynamic["source_month"] = dynamic["source_month"].astype(str)

    return dynamic


def load_dynamic(dynamic_dir: Path) -> pd.DataFrame:
    files = sorted(dynamic_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No dynamic AIS files found in {dynamic_dir}")
    return load_dynamic_files(files)


def build_master(dynamic: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    master = dynamic.merge(static, on="mmsi", how="left", validate="many_to_one")

    # quick flags for coverage
    master["has_static"] = master["ship_name"].notna()
    return master


def _find_existing_master(out_dir: Path) -> Path | None:
    if MASTER_CSV.exists():
        return MASTER_CSV
    if MASTER_PARQUET.exists():
        return MASTER_PARQUET

    candidates = sorted(out_dir.glob("ais_master_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates_csv = sorted(out_dir.glob("ais_master_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates_csv:
        return candidates_csv[0]
    if candidates:
        return candidates[0]
    return None


def load_existing_master(out_dir: Path) -> pd.DataFrame:
    existing = _find_existing_master(out_dir)
    if existing is None:
        return pd.DataFrame()

    if existing.suffix == ".parquet":
        try:
            master = pd.read_parquet(existing)
        except Exception:
            csv_fallback = existing.with_suffix(".csv")
            if not csv_fallback.exists():
                raise
            master = pd.read_csv(csv_fallback)
    else:
        master = pd.read_csv(existing)

    if "datetime" in master.columns:
        master["datetime"] = pd.to_datetime(master["datetime"], errors="coerce", utc=True)

    if "source_month" in master.columns:
        master["source_month"] = master["source_month"].astype(str)

    return master


def get_processed_months(existing_master: pd.DataFrame) -> set[str]:
    if existing_master.empty or "source_month" not in existing_master.columns:
        return set()
    return set(existing_master["source_month"].dropna().astype(str).unique().tolist())


def get_new_dynamic_files(dynamic_dir: Path, processed_months: set[str]) -> list[Path]:
    files = sorted(dynamic_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No dynamic AIS files found in {dynamic_dir}")
    return [f for f in files if f.stem not in processed_months]


def combine_existing_and_new(existing_master: pd.DataFrame, new_master: pd.DataFrame) -> pd.DataFrame:
    if existing_master.empty:
        combined = new_master.copy()
    else:
        combined = pd.concat([existing_master, new_master], ignore_index=True)

    dedup_keys = [k for k in ["mmsi", "datetime", "lat", "lon", "source_month"] if k in combined.columns]
    if dedup_keys:
        combined = combined.drop_duplicates(subset=dedup_keys, keep="first")

    if "datetime" in combined.columns:
        combined = combined.sort_values(["datetime", "mmsi"], ascending=[True, True], kind="mergesort")

    return combined.reset_index(drop=True)


def save_outputs(master: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    months = sorted(master["source_month"].dropna().astype(str).unique().tolist())
    month_tag = f"{months[0]}_{months[-1]}" if months else "unknown"

    parquet_path = out_dir / f"ais_master_{month_tag}.parquet"
    csv_path = out_dir / f"ais_master_{month_tag}.csv"
    report_path = out_dir / f"ais_master_{month_tag}_report.json"

    saved_parquet = False
    try:
        master.to_parquet(parquet_path, index=False)
        saved_parquet = True
    except Exception:
        pass

    # CSV fallback and audit copy
    master.to_csv(csv_path, index=False)

    report = {
        "rows": int(len(master)),
        "unique_mmsi": int(master["mmsi"].nunique()),
        "months": months,
        "static_match_rate": float(master["has_static"].mean()) if len(master) else 0.0,
        "precision_policy": {
            "lat_lon": "preserved for spatial accuracy",
            "lat_lon_index_helpers": "rounded to 3-4 decimals",
            "spd_crs_hdg": "rounded to nearest integer",
            "dimensions_draft": "rounded to 2 decimals",
            "estimated_tonnage": "rounded to nearest integer",
        },
        "saved_parquet": saved_parquet,
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # also store canonical latest artifacts for incremental workflows
    try:
        master.to_parquet(MASTER_PARQUET, index=False)
        report["latest_parquet_path"] = str(MASTER_PARQUET)
    except Exception:
        report["latest_parquet_path"] = None

    master.to_csv(MASTER_CSV, index=False)
    report["latest_csv_path"] = str(MASTER_CSV)

    with open(MASTER_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    static = load_static(STATIC_FILE)
    existing_master = load_existing_master(OUT_DIR)
    processed_months = get_processed_months(existing_master)

    new_files = get_new_dynamic_files(DYNAMIC_DIR, processed_months)
    if not new_files and not existing_master.empty:
        save_outputs(existing_master, OUT_DIR)
        print("No new dynamic AIS month found. Existing master is up to date.")
        print(f"Processed months: {sorted(processed_months)}")
        return

    dynamic = load_dynamic_files(new_files) if new_files else pd.DataFrame()
    new_master = build_master(dynamic, static) if not dynamic.empty else pd.DataFrame()
    master = combine_existing_and_new(existing_master, new_master)

    report = save_outputs(master, OUT_DIR)

    action = "AIS master appended" if not existing_master.empty else "AIS master created"
    print(action)
    print("New months:", [p.stem for p in new_files])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
