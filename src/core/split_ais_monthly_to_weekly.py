from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "ais_dynamic"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ais_master" / "weeks"
DEFAULT_TIMESTAMP_COL = "datetime"


@dataclass
class WeeklyFileStats:
    rows: int = 0
    min_datetime: str | None = None
    max_datetime: str | None = None

    def update(self, timestamps: pd.Series) -> None:
        valid = timestamps.dropna()
        if valid.empty:
            return

        current_min = valid.min()
        current_max = valid.max()

        self.rows += int(len(valid))

        min_iso = current_min.isoformat()
        max_iso = current_max.isoformat()

        if self.min_datetime is None or min_iso < self.min_datetime:
            self.min_datetime = min_iso
        if self.max_datetime is None or max_iso > self.max_datetime:
            self.max_datetime = max_iso


@dataclass
class MonthlySplitReport:
    source_file: str
    month: str
    output_dir: str
    timestamp_column: str
    chunk_size: int
    total_rows_read: int = 0
    rows_written: int = 0
    rows_with_invalid_timestamp: int = 0
    weekly_files: dict[str, WeeklyFileStats] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "month": self.month,
            "output_dir": self.output_dir,
            "timestamp_column": self.timestamp_column,
            "chunk_size": self.chunk_size,
            "total_rows_read": self.total_rows_read,
            "rows_written": self.rows_written,
            "rows_with_invalid_timestamp": self.rows_with_invalid_timestamp,
            "weekly_files": {
                name: {
                    "rows": stats.rows,
                    "min_datetime": stats.min_datetime,
                    "max_datetime": stats.max_datetime,
                }
                for name, stats in sorted(self.weekly_files.items())
            },
        }


def _resolve_input_files(files: list[str] | None, input_dir: Path) -> list[Path]:
    if files:
        resolved = [Path(f).expanduser().resolve() for f in files]
    else:
        resolved = sorted(input_dir.glob("*.csv"))

    if not resolved:
        raise FileNotFoundError(f"No monthly AIS CSV files found in {input_dir}")

    return resolved


def _extract_month_tag(path: Path) -> str:
    stem = path.stem
    if len(stem) == 7 and stem[4] == "-" and stem[:4].isdigit() and stem[5:7].isdigit():
        return stem
    raise ValueError(
        f"Expected a monthly filename like YYYY-MM.csv, but got: {path.name}"
    )


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _remove_existing_week_files(output_dir: Path, month_tag: str) -> None:
    for path in output_dir.glob(f"{month_tag}-week*.csv"):
        path.unlink()
    report_path = output_dir / f"{month_tag}-weekly-report.json"
    if report_path.exists():
        report_path.unlink()


def _week_index_from_timestamp(ts: pd.Series) -> pd.Series:
    return ((ts.dt.day - 1) // 7 + 1).astype("Int64")


def split_monthly_file(
    file_path: Path,
    output_dir: Path,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    chunk_size: int = 500_000,
    overwrite: bool = True,
) -> MonthlySplitReport:
    month_tag = _extract_month_tag(file_path)
    _prepare_output_dir(output_dir)

    if overwrite:
        _remove_existing_week_files(output_dir, month_tag)

    report = MonthlySplitReport(
        source_file=str(file_path),
        month=month_tag,
        output_dir=str(output_dir),
        timestamp_column=timestamp_col,
        chunk_size=chunk_size,
    )
    header_written: set[Path] = set()

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        if timestamp_col not in chunk.columns:
            raise KeyError(f"Timestamp column '{timestamp_col}' was not found in {file_path}")

        report.total_rows_read += int(len(chunk))

        chunk[timestamp_col] = pd.to_datetime(chunk[timestamp_col], errors="coerce", format="mixed")
        invalid_mask = chunk[timestamp_col].isna()
        report.rows_with_invalid_timestamp += int(invalid_mask.sum())

        valid_chunk = chunk.loc[~invalid_mask].copy()
        if valid_chunk.empty:
            continue

        valid_chunk["_week_index"] = _week_index_from_timestamp(valid_chunk[timestamp_col])
        valid_chunk = valid_chunk.loc[valid_chunk["_week_index"].notna()].copy()

        for week_idx, week_df in valid_chunk.groupby("_week_index", sort=True):
            week_number = int(week_idx)
            out_path = output_dir / f"{month_tag}-week{week_number}.csv"
            cleaned = week_df.drop(columns="_week_index")
            cleaned.to_csv(
                out_path,
                index=False,
                mode="a",
                header=out_path not in header_written and not out_path.exists(),
            )
            header_written.add(out_path)

            file_key = out_path.name
            if file_key not in report.weekly_files:
                report.weekly_files[file_key] = WeeklyFileStats()
            report.weekly_files[file_key].update(cleaned[timestamp_col])
            report.rows_written += int(len(cleaned))

    report_path = output_dir / f"{month_tag}-weekly-report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report.as_dict(), fh, indent=2)

    return report


def build_manifest(reports: list[MonthlySplitReport], output_dir: Path) -> Path:
    manifest = {
        "months_processed": [report.month for report in reports],
        "output_dir": str(output_dir),
        "reports": [report.as_dict() for report in reports],
    }
    manifest_path = output_dir / "weekly_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split monthly AIS CSV files into week1-week5 CSV files using the datetime column."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing monthly AIS CSV files named YYYY-MM.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where weekly CSV files will be written",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit monthly CSV files to split instead of scanning --input-dir",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default=DEFAULT_TIMESTAMP_COL,
        help="Timestamp column used to assign rows to weeks",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Rows per chunk while reading the monthly CSV",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Append to existing weekly outputs instead of replacing matching month files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    files = _resolve_input_files(args.files, input_dir)

    reports: list[MonthlySplitReport] = []
    for file_path in files:
        report = split_monthly_file(
            file_path=file_path,
            output_dir=output_dir,
            timestamp_col=args.timestamp_col,
            chunk_size=args.chunk_size,
            overwrite=not args.no_overwrite,
        )
        reports.append(report)

        print(
            f"{report.month}: wrote {report.rows_written:,} rows to "
            f"{len(report.weekly_files)} weekly files "
            f"({report.rows_with_invalid_timestamp:,} rows skipped with invalid {report.timestamp_column})"
        )
        for weekly_name, weekly_stats in sorted(report.weekly_files.items()):
            print(
                f"  - {weekly_name}: {weekly_stats.rows:,} rows "
                f"[{weekly_stats.min_datetime} -> {weekly_stats.max_datetime}]"
            )

    manifest_path = build_manifest(reports, output_dir)
    print(f"Saved weekly manifest: {manifest_path}")


if __name__ == "__main__":
    main()
