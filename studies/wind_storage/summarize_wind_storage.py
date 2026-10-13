"""Print publication tables from the verified three-arm seasonal storage experiment.

Run after ``measure_wind_storage.py report``. Percentages compare pooled Parquet file bytes;
gap statistics weight each native forecast row equally, rather than each hour of exposure.
"""

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl
from measure_wind_storage import (
    HEIGHTS,
    REPRESENTATIONS,
    _identity,
    _paired_records,
    _resolved_dates,
)


def summarize(*, output: Path, manifest: Path, grid: Path) -> None:
    """Write storage contrasts and initialization-season scalar-gap summaries."""
    requested = json.loads(manifest.read_text())["dates"]
    identity = _identity(manifest=manifest, grid=grid)
    dates = _resolved_dates(output=output, requested=requested, identity=identity)
    records = _paired_records(output=output, dates=dates, identity=identity)
    if json.loads((output / "measurements.json").read_text()) != records:
        raise ValueError("Run the full measurement report before summarizing")
    total_rows = sum(r["rows"] for r in records if r["arm"] == "polar")
    lines = [
        "# Wind storage: seasonal sample",
        "",
        f"Sample: {len(dates)} full 00Z runs, {total_rows:,} rows per arm.",
        "",
        "Dates: " + ", ".join(dates) + ".",
        "",
        "Parquet file bytes include metadata. Delta-log bytes are separate. MB means 10⁶ bytes.",
        "",
        (
            "| Run | Polar MB | Components MB | Components + scalar MB | "
            "Components vs polar | Scalar vs components |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for date in [*dates, "Pooled"]:
        sizes = {
            arm: sum(
                r["file_bytes"]
                for r in records
                if r["arm"] == arm and (date == "Pooled" or r["date"] == date)
            )
            for arm in REPRESENTATIONS
        }
        lines.append(
            f"| {date} | {sizes['polar'] / 1e6:.3f} | {sizes['components'] / 1e6:.3f} | "
            f"{sizes['scalar'] / 1e6:.3f} | "
            f"{100 * (sizes['components'] / sizes['polar'] - 1):+.2f}% | "
            f"{100 * (sizes['scalar'] / sizes['components'] - 1):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "| Initialization season | Components vs polar | Scalar vs components |",
            "|---|---:|---:|",
        ]
    )
    for season, label in enumerate(("DJF", "MAM", "JJA", "SON")):
        sizes = {
            arm: sum(
                r["file_bytes"]
                for r in records
                if r["arm"] == arm and int(r["date"][5:7]) % 12 // 3 == season
            )
            for arm in REPRESENTATIONS
        }
        lines.append(
            f"| {label} | {100 * (sizes['components'] / sizes['polar'] - 1):+.2f}% | "
            f"{100 * (sizes['scalar'] / sizes['components'] - 1):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "Scalar minus vector speed measures spatial cancellation, not turbine-power loss.",
            "Native forecast rows have equal weight, including both 3-hour and 6-hour steps.",
            "Seasons group runs by initialization month; forecasts may cross season boundaries.",
            "Relative gaps divide by vector speed and can grow very large near calm.",
            "Float32 reduction can give negative gaps of a few millionths of a metre per second.",
            "",
            (
                "| Initialization season | Height | Median gap m/s | Mean gap m/s | p90 gap m/s | "
                "p99 gap m/s | Max gap m/s | Median relative gap | p99 relative gap |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    gaps = pl.concat(
        [
            pl.scan_parquet(output / "scalar" / f"{date}-gaps.parquet").with_columns(
                season=pl.lit(int(date[5:7]) % 12 // 3)
            )
            for date in dates
        ]
    )
    for season, label in [(None, "All"), *enumerate(("DJF", "MAM", "JJA", "SON"))]:
        selected = gaps if season is None else gaps.filter(pl.col("season") == season)
        for height in HEIGHTS:
            gap, relative = pl.col(f"gap_{height}"), pl.col(f"relative_gap_{height}")
            values = selected.select(
                median=gap.median(),
                mean=gap.mean(),
                p90=gap.quantile(0.9),
                p99=gap.quantile(0.99),
                maximum=gap.max(),
                relative_median=relative.median(),
                relative_p99=relative.quantile(0.99),
            ).collect()
            lines.append(
                f"| {label} | {height} | " + " | ".join(f"{v:.6f}" for v in values.row(0)) + " |"
            )
    lines.extend(
        [
            "",
            "| Arm | Delta-log bytes |",
            "|---|---:|",
            *(
                f"| {arm} | {sum(r['delta_log_bytes'] for r in records if r['arm'] == arm)} |"
                for arm in REPRESENTATIONS
            ),
            "",
            "Full per-column compression, calm counts, cell summaries, raw digests, source hashes,",
            "dependency versions and availability inventory are in the sibling report and records.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(summary)
    provenance = {
        **identity,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records_sha256": hashlib.sha256((output / "measurements.json").read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
    }
    (output / "summary-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    """Summarize only the complete paired sample, never a partial pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    args = parser.parse_args()
    summarize(output=args.output, manifest=args.manifest, grid=args.grid)


if __name__ == "__main__":
    main()
