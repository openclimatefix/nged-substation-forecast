"""Draw descriptive point-control gain shares from an immutable public importance report.

Export text first with --text-only. Render only after the exact figure text has been reviewed,
using --reviewed-text. Charts convert reported shares to percentages and estimate no statistics.
"""

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
import vl_convert as vlc

from studies import charts

LABELS: Final[dict[str, str]] = {
    "categorical_precipitation_type_surface": "Precipitation type",
    "dew_point_temperature_2m": "Dew-point temperature",
    "downward_short_wave_radiation_flux_surface": "Downward short-wave radiation",
    "geopotential_height_500hpa": "500 hPa geopotential height",
    "local_day_of_week": "Weekday ordinal",
    "local_day_of_week_cos": "Weekday cosine",
    "local_day_of_week_sin": "Weekday sine",
    "local_time_of_day_cos": "Time-of-day cosine",
    "local_time_of_day_sin": "Time-of-day sine",
    "local_time_of_year_cos": "Time-of-year cosine",
    "local_time_of_year_sin": "Time-of-year sine",
    "local_utc_offset_minutes": "Local UTC offset",
    "power_lag_168h": "Power lag: 1 week",
    "power_lag_24h": "Power lag: 24 hours",
    "power_lag_336h": "Power lag: 2 weeks",
    "power_lag_48h": "Power lag: 48 hours",
    "pressure_reduced_to_mean_sea_level": "Mean sea-level pressure",
    "pressure_surface": "Surface pressure",
    "temperature_2m": "Air temperature at 2 m",
    "wind_direction_100m": "Wind direction at 100 m",
    "wind_direction_10m": "Wind direction at 10 m",
    "wind_speed_100m": "Wind speed at 100 m",
    "wind_speed_10m": "Wind speed at 10 m",
    "windchill": "Windchill",
}
PANELS: Final[dict[str, str]] = {"PV": "Solar farms", "Wind": "Wind farms"}
TITLE: Final[str] = "Training gain describes splits; feature tests decide usefulness"
SUBTITLE: Final[tuple[str, ...]] = (
    "Combined pipeline; full-feature point control; two folds and three seeds.",
    "Blue dot: mean gain share. Blue line: min/max across six fold/seed refits.",
    "Equal weight per series within each refit, then per refit; unsplit features have zero.",
    "Ranges describe variability across refits, not confidence intervals.",
    "Gain divides credit between correlated features and does not establish benefit.",
)
AXIS_TITLE: Final[str] = "Share of training gain (%; larger is more gain, not more benefit)"
REPORT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "series_type",
        "feature",
        "mean_share",
        "min_share",
        "max_share",
        "refits",
        "min_series_per_refit",
        "max_series_per_refit",
    }
)


def _sha256(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_report(*, root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["recipe"] != "production_cleaned"
        or "importance.parquet" not in manifest["outputs"]
    ):
        raise ValueError("chart requires a complete point-control importance report")
    for name, digest in manifest["outputs"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or _sha256(path=path) != digest:
            raise ValueError("importance output path or fingerprint mismatch")
    support = manifest["refits"]
    if (
        len(support) != 6
        or len({(run["fold"], run["seed"]) for run in support}) != 6
        or {run["seed"] for run in support} != {0, 1, 2}
        or len({run["fold"] for run in support}) != 2
        or len({run["device"] for run in support}) != 1
    ):
        raise ValueError("importance report requires both folds and three fitting seeds")
    return manifest


def _read_table(*, root: Path) -> pl.DataFrame:
    table = pl.read_parquet(root / "importance.parquet")
    if set(table.columns) != REPORT_COLUMNS:
        raise ValueError("importance table differs from the approved public aggregate schema")
    selected = table.filter(pl.col("series_type").is_in(list(PANELS)))
    if selected.null_count().sum_horizontal().item():
        raise ValueError("importance chart cells must be non-null")
    numeric = ["mean_share", "min_share", "max_share"]
    if selected.filter(
        pl.any_horizontal(~pl.col(*numeric).is_finite() | ~pl.col(*numeric).is_between(0, 1))
        | (pl.col("min_share") > pl.col("mean_share"))
        | (pl.col("mean_share") > pl.col("max_share"))
        | (pl.col("refits") != 6)
        | (pl.col("min_series_per_refit") <= 0)
        | (pl.col("max_series_per_refit") < pl.col("min_series_per_refit"))
    ).height:
        raise ValueError("importance ranges or support are invalid")
    for kind in PANELS:
        rows = selected.filter(pl.col("series_type") == kind)
        if rows.height != len(LABELS) or set(rows["feature"]) != set(LABELS):
            raise ValueError("each importance panel must contain all 24 unique production features")
    return selected.with_columns(
        label=pl.col("feature").replace_strict(LABELS),
        mean_percent=100 * pl.col("mean_share"),
        min_percent=100 * pl.col("min_share"),
        max_percent=100 * pl.col("max_share"),
    ).sort("series_type", "mean_share", "feature", descending=[False, True, False])


def _text(*, table: pl.DataFrame, number: str) -> str:
    lines = ["# Point-control importance figure text", "", f"Figure {number}: {TITLE}"]
    lines.extend([*SUBTITLE, ""])
    for kind, title in PANELS.items():
        rows = table.filter(pl.col("series_type") == kind)
        lines.extend([title, AXIS_TITLE, *rows["label"].to_list(), ""])
    lines.append("Horizontal axes are linear and start at zero; zero shares remain at zero.")
    return "\n".join(lines) + "\n"


def _panel(*, rows: pl.DataFrame, title: str) -> charts.Panel:
    labels = rows["label"].to_list()
    base = alt.Chart(rows).encode(
        y=alt.Y(
            "label:N",
            sort=labels,
            title=None,
            axis=alt.Axis(
                labelLimit=charts.LABEL_WIDTH_PX,
                minExtent=charts.LABEL_WIDTH_PX,
                maxExtent=charts.LABEL_WIDTH_PX,
                labelPadding=8,
                ticks=False,
                domain=False,
            ),
        ),
    )
    interval = cast(alt.Chart, base.mark_rule(color=ocf.DATA_BLUE, strokeWidth=2)).encode(
        x=alt.X("min_percent:Q", title=AXIS_TITLE, scale=alt.Scale(zero=True)),
        x2=alt.X2("max_percent:Q"),
    )
    point = cast(alt.Chart, base.mark_point(color=ocf.DATA_BLUE, filled=True, size=55)).encode(
        x=alt.X("mean_percent:Q", title=AXIS_TITLE, scale=alt.Scale(zero=True)),
    )
    return alt.layer(interval, point).properties(
        width=charts.PLOT_WIDTH_PX,
        height=alt.Step(23),
        title=alt.TitleParams(title, anchor="start", fontSize=15),
    )


def main() -> None:
    """Export all figure text or render a verified public report after exact text review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--number", default="6")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--reviewed-text", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("chart output exists; use a new immutable directory")
    root = args.report.resolve()
    report_sha = _sha256(path=root / "manifest.json")
    manifest = _verify_report(root=root)
    table = _read_table(root=root)
    text = _text(table=table, number=args.number)
    if not args.text_only and (
        args.reviewed_text is None or args.reviewed_text.read_text() != text
    ):
        raise ValueError("rendering requires the exact previously reviewed figure text")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "figure-text.md").write_text(text)
    if not args.text_only:
        panels = [
            _panel(rows=table.filter(pl.col("series_type") == kind), title=title)
            for kind, title in PANELS.items()
        ]
        chart = charts.figure(
            panels=panels,
            number=args.number,
            title=TITLE,
            subtitle=[
                *SUBTITLE,
                "Horizontal axes are linear and start at zero; zero shares remain at zero.",
            ],
            figure_planning=None,
        )
        svg = args.output / "fixed_pipeline_importance.svg"
        chart.save(svg)
        subprocess.run(
            ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(svg)],
            check=True,
            capture_output=True,
        )
        (args.output / "fixed_pipeline_importance.png").write_bytes(
            vlc.svg_to_png(svg.read_text(), scale=2)
        )
    if _sha256(path=root / "manifest.json") != report_sha or _verify_report(root=root) != manifest:
        raise ValueError("importance report changed during chart export")
    configuration = {"report": str(root), "number": args.number, "text_only": args.text_only}
    provenance = {
        "script_sha256": _sha256(path=Path(__file__)),
        "report_manifest_sha256": report_sha,
        "report_output_hashes": manifest["outputs"],
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        "chart_library_sha256": _sha256(path=Path(charts.__file__)),
        "theme_sha256": _sha256(path=Path(ocf.__file__)),
        "palette": [ocf.DATA_BLUE],
        "feature_labels": LABELS,
        "reviewed_text_sha256": (
            _sha256(path=args.reviewed_text) if args.reviewed_text is not None else None
        ),
        "statistics_recomputed": False,
        "dependencies": {name: version(name) for name in ("altair", "polars", "vl-convert-python")},
        "outputs": {
            path.name: _sha256(path=path)
            for path in sorted(args.output.iterdir())
            if path.is_file()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
