"""Plot deterministic radiation references through the production feature pipeline.

Run from the repository root with `uv run python
studies/radiation_resampling/plot_radiation_resampling.py`. All inputs are synthetic. The reference
uses independent pvlib solar geometry; conservation is checked against the supplied Float32 means.
The synthetic comparison illustrates interval semantics, not forecast accuracy or recovery of
unknown cloud changes.
"""

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import patito as pt
import plotting.ocf_theme as ocf
import polars as pl
import pvlib
import vl_convert as vlc
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp, NwpModelId
from ml_core.features import TabularFeatureEngineer
from studies.charts import CONTENT_WIDTH_PX

SHORT_WAVE: Final[str] = "downward_short_wave_radiation_flux_surface"
LONG_WAVE: Final[str] = "downward_long_wave_radiation_flux_surface"
FIELDS: Final[tuple[str, str]] = (SHORT_WAVE, LONG_WAVE)
CURVES: Final[list[str]] = [
    "Synthetic reference",
    "Before: point interpolation",
    "After: interval-aware reconstruction",
]
PLOT_DAY: Final[datetime] = datetime(2026, 3, 21, tzinfo=UTC)


def reference_means(*, stamps: pl.Series, cloudy_afternoon: bool = False) -> pl.DataFrame:
    """Prescribe half-hour means at an invented coordinate, using two midpoint samples."""
    short_wave, long_wave = [], []
    for offset in (450, 1350):
        samples = stamps - timedelta(seconds=offset)
        position = pvlib.solarposition.get_solarposition(
            time=samples.to_numpy(), latitude=52.0, longitude=0.0
        )
        cosine = np.maximum(np.cos(np.deg2rad(position["zenith"].to_numpy())), 0)
        hour = (
            samples.dt.hour().to_numpy()
            + samples.dt.minute().to_numpy() / 60
            + samples.dt.second().to_numpy() / 3600
        )
        attenuation = np.where(hour < 12, 0.85, 0.2) if cloudy_afternoon else 0.7
        short_wave.append(attenuation * 1098 * cosine * np.exp(-0.059 / np.maximum(cosine, 1e-12)))
        long_wave.append(310 + 30 * np.cos(2 * np.pi * (hour - 15) / 24))
    return pl.DataFrame(
        {
            "valid_time": stamps,
            SHORT_WAVE: np.mean(short_wave, axis=0),
            LONG_WAVE: np.mean(long_wave, axis=0),
        }
    )


def reconstruct(*, truth: pl.DataFrame, width: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate identical native inputs and compare baseline with the public production path."""
    init = PLOT_DAY - timedelta(days=1 if width == 3 else 7)
    native = (
        truth.group_by_dynamic("valid_time", every=f"{width}h", closed="right", label="right")
        .agg(pl.col(*FIELDS).mean())
        .with_columns(
            init_time=pl.lit(init),
            h3_index=pl.lit(1, dtype=pl.Int64),
            ensemble_member=pl.lit(0),
            nwp_model_id=pl.lit(NwpModelId.ECMWF_ENS_0_25_degree.name),
        )
    )
    native = native.cast({column: Nwp.dtypes[column] for column in native.columns})
    metadata = pt.DataFrame(
        pl.DataFrame(
            {"time_series_id": [1], "h3_res_5": [1], "latitude": [52.0], "longitude": [0.0]}
        ).cast({"time_series_id": pl.Int32, "h3_res_5": pl.UInt64})
    ).set_model(TimeSeriesMetadata)
    power = pt.LazyFrame.from_existing(
        truth.select(time=pl.col("valid_time"))
        .with_columns(time_series_id=pl.lit(1, dtype=pl.Int32), power=pl.lit(0.0, dtype=pl.Float32))
        .lazy()
    ).set_model(PowerTimeSeries)
    after = (
        TabularFeatureEngineer()
        .engineer(
            selected_features=set(FIELDS),
            power_time_series=power,
            time_series_metadata=metadata,
            nwp=pt.LazyFrame.from_existing(native.lazy()).set_model(Nwp),
        )
        .collect()
        .select("valid_time", *FIELDS)
    )
    # This is the baseline algorithm: join period-ending means as points, then interpolate.
    before = (
        truth.select("valid_time")
        .join(native.select("valid_time", *FIELDS), on="valid_time", how="left")
        .sort("valid_time")
        .with_columns(pl.col(*FIELDS).interpolate())
    )
    shared_day = (pl.col("valid_time") > PLOT_DAY) & (
        pl.col("valid_time") <= PLOT_DAY + timedelta(days=1)
    )
    comparison = (
        truth.filter(shared_day)
        .join(before.rename({column: f"{column}_before" for column in FIELDS}), on="valid_time")
        .join(after.rename({column: f"{column}_after" for column in FIELDS}), on="valid_time")
        .sort("valid_time")
    )
    return comparison, native.filter(shared_day).sort("valid_time")


def main() -> None:
    """Write the figure and a report from the same complete-window comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/architecture/assets/radiation-resampling.svg")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("studies/radiation_resampling/report.md")
    )
    args = parser.parse_args()
    stamps = pl.datetime_range(
        PLOT_DAY - timedelta(days=1) + timedelta(minutes=30),
        PLOT_DAY + timedelta(days=2),
        interval="30m",
        eager=True,
    )
    panels = []
    report = [
        "# Synthetic radiation resampling check",
        "",
        "Invented coordinate: 52°N, 0°E. Reference day: 21 March 2026 (UTC).",
        "Before uses point interpolation; after uses production interval-aware reconstruction.",
        "Errors compare the average of reconstructed half-hours within each complete native window",
        "with that window's supplied Float32 mean.",
        "The energy centroid is the average half-hour midpoint weighted by radiation energy.",
        "Centroid shifts include all 48 half-hours and are relative to the synthetic reference.",
        "Positive shifts mean later. These checks do not measure forecast skill.",
        "The cloud-transition reference uses 85% of clear sky before noon and 20% afterwards.",
        "Short-wave conserves each native mean. Long-wave evaluates a piecewise cubic Hermite",
        (
            "interpolating polynomial (PCHIP) at half-hour midpoints "
            "without enforcing interval energy."
        ),
        "Neither method recovers unknown sub-window changes.",
        "",
        (
            "| Field / scenario | Native step (h) | Maximum mean error before (W/m²) | "
            "After (W/m²) | "
            "Centroid shift before (h) | After (h) |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for field, label, width, cloudy in (
        (SHORT_WAVE, "Short-wave, constant attenuation", 3, False),
        (SHORT_WAVE, "Short-wave, constant attenuation", 6, False),
        (SHORT_WAVE, "Short-wave, sunny morning (85%) / dull afternoon (20%)", 3, True),
        (LONG_WAVE, "Long-wave", 3, False),
        (LONG_WAVE, "Long-wave", 6, False),
    ):
        truth = reference_means(stamps=stamps, cloudy_afternoon=cloudy)
        comparison, native = reconstruct(truth=truth, width=width)
        assert comparison.height == 48
        expected = comparison[field].to_numpy()
        before, after = (
            comparison[f"{field}_{name}"].to_numpy().astype(np.float64)
            for name in ("before", "after")
        )
        assert np.isfinite(before).all()
        assert np.isfinite(after).all()
        source = native[field].to_numpy().astype(np.float64)
        assert len(source) == 24 // width
        errors = [
            float(np.max(np.abs(values.reshape(-1, width * 2).mean(axis=1) - source)))
            for values in (before, after)
        ]
        if field == SHORT_WAVE:
            assert errors[1] < 0.001, (field, width, errors)
        hours = np.arange(0.25, 24, 0.5)
        shifts = [
            float(np.sum(hours * values) / values.sum() - np.sum(hours * expected) / expected.sum())
            for values in (before, after)
        ]
        if field == SHORT_WAVE and not cloudy:
            assert abs(shifts[1]) < 0.15
            assert shifts[0] > 1
        report.append(
            f"| {label} | {width} | {errors[0]:.6f} | {errors[1]:.6f} | "
            f"{shifts[0]:.6f} | {shifts[1]:.6f} |"
        )
        data = pl.DataFrame(
            {"hour": hours, CURVES[0]: expected, CURVES[1]: before, CURVES[2]: after}
        ).unpivot(index="hour", variable_name="curve", value_name="flux")
        lines = [
            alt.Chart(data.filter(pl.col("curve") == curve))
            .mark_line(
                interpolate="linear",
                strokeWidth=2.5,
            )
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "hour:Q",
                    title="Half-hour midpoint (UTC hour)",
                    scale=alt.Scale(domain=[0, 24]),
                    axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                ),
                y=alt.Y(
                    "flux:Q",
                    title="Radiation (W/m²)",
                    scale=alt.Scale(zero=field == SHORT_WAVE),
                ),
                color=alt.Color(
                    "curve:N",
                    title=None,
                    scale=alt.Scale(
                        domain=CURVES, range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                    ),
                    legend=alt.Legend(orient="top", columns=1, labelLimit=500),
                ),
                strokeDash=alt.StrokeDash(
                    "curve:N",
                    scale=alt.Scale(domain=CURVES, range=[[6, 3], [2, 2], [1, 0]]),
                    legend=None,
                ),
            )
            for curve in CURVES
        ]
        panel = alt.layer(*lines).properties(
            width=CONTENT_WIDTH_PX - 65,
            height=150,
            title=alt.TitleParams(
                text=f"{label}: {width}-hour native means",
                subtitle=(
                    f"Maximum interval-mean deviation: before {errors[0]:.2f}, "
                    f"after {errors[1]:.5f} W/m²"
                ),
                anchor="start",
            ),
        )
        panels.append(panel)
    chart = (
        alt.vconcat(*panels, spacing=25)
        .resolve_scale(y="independent")
        .properties(
            title=alt.TitleParams(
                text="Radiation resampling uses interval timing and changing cloud attenuation",
                subtitle=[
                    "Synthetic examples: 52°N, 0°E; 21 March 2026. Reference: half-hour means.",
                    "Reconstructed values are shown at half-hour midpoints.",
                    (
                        "Unknown cloud and thermal changes are not recovered; "
                        "forecast skill is not measured."
                    ),
                    "Short-wave retains each interval's energy.",
                    "Long-wave samples a shape-preserving cubic spline at half-hour midpoints.",
                ],
                anchor="start",
            )
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(vlc.vegalite_to_svg(chart.to_dict()))
    args.output.with_suffix(".png").write_bytes(vlc.vegalite_to_png(chart.to_dict(), scale=1.2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
