"""Plot aggregate feature diagnostics; emit figure text for review before rendering."""

import argparse
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
import vl_convert as vlc

FIGURES: Final[dict[str, tuple[str, str]]] = {
    "radiation_probe": (
        "Radiation means are shifted late by the current resampler",
        "Synthetic half-hour means; production interpolates between 3 h or 6 h backward means.",
    ),
    "radiation_profile": (
        "Engineered radiation profiles differ by forecast lead",
        "31 NGED series, control member; July 2025–June 2026; means of finite rows.",
    ),
    "direction": (
        "Linear interpolation can turn north into south",
        "Synthetic instantaneous samples 350° to 10° over 3 h; 10° and 370° are equivalent.",
    ),
    "windchill": (
        "Windchill is calculated outside the formula's domain",
        "Control member; valid domain: air ≤10°C and 10 m wind ≥4.8 km/h; finite inputs only.",
    ),
    "precipitation": (
        "Precipitation-type missingness by weather-run month",
        "31 NGED series, control member; null or NaN rows grouped by weather-run month.",
    ),
    "lags": (
        "Some eligible two-week power lags are missing",
        "31 NGED series, control member; denominator: rows with issuance lead below 336 h.",
    ),
    "weekday": (
        "Production already uses weekday sine and cosine",
        "Synthetic seven-day cycle; sine and cosine repeat weekly alongside the ordinal.",
    ),
    "temperature": (
        "Six-hour sampling loses some three-hour temperature extrema",
        "Control member, July 2025–June 2026; matched UTC days; weather-run leads >0 to 144 h.",
    ),
    "completeness": (
        "Missing inputs differ between training and validation",
        "24 inputs, control member; April 2024–June 2025 training, July 2025–June 2026 validation.",
    ),
}
NOTES: Final[dict[str, str]] = {
    "radiation_profile": (
        "Legend: lead from power issuance, [0,72) h versus ≥264 h; row-weighted means."
    ),
    "windchill": "Training April 2024–June 2025; validation July 2025–June 2026; 31 NGED series.",
    "lags": (
        "Training April 2024–June 2025; validation July 2025–June 2026; final band is [264,336) h."
    ),
    "temperature": (
        "Each retained series/run-day has 8 finite three-hour values and 4 six-hour values."
    ),
    "completeness": (
        "Row-weighted missing shares include expected unavailable lags at long forecast leads."
    ),
}
FEATURE_LABELS: Final[dict[str, str]] = {
    "downward_short_wave_radiation_flux_surface": "Short-wave radiation",
    "categorical_precipitation_type_surface": "Precipitation type",
    "pressure_reduced_to_mean_sea_level": "Mean sea-level pressure",
    "dew_point_temperature_2m": "2 m dew-point temperature",
    "geopotential_height_500hpa": "500 hPa geopotential height",
    "local_utc_offset_minutes": "UTC offset (minutes)",
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]
WIDTH: Final[int] = 600
BANDS: Final[list[str]] = ["0–72 h", "72–144 h", "144–264 h", "264 h and later"]


def colour(column: str, *, domain: list[str]) -> alt.Color:
    """Assign contrasting brand colours in an explicit order."""
    return alt.Color(f"{column}:N", scale=alt.Scale(domain=domain, range=COLOURS), title=None)


def lines(
    data: pl.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    domain: list[str],
    x_title: str,
    y_title: str,
) -> alt.Chart:
    """Draw two explicitly coloured curves with redundant line styles."""
    return (
        alt.Chart(data)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(f"{x}:Q", title=x_title),
            y=alt.Y(f"{y}:Q", title=y_title),
            color=colour(group, domain=domain),
            strokeDash=alt.StrokeDash(f"{group}:N", legend=None),
        )
        .properties(width=WIDTH, height=190)
    )


def plot_probe(data: pl.DataFrame) -> alt.VConcatChart:
    """Compare independent radiation truth with production reconstruction at both step widths."""
    panels = []
    for width in (3, 6):
        panel = lines(
            data.filter(pl.col("step_hours") == width),
            x="hour",
            y="irradiance",
            group="curve",
            domain=["truth", "production"],
            x_title="Hour ending in synthetic day",
            y_title="Short-wave radiation (W/m²)",
        ).encode(
            x=alt.X(
                "hour:Q",
                title="Hour ending in synthetic day",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
            )
        )
        panels.append(
            panel.properties(title=f"Native means over {width} hours; lines: truth and production")
        )
    return alt.vconcat(*panels)


def plot_radiation(data: pl.DataFrame) -> alt.Chart:
    """Plot pooled engineered radiation profiles in short and long forecast bands."""
    subset = data.filter(
        pl.col("split") == "Validation", pl.col("band").is_in([BANDS[0], BANDS[-1]])
    ).with_columns(irradiance=pl.col("radiation_sum") / pl.col("rows"))
    return lines(
        subset,
        x="hour",
        y="irradiance",
        group="band",
        domain=[BANDS[0], BANDS[-1]],
        x_title="UTC hour of the period end",
        y_title="Mean short-wave radiation (W/m²)",
    ).encode(
        x=alt.X(
            "hour:Q", title="UTC hour of the period end", axis=alt.Axis(values=[0, 6, 12, 18, 24])
        )
    )


def plot_direction(data: pl.DataFrame) -> alt.Chart:
    """Display the wrong long rotation alongside a shortest-arc reference."""
    subset = (
        data.filter(pl.col("case") == "Crossing north")
        .with_columns(
            shortest_arc=pl.when(pl.col("shortest_arc") < 180)
            .then(pl.col("shortest_arc") + 360)
            .otherwise(pl.col("shortest_arc")),
        )
        .unpivot(
            index="hour",
            on=["production", "shortest_arc"],
            variable_name="curve",
            value_name="bearing",
        )
        .with_columns(
            curve=pl.col("curve").replace(
                {
                    "shortest_arc": "Shortest-arc reference",
                    "production": "Production linear interpolation",
                }
            )
        )
    )
    return lines(
        subset,
        x="hour",
        y="bearing",
        group="curve",
        domain=["Shortest-arc reference", "Production linear interpolation"],
        x_title="Hours after first native direction",
        y_title="Bearing (°; north unwrapped at 360°)",
    )


def plot_rates(data: pl.DataFrame, *, ylabel: str, last_band_label: str = BANDS[-1]) -> alt.Chart:
    """Compare training and validation prevalence by lead band."""
    data = data.with_columns(band=pl.col("band").replace({BANDS[-1]: last_band_label}))
    order = [*BANDS[:-1], last_band_label]
    return (
        alt.Chart(data)
        .mark_point(size=80, filled=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "band:N",
                title="Forecast lead from power issuance",
                sort=order,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("percent:Q", title=ylabel, scale=alt.Scale(zero=True)),
            color=colour("split", domain=["Training", "Validation"]),
            shape=alt.Shape("split:N", legend=None),
        )
        .properties(width=WIDTH, height=180)
    )


def plot_windchill(rates: pl.DataFrame, profile: pl.DataFrame) -> alt.VConcatChart:
    """Show domain violations and actual engineered windchill-minus-temperature values."""
    subset = profile.filter(pl.col("split") == "Validation").with_columns(
        mean_excess=pl.col("excess_sum") / pl.col("rows"),
    )
    curve = lines(
        subset,
        x="temperature_bin",
        y="mean_excess",
        group="domain",
        domain=["Inside formula domain", "Outside formula domain"],
        x_title="Air temperature (°C; 2°C bins)",
        y_title="Mean windchill minus air temperature (°C)",
    ).properties(title="Validation values; zero means windchill equals air temperature")
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0]}))
        .mark_rule(color=ocf.BLACK_1)
        .encode(  # ty: ignore[unresolved-attribute]
            y="zero:Q"
        )
    )
    return alt.vconcat(plot_rates(rates, ylabel="Rows outside formula domain (%)"), curve + zero)


def plot_precipitation(data: pl.DataFrame) -> alt.Chart:
    """Plot the training/validation feature availability boundary."""
    return (
        alt.Chart(data)
        .mark_point(size=75, filled=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "month:O", title="Weather-run start month (UTC)", axis=alt.Axis(labelAngle=-45)
            ),
            y=alt.Y(
                "percent:Q",
                title="Missing precipitation-type rows (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=colour("split", domain=["Training", "Validation"]),
            shape=alt.Shape("split:N", legend=None),
        )
        .properties(width=WIDTH, height=200)
    )


def plot_weekday(data: pl.DataFrame) -> alt.VConcatChart:
    """Show the ordinal and the two existing cyclic weekday columns."""
    order = data["weekday"].to_list()
    top = (
        alt.Chart(data)
        .mark_line(point=True, color=ocf.BRAND_ORANGE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("weekday:N", sort=order, title=None),
            y=alt.Y("ordinal:Q", title="Ordinal code (0–6)"),
        )
        .properties(width=WIDTH, height=100)
    )
    cyclic = data.unpivot(
        index=["weekday", "ordinal"],
        on=["sine", "cosine"],
        variable_name="column",
        value_name="value",
    )
    bottom = (
        alt.Chart(cyclic)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("weekday:N", sort=order, title="Local weekday (Monday to Sunday)"),
            y=alt.Y("value:Q", title="Circular feature value"),
            color=colour("column", domain=["sine", "cosine"]),
            strokeDash=alt.StrokeDash("column:N", legend=None),
        )
        .properties(width=WIDTH, height=140)
    )
    return alt.vconcat(top, bottom)


def plot_temperature(data: pl.DataFrame) -> alt.Chart:
    """Compare complete-day mean ranges on matched short-lead temperature knots."""
    subset = (
        data.filter(pl.col("split") == "Validation")
        .select(
            pl.col("mean_range_3h").alias("3-hour samples"),
            pl.col("mean_range_6h").alias("6-hour subset"),
        )
        .unpivot(variable_name="sampling", value_name="range")
    )
    return (
        alt.Chart(subset)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("range:Q", title="Mean daily temperature range (°C)"),
            y=alt.Y("sampling:N", title=None, sort=["3-hour samples", "6-hour subset"]),
            color=colour("sampling", domain=["3-hour samples", "6-hour subset"]),
        )
        .properties(width=WIDTH, height=100)
    )


def plot_completeness(data: pl.DataFrame) -> alt.Chart:
    """Show missingness for every production feature, including structurally null lags."""
    subset = data.with_columns(
        percent=100 * pl.col("missing") / pl.col("rows"),
        feature=pl.col("feature").replace(FEATURE_LABELS),
    )
    return (
        alt.Chart(subset)
        .mark_point(filled=True, size=55, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("percent:Q", title="Null or NaN rows (%; all forecast leads)"),
            y=alt.Y("feature:N", title=None, axis=alt.Axis(labelLimit=330)),
            color=colour("split", domain=["Training", "Validation"]),
            shape=alt.Shape("split:N", legend=None),
        )
        .properties(width=WIDTH - 150, height=480)
    )


def figure_note(key: str) -> str:
    """Return the exact population note used in both text review and rendering."""
    note = NOTES.get(key, "")
    if key not in {"radiation_probe", "direction", "weekday"}:
        note += " Control member: unperturbed weather ensemble member 0."
    return note


def main() -> None:
    """Write reviewable figure text or render SVG and PNG from saved audit tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png-dir", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    text = "\n\n".join(
        f"{key}\n{title}\n{subtitle}\n{figure_note(key)}"
        for key, (title, subtitle) in FIGURES.items()
    )
    args.png_dir.mkdir(parents=True, exist_ok=True)
    (args.png_dir / "figure_text.txt").write_text(text + "\n")
    if args.text_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    alt.data_transformers.disable_max_rows()

    def read(name: str) -> pl.DataFrame:
        return pl.read_parquet(args.input / f"{name}.parquet")

    plots = {
        "radiation_probe": plot_probe(read("radiation_probe")),
        "radiation_profile": plot_radiation(read("radiation_profile")),
        "direction": plot_direction(read("direction_probe")),
        "windchill": plot_windchill(read("windchill"), read("windchill_profile")),
        "precipitation": plot_precipitation(read("precip_months")),
        "lags": plot_rates(
            read("lag_missing"),
            ylabel="Missing eligible 336-hour power lags (%)",
            last_band_label="264–336 h",
        ),
        "weekday": plot_weekday(read("weekday_probe")),
        "temperature": plot_temperature(read("temperature")),
        "completeness": plot_completeness(read("ranges")),
    }
    for key, plot in plots.items():
        title, subtitle = FIGURES[key]
        note = figure_note(key)
        chart = plot.properties(
            title=alt.Title(
                textwrap.wrap(title, width=64),
                subtitle=textwrap.wrap(subtitle + " " + note, width=95),
                anchor="start",
                fontSize=18,
                subtitleFontSize=11,
            )
        ).configure_legend(orient="top", labelLimit=400)
        spec = chart.to_dict()
        (args.output / f"production_feature_faults_{key}.svg").write_text(vlc.vegalite_to_svg(spec))
        (args.png_dir / f"{key}.png").write_bytes(vlc.vegalite_to_png(spec, scale=2))
        print(f"rendered {key}", flush=True)


if __name__ == "__main__":
    main()
