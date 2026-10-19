"""Draw the three figures on the data-processing page of `docs/ml-experiments/`.

Run with `uv run python scripts/diagnostics/plot_data_processing_figures.py`, with
`DATA_PATH_INTERNAL` set to the main checkout's `data/` folder when run from a worktree.
The script only reads: the raw and cleaned power tables, the private table that maps each
anonymised label D1 to D22 to a series, and the CAMS irradiance downloaded at the demand series'
locations. It writes three SVG files to `docs/ml-experiments/data-processing/`; optimise them
afterwards with `npx svgo@4 --multipass --precision=1 --final-newline`.

The cut dates in `data_processing_cut_dates.csv` come from the differentiable-physics design
session's analysis and are provisional. Only Primary substations are drawn, under their `D` labels:
no site name, no coordinate, and no metered generator appears.
"""

import textwrap
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from contracts.settings import Settings
from plotting.ocf_theme import (
    BRAND_ORANGE,
    DATA_BLUE,
    DATA_GREEN,
    DATA_PURPLE,
    DATA_SKY,
    GREY_3,
)

CUT_DATES_PATH: Final[Path] = Path(__file__).with_name("data_processing_cut_dates.csv")
OUTPUT_DIR: Final[Path] = Path("docs/ml-experiments/data-processing")
_SETTINGS: Final[Settings] = Settings()
DATA_DIR: Final[Path] = Path(_SETTINGS.data_path_internal)
RAW_POWER_PATH: Final[str] = _SETTINGS.power_time_series_data_path
CLEANED_POWER_PATH: Final[str] = _SETTINGS.cleaned_power_time_series_data_path
METADATA_PATH: Final[Path] = DATA_DIR / "NGED" / "metadata.parquet"
SERIES_PATH: Final[Path] = (
    DATA_DIR / "studies" / "per_study" / "stage1_pilot" / "demand" / "inputs" / "series.parquet"
)
"""The private table that maps each label D1 to D22 to a `time_series_id`."""
CAMS_PATH: Final[Path] = (
    DATA_DIR
    / "studies"
    / "downloads"
    / "reanalysis"
    / "CAMS"
    / "demand_points"
    / "cams_demand_points.parquet"
)
CONTENT_WIDTH_PX: Final[int] = 680
"""The width of the docs text column, which every figure on a docs page matches."""

TRIM_SERIES: Final[tuple[str, ...]] = ("D4", "D14", "D9", "D16", "D10", "D18", "D5")
"""Primaries with clear cuts, drawn in this order."""
LEVEL_SERIES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The Primaries with histories long enough to span the 2020 lockdown and season-year 2022."""
LEVEL_COLOURS: Final[tuple[str, ...]] = (BRAND_ORANGE, DATA_BLUE, DATA_SKY, DATA_PURPLE, DATA_GREEN)
DARK_SERIES: Final[str] = "D2"
"""An MVA-only Primary with no cut, so its summer weeks are clean."""
DARK_WEEKS: Final[tuple[date, date]] = (date(2025, 7, 7), date(2025, 7, 21))
DARK_IRRADIANCE_W_M2: Final[float] = 1.0
LOCKDOWN: Final[tuple[date, date]] = (date(2020, 3, 23), date(2021, 7, 19))
NIGHT_HOURS_UTC: Final[tuple[int, int]] = (0, 6)
"""Half-hours ending after 00:00 and up to 06:00 UTC count as night."""
PANEL_HEIGHT_PX: Final[int] = 95
PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 70
GREY_TEXT: Final[str] = "#6b6560"
NARROW_SHADE_DAYS: Final[int] = 150
"""A shaded band shorter than this holds no room for its label, which goes right of the cut."""


def _wrapped(*, text: str, width: int) -> list[str]:
    """Split a subtitle into lines no longer than `width` characters."""
    return textwrap.wrap(text, width=width, break_long_words=False)


def _series_ids() -> pl.DataFrame:
    """Map each anonymised label to its series and unit, keeping Primary substations only."""
    series = pl.read_parquet(
        SERIES_PATH,
        columns=["time_series_id", "site"],
    )
    metadata = pl.read_parquet(
        METADATA_PATH, columns=["time_series_id", "time_series_type", "units"]
    )
    return (
        series.join(metadata, on="time_series_id", how="left", validate="1:1")
        .filter(pl.col("time_series_type") == "Disaggregated Demand")
        .with_columns(pl.col("units").cast(pl.String))
        .rename({"site": "series_label"})
    )


def _cut_dates() -> pl.DataFrame:
    return pl.read_csv(CUT_DATES_PATH, try_parse_dates=True)


def _daily_mean(*, scan: pl.LazyFrame, series_ids: list[int]) -> pl.DataFrame:
    """Daily mean power, in the unit the series is metered in."""
    return (
        scan.filter(pl.col("time_series_id").is_in(series_ids))
        .group_by("time_series_id", pl.col("time").dt.date().alias("date"))
        .agg(power=pl.col("power").mean())
        .collect()
    )


def _trim_frame(*, ids: pl.DataFrame, cuts: pl.DataFrame) -> pl.DataFrame:
    """Daily means for the trim figure: raw before each cut, cleaned from the cut on."""
    chosen = ids.filter(pl.col("series_label").is_in(TRIM_SERIES)).join(
        cuts, on="series_label", how="inner", validate="1:1"
    )
    id_list = chosen["time_series_id"].to_list()
    raw = _daily_mean(scan=pl.scan_delta(RAW_POWER_PATH), series_ids=id_list)
    cleaned = _daily_mean(
        scan=pl.scan_delta(CLEANED_POWER_PATH).filter(pl.col("drop_reason").is_null()),
        series_ids=id_list,
    )
    frames = []
    for row in chosen.iter_rows(named=True):
        cut: date = row["recommended_first_usable_date"]
        one_id = row["time_series_id"]
        dropped = raw.filter((pl.col("time_series_id") == one_id) & (pl.col("date") < cut))
        kept = cleaned.filter((pl.col("time_series_id") == one_id) & (pl.col("date") >= cut))
        frames.append(
            pl.concat(
                [
                    dropped.with_columns(kind=pl.lit("dropped")),
                    kept.with_columns(kind=pl.lit("kept")),
                ]
            )
            .with_columns(
                series_label=pl.lit(row["series_label"]),
                unit=pl.lit(row["units"]),
                cut=pl.lit(cut),
                months_cut=pl.lit(row["months_cut"]),
            )
            .drop("time_series_id")
        )
    return pl.concat(frames)


def trim_figure(*, ids: pl.DataFrame, cuts: pl.DataFrame) -> alt.VConcatChart:
    """Daily mean power over the full history, with the dropped months shaded grey."""
    frame = _trim_frame(ids=ids, cuts=cuts)
    panels = []
    for label in TRIM_SERIES:
        part = frame.filter(pl.col("series_label") == label)
        first = part["date"].min()
        cut = part["cut"][0]
        unit = part["unit"][0]
        months = part["months_cut"][0]
        base = alt.Chart(part)
        shade = (
            alt.Chart(pl.DataFrame({"start": [first], "end": [cut]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(color=GREY_3, opacity=0.5)
            .encode(x="start:T", x2="end:T")
        )
        narrow = (cut - first).days < NARROW_SHADE_DAYS
        label_text = (
            alt.Chart(pl.DataFrame({"x": [cut if narrow else first], "text": ["dropped"]}))  # ty: ignore[unresolved-attribute]
            .mark_text(
                align="left",
                baseline="top",
                dx=4,
                dy=3,
                color=GREY_TEXT,
                fontSize=10,
            )
            .encode(x="x:T", y=alt.value(0), text="text:N")
        )
        line = base.mark_line(strokeWidth=1, clip=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
            y=alt.Y(
                "power:Q",
                title=f"Daily mean ({unit})",
                scale=alt.Scale(zero=False, nice=True),
                axis=alt.Axis(tickCount=3),
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["dropped", "kept"], range=[GREY_TEXT, DATA_BLUE]),
                legend=None,
            ),
            detail="kind:N",
        )
        rule = (
            alt.Chart(pl.DataFrame({"cut": [cut]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=BRAND_ORANGE, strokeDash=[4, 3])
            .encode(x="cut:T")
        )
        panels.append(
            (shade + line + rule + label_text).properties(
                width=PLOT_WIDTH_PX,
                height=PANEL_HEIGHT_PX,
                title=alt.TitleParams(
                    f"{label}: first {months:g} months dropped (first usable date {cut})",
                    anchor="start",
                    fontSize=11,
                ),
            )
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(x="shared")
        .properties(
            title=alt.TitleParams(
                text=["Power is dropped from the start of each series (provisional cut dates)"],
                subtitle=_wrapped(
                    text="Daily mean power of 7 Primary substations over their full history. "
                    "Grey: dropped months, from the raw table so flagged rows still show. "
                    "Blue: kept months, from the cleaned table. Orange rule: first usable date.",
                    width=120,
                ),
                anchor="start",
            )
        )
    )


def _half_hourly(*, series_id: int, start: date, end: date, scan: pl.LazyFrame) -> pl.DataFrame:
    low = datetime(start.year, start.month, start.day, tzinfo=UTC)
    high = datetime(end.year, end.month, end.day, tzinfo=UTC)
    return (
        scan.filter(
            (pl.col("time_series_id") == series_id)
            & (pl.col("time") > low)
            & (pl.col("time") <= high)
        )
        .select("time", "power")
        .sort("time")
        .collect()
    )


def dark_figure(*, ids: pl.DataFrame) -> alt.Chart:
    """Two summer weeks of half-hourly power, with the dark half-hours highlighted."""
    row = ids.filter(pl.col("series_label") == DARK_SERIES).row(0, named=True)
    power = _half_hourly(
        series_id=row["time_series_id"],
        start=DARK_WEEKS[0],
        end=DARK_WEEKS[1],
        scan=pl.scan_delta(CLEANED_POWER_PATH).filter(pl.col("drop_reason").is_null()),
    )
    cams = (
        pl.scan_parquet(CAMS_PATH)
        .filter(pl.col("site") == DARK_SERIES)
        .select("time", "ghi_w_m2")
        .collect()
    )
    # Power and CAMS timestamps both mark the end of their period, so a half-hour belongs to the
    # hour that ends at or after it.
    joined = (
        power.with_columns(
            hour=(pl.col("time") - pl.duration(microseconds=1)).dt.truncate("1h")
            + pl.duration(hours=1)
        )
        .join(cams, left_on="hour", right_on="time", how="left", validate="m:1")
        .with_columns(
            kind=pl.when(pl.col("ghi_w_m2") < DARK_IRRADIANCE_W_M2)
            .then(pl.lit("Dark (demand only)"))
            .otherwise(pl.lit("Daylight (solar leaks in)"))
        )
    )
    unit = row["units"]
    line = (
        alt.Chart(joined)  # ty: ignore[unresolved-attribute]
        .mark_line(color=GREY_3, strokeWidth=1, aria=False)
        .encode(x=alt.X("time:T"), y=alt.Y("power:Q"))
    )
    points = (
        alt.Chart(joined)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=14, aria=False)
        .encode(
            x=alt.X(
                "time:T",
                title="Half-hour (UTC)",
                axis=alt.Axis(format="%-d %b", tickCount=14),
            ),
            y=alt.Y("power:Q", title=f"Power ({unit}; apparent power, never negative)"),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(
                    domain=["Dark (demand only)", "Daylight (solar leaks in)"],
                    range=[DATA_BLUE, GREY_TEXT],
                ),
                legend=alt.Legend(orient="top"),
            ),
            opacity=alt.Opacity(
                "kind:N",
                scale=alt.Scale(
                    domain=["Dark (demand only)", "Daylight (solar leaks in)"], range=[1.0, 0.35]
                ),
                legend=None,
            ),
        )
    )
    return (line + points).properties(
        width=PLOT_WIDTH_PX,
        height=230,
        title=alt.TitleParams(
            text=["Only the dark half-hours of an MVA-only Primary are demand alone"],
            subtitle=_wrapped(
                text=f"Half-hourly power of Primary {DARK_SERIES}, two summer weeks in July "
                "2025. Blue: half-hours with CAMS irradiance below 1 W/m². Grey: daylight "
                "half-hours, where embedded solar output is folded into the apparent power.",
                width=120,
            ),
            anchor="start",
        ),
    )


def level_figure(*, ids: pl.DataFrame, cuts: pl.DataFrame) -> alt.LayerChart:
    """Monthly mean night power as a share of each series' own mean, with the lockdown shaded."""
    chosen = (
        ids.filter(pl.col("series_label").is_in(LEVEL_SERIES))
        .join(
            cuts.select("series_label", "recommended_first_usable_date"),
            on="series_label",
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("recommended_first_usable_date").fill_null(date(2019, 9, 1)))
    )
    id_list = chosen["time_series_id"].to_list()
    hours = (pl.col("time") - pl.duration(microseconds=1)).dt.hour()
    night = (
        pl.scan_delta(CLEANED_POWER_PATH)
        .filter(pl.col("drop_reason").is_null() & pl.col("time_series_id").is_in(id_list))
        .filter(hours.is_between(NIGHT_HOURS_UTC[0], NIGHT_HOURS_UTC[1] - 1))
        .group_by("time_series_id", pl.col("time").dt.truncate("1mo").alias("month"))
        .agg(power=pl.col("power").mean(), half_hours=pl.len())
        .collect()
        .join(chosen, on="time_series_id", how="inner", validate="m:1")
        .with_columns(
            cut_month=pl.col("recommended_first_usable_date").cast(pl.Datetime("us", "UTC"))
        )
        .filter(pl.col("month") >= pl.col("cut_month"))
        .filter(pl.col("half_hours") >= 100)
        .with_columns(share=100 * pl.col("power") / pl.col("power").mean().over("series_label"))
        .select("series_label", "month", "share")
    )
    shade = (
        alt.Chart(pl.DataFrame({"start": [LOCKDOWN[0]], "end": [LOCKDOWN[1]]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=GREY_3, opacity=0.5)
        .encode(x="start:T", x2="end:T")
    )
    lines = (
        alt.Chart(night)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.5, aria=False)
        .encode(
            x=alt.X("month:T", title="Month", axis=alt.Axis(format="%Y", tickCount="year")),
            y=alt.Y(
                "share:Q",
                title="Night power (% of the series' own mean)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "series_label:N",
                title=None,
                scale=alt.Scale(domain=list(LEVEL_SERIES), range=list(LEVEL_COLOURS)),
                legend=alt.Legend(orient="top"),
            ),
        )
    )
    return (shade + lines).properties(
        width=PLOT_WIDTH_PX,
        height=240,
        title=alt.TitleParams(
            text=["Night demand at the long-history Primaries follows the same seasons"],
            subtitle=_wrapped(
                text="Monthly mean power from 00:00 to 06:00 UTC at 5 Primaries, after each "
                "series' cut, as a percentage of that series' own mean. Grey band: the fixed "
                "lockdown window, 2020-03-23 to 2021-07-19.",
                width=120,
            ),
            anchor="start",
        ),
    )


def main() -> None:
    """Draw the three figures and write them as SVG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = _series_ids()
    cuts = _cut_dates()
    trim_figure(ids=ids, cuts=cuts).save(OUTPUT_DIR / "figure_1_trim.svg")
    dark_figure(ids=ids).save(OUTPUT_DIR / "figure_2_dark_rows.svg")
    level_figure(ids=ids, cuts=cuts).save(OUTPUT_DIR / "figure_3_shared_level.svg")


if __name__ == "__main__":
    main()
