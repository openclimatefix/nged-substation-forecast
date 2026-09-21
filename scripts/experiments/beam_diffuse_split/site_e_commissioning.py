"""Draw site E's output relative to the rest of the fleet, day by day, through its first year.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**The measurement is a ratio, because a ratio cancels the weather.** Each day's value is the median,
over the day's bright half-hours, of site E's capacity factor divided by the median capacity factor
of the other five photovoltaic sites. A half-hour counts as bright when at least four of those five
sites reported and their median capacity factor exceeded 0.15. Dividing by the fleet removes cloud,
season and time of day, so what is left is how much of site E was connected.

The plateau dates in `SEGMENTS` are a fitted result rather than an input: binary segmentation with
an L2 cost over the daily medians put every changepoint where the adjacent plateaus differ by a
Cohen's d of at least 2.0. This script takes those dates as given and recomputes each plateau's
level, so the levels in the figure always match the data the figure plots.

Run it with `uv run --no-project` plus `--with polars --with numpy --with altair --with pandas
--with deltalake --with vl-convert-python`, then copy the SVG it writes into
`docs/background/assets/`.
"""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import altair as alt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "plotting" / "src"))

# Importing the theme module registers and enables the OCF Altair theme as a side effect.
import plotting.ocf_theme as ocf

_LOG: Final[logging.Logger] = logging.getLogger("site_e_commissioning")

REPO_DATA_DIR: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data")
FIGURES_DIR: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse_figures"

SITE_IDS: Final[dict[int, str]] = {31: "A", 22: "B", 30: "C", 29: "D", 23: "E", 21: "F"}
"""NGED's `time_series_id` for each photovoltaic site, mapped to its anonymous label."""

SUBJECT: Final[str] = "E"
"""The site whose commissioning this figure traces."""

CAP_TOLERANCE_MW: Final[float] = 0.001
"""How far below the connection limit a cap must sit before its half-hour counts as curtailed."""

ALIGNMENT_FIXED_AT: Final[datetime] = datetime(2026, 3, 26, 8, 30, tzinfo=UTC)
"""The instant NGED corrected the half-hourly power stamps.

A reading stamped before this instant is half an hour late, and `build_dataset.ALIGNMENT_FIXED_AT`
carries the evidence. The ratio this figure plots is insensitive to the lateness, because every
site is late by the same half hour and the two capacity factors cancel it. The export cap is not:
the cap is a second-resolution event log with no averaging window to mislabel, so joining a late
power stamp to a correct cap stamp reads the wrong half-hour's setpoint.
"""

MIN_REFERENCE_SITES: Final[int] = 4
"""How many of the five reference sites must report before a half-hour's fleet median is usable."""

MIN_FLEET_CAPACITY_FACTOR: Final[float] = 0.15
"""How hard the fleet must be generating before a ratio against it is stable enough to plot."""

MIN_HALF_HOURS_PER_DAY: Final[int] = 4
"""How many bright half-hours a day needs before its median ratio is plotted."""

SEGMENTS: Final[list[tuple[str, str, str]]] = [
    ("2024-03-16", "2024-03-28", "first generation"),
    ("2024-03-29", "2024-04-02", "energisation step 1"),
    ("2024-04-03", "2024-04-12", "energisation step 2"),
    ("2024-04-13", "2024-04-18", "energisation step 3"),
    ("2024-04-19", "2024-04-24", "energisation step 4"),
    ("2024-04-25", "2024-04-29", "energisation step 5"),
    ("2024-04-30", "2024-05-06", "partial shutdown"),
    ("2024-05-13", "2024-06-06", "May plateau"),
    ("2024-06-08", "2024-07-01", "June outage"),
    ("2024-07-02", "2024-07-05", "re-energisation step 1"),
    ("2024-07-06", "2024-07-08", "re-energisation step 2"),
    ("2024-07-09", "2024-07-10", "re-energisation step 3"),
    ("2024-07-11", "2024-10-05", "summer plateau"),
    ("2024-10-06", "2025-07-31", "settled level"),
]
"""The fitted plateaus, each a start date, an end date, and a name for the log."""

SETTLED: Final[str] = "settled level"
"""The plateau every other plateau is expressed as a percentage of."""

ANNOTATIONS: Final[list[tuple[str, str]]] = [
    ("2024-08-06", "6 Aug 2024\nactive-network-management\nscheme goes live"),
    ("2024-10-06", "6 Oct 2024\noutput reaches\nits settled level"),
]

LABEL_PLACEMENTS: Final[list[tuple[Literal["left", "right"], int]]] = [
    ("right", -7),
    ("left", 7),
]
"""How each dated annotation sits against its rule: which side, and how far from it."""

Y_TITLE: Final[str] = "Relative output, % of site E's settled level"
INK: Final[str] = "#292B2B"
GRID_GREY: Final[str] = "#9B9B9B"


def _day(*, text: str) -> pl.Series:
    """Parse one ISO date, since Polars parses a Series rather than a scalar."""
    return pl.Series([text]).str.to_date()


def _went_live() -> tuple[pl.Series, float]:
    """Read the export cap and find the first half-hour it reaches the connection limit.

    Returns:
        That half-hour, and the connection limit itself.
    """
    cap = pl.read_parquet(REPO_DATA_DIR / "NGED" / "anm" / "export_cap_23.parquet").sort("time")
    limit = float(np.max(cap["cap_mw"].to_numpy()))
    return cap.filter(pl.col("cap_mw") >= limit - CAP_TOLERANCE_MW)["time"][0], limit


def _half_hourly_gain() -> pl.DataFrame:
    """Build site E's ratio to the fleet, one row per bright half-hour.

    Returns:
        `(time, date, gain)`, with the half-hours the live export cap had moved removed.
    """
    went_live, limit = _went_live()
    cap = pl.read_parquet(REPO_DATA_DIR / "NGED" / "anm" / "export_cap_23.parquet")
    capacity = (
        pl.scan_delta(str(REPO_DATA_DIR / "effective_capacity"))
        .sort("time")
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").last())
        .collect()
    )
    power = (
        pl.scan_delta(str(REPO_DATA_DIR / "NGED" / "power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(list(SITE_IDS)))
        .select("time_series_id", "time", "power")
        .collect()
        .join(capacity, on="time_series_id")
        .with_columns(
            # Correct the stamp before anything joins on it, so the cap join below compares a
            # half-hour of power against the setpoint that was actually in force during it.
            time=pl.when(pl.col("time") < pl.lit(ALIGNMENT_FIXED_AT))
            .then(pl.col("time").dt.offset_by("-30m"))
            .otherwise(pl.col("time")),
            site=pl.col("time_series_id").replace_strict(SITE_IDS),
            capacity_factor=pl.col("power") / pl.col("effective_capacity_mw"),
        )
    )
    fleet = (
        power.filter(pl.col("site") != SUBJECT)
        .group_by("time")
        .agg(
            fleet_capacity_factor=pl.col("capacity_factor").median(),
            reporting=pl.len(),
        )
    )
    return (
        power.filter(pl.col("site") == SUBJECT)
        .select("time", "capacity_factor")
        .join(fleet, on="time")
        .filter(
            pl.col("reporting") >= MIN_REFERENCE_SITES,
            pl.col("fleet_capacity_factor") > MIN_FLEET_CAPACITY_FACTOR,
        )
        .join(cap, on="time", how="left")
        .filter(
            ~(
                (pl.col("time") >= went_live)
                & pl.col("lowest_cap_mw").is_not_null()
                & (pl.col("lowest_cap_mw") < limit - CAP_TOLERANCE_MW)
            )
        )
        .with_columns(
            gain=pl.col("capacity_factor") / pl.col("fleet_capacity_factor"),
            date=pl.col("time").dt.date(),
        )
    )


def _plateau_levels(*, half_hourly: pl.DataFrame) -> pl.DataFrame:
    """Recompute each fitted plateau's level as a percentage of the settled level.

    Args:
        half_hourly: One row per bright half-hour, carrying `date` and `gain`.

    Returns:
        `(start, end, level)`, one row per plateau, ready to draw as a horizontal bar.
    """
    levels: dict[str, float] = {}
    for start, end, name in SEGMENTS:
        window = half_hourly.filter(
            pl.col("date") >= _day(text=start)[0], pl.col("date") <= _day(text=end)[0]
        )["gain"]
        levels[name] = float(np.median(window.to_numpy()))
        _LOG.info(
            "%-24s %s..%s median gain %.3f over %d half-hours",
            name,
            start,
            end,
            levels[name],
            window.len(),
        )
    settled = levels[SETTLED]
    bars = [
        (start, end, 100 * levels[name] / settled)
        for start, end, name in SEGMENTS
        if name != SETTLED
    ]
    bars.append(("2024-10-06", "2024-12-30", 100.0))
    return pl.DataFrame(
        {
            "start": [_day(text=start)[0] for start, _, _ in bars],
            "end": [_day(text=end)[0] for _, end, _ in bars],
            "level": [level for _, _, level in bars],
        }
    ).with_columns(
        pl.col("start").cast(pl.Datetime),
        pl.col("end").cast(pl.Datetime) + pl.duration(days=1),
    )


def _annotations() -> pl.DataFrame:
    """Return the two dated rules the lower panel carries."""
    return pl.DataFrame(
        {
            "date": [_day(text=date)[0] for date, _ in ANNOTATIONS],
            "label": [label for _, label in ANNOTATIONS],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime))


def _panel(
    *,
    daily: pl.DataFrame,
    plateaus: pl.DataFrame | None,
    domain: list[str] | None,
    title: str,
    tick_count: int,
) -> alt.LayerChart | alt.FacetChart:
    """Draw one panel of the figure.

    Args:
        daily: One row per day, carrying `date` and `percent`.
        plateaus: The fitted levels to draw as bars, or None to leave them off.
        domain: The x-axis date range, or None to fit the data.
        title: The panel's heading.
        tick_count: Roughly how many date ticks to label.

    Returns:
        The layered panel.
    """
    events = _annotations()
    x_scale = alt.Scale(domain=domain) if domain else alt.Undefined
    hundred = (
        alt.Chart(pl.DataFrame({"y": [100.0]}).to_pandas())
        .mark_rule(color=GRID_GREY, strokeDash=[4, 4])
        .encode(y=alt.Y("y:Q"))  # ty: ignore[unresolved-attribute]
    )
    points = (
        alt.Chart(daily.to_pandas())
        .mark_circle(size=26, opacity=0.5, color=ocf.BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "date:T",
                title=None,
                scale=x_scale,
                axis=alt.Axis(tickCount=tick_count, format="%b %Y"),
            ),
            y=alt.Y("percent:Q", title=Y_TITLE, scale=alt.Scale(domain=[0, 135])),
        )
    )
    rules = (
        alt.Chart(events.to_pandas())
        .mark_rule(color=INK, strokeWidth=1.5, strokeDash=[6, 3])
        .encode(x=alt.X("date:T", scale=x_scale))  # ty: ignore[unresolved-attribute]
    )
    if plateaus is None:
        return alt.layer(hundred, points, rules).properties(width=760, height=215, title=title)
    bars = (
        alt.Chart(plateaus.to_pandas())
        .mark_rect(color=ocf.ORANGE_RED, height=4)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("start:T", scale=x_scale), x2="end:T", y=alt.Y("level:Q")
        )
    )
    labels = [
        alt.Chart(events.to_pandas()[index : index + 1])
        .mark_text(
            fontSize=11,
            color=INK,
            lineBreak="\n",
            baseline="top",
            align=align,
            dx=dx,
            lineHeight=13,
        )
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("date:T", scale=x_scale), y=alt.value(4), text="label:N"
        )
        for index, (align, dx) in enumerate(LABEL_PLACEMENTS)
    ]
    return alt.layer(hundred, bars, points, rules, *labels).properties(
        width=760, height=215, title=title
    )


def main() -> None:
    """Build the gain series, recompute the plateau levels, and write the figure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    half_hourly = _half_hourly_gain()
    plateaus = _plateau_levels(half_hourly=half_hourly)
    daily = (
        half_hourly.group_by("date")
        .agg(gain=pl.col("gain").median(), half_hours=pl.len())
        .filter(pl.col("half_hours") >= MIN_HALF_HOURS_PER_DAY)
        .with_columns(percent=100 * pl.col("gain"))
        .sort("date")
    )
    chart = (
        alt.vconcat(
            _panel(
                daily=daily,
                plateaus=None,
                domain=None,
                title=(
                    "The whole record: almost nothing until late March 2024, "
                    "a settled level from October 2024"
                ),
                tick_count=10,
            ),
            _panel(
                daily=daily.filter(pl.col("date") <= pl.date(2024, 12, 31)),
                plateaus=plateaus,
                domain=["2024-02-01", "2024-12-31"],
                title=(
                    "2024 in detail: a staircase in April, an outage in June, "
                    "the staircase repeated in July"
                ),
                tick_count=11,
            ),
            spacing=34,
        )
        .properties(
            title=alt.TitleParams(
                text=(
                    "Site E was energised in stages, and kept rising after "
                    "the network scheme went live"
                ),
                subtitle=[
                    (
                        "Each point is one day's median ratio of site E's capacity factor "
                        "to the median capacity factor of the five other"
                    ),
                    (
                        "solar farms, over half-hours when the fleet was generating well. "
                        "Half-hours the export cap had moved, once the"
                    ),
                    (
                        "scheme was live, are excluded. Orange bars are fitted constant "
                        "levels. Site E has no telemetry between"
                    ),
                    "9 February and 11 March 2024, so the record starts in mid-March.",
                ],
                anchor="start",
                fontSize=17,
                subtitleFontSize=11,
            )
        )
        .configure_view(stroke=None)
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "site_e_commissioning.svg"
    chart.save(path)
    _LOG.info("wrote %s", path)


if __name__ == "__main__":
    main()
