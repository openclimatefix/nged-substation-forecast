"""Draw the lagged-power-features page's figures from the saved losses and the report's tables.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. It reads
`tables_<product>/*.parquet` (written by `report_lag_features.py`, so a chart cannot disagree with
the report), `losses_<product>.parquet` and the built frames, and writes one SVG per figure under
`<output-root>/<product>/figures_<product>/`, together with `figure_text_<product>.txt`, which
collects every figure's title, subtitle lines, axis titles and legend entries in one file for the
text reviews (`.claude/skills/study/SKILL.md`, "Review the text of every figure"). **Run it with
`--text-only` first, review the text, then render.**

**No NGED metered generator's identifier, name, coordinates or calendar date appears.** Plants are
the anonymous letters `studies.anonymise` assigns, power is a share of capacity, and a week's axis
counts days 1 to 7.

**Lead-day 0 is labelled "not usable for its early hours in a live service" wherever it is drawn**,
because a 00 UTC run is published after its first hours have passed.

The figures:

1. The planned contrasts with their 99% intervals, at both hyperparameter settings (the headline).
2. The data: forecast irradiance against measured power, and the lag against the target.
3. The techniques with known answers: the positive control, and the controls beside L1.
4. The XGBoost models work: three weeks of out-of-fold predictions, and each plant's error.
5. The sweep, ranked, with the no-fit references.
6. Results: gain over B0 against lead-day with climatology drawn, the monthly paired differences,
   the interval coverage and width, and global against per-plant error.

Run it with `uv run python studies/lag_features/lag_features_charts.py`.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from build_lag_frame import (
    FULL_SWEEP_LEAD_DAY,
    LAG_FEATURES_DIR,
    WEATHER_PRODUCTS,
    WeatherProduct,
    output_paths,
)
from studies.charts import CONTENT_WIDTH_PX, figure
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("lag_features_charts")

PERCENTAGE_POINTS: Final[float] = 100.0
"""Losses are fractions of capacity; charts show percentage points."""

LEAD_ZERO_TICK: Final[str] = "0*"
"""How lead-day 0 is ticked on an axis. Each figure that shows it explains the asterisk: lead-day 0
is not usable for its early hours in a live service."""

HOURS_PER_DAY: Final[int] = 24
"""Converts an hour of the day into a fraction of a day on a week's axis."""

WEEK_RULES: Final[tuple[str, ...]] = ("clearest", "most variable", "dullest")
"""The rules that pick the three weeks drawn, stated in the figure text and never chosen by eye."""

PANEL_HEIGHT_PX: Final[int] = 200
"""A single panel's height."""

ARM_COLOURS: Final[dict[str, str]] = {
    "B0": ocf.DATA_BLUE,
    "L1": ocf.BRAND_ORANGE,
    "W7": ocf.DATA_GREEN,
    "Q30": ocf.DATA_PURPLE,
    "T1": ocf.DATA_SKY,
    "N2": ocf.BLACK_1,
    "climatology": ocf.DATA_AMBER,
}
"""One colour per arm of the by-lead figure. Climatology uses an additional data colour."""


class FigureText:
    """Collects the text of every figure for the text reviews."""

    def __init__(self) -> None:
        """Start with no figures."""
        self.entries: dict[str, list[str]] = {}

    def add(self, *, figure_id: str, plots: str, lines: list[str]) -> None:
        """Record one figure's text.

        Args:
            figure_id: The figure's number on the page.
            plots: What the figure plots, for the reviewer.
            lines: The title, subtitle lines, axis titles and legend entries, one per line.
        """
        self.entries[figure_id] = [f"[plots] {plots}", *lines]

    def write(self, *, path: Path) -> None:
        """Write every figure's text to one file.

        Args:
            path: The file to write.
        """
        blocks = [f"Figure {key}\n" + "\n".join(lines) for key, lines in self.entries.items()]
        path.write_text("\n\n".join(blocks) + "\n")


def _intervals_chart(
    *, data: pl.DataFrame, y: str, x_title: str, colour: str, shape: str
) -> alt.LayerChart:
    """Draw dots with interval lines against a zero rule, one row per `y` value.

    Args:
        data: Rows with `difference`, `lower`, `upper`, the `y` column, and `setting`.
        y: The column naming each row.
        x_title: The axis title.
        colour: The column giving the colour field.
        shape: The column giving the shape field.

    Returns:
        The layered chart.
    """
    base = alt.Chart(data)
    rule = alt.Chart(pl.DataFrame({"zero": [0.0]})).mark_rule(color=ocf.BLACK_1).encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    lines = base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(f"{y}:N", title=None, sort=None),
        x=alt.X("lower:Q", title=x_title),
        x2="upper:Q",
        color=alt.Color(f"{colour}:N", legend=alt.Legend(title="Hyperparameter setting")),
    )
    dots = base.mark_point(filled=True, size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(f"{y}:N", sort=None),
        x="difference:Q",
        color=f"{colour}:N",
        shape=alt.Shape(f"{shape}:N", legend=None),
    )
    return (rule + lines + dots).properties(width=CONTENT_WIDTH_PX - 260, height=200)


def headline_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 1: the five planned contrasts with their 99% intervals.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    planned = pl.read_parquet(tables / "planned.parquet")
    rows = planned.with_columns(
        label=pl.col("contrast") + ": " + pl.col("treatment") + " minus " + pl.col("reference"),
        difference=pl.col("difference") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
    )
    x_title = (
        "Change in error, treatment minus reference (percentage points of capacity; "
        "more negative is better)"
    )
    title = "The five planned contrasts: what lagged power changes at the day-ahead lead"
    subtitle = [
        (
            "Mean absolute error, except P4, which is the continuous ranked probability score "
            "approximated from nine quantiles (0.1 to 0.9)."
        ),
        "Dot: estimate. Line: 99% interval from resampling whole months. Zero: no change.",
        "All rows are planned: written into the study plan before any result existed.",
    ]
    text.add(
        figure_id="1",
        plots="The five planned contrasts, both settings",
        lines=[title, *subtitle, x_title, "Hyperparameter setting: primary, sensitivity"],
    )
    return figure(
        panels=[
            _intervals_chart(
                data=rows, y="label", x_title=x_title, colour="setting", shape="setting"
            )
        ],
        number=1,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )


def data_figure(*, root: Path, product: WeatherProduct, text: FigureText) -> alt.VConcatChart:
    """Figure 2: forecast irradiance against power, and the lag against the target.

    Args:
        root: The output root.
        product: The weather product.
        text: The text collector.

    Returns:
        The figure.
    """
    frame = pl.read_parquet(
        output_paths(root=root, product=product, lead_days=(FULL_SWEEP_LEAD_DAY,))["day1"]
    ).with_columns(
        share=pl.col("power_mw") / pl.col("effective_capacity_mw"),
        lag_share=pl.col("lag_d1") / pl.col("effective_capacity_mw"),
    )
    sample = frame.sample(n=min(6000, frame.height), seed=0)
    forecast = (
        alt.Chart(sample)
        .mark_circle(size=6, opacity=0.35, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("nwp_ghi:Q", title="Forecast global horizontal irradiance (W/m²)"),
            y=alt.Y("share:Q", title="Measured power (share of capacity)"),
            color=alt.Color("site:N", legend=alt.Legend(title="Plant")),
        )
        .properties(width=CONTENT_WIDTH_PX - 80, height=PANEL_HEIGHT_PX)
    )
    lag = (
        alt.Chart(sample)
        .mark_circle(size=6, opacity=0.35, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lag_share:Q", title="Power at the same hour two days earlier (share of capacity)"
            ),
            y=alt.Y("share:Q", title="Measured power (share of capacity)"),
            color="site:N",
        )
        .properties(width=CONTENT_WIDTH_PX - 80, height=PANEL_HEIGHT_PX)
    )
    title = "Measured power against forecast irradiance, and against the lag"
    subtitle = [
        (
            "One dot is one daylight hour at one of six solar plants, day-ahead forecasts, "
            "a random sample of 6,000 hours."
        ),
        "The lag is the same clock hour on the latest whole day before the forecast was issued.",
    ]
    text.add(
        figure_id="2",
        plots="Forecast irradiance vs power; lag vs power, six plants",
        lines=[
            title,
            *subtitle,
            "Forecast global horizontal irradiance (W/m²)",
            "Measured power (share of capacity)",
            "Power at the same hour two days earlier (share of capacity)",
            "Plant: A to F",
        ],
    )
    return figure(
        panels=[forecast, lag], number=2, title=title, subtitle=subtitle, figure_planning=None
    )


def control_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 3: which synthetic level shifts L1 recovers, and the controls beside L1.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    control = pl.read_parquet(tables / "positive_control.parquet").with_columns(
        label=(pl.col("shift") * 100).round(0).cast(pl.Int32).cast(pl.String) + "% of power lost",
        setting=pl.lit("primary"),
        difference=pl.col("difference") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
    )
    phase1 = (
        pl.read_parquet(tables / "phase1.parquet")
        .filter(pl.col("arm").is_in(["B0", "L1", "N1", "N2"]))
        .with_columns(
            error=pl.col("error") * PERCENTAGE_POINTS,
            lower=pl.col("lower") * PERCENTAGE_POINTS,
            upper=pl.col("upper") * PERCENTAGE_POINTS,
        )
    )
    x_title = "L1 minus B0 error (percentage points of capacity; more negative is better)"
    recovered = _intervals_chart(
        data=control.with_columns(setting=pl.lit("99% interval")),
        y="label",
        x_title=x_title,
        colour="setting",
        shape="setting",
    )
    controls = (
        alt.Chart(phase1)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", title=None, sort=["B0", "N2", "N1", "L1"]),
            x=alt.X(
                "error:Q",
                title="Mean absolute error, screening months (% of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("arm:N", legend=None),
        )
        .properties(width=CONTENT_WIDTH_PX - 120, height=120)
    )
    title = "Known-answer tests: a synthetic loss of power, and the lag arms' controls"
    subtitle = [
        (
            "Top: the same pipeline on a synthetic target in which power after a fixed date is "
            "multiplied by one minus the shift. Line: 99% interval from resampling whole months."
        ),
        (
            "Bottom: B0 is the baseline, N2 reads a random day (a true null), N1 reads a day "
            "8 to 28 days earlier (the plant's slow level), L1 reads the latest whole day."
        ),
    ]
    text.add(
        figure_id="3",
        plots="Positive-control recovery; B0, N2, N1, L1 errors",
        lines=[
            title,
            *subtitle,
            x_title,
            "Mean absolute error, screening months (% of capacity; smaller is better)",
        ],
    )
    return figure(
        panels=[recovered, controls], number=3, title=title, subtitle=subtitle, figure_planning=None
    )


def _week_starts(*, losses: pl.DataFrame) -> dict[str, datetime]:
    """Pick the three weeks drawn: the clearest, most variable and dullest.

    Args:
        losses: Lead-day 1 B0 losses carrying `actual`.

    Returns:
        The first day of each rule's week.
    """
    weekly = (
        losses.with_columns(week=pl.col("time").dt.truncate("1w"))
        .group_by("week", "site")
        .agg(mean=(pl.col("actual") / pl.col("effective_capacity_mw")).mean())
        .group_by("week")
        .agg(level=pl.col("mean").mean(), spread=pl.col("mean").std(), n=pl.len())
        .filter(pl.col("n") >= 6)
    )
    return {
        "clearest": weekly.sort("level")["week"][-1],
        "most variable": weekly.sort("spread")["week"][-1],
        "dullest": weekly.sort("level")["week"][0],
    }


def models_work_figure(
    *, root: Path, product: WeatherProduct, text: FigureText
) -> alt.VConcatChart:
    """Figure 4: out-of-fold predictions against measured power for three weeks.

    Args:
        root: The output root.
        product: The weather product.
        text: The text collector.

    Returns:
        The figure.
    """
    losses = pl.read_parquet(root / product / f"losses_{product}.parquet").filter(
        (pl.col("scope") == "lead1")
        & (pl.col("setting") == "primary")
        & (pl.col("seed") == 0)
        & pl.col("arm").is_in(["B0", "L1"])
    )
    starts = _week_starts(losses=losses.filter(pl.col("arm") == "B0"))
    panels = []
    for rule, start in starts.items():
        week = losses.filter(
            (pl.col("time") >= start) & (pl.col("time") < pl.lit(start) + pl.duration(days=7))
        ).with_columns(
            day=(pl.col("time") - pl.lit(start)).dt.total_seconds() / 86400.0 + 1.0,
            predicted=pl.col("prediction") / pl.col("effective_capacity_mw"),
            measured=pl.col("actual") / pl.col("effective_capacity_mw"),
        )
        measured = week.filter(pl.col("arm") == "B0").with_columns(
            series=pl.lit("Measured"), value=pl.col("measured")
        )
        forecasts = week.with_columns(series=pl.col("arm"), value=pl.col("predicted"))
        stacked = pl.concat(
            [
                measured.select("site", "day", "series", "value"),
                forecasts.select("site", "day", "series", "value"),
            ]
        )
        panels.append(
            alt.Chart(stacked)
            .mark_line(aria=False, strokeWidth=1.2)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("day:Q", title="Day of the week (1 to 7)", scale=alt.Scale(domain=[1, 8])),
                y=alt.Y("value:Q", title="Power (share of capacity)"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["Measured", "B0", "L1"],
                        range=[ocf.BLACK_1, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title="Series"),
                ),
                row=alt.Row("site:N", title=f"Plant ({rule} week)"),
            )
            .properties(width=CONTENT_WIDTH_PX - 140, height=70)
        )
    title = "Day-ahead forecasts against measured power in three weeks chosen by rule"
    subtitle = [
        "Out-of-fold forecasts at lead-day 1 from B0 (no lags) and L1 (one lag), against power.",
        (
            "Weeks are chosen by rule, not by eye: the clearest, the most variable between "
            "plants, and the dullest, of the weeks with six plants."
        ),
    ]
    text.add(
        figure_id="4",
        plots="Three weeks of B0 and L1 forecasts against measured power",
        lines=[
            title,
            *subtitle,
            "Day of the week (1 to 7)",
            "Power (share of capacity)",
            "Series: Measured, B0, L1",
        ],
    )
    return figure(panels=panels, number=4, title=title, subtitle=subtitle, figure_planning=None)


def sweep_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 5: the sweep ranked, with the no-fit references.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    phase1 = pl.read_parquet(tables / "phase1.parquet").with_columns(
        error=pl.col("error") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
        kind=pl.when(pl.col("fitted"))
        .then(pl.lit("Fitted XGBoost arm"))
        .otherwise(pl.lit("No fit")),
    )
    baseline = float(phase1.filter(pl.col("arm") == "B0")["error"][0])
    order = phase1.sort("error")["arm"].to_list()
    bars = (
        alt.Chart(phase1)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=order, title=None),
            x=alt.X("error:Q", title="Mean absolute error (% of capacity; smaller is better)"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.DATA_AMBER]),
                legend=alt.Legend(title="Arm"),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [baseline]})).mark_rule(color=ocf.BRAND_ORANGE).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    title = "A screen of the lag ideas, ranked by error, to generate hypotheses"
    subtitle = [
        "Mean absolute error of each arm at lead-day 1 over the first 13 calendar months.",
        "Orange line: B0, the baseline with no lag. Climatology, persistence and R1 need no fit.",
        "All rows are exploratory. No pairwise comparison is made here.",
    ]
    text.add(
        figure_id="5",
        plots="Every arm's screening-month error, ranked",
        lines=[
            title,
            *subtitle,
            "Mean absolute error (% of capacity; smaller is better)",
            "Arm: Fitted XGBoost arm, No fit",
        ],
    )
    return figure(
        panels=[(bars + rule).properties(width=CONTENT_WIDTH_PX - 120, height=360)],
        number=5,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )


def by_lead_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 6: each arm's gain over B0 against lead-day, and the error of B0 and climatology.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    by_lead = pl.read_parquet(tables / "by_lead.parquet").with_columns(
        lead=pl.col("lead_day").cast(pl.String).replace({"0": LEAD_ZERO_TICK}),
        error=pl.col("error") * PERCENTAGE_POINTS,
        gain=pl.col("difference_to_b0") * PERCENTAGE_POINTS,
        lower=pl.col("difference_lower") * PERCENTAGE_POINTS,
        upper=pl.col("difference_upper") * PERCENTAGE_POINTS,
    )
    order = [
        LEAD_ZERO_TICK if d == 0 else str(d) for d in sorted(by_lead["lead_day"].unique().to_list())
    ]
    x_title = "Lead-day (days between the run and the target day)"
    gains = by_lead.filter(pl.col("arm").is_in(["L1", "W7", "Q30", "T1", "N2"]))
    gain_colour = alt.Color(
        "arm:N",
        scale=alt.Scale(
            domain=["L1", "W7", "Q30", "T1", "N2"],
            range=[ARM_COLOURS[a] for a in ("L1", "W7", "Q30", "T1", "N2")],
        ),
        legend=alt.Legend(title="Arm"),
    )
    gain_x = alt.X("lead:N", sort=order, title=x_title)
    gain_lines = (
        alt.Chart(gains)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=gain_x,
            y=alt.Y("gain:Q", title="Error minus B0's (pp of capacity)"),
            color=gain_colour,
        )
    )
    gain_bars = (
        alt.Chart(gains)
        .mark_rule(aria=False)
        .encode(x=gain_x, y="lower:Q", y2="upper:Q", color=gain_colour)  # ty: ignore[unresolved-attribute]
    )
    levels = by_lead.filter(pl.col("arm").is_in(["B0", "L1", "climatology"]))
    level_lines = (
        alt.Chart(levels)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lead:N", sort=order, title=x_title),
            y=alt.Y("error:Q", title="Mean absolute error (% of capacity; smaller is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["B0", "L1", "climatology"],
                    range=[ARM_COLOURS["B0"], ARM_COLOURS["L1"], ARM_COLOURS["climatology"]],
                ),
                legend=alt.Legend(title="Arm"),
            ),
        )
    )
    title = "Gain from lagged power and the error of climatology, by lead-day"
    subtitle = [
        (
            "Top: each arm's mean absolute error minus B0's at the same lead-day. Below zero beats "
            "B0. Line: 95% interval from resampling whole months."
        ),
        "Bottom: the error of B0, L1 and climatology (the median for the month and hour, no fit).",
        "ENS-mean weather, primary setting, per plant. All rows are exploratory.",
        f"{LEAD_ZERO_TICK}: lead-day 0 is not usable for its early hours in a live service.",
    ]
    text.add(
        figure_id="6",
        plots="Gain over B0 against lead-day; error of B0, L1 and climatology",
        lines=[
            title,
            *subtitle,
            x_title,
            "Error minus B0's (pp of capacity)",
            "Mean absolute error (% of capacity; smaller is better)",
        ],
    )
    return figure(
        panels=[
            (gain_lines + gain_bars).properties(width=CONTENT_WIDTH_PX - 100, height=180),
            level_lines.properties(width=CONTENT_WIDTH_PX - 100, height=180),
        ],
        number=6,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )


def coverage_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart | None:
    """Figure 7: the share of rows inside the 10% to 90% interval, and its width.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure, or `None` if no interval predictions were saved.
    """
    path = tables / "coverage.parquet"
    if not path.exists():
        return None
    coverage = pl.read_parquet(path).with_columns(width=pl.col("width") * PERCENTAGE_POINTS)
    colours = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=["B0", "L1"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(title="Arm"),
    )
    sky = alt.X(
        "sky:N", sort=["clear", "mixed", "cloudy", "all"], title="Forecast sky (clear-sky index)"
    )
    share = (
        alt.Chart(coverage)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=sky,
            xOffset="arm:N",
            y=alt.Y("coverage:Q", title="Share inside the 10% to 90% interval"),
            color=colours,
        )
    ).properties(width=CONTENT_WIDTH_PX - 100, height=160)
    nominal = (
        alt.Chart(pl.DataFrame({"y": [0.8]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    width = (
        alt.Chart(coverage)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=sky,
            xOffset="arm:N",
            y=alt.Y("width:Q", title="Mean interval width (pp of capacity)"),
            color=colours,
        )
    ).properties(width=CONTENT_WIDTH_PX - 100, height=160)
    title = "Coverage and width of the 10% to 90% interval, with and without the lag"
    subtitle = [
        "Out-of-fold quantile predictions at lead-day 1, one seed. Dashed line: the nominal 80%.",
        "Descriptive: no planned contrast rests on this figure.",
    ]
    text.add(
        figure_id="7",
        plots="Coverage and width of the 10% to 90% interval for B0 and L1",
        lines=[
            title,
            *subtitle,
            "Share inside the 10% to 90% interval",
            "Mean interval width (pp of capacity)",
            "Arm: B0, L1",
        ],
    )
    return figure(
        panels=[share + nominal, width],
        number=7,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )


def main() -> int:
    """Write the figure text, and render the figures unless `--text-only` is given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-product", choices=WEATHER_PRODUCTS, default="ens_mean")
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument("--text-only", action="store_true", help="Write the figure text and stop.")
    arguments = parser.parse_args()
    product: WeatherProduct = arguments.weather_product
    root: Path = arguments.output_root
    directory = root / product
    tables = directory / f"tables_{product}"
    text = FigureText()
    figures = {
        "1_headline": headline_figure(tables=tables, text=text),
        "2_data": data_figure(root=root, product=product, text=text),
        "3_controls": control_figure(tables=tables, text=text),
        "4_models_work": models_work_figure(root=root, product=product, text=text),
        "5_sweep": sweep_figure(tables=tables, text=text),
        "6_by_lead": by_lead_figure(tables=tables, text=text),
    }
    coverage = coverage_figure(tables=tables, text=text)
    if coverage is not None:
        figures["7_coverage"] = coverage
    text_path = directory / f"figure_text_{product}.txt"
    text.write(path=text_path)
    _LOG.info("wrote %s", text_path)
    if arguments.text_only:
        return 0
    output = directory / f"figures_{product}"
    output.mkdir(exist_ok=True)
    refuse_to_overwrite(paths=[output / f"{name}.svg" for name in figures])
    for name, chart in figures.items():
        chart.save(str(output / f"{name}.svg"))
        _LOG.info("wrote %s", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
