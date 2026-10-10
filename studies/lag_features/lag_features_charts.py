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
    ARM_LABELS,
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
    "climatology": ocf.ENSEMBLE_LINE,
}
"""One colour per arm of the by-lead figure. Climatology is grey, which is not a data colour."""


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
    *, data: pl.DataFrame, y: str, x_title: str, colour: str, shape: str, legend_title: str | None
) -> alt.LayerChart:
    """Draw dots with interval lines against a zero rule, one row per `y` value.

    Args:
        data: Rows with `difference`, `lower`, `upper`, the `y` column, and `setting`.
        y: The column naming each row.
        x_title: The axis title.
        colour: The column giving the colour field.
        shape: The column giving the shape field.
        legend_title: The legend's title, or `None` for no legend (a chart of one series).

    Returns:
        The layered chart.
    """
    base = alt.Chart(data)
    rule = alt.Chart(pl.DataFrame({"zero": [0.0]})).mark_rule(color=ocf.BLACK_1).encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    lines = base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(f"{y}:N", title=None, sort=None),
        x=alt.X("lower:Q", title=x_title),
        x2="upper:Q",
        color=alt.Color(
            f"{colour}:N", legend=None if legend_title is None else alt.Legend(title=legend_title)
        ),
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
        label=pl.col("contrast")
        + ": "
        + pl.col("treatment")
        + " minus "
        + pl.col("reference")
        + " ("
        + pl.col("reference")
        + " "
        + (pl.col("reference_value") * PERCENTAGE_POINTS).round(2).cast(pl.String)
        + "%), "
        + pl.col("combined_verdict").fill_null("not both settings"),
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
        "Each row names its reference arm's own error, and the verdict from both settings.",
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
                data=rows,
                y="label",
                x_title=x_title,
                colour="setting",
                shape="setting",
                legend_title="Hyperparameter setting",
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


CONTROL_PAIRS: Final[tuple[tuple[str, str], ...]] = (("L1", "N1"), ("L1", "N2"), ("N2", "B0"))
"""The paired differences the controls panel draws: the lag against each control, and the true null
against the baseline (the column-count bias plus the noise floor)."""


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
        series=pl.lit("L1 minus B0"),
        difference=pl.col("difference") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
    )
    exploratory = pl.read_parquet(tables / "exploratory.parquet")
    pairs = pl.concat(
        [
            exploratory.filter(
                (pl.col("treatment") == treatment) & (pl.col("reference") == reference)
            )
            for treatment, reference in CONTROL_PAIRS
        ]
    ).with_columns(
        label=pl.col("treatment") + " minus " + pl.col("reference"),
        series=pl.lit("95% interval"),
        difference=pl.col("difference") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
    )
    top_title = (
        "Change in error, L1 minus B0 (percentage points of capacity; more negative is better)"
    )
    bottom_title = (
        "Change in error, first arm minus second (percentage points of capacity; "
        "more negative is better)"
    )
    recovered = _intervals_chart(
        data=control,
        y="label",
        x_title=top_title,
        colour="series",
        shape="series",
        legend_title=None,
    )
    controls = _intervals_chart(
        data=pairs,
        y="label",
        x_title=bottom_title,
        colour="series",
        shape="series",
        legend_title=None,
    )
    title = "Known-answer tests: a synthetic loss of power, and the lag arms' controls"
    subtitle = [
        (
            "Top: the same pipeline on a synthetic target in which power is multiplied by one "
            "minus the shift in a seeded random half of the calendar months. Dot: estimate. "
            "Line: 99% interval from resampling whole months."
        ),
        (
            "Bottom: N2 reads a random day outside the row's fold (a true null), N1 reads a day "
            "8 to 28 days earlier (the plant's slow level), L1 reads the latest whole day. "
            "Line: 95% interval."
        ),
        "Zero: no change. All rows are exploratory.",
    ]
    text.add(
        figure_id="3",
        plots="Positive-control recovery; paired differences of L1, N1, N2 and B0",
        lines=[title, *subtitle, top_title, bottom_title],
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
            predicted=(pl.col("actual") + pl.col("signed_error_capped_mw"))
            / pl.col("effective_capacity_mw"),
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


NO_FIT_COLOUR: Final[str] = ocf.ENSEMBLE_LINE
"""Arms that need no fit are grey, which is not a data colour."""


def sweep_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 5: the sweep ranked, with the no-fit references and the CRPS corrections.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    phase1 = pl.read_parquet(tables / "phase1.parquet").with_columns(
        label=pl.col("arm").replace(ARM_LABELS),
        error=pl.col("error") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
        kind=pl.when(pl.col("fitted"))
        .then(pl.lit("Fitted XGBoost arm"))
        .otherwise(pl.lit("No fit")),
    )
    baseline = float(phase1.filter(pl.col("arm") == "B0")["error"][0])
    order = phase1.sort("error")["label"].to_list()
    bars = (
        alt.Chart(phase1)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order, title=None),
            x=alt.X("error:Q", title="Mean absolute error (% of capacity; smaller is better)"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Fitted XGBoost arm", "No fit"], range=[ocf.DATA_BLUE, NO_FIT_COLOUR]
                ),
                legend=alt.Legend(title="Arm"),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [baseline]})).mark_rule(color=ocf.BRAND_ORANGE).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    panels = [(bars + rule).properties(width=CONTENT_WIDTH_PX - 120, height=420)]
    crps_title = "CRPS approximated from nine quantiles (% of capacity; smaller is better)"
    crps_rows = _crps_rows(tables=tables)
    if not crps_rows.is_empty():
        crps_bars = (
            alt.Chart(crps_rows)
            .mark_bar(color=ocf.DATA_BLUE)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("label:N", sort=crps_rows["label"].to_list(), title=None),
                x=alt.X("crps:Q", title=crps_title),
            )
            .properties(width=CONTENT_WIDTH_PX - 120, height=110)
        )
        panels.append(crps_bars)
    title = "A screen of the lag ideas, ranked by error, to generate hypotheses"
    subtitle = [
        "Top: mean absolute error of each arm at lead-day 1 over the first 10 calendar months.",
        (
            "Orange line: B0, the baseline with no lag. Climatology, persistence, R1s, R2 and R4 "
            "need no fit (R1s and R2 rescale B0's forecast, R4 clamps it at the ceiling)."
        ),
        (
            "Bottom: R4 and R5 are scored on the continuous ranked probability score, each beside "
            "B0 on the same rows, with one fitting seed."
        ),
        "All rows are exploratory. No pairwise comparison is made here.",
    ]
    text.add(
        figure_id="5",
        plots="Every arm's screening-month error, ranked; CRPS of R4 and R5 beside B0",
        lines=[
            title,
            *subtitle,
            "Mean absolute error (% of capacity; smaller is better)",
            crps_title,
            "Arm: Fitted XGBoost arm, No fit",
        ],
    )
    return figure(panels=panels, number=5, title=title, subtitle=subtitle, figure_planning=None)


def _crps_rows(*, tables: Path) -> pl.DataFrame:
    """Return the CRPS of R4 and R5 beside B0 on each one's own rows, in percentage points.

    Args:
        tables: The report's tables directory.

    Returns:
        `label` and `crps`, empty if the report saved neither table.
    """
    parts = []
    for name, correction in (("r4", "R4"), ("r5", "R5")):
        path = tables / f"{name}.parquet"
        if path.exists():
            parts.append(
                pl.read_parquet(path)
                .with_columns(
                    label=pl.when(pl.col("arm") == "B0")
                    .then(pl.lit(f"B0 (on {correction}'s rows)"))
                    .otherwise(pl.col("arm")),
                    crps=pl.col("crps") * PERCENTAGE_POINTS,
                )
                .select("label", "crps")
            )
    return pl.concat(parts).select("label", "crps") if parts else pl.DataFrame()


def by_lead_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 6: each arm's gain over B0 against lead-day, and the error of B0 and climatology.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    by_lead = pl.read_parquet(tables / "by_lead.parquet").with_columns(
        arm=pl.col("arm").replace(ARM_LABELS),
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
    gain_arms = ["L1", "W7", "Q30", ARM_LABELS["T1"], "N2"]
    gains = by_lead.filter(pl.col("arm").is_in(gain_arms))
    gain_colour = alt.Color(
        "arm:N",
        scale=alt.Scale(
            domain=gain_arms,
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


def lag_construction_figure(
    *, root: Path, product: WeatherProduct, text: FigureText
) -> alt.VConcatChart:
    """Figure 8: how every lag column is built for one example target day.

    The day is the plant-day, over all plants, with the most hours that hold all seven lags, and
    among those the highest mean power, so a rule and not an eye picks it.

    Args:
        root: The output root.
        product: The weather product.
        text: The text collector.

    Returns:
        The figure.
    """
    lag_columns = [f"lag_d{day}" for day in range(1, 8)]
    frame = pl.read_parquet(
        output_paths(root=root, product=product, lead_days=(FULL_SWEEP_LEAD_DAY,))["day1"]
    ).with_columns(
        day=(pl.col("time") - pl.duration(minutes=30)).dt.date(),
        share=pl.col("power_mw") / pl.col("effective_capacity_mw"),
    )
    complete = frame.drop_nulls([*lag_columns, "week_min", "week_max", "week_mean"])
    chosen = (
        complete.group_by("site", "day")
        .agg(hours=pl.len(), level=pl.col("share").mean())
        .sort("hours", "level", descending=True)
        .row(0, named=True)
    )
    example = complete.filter((pl.col("site") == chosen["site"]) & (pl.col("day") == chosen["day"]))
    example = example.with_columns(
        clock=(pl.col("time") - pl.duration(minutes=30)).dt.hour().cast(pl.Float64) + 0.5,
        **{
            c: pl.col(c) / pl.col("effective_capacity_mw")
            for c in (*lag_columns, "week_min", "week_max", "week_mean")
        },
    )
    long = example.select("clock", "share", *lag_columns).unpivot(
        index="clock", variable_name="series", value_name="value"
    )
    long = long.with_columns(
        kind=pl.col("series").replace_strict(
            {
                "share": "Target: measured power",
                "lag_d1": "L1: day 1 back",
            },
            default="CTX7: days 2 to 7 back",
        )
    )
    x = alt.X("clock:Q", title="Hour of the day (UTC, hour midpoint)")
    y = alt.Y("value:Q", title="Power (share of capacity)")
    lines = (
        alt.Chart(long)
        .mark_line(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=y,
            detail="series:N",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Target: measured power", "L1: day 1 back", "CTX7: days 2 to 7 back"],
                    range=[ocf.BLACK_1, ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title="Series"),
            ),
        )
    )
    band = (
        alt.Chart(example)
        .mark_area(opacity=0.25, color=ocf.DATA_GREEN)
        .encode(x="clock:Q", y="week_min:Q", y2="week_max:Q")  # ty: ignore[unresolved-attribute]
    )
    mean = (
        alt.Chart(example)
        .mark_line(color=ocf.DATA_GREEN, strokeDash=[4, 3])
        .encode(x="clock:Q", y="week_mean:Q")  # ty: ignore[unresolved-attribute]
    )
    title = "How each lag column is read for one target day"
    subtitle = [
        f"One example target day at plant {chosen['site']}, at the day-ahead lead, chosen by rule.",
        "The rule: the plant-day with the most hours holding all seven lags, then the most power.",
        (
            "Green band: W7's minimum to maximum over days 1 to 7 back, dashed: its mean. "
            "Each lag is the same clock hour on an earlier whole day, before the forecast issue."
        ),
    ]
    text.add(
        figure_id="8",
        plots="Power, L1, CTX7 lags and the W7 band for one example target day",
        lines=[
            title,
            *subtitle,
            "Hour of the day (UTC, hour midpoint)",
            "Power (share of capacity)",
        ],
    )
    return figure(
        panels=[(band + mean + lines).properties(width=CONTENT_WIDTH_PX - 100, height=260)],
        number=8,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )


def withheld_folds_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 9: which months the two-step arms' stage-1 models withhold, drawn as months.

    For scored fold `k`, a stage-1 prediction for an hour in fold `j` comes from a model that
    withheld folds `k` and `j`. The panel draws the months one plant has, for the hours of the
    middle fold, and for each scored fold `k` marks the months the stage-1 model trains on.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    folds = pl.read_parquet(tables / "folds.parquet")
    site = min(folds["site"].unique().to_list())
    months = folds.filter(pl.col("site") == site).sort("month")
    own = int(months["fold"].to_list()[len(months) // 2])
    records = [
        {
            "month": row["month"],
            "scored fold": f"Scored fold {scored}",
            "role": (
                "Withheld: scored fold"
                if row["fold"] == scored
                else "Withheld: predicted hour's fold"
                if row["fold"] == own
                else "Stage-1 training month"
            ),
        }
        for scored in range(5)
        for row in months.iter_rows(named=True)
    ]
    chart = (
        alt.Chart(pl.DataFrame(records))
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "month:O", title="Calendar month of the plant's rows", axis=alt.Axis(labelAngle=-90)
            ),
            y=alt.Y("scored fold:N", title=None),
            color=alt.Color(
                "role:N",
                scale=alt.Scale(
                    domain=[
                        "Stage-1 training month",
                        "Withheld: scored fold",
                        "Withheld: predicted hour's fold",
                    ],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title="Role of the month"),
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 120, height=150)
    )
    title = "The stage-1 models never see the months they predict or score"
    subtitle = [
        (
            f"Each row is one scored fold. Each column is a month of plant {site}'s rows, "
            "coloured by what the stage-1 model for an hour in the middle fold does with it."
        ),
        "An hour outside every fold is predicted by a model that withholds the scored fold alone.",
    ]
    text.add(
        figure_id="9",
        plots="Which months the stage-1 models withhold, per scored fold",
        lines=[title, *subtitle, "Calendar month of the plant's rows", "Role of the month"],
    )
    return figure(panels=[chart], number=9, title=title, subtitle=subtitle, figure_planning=None)


def by_plant_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 10: each arm's error at each plant, and its difference from B0 there.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    by_plant = pl.read_parquet(tables / "by_plant.parquet")
    base = by_plant.filter(pl.col("arm") == "B0").select("site", base=pl.col("error"))
    rows = (
        by_plant.join(base, on="site")
        .with_columns(
            label=pl.col("arm").replace(ARM_LABELS),
            error=pl.col("error") * PERCENTAGE_POINTS,
            gain=(pl.col("error") - pl.col("base")) * PERCENTAGE_POINTS,
        )
        .with_columns(text=pl.col("error").round(2).cast(pl.String))
    )
    order = rows.group_by("label").agg(pl.col("error").mean()).sort("error")["label"].to_list()
    heat = (
        alt.Chart(rows)
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", title="Plant"),
            y=alt.Y("label:N", sort=order, title=None),
            color=alt.Color(
                "gain:Q",
                title="Error minus B0's at the plant (pp)",
                scale=alt.Scale(scheme="redblue", reverse=True, domainMid=0),
            ),
        )
    )
    labels = (
        alt.Chart(rows)
        .mark_text(fontSize=9, aria=False)
        .encode(x="site:N", y=alt.Y("label:N", sort=order), text="text:N")  # ty: ignore[unresolved-attribute]
    )
    title = "Each arm's error at each plant, and its gain over B0 there"
    subtitle = [
        "Cell text: mean absolute error (% of capacity; smaller is better), all months.",
        "Cell colour: the arm's error minus B0's at the same plant (blue is better than B0).",
        "All rows are exploratory.",
    ]
    text.add(
        figure_id="10",
        plots="Per-plant error of every arm",
        lines=[title, *subtitle, "Plant", "Error minus B0's at the plant (pp)"],
    )
    return figure(
        panels=[(heat + labels).properties(width=CONTENT_WIDTH_PX - 220, height=380)],
        number=10,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )


def monthly_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 11: each calendar month's paired difference from B0.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    monthly = pl.read_parquet(tables / "monthly_differences.parquet").with_columns(
        difference=pl.col("difference") * PERCENTAGE_POINTS
    )
    chart = (
        alt.Chart(monthly)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:O", title="Calendar month", axis=alt.Axis(labelAngle=-90)),
            xOffset="contrast:N",
            y=alt.Y("difference:Q", title="Error minus B0's (pp of capacity; negative beats B0)"),
            color=alt.Color(
                "contrast:N",
                scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_GREEN]),
                legend=alt.Legend(title="Arm minus B0"),
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 100, height=220)
    )
    title = "Month by month, the paired difference from B0 at the day-ahead lead"
    subtitle = [
        "Each bar: the mean over plants, seeds and hours of one arm's error minus B0's in a month.",
        "Per-plant fits, primary setting. Below zero beats B0. All rows are exploratory.",
    ]
    text.add(
        figure_id="11",
        plots="Monthly paired differences of L1 and X from B0",
        lines=[
            title,
            *subtitle,
            "Calendar month",
            "Error minus B0's (pp of capacity; negative beats B0)",
        ],
    )
    return figure(panels=[chart], number=11, title=title, subtitle=subtitle, figure_planning=None)


def scope_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 12: B0 and L1 fitted per plant against fitted as one global model.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    scopes = pl.read_parquet(tables / "scope_comparison.parquet").with_columns(
        error=pl.col("error") * PERCENTAGE_POINTS
    )
    chart = (
        alt.Chart(scopes)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", title=None),
            yOffset="scope:N",
            x=alt.X("error:Q", title="Mean absolute error (% of capacity; smaller is better)"),
            color=alt.Color(
                "scope:N",
                scale=alt.Scale(
                    domain=["per-plant", "global"], range=[ocf.DATA_BLUE, ocf.DATA_PURPLE]
                ),
                legend=alt.Legend(title="One model per plant, or one global model"),
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 120, height=120)
    )
    title = "One global model against one model per plant, with and without the lag"
    subtitle = [
        "Mean absolute error at lead-day 1 on the rows both scopes score, primary setting.",
        "The two scopes cut their folds differently, so the comparison is descriptive.",
        "All rows are exploratory.",
    ]
    text.add(
        figure_id="12",
        plots="Per-plant against global error of B0 and L1",
        lines=[title, *subtitle, "Mean absolute error (% of capacity; smaller is better)"],
    )
    return figure(panels=[chart], number=12, title=title, subtitle=subtitle, figure_planning=None)


def fingerprint_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 13: the global fingerprint mini-sweep, with the leave-one-plant-out fits.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    sweep = pl.read_parquet(tables / "fingerprint.parquet").with_columns(
        difference=pl.col("error") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
        group=pl.when(pl.col("arm").str.starts_with("LOPO"))
        .then(pl.lit("Leave one plant out"))
        .otherwise(pl.lit("All plants in training")),
    )
    x_title = "Mean absolute error (% of capacity; smaller is better)"
    chart = _intervals_chart(
        data=sweep, y="arm", x_title=x_title, colour="group", shape="group", legend_title="Training"
    )
    title = "A fingerprint of each plant against the plant code, for one global model"
    subtitle = [
        "G-B0: no fingerprint. G-ID: plant code. G-FP: transfer function, ceiling, diurnal shape.",
        "G-L1 adds the lag. LOPO rows leave the scored plant out of training.",
        "Dot: estimate. Line: 95% interval from resampling whole months. All rows are exploratory.",
    ]
    text.add(
        figure_id="13",
        plots="Errors of G-B0, G-ID, G-L1, G-FP and the leave-one-plant-out fits",
        lines=[title, *subtitle, x_title, "Training: All plants in training, Leave one plant out"],
    )
    return figure(panels=[chart], number=13, title=title, subtitle=subtitle, figure_planning=None)


def importance_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart | None:
    """Figure 14: each arm's share of total gain by feature group, descriptive only.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure, or `None` if the report saved no importance table.
    """
    path = tables / "importance.parquet"
    if not path.exists():
        return None
    shares = pl.read_parquet(path).with_columns(
        label=pl.col("arm").replace(ARM_LABELS), text=pl.col("share").round(2).cast(pl.String)
    )
    heat = (
        alt.Chart(shares)
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("label:N", title="Arm"),
            y=alt.Y("group:N", title=None),
            color=alt.Color("share:Q", title="Mean share of gain", scale=alt.Scale(scheme="blues")),
        )
    )
    labels = (
        alt.Chart(shares)
        .mark_text(fontSize=9, aria=False)
        .encode(x="label:N", y="group:N", text="text:N")  # ty: ignore[unresolved-attribute]
    )
    title = "Where each arm's trees split, by feature group (descriptive)"
    subtitle = [
        "Share of total gain from a separate refit of the point model, mean over plants and folds.",
        "Gain is measured on training rows and splits credit between correlated columns by chance.",
        "Importance is never evidence that an input helps; the planned contrasts are the evidence.",
    ]
    text.add(
        figure_id="14",
        plots="Share of total gain by feature group for B0, L1, L2, S2 and X",
        lines=[title, *subtitle, "Arm", "Mean share of gain"],
    )
    return figure(
        panels=[(heat + labels).properties(width=CONTENT_WIDTH_PX - 220, height=300)],
        number=14,
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
    optional = {
        "7_coverage": coverage_figure(tables=tables, text=text),
        "14_importance": importance_figure(tables=tables, text=text),
    }
    figures |= {name: chart for name, chart in optional.items() if chart is not None}
    figures["8_lag_construction"] = lag_construction_figure(root=root, product=product, text=text)
    figures["9_withheld_folds"] = withheld_folds_figure(tables=tables, text=text)
    figures["10_by_plant"] = by_plant_figure(tables=tables, text=text)
    figures["11_monthly"] = monthly_figure(tables=tables, text=text)
    figures["12_per_plant_vs_global"] = scope_figure(tables=tables, text=text)
    figures["13_fingerprint"] = fingerprint_figure(tables=tables, text=text)
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
