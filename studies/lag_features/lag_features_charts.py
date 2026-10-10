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
import subprocess
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
    run_suffix,
    smoke_subsample,
)
from contracts.settings import PROJECT_ROOT
from fit_lag_arms import verify_manifest_inputs
from studies.charts import CONTENT_WIDTH_PX, figure
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("lag_features_charts")

NOMINAL_COVERAGE: Final[float] = 0.8
"""The share of hours a 10% to 90% interval should hold."""

PERCENTAGE_POINTS: Final[float] = 100.0
"""Losses are fractions of capacity; charts show percentage points."""

LEAD_ZERO_TICK: Final[str] = "0*"
"""How lead-day 0 is ticked on an axis. Each figure that shows it explains the asterisk: lead-day 0
is not usable for its early hours in a live service."""

HOUR_TICKS: Final[list[int]] = [0, 6, 12, 18, 24]
"""An hour-of-day axis is ticked at these hours and clipped to the hours with daylight."""

HOURS_PER_DAY: Final[int] = 24
"""Converts an hour of the day into a fraction of a day on a week's axis."""

WEEK_RULES: Final[tuple[str, ...]] = ("sunniest", "most variable", "cloudiest")
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

ARM_KEY: Final[dict[str, str]] = {
    "B0": "the baseline with no lag",
    "L1": (
        "B0 plus the power at the same hour on the latest whole day before the forecast was "
        "issued (two days before the target day for a day-ahead forecast)"
    ),
    "L2": "L1 plus that hour's weather forecast",
    "S2": ("a two-step forecast that adds a first-stage forecast and its residual at the lag hour"),
    "W7": "the minimum, maximum and mean of the same hour over the last 7 whole days",
    "Q30": "the 30-day 90th percentile and median of the same hour",
    "CTX7": "seven separate daily lags and their weather forecasts",
    "AN": (
        "the analogue ensemble, the power on the 5 recent days whose forecast sky was most similar"
    ),
    "TF": "the 30-day ratio of power to forecast irradiance (the transfer ratio)",
    "PC": "the ratio of power to CAMS satellite irradiance",
    "CK": "the plant's clipping ceiling",
    "RP": "the plant's recent energy relative to the other plants",
    "IM": "the morning's power on the forecast day",
    "S3": (
        "a two-step forecast that adds the first-stage forecast's mean residual over the last "
        "1, 7 and 30 days"
    ),
    "KS": "all of L1, IM, W7, Q30, TF, S3 and RP together",
    "N1": "a random day 8 to 28 days earlier (keeps the slow level)",
    "N2": "a random day from months not being scored (a null)",
    "N2-3": "three random-day lags, as wide as W7",
    "T1": "days since the record began (an interpolation bound)",
    "CL": "B0 plus the climatology as a column",
    "W7+CL": "W7 plus the climatology as a column",
    "B0xCL": "B0 blended half and half with the climatology, with no fit",
    "W7xCL": "W7 blended half and half with the climatology, with no fit",
    "climatology": (
        "the median power of the plant's calendar month and hour, from months outside the scored "
        "block, with no fit"
    ),
    "persistence": "the last observed hour's power when the forecast was issued, with no fit",
    "diurnal_persistence": (
        "the power at the target's hour on the last whole day before the issue day, with no fit"
    ),
    "R1s": "B0 rescaled by the last 7 days' observed over predicted energy, with no fit",
    "R2": (
        "B0 rescaled by the last 30 days' observed over predicted energy per clock hour, "
        "with no fit"
    ),
    "R4": "B0 clamped at the plant's clipping ceiling, with no fit",
    "R5": "B0 plus recent residual quantiles, with no fit",
    "G-B0": "one global model with no plant information",
    "G-ID": "one global model given a plant code",
    "G-L1": "one global model given L1",
    "G-FP": "one global model given the plant's TF, CK and daily shape (a fingerprint)",
    "LOPO": "leave one plant out: the scored plant is left out of training",
}
"""What each arm code means, in the wording every figure shares. A figure lists only the arms it
shows, through `arm_key`."""

SHARED_TERMS: Final[dict[str, str]] = {
    "lag": "A lag is the power at the same clock hour on an earlier whole day.",
    "lead-day 1": "Lead-day 1 is the forecast for the day after the day it is issued.",
    "out-of-fold": (
        "Out-of-fold means made by a model that never trained on that month: whole blocks of "
        "months (folds) are held out in turn."
    ),
    "seed": "One fitting seed is one training run.",
    "setting": (
        "The primary hyperparameter setting is the default XGBoost setting. The sensitivity "
        "setting is shallower and more regularised."
    ),
    "CRPS": "CRPS is the continuous ranked probability score.",
    "ENS mean": "ENS mean is the mean of the ECMWF ensemble weather forecast.",
    "era": "An era is a period during which one version of the weather forecast product ran.",
}
"""One sentence per shared term, so that a figure defining a term uses the shared wording."""


def arm_key(*arms: str) -> str:
    """Return the one-line arm key of the arms a figure shows.

    Args:
        arms: The arm codes the figure shows, in the order to list them.

    Returns:
        A line of the form `Arms. B0: the baseline with no lag. L1: ...`.

    Raises:
        KeyError: If an arm has no entry in `ARM_KEY`.
    """
    return "Arms. " + " ".join(f"{arm}: {ARM_KEY[arm]}." for arm in arms)


def read_frame(*, path: Path, smoke: bool) -> pl.DataFrame:
    """Read a built frame, subsampled as a smoke run subsamples it.

    Args:
        path: The frame's parquet file.
        smoke: Whether the output being drawn is a smoke run's.

    Returns:
        The frame.
    """
    frame = pl.read_parquet(path)
    return smoke_subsample(frame=frame) if smoke else frame


def claim(*, holds: bool, what: str) -> None:
    """Raise if a claim a figure's title makes does not hold in the tables it draws.

    A title states a finding, so a re-run that changes the tables must not leave a stale title.

    Args:
        holds: Whether the claim holds.
        what: The claim, for the message.

    Raises:
        ValueError: If the claim does not hold.
    """
    if not holds:
        msg = f"the figure's title claims that {what}, and the tables say otherwise"
        raise ValueError(msg)


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
    *,
    data: pl.DataFrame,
    y: str,
    x_title: str,
    colour: str,
    shape: str,
    legend_title: str | None,
    verdicts: pl.DataFrame | None = None,
    x_domain: list[float] | None = None,
) -> alt.LayerChart:
    """Draw dots with interval lines against a zero rule, one row per `y` value.

    Args:
        data: Rows with `difference`, `lower`, `upper`, the `y` column, and `setting`.
        y: The column naming each row.
        x_title: The axis title.
        colour: The column giving the colour field.
        shape: The column giving the shape field.
        legend_title: The legend's title, or `None` for no legend (a chart of one series).
        verdicts: Optional rows with the `y` column, `verdict` and `x` (where the text starts),
            drawn as a text mark to the right of each row's interval.
        x_domain: Optional lower and upper end of the x axis.

    Returns:
        The layered chart.
    """
    base = alt.Chart(data)
    rule = alt.Chart(pl.DataFrame({"zero": [0.0]})).mark_rule(color=ocf.BLACK_1).encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    lines = base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(f"{y}:N", title=None, sort=None, axis=alt.Axis(labelLimit=0)),
        x=alt.X(
            "lower:Q",
            title=x_title,
            scale=alt.Undefined if x_domain is None else alt.Scale(domain=x_domain),
        ),
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
    layers = rule + lines + dots
    if verdicts is not None:
        words = (
            alt.Chart(verdicts)
            .mark_text(align="left", dx=8, fontSize=11, aria=False, color=ocf.BLACK_1)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y(f"{y}:N", sort=None), x="x:Q", text="verdict:N"
            )
        )
        layers = layers + words
    return layers.properties(width=CONTENT_WIDTH_PX - 260, height=200)


def headline_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 1: the five planned contrasts with their 99% intervals.

    Args:
        tables: The report's tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    planned = pl.read_parquet(tables / "planned.parquet")
    primary = planned.filter(pl.col("setting") == "primary")
    missing = set(planned["contrast"].to_list()) - set(primary["contrast"].to_list())
    if missing:
        msg = f"contrasts {sorted(missing)} have no primary-setting row; Figure 1 cannot label them"
        raise ValueError(msg)
    primary = primary.select(
        "contrast",
        label=pl.Series(
            [
                f"{row['contrast']}: {row['treatment']} minus {row['reference']} "
                f"({row['reference']} {row['reference_value'] * PERCENTAGE_POINTS:.2f}%)"
                for row in primary.iter_rows(named=True)
            ]
        ),
    )
    rows = planned.join(primary, on="contrast").with_columns(
        difference=pl.col("difference") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
    )
    span = float(rows["upper"].max()) - float(rows["lower"].min())  # ty: ignore[invalid-argument-type]
    verdicts = (
        rows.group_by("label", maintain_order=True)
        .agg(
            verdict=pl.col("combined_verdict").first().fill_null("not both settings"),
            x=pl.col("upper").max(),
        )
        .with_columns(x=pl.col("x") + 0.02 * span)
    )
    domain = [float(rows["lower"].min()) - 0.05 * span, float(rows["upper"].max()) + 0.55 * span]  # ty: ignore[invalid-argument-type]
    x_title = (
        "Change in error, treatment minus reference (percentage points of capacity; "
        "more negative is better)"
    )
    claim(
        holds=bool(((rows["lower"] < 0) & (rows["upper"] > 0)).all()),
        what="every planned contrast's 99% interval includes zero",
    )
    title = "No planned contrast found lagged power changing day-ahead error significantly"
    subtitle = [
        (
            "Lagged power means a lag: the power at the same clock hour on an earlier whole day. "
            "Day-ahead means lead-day 1, the forecast for the day after the day it is issued. "
            "Each planned contrast, P1 to P5, is a treatment arm minus a reference arm, where an "
            "arm is one forecast configuration."
        ),
        (
            "P1: L1 minus B0. P2: AN minus L1, on the last 8 months only. P3: S2 minus L2. "
            "P4: L1 minus B0 scored on the continuous ranked probability score (CRPS), "
            "approximated from nine quantiles (0.1 to 0.9). P5: L1 minus B0 in one model shared "
            "by all plants. The other rows are scored on mean absolute error."
        ),
        (
            "Dot: estimate. Line: 99% interval from resampling whole months. Zero: no change. "
            "Each row names its reference arm's own error. The words at the right of a row give "
            "the verdict from both settings."
        ),
        f"Dots and lines are drawn at both settings. {SHARED_TERMS['setting']}",
        arm_key("B0", "L1", "L2", "S2", "AN"),
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
                verdicts=verdicts,
                x_domain=domain,
            )
        ],
        number=1,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )


def data_figure(
    *, root: Path, product: WeatherProduct, text: FigureText, smoke: bool
) -> alt.VConcatChart:
    """Figure 2: forecast irradiance against power, and the lag against the target.

    Args:
        root: The output root.
        product: The weather product.
        text: The text collector.
        smoke: Whether to read a smoke run's subsampled frame.

    Returns:
        The figure.
    """
    frame = read_frame(
        path=output_paths(root=root, product=product, lead_days=(FULL_SWEEP_LEAD_DAY,))["day1"],
        smoke=smoke,
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
        (
            "The lag is the power at the same clock hour on the latest whole day before the "
            "forecast was issued, which is two days before the target day for a day-ahead "
            "forecast."
        ),
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
    claim(
        holds=bool((control["upper"] >= 0).all()),
        what="no planted power loss is detected by L1",
    )
    claim(
        holds=bool(
            pairs.filter(pl.col("treatment") == "L1")
            .select(((pl.col("lower") <= 0) & (pl.col("upper") >= 0)).all())
            .item()
        ),
        what="L1 is not distinguishable from the random-lag controls",
    )
    labels = [f"{shift:.0%}" for shift in sorted(control["shift"].to_list())]
    shifts = f"{', '.join(labels[:-1])} or {labels[-1]}"
    title = "A single day's lag neither detects a planted power loss nor beats a random-day lag"
    subtitle = [
        (
            f"Top: the forecasts are re-fitted on a synthetic target in which measured power is "
            f"cut by {shifts} in 9 of the 18 scored months, chosen at random with a fixed seed "
            "(and in a random half of the other months). A forecast that used the lag well would "
            "gain more the bigger the cut. The interval is 99%, as for the planned contrasts."
        ),
        (
            "Bottom: the change in error from L1 compared with two random-day lags, and from a "
            "random-day lag compared with B0 (the change from adding any extra column). "
            "The interval is 95%, as for every exploratory contrast."
        ),
        (
            "Dot: estimate. Line: interval from resampling whole months. Zero: no change. "
            "Months being scored are the months the fitted forecasts are tested on. "
            "All rows are exploratory."
        ),
        arm_key("B0", "L1", "N1", "N2"),
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
    """Pick the three weeks drawn: the sunniest, most variable and cloudiest.

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
        "sunniest": weekly.sort("level")["week"][-1],
        "most variable": weekly.sort("spread")["week"][-1],
        "cloudiest": weekly.sort("level")["week"][0],
    }


def models_work_figure(
    *, root: Path, product: WeatherProduct, text: FigureText, smoke: bool
) -> alt.VConcatChart:
    """Figure 4: out-of-fold predictions against measured power for three weeks.

    Args:
        root: The output root.
        product: The weather product.
        text: The text collector.
        smoke: Whether the output being drawn is a smoke run's.

    Returns:
        The figure.
    """
    losses = pl.read_parquet(
        root / product / f"losses_{product}{run_suffix(smoke=smoke)}.parquet"
    ).filter(
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
                x=alt.X(
                    "day:Q", title="Day of the chosen week (1 to 7)", scale=alt.Scale(domain=[1, 8])
                ),
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
        (
            "Each row is one plant, and each group of six rows is one week. Black: measured "
            "power. Blue and orange: the forecasts of B0 and L1 for that power."
        ),
        (
            f"{SHARED_TERMS['out-of-fold']} {SHARED_TERMS['lead-day 1']} "
            "One fitting seed is one training run, and this is one of three."
        ),
        (
            "Weeks are chosen by rule, not by eye, from the weeks with data for all six plants: "
            "the sunniest (highest mean power across plants), the most variable between plants, "
            "and the cloudiest (lowest mean power)."
        ),
        arm_key("B0", "L1"),
    ]
    text.add(
        figure_id="4",
        plots="Three weeks of B0 and L1 forecasts against measured power",
        lines=[
            title,
            *subtitle,
            "Day of the chosen week (1 to 7)",
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
    error_title = "Mean absolute error (% of capacity; smaller is better)"
    bars = (
        alt.Chart(phase1)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order, title=None),
            x=alt.X("error:Q", title=error_title),
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
    panels = [(bars + rule).properties(width=CONTENT_WIDTH_PX - 180, height=420)]
    crps_title = (
        "Continuous ranked probability score (CRPS), approximated from nine quantiles "
        "(% of capacity; smaller is better)"
    )
    crps_rows = _crps_rows(tables=tables)
    if not crps_rows.is_empty():
        crps_bars = (
            alt.Chart(crps_rows)
            .mark_bar(color=ocf.DATA_BLUE)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("label:N", sort=crps_rows["label"].to_list(), title=None),
                x=alt.X("crps:Q", title=crps_title),
            )
            .properties(width=CONTENT_WIDTH_PX - 180, height=110)
        )
        panels.append(crps_bars)
    fitted = phase1.filter(pl.col("fitted")).sort("error")
    best = str(fitted["label"][0])
    b0_row = phase1.filter(pl.col("arm") == "B0").row(0, named=True)
    others = fitted.filter(pl.col("arm") != "B0")
    claim(
        holds=bool(
            (others["lower"] <= b0_row["upper"]).all() & (others["upper"] >= b0_row["lower"]).all()
        ),
        what="every arm's 95% interval overlaps B0's",
    )
    claim(
        holds=int((others["error"] >= b0_row["error"]).sum()) > others.height / 2,
        what="most arms were no better than B0",
    )
    claim(holds=best != "B0", what="an arm other than B0 had the lowest error")
    title = f"In the screening months no arm clearly beat B0, though {best} had the lowest error"
    arms_drawn = [*phase1.sort("error")["arm"].to_list(), "R5"]
    subtitle = [
        (
            "A screen, not a test: the 95% intervals of the arms overlap B0's, so no arm is shown "
            "to beat B0. Top: mean absolute error of each arm at lead-day 1 (the forecast for the "
            "day after the day it is issued) over the first 10 of the 18 scored months."
        ),
        (
            "Orange line: B0's error. Grey bars need no fit. Blue bars are fitted per plant, "
            "out-of-fold (made by a model that never trained on that month), with 3 training "
            "runs (seeds)."
        ),
        (
            "Bottom: R4 and R5 are scored on CRPS, each beside B0 on the same rows, "
            "with one training run."
        ),
        "All rows are exploratory. No pairwise comparison is made here.",
        arm_key(*arms_drawn),
    ]
    text.add(
        figure_id="5",
        plots="Every arm's screening-month error, ranked; CRPS of R4 and R5 beside B0",
        lines=[
            title,
            *subtitle,
            error_title,
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
    gain_title = "Error minus B0's (percentage points of capacity; negative beats B0)"
    level_title = "Mean absolute error (percentage points of capacity; smaller is better)"
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
            y=alt.Y("gain:Q", title=gain_title),
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
            y=alt.Y("error:Q", title=level_title),
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
    longest = int(by_lead["lead_day"].max())  # ty: ignore[invalid-argument-type]
    drawn = by_lead.filter(pl.col("arm").is_in(["L1", "W7", "Q30", "N2"]))
    best_gain = drawn.sort("difference_to_b0").row(0, named=True)
    claim(
        holds=best_gain["lead_day"] == longest,
        what="the largest gain over B0 is at the longest lead",
    )
    levels = by_lead.filter(pl.col("lead_day").is_in([10, longest]))
    claim(
        holds=bool(
            all(
                levels.filter((pl.col("lead_day") == lead) & (pl.col("arm") == "B0"))["error"][0]
                > levels.filter((pl.col("lead_day") == lead) & (pl.col("arm") == "climatology"))[
                    "error"
                ][0]
                for lead in (10, longest)
            )
        ),
        what="B0 is worse than climatology at lead-days 10 and the longest",
    )
    claim(
        holds=bool(
            drawn.filter((pl.col("arm") == "N2") & (pl.col("lead_day") == longest))["upper"][0] < 0
        ),
        what="the random-day null also gains at the longest lead",
    )
    title = (
        f"At lead-day {longest} B0 is worse than climatology; lags gain most, but so does a "
        "random-day lag (N2)"
    )
    subtitle = [
        (
            "Top: each arm's mean absolute error minus B0's at the same lead-day. Below zero beats "
            "B0. Line: 95% interval from resampling whole months."
        ),
        (
            "Bottom: the mean absolute error of B0, L1 and climatology. Lead-day is the number of "
            "days between the weather forecast run and the target day."
        ),
        (
            "A random-day lag (N2) gains at the longest lead too, so part of that gain is not "
            "specific to recent days. The weather forecast is the ENS mean, the mean of the ECMWF "
            "ensemble; fits are per plant, at the primary (default XGBoost) setting. "
            "All rows are exploratory."
        ),
        f"{LEAD_ZERO_TICK}: lead-day 0 is not usable for its early hours in a live service.",
        arm_key("B0", "L1", "W7", "Q30", "T1", "N2", "climatology"),
    ]
    text.add(
        figure_id="6",
        plots="Gain over B0 against lead-day; error of B0, L1 and climatology",
        lines=[
            title,
            *subtitle,
            x_title,
            gain_title,
            level_title,
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
        legend=alt.Legend(title="B0 (no lag) or L1 (one lag)"),
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
            y=alt.Y("width:Q", title="Mean interval width (percentage points of capacity)"),
            color=colours,
        )
    ).properties(width=CONTENT_WIDTH_PX - 100, height=160)
    overall = coverage.filter(pl.col("sky") == "all")
    claim(
        holds=bool((overall["coverage"] < NOMINAL_COVERAGE).all()),
        what="the interval holds less than its nominal share of hours",
    )
    claim(
        holds=float(overall["coverage"].max() - overall["coverage"].min()) < 0.02,  # ty: ignore[unsupported-operator]
        what="the lag changes the share by less than 2 points",
    )
    typical = float(overall["coverage"].mean())  # ty: ignore[invalid-argument-type]
    title = (
        f"The 10% to 90% interval holds about {typical:.0%} of hours rather than "
        f"{NOMINAL_COVERAGE:.0%}, with and without the lag"
    )
    subtitle = [
        (
            f"{SHARED_TERMS['out-of-fold']} {SHARED_TERMS['lead-day 1']} "
            "One fitting seed is one training run."
        ),
        (
            "Top: the share of hours whose measured power falls between the forecast's "
            "10% and 90% quantiles. Dashed line: the nominal 80%. Bottom: the mean distance "
            "between those quantiles. Forecast sky is the clear-sky index of the weather forecast."
        ),
        "Descriptive: no planned contrast rests on this figure.",
        arm_key("B0", "L1"),
    ]
    text.add(
        figure_id="7",
        plots="Coverage and width of the 10% to 90% interval for B0 and L1",
        lines=[
            title,
            *subtitle,
            "Share inside the 10% to 90% interval",
            "Mean interval width (percentage points of capacity)",
            "B0 (no lag) or L1 (one lag)",
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
    *, root: Path, product: WeatherProduct, text: FigureText, smoke: bool
) -> alt.VConcatChart:
    """Figure 8: how every lag column is built for one example target day.

    The day is the plant-day, over all plants, with the most hours that hold all seven lags, and
    among those the highest mean power, so a rule and not an eye picks it.

    Args:
        root: The output root.
        product: The weather product.
        text: The text collector.
        smoke: Whether the output being drawn is a smoke run's.

    Returns:
        The figure.
    """
    lag_columns = [f"lag_d{day}" for day in range(1, 8)]
    frame = read_frame(
        path=output_paths(root=root, product=product, lead_days=(FULL_SWEEP_LEAD_DAY,))["day1"],
        smoke=smoke,
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
    daylight = [float(example["clock"].min()) - 0.5, float(example["clock"].max()) + 0.5]  # ty: ignore[invalid-argument-type]
    x = alt.X(
        "clock:Q",
        title="Hour of the day (UTC, hour midpoint)",
        axis=alt.Axis(values=HOUR_TICKS),
        scale=alt.Scale(domain=daylight),
    )
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
        .encode(x=x, y="week_min:Q", y2="week_max:Q")  # ty: ignore[unresolved-attribute]
    )
    mean = (
        alt.Chart(example)
        .mark_line(color=ocf.DATA_GREEN, strokeDash=[4, 3])
        .encode(x=x, y="week_mean:Q")  # ty: ignore[unresolved-attribute]
    )
    title = "How each lag column is read for one target day"
    subtitle = [
        (
            f"One example target day at plant {chosen['site']}, at the day-ahead lead (the "
            "forecast for the day after the day it is issued). Each line or band is read at the "
            "same clock hour as the target hour, on an earlier whole day before the forecast "
            "was issued."
        ),
        (
            "Black line: the measured power on the target day. Orange line: L1, the latest whole "
            "day before the forecast was issued (day 1 back). Blue lines: CTX7's other six daily "
            "lags (days 2 to 7 back). Green band: W7's minimum to maximum over days 1 to 7 back. "
            "Green dashed line: W7's mean over those days."
        ),
        (
            "The example is chosen by rule, not by eye: the plant-day with the most hours that "
            "hold all seven lags, then the most power."
        ),
        arm_key("L1", "CTX7", "W7"),
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


ROLE_TRAINED: Final[str] = "Trained on"
ROLE_SCORED: Final[str] = "Skipped (scored block)"
ROLE_PREDICTED: Final[str] = "Skipped (predicted hours' block)"
"""The three things a first-stage model does with a month, as Figure 9's legend names them."""


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
                ROLE_SCORED
                if row["fold"] == scored
                else ROLE_PREDICTED
                if row["fold"] == own
                else ROLE_TRAINED
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
                    domain=[ROLE_TRAINED, ROLE_SCORED, ROLE_PREDICTED],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title="What the first-stage model does with the month"),
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 120, height=150)
    )
    title = "The first-stage forecasts never train on the months they are scored on"
    subtitle = [
        (
            "Some arms feed a first-stage XGBoost forecast into a second model. Each row is one "
            "scored block of months (a fold). Each cell is one month of plant "
            f"{site}, coloured by whether the first-stage model trained on it, or skipped it."
        ),
        (
            f"{ROLE_TRAINED}: the model learned from the month. {ROLE_SCORED}: the month is in "
            f"the block being scored. {ROLE_PREDICTED}: the first-stage forecast is made for "
            "hours in the middle block (here), so the model skips that block too."
        ),
        (
            "A first-stage forecast for an hour in no block is made by a model that skips the "
            "scored block alone."
        ),
        arm_key("S2", "S3", "KS"),
    ]
    text.add(
        figure_id="9",
        plots="Which months the stage-1 models withhold, per scored fold",
        lines=[
            title,
            *subtitle,
            "Calendar month of the plant's rows",
            "What the first-stage model does with the month",
            f"Legend: {ROLE_TRAINED}, {ROLE_SCORED}, {ROLE_PREDICTED}",
        ],
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
    arm_order = rows.group_by("arm").agg(pl.col("error").mean()).sort("error")["arm"].to_list()
    gain_title = "Error minus B0's at the plant (percentage points of capacity)"
    heat = (
        alt.Chart(rows)
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", title="Plant"),
            y=alt.Y("label:N", sort=order, title=None),
            color=alt.Color(
                "gain:Q",
                title=gain_title,
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
        (
            "Cell text: mean absolute error at lead-day 1 (the forecast for the day after the "
            "day it is issued), % of capacity, smaller is better, all months, per-plant fits."
        ),
        (
            "Cell colour: the arm's error minus B0's at the same plant, in percentage points of "
            "capacity (blue is better than B0). AN is the arm the shortlist rule picked from the "
            "first 10 months."
        ),
        "All rows are exploratory.",
        arm_key(*arm_order),
    ]
    text.add(
        figure_id="10",
        plots="Per-plant error of every arm",
        lines=[title, *subtitle, "Plant", gain_title],
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
    difference_title = "Error minus B0's (percentage points of capacity; negative beats B0)"
    chart = (
        alt.Chart(monthly)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:O", title="Calendar month", axis=alt.Axis(labelAngle=-90)),
            xOffset="contrast:N",
            y=alt.Y("difference:Q", title=difference_title),
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
        (
            "Each bar: the mean over plants, training runs (seeds) and hours of one arm's mean "
            "absolute error minus B0's in one calendar month, at lead-day 1 (the forecast for "
            "the day after the day it is issued)."
        ),
        (
            "Per-plant fits, primary (default XGBoost) setting. Below zero beats B0. "
            "AN is the arm the shortlist rule picked from the first 10 months. "
            "All rows are exploratory."
        ),
        arm_key("B0", "L1", "AN"),
    ]
    text.add(
        figure_id="11",
        plots="Monthly paired differences of L1 and AN from B0",
        lines=[title, *subtitle, "Calendar month", difference_title],
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
    wide = scopes.pivot(on="scope", index="arm", values="error")
    claim(
        holds=bool((wide["global"] > wide["per-plant"]).all()),
        what="the global model is less accurate than one model per plant",
    )
    title = "One model shared by all plants scored worse than one model per plant (descriptive)"
    subtitle = [
        (
            "Mean absolute error at lead-day 1 (the forecast for the day after the day it is "
            "issued) on the rows both versions score, primary (default XGBoost) setting. "
            "The shared (global) model is given no information about which plant a row "
            "belongs to."
        ),
        (
            "The two versions split the months into held-out blocks differently, so this "
            "comparison is descriptive. The fingerprint figure compares shared models with each "
            "other, with 95% intervals."
        ),
        "All rows are exploratory.",
        arm_key("B0", "L1"),
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
    errors = {row["arm"]: float(row["difference"]) for row in sweep.iter_rows(named=True)}
    seen_gain = errors["G-B0"] - errors["G-FP"]
    unseen_gain = errors["LOPO G-B0"] - errors["LOPO G-FP"]
    claim(
        holds=seen_gain > unseen_gain > 0,
        what="the fingerprint's gain is positive and smaller for an unseen plant",
    )
    title = (
        "Describing each plant to the shared model lowers its error, less so for an unseen plant"
    )
    subtitle = [
        (
            "Every row is one model shared by all plants, scored by mean absolute error at "
            "lead-day 1 (the forecast for the day after the day it is issued). The fingerprint "
            "(G-FP) describes each plant by its transfer ratio (TF), clipping ceiling (CK) and "
            "daily shape. An unseen plant is one left out of training (the LOPO rows)."
        ),
        (
            "A post hoc decomposition in the follow-up report attributes the gain to the transfer "
            "ratio, and finds the clipping ceiling does not help a plant the model has not seen."
        ),
        "Dot: estimate. Line: 95% interval from resampling whole months. All rows are exploratory.",
        arm_key("G-B0", "G-ID", "G-L1", "G-FP", "LOPO", "TF", "CK"),
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
            color=alt.Color(
                "share:Q", title="Mean share of split gain", scale=alt.Scale(scheme="blues")
            ),
        )
    )
    labels = (
        alt.Chart(shares)
        .mark_text(fontSize=9, aria=False)
        .encode(x="label:N", y="group:N", text="text:N")  # ty: ignore[unresolved-attribute]
    )
    title = "Where each arm's trees split, by feature group (descriptive)"
    subtitle = [
        (
            "Gain here is XGBoost's split importance: how much a column's splits reduce the "
            "training loss. It is not the error gain over B0 shown elsewhere. Each cell is the "
            "feature group's share of the arm's total split gain, averaged over plants and "
            "folds, from a separate refit of the arm's single-value (not quantile) forecast."
        ),
        "Gain is measured on training rows and splits credit between correlated columns by chance.",
        "Importance is never evidence that an input helps; the planned contrasts are the evidence.",
        arm_key(*sorted(shares["arm"].unique().to_list())),
    ]
    text.add(
        figure_id="14",
        plots="Share of total split gain by feature group for B0, L1, L2, S2 and AN",
        lines=[title, *subtitle, "Arm", "Mean share of split gain"],
    )
    return figure(
        panels=[(heat + labels).properties(width=CONTENT_WIDTH_PX - 220, height=300)],
        number=14,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )


# --- The post hoc follow-ups ----------------------------------------------------------------------

POST_HOC: Final[str] = "All rows are post hoc: written after the first run's results were known."
"""The subtitle line every follow-up figure ends with."""

KIND_COLOURS: Final[dict[str, str]] = {"month": ocf.DATA_BLUE, "step": ocf.BRAND_ORANGE}
"""The two positive controls' colours."""

KIND_LABELS: Final[dict[str, str]] = {"month": "Month-level", "step": "Plant steps"}
"""How a figure names each positive control."""


def followup_controls_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Follow-up figure A: the share of the oracle's gain each arm recovers, in both controls.

    Args:
        tables: The follow-ups' tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    controls = (
        pl.read_parquet(tables / "controls.parquet")
        .filter(pl.col("arm") != "O")
        .with_columns(
            control=pl.col("kind").replace(KIND_LABELS),
            shift_label=(pl.col("shift") * 100).round(0).cast(pl.Int32).cast(pl.String) + "%",
            interval=pl.when(pl.col("wholly_below_zero"))
            .then(pl.lit("Wholly below zero"))
            .otherwise(pl.lit("Includes zero")),
        )
        .drop_nulls("share_of_oracle_gain")
    )
    x_title = "Share of the oracle's gain over B0 recovered (0: none, 1: all of it)"
    order = (
        controls.group_by("arm")
        .agg(pl.col("share_of_oracle_gain").mean())
        .sort("share_of_oracle_gain")
    )["arm"].to_list()
    base = alt.Chart(controls)
    dots = base.mark_point(filled=True, size=90, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("arm:N", sort=order, title=None),
        x=alt.X("share_of_oracle_gain:Q", title=x_title),
        color=alt.Color(
            "control:N",
            scale=alt.Scale(domain=list(KIND_LABELS.values()), range=list(KIND_COLOURS.values())),
            legend=alt.Legend(title="Control (planted-loss test)"),
        ),
        shape=alt.Shape("shift_label:N", legend=alt.Legend(title="Shift (planted loss)")),
        opacity=alt.Opacity(
            "interval:N",
            scale=alt.Scale(
                domain=["Wholly below zero", "Includes zero"],
                range=[1.0, 0.3],
            ),
            legend=alt.Legend(title="Arm's 99% interval for its gain over B0"),
        ),
    )
    rules = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    mean_share = controls.group_by("arm").agg(pl.col("share_of_oracle_gain").mean())
    shares = dict(zip(mean_share["arm"], mean_share["share_of_oracle_gain"], strict=True))
    claim(
        holds=all(shares[arm] > shares["L1"] for arm in ("W7", "Q30", "AN", "TF", "PC")),
        what="W7, Q30, AN, TF and PC each recover more of the planted loss than L1",
    )
    title = "Multi-day and satellite inputs recover more of a planted power loss than one day's lag"
    subtitle = [
        (
            "A control is a planted-loss test: the forecasts are re-fitted on a synthetic target "
            "in which measured power is cut by the shift (5% or 10%). Month-level: the cut applies "
            "in 9 of the 18 scored months and a random half of the others. Plant steps: each "
            "plant's power is cut for a random 4 to 12 weeks."
        ),
        (
            "Each dot is one arm in one control at one shift size: its error gain over B0 divided "
            "by the oracle's gain, where the oracle is B0 given the true shift factor."
        ),
        (
            "Dashed lines: no gain (0) and the oracle's gain (1). Faint dots: zero lies inside the "
            "arm's 99% interval, so the gain is not clearly different from none."
        ),
        "No dot is drawn where the oracle's own 99% interval includes zero.",
        POST_HOC,
        arm_key("B0", *order),
    ]
    text.add(
        figure_id="A",
        plots="Share of the oracle's gain recovered, per arm, control and shift",
        lines=[
            title,
            *subtitle,
            x_title,
            "Control (planted-loss test): Month-level, Plant steps",
            "Shift (planted loss): 5%, 10%",
            "Arm's 99% interval for its gain over B0: wholly below zero, includes zero",
        ],
    )
    return figure(
        panels=[(rules + dots).properties(width=CONTENT_WIDTH_PX - 120, height=240)],
        number="A",
        title=title,
        subtitle=subtitle,
        figure_planning=None,
        label="Follow-up figure",
    )


LONG_LEAD_COLOURS: Final[dict[str, str]] = {
    "CL": ocf.DATA_BLUE,
    "W7+CL": ocf.BRAND_ORANGE,
    "B0xCL": ocf.DATA_PURPLE,
    "W7xCL": ocf.DATA_SKY,
    "climatology": ocf.ENSEMBLE_LINE,
    "W7": ocf.DATA_GREEN,
    "Q30": ocf.DATA_PURPLE_LIGHT,
    "N2": ocf.BLACK_1,
    "N2-3": ocf.DATA_BLUE_LIGHT,
}
"""One colour per long-lead arm. Grey is for climatology, the no-fit forecast."""

CLIMATOLOGY_ARMS: Final[tuple[str, ...]] = ("CL", "W7+CL", "B0xCL", "W7xCL", "climatology")
"""The long-lead arms that use climatology, drawn in the top panel."""

LAG_ARMS: Final[tuple[str, ...]] = ("W7", "Q30", "N2", "N2-3")
"""The long-lead arms that use lags and no climatology, drawn in the bottom panel."""


def followup_long_leads_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Follow-up figure B: the long-lead gains over B0, with climatology and its blends drawn.

    Args:
        tables: The follow-ups' tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    leads = pl.read_parquet(tables / "long_leads.parquet").with_columns(
        difference=pl.col("difference") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
        lead=pl.col("lead_day").cast(pl.String),
    )
    order = ["7", "10", "14"]
    y_title = "Error minus B0's (percentage points of capacity; negative beats B0)"
    x_title = "Lead-day (days between the run and the target day)"
    x = alt.X("lead:N", sort=order, title=x_title)

    def panel(arms: tuple[str, ...]) -> alt.LayerChart:
        rows = leads.filter(pl.col("arm").is_in(arms))
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(arms), range=[LONG_LEAD_COLOURS[arm] for arm in arms]),
            legend=alt.Legend(title="Arm"),
        )
        lines = (
            alt.Chart(rows)
            .mark_line(point=True, aria=False)
            .encode(x=x, y=alt.Y("difference:Q", title=y_title), color=colour)  # ty: ignore[unresolved-attribute]
        )
        bars = (
            alt.Chart(rows)
            .mark_rule(aria=False)
            .encode(x=x, y="lower:Q", y2="upper:Q", color=colour)  # ty: ignore[unresolved-attribute]
        )
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=ocf.BLACK_1).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        return (zero + lines + bars).properties(width=CONTENT_WIDTH_PX - 120, height=200)

    by_lead_diff = {(row["lead_day"], row["arm"]): row for row in leads.iter_rows(named=True)}
    for lead in (7, 10, 14):
        blend = by_lead_diff[(lead, "B0xCL")]["difference"]
        claim(
            holds=all(blend < by_lead_diff[(lead, arm)]["difference"] for arm in LAG_ARMS),
            what=f"the B0 and climatology blend beats every lag arm at lead-day {lead}",
        )
        cl = by_lead_diff[(lead, "CL")]
        claim(
            holds=cl["lower"] <= 0 <= cl["upper"],
            what="the 95% interval of CL, climatology as a column, includes zero",
        )
    title = "Blending B0 with climatology lowers long-lead error more than any lag arm does"
    subtitle = [
        (
            "Top: arms that use climatology, shown as the error minus B0's. The climatology of "
            "every top-panel arm is the median power of the plant's calendar month and hour from "
            "months outside the scored block (out-of-fold). Those months include later ones, "
            "which a live service would not have."
        ),
        (
            "Adding climatology as a column (CL) leaves the 95% interval across zero at every "
            "lead-day shown, so it does not clearly beat B0. Blending B0 with climatology, or W7 "
            "with climatology, has no fit."
        ),
        (
            "Bottom: lag arms. W7 and Q30 read recent days. N2 reads one random day from months "
            "not being scored, and N2-3 reads three, as wide as W7, so they show what a lag "
            "with no recent information gains. Line: 95% interval from resampling whole "
            "months. Zero is B0."
        ),
        POST_HOC,
        arm_key("CL", "W7+CL", "B0xCL", "W7xCL", "climatology", "W7", "Q30", "N2", "N2-3"),
    ]
    text.add(
        figure_id="B",
        plots="Long-lead gain over B0, climatology arms and lag arms",
        lines=[title, *subtitle, x_title, y_title, "Arm"],
    )
    return figure(
        panels=[panel(CLIMATOLOGY_ARMS), panel(LAG_ARMS)],
        number="B",
        title=title,
        subtitle=subtitle,
        figure_planning=None,
        label="Follow-up figure",
    ).resolve_scale(color="independent")


ERA_COLOURS: Final[dict[str, str]] = {
    "2025-10 onwards (8 months)": ocf.BLACK_1,
    "2025-10 to 2025-12": ocf.DATA_BLUE,
    "2026-02 onwards": ocf.BRAND_ORANGE,
}
"""The unselected months and each forecast-product era."""


def followup_unselected_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Follow-up figure C: every arm's gain over B0 on the 8 months the sweep never screened.

    Args:
        tables: The follow-ups' tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    rows = pl.read_parquet(tables / "unselected.parquet").with_columns(
        label=pl.col("arm").replace(ARM_LABELS),
        difference=pl.col("difference") * PERCENTAGE_POINTS,
        lower=pl.col("lower") * PERCENTAGE_POINTS,
        upper=pl.col("upper") * PERCENTAGE_POINTS,
    )
    order = (
        rows.filter(pl.col("era") == "2025-10 onwards (8 months)")
        .sort("difference")["label"]
        .to_list()
    )
    rows = rows.with_columns(
        label=pl.col("label").cast(pl.Enum(order)), era=pl.col("era").cast(pl.String)
    ).sort("label")
    x_title = "Error minus B0's (percentage points of capacity; negative beats B0)"
    eras = list(ERA_COLOURS)
    colour = alt.Color(
        "era:N",
        scale=alt.Scale(domain=eras, range=list(ERA_COLOURS.values())),
        legend=alt.Legend(title="Months scored (era)"),
    )
    shape = alt.Shape(
        "era:N",
        scale=alt.Scale(domain=eras, range=["circle", "square", "triangle-up"]),
        legend=alt.Legend(title="Months scored (era)"),
    )
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=0))
    y_offset = alt.YOffset("era:N", scale=alt.Scale(domain=eras))
    base = alt.Chart(rows)
    lines = base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        yOffset=y_offset,
        x=alt.X("lower:Q", title=x_title),
        x2="upper:Q",
        color=colour,
    )
    dots = base.mark_point(filled=True, size=50, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, yOffset=y_offset, x="difference:Q", color=colour, shape=shape
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=ocf.BLACK_1).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + lines + dots).properties(width=CONTENT_WIDTH_PX - 260, height=44 * len(order))
    overall = rows.filter(pl.col("era") == "2025-10 onwards (8 months)").sort("difference")
    top_three = set(overall["arm"].head(3).to_list())
    claim(
        holds=top_three == {"PC", "W7", "TF"} and "AN" not in top_three,
        what="PC, W7 and TF gained most and AN did not",
    )
    title = "On the 8 months not used for screening, PC, W7 and TF gained most over B0, AN did not"
    subtitle = [
        (
            "Mean absolute error minus B0's at lead-day 1 (the forecast for the day after the day "
            "it is issued), per-plant fits, primary (default XGBoost) setting. Dot: estimate. "
            "Line: 95% interval from resampling months. Zero is B0."
        ),
        (
            f"{SHARED_TERMS['era']} The black rows use all 8 months. The study has no 2026-01, "
            "so the later era starts at 2026-02. An era of fewer than 6 months has a wide "
            "interval."
        ),
        (
            "The sweep picked AN by the screening rule on the first 10 months, so its row here is "
            "out of sample."
        ),
        POST_HOC,
        arm_key("B0", *overall["arm"].to_list()),
    ]
    text.add(
        figure_id="C",
        plots="Per-arm gain over B0 on the unselected months, by era",
        lines=[title, *subtitle, x_title, "Months: all 8, 2025-10 to 2025-12, 2026-02 onwards"],
    )
    return figure(
        panels=[chart],
        number="C",
        title=title,
        subtitle=subtitle,
        figure_planning=None,
        label="Follow-up figure",
    )


def followup_hit_rate_figure(*, tables: Path, text: FigureText) -> alt.VConcatChart:
    """Follow-up figure D: the share of rows at or below each quantile, against the quantile level.

    Args:
        tables: The follow-ups' tables directory.
        text: The text collector.

    Returns:
        The figure.
    """
    hits = pl.read_parquet(tables / "hit_rates.parquet").filter(pl.col("month") == "all")
    y_title = "Share of rows with measured power at or below the quantile"
    x_title = "Quantile level of the forecast"
    diagonal = (
        alt.Chart(pl.DataFrame({"level": [0.0, 1.0], "hit_rate": [0.0, 1.0]}))
        .mark_line(color=ocf.BLACK_1, strokeDash=[4, 3])
        .encode(x="level:Q", y="hit_rate:Q")  # ty: ignore[unresolved-attribute]
    )
    lines = (
        alt.Chart(hits)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("level:Q", title=x_title, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("hit_rate:Q", title=y_title, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=["B0", "L1"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title="B0 (no lag) or L1 (one lag)"),
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 120, height=240)
    )
    low, high = float(hits["level"].min()), float(hits["level"].max())  # ty: ignore[invalid-argument-type]
    claim(
        holds=bool(
            (hits.filter(pl.col("level") == low)["hit_rate"] > low).all()
            and (hits.filter(pl.col("level") == high)["hit_rate"] < high).all()
        ),
        what="both tails' quantiles hit away from their levels, on the over-confident side",
    )
    title = (
        f"Quantile forecasts are too narrow: too many hours fall outside the {low:.0%} and "
        f"{high:.0%} quantiles"
    )
    subtitle = [
        (
            f"{SHARED_TERMS['out-of-fold']} {SHARED_TERMS['lead-day 1']} One fitting seed is "
            "one training run. All 18 scored months."
        ),
        (
            "A calibrated forecast has, say, 10% of hours at or below its 10% quantile. "
            "Dashed line: a calibrated forecast. Above it the quantile is too high, below it "
            "too low."
        ),
        POST_HOC,
        arm_key("B0", "L1"),
    ]
    text.add(
        figure_id="D",
        plots="Quantile hit rate against level for B0 and L1",
        lines=[title, *subtitle, x_title, y_title, "B0 (no lag) or L1 (one lag)"],
    )
    return figure(
        panels=[diagonal + lines],
        number="D",
        title=title,
        subtitle=subtitle,
        figure_planning=None,
        label="Follow-up figure",
    )


DOCS_ASSETS_DIR: Final[Path] = PROJECT_ROOT / "docs" / "studies" / "assets"
"""Where a render publishes each figure, optimised, under a name that carries its number."""

SVGO_COMMAND: Final[tuple[str, ...]] = (
    "npx",
    "svgo@4",
    "--multipass",
    "--precision=1",
    "--final-newline",
)
"""The optimisation `CLAUDE.md` asks for before a chart image is committed."""


def render_figures(
    *, figures: dict[str, alt.VConcatChart], output: Path, asset_prefix: str | None
) -> None:
    """Save every figure as SVG, and publish an optimised copy named by the figure's number.

    A figure's key is its number or letter, an underscore and a slug, such as `5_sweep`. The raw
    SVG goes to `output/<key>.svg`. Unless `asset_prefix` is `None`, svgo writes
    `docs/studies/assets/<asset_prefix><number>.svg`, which the page links.

    Args:
        figures: The figures by key.
        output: The folder for the raw SVGs.
        asset_prefix: The published file name's start, or `None` to publish nothing (a smoke run).

    Raises:
        FileExistsError: If a raw or published file already exists.
        subprocess.CalledProcessError: If svgo fails.
    """
    output.mkdir(exist_ok=True)
    published = {
        name: DOCS_ASSETS_DIR / f"{asset_prefix}{name.split('_', maxsplit=1)[0]}.svg"
        for name in figures
        if asset_prefix is not None
    }
    refuse_to_overwrite(paths=[*(output / f"{name}.svg" for name in figures), *published.values()])
    for name, chart in figures.items():
        raw = output / f"{name}.svg"
        chart.save(str(raw))
        _LOG.info("wrote %s", raw)
        if name in published:
            DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [*SVGO_COMMAND, "--input", str(raw), "--output", str(published[name])],
                check=True,
            )
            _LOG.info("published %s", published[name])


def draw_followups(*, root: Path, smoke: bool, text_only: bool) -> int:
    """Write the follow-up figures' text, and render them unless `text_only`.

    Args:
        root: The output root.
        smoke: Whether to draw a smoke run's follow-up tables.
        text_only: Whether to write the figure text and stop.

    Returns:
        0.
    """
    suffix = run_suffix(smoke=smoke)
    directory = root / "ens_mean" / "followups"
    tables = directory / f"tables_followups_ens_mean{suffix}"
    text = FigureText()
    figures = {
        "A_controls": followup_controls_figure(tables=tables, text=text),
        "B_long_leads": followup_long_leads_figure(tables=tables, text=text),
        "C_unselected_months": followup_unselected_figure(tables=tables, text=text),
        "D_hit_rates": followup_hit_rate_figure(tables=tables, text=text),
    }
    text_path = directory / f"figure_text_followups_ens_mean{suffix}.txt"
    text.write(path=text_path)
    _LOG.info("wrote %s", text_path)
    if text_only:
        return 0
    render_figures(
        figures=figures,
        output=directory / f"figures_followups_ens_mean{suffix}",
        asset_prefix=None if smoke else "lag_features_followup_figure_",
    )
    return 0


def main() -> int:
    """Write the figure text, and render the figures unless `--text-only` is given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-product", choices=WEATHER_PRODUCTS, default="ens_mean")
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument("--text-only", action="store_true", help="Write the figure text and stop.")
    parser.add_argument("--smoke", action="store_true", help="Draw a `--smoke` run's output.")
    parser.add_argument(
        "--followups", action="store_true", help="Draw the post hoc follow-up figures instead."
    )
    arguments = parser.parse_args()
    if arguments.followups:
        return draw_followups(
            root=arguments.output_root, smoke=arguments.smoke, text_only=arguments.text_only
        )
    product: WeatherProduct = arguments.weather_product
    smoke: bool = arguments.smoke
    suffix = run_suffix(smoke=smoke)
    root: Path = arguments.output_root
    directory = root / product
    tables = directory / f"tables_{product}{suffix}"
    verify_manifest_inputs(root=root, product=product, smoke=smoke)
    text = FigureText()
    figures = {
        "1_headline": headline_figure(tables=tables, text=text),
        "2_data": data_figure(root=root, product=product, text=text, smoke=smoke),
        "3_controls": control_figure(tables=tables, text=text),
        "4_models_work": models_work_figure(root=root, product=product, text=text, smoke=smoke),
        "5_sweep": sweep_figure(tables=tables, text=text),
        "6_by_lead": by_lead_figure(tables=tables, text=text),
    }
    optional = {
        "7_coverage": coverage_figure(tables=tables, text=text),
        "14_importance": importance_figure(tables=tables, text=text),
    }
    figures |= {name: chart for name, chart in optional.items() if chart is not None}
    figures["8_lag_construction"] = lag_construction_figure(
        root=root, product=product, text=text, smoke=smoke
    )
    figures["9_withheld_folds"] = withheld_folds_figure(tables=tables, text=text)
    figures["10_by_plant"] = by_plant_figure(tables=tables, text=text)
    figures["11_monthly"] = monthly_figure(tables=tables, text=text)
    figures["12_per_plant_vs_global"] = scope_figure(tables=tables, text=text)
    figures["13_fingerprint"] = fingerprint_figure(tables=tables, text=text)
    text_path = directory / f"figure_text_{product}{suffix}.txt"
    text.write(path=text_path)
    _LOG.info("wrote %s", text_path)
    if arguments.text_only:
        return 0
    render_figures(
        figures=figures,
        output=directory / f"figures_{product}{suffix}",
        asset_prefix=None if smoke or product != "ens_mean" else "lag_features_figure_",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
