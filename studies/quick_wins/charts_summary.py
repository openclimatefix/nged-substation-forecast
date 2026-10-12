# ruff: noqa: E501, PLR0915
"""Draw Figures 1 to 6 of the quick-wins study: the leaderboard, the contrasts, and calibration.

Every number is read from the report tables written by `report_quick_wins.py` and from the
calibration tables written by `calibrate.py`. The only quantities computed here are the two
baselines' levels (manual heuristic and climatology) on the rows they share with the arms, and the
per-series change of the chosen cleaning rule in Figure 5.

The text of every figure (title, subtitle, panel titles, axis titles, legend entries, annotations)
is written to `studies/quick_wins/figure_text.md` before any chart is rendered, so a
reviewer can read it first. Another script writes Figures 7 onward into the same file, so this
script replaces only its own `## Figure N` sections.

    uv run python studies/quick_wins/charts_summary.py [--fold ...] [--seeds 0 1 2]
"""

import argparse
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import altair as alt
import arms as arm_definitions
import month_bootstrap
import plotting.ocf_theme as ocf
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import PROJECT_ROOT, Settings

from studies import charts

ASSET_DIR: Final[Path] = PROJECT_ROOT / "docs" / "studies" / "assets"
"""Where the SVG files and the figure-text file are written."""

TEXT_FILE: Final[Path] = Path(__file__).parent / "figure_text.md"
"""The text of every figure, one `## Figure N` section each, for review before rendering."""

SVGO_COMMAND: Final[tuple[str, ...]] = (
    "npx",
    "svgo@4",
    "--multipass",
    "--precision=1",
    "--final-newline",
)
"""The command `CLAUDE.md` prescribes for shrinking an exported chart."""

FIGURE_NUMBERS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
"""The figures this script owns."""

PERCENT: Final[float] = 100.0
"""Converts a fraction of capacity into a percentage of capacity."""

MID_GREY: Final[str] = "#8C8C8C"
"""The grey of the reference arms: references are drawn in grey, per the dataviz skill."""

FAMILY_ORDER: Final[tuple[str, ...]] = (
    "reference",
    "cleaning",
    "features",
    "training",
    "quantile",
    "combination",
)
FAMILY_LABELS: Final[dict[str, str]] = {
    "reference": "Production (reference)",
    "cleaning": "Round 0: cleaning",
    "features": "Round 1: columns shown",
    "training": "Round 1: training",
    "quantile": "Quantile training (post hoc)",
    "combination": "Round 2: combinations",
}
FAMILY_COLOURS: Final[dict[str, str]] = {
    "reference": MID_GREY,
    "cleaning": ocf.DATA_SKY,
    "features": ocf.DATA_BLUE,
    "training": ocf.DATA_PURPLE,
    "quantile": ocf.DATA_DEEP_TEAL,
    "combination": ocf.BRAND_ORANGE,
}
"""The colour of each kind of arm; the colour carries the kind and the row label names the arm."""

ARM_LABELS: Final[dict[str, str]] = {
    "production_colsample0.8": "Production as stored",
    "production": "Production (all columns per tree)",
    "production_cleaned": "Production with cleaning",
    "clean_first_1_months": "Drop first month of each series",
    "clean_first_2_months": "Drop first 2 months",
    "clean_first_3_months": "Drop first 3 months",
    "clean_isolated_zeros": "Drop isolated zeros",
    "clean_spikes": "Drop single-reading spikes",
    "clean_commissioning_ramp": "Drop commissioning ramp",
    "clean_new_meter_settling": "Drop new-meter settling",
    "clean_pv_dropouts": "Drop solar and wind dropouts",
    "clean_switching_days": "Drop switching days",
    "clean_combined": "Four cleaning rules together",
    "a1_generators_without_lags": "Generators without power lags",
    "a2_per_type_physics": "Per-type physics columns",
    "a3_lean_generators": "Lean columns for generators",
    "a3w_lean_wind_only": "Lean columns for wind only",
    "c1_lead_time": "Forecast lead-time column",
    "c2_early_stopping": "Early stopping",
    "c3_recency_weights": "Recency weights",
    "c4_absolute_error": "Absolute-error loss",
    "c7_calendar": "Calendar columns for demand",
    "d1_min_child_weight_20": "Minimum child weight 20",
    "d2_pooled_by_type": "One model per series type",
    "d3_pooled_with_site_id": "Per type, with site identifier",
    "g1_heuristic_inputs": "Manual heuristic's inputs as columns",
    "e1_quantile": "Quantile training, 9 levels",
    "r2_e1_c1_a3w_d3": "Quantile, lead, wind, pooled",
    "r2_c4_c2": "Abs. loss, stop",
    "r2_c4_c1": "Abs. loss, lead time",
    "r2_c4_c2_c1": "Abs. loss, stop, lead time",
    "r2_c4_c2_c1_a3w": "Abs. loss, stop, lead, wind",
    "r2_c4_c2_c1_a3w_d3": "Abs. loss, stop, lead, wind, pooled",
    "r2_c4_c1_a3w_d3": "Abs. loss, lead, wind, pooled",
}
"""A short label for every arm, at most 40 characters so a row label stays on one line."""

ARM_FAMILIES: Final[dict[str, str]] = {
    name: "quantile" if arm.quantile_levels else arm.family
    for name, arm in arm_definitions.ALL_ARMS.items()
}
"""The kind of each arm: reference, cleaning, features, training, quantile, or combination.

Every arm trained on quantiles is in the `quantile` family, whether it is a round 1 idea or a
round 2 combination, because its CRPS is scored on a different footing (see `ROUND2_LEGEND`).
"""

LEADERBOARD_EXCLUDED: Final[frozenset[str]] = frozenset(
    {"clean_new_meter_settling"}
    | {name for name in arm_definitions.ALL_ARMS if name.startswith("f1_")}
)
"""Arms left off Figures 1 to 4: the chosen cleaning rule is the control, and the learning-curve arms
have their own figure (Figure 14)."""

ROUND2_LEGEND: Final[str] = (
    "Round 2 labels list the ideas combined: abs. loss is the absolute-error loss, stop is "
    "early stopping, lead is the forecast lead-time column, wind is the lean wind-farm columns, "
    "pooled is one model per series type with a site identifier, and quantile is training on nine "
    "quantile levels with the pinball loss (the quantile arms' CRPS is scored on 51 members x 9 "
    "levels samples)."
)

PLANNED_ARMS: Final[frozenset[str]] = frozenset(
    {
        "a1_generators_without_lags",
        "a2_per_type_physics",
        "c1_lead_time",
        "c2_early_stopping",
        "c3_recency_weights",
        "c4_absolute_error",
        "c7_calendar",
        "clean_first_1_months",
        "clean_first_2_months",
        "clean_first_3_months",
        "clean_isolated_zeros",
        "clean_spikes",
        "clean_commissioning_ramp",
    }
)
"""The arms written into the study plan before any result existed."""

ROUND1_ARMS: Final[tuple[str, ...]] = (
    "a1_generators_without_lags",
    "a2_per_type_physics",
    "a3_lean_generators",
    "a3w_lean_wind_only",
    "c1_lead_time",
    "c2_early_stopping",
    "c3_recency_weights",
    "c4_absolute_error",
    "c7_calendar",
    "d1_min_child_weight_20",
    "d2_pooled_by_type",
    "d3_pooled_with_site_id",
    "g1_heuristic_inputs",
    "e1_quantile",
)
ROUND2_LADDER: Final[tuple[str, ...]] = (
    "c4_absolute_error",
    "r2_c4_c2",
    "r2_c4_c1",
    "r2_c4_c2_c1",
    "r2_c4_c2_c1_a3w",
    "r2_c4_c2_c1_a3w_d3",
    "r2_c4_c1_a3w_d3",
    "r2_e1_c1_a3w_d3",
)
"""The round 1 winner, then the round 2 arms, in the order the ladder adds ideas."""

CLEANING_ARMS: Final[tuple[str, ...]] = (
    "clean_first_1_months",
    "clean_first_2_months",
    "clean_first_3_months",
    "clean_isolated_zeros",
    "clean_spikes",
    "clean_commissioning_ramp",
    "clean_new_meter_settling",
    "clean_pv_dropouts",
    "clean_switching_days",
    "clean_combined",
)
BY_TYPE_ARMS: Final[tuple[str, ...]] = (
    "c4_absolute_error",
    "c2_early_stopping",
    "c1_lead_time",
    "a1_generators_without_lags",
    "a3_lean_generators",
    "a3w_lean_wind_only",
    "d2_pooled_by_type",
    "d3_pooled_with_site_id",
    "g1_heuristic_inputs",
    "e1_quantile",
)
"""The ideas Figure 3 breaks down by series type, before the best round 2 arm is appended."""

CHOSEN_RULE: Final[str] = "new_meter_settling"
CHOSEN_RULE_ARM: Final[str] = "clean_new_meter_settling"

TYPE_LABELS: Final[dict[str, str]] = {
    "PV": "Solar farms",
    "Wind": "Wind farms",
    "Disaggregated Demand": "Demand, disaggregated",
    "Raw Flow": "Raw flow",
    "BESS": "Battery storage",
    "Biofuel": "Biofuel",
}
TYPE_SHORT: Final[dict[str, str]] = {
    "PV": "PV",
    "Wind": "Wind",
    "Disaggregated Demand": "Demand",
    "Raw Flow": "Raw flow",
    "BESS": "BESS",
    "Biofuel": "Biofuel",
}
FIGURE3_TYPES: Final[tuple[str, ...]] = ("PV", "Wind", "Disaggregated Demand", "Raw Flow")
FIGURE6_TYPES: Final[tuple[str, ...]] = (
    "all types",
    "PV",
    "Wind",
    "Disaggregated Demand",
    "Raw Flow",
    "BESS",
    "Biofuel",
)

BAND_ALL: Final[str] = "all"
BANDS: Final[tuple[str, ...]] = ("all", "days 0 to 2", "days 3 to 10")
BAND_LABELS: Final[dict[str, str]] = {
    "all": "All leads",
    "days 0 to 2": "Days 0 to 2",
    "days 3 to 10": "Days 3 to 10",
}
BAND_COLOURS: Final[tuple[str, ...]] = (ocf.BLACK_1, ocf.BRAND_ORANGE, ocf.DATA_BLUE)

DOMAIN_STEP: Final[float] = 0.5
"""Axis ranges are rounded outwards to a multiple of this many percentage points."""

DUMMY_FAMILY: Final = "weather model"
"""`interval_panel` wants a product family on every row; colours here come from the condition."""

ROW_STEP_PX: Final[int] = 18
"""The height of a leaderboard row in pixels."""

POST_HOC_LINE: Final[str] = (
    "(post hoc) marks an arm or rule chosen after seeing earlier results; a bold (planned) row was "
    "written into the study plan beforehand."
)


@dataclass
class FigureText:
    """Everything a reader sees in a figure besides the marks, kept for review before rendering."""

    number: int
    plots: str
    title: str
    subtitle: list[str]
    panel_titles: list[str] = field(default_factory=list)
    axis_titles: list[str] = field(default_factory=list)
    legend: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)

    def markdown(self) -> str:
        """Return the figure's `## Figure N` section."""
        lines = [f"## Figure {self.number}", "", f"Plots: {self.plots}", ""]
        lines += [f"Title: Figure {self.number}: {self.title}", "", "Subtitle lines:"]
        lines += [f"- {line}" for line in self.subtitle]
        for heading, items in (
            ("Panel titles", self.panel_titles),
            ("Axis titles", self.axis_titles),
            ("Legend entries", self.legend),
            ("Annotations", self.annotations),
        ):
            if items:
                lines += ["", f"{heading}:"]
                lines += [f"- {item}" for item in items]
        return "\n".join(lines) + "\n"


@dataclass
class Figure:
    """A figure's text and the function that draws it once the text is on disk."""

    text: FigureText
    draw: Callable[[], alt.TopLevelMixin]


@dataclass
class Context:
    """The tables and wording every figure shares."""

    fold: str
    seeds: list[int]
    levels: pl.DataFrame
    contrasts: pl.DataFrame
    calibration: pl.DataFrame | None
    calibration_seed: int | None
    calibration_runs: int
    scope: str
    n_series: int
    series_counts: dict[str, int]
    types: pl.DataFrame
    capacity: pl.DataFrame
    baselines: dict[str, dict[str, float]]
    best_round2: str
    seeds_text: str


# ---------------------------------------------------------------------------------------------
# Reading the tables
# ---------------------------------------------------------------------------------------------


def _read_report_table(*, kind: str, fold: str, seeds: list[int]) -> pl.DataFrame:
    """Read `levels` or `contrasts`, preferring the tag of the requested seeds."""
    folder = inputs.OUTPUT_ROOT / "report" / fold
    tags = ["seeds" + "".join(str(seed) for seed in seeds), f"seed{seeds[0]}"]
    for tag in tags:
        path = folder / f"{kind}_{tag}.parquet"
        if path.exists():
            return pl.read_parquet(path)
    msg = f"no {kind} table for seeds {seeds} in {folder}; run report_quick_wins.py first"
    raise SystemExit(msg)


def _read_calibration(*, fold: str, seeds: list[int]) -> tuple[pl.DataFrame | None, int | None]:
    """Read the calibration table of the first requested seed that has one."""
    folder = inputs.OUTPUT_ROOT / "calibration" / fold
    for seed in seeds:
        path = folder / f"calibration_seed{seed}.parquet"
        if path.exists():
            return pl.read_parquet(path), seed
    return None, None


def _month_year(*, moment: datetime) -> str:
    return moment.strftime("%B %Y")


def _scope(*, fold: str) -> str:
    """Name the runs and the months the scores cover."""
    window = inputs.window_for(fold_id=fold)
    runs = len(inputs.sample_init_times(window=window))
    return (
        f"{runs} weekly ECMWF ensemble (ENS) forecast runs, scored from "
        f"{_month_year(moment=window.val_start)} to {_month_year(moment=window.val_end)}"
    )


def _baseline_levels(*, capacity: pl.DataFrame, fold: str, seeds: list[int]) -> dict[str, dict]:
    """Score the two naive baselines on the rows they share with the arms.

    The baselines were scored once, on the stored production run, so they exist for fold A only.
    Each level is the mean over series of each series' mean, as `month_bootstrap` computes it, on
    the rows the baseline and production with cleaning both scored.
    """
    folder = inputs.OUTPUT_ROOT / "production_arm" / "colsample0.8_seed0" / "report"
    reference_path = (
        inputs.OUTPUT_ROOT / "arms" / fold / f"production_cleaned_seed{seeds[0]}" / "scores.parquet"
    )
    if fold != inputs.FOLD_A or not reference_path.exists():
        return {}
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    reference = pl.read_parquet(reference_path).select(keys)
    found: dict[str, dict[str, float]] = {}
    for name in ("manual_heuristic", "climatology"):
        path = folder / f"scores_stored_{name}.parquet"
        if not path.exists():
            continue
        scored = (
            pl.read_parquet(path)
            .join(capacity.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
            .with_columns(
                crps_norm=pl.col("crps") / pl.col("effective_capacity_mw"),
                abs_error_norm=pl.col("abs_error") / pl.col("effective_capacity_mw"),
            )
            .join(reference, on=keys)
        )
        found[name] = {
            metric: month_bootstrap.level_interval(
                table=month_bootstrap.monthly_table(scores=scored, metric=column)
            ).estimate
            * PERCENT
            for metric, column in (("crps", "crps_norm"), ("mae", "abs_error_norm"))
        }
    return found


def build_context(*, fold: str, seeds: list[int]) -> Context:
    """Read every table the figures need."""
    settings = Settings()
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))
    capacity = pl.DataFrame(inputs.effective_capacity(settings=settings)).filter(
        pl.col("time_series_id").is_in(series_ids)
    )
    levels = _read_report_table(kind="levels", fold=fold, seeds=seeds)
    contrasts = _read_report_table(kind="contrasts", fold=fold, seeds=seeds)
    calibration, calibration_seed = _read_calibration(fold=fold, seeds=seeds)
    runs = 0
    best_round2 = ""
    round2 = levels.filter(
        pl.col("arm").str.starts_with("r2_"),
        pl.col("group") == "all types",
        pl.col("lead_band") == BAND_ALL,
        pl.col("metric") == "crps",
    ).sort("value")
    if not round2.is_empty():
        best_round2 = round2["arm"][0]
    if calibration is not None:
        calibrated_round2 = round2.filter(
            pl.col("arm").is_in(calibration["arm"].unique().to_list())
        )
        calibration = calibration.filter(
            pl.col("arm").is_in(["production_cleaned", *calibrated_round2["arm"][:1].to_list()])
        )
    if calibration is not None and calibration_seed is not None:
        first = calibration["arm"][0]
        raw = pl.read_parquet(
            inputs.OUTPUT_ROOT
            / "calibration"
            / fold
            / f"{first}_seed{calibration_seed}"
            / "raw.parquet"
        )
        runs = raw["power_fcst_init_time"].n_unique()
    counts = types.group_by("time_series_type").len()
    seeds_text = (
        f"average of seeds {', '.join(str(seed) for seed in seeds)}"
        if len(seeds) > 1
        else f"seed {seeds[0]}"
    )
    return Context(
        fold=fold,
        seeds=seeds,
        levels=levels,
        contrasts=contrasts,
        calibration=calibration,
        calibration_seed=calibration_seed,
        calibration_runs=runs,
        scope=_scope(fold=fold),
        n_series=types.height,
        series_counts=dict(zip(counts["time_series_type"], counts["len"], strict=True)),
        types=types,
        capacity=capacity,
        baselines=_baseline_levels(capacity=capacity, fold=fold, seeds=seeds),
        best_round2=best_round2,
        seeds_text=seeds_text,
    )


# ---------------------------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------------------------


def _level_rows(*, context: Context, metric: str) -> pl.DataFrame:
    """Return every arm's level, all types and leads, in percent of capacity, best CRPS first."""
    rows = context.levels.filter(
        pl.col("group") == "all types", pl.col("lead_band") == BAND_ALL, pl.col("metric") == metric
    ).with_columns(pl.col("value", "lower_95", "upper_95") * PERCENT)
    return rows.filter(~pl.col("arm").is_in(LEADERBOARD_EXCLUDED))


def _arm_label(*, arm: str, post_hoc: bool = False) -> str:
    label = ARM_LABELS[arm]
    if post_hoc and arm not in PLANNED_ARMS and arm_definitions.ALL_ARMS[arm].family != "reference":
        return label + charts.POST_HOC_SUFFIX
    return label


def _contrast_rows(
    *,
    context: Context,
    arms: Sequence[str],
    reference: str,
    metric: str,
    group: str = "all types",
    band: str = BAND_ALL,
    condition: str | Callable[[str], str] = "",
    mark_planned: bool = True,
    mark_post_hoc: bool = True,
) -> pl.DataFrame:
    """Return the paired contrasts of `arms` against `reference`, in percentage points."""
    wanted = context.contrasts.filter(
        pl.col("reference") == reference,
        pl.col("group") == group,
        pl.col("lead_band") == band,
        pl.col("metric") == metric,
        pl.col("arm").is_in(list(arms)),
    )
    found = {row["arm"]: row for row in wanted.iter_rows(named=True)}
    rows = []
    for arm in arms:
        if arm not in found:
            continue
        row = found[arm]
        rows.append(
            {
                "arm": arm,
                "label": _arm_label(arm=arm, post_hoc=mark_post_hoc),
                "family": DUMMY_FAMILY,
                "difference": row["difference"] * PERCENT,
                "lower_95": row["lower_95"] * PERCENT,
                "upper_95": row["upper_95"] * PERCENT,
                "condition": condition(arm) if callable(condition) else condition,
                "planned": mark_planned and arm in PLANNED_ARMS,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "arm": pl.String,
            "label": pl.String,
            "family": pl.String,
            "difference": pl.Float64,
            "lower_95": pl.Float64,
            "upper_95": pl.Float64,
            "condition": pl.String,
            "planned": pl.Boolean,
        },
    )


def _domain(*, frames: Sequence[pl.DataFrame]) -> tuple[float, float]:
    """Cover every interval and estimate, and zero, rounded outwards to `DOMAIN_STEP`."""
    lows = [0.0, *(v for f in frames for v in f["lower_95"].to_list())]
    highs = [0.0, *(v for f in frames for v in f["upper_95"].to_list())]
    lows += [v for f in frames for v in f["difference"].to_list()]
    highs += [v for f in frames for v in f["difference"].to_list()]
    low = -DOMAIN_STEP * round(-min(lows) / DOMAIN_STEP + 0.4999999)
    high = DOMAIN_STEP * round(max(highs) / DOMAIN_STEP + 0.4999999)
    return (low, high)


def _order_by(*, rows: pl.DataFrame, order: Sequence[str]) -> pl.DataFrame:
    """Sort `rows` by the position of each row's `arm` in `order`."""
    position = {arm: index for index, arm in enumerate(order)}
    return rows.sort(pl.col("arm").replace_strict(position, default=len(position)))


def _kind_of(arm: str) -> str:
    """Return the colour group of an arm in a contrast figure."""
    family = ARM_FAMILIES[arm]
    return {
        "features": "Columns shown",
        "training": "Training",
        "quantile": "Quantile training",
        "combination": "Combination",
    }.get(family, "Cleaning")


KIND_ORDER: Final[tuple[str, ...]] = (
    "Cleaning",
    "Columns shown",
    "Training",
    "Quantile training",
    "Combination",
)
KIND_COLOURS: Final[tuple[str, ...]] = (
    ocf.DATA_SKY,
    ocf.DATA_BLUE,
    ocf.DATA_PURPLE,
    ocf.DATA_DEEP_TEAL,
    ocf.BRAND_ORANGE,
)


def _kind_args(*, kinds: Sequence[str]) -> dict[str, object]:
    """Return the `interval_panel` arguments that colour rows by the kinds present."""
    present = [kind for kind in KIND_ORDER if kind in set(kinds)]
    return {
        "conditions": present,
        "condition_colours": [KIND_COLOURS[KIND_ORDER.index(kind)] for kind in present],
        "condition_title": "Kind of change",
    }


# ---------------------------------------------------------------------------------------------
# Generic dot panel (levels, not contrasts)
# ---------------------------------------------------------------------------------------------


def _dot_panel(
    *,
    rows: pl.DataFrame,
    x_domain: tuple[float, float],
    x_title: str,
    panel_title: str,
    colours: dict[str, str],
    legend_title: str | None,
    rules: Sequence[tuple[float, str, str]] = (),
    zero_rule: str | None = None,
    row_step_px: int = ROW_STEP_PX,
    width: int = charts.PLOT_WIDTH_PX,
    tick_values: Sequence[float] | None = None,
    x_format: str = ".1~f",
    key_title: str | None = None,
) -> alt.LayerChart | alt.VConcatChart:
    """Draw one dot per row, with an interval where the row has one, coloured by `key`.

    Args:
        rows: Columns `label`, `value`, `key` (a key of `colours`), and optionally `lower_95` and
            `upper_95`; rows with the same label share a row and are offset by key.
        x_domain: The x range.
        x_title: The axis title.
        panel_title: The title above the panel.
        colours: Each key's colour, in legend order.
        legend_title: The legend's title, or None for no legend.
        rules: Vertical reference lines as (x, label, alignment of the text).
        zero_rule: Label of a rule at zero, or None for none.
        row_step_px: The row height.
        width: The plot width.
        tick_values: Axis ticks, or the round ticks of the domain.
        x_format: Format of the tick labels.
        key_title: Where given, colours are drawn literally (outside the figure's shared colour
            scale) and a key with this title sits above the panel; `legend_title` is then ignored.

    Returns:
        The dot panel, a layered chart with the optional key concatenated above it.
    """
    labels = list(dict.fromkeys(rows["label"].to_list()))
    data = rows.with_columns(pl.col("value").round(3))
    has_interval = "lower_95" in rows.columns
    if has_interval:
        data = data.with_columns(pl.col("lower_95", "upper_95").round(3))
    scale = alt.Scale(domain=list(x_domain), nice=False, zero=False)
    axis = alt.Axis(
        values=list(tick_values) if tick_values is not None else charts.ticks(x_domain=x_domain),
        format=x_format,
    )
    title_lines = charts.wrapped(text=x_title, width=78)
    if key_title is not None:
        data = data.with_columns(colour=pl.col("key").replace_strict(colours))
    colour = (
        alt.Color("colour:N", scale=None, legend=None)
        if key_title is not None
        else alt.Color(
            "key:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=(
                alt.Legend(
                    title=legend_title,
                    symbolType="circle",
                    symbolOpacity=1,
                    columns=2,
                    labelLimit=300,
                )
                if legend_title
                else None
            ),
        )
    )
    y = alt.Y(
        "label:N",
        sort=labels,
        title=None,
        axis=alt.Axis(
            labelLimit=charts.LABEL_WIDTH_PX,
            minExtent=charts.LABEL_WIDTH_PX,
            maxExtent=charts.LABEL_WIDTH_PX,
            labelPadding=6,
            ticks=False,
            domain=False,
        ),
    )
    offsets = rows["key"].n_unique() if rows["label"].is_duplicated().any() else 1
    extra: dict[str, object] = {}
    if offsets > 1:
        extra["yOffset"] = alt.YOffset("key:N", sort=list(colours))
    layers: list[alt.Chart] = []
    if has_interval:
        layers.append(
            alt.Chart(data.filter(pl.col("lower_95").is_not_null()))
            .mark_rule(strokeWidth=2, clip=True, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lower_95:Q", scale=scale, title=title_lines, axis=axis),
                x2="upper_95:Q",
                y=y,
                color=colour,
                **extra,
            )
        )
    layers.append(
        alt.Chart(data)
        .mark_point(filled=True, size=charts._POINT_SIZE, opacity=1, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", scale=scale, title=title_lines, axis=axis),
            y=y,
            color=colour,
            tooltip=[alt.Tooltip("label:N", title="Row"), alt.Tooltip("value:Q", title="Value")],
            **extra,
        )
    )
    reference_frames = []
    if zero_rule is not None:
        reference_frames.append((0.0, zero_rule, "left", 0))
    reference_frames += [
        (value, text, align, 1 + index) for index, (value, text, align) in enumerate(rules)
    ]
    for value, text, align, level in reference_frames:
        anchor = pl.DataFrame({"x": [value]})
        layers.append(
            alt.Chart(anchor)
            .mark_rule(color=ocf.BLACK_1, strokeWidth=1, strokeDash=list(charts.DASH), opacity=0.7)
            .encode(x=alt.X("x:Q", scale=scale))  # ty: ignore[unresolved-attribute]
        )
        layers.append(
            alt.Chart(anchor)
            .mark_text(
                align=align,  # ty: ignore[invalid-argument-type]
                dx=4 if align == "left" else -4,
                baseline="bottom",
                dy=-4 - 13 * level,
                color=ocf.BLACK_1,
            )
            .encode(x="x:Q", y=alt.value(0), text=alt.value(text))  # ty: ignore[unresolved-attribute]
        )
    panel = alt.LayerChart(
        layer=layers,
        width=width,
        height=alt.Step(row_step_px / offsets),
        title=alt.TitleParams(
            charts.wrapped(text=panel_title, width=85),
            anchor="start",
            frame="group",
            fontSize=charts._PANEL_TITLE_PX,
            offset=13 * (len(reference_frames) + 1) if reference_frames else 5,
        ),
    )
    if key_title is None:
        return panel
    key = charts._key(
        title=key_title,
        labels=list(colours),
        shapes=["circle"] * len(colours),
        filled=[True] * len(colours),
        colours=list(colours.values()),
    )
    return alt.vconcat(key, panel, spacing=8)


# ---------------------------------------------------------------------------------------------
# Figure 1: the leaderboard
# ---------------------------------------------------------------------------------------------


def figure_1(*, context: Context) -> Figure:
    """The leaderboard: every arm's CRPS and MAE with a month-resampled interval."""
    crps = _level_rows(context=context, metric="crps").sort("value")
    mae = _level_rows(context=context, metric="mae")
    order = crps["arm"].to_list()
    mae = _order_by(rows=mae, order=order)

    def shape(rows: pl.DataFrame) -> pl.DataFrame:
        return rows.select(
            pl.col("arm").replace_strict(ARM_LABELS).alias("label"),
            "value",
            "lower_95",
            "upper_95",
            pl.col("arm").replace_strict(ARM_FAMILIES).replace_strict(FAMILY_LABELS).alias("key"),
        )

    crps_rows, mae_rows = shape(crps), shape(mae)
    baselines = context.baselines
    best = crps.row(0, named=True)
    control = crps.filter(pl.col("arm") == "production_cleaned").row(0, named=True)
    stored = crps.filter(pl.col("arm") == "production_colsample0.8").row(0, named=True)

    def domain(rows: pl.DataFrame, metric: str) -> tuple[float, float]:
        values = [*rows["lower_95"], *rows["upper_95"]]
        values += [baselines[name][metric] for name in baselines]
        low, high = min(values), max(values)
        return (DOMAIN_STEP * int(low / DOMAIN_STEP), DOMAIN_STEP * (int(high / DOMAIN_STEP) + 1))

    crps_domain, mae_domain = domain(crps_rows, "crps"), domain(mae_rows, "mae")

    def rules(metric: str, stored_value: float) -> list[tuple[float, str, str]]:
        found = [(stored_value, "Production as stored", "left")]
        if "manual_heuristic" in baselines:
            found.append((baselines["manual_heuristic"][metric], "Manual heuristic", "right"))
        if "climatology" in baselines:
            found.append((baselines["climatology"][metric], "Climatology", "left"))
        return sorted(found)

    stored_mae = (
        mae.filter(pl.col("arm") == "production_colsample0.8")["value"][0]
        if not mae.is_empty()
        else stored["value"]
    )
    rule_text_crps = [
        f"Dashed line, {text}: {value:.1f}%" for value, text, _ in rules("crps", stored["value"])
    ]
    assert best["value"] < control["value"]
    assert best["arm"] not in ("production", "production_cleaned", "production_colsample0.8")
    best_name = ARM_LABELS[best["arm"]].lower()
    title_numbers = (
        f'{best["value"]:.1f}% of capacity ("{best_name}"), against '
        f"{control['value']:.1f}% for production with cleaning"
    )
    if "manual_heuristic" in baselines:
        heuristic = baselines["manual_heuristic"]["crps"]
        tail = "and below" if best["value"] < heuristic else "but still above"
        title = (
            f"The lowest CRPS is {title_numbers}, {tail} the manual heuristic's {heuristic:.1f}%"
        )
    else:
        title = f"The lowest CRPS is {title_numbers}"
    x_crps = "Continuous ranked probability score, CRPS (% of capacity; smaller is better)"
    x_mae = "Mean absolute error of the ensemble mean, MAE (% of capacity; smaller is better)"
    panel_crps = "CRPS of each arm, best first (scores the whole ensemble)"
    panel_mae = "MAE of each arm, in the same order (scores the ensemble mean only)"
    n_calibrated = context.calibration_runs
    # The calibration panel's rows and wording.
    calibration_rows = _calibration_level_rows(context=context)
    cal_x = (
        f"CRPS on the {n_calibrated} later forecast runs only (% of capacity; smaller is better)"
    )
    cal_panel = (
        f"Calibrated against uncalibrated, on the {n_calibrated} runs after the first 8 "
        "(not comparable with the panels above; no interval shown)"
    )
    subtitle = [
        (
            f"Each dot is an arm's mean score over {context.n_series} series (solar, wind, demand, "
            f"battery, and biofuel) and {context.scope}; {context.seeds_text}. "
            "Each score is a share of the series' own effective capacity."
        ),
        (
            "Each line is the 95% interval from resampling whole months. Month-to-month swings are "
            "shared by all arms, so overlapping intervals can hide a real gap (see Figure 2). Each "
            "dashed vertical line is the score of the named reference: the manual heuristic, "
            "climatology (the typical power at that time of day and year), or production as stored."
        ),
        (
            "Every arm is an XGBoost model fitted to the same weather data; arms differ only in "
            "the changes named in the label."
        ),
        ROUND2_LEGEND,
    ]
    legend = [FAMILY_LABELS[f] for f in FAMILY_ORDER]
    legend += ["Spread calibration: uncalibrated (black)", "Spread calibration: calibrated (green)"]
    text = FigureText(
        number=1,
        plots=(
            f"Fold {context.fold}: CRPS and MAE (% of capacity) of every arm, all series types, "
            "all leads, with 95% month-resampled intervals; baselines and production as lines; "
            "a third panel compares calibrated and uncalibrated CRPS."
        ),
        title=title,
        subtitle=subtitle,
        panel_titles=[panel_crps, panel_mae, cal_panel],
        axis_titles=[x_crps, x_mae, cal_x],
        legend=legend,
        annotations=[
            *rule_text_crps,
            "The MAE panel carries the same three dashed lines, at their MAE.",
            (
                "Manual heuristic and climatology are scored on the rows they share with production "
                "with cleaning; they are drawn as lines without intervals."
            ),
        ],
    )

    def draw() -> alt.TopLevelMixin:
        colours = {
            **{FAMILY_LABELS[family]: FAMILY_COLOURS[family] for family in FAMILY_ORDER},
            "Uncalibrated": ocf.BLACK_1,
            "Calibrated": ocf.DATA_GREEN,
        }
        top = _dot_panel(
            rows=crps_rows,
            x_domain=crps_domain,
            x_title=x_crps,
            panel_title=panel_crps,
            colours=colours,
            legend_title="Kind of arm, then calibration",
            rules=rules("crps", stored["value"]),
            row_step_px=ROW_STEP_PX,
        )
        middle = _dot_panel(
            rows=mae_rows,
            x_domain=mae_domain,
            x_title=x_mae,
            panel_title=panel_mae,
            colours=colours,
            legend_title=None,
            rules=rules("mae", stored_mae),
            row_step_px=ROW_STEP_PX,
        )
        panels: list[charts.Panel] = [top, middle]
        if calibration_rows is not None:
            low = min(calibration_rows["value"])
            high = max(calibration_rows["value"])
            cal_domain = (
                DOMAIN_STEP * int(low / DOMAIN_STEP),
                DOMAIN_STEP * (int(high / DOMAIN_STEP) + 1),
            )
            panels.append(
                _dot_panel(
                    rows=calibration_rows,
                    x_domain=cal_domain,
                    x_title=cal_x,
                    panel_title=cal_panel,
                    colours=colours,
                    legend_title=None,
                    row_step_px=34,
                )
            )
        return charts.figure(
            panels=panels,
            number=1,
            title=title,
            subtitle=subtitle,
            figure_planning=None,
        )

    return Figure(text=text, draw=draw)


def _calibration_level_rows(*, context: Context) -> pl.DataFrame | None:
    """Return raw and calibrated CRPS (% of capacity) of the calibrated arms, all types."""
    if context.calibration is None:
        return None
    rows = []
    for row in context.calibration.filter(pl.col("group") == "all types").iter_rows(named=True):
        label = ARM_LABELS[row["arm"]]
        rows.append({"label": label, "value": row["raw_crps"] * PERCENT, "key": "Uncalibrated"})
        rows.append(
            {"label": label, "value": row["calibrated_crps"] * PERCENT, "key": "Calibrated"}
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Figures 2 and 4: contrasts against production with cleaning
# ---------------------------------------------------------------------------------------------

CONTRAST_X: Final[str] = (
    "Change in {metric} from production with cleaning (percentage points of capacity)"
)
ZERO_LABEL: Final[str] = "same as production with cleaning"
BETTER_LABEL: Final[str] = "better than production with cleaning"
SHORT_ZERO_LABEL: Final[str] = "no change"
SHORT_BETTER_LABEL: Final[str] = "better"
"""Figure 3's wide axis leaves no room for the long labels; its axis title gives the full meaning."""


def _round1_and_round2_domain(*, context: Context) -> tuple[float, float]:
    """One x range for Figures 2 and 4, so a reader can compare their lengths."""
    frames = []
    for metric in ("crps", "mae"):
        frames.append(
            _contrast_rows(
                context=context, arms=ROUND1_ARMS, reference="production_cleaned", metric=metric
            )
        )
        frames.extend(
            _contrast_rows(
                context=context,
                arms=ROUND2_LADDER,
                reference="production_cleaned",
                metric=metric,
                band=band,
            )
            for band in BANDS
        )
    return _domain(frames=[f for f in frames if not f.is_empty()])


def figure_2(*, context: Context) -> Figure:
    """Paired contrasts of the round 1 arms against production with cleaning."""
    crps = _contrast_rows(
        context=context, arms=ROUND1_ARMS, reference="production_cleaned", metric="crps"
    ).sort("difference")
    order = crps["arm"].to_list()
    mae = _order_by(
        rows=_contrast_rows(
            context=context, arms=ROUND1_ARMS, reference="production_cleaned", metric="mae"
        ),
        order=order,
    )
    kinds = [_kind_of(arm) for arm in order]
    domain = _round1_and_round2_domain(context=context)
    best = crps.row(0, named=True)
    n_better = crps.filter(pl.col("upper_95") < 0).height
    title = (
        f"{n_better} of {crps.height} round 1 ideas lower CRPS reliably; "
        f'the largest, "{ARM_LABELS[best["arm"]]}", lowers it by {-best["difference"]:.1f} '
        "points of capacity"
    )
    subtitle = [
        (
            f"Each dot is the paired difference between an arm and production with cleaning in mean "
            f"score, over {context.n_series} series and {context.scope}; {context.seeds_text}. "
            "Zero means no change."
        ),
        (
            "Each line is the 95% interval from resampling whole months with the same months drawn "
            "for both arms. A line that excludes zero is a reliable difference."
        ),
    ]
    x_crps = CONTRAST_X.format(metric="CRPS")
    x_mae = CONTRAST_X.format(metric="MAE")
    panel_crps = "CRPS: scores the whole ensemble"
    panel_mae = "MAE: scores the ensemble mean only"
    text = FigureText(
        number=2,
        plots=(
            "Paired differences in CRPS and MAE (percentage points of capacity) between each "
            "round 1 arm and production with cleaning, with 95% month-resampled intervals."
        ),
        title=title,
        subtitle=subtitle,
        panel_titles=[panel_crps, panel_mae],
        axis_titles=[
            charts.axis_title_with_direction(
                x_title=x_crps, better_label=BETTER_LABEL, better_direction="negative"
            ),
            charts.axis_title_with_direction(
                x_title=x_mae, better_label=BETTER_LABEL, better_direction="negative"
            ),
        ],
        legend=[f"Kind of change: {kind}" for kind in KIND_ORDER if kind in set(kinds)],
        annotations=[
            ZERO_LABEL,
            f"← {BETTER_LABEL}",
            f"Added by the figure template: {charts.POST_HOC_PLANNING_NOTE}",
            "Row labels ending (planned) are in bold; rows ending (post hoc) were chosen later.",
        ],
    )

    def draw() -> alt.TopLevelMixin:
        panels = []
        for index, (rows, x_title, panel_title) in enumerate(
            ((crps, x_crps, panel_crps), (mae, x_mae, panel_mae))
        ):
            data = rows.with_columns(condition=pl.Series(kinds))
            panels.append(
                charts.interval_panel(
                    rows=data,
                    x_domain=domain,
                    x_title=x_title,
                    zero_label=ZERO_LABEL,
                    better_label=BETTER_LABEL,
                    panel_title=panel_title,
                    figure_planning="mixed",
                    condition_key=index == 0,
                    **_kind_args(kinds=kinds),  # ty: ignore[invalid-argument-type]
                )
            )
        return charts.figure(
            panels=panels,
            number=2,
            title=title,
            subtitle=subtitle,
            figure_planning="mixed",
            post_hoc=True,
        )

    return Figure(text=text, draw=draw)


def figure_4(*, context: Context) -> Figure:
    """The round 2 ladder, with its lead bands."""
    arms = [arm for arm in ROUND2_LADDER if arm in set(context.contrasts["arm"].to_list())]
    domain = _round1_and_round2_domain(context=context)
    frames = {}
    for metric in ("crps", "mae"):
        parts = [
            _contrast_rows(
                context=context,
                arms=arms,
                reference="production_cleaned",
                metric=metric,
                band=band,
                condition=BAND_LABELS[band],
                mark_planned=False,
            )
            for band in BANDS
        ]
        frames[metric] = pl.concat([p for p in parts if not p.is_empty()])
        frames[metric] = _order_by(rows=frames[metric], order=arms)
    all_leads = frames["crps"].filter(pl.col("condition") == BAND_LABELS[BAND_ALL])
    best = all_leads.sort("difference").row(0, named=True)
    title = (
        f"Combining ideas lowers CRPS by up to {-best['difference']:.1f} points of capacity, "
        "at every lead"
    )
    near = frames["crps"].filter(pl.col("condition") != BAND_LABELS[BAND_ALL])
    if near.filter(pl.col("upper_95") >= 0).height:
        title = (
            f"Combining ideas lowers CRPS by up to {-best['difference']:.1f} points of capacity "
            "overall; the gain differs between short and long leads"
        )
    subtitle = [
        (
            f"Each dot is the paired difference between an arm and production with cleaning in mean "
            f"score, over {context.n_series} series and {context.scope}; {context.seeds_text}. "
            "Zero means no change."
        ),
        (
            "Each line is the 95% interval from resampling whole months. Days 0 to 2 are forecasts "
            "up to 72 hours after the run starts; days 3 to 10 are 72 to 264 hours."
        ),
        ("The first row, absolute-error loss, is the best round 1 arm; each later row adds ideas."),
        "The horizontal axis has the same range as Figure 2.",
        ROUND2_LEGEND,
    ]
    x_crps = CONTRAST_X.format(metric="CRPS")
    x_mae = CONTRAST_X.format(metric="MAE")
    panel_crps = "CRPS: scores the whole ensemble"
    panel_mae = "MAE: scores the ensemble mean only"
    text = FigureText(
        number=4,
        plots=(
            "Paired differences in CRPS and MAE (percentage points of capacity) between each "
            "round 2 arm and production with cleaning, for all leads and for two lead bands."
        ),
        title=title,
        subtitle=subtitle,
        panel_titles=[panel_crps, panel_mae],
        axis_titles=[
            charts.axis_title_with_direction(
                x_title=x_crps, better_label=BETTER_LABEL, better_direction="negative"
            ),
            charts.axis_title_with_direction(
                x_title=x_mae, better_label=BETTER_LABEL, better_direction="negative"
            ),
        ],
        legend=[BAND_LABELS[b] for b in BANDS],
        annotations=[ZERO_LABEL, f"← {BETTER_LABEL}"],
    )

    def draw() -> alt.TopLevelMixin:
        panels = []
        for index, (metric, x_title, panel_title) in enumerate(
            (("crps", x_crps, panel_crps), ("mae", x_mae, panel_mae))
        ):
            panels.append(
                charts.interval_panel(
                    rows=frames[metric],
                    x_domain=domain,
                    x_title=x_title,
                    zero_label=ZERO_LABEL,
                    better_label=BETTER_LABEL,
                    panel_title=panel_title,
                    figure_planning="exploratory",
                    conditions=[BAND_LABELS[b] for b in BANDS],
                    condition_colours=list(BAND_COLOURS),
                    condition_title="Forecast lead",
                    condition_key=index == 0,
                    row_step_px=62,
                )
            )
        return charts.figure(
            panels=panels,
            number=4,
            title=title,
            subtitle=subtitle,
            figure_planning="exploratory",
            planning_note="Every row is post hoc: chosen after seeing the round 1 results.",
        )

    return Figure(text=text, draw=draw)


# ---------------------------------------------------------------------------------------------
# Figure 3: where the gains come from
# ---------------------------------------------------------------------------------------------


def figure_3(*, context: Context) -> Figure:
    """Contrasts against production with cleaning, by series type."""
    arms = [*BY_TYPE_ARMS, *([context.best_round2] if context.best_round2 else [])]
    present = [t for t in FIGURE3_TYPES if t in set(context.contrasts["group"].to_list())]
    blocks = {
        series_type: _order_by(
            rows=_contrast_rows(
                context=context,
                arms=arms,
                reference="production_cleaned",
                metric="crps",
                group=series_type,
                mark_planned=False,
            ),
            order=arms,
        )
        for series_type in present
    }
    domain = _domain(frames=list(blocks.values()))
    biggest = max(blocks, key=lambda t: -min(blocks[t]["difference"]))
    best_arm = blocks[biggest].sort("difference").row(0, named=True)
    best_name = (
        "the best round 2 arm"
        if best_arm["arm"] == context.best_round2
        else f'"{ARM_LABELS[best_arm["arm"]].lower()}"'
    )
    title = (
        f"The largest gains are in {TYPE_LABELS[biggest].lower()}: "
        f"{best_name} lowers their CRPS by {-best_arm['difference']:.1f} points of capacity"
    )
    subtitle = [
        (
            "Each dot is the paired difference in mean CRPS between an arm and production with "
            f"cleaning, within one type of series, over {context.scope}; {context.seeds_text}. "
            "Zero means no change."
        ),
        (
            "Each line is the 95% interval from resampling whole months. Battery storage (1 series) "
            "and biofuel (1 series) are in the all-series figures but not drawn here."
        ),
        (
            f'The best round 2 arm, "{ARM_LABELS[context.best_round2]}", is '
            f"{arm_definitions.ALL_ARMS[context.best_round2].description[:-1].lower()}."
        ),
        "(post hoc) marks an arm chosen after seeing earlier results; unmarked arms were planned.",
    ]
    x_crps = CONTRAST_X.format(metric="CRPS")
    panel_titles = [f"{TYPE_LABELS[t]} ({context.series_counts.get(t, 0)} series)" for t in blocks]
    text = FigureText(
        number=3,
        plots=(
            "Paired difference in CRPS (percentage points of capacity) from production with "
            "cleaning, for the main ideas and the best round 2 arm, one panel per series type."
        ),
        title=title,
        subtitle=subtitle,
        panel_titles=panel_titles,
        axis_titles=[
            charts.axis_title_with_direction(
                x_title=x_crps, better_label=BETTER_LABEL, better_direction="negative"
            )
        ],
        legend=[f"Kind of change: {k}" for k in KIND_ORDER[1:]],
        annotations=[SHORT_ZERO_LABEL, f"← {SHORT_BETTER_LABEL}", ROUND2_LEGEND],
    )

    def draw() -> alt.TopLevelMixin:
        panels = []
        kinds_all = [_kind_of(arm) for arm in arms]
        for index, rows in enumerate(blocks.values()):
            kinds = [_kind_of(arm) for arm in rows["arm"]]
            panels.append(
                charts.interval_panel(
                    rows=rows.with_columns(condition=pl.Series(kinds)),
                    x_domain=domain,
                    x_title=x_crps if index == len(blocks) - 1 else "",
                    zero_label=SHORT_ZERO_LABEL,
                    better_label=SHORT_BETTER_LABEL,
                    panel_title=panel_titles[index],
                    figure_planning="exploratory",
                    condition_key=index == 0,
                    reference_labels=index == 0,
                    row_step_px=30,
                    **_kind_args(kinds=kinds_all),  # ty: ignore[invalid-argument-type]
                )
            )
        return charts.figure(
            panels=panels,
            number=3,
            title=title,
            subtitle=subtitle,
            figure_planning="exploratory",
            planning_note="The breakdown by series type was not planned, so every row is exploratory.",
        )

    return Figure(text=text, draw=draw)


# ---------------------------------------------------------------------------------------------
# Figure 5: round 0 cleaning
# ---------------------------------------------------------------------------------------------


def _changed_series(*, context: Context) -> pl.DataFrame:
    """Return the per-series change in mean score of the chosen rule against production.

    Columns: `time_series_id`, `crps` and `mae` (changes, percentage points of capacity), and a
    label ranking the series within its type by effective capacity (1 is the largest).
    """
    per_seed = []
    for seed in context.seeds:
        frames = {}
        for arm in ("production", CHOSEN_RULE_ARM):
            path = (
                inputs.OUTPUT_ROOT / "arms" / context.fold / f"{arm}_seed{seed}" / "scores.parquet"
            )
            if not path.exists():
                break
            frames[arm] = (
                pl.read_parquet(path)
                .group_by("time_series_id")
                .agg(
                    crps=pl.col("crps_norm").mean(),
                    mae=pl.col("abs_error_norm").cast(pl.Float64).mean(),
                )
            )
        else:
            joined = frames[CHOSEN_RULE_ARM].join(
                frames["production"], on="time_series_id", suffix="_production"
            )
            per_seed.append(
                joined.select(
                    "time_series_id",
                    crps=(pl.col("crps") - pl.col("crps_production")) * PERCENT,
                    mae=(pl.col("mae") - pl.col("mae_production")) * PERCENT,
                )
            )
    if not per_seed:
        return pl.DataFrame(schema={"time_series_id": pl.Int32, "crps": pl.Float64})
    change = pl.concat(per_seed).group_by("time_series_id").agg(pl.col("crps", "mae").mean())
    ranked = (
        context.types.join(
            context.capacity.select("time_series_id", "effective_capacity_mw"), on="time_series_id"
        )
        .with_columns(
            rank=pl.col("effective_capacity_mw")
            .rank(method="ordinal", descending=True)
            .over("time_series_type")
        )
        .with_columns(
            label=pl.col("time_series_type").replace(TYPE_SHORT)
            + " "
            + pl.col("rank").cast(pl.String)
        )
        .select("time_series_id", "label", "time_series_type")
    )
    return change.join(ranked, on="time_series_id")


def _rows_removed(*, context: Context) -> dict[int, int]:
    """Return the training half-hours the chosen rule removes from each series."""
    path = inputs.OUTPUT_ROOT / "cleaning" / context.fold / f"{CHOSEN_RULE}.parquet"
    if not path.exists():
        path = inputs.OUTPUT_ROOT / "cleaning" / inputs.FOLD_A / f"{CHOSEN_RULE}.parquet"
    if not path.exists():
        return {}
    counted = pl.read_parquet(path).group_by("time_series_id").len()
    return dict(zip(counted["time_series_id"], counted["len"], strict=True))


def figure_5(*, context: Context) -> Figure | None:
    """Round 0 cleaning: contrasts against production, then the per-series change."""
    arms = [arm for arm in CLEANING_ARMS if arm in set(context.contrasts["arm"].to_list())]
    crps = _contrast_rows(context=context, arms=arms, reference="production", metric="crps")
    if crps.is_empty():
        return None
    mae = _contrast_rows(context=context, arms=arms, reference="production", metric="mae")
    kinds = ["Cleaning"] * len(arms)
    changed_all = _changed_series(context=context)
    changed = (
        changed_all.filter((pl.col("crps").abs() > 1e-9) | (pl.col("mae").abs() > 1e-9)).sort(
            "crps"
        )
        if not changed_all.is_empty()
        else changed_all
    )
    removed = _rows_removed(context=context)
    unchanged = changed_all.height - changed.height
    chosen = crps.filter(pl.col("arm") == CHOSEN_RULE_ARM)
    delta = float(chosen["difference"][0]) if not chosen.is_empty() else float("nan")
    title = (
        f"Dropping each new meter's first 6 weeks lowers CRPS by {-delta:.1f} points of capacity, "
        f"and only {changed.height} series change"
    )
    domain = _domain(frames=[f for f in (crps, mae) if not f.is_empty()])
    x_crps = "Change in CRPS from production (percentage points of capacity)"
    x_mae = "Change in MAE from production (percentage points of capacity)"
    zero = "same as production"
    better = "better than production"
    panel_crps = "CRPS: scores the whole ensemble"
    panel_mae = "MAE: scores the ensemble mean only"
    panel_series = (
        f"Change for the {changed.height} series whose forecast changed "
        f"({unchanged} others did not change)"
    )
    x_series = "Change from production for one series (percentage points of that series' capacity"
    rows_label = {
        row["label"]: (
            f"{row['label']} ({removed.get(row['time_series_id'], 0):,} half-hours removed)"
            if row["time_series_type"] == "Disaggregated Demand"
            else row["label"]
        )
        for row in changed.iter_rows(named=True)
    }
    subtitle = [
        (
            f"Each dot is the paired difference between a cleaning rule and production, which uses the "
            f"minimal cleaning already in the stored power table, over {context.n_series} series and "
            f"{context.scope}; {context.seeds_text}. Rules change only the training data. Zero means "
            "no change."
        ),
        (
            "Each line is the 95% interval from resampling whole months. Dropping the first 6 months "
            "of each series was also tried but removes a whole series, so it was not scored."
        ),
        (
            "The chosen rule, drop new-meter settling, removes the 6 weeks after a demand meter first "
            "reads live. It was written after seeing the results of the other rules. In the bottom "
            "panel, a series is named by its type and its rank by capacity within the type "
            "(Demand 1 is the largest)."
        ),
    ]
    text = FigureText(
        number=5,
        plots=(
            "Paired differences in CRPS and MAE (percentage points of capacity) between each "
            "training-data cleaning rule and production, and below them the per-series change "
            "for the chosen rule."
        ),
        title=title,
        subtitle=subtitle,
        panel_titles=[panel_crps, panel_mae, panel_series],
        axis_titles=[
            charts.axis_title_with_direction(
                x_title=x_crps, better_label=better, better_direction="negative"
            ),
            charts.axis_title_with_direction(
                x_title=x_mae, better_label=better, better_direction="negative"
            ),
            x_series + "; smaller is better)",
        ],
        legend=["Change in CRPS (blue)", "Change in MAE (orange)"],
        annotations=[
            zero,
            f"← {better}",
            f"Added by the figure template: {charts.POST_HOC_PLANNING_NOTE}",
            *rows_label.values(),
        ],
    )

    def draw() -> alt.TopLevelMixin:
        panels: list[charts.Panel] = []
        for rows, x_title, panel_title in (
            (crps, x_crps, panel_crps),
            (mae, x_mae, panel_mae),
        ):
            panels.append(
                charts.interval_panel(
                    rows=rows.with_columns(condition=pl.Series(kinds)),
                    x_domain=domain,
                    x_title=x_title,
                    zero_label=zero,
                    better_label=better,
                    panel_title=panel_title,
                    figure_planning="mixed",
                    condition_key=False,
                    **_kind_args(kinds=kinds),  # ty: ignore[invalid-argument-type]
                )
            )
        if not changed.is_empty():
            long = pl.concat(
                [
                    changed.select(
                        label=pl.col("label").replace(rows_label),
                        value=pl.col(column),
                        key=pl.lit(name),
                    )
                    for column, name in (("crps", "Change in CRPS"), ("mae", "Change in MAE"))
                ]
            )
            low = min(0.0, float(long["value"].min()))  # ty: ignore[invalid-argument-type]
            high = max(0.0, float(long["value"].max()))  # ty: ignore[invalid-argument-type]
            panels.append(
                _dot_panel(
                    rows=long,
                    x_domain=(
                        DOMAIN_STEP * (int(low / DOMAIN_STEP) - 1),
                        DOMAIN_STEP * (int(high / DOMAIN_STEP) + 1),
                    ),
                    x_title=x_series + "; smaller is better)",
                    panel_title=panel_series,
                    colours={"Change in CRPS": ocf.DATA_BLUE, "Change in MAE": ocf.BRAND_ORANGE},
                    legend_title=None,
                    key_title="Metric",
                    zero_rule="same as production",
                    row_step_px=44,
                )
            )
        return charts.figure(
            panels=panels,
            number=5,
            title=title,
            subtitle=subtitle,
            figure_planning="mixed",
            post_hoc=True,
        )

    return Figure(text=text, draw=draw)


# ---------------------------------------------------------------------------------------------
# Figure 6: calibration
# ---------------------------------------------------------------------------------------------


def figure_6(*, context: Context) -> Figure | None:
    """Spread calibration: band coverage, then the change in CRPS by series type."""
    calibration = context.calibration
    if calibration is None:
        return None
    arms = list(dict.fromkeys(calibration["arm"].to_list()))
    arm_titles = {arm: ARM_LABELS[arm] for arm in arms}
    groups = [g for g in FIGURE6_TYPES if g in set(calibration["group"].to_list())]
    group_labels = {g: "All series" if g == "all types" else TYPE_LABELS[g] for g in groups}
    overall = calibration.filter(pl.col("group") == "all types")
    first = overall.filter(pl.col("arm") == arms[0]).row(0, named=True)
    title = (
        f"Rescaling the ensemble spread lifts 80% band coverage from "
        f"{first['raw_band_coverage'] * PERCENT:.0f}% to "
        f"{first['calibrated_band_coverage'] * PERCENT:.0f}% and lowers CRPS by "
        f"{-first['difference'] * PERCENT:.1f} points of capacity"
    )
    subtitle = [
        (
            "Spread calibration (post hoc) multiplies how far each ensemble member sits from the "
            "ensemble mean by a factor chosen per series from earlier forecast runs only, so the "
            "ensemble mean and the MAE do not change."
        ),
        (
            f"Scored only on the {context.calibration_runs} forecast runs after the first 8, which "
            f"are left out while history builds up; {context.scope}; seed {context.calibration_seed}. "
            "Panels a use the two arms named in the panel titles; the second arm combines "
            "absolute-error loss, early stopping, the lead-time column, lean wind columns, and one "
            "model per type with a site identifier."
        ),
        (
            "In panel a, a dot is the share of outcomes that fell inside the ensemble's 10% to 90% "
            "band; a well-spread ensemble sits on the dashed 80% line. In panel b, a dot is the "
            "paired difference in CRPS (calibrated minus uncalibrated), and a line is the 95% interval "
            "from resampling whole months; zero means no change."
        ),
    ]
    x_a = "Share of outcomes inside the ensemble's 10% to 90% band (%; 80% is ideal)"
    x_b = "Change in CRPS from calibrating the spread (percentage points of capacity)"
    zero = "same as uncalibrated"
    better = "better than uncalibrated"
    panel_a = [f"a. 10% to 90% band coverage, {arm_titles[arm]}" for arm in arms]
    panel_b = "b. Change in CRPS from calibrating, by series type"
    text = FigureText(
        number=6,
        plots=(
            "Share of outcomes inside the ensemble's 10% to 90% band before and after spread "
            "calibration (a), and the paired change in CRPS from calibrating, by series type (b), "
            "for production with cleaning and the best point-forecast round 2 arm."
        ),
        title=title,
        subtitle=subtitle,
        panel_titles=[*panel_a, panel_b],
        axis_titles=[
            x_a,
            charts.axis_title_with_direction(
                x_title=x_b, better_label=better, better_direction="negative"
            ),
        ],
        legend=["Before calibration (black)", "After calibration (green)", *arm_titles.values()],
        annotations=[
            "Dashed line: 80% band coverage if the ensemble were well spread",
            zero,
            f"← {better}",
        ],
    )

    def draw() -> alt.TopLevelMixin:
        panels: list[charts.Panel] = []
        for index, arm in enumerate(arms):
            rows = []
            for group in groups:
                row = calibration.filter((pl.col("arm") == arm) & (pl.col("group") == group)).row(
                    0, named=True
                )
                rows.append(
                    {
                        "label": group_labels[group],
                        "value": row["raw_band_coverage"] * PERCENT,
                        "key": "Before calibration",
                    }
                )
                rows.append(
                    {
                        "label": group_labels[group],
                        "value": row["calibrated_band_coverage"] * PERCENT,
                        "key": "After calibration",
                    }
                )
            panels.append(
                _dot_panel(
                    rows=pl.DataFrame(rows),
                    x_domain=(-4.0, 100.0),
                    x_title=x_a,
                    panel_title=panel_a[index],
                    colours={
                        "Before calibration": ocf.BLACK_1,
                        "After calibration": ocf.DATA_GREEN,
                    },
                    legend_title=None,
                    key_title="Ensemble spread",
                    rules=[(80.0, "80% if well spread", "left")],
                    row_step_px=34,
                    tick_values=[0, 20, 40, 60, 80, 100],
                    x_format="d",
                )
            )
        rows_b = []
        for arm in arms:
            for group in groups:
                row = calibration.filter((pl.col("arm") == arm) & (pl.col("group") == group)).row(
                    0, named=True
                )
                rows_b.append(
                    {
                        "arm": arm,
                        "label": group_labels[group],
                        "family": DUMMY_FAMILY,
                        "difference": row["difference"] * PERCENT,
                        "lower_95": row["lower_95"] * PERCENT,
                        "upper_95": row["upper_95"] * PERCENT,
                        "condition": arm_titles[arm],
                    }
                )
        frame_b = pl.DataFrame(rows_b)
        panels.append(
            charts.interval_panel(
                rows=frame_b,
                x_domain=_domain(frames=[frame_b]),
                x_title=x_b,
                zero_label=zero,
                better_label=better,
                panel_title=panel_b,
                figure_planning="exploratory",
                conditions=list(arm_titles.values()),
                condition_colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE][: len(arms)],
                condition_title="Arm calibrated",
                row_step_px=40,
            )
        )
        return charts.figure(
            panels=panels,
            number=6,
            title=title,
            subtitle=subtitle,
            figure_planning=None,
        )

    return Figure(text=text, draw=draw)


# ---------------------------------------------------------------------------------------------
# Writing and rendering
# ---------------------------------------------------------------------------------------------

_SECTION: Final[re.Pattern[str]] = re.compile(r"^## Figure (\d+)\s*$", re.MULTILINE)


def write_text(*, figures: dict[int, Figure]) -> None:
    """Replace this script's `## Figure N` sections in the shared text file, keeping the rest."""
    existing = TEXT_FILE.read_text() if TEXT_FILE.exists() else ""
    matches = list(_SECTION.finditer(existing))
    sections: dict[int, str] = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(existing)
        sections[int(match.group(1))] = existing[match.start() : end].rstrip() + "\n"
    preamble = existing[: matches[0].start()] if matches else existing
    for number, figure in figures.items():
        sections[number] = figure.text.markdown()
    body = "\n".join(sections[number] for number in sorted(sections))
    header = preamble if preamble.strip() else "# Quick-wins figure text\n\n"
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_FILE.write_text(header.rstrip() + "\n\n" + body)


def render(*, figures: dict[int, Figure], png_dir: Path | None = None) -> None:
    """Draw each figure to SVG and shrink it with svgo."""
    for number, figure in figures.items():
        path = ASSET_DIR / f"quick_wins_figure_{number}.svg"
        chart = figure.draw()
        chart.save(str(path))
        if png_dir is not None:
            png_dir.mkdir(parents=True, exist_ok=True)
            chart.save(png_dir / f"quick_wins_figure_{number}.png", scale_factor=2)
        subprocess.run([*SVGO_COMMAND, str(path)], check=True, capture_output=True)
        print(f"wrote {path}")


def main() -> None:
    """Build Figures 1 to 6, write their text, then render them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default=inputs.FOLD_A)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--png-dir", type=Path, default=None, help="also write PNG renders here")
    parser.add_argument("--text-only", action="store_true", help="write the text file, no charts")
    args = parser.parse_args()

    context = build_context(fold=args.fold, seeds=args.seeds)
    built: dict[int, Figure | None] = {
        1: figure_1(context=context),
        2: figure_2(context=context),
        3: figure_3(context=context),
        4: figure_4(context=context),
        5: figure_5(context=context),
        6: figure_6(context=context),
    }
    figures = {number: figure for number, figure in built.items() if figure is not None}
    write_text(figures=figures)
    print(f"wrote {TEXT_FILE}")
    if not args.text_only:
        render(figures=figures, png_dir=args.png_dir)


if __name__ == "__main__":
    main()
