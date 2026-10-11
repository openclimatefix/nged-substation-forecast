"""Figures 7 to 12 of the quick-wins study: example forecasts drawn beside the actual power.

Each of figures 7 to 10 shows one anonymised series (a PV farm, a wind farm, a Disaggregated Demand
series, a Raw Flow series) in three stacked panels, one per forecast run. The runs are chosen by a
rule that uses the actual power only: among the validation year's weekly runs, the run with the
highest, the median, and the lowest mean actual power over the run's first 7 days. The series is
chosen by a rule too: the lower-middle series of its type when the type's series are ranked by the
`production_cleaned` arm's mean normalised CRPS. Figure 11 draws the 10% to 90% band of the
members before and after the post hoc B1 spread calibration. Figure 12 draws the ensemble mean's
absolute error by lead day for every run.

Every figure's text is written to `docs/studies/assets/quick_wins_figure_text.md` (replacing only
sections 7 to 12) before any figure is rendered. An NGED generator's power is normalised by its
effective capacity, labelled with `studies.anonymise`, and plotted against days since the run began,
never against a calendar date.

    uv run python studies/quick_wins/charts_examples.py [--fold ...] [--seed 0] [--figures 7 8 ...]
"""

import argparse
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings
from studies.anonymise import (
    WIND_LABEL_PERMUTATION_SEED,
    WIND_SITE_LABELS,
    site_labels_for,
)
from studies.charts import CONTENT_WIDTH_PX, wrapped

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the figures and the figure text file are written."""

TEXT_FILE: Final[Path] = ASSETS_DIR / "quick_wins_figure_text.md"
"""The file holding every quick-wins figure's text, one `## Figure N` section per figure."""

SHOWN_DAYS: Final[int] = 7
"""Each example panel shows the first this-many days after a run's initialisation time."""

LEAD_DAY_LIMIT_HOURS: Final[int] = 264
"""Leads from this many hours up are left out of the lead-day view (the study's days 3 to 10)."""

PERCENT: Final[float] = 100.0
"""Converts a fraction of effective capacity to a percentage."""

TITLE_CHARACTERS: Final[int] = 72
SUBTITLE_CHARACTERS: Final[int] = 115
PANEL_HEIGHT_PX: Final[int] = 150
PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 90
"""Plot-area width, leaving room for the y axis labels and title."""

ACTUAL: Final[str] = "Actual power"

PRODUCTION: Final[str] = "production_colsample0.8"
CLEANED: Final[str] = "production_cleaned"
BEST_ROUND_2: Final[str] = "r2_c4_c2_c1_a3w_d3"
"""The best round 2 arm, which is the study's final recipe (chosen after seeing round 1)."""

ARM_LABELS: Final[dict[str, str]] = {
    PRODUCTION: "Production as stored",
    CLEANED: "Production, cleaned training data",
    "c4_absolute_error": "C4: absolute-error loss",
    "a3w_lean_wind_only": "A3w: lean wind columns (post hoc)",
    "d3_pooled_with_site_id": "D3: pooled with site identifier (post hoc)",
    BEST_ROUND_2: "Best round 2 arm (post hoc)",
}
"""The legend entry of each arm. One entry per arm is used in every figure."""

ARM_COLOURS: Final[dict[str, str]] = {
    PRODUCTION: ocf.ENSEMBLE_LINE,
    CLEANED: ocf.DATA_SKY,
    "c4_absolute_error": ocf.DATA_PURPLE,
    "a3w_lean_wind_only": ocf.DATA_GREEN,
    "d3_pooled_with_site_id": ocf.BRAND_ORANGE,
    BEST_ROUND_2: ocf.DATA_BLUE,
}
"""One colour per arm, the same in every figure."""

RAW_BAND_LABEL: Final[str] = "10% to 90% member band, production (cleaned)"
CALIBRATED_BAND_LABEL: Final[str] = "10% to 90% member band after spread calibration (post hoc)"

RUN_KINDS: Final[tuple[str, ...]] = ("highest", "median", "lowest")
RUN_KIND_TITLES: Final[dict[str, str]] = {
    "highest": "Run with the highest mean actual power in its first 7 days",
    "median": "Run with the median mean actual power in its first 7 days",
    "lowest": "Run with the lowest mean actual power in its first 7 days",
}
PV_RUN_KIND_TITLES: Final[dict[str, str]] = {
    "highest": "Run with the highest daily energy in its first 7 days",
    "median": "Run with the median daily energy in its first 7 days",
    "lowest": "Run with the lowest daily energy in its first 7 days",
}

X_TITLE: Final[str] = "Days since the run's initialisation time (day 1 is the first 24 hours)"
Y_TITLE: Final[str] = "Power (% of effective capacity)"

RUN_SCOPE_SENTENCE: Final[str] = (
    "Each panel shows one forecast run from the weekly runs of the validation year (July 2025 to "
    "June 2026), chosen by the actual power alone and not by forecast error."
)
SERIES_RULE_SENTENCE: Final[str] = (
    "The series is the lower-middle one of its type when ranked by the cleaned production "
    "forecast's mean normalised CRPS (continuous ranked probability score) over the 53 weekly "
    "runs."
)


@dataclass(frozen=True)
class FigureText:
    """Every piece of text a figure draws, so the text file and the chart cannot disagree."""

    number: int
    title: str
    subtitle: list[str]
    panel_titles: list[str] = field(default_factory=list)
    x_title: str = X_TITLE
    y_title: str = Y_TITLE
    legend: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def markdown(self) -> str:
        """Return this figure's section of the text file."""
        lines = [f"## Figure {self.number}", "", f"Title: Figure {self.number}: {self.title}", ""]
        lines += ["Subtitle:", *[f"- {line}" for line in self.subtitle], ""]
        lines += ["Panel titles:", *[f"- {line}" for line in self.panel_titles], ""]
        lines += [f"X axis title: {self.x_title}", f"Y axis title: {self.y_title}", ""]
        lines += ["Legend entries:", *[f"- {line}" for line in self.legend], ""]
        if self.notes:
            lines += ["Other text:", *[f"- {line}" for line in self.notes], ""]
        return "\n".join(lines)


def write_text_sections(*, texts: Sequence[FigureText]) -> None:
    """Replace the sections of the given figures in the text file, keeping every other section."""
    existing = TEXT_FILE.read_text() if TEXT_FILE.exists() else ""
    parts = re.split(r"(?m)^(?=## Figure \d+\s*$)", existing)
    header = parts[0] if parts and not parts[0].startswith("## Figure") else ""
    sections: dict[int, str] = {}
    for part in parts:
        match = re.match(r"## Figure (\d+)", part)
        if match:
            sections[int(match.group(1))] = part.rstrip() + "\n"
    for text in texts:
        sections[text.number] = text.markdown().rstrip() + "\n"
    header = header or "# Quick-wins study: text of every figure\n\n"
    body = "\n".join(sections[number] for number in sorted(sections))
    TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEXT_FILE.write_text(header.rstrip() + "\n\n" + body)


@dataclass(frozen=True)
class Context:
    """Everything the figures share: scores of each arm, series types, and anonymous labels."""

    fold: str
    seed: int
    scores: dict[str, pl.DataFrame]
    types: pl.DataFrame
    labels: dict[int, str]
    arm_root: Path


def _arm_dir(*, fold: str, arm: str, seed: int) -> Path:
    return inputs.OUTPUT_ROOT / "arms" / fold / f"{arm}_seed{seed}"


def load_context(*, fold: str, seed: int) -> Context:
    """Read the scores of every plotted arm and label the generators."""
    settings = Settings()
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))
    pv_ids = types.filter(pl.col("time_series_type") == "PV")["time_series_id"].to_list()
    wind_ids = types.filter(pl.col("time_series_type") == "Wind")["time_series_id"].to_list()
    labels = {
        **site_labels_for(eligible_ids=pv_ids),
        **site_labels_for(
            eligible_ids=wind_ids, labels=WIND_SITE_LABELS, seed=WIND_LABEL_PERMUTATION_SEED
        ),
    }
    scores = {}
    for arm in ARM_LABELS:
        path = _arm_dir(fold=fold, arm=arm, seed=seed) / "scores.parquet"
        if path.exists():
            scores[arm] = pl.read_parquet(path)
    return Context(
        fold=fold,
        seed=seed,
        scores=scores,
        types=types,
        labels=labels,
        arm_root=inputs.OUTPUT_ROOT / "arms" / fold,
    )


def choose_series(*, context: Context, series_type: str) -> int:
    """Return the lower-middle series of a type, ranked by cleaned production's mean CRPS."""
    ranked = (
        context.scores[CLEANED]
        .join(context.types, on="time_series_id")
        .filter(pl.col("time_series_type") == series_type)
        .group_by("time_series_id")
        .agg(crps=pl.col("crps_norm").mean())
        .sort("crps", "time_series_id")
    )
    return int(ranked["time_series_id"][(ranked.height - 1) // 2])


def series_label(*, context: Context, series_id: int, series_type: str) -> str:
    """Return the anonymous label of a series."""
    if series_type == "PV":
        return f"PV farm {context.labels[series_id]}"
    if series_type == "Wind":
        return f"Wind farm {context.labels[series_id]}"
    return {"Disaggregated Demand": "Demand 1", "Raw Flow": "Raw Flow 1"}[series_type]


def first_week(*, scores: pl.DataFrame, series_id: int) -> pl.DataFrame:
    """Return one series' rows within `SHOWN_DAYS` of each run's initialisation time."""
    return scores.filter(
        pl.col("time_series_id") == series_id, pl.col("lead_hours") < SHOWN_DAYS * 24
    )


def choose_runs(*, actual_by_run: pl.DataFrame) -> dict[str, datetime]:
    """Return the initialisation time of the highest, median, and lowest mean-actual run.

    Args:
        actual_by_run: One row per run with `power_fcst_init_time` and `mean_actual`.
    """
    ranked = actual_by_run.sort("mean_actual", "power_fcst_init_time")
    return {
        "lowest": ranked["power_fcst_init_time"][0],
        "median": ranked["power_fcst_init_time"][ranked.height // 2],
        "highest": ranked["power_fcst_init_time"][-1],
    }


def _line_data(
    *, context: Context, series_id: int, init_time: datetime, arms: Sequence[str]
) -> pl.DataFrame:
    """Long-form rows (days, percent of capacity, line name) of the actual power and each arm."""
    frames = []
    for index, arm in enumerate(arms):
        week = first_week(scores=context.scores[arm], series_id=series_id).filter(
            pl.col("power_fcst_init_time") == init_time
        )
        if index == 0:
            frames.append(
                week.select(
                    days=pl.col("lead_hours") / 24,
                    percent=pl.col("actual") / pl.col("effective_capacity_mw") * PERCENT,
                    line=pl.lit(ACTUAL),
                )
            )
        frames.append(
            week.select(
                days=pl.col("lead_hours") / 24,
                percent=pl.col("ensemble_mean") / pl.col("effective_capacity_mw") * PERCENT,
                line=pl.lit(ARM_LABELS[arm]),
            )
        )
    return pl.concat(frames).sort("line", "days")


def _colour_scale(*, entries: Mapping[str, str]) -> alt.Scale:
    return alt.Scale(domain=list(entries), range=list(entries.values()))


def _x_axis(*, title: str | None) -> alt.Axis:
    return alt.Axis(values=list(range(SHOWN_DAYS + 1)), title=title, grid=False, format="d")


def _line_panel(*, data: pl.DataFrame, arms: Sequence[str], title: str, last: bool) -> alt.Chart:
    colours = {ACTUAL: ocf.BLACK_1, **{ARM_LABELS[arm]: ARM_COLOURS[arm] for arm in arms}}
    widths = {ACTUAL: 2.4, **{ARM_LABELS[arm]: 1.4 for arm in arms}}
    return (
        alt.Chart(data, title=alt.TitleParams(title, anchor="start", fontSize=12))
        .mark_line(aria=False, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "days:Q",
                axis=_x_axis(title=X_TITLE if last else None),
                scale=alt.Scale(domain=[0, SHOWN_DAYS], nice=False),
            ),
            y=alt.Y(
                "percent:Q",
                title=Y_TITLE,
                scale=alt.Scale(zero=True),
                axis=alt.Axis(grid=True),
            ),
            color=alt.Color(
                "line:N",
                title=None,
                scale=_colour_scale(entries=colours),
                legend=alt.Legend(columns=2, symbolStrokeWidth=3, symbolType="stroke"),
            ),
            strokeWidth=alt.StrokeWidth(
                "line:N",
                scale=alt.Scale(domain=list(widths), range=list(widths.values())),
                legend=None,
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=PANEL_HEIGHT_PX)
    )


def _stack(*, panels: Sequence[alt.Chart], text: FigureText) -> alt.VConcatChart:
    caption = alt.TitleParams(
        wrapped(text=f"Figure {text.number}: {text.title}", width=TITLE_CHARACTERS),
        subtitle=[
            line
            for sentence in text.subtitle
            for line in wrapped(text=sentence, width=SUBTITLE_CHARACTERS)
        ],
        anchor="start",
        offset=14,
        subtitleColor=ocf.BLACK_1,
        subtitlePadding=6,
    )
    return (
        alt.vconcat(*panels, spacing=26)
        .properties(title=caption)
        .resolve_scale(color="shared", strokeWidth="shared")
        .configure_view(stroke=None)
        .configure_legend(orient="top", direction="horizontal", labelLimit=400)
    )


def example_figure(
    *, context: Context, number: int, series_type: str, arms: Sequence[str], title: str
) -> tuple[FigureText, alt.VConcatChart]:
    """Build one of figures 7 to 10: three runs of one series, actual beside each arm's mean."""
    series_id = choose_series(context=context, series_type=series_type)
    name = series_label(context=context, series_id=series_id, series_type=series_type)
    week = first_week(scores=context.scores[CLEANED], series_id=series_id)
    by_run = week.group_by("power_fcst_init_time").agg(
        mean_actual=(pl.col("actual") / pl.col("effective_capacity_mw")).mean()
    )
    runs = choose_runs(actual_by_run=by_run)
    run_titles = PV_RUN_KIND_TITLES if series_type == "PV" else RUN_KIND_TITLES
    panel_titles = [f"{name}: {run_titles[kind]}" for kind in RUN_KINDS]
    arm_names = [ARM_LABELS[arm] for arm in arms]
    text = FigureText(
        number=number,
        title=title,
        subtitle=[
            (
                f"Black line: actual power of {name}. Coloured lines: each forecast's ensemble "
                "mean (the average of the 51 ensemble members), as a percentage of the series' "
                "effective capacity. Zero is no power."
            ),
            RUN_SCOPE_SENTENCE,
            "Panels run from the run with the highest mean actual power (top) to the lowest "
            "(bottom). " + SERIES_RULE_SENTENCE,
        ],
        panel_titles=panel_titles,
        legend=[ACTUAL, *arm_names],
        notes=[f"Models were trained on one random seed (seed {context.seed})."],
    )
    panels = []
    for index, kind in enumerate(RUN_KINDS):
        data = _line_data(
            context=context, series_id=series_id, init_time=runs[kind], arms=list(arms)
        )
        panels.append(
            _line_panel(
                data=data, arms=arms, title=panel_titles[index], last=index == len(RUN_KINDS) - 1
            )
        )
    return text, _stack(panels=panels, text=text)


def _members(
    *, context: Context, arm: str, series_id: int, init_time: datetime
) -> tuple[pl.DataFrame, float]:
    """Read one run's members for one series, as a fraction of capacity, within the shown days."""
    capacity = float(
        context.scores[arm].filter(pl.col("time_series_id") == series_id)["effective_capacity_mw"][
            0
        ]
    )
    path = _arm_dir(fold=context.fold, arm=arm, seed=context.seed) / "forecasts"
    stamp = init_time.strftime("%Y-%m-%d")
    members = pl.read_parquet(path / f"init={stamp}.parquet").filter(
        pl.col("time_series_id") == series_id,
        pl.col("power_fcst_init_time") == init_time,
    )
    return (
        members.with_columns(
            days=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 86400,
            power=pl.col("power_fcst") / capacity,
        ).filter(pl.col("days") < SHOWN_DAYS),
        capacity,
    )


def _band(*, members: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Return the 10% and 90% band of the members after `factor` scaling about their mean."""
    mean = pl.col("power").mean().over("valid_time")
    scaled = members.with_columns(scaled=mean + factor * (pl.col("power") - mean))
    return (
        scaled.group_by("days")
        .agg(low=pl.col("scaled").quantile(0.1), high=pl.col("scaled").quantile(0.9))
        .sort("days")
        .with_columns(pl.col("low", "high") * PERCENT)
    )


def band_figure(*, context: Context) -> tuple[FigureText, alt.VConcatChart]:
    """Build figure 11: the production member band before and after spread calibration."""
    calibration_dir = inputs.OUTPUT_ROOT / "calibration" / context.fold
    coverage = pl.read_parquet(calibration_dir / f"calibration_seed{context.seed}.parquet").filter(
        pl.col("arm") == CLEANED
    )
    factors = pl.read_parquet(calibration_dir / f"{CLEANED}_seed{context.seed}" / "factors.parquet")
    panels, panel_titles = [], []
    notes = []
    for series_type in ("PV", "Wind"):
        series_id = choose_series(context=context, series_type=series_type)
        name = series_label(context=context, series_id=series_id, series_type=series_type)
        week = first_week(scores=context.scores[CLEANED], series_id=series_id)
        scored = factors.filter(pl.col("time_series_id") == series_id)["power_fcst_init_time"]
        by_run = (
            week.filter(pl.col("power_fcst_init_time").is_in(scored.implode()))
            .group_by("power_fcst_init_time")
            .agg(mean_actual=(pl.col("actual") / pl.col("effective_capacity_mw")).mean())
        )
        init_time = choose_runs(actual_by_run=by_run)["median"]
        factor = float(
            factors.filter(
                pl.col("time_series_id") == series_id, pl.col("power_fcst_init_time") == init_time
            )["factor"][0]
        )
        members, _ = _members(
            context=context, arm=CLEANED, series_id=series_id, init_time=init_time
        )
        raw = _band(members=members, factor=1.0).with_columns(line=pl.lit(RAW_BAND_LABEL))
        calibrated = _band(members=members, factor=factor).with_columns(
            line=pl.lit(CALIBRATED_BAND_LABEL)
        )
        actual = (
            week.filter(pl.col("power_fcst_init_time") == init_time)
            .select(
                days=pl.col("lead_hours") / 24,
                percent=pl.col("actual") / pl.col("effective_capacity_mw") * PERCENT,
                line=pl.lit(ACTUAL),
            )
            .sort("days")
        )
        group = "PV" if series_type == "PV" else "Wind"
        rates = coverage.filter(pl.col("group") == group)
        title = (
            f"{name}: median run (by daily energy); this series' spread was multiplied by "
            f"{factor:.1f}"
            if series_type == "PV"
            else f"{name}: median run (by mean actual power); this series' spread was multiplied "
            f"by {factor:.1f}"
        )
        panel_titles.append(title)
        if rates.height:
            notes.append(
                f"{group}, all runs after the first 8: share of actual values inside the band "
                f"{rates['raw_band_coverage'][0]:.0%} before and "
                f"{rates['calibrated_band_coverage'][0]:.0%} after calibration (nominal 80%)."
            )
        entries = {
            ACTUAL: ocf.BLACK_1,
            RAW_BAND_LABEL: ocf.DATA_SKY,
            CALIBRATED_BAND_LABEL: ocf.DATA_PURPLE,
        }
        bands = pl.concat([raw, calibrated])
        shade = (
            alt.Chart(bands)
            .mark_area(aria=False, opacity=0.4, clip=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "days:Q",
                    scale=alt.Scale(domain=[0, SHOWN_DAYS], nice=False),
                    axis=_x_axis(title=X_TITLE if series_type == "Wind" else None),
                ),
                y=alt.Y("low:Q", title=Y_TITLE, scale=alt.Scale(zero=True)),
                y2="high:Q",
                color=alt.Color(
                    "line:N",
                    title=None,
                    scale=_colour_scale(entries=entries),
                    legend=alt.Legend(columns=1, symbolType="square"),
                ),
            )
        )
        line = (
            alt.Chart(actual)
            .mark_line(aria=False, clip=True, strokeWidth=2.2)
            .encode(  # ty: ignore[unresolved-attribute]
                x="days:Q",
                y="percent:Q",
                color=alt.Color("line:N", scale=_colour_scale(entries=entries), legend=None),
            )
        )
        panels.append(
            alt.layer(shade, line).properties(
                title=alt.TitleParams(title, anchor="start", fontSize=12),
                width=PLOT_WIDTH_PX,
                height=PANEL_HEIGHT_PX,
            )
        )
    text = FigureText(
        number=11,
        title="The raw ensemble band is too narrow, and rescaling the spread widens it",
        subtitle=[
            (
                "Black line: actual power as a percentage of effective capacity. Blue shade: the "
                "band holding the middle 80% of the 51 ensemble members (10% to 90%) of the "
                "cleaned production forecast. Purple shade: the same band after the post hoc B1 "
                "spread calibration, which rescales each series' members about their mean using "
                "only past outcomes. The ensemble mean, and so the mean absolute error, is "
                "unchanged."
            ),
            (
                "A well-calibrated band holds the actual power 80% of the time. Each panel shows "
                "one forecast run from the weekly runs of the validation year (July 2025 to June "
                "2026): the run with the median mean actual power in its first 7 days among the "
                "runs the calibration scored, chosen by the actual power alone."
            ),
        ],
        panel_titles=panel_titles,
        legend=[ACTUAL, RAW_BAND_LABEL, CALIBRATED_BAND_LABEL],
        notes=notes,
    )
    return text, _stack(panels=panels, text=text)


def lead_day_figure(*, context: Context) -> tuple[FigureText, alt.VConcatChart]:
    """Build figure 12: mean absolute error by lead day for production and the best round 2 arm."""
    arms = [PRODUCTION, BEST_ROUND_2]
    lead_type = [
        ("PV", "PV farms"),
        ("Wind", "Wind farms"),
        ("Disaggregated Demand", "Disaggregated Demand series"),
        ("Raw Flow", "Raw Flow series"),
    ]
    frames = [
        (
            context.scores[arm]
            .join(context.types, on="time_series_id")
            .filter(pl.col("lead_hours") < LEAD_DAY_LIMIT_HOURS)
            .group_by("time_series_type", (pl.col("lead_hours") // 24 + 1).alias("day"))
            .agg(
                error=pl.col("abs_error_norm").mean() * PERCENT,
                series=pl.col("time_series_id").n_unique(),
            )
            .with_columns(line=pl.lit(ARM_LABELS[arm]))
        )
        for arm in arms
    ]
    data = pl.concat(frames)
    entries = {ARM_LABELS[arm]: ARM_COLOURS[arm] for arm in arms}
    panel_titles, panels = [], []
    for index, (series_type, plural) in enumerate(lead_type):
        count = int(data.filter(pl.col("time_series_type") == series_type)["series"][0])
        panel_titles.append(f"Mean over the {count} {plural}")
        panels.append(
            alt.Chart(
                data.filter(pl.col("time_series_type") == series_type),
                title=alt.TitleParams(panel_titles[-1], anchor="start", fontSize=12),
            )
            .mark_line(aria=False, point=True, strokeWidth=1.8)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "day:Q",
                    scale=alt.Scale(domain=[0.5, 11.5], nice=False),
                    axis=alt.Axis(
                        values=list(range(1, 12)),
                        title=LEAD_X_TITLE if index == len(lead_type) - 1 else None,
                        grid=False,
                    ),
                ),
                y=alt.Y("error:Q", title=LEAD_Y_TITLE, scale=alt.Scale(zero=True)),
                color=alt.Color(
                    "line:N",
                    title=None,
                    scale=_colour_scale(entries=entries),
                    legend=alt.Legend(columns=2, symbolStrokeWidth=3, symbolType="stroke"),
                ),
            )
            .properties(width=PLOT_WIDTH_PX, height=PANEL_HEIGHT_PX - 20)
        )
    wide = data.pivot(on="line", index=["time_series_type", "day"], values="error")
    assert (wide[ARM_LABELS[BEST_ROUND_2]] < wide[ARM_LABELS[PRODUCTION]]).all(), (
        "figure 12's title says the best round 2 arm is lower at every lead day"
    )
    text = FigureText(
        number=12,
        title=(
            "The best round 2 arm has a lower error than production at every lead day, for all "
            "four series types"
        ),
        subtitle=[
            (
                "Mean absolute error of the ensemble mean (the average of the 51 ensemble "
                "members) as a percentage of effective capacity; smaller is better, zero is a "
                "perfect forecast. Each point averages every half-hourly forecast of the type's "
                "series, over all 53 weekly runs of the validation year (July 2025 to June 2026)."
            ),
            (
                "Day 1 is the first 24 hours after a run's initialisation time. Leads of 11 days "
                "and more are left out. The best round 2 arm was chosen after seeing round 1 "
                "(post hoc)."
            ),
        ],
        panel_titles=panel_titles,
        x_title=LEAD_X_TITLE,
        y_title=LEAD_Y_TITLE,
        legend=list(entries),
        notes=[f"Models were trained on one random seed (seed {context.seed})."],
    )
    return text, _stack(panels=panels, text=text)


LEAD_X_TITLE: Final[str] = "Lead day (day 1 is the first 24 hours after the run's initialisation)"
LEAD_Y_TITLE: Final[str] = "Mean abs. error (% of capacity)"


def build_figures(
    *, context: Context, wanted: set[int]
) -> dict[int, tuple[FigureText, alt.VConcatChart]]:
    """Build every wanted figure whose arms have all been scored."""
    best = BEST_ROUND_2 in context.scores
    demand_arms = [PRODUCTION, "c4_absolute_error", "d3_pooled_with_site_id", BEST_ROUND_2]
    specs: dict[int, tuple[str, list[str], str]] = {
        7: (
            "PV",
            [PRODUCTION, CLEANED, BEST_ROUND_2],
            (
                "Forecasts of a PV farm follow the daily solar cycle but often miss the height of "
                "each day's peak"
            ),
        ),
        8: (
            "Wind",
            [PRODUCTION, "a3w_lean_wind_only", BEST_ROUND_2],
            (
                "Forecasts of a wind farm follow the broad swings in power but miss its fast "
                "changes and highest peaks"
            ),
        ),
        9: (
            "Disaggregated Demand",
            demand_arms,
            (
                "Forecasts of a demand series follow its daily pattern but can sit well above or "
                "below its level"
            ),
        ),
        10: (
            "Raw Flow",
            demand_arms,
            (
                "Forecasts of a Raw Flow series follow its daily swing but miss longer departures "
                "from it"
            ),
        ),
    }
    built: dict[int, tuple[FigureText, alt.VConcatChart]] = {}
    for number, (series_type, arms, title) in specs.items():
        if number in wanted and best and all(arm in context.scores for arm in arms):
            built[number] = example_figure(
                context=context, number=number, series_type=series_type, arms=arms, title=title
            )
    if 11 in wanted:
        built[11] = band_figure(context=context)
    if 12 in wanted and best:
        built[12] = lead_day_figure(context=context)
    return built


def save(*, chart: alt.TopLevelMixin, number: int) -> Path:
    """Write a figure to SVG, optimise it with svgo, and write a PNG beside it for checking."""
    path = ASSETS_DIR / f"quick_wins_figure_{number}.svg"
    chart.save(path)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True,
        capture_output=True,
    )
    return path


def main() -> None:
    """Build the figures, write their text, then render them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default=inputs.FOLD_A)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figures", type=int, nargs="+", default=list(range(7, 13)))
    parser.add_argument("--png-dir", type=Path, default=None, help="also write PNG renders here")
    args = parser.parse_args()
    context = load_context(fold=args.fold, seed=args.seed)
    built = build_figures(context=context, wanted=set(args.figures))
    write_text_sections(texts=[text for text, _ in built.values()])
    for number, (_, chart) in built.items():
        print(save(chart=chart, number=number))
        if args.png_dir is not None:
            args.png_dir.mkdir(parents=True, exist_ok=True)
            chart.save(args.png_dir / f"quick_wins_figure_{number}.png", scale_factor=2)


if __name__ == "__main__":
    main()
