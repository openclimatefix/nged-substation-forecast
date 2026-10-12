"""Draw Figures 13 and 14 of the quick-wins study: the two diagnostic comparisons.

Figure 13 compares the XGBoost arms with the manual heuristic and climatology by lead band and
series type. Figure 14 is a learning curve: the CRPS of models trained on only the last 6, 9, or 12
months of the 15-month training window, against the 15-month production arm with cleaning.

The text of each figure is written to `docs/studies/assets/quick_wins_figure_text.md` before any
chart is rendered, replacing only these two figures' sections. Figure 13 is computed here from the
per-row scores (the mean over rows per series, then over the series of a type, as the leaderboard
does); Figure 14 reads the contrasts the report script wrote.

    uv run python studies/quick_wins/charts_diagnostics.py [--seeds 0 1 2] [--text-only]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import altair as alt
import charts_examples as examples
import per_run_scores
import plotting.ocf_theme as ocf
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings

from studies import charts

FOLD: Final[str] = inputs.FOLD_A
PERCENT: Final[float] = 100.0
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
HOURS_PER_DAY: Final[int] = 24
SHORT_LEAD_HOURS: Final[int] = 72
LONG_LEAD_HOURS: Final[int] = 264
MID_GREY: Final[str] = "#8C8C8C"

BAND_LABELS: Final[dict[int, str]] = {0: "0-2", 1: "3-10", 2: "11+"}
BAND_ORDER: Final[list[str]] = list(BAND_LABELS.values())

HEURISTIC: Final[str] = "manual_heuristic"
CLIMATOLOGY: Final[str] = "climatology"
CLEANED: Final[str] = "production_cleaned"
BEST_POINT: Final[str] = "r2_c4_c2_c1_a3w_d3"
BEST_QUANTILE: Final[str] = "r2_e1_c1_a3w_d3"
MODEL_LABELS: Final[dict[str, str]] = {
    HEURISTIC: "Manual heuristic",
    CLIMATOLOGY: "Climatology",
    CLEANED: "XGBoost, production with cleaning",
    BEST_POINT: "XGBoost, best point-forecast arm (post hoc)",
    BEST_QUANTILE: "XGBoost, best quantile-trained arm (post hoc)",
}
MODEL_COLOURS: Final[dict[str, str]] = {
    HEURISTIC: ocf.BLACK_1,
    CLIMATOLOGY: MID_GREY,
    CLEANED: ocf.DATA_SKY,
    BEST_POINT: ocf.DATA_BLUE,
    BEST_QUANTILE: ocf.BRAND_ORANGE,
}

TYPE_ORDER: Final[list[str]] = ["PV", "Wind", "Disaggregated Demand", "Raw Flow", "all"]
TYPE_LABELS: Final[dict[str, str]] = {
    "PV": "PV farms",
    "Wind": "Wind farms",
    "Disaggregated Demand": "Demand series",
    "Raw Flow": "Raw Flow series",
    "all": "All types",
}
GROUP_NAMES: Final[dict[str, str]] = {"all": "all types", **{t: t for t in TYPE_ORDER[:-1]}}

PANEL_WIDTH_PX: Final[int] = 78
PANEL_HEIGHT_PX: Final[int] = 125

STEPS_ORDER: Final[list[int]] = [6, 9, 12, 15]
F1_ARMS: Final[dict[int, str]] = {
    6: "f1_train_last_6_months",
    9: "f1_train_last_9_months",
    12: "f1_train_last_12_months",
}
CURVE_GROUPS: Final[list[str]] = ["all types", "PV", "Wind", "Disaggregated Demand", "Raw Flow"]
CURVE_LABELS: Final[dict[str, str]] = {
    "all types": "All types",
    "PV": "PV farms",
    "Wind": "Wind farms",
    "Disaggregated Demand": "Demand series",
    "Raw Flow": "Raw Flow series",
}


PANEL_TITLE_14: Final[str] = (
    "CRPS of a model trained on only the last N months, minus the CRPS of the 15-month "
    "production arm with cleaning (points of capacity; smaller is better)"
)


@dataclass(frozen=True)
class Figure:
    """A figure's text and the chart drawn from it."""

    text: examples.FigureText
    chart: alt.TopLevelMixin


def _seeds_text(*, seeds: list[int]) -> str:
    """Name the random seeds the XGBoost scores average over."""
    if len(seeds) == 1:
        return f"scores of one random seed (seed {seeds[0]})"
    return f"average over random seeds {', '.join(str(seed) for seed in seeds)}"


def _type_means(*, scored: pl.DataFrame, types: pl.DataFrame) -> pl.DataFrame:
    """Mean over rows per series, then over the series of each type, per lead band."""
    band = (
        pl.when(pl.col("lead_hours") < SHORT_LEAD_HOURS)
        .then(0)
        .when(pl.col("lead_hours") < LONG_LEAD_HOURS)
        .then(1)
        .otherwise(2)
        .alias("band")
    )
    per_series = (
        scored.join(types, on="time_series_id")
        .with_columns(band)
        .group_by("time_series_id", "time_series_type", "band")
        .agg(
            mae=pl.col("abs_error_norm").mean() * PERCENT,
            crps=pl.col("crps_norm").mean() * PERCENT,
        )
    )
    by_type = per_series.group_by("time_series_type", "band").agg(pl.col("mae", "crps").mean())
    overall = (
        per_series.group_by("band")
        .agg(pl.col("mae", "crps").mean())
        .with_columns(time_series_type=pl.lit("all"))
        .select(by_type.columns)
    )
    return pl.concat([by_type, overall]).with_columns(
        pl.col("mae", "crps").cast(pl.Float64), pl.col("band").replace_strict(BAND_LABELS)
    )


def lead_band_scores(*, seeds: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return every model's lead-band means on the rows all five models scored, and the types.

    The XGBoost arms' means are averaged over `seeds`; the two baselines have no seed.
    """
    settings = Settings()
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))
    capacity = pl.DataFrame(inputs.effective_capacity(settings=settings))
    baselines = inputs.OUTPUT_ROOT / "production_arm" / "colsample0.8_seed0" / "report"
    frames: dict[str, pl.DataFrame] = {
        name: per_run_scores.normalised_by_capacity(
            scores=pl.read_parquet(baselines / f"scores_stored_{name}.parquet"), capacity=capacity
        )
        for name in (HEURISTIC, CLIMATOLOGY)
    }
    arm_frames = {
        (arm, seed): pl.read_parquet(
            inputs.OUTPUT_ROOT / "arms" / FOLD / f"{arm}_seed{seed}" / "scores.parquet"
        )
        for arm in (CLEANED, BEST_POINT, BEST_QUANTILE)
        for seed in seeds
    }
    shared = None
    for frame in (*frames.values(), *arm_frames.values()):
        shared = frame.select(KEYS) if shared is None else shared.join(frame.select(KEYS), on=KEYS)
    assert shared is not None
    per_seed = [
        _type_means(scored=frame.join(shared, on=KEYS), types=types).with_columns(
            model=pl.lit(name)
        )
        for name, frame in frames.items()
    ]
    per_seed += [
        _type_means(scored=frame.join(shared, on=KEYS), types=types).with_columns(model=pl.lit(arm))
        for (arm, _seed), frame in arm_frames.items()
    ]
    rows = (
        pl.concat(per_seed)
        .group_by("time_series_type", "band", "model")
        .agg(pl.col("mae", "crps").mean())
    )
    return rows, types


def _facet(*, data: pl.DataFrame, title: str) -> alt.FacetChart:
    """Draw one metric: a column of lines per series type, one line per model."""
    chart = (
        alt.Chart(data)
        .mark_line(aria=False, point=True, strokeWidth=1.8)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "band:N",
                sort=BAND_ORDER,
                title="Lead (days)",
                axis=alt.Axis(labelAngle=0, labelFontSize=9, grid=False),
            ),
            y=alt.Y("score:Q", title=None, scale=alt.Scale(zero=True)),
            color=alt.Color(
                "label:N",
                title=None,
                sort=list(MODEL_LABELS.values()),
                scale=alt.Scale(
                    domain=list(MODEL_LABELS.values()), range=list(MODEL_COLOURS.values())
                ),
                legend=alt.Legend(
                    columns=1, symbolStrokeWidth=3, symbolType="stroke", labelLimit=400
                ),
            ),
        )
        .properties(width=PANEL_WIDTH_PX, height=PANEL_HEIGHT_PX)
    )
    return (
        chart.facet(
            column=alt.Column(
                "group:N",
                sort=[TYPE_LABELS[t] for t in TYPE_ORDER],
                title=None,
                header=alt.Header(labelOrient="top", labelFontSize=11, labelAnchor="start"),
            )
        )
        .resolve_scale(y="independent")
        .properties(title=alt.TitleParams(charts.wrapped(text=title, width=85), anchor="start"))
    )


def figure_13(*, seeds: list[int]) -> Figure:
    """Build Figure 13: lead-band MAE and CRPS of the heuristic, climatology, and XGBoost arms."""
    rows, types = lead_band_scores(seeds=seeds)
    counts = dict(types.group_by("time_series_type").len().iter_rows())
    counts["all"] = types.height
    tidy = (
        rows.rename({"time_series_type": "type"})
        .filter(pl.col("type").is_in(TYPE_ORDER))
        .with_columns(
            label=pl.col("model").replace_strict(MODEL_LABELS),
            group=pl.col("type").replace_strict(TYPE_LABELS),
        )
        .unpivot(
            on=["mae", "crps"],
            index=["type", "band", "model", "label", "group"],
            variable_name="metric",
            value_name="score",
        )
    )

    def score(model: str, series_type: str, band: str, metric: str) -> float:
        row = tidy.filter(
            pl.col("model") == model,
            pl.col("type") == series_type,
            pl.col("band") == band,
            pl.col("metric") == metric,
        )
        assert row.height == 1, (model, series_type, band, metric)
        return float(row["score"][0])

    short = BAND_LABELS[0]
    for series_type in ("PV", "Wind", "Raw Flow", "all"):
        assert score(BEST_POINT, series_type, short, "mae") < score(
            HEURISTIC, series_type, short, "mae"
        ), f"title: best point arm MAE at days 0 to 2, {series_type}"
        assert score(BEST_QUANTILE, series_type, short, "crps") < score(
            HEURISTIC, series_type, short, "crps"
        ), f"title: quantile arm CRPS at days 0 to 2, {series_type}"
    behind = [
        score(arm, series_type, band, metric) > score(HEURISTIC, series_type, band, metric)
        for arm in (BEST_POINT, BEST_QUANTILE)
        for series_type in TYPE_ORDER
        for band in BAND_ORDER[1:]
        for metric in ("mae", "crps")
    ]
    share_behind = sum(behind) / len(behind)
    assert share_behind > 0.5, "the title says the heuristic is usually ahead from day 3"

    mae = _facet(
        data=tidy.filter(pl.col("metric") == "mae"),
        title="Mean absolute error of the ensemble mean, MAE (% of capacity; smaller is better)",
    )
    crps = _facet(
        data=tidy.filter(pl.col("metric") == "crps"),
        title=(
            "Continuous ranked probability score, CRPS, of the whole ensemble "
            "(% of capacity; smaller is better)"
        ),
    )
    title = (
        "XGBoost beats the manual heuristic at days 0 to 2 for PV, wind, and Raw Flow, and is "
        "usually behind it from day 3"
    )
    subtitle = [
        (
            "Each dot is the mean over the series of a type of each series' mean score, as a "
            "percentage of the series' own effective capacity; zero is a perfect forecast. "
            "The last column averages all 31 series. The scores come from 53 weekly ECMWF "
            "ensemble forecast runs, July 2025 to June 2026 (fold A), on the rows that all five "
            f"models scored. XGBoost scores are the {_seeds_text(seeds=seeds)}. "
            "No intervals are shown."
        ),
        (
            "Lead is the time from a run's initialisation to the valid time: days 0 to 2 is "
            "under 72 hours, days 3 to 10 is 72 to 263 hours, and 11+ is 264 hours and more. "
        ),
        (
            "The best arms were chosen after seeing earlier results (post hoc). The quantile "
            "arm trains on 9 quantile levels and is scored on its 51 members at each of 9 "
            "levels (459 values)."
        ),
    ]
    chart = charts.figure(
        panels=[mae, crps], number=13, title=title, subtitle=subtitle, figure_planning=None
    )
    text = examples.FigureText(
        number=13,
        title=title,
        subtitle=subtitle,
        panel_titles=[
            "Mean absolute error of the ensemble mean, MAE (% of capacity; smaller is better)",
            (
                "Continuous ranked probability score, CRPS, of the whole ensemble "
                "(% of capacity; smaller is better)"
            ),
            *[f"Column: {TYPE_LABELS[t]} ({counts[t]} series)" for t in TYPE_ORDER],
        ],
        x_title="Lead (days)",
        y_title="(none: the panel title carries the metric, unit and direction)",
        legend=list(MODEL_LABELS.values()),
        notes=[
            (
                f"In {share_behind:.0%} of the {len(behind)} comparisons of an XGBoost arm with "
                "the heuristic at days 3 to 10 and 11+ (both arms, both metrics, five columns), "
                "the heuristic's error is lower."
            )
        ],
    )
    return Figure(text=text, chart=chart)


def figure_14(*, seeds: list[int]) -> Figure:
    """Build Figure 14: CRPS against months of training data, as a difference from 15 months."""
    contrasts = pl.read_parquet(
        inputs.OUTPUT_ROOT
        / "report"
        / FOLD
        / ("contrasts_seeds" + "".join(map(str, seeds)) + ".parquet")
    ).filter(
        pl.col("reference") == CLEANED,
        pl.col("lead_band") == "all",
        pl.col("metric") == "crps",
        pl.col("arm").is_in(list(F1_ARMS.values())),
    )
    months_of = {arm: months for months, arm in F1_ARMS.items()}
    steps = contrasts.with_columns(
        months=pl.col("arm").replace_strict(months_of),
        difference=pl.col("difference") * PERCENT,
        lower_95=pl.col("lower_95") * PERCENT,
        upper_95=pl.col("upper_95") * PERCENT,
    ).select("group", "months", "difference", "lower_95", "upper_95")
    zero = pl.DataFrame(
        {
            "group": CURVE_GROUPS,
            "months": [15] * len(CURVE_GROUPS),
            "difference": [0.0] * len(CURVE_GROUPS),
            "lower_95": [0.0] * len(CURVE_GROUPS),
            "upper_95": [0.0] * len(CURVE_GROUPS),
        }
    ).cast({"months": pl.Int64})
    data = (
        pl.concat([steps.cast({"months": pl.Int64}), zero])
        .filter(pl.col("group").is_in(CURVE_GROUPS))
        .with_columns(label=pl.col("group").replace_strict(CURVE_LABELS))
    )
    everything = data.filter(pl.col("group") == "all types").sort("months")
    values = everything["difference"].to_list()
    gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    assert all(gap > 0 for gap in gaps), "the title says more months lower the all-types CRPS"
    assert gaps[-1] == min(gaps), "the title says the last 3 months add the least"
    assert everything.filter(pl.col("months") == 6)["lower_95"][0] > 0, (
        "the title relies on the 6-month interval excluding zero"
    )

    base = alt.Chart().encode(
        x=alt.X(
            "months:O",
            sort=STEPS_ORDER,
            title="Months of training data",
            axis=alt.Axis(labelAngle=0, grid=False),
        ),
    )
    zero_rule = alt.Chart().mark_rule(color=MID_GREY, strokeDash=[3, 3], aria=False)
    zero_rule = zero_rule.encode(y=alt.datum(0))  # ty: ignore[unresolved-attribute]
    band = base.mark_rule(color=ocf.DATA_BLUE, strokeWidth=1.5, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lower_95:Q", title=None), y2="upper_95:Q"
    )
    line = base.mark_line(
        color=ocf.DATA_BLUE,
        aria=False,
        point=alt.OverlayMarkDef(size=45, color=ocf.DATA_BLUE, filled=True),
    ).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("difference:Q", title=None)
    )
    panel = (
        alt.layer(zero_rule, band, line)
        .properties(
            width=PANEL_WIDTH_PX + 6,
            height=PANEL_HEIGHT_PX + 20,
        )
        .facet(
            column=alt.Column(
                "label:N",
                sort=[CURVE_LABELS[g] for g in CURVE_GROUPS],
                title=None,
                header=alt.Header(labelOrient="top", labelFontSize=11, labelAnchor="start"),
            ),
            data=data,
        )
        .resolve_scale(y="independent")
        .properties(
            title=alt.TitleParams(
                charts.wrapped(text=PANEL_TITLE_14, width=85),
                anchor="start",
            )
        )
    )
    title = (
        "Over all series types, more months of training data lower the CRPS, and the last 3 "
        "months add the least"
    )
    subtitle = [
        (
            "Each dot is the continuous ranked probability score (CRPS) of a model trained on "
            "only the last 6, 9, or 12 months of the 15-month training window, minus the CRPS of "
            "the production arm with cleaning, which uses all 15 months. Zero (dashed line, and "
            "the 15-month dot by definition) is no change; above zero is a worse forecast."
        ),
        (
            "Each vertical line is a 95% interval from resampling whole months of the validation "
            "year, July 2025 to June 2026 (53 weekly forecast runs, fold A), with the same months "
            f"drawn for both arms. Scores are the {_seeds_text(seeds=seeds)}. A "
            "shorter window drops whole seasons from the training data. The first column also "
            "includes the one battery and the one biofuel series."
        ),
    ]
    chart = charts.figure(
        panels=[panel], number=14, title=title, subtitle=subtitle, figure_planning=None
    )
    text = examples.FigureText(
        number=14,
        title=title,
        subtitle=subtitle,
        panel_titles=[
            PANEL_TITLE_14,
            *[f"Column: {CURVE_LABELS[g]}" for g in CURVE_GROUPS],
        ],
        x_title="Months of training data",
        y_title="(none: the panel title carries the unit and direction)",
        legend=["(none: one colour)"],
        notes=[
            "All-types differences from the 15-month arm, in points of capacity: "
            + ", ".join(
                f"{m} months {v:+.2f}"
                for m, v in zip(everything["months"].to_list(), values, strict=True)
            )
            + "."
        ],
    )
    return Figure(text=text, chart=chart)


def main() -> None:
    """Write the text of Figures 13 and 14, then render them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--png-dir", type=Path, default=None, help="also write PNG renders here")
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    figures = {13: figure_13(seeds=args.seeds), 14: figure_14(seeds=args.seeds)}
    examples.write_text_sections(texts=[figure.text for figure in figures.values()])
    print(f"wrote {examples.TEXT_FILE}")
    if args.text_only:
        return
    for number, figure in figures.items():
        path = examples.save(chart=figure.chart, number=number)
        print(f"wrote {path}")
        if args.png_dir is not None:
            args.png_dir.mkdir(parents=True, exist_ok=True)
            figure.chart.save(args.png_dir / f"quick_wins_figure_{number}.png", scale_factor=2)


if __name__ == "__main__":
    main()
