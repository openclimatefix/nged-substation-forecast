"""Draw experiment 00045's figures 5 to 17 as SVG.

Every number a chart shares with the write-up is read from the parquet tables that
`e00045_report.py` prints into `report.txt` (`scores/*.parquet`, `scores/post_hoc_tables/*.parquet`,
`scores/family_intervals.parquet`, `capacity_weekly*.parquet`, `outside_checks.parquet`). Two
figures recompute a path with the experiment's own functions, which no table stores: figure 5 reads
the fitted paths of one unit, and figure 15 refits the symmetric path of the gap diagnostic.

No chart shows the output of a solar farm. A farm appears only as its normalised injected capacity
path (Solar farm A to F, as a share of the path's mean), never in megawatts. Substations are named,
in megawatts, on calendar dates. Each SVG is afterwards optimised with
`npx svgo@4 --multipass --precision=1 --final-newline`, which `--optimise` runs.

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python \
        ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_charts.py --optimise
"""

import argparse
import subprocess
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Final

import altair as alt
import e00045_common as common
import e00045_inject as inject
import e00045_model as model
import e00045_outputs as outputs
import e00045_score as score
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
import vl_convert

WIDTH: Final[int] = 560
"""The width of the plot area, in pixels, which fits the docs page's text column."""
WRAP: Final[int] = 92
SCORES: Final[Path] = score.SCORES_DIR
POST_HOC_TABLES: Final[Path] = SCORES / "post_hoc_tables"
PREVIEW_DIR: Final[Path] = common.DATA_DIR / "figure_previews"
MAX_SVG_BYTES: Final[int] = 500_000

FROZEN: Final[str] = "frozen asymmetric (a=4, r=30)"
POST_HOC: Final[str] = "post hoc asymmetric (a=2, r=10)"
FREE: Final[str] = "free weekly"
SYMMETRIC: Final[str] = "symmetric"
NON_DECREASING: Final[str] = "non-decreasing"
SEASON_YEAR: Final[str] = "season-year constant"
FIT_ORDER: Final[list[str]] = [SEASON_YEAR, POST_HOC, FROZEN, NON_DECREASING, SYMMETRIC, FREE]
"""The fits from the smallest to the largest path error in the headline figure."""
FIT_COLOURS: Final[dict[str, str]] = {
    SEASON_YEAR: ocf.DATA_GREEN,
    POST_HOC: ocf.DATA_BLUE,
    FROZEN: ocf.BRAND_ORANGE,
    NON_DECREASING: "#6B6B6B",
    SYMMETRIC: ocf.DATA_PURPLE,
    FREE: ocf.DATA_SKY,
}
"""Each fit's colour. Only the five main data colours are used, plus a neutral grey for the
non-decreasing fit, which equals the frozen asymmetric fit and is drawn dashed beneath it."""
REPORT_FIT_LABELS: Final[dict[str, str]] = {
    "asymmetric": FROZEN,
    "non-decreasing": NON_DECREASING,
    "free weekly": FREE,
    "symmetric": SYMMETRIC,
    "season-year constant": SEASON_YEAR,
    "post hoc (a=2, r=10)": POST_HOC,
}
"""Maps the fit names in `report.txt` to the chart labels."""
SCORE_FIT_LABELS: Final[dict[str, str]] = {
    "asymmetric": FROZEN,
    "non_decreasing": NON_DECREASING,
    "free": FREE,
    "symmetric": SYMMETRIC,
    "season_year": SEASON_YEAR,
}
VARIANT_LABELS: Final[dict[str, str]] = {
    "semi_synthetic": "semi-synthetic baseline",
    "real_plus_injection": "real series plus injection",
}
VARIANT_COLOURS: Final[dict[str, str]] = {
    "semi-synthetic baseline": ocf.BRAND_ORANGE,
    "real series plus injection": ocf.DATA_BLUE,
}
PRIMARY_LABEL: Final[dict[str, str]] = {
    site: common.PRIMARY_NAMES[site]
    if common.PRIMARY_STATUS[site] == "clean"
    else f"{common.PRIMARY_NAMES[site]} ({common.PRIMARY_STATUS[site]})"
    for site in common.PRIMARIES
}
PRIMARY_ORDER: Final[list[str]] = [PRIMARY_LABEL[site] for site in common.PRIMARIES]
SERIES_ID_TO_LABEL: Final[dict[int, str]] = {
    common.PRIMARY_SERIES_IDS[site]: PRIMARY_LABEL[site] for site in common.PRIMARIES
}
GREY: Final[str] = "#999999"
LIGHT_GREY: Final[str] = "#D9D9D9"
FAMILY_ORDER: Final[list[str]] = [
    "ramps",
    "steps",
    "falls",
    "transfers",
    "seasonal loads",
    "constant path",
]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    """Return a left-aligned title with subtitle lines wrapped to the plot area's width."""
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=textwrap.wrap(text, WRAP - 14),
        subtitle=wrapped,
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _fit_colour(
    *, title: str | None = None, legend: bool = True, fits: list[str] | None = None
) -> alt.Color:
    domain = fits or FIT_ORDER
    return alt.Color(
        "fit:N",
        title=title,
        sort=domain,
        scale=alt.Scale(domain=domain, range=[FIT_COLOURS[f] for f in domain]),
        legend=alt.Legend(orient="top", columns=3, labelLimit=260) if legend else None,
    )


def _zero_rule(*, x: float = 0.0, dash: tuple[int, int] = (4, 3)) -> alt.Chart:
    """Return a vertical reference rule that draws in every facet panel."""
    return alt.Chart().mark_rule(color=GREY, strokeDash=list(dash)).encode(x=alt.datum(x))  # ty: ignore[unresolved-attribute]


def _horizontal_rule(*, y: float) -> alt.Chart:
    """Return a horizontal reference rule that draws in every facet panel."""
    return alt.Chart().mark_rule(color=GREY, strokeDash=[4, 3]).encode(y=alt.datum(y))  # ty: ignore[unresolved-attribute]


def _variant_encodings() -> dict[str, alt.Color | alt.Shape]:
    """Return the colour and shape encodings of the two baseline variants, one merged legend."""
    domain = list(VARIANT_COLOURS)
    return {
        "color": alt.Color(
            "variant:N",
            title=None,
            sort=domain,
            scale=alt.Scale(domain=domain, range=list(VARIANT_COLOURS.values())),
            legend=alt.Legend(orient="top", labelLimit=300),
        ),
        "shape": alt.Shape(
            "variant:N",
            title=None,
            sort=domain,
            scale=alt.Scale(domain=domain, range=["square", "circle"]),
            legend=alt.Legend(orient="top", labelLimit=300),
        ),
    }


def _key(*, colours: dict[str, str]) -> alt.Chart:
    """Return a one-pixel chart whose only job is to draw a legend of line colours."""
    domain = list(colours)
    return (
        alt.Chart(pl.DataFrame({"fit": domain, "x": [0.0] * len(domain)}))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=1, opacity=1)
        .encode(
            x=alt.X("x:Q", axis=None),
            color=alt.Color(
                "fit:N",
                title=None,
                sort=domain,
                scale=alt.Scale(domain=domain, range=list(colours.values())),
                legend=alt.Legend(
                    orient="top",
                    columns=len(domain),
                    labelLimit=320,
                    symbolType="stroke",
                    symbolStrokeWidth=3,
                    symbolOpacity=1,
                ),
            ),
        )
        .properties(width=1, height=1)
    )


def _read(*, name: str, folder: Path = SCORES) -> pl.DataFrame:
    return pl.read_parquet(folder / f"{name}.parquet")


def _chart_fit_names(*, frame: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    return frame.with_columns(fit=pl.col("fit").replace(mapping))


def _both_fit_tables(*, name: str) -> pl.DataFrame:
    """Return a breakdown table with the frozen scoring's fits and the post hoc pair."""
    frozen = _chart_fit_names(frame=_read(name=name), mapping=SCORE_FIT_LABELS)
    post_hoc = _chart_fit_names(
        frame=_read(name=name, folder=POST_HOC_TABLES), mapping={"asymmetric": POST_HOC}
    )
    return pl.concat([frozen, post_hoc], how="vertical_relaxed")


# ---------------------------------------------------------------------------------------------
# Figure 5: one injected path through the fits
# ---------------------------------------------------------------------------------------------

FIG5_PRIMARY: Final[str] = "D16"
FIG5_FARM: Final[str] = "D"
FIG5_SEED: Final[int] = 1000
FIG5_CASES: Final[dict[str, str]] = {
    "ramp_50": "Ramp: capacity rises by 50%",
    "step_20": "Step: capacity rises by 20% in one week",
    "fall_30_26": "Fall: capacity falls by 30% for 26 weeks, then returns",
}


def _unit_estimates(*, folder: Path, fits: list[str], cases: tuple[str, ...]) -> pl.DataFrame:
    """Return the truth and each fit's estimate of one unit's capacity path, as a share of its mean.

    Args:
        folder: The scoring folder (`scores` or `scores/post_hoc`).
        fits: The fit names in that folder to read.
        cases: Injection cases to read.

    Returns:
        One row per case, fit (`truth` for the known path), and week with the share of the
        path's mean capacity.
    """
    stem = f"scoring_semi_synthetic_{FIG5_PRIMARY}_{FIG5_FARM}"
    units = pl.read_parquet(folder / f"{stem}_units.parquet").filter(
        (pl.col("seed") == FIG5_SEED) & pl.col("case").is_in(list(cases))
    )
    paths = pl.read_parquet(folder / f"{stem}_paths.parquet").filter(
        (pl.col("seed") == FIG5_SEED) & pl.col("case").is_in(list(cases))
    )
    ratios = score.read_ratios(out_dir=folder)
    rows = []
    for unit in units.iter_rows(named=True):
        increment = np.array(unit["increment"], dtype=float)
        observed = np.array(unit["observed"], dtype=bool)
        base = float(unit["base_capacity"])
        truth = inject.true_path(increment=increment, base_capacity=base, variant="semi_synthetic")
        scale = float(np.mean((base + increment)[observed]))
        weeks = np.arange(len(truth))
        starts = [common.week_start(week=int(w)).date() for w in weeks]
        rows.extend(
            {"case": unit["case"], "fit": "truth", "week_start": d, "share": v / scale}
            for d, v in zip(starts, truth, strict=True)
        )
        for path in paths.filter(pl.col("case") == unit["case"]).iter_rows(named=True):
            if path["fit"] not in fits:
                continue
            estimate = inject.estimate_path(
                with_injection=np.array(path["with_injection"], dtype=float),
                baseline=None,
                ratio=ratios[(unit["primary"], path["fit"])],
                variant="semi_synthetic",
                base_capacity=base,
            )
            label = SCORE_FIT_LABELS.get(path["fit"], POST_HOC if folder != SCORES else path["fit"])
            rows.extend(
                {"case": unit["case"], "fit": label, "week_start": d, "share": v / scale}
                for d, v, seen in zip(starts, estimate, observed, strict=True)
                if seen and np.isfinite(v)
            )
    return pl.DataFrame(rows)


def figure_05() -> alt.VConcatChart:
    """Method: one injected path through the fits, three cases at one Primary."""
    cases = tuple(FIG5_CASES)
    frozen = _unit_estimates(folder=SCORES, fits=list(score.FIT_NAMES), cases=cases)
    post_hoc = _unit_estimates(
        folder=SCORES / score.POST_HOC_DIR_NAME, fits=["asymmetric"], cases=cases
    ).filter(pl.col("fit") != "truth")
    post_hoc = post_hoc.with_columns(fit=pl.lit(POST_HOC))
    data = pl.concat([frozen, post_hoc])
    first_seen = data.filter(pl.col("fit") != "truth")["week_start"].min()
    data = data.filter(pl.col("week_start") >= first_seen)
    domain = ["truth", *FIT_ORDER]
    colours = [ocf.DATA_BLUE if False else "#222222", *[FIT_COLOURS[f] for f in FIT_ORDER]]
    panels = []
    for case, label in FIG5_CASES.items():
        d = data.filter(pl.col("case") == case)
        lines = (
            alt.Chart(d)  # ty: ignore[unresolved-attribute]
            .mark_line(clip=True, interpolate="step-after")
            .encode(
                x=alt.X(
                    "week_start:T",
                    title=None,
                    axis=alt.Axis(format="%Y", tickCount="year", labelOverlap=True),
                ),
                y=alt.Y(
                    "share:Q",
                    title="Share of mean capacity",
                    scale=alt.Scale(zero=False, nice=True),
                ),
                color=alt.Color(
                    "fit:N",
                    title=None,
                    sort=domain,
                    scale=alt.Scale(domain=domain, range=colours),
                    legend=alt.Legend(orient="top", columns=3, labelLimit=260)
                    if case == "ramp_50"
                    else None,
                ),
                strokeWidth=alt.StrokeWidth(
                    "fit:N",
                    legend=None,
                    scale=alt.Scale(domain=domain, range=[3.5, 1.6, 1.6, 1.6, 1.0, 1.6, 1.6]),
                ),
                strokeDash=alt.StrokeDash(
                    "fit:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=domain,
                        range=[[1, 0], [1, 0], [1, 0], [1, 0], [5, 3], [1, 0], [1, 0]],
                    ),
                ),
            )
            .properties(width=WIDTH, height=110, title=label)
        )
        panels.append(lines)
    return alt.vconcat(*panels).properties(
        title=_title(
            text=(
                "The frozen asymmetric fit misses a 26-week fall; the free, symmetric, and post "
                "hoc fits follow it"
            ),
            lines=[
                (
                    "Thick dark line: the known capacity path injected into a semi-synthetic "
                    "baseline"
                    f"at {common.PRIMARY_NAMES[FIG5_PRIMARY]} using Solar farm "
                    f"{FIG5_FARM}'s sunshine "
                    "pattern, as a share of the path's mean capacity (dimensionless, so no "
                    "megawatts). Coloured lines: each fit's estimate after the recovery-ratio "
                    "correction, observed weeks only. The non-decreasing fit equals the frozen "
                    "asymmetric fit (dashed grey beneath orange)."
                ),
            ],
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 6: the penalty grid
# ---------------------------------------------------------------------------------------------


def figure_06() -> alt.VConcatChart:
    """Method: the tuning-set results over the penalty grid, both attempts."""
    import json

    frozen = json.loads(score.FROZEN_PATH.read_text())
    attempts = {
        "Second attempt (used)": frozen["tuning_table"],
        "First attempt (superseded)": frozen["first_attempt_real_baseline"]["tuning_table"],
    }
    metrics = {
        "steps_constant_median": ("Constant-path false steps per Primary-year, median", 0.5, "max"),
        "fall_26_recovered": ("Share of a 26-week 30% fall recovered", 0.5, "min"),
        "path_error_mixture": ("Path error, mixture of cases", None, None),
    }
    rows = []
    for attempt, table in attempts.items():
        for row in table:
            for key, (label, limit, kind) in metrics.items():
                value = row[key]
                met = (
                    "no limit"
                    if limit is None
                    else "limit met"
                    if (value <= limit if kind == "max" else value >= limit)
                    else "limit not met"
                )
                mark = ""
                if (row["a"], row["r"]) == (frozen["a"], frozen["r"]):
                    mark = " F"
                if (row["a"], row["r"]) == score.POST_HOC_PAIR:
                    mark = " P"
                rows.append(
                    {
                        "attempt": attempt,
                        "metric": label,
                        "a": f"a = {row['a']:g}",
                        "r": f"r = {row['r']:g}",
                        "value": value,
                        "text": f"{value:.2f}{mark}",
                        "limit": met,
                    }
                )
    data = pl.DataFrame(rows)
    a_order = [f"a = {a:g}" for a in (0.5, 1.0, 2.0, 4.0)]
    r_order = ["r = 30", "r = 10"]
    panels = []
    for metric in [v[0] for v in metrics.values()]:
        d = data.filter(pl.col("metric") == metric)
        base = alt.Chart(d).encode(
            x=alt.X("a:N", sort=a_order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("r:N", sort=r_order, title=None),
        )
        rect = base.mark_rect(stroke="white").encode(  # ty: ignore[unresolved-attribute]
            color=alt.Color(
                "limit:N",
                title="Against the tuning limit",
                scale=alt.Scale(
                    domain=["limit met", "limit not met", "no limit"],
                    range=["#9FEBC8", "#FFB59E", "#E6E6E6"],
                ),
                legend=alt.Legend(orient="top")
                if metric == next(iter(metrics.values()))[0]
                else None,
            )
        )
        text = base.mark_text(fontSize=12).encode(text="text:N")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(rect, text)
            .properties(width=210, height=60)
            .facet(column=alt.Column("attempt:N", title=None, sort=list(attempts)))
            .properties(title=metric)
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="No penalty pair met the tuning limits; the frozen pair was the nearest miss",
            lines=[
                (
                    "Each cell is one pair of penalties on the tuning units (Solar farms A and B). "
                    "a is the weight on weekly capacity changes and r is the factor by which a "
                    "fall costs more than a rise. F marks the frozen pair (a=4, r=30) and P the "
                    "post hoc pair (a=2, r=10), which was picked after the scoring run. Green: the "
                    "tuning limit is met (at most 0.5 constant-path false steps per Primary-year; "
                    "at least 0.5 of a 26-week fall recovered). The path error has no limit."
                ),
            ],
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 7: the headline
# ---------------------------------------------------------------------------------------------


def figure_07() -> alt.Chart:
    """Headline: path error by case family and fit with 95% intervals, scoring units."""
    data = (
        _read(name="family_intervals")
        .with_columns(
            fit=pl.col("fit").replace(REPORT_FIT_LABELS),
            variant=pl.col("variant").replace(VARIANT_LABELS),
        )
        .filter(pl.col("family") != "seasonal loads")
    )
    families = [f for f in FAMILY_ORDER if f != "seasonal loads"]
    encodings = _variant_encodings()
    base = alt.Chart().encode(
        y=alt.Y("fit:N", sort=FIT_ORDER, title=None, axis=alt.Axis(labelLimit=230)),
        yOffset=alt.YOffset("variant:N", sort=list(VARIANT_COLOURS)),
        color=encodings["color"],  # ty: ignore[invalid-argument-type]
    )
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Path error (share of mean capacity; smaller is better)"),
        x2="high:Q",
    )
    dot = base.mark_point(filled=True, size=45, opacity=1).encode(  # ty: ignore[unresolved-attribute]
        x="path_error:Q", shape=encodings["shape"]
    )
    ref = _zero_rule(x=score.LIMIT_I)
    panel = alt.layer(ref, rule, dot).properties(width=WIDTH, height=108)
    return panel.facet(  # ty: ignore[invalid-return-type]
        row=alt.Row(
            "family:N",
            sort=families,
            title=None,
            header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11, labelPadding=4),
        ),
        data=data,
    ).properties(
        title=_title(
            text=(
                "On the semi-synthetic baseline the season-year constant has the lowest path "
                "error and the free weekly fit the highest"
            ),
            lines=[
                (
                    "Dot: median over the 8 Primary substations of each Primary's median path "
                    "error (mean absolute error of the weekly capacity path against the known "
                    "path, as a share of its mean) on the scoring units (Solar farms D, E, and F, "
                    "4 path seeds). Line: 95% interval from resampling Primaries, then farm-seed "
                    "units within each. Dashed line: 0.15, the plan's ceiling for ramps and steps. "
                    "All rows are exploratory except the frozen asymmetric fit's ramps and steps."
                )
            ],
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 8: the detection curve
# ---------------------------------------------------------------------------------------------

STEP_PERCENT: Final[dict[str, float]] = {
    "step_05": 5.0,
    "step_10": 10.0,
    "step_20": 20.0,
    "step_30": 30.0,
}
SEASON_LABELS: Final[dict[str, str]] = {
    "Apr-Sep": "Step placed April to September",
    "Oct-Mar": "Step placed October to March",
}
DETECTION_FITS: Final[list[str]] = [POST_HOC, FROZEN, SYMMETRIC, FREE, SEASON_YEAR]


def figure_08() -> alt.VConcatChart:
    """The detection curve: share of steps detected against step size, by season and fit."""
    data = (
        _read(name="detection_by_primary")
        .with_columns(
            fit=pl.col("fit").replace(REPORT_FIT_LABELS),
            size=pl.col("case").replace_strict(STEP_PERCENT, return_dtype=pl.Float64),
            season=pl.col("season").replace(SEASON_LABELS),
        )
        .filter(pl.col("fit").is_in(DETECTION_FITS))
    )
    pooled = (
        data.group_by("fit", "season", "size")
        .agg(share=(pl.col("share_detected") * pl.col("n")).sum() / pl.col("n").sum())
        .sort("fit", "season", "size")
    )
    x = alt.X(
        "size:Q",
        title="Step size (% of the Primary's mean capacity)",
        axis=alt.Axis(values=[5, 10, 20, 30]),
        scale=alt.Scale(domain=[4, 31]),
    )
    y = alt.Y(
        "share:Q",
        title="Share of steps detected",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(values=[0, 0.5, 0.8, 1]),
    )
    eighty = _horizontal_rule(y=0.8)
    lines = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(x=x, y=y, color=_fit_colour(title=None))
    )
    top = (
        alt.layer(eighty, lines)
        .properties(width=265, height=150)
        .facet(
            column=alt.Column(
                "season:N",
                title=None,
                sort=list(SEASON_LABELS.values()),
                header=alt.Header(labelFontSize=11),
            ),
            data=pooled,
        )
        .properties(title="All 8 Primaries pooled, by fit")
    )
    per_primary = data.filter(pl.col("fit") == POST_HOC).with_columns(
        share=pl.col("share_detected"),
        name=pl.col("primary").replace(PRIMARY_LABEL),
    )
    thin = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2, opacity=0.9)
        .encode(
            x=x,
            y=y,
            detail="name:N",
            color=alt.Color(
                "name:N",
                title="Primary",
                sort=PRIMARY_ORDER,
                scale=alt.Scale(
                    domain=PRIMARY_ORDER,
                    range=[
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_SKY,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        "#6B6B6B",
                        "#B0B0B0",
                        "#222222",
                    ],
                ),
                legend=alt.Legend(orient="top", columns=4, labelLimit=240),
            ),
        )
    )
    bottom = (
        alt.layer(eighty, thin)
        .properties(width=265, height=150)
        .facet(
            column=alt.Column(
                "season:N",
                title=None,
                sort=list(SEASON_LABELS.values()),
                header=alt.Header(labelFontSize=11),
            ),
            data=per_primary,
        )
        .properties(title="Post hoc asymmetric fit (a=2, r=10), one line per Primary")
    )
    return (
        alt.vconcat(top, bottom)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text=(
                    "Only the free weekly fit comes near detecting 80% of steps, and the "
                    "season-year "
                    ""
                    "constant detects none from October to March"
                ),
                lines=[
                    (
                        "A step counts as detected when the estimate rises by at least half the "
                        "step within 3 weeks of it. Share of semi-synthetic steps detected, on the "
                        "scoring units. Dashed line: 80%, the share the plan's detection limit "
                        "asks for. Steps of 5%, 10%, 20%, and 30% of the Primary's mean capacity. "
                        "Each point in the lower panels rests on few units."
                    )
                ],
            )
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 9: falls
# ---------------------------------------------------------------------------------------------

FALL_LABELS: Final[dict[str, str]] = {
    "fall_30_26": "30% fall lasting 26 weeks",
    "fall_15_26": "15% fall lasting 26 weeks",
    "fall_30_8": "30% fall lasting 8 weeks",
}


def figure_09() -> alt.Chart:
    """Falls: the share of a temporary fall's depth that each fit shows."""
    data = _both_fit_tables(name="fall_recovery").with_columns(
        case=pl.col("case").replace(FALL_LABELS),
        variant=pl.col("variant").replace(VARIANT_LABELS),
    )
    encodings = _variant_encodings()
    base = alt.Chart().encode(
        y=alt.Y("fit:N", sort=FIT_ORDER, title=None, axis=alt.Axis(labelLimit=230)),
        yOffset=alt.YOffset("variant:N", sort=list(VARIANT_COLOURS)),
        color=encodings["color"],  # ty: ignore[invalid-argument-type]
    )
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "worst_primary:Q",
            title="Share of the fall's depth shown (median over Primaries; line to worst Primary)",
            scale=alt.Scale(domain=[-0.2, 1.6]),
        ),
        x2="fall_recovered:Q",
    )
    dot = base.mark_point(filled=True, size=50, opacity=1).encode(  # ty: ignore[unresolved-attribute]
        x="fall_recovered:Q", shape=encodings["shape"]
    )
    half = _zero_rule(x=0.5)
    panel = alt.layer(half, rule, dot).properties(width=WIDTH, height=110)
    return panel.facet(  # ty: ignore[invalid-return-type]
        row=alt.Row(
            "case:N",
            sort=list(FALL_LABELS.values()),
            title=None,
            header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11, labelPadding=4),
        ),
        data=data,
    ).properties(
        title=_title(
            text=(
                "The symmetric, free, and post hoc fits show a 26-week fall; the frozen pair and "
                "the season-year constant show none"
            ),
            lines=[
                (
                    "Share of a temporary fall's depth that the weekly capacity path shows during "
                    "the fall (0 means the path ignores the fall; 1 means it follows it fully; "
                    "more than 1 means it overshoots). Dot: median over the 8 Primaries on the "
                    "scoring units. Line: from the dot to the worst Primary. Dashed line: 0.5, the "
                    "plan's limit for a 26-week fall."
                )
            ],
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 10: transfers and seasonal load
# ---------------------------------------------------------------------------------------------

LOAD_LABELS: Final[dict[str, str]] = {
    "transfer_inside_mask": "Transfer inside the masked closure window",
    "transfer_outside_mask": "Transfer not logged, outside the mask",
    "seasonal_load_independent": "Seasonal load independent of the weather",
    "seasonal_load_sunshine": "Seasonal load that rises with sunshine",
}


def figure_10() -> alt.Chart:
    """Transfers and seasonal load: the error each leaves in the path, semi-synthetic baseline."""
    transfers = (
        _both_fit_tables(name="transfers")
        .select("variant", "fit", "case", value="path_error")
        .with_columns(measure=pl.lit("Path error (share of mean capacity)"))
    )
    loads = (
        _both_fit_tables(name="seasonal_load")
        .select("variant", "fit", "case", value="swing")
        .drop_nulls("value")
        .with_columns(measure=pl.lit("Seasonal swing of the error (share of mean capacity)"))
    )
    data = (
        pl.concat([transfers, loads])
        .filter(pl.col("variant") == "semi_synthetic")
        .with_columns(case=pl.col("case").replace(LOAD_LABELS))
    )
    order = list(LOAD_LABELS.values())
    chart = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, opacity=1)
        .encode(
            x=alt.X(
                "value:Q",
                title="Error left in the path (share of mean capacity; smaller is better)",
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y("fit:N", sort=FIT_ORDER, title=None, axis=alt.Axis(labelLimit=230)),
            color=_fit_colour(),
        )
        .properties(width=WIDTH, height=110)
    )
    return chart.facet(
        row=alt.Row(
            "case:N",
            sort=order,
            title=None,
            header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11, labelPadding=4),
        ),
        data=data,
    ).properties(
        title=_title(
            text=(
                "A transfer or a seasonal load leaves the least error in the season-year constant"
                " and the most in the free weekly fit"
            ),
            lines=[
                (
                    "Semi-synthetic baseline, scoring units, median over the 8 Primaries. "
                    "Transfers: the path error when load is moved between Primaries for a period, "
                    "inside the masked closure window or outside it. Seasonal loads: the seasonal "
                    "swing that the error of the path acquires when a seasonal load (independent "
                    "of the weather, or rising with sunshine) is added; a swing of 0 means the "
                    "path stays clear of the seasonal load."
                )
            ],
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 11: false steps and drift
# ---------------------------------------------------------------------------------------------

FALSE_FITS: Final[list[str]] = [SEASON_YEAR, POST_HOC, FROZEN, SYMMETRIC]


def _distribution_panel(
    *,
    data: pl.DataFrame,
    value: str,
    low: str | None,
    high: str | None,
    title: str,
    x_title: str,
    legend: bool = False,
) -> alt.LayerChart:
    base = alt.Chart(data).encode(
        y=alt.Y("name:N", sort=PRIMARY_ORDER, title=None, axis=alt.Axis(labelLimit=230)),
        yOffset=alt.YOffset("fit:N", sort=FALSE_FITS),
        color=_fit_colour(legend=legend, fits=FALSE_FITS),
    )
    dot = base.mark_point(filled=True, size=40, opacity=1).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(f"{value}:Q", title=x_title)
    )
    layers = [dot]
    if low and high:
        layers.insert(
            0,
            base.mark_rule(strokeWidth=1.5, opacity=0.8).encode(x=f"{low}:Q", x2=f"{high}:Q"),  # ty: ignore[unresolved-attribute]
        )
    return alt.layer(*layers).properties(width=WIDTH, height=190, title=title)  # ty: ignore[invalid-return-type]


def figure_11() -> alt.VConcatChart:
    """False steps: the spread over placebo permutations and constant-path units, and the drift."""
    data = (
        _read(name="false_step_distribution")
        .with_columns(
            fit=pl.col("fit").replace(REPORT_FIT_LABELS),
            name=pl.col("primary").replace(PRIMARY_LABEL),
        )
        .filter(pl.col("fit").is_in(FALSE_FITS))
    )
    constant = data.filter(pl.col("sky") == "constant path")
    sky = data.filter(pl.col("sky") == "placebo sky")
    limit = (
        alt.Chart(pl.DataFrame({"x": [score.LIMIT_FALSE_WORST]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    half = (
        alt.Chart(pl.DataFrame({"x": [score.LIMIT_FALSE_MEDIAN]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[1, 2])
        .encode(x="x:Q")
    )
    steps_title = (
        "False steps per Primary-year (mean; line: 10th to 90th percentile; smaller is better)"
    )
    a = alt.layer(
        half,
        limit,
        _distribution_panel(
            data=constant,
            value="mean_steps",
            low="p10_steps",
            high="p90_steps",
            title="Constant injected path (12 units per Primary)",
            x_title=steps_title,
            legend=True,
        ),
    )
    b = alt.layer(
        half,
        limit,
        _distribution_panel(
            data=sky,
            value="mean_steps",
            low="p10_steps",
            high="p90_steps",
            title="Placebo sky (200 permutations of the sunshine pattern per Primary)",
            x_title=steps_title,
        ),
    )
    drift_limit = (
        alt.Chart(pl.DataFrame({"x": [score.LIMIT_DRIFT]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    drift = alt.layer(
        _zero_rule(),
        drift_limit,
        _distribution_panel(
            data=data.filter(pl.col("sky") == "placebo sky"),
            value="mean_drift",
            low=None,
            high=None,
            title="Ratchet drift on the placebo sky",
            x_title="Rise of the path over 5 years (share of mean capacity; near 0 is better)",
        ),
    )
    return alt.vconcat(a, b, drift).properties(
        title=_title(
            text=(
                "The post hoc pair makes more false steps than the season-year constant at most "
                "Primaries"
            ),
            lines=[
                (
                    "A false step is a rise of 10% of the real series' mean capacity within 4 "
                    "weeks that the injected path does not contain. Dots: mean over replicates. "
                    "Dotted and dashed lines: the plan's limits for the median Primary (0.5) and "
                    "the worst Primary (1.0), or 0.05 for the drift. Four fits; the free fit is "
                    "omitted because it gives constant-path false steps of 11 or more."
                )
            ],
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 12: vintages
# ---------------------------------------------------------------------------------------------


def figure_12() -> alt.VConcatChart:
    """Real-time vintages: the first estimate of a week against the final fit."""
    by_age = _read(name="vintages_by_age").select("age", first="error_vintage", final="error_final")
    long = by_age.unpivot(index="age", variable_name="which", value_name="error").with_columns(
        which=pl.col("which").replace(
            {
                "first": "Estimate made when the week had just ended",
                "final": "Estimate from the final fit on all data",
            }
        )
    )
    order = [
        "Estimate made when the week had just ended",
        "Estimate from the final fit on all data",
    ]
    injection = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "age:Q",
                title="Age of the week when the first estimate was made (weeks)",
                axis=alt.Axis(values=[1, 2, 4, 8, 13, 26]),
            ),
            y=alt.Y(
                "error:Q", title="Path error (share of mean capacity)", scale=alt.Scale(zero=True)
            ),
            color=alt.Color(
                "which:N",
                title=None,
                sort=order,
                scale=alt.Scale(domain=order, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top", columns=1, labelLimit=420),
            ),
        )
        .properties(
            width=WIDTH,
            height=150,
            title="Injected paths, frozen asymmetric fit: revision by the age of the week",
        )
    )
    final = pl.read_parquet(common.DATA_DIR / "capacity_weekly.parquet")
    vintages = pl.read_parquet(common.DATA_DIR / "capacity_weekly_vintages.parquet")
    ratios = outputs.vintage_ratios(final=final, vintages=vintages).with_columns(
        name=pl.col("time_series_id").replace_strict(SERIES_ID_TO_LABEL, return_dtype=pl.String),
        fit=pl.col("variant").replace({"frozen": FROZEN, "alt_post_hoc": POST_HOC}),
    )
    base = alt.Chart(ratios).encode(
        y=alt.Y("name:N", sort=PRIMARY_ORDER, title=None, axis=alt.Axis(labelLimit=230)),
        yOffset=alt.YOffset("fit:N", sort=[POST_HOC, FROZEN]),
        color=alt.Color(
            "fit:N",
            title=None,
            sort=[POST_HOC, FROZEN],
            scale=alt.Scale(
                domain=[POST_HOC, FROZEN], range=[FIT_COLOURS[POST_HOC], FIT_COLOURS[FROZEN]]
            ),
            legend=alt.Legend(orient="top", labelLimit=400),
        ),
    )
    real = alt.layer(
        _zero_rule(x=1.0),
        base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "p10:Q",
                title=(
                    "First estimate of the latest week over its final-fit value (1 means no "
                    "revision)"
                ),
                scale=alt.Scale(domain=[0.9, 1.3]),
            ),
            x2="p90:Q",
        ),
        base.mark_point(filled=True, size=45, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("median_ratio:Q", scale=alt.Scale(domain=[0.9, 1.3]))
        ),
    ).properties(
        width=WIDTH,
        height=190,
        title="Real series: median and 10th to 90th percentile over the origins",
    )
    return (
        alt.vconcat(injection, real)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text=(
                    "On the real series the first estimate of the newest week is typically "
                    "0% to 10% "
                    ""
                    "above its final value; on injected paths its error is 3.6 times the final "
                    "error"
                ),
                lines=[
                    (
                        "Top: median over the 8 Primaries on the scoring units, from vintages "
                        "every 4 weeks over the last 3 years. Bottom: at each origin Monday the "
                        "fit uses only earlier data; the dot is the median, over origins, of that "
                        "fit's value for the week just ended divided by the final fit's value for "
                        "the same week."
                    )
                ],
            )
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 13: interval calibration
# ---------------------------------------------------------------------------------------------


def figure_13() -> alt.VConcatChart:
    """Interval calibration: coverage of the known path against the nominal level."""
    frozen = _read(name="coverage_summary_shown").with_columns(fit=pl.lit(FROZEN))
    post_hoc = _read(name="coverage_summary_post_hoc").with_columns(fit=pl.lit(POST_HOC))
    allrows = pl.concat([frozen, post_hoc])
    pooled = (
        allrows.filter(pl.col("group") == "all")
        .unpivot(
            index=["fit", "weeks"],
            on=["coverage_50", "coverage_80", "coverage_90"],
            variable_name="nominal",
            value_name="coverage",
        )
        .with_columns(
            nominal=pl.col("nominal").str.replace("coverage_", "").cast(pl.Float64) / 100,
            weeks=pl.col("weeks").replace({"non-step": "Weeks with no step", "step": "Step weeks"}),
        )
    )
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.4, 0.95], "y": [0.4, 0.95]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")
    )
    lines = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "nominal:Q",
                title="Nominal coverage of the interval",
                axis=alt.Axis(values=[0.5, 0.8, 0.9], format="%"),
                scale=alt.Scale(domain=[0.4, 0.95]),
            ),
            y=alt.Y(
                "coverage:Q",
                title="Share of weeks where the known path lies inside",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=_fit_colour(title=None, fits=[POST_HOC, FROZEN]),
        )
    )
    top = (
        alt.layer(diagonal, lines)
        .properties(width=265, height=190)
        .facet(
            column=alt.Column(
                "weeks:N",
                title=None,
                sort=["Weeks with no step", "Step weeks"],
                header=alt.Header(labelFontSize=11),
            ),
            data=pooled,
        )
    )
    per = allrows.filter(pl.col("group").str.starts_with("primary:")).with_columns(
        name=pl.col("group").str.replace("primary:", "").replace(PRIMARY_LABEL),
        weeks=pl.col("weeks").replace({"non-step": "Weeks with no step", "step": "Step weeks"}),
    )
    base = alt.Chart().encode(
        y=alt.Y("name:N", sort=PRIMARY_ORDER, title=None, axis=alt.Axis(labelLimit=230)),
        yOffset=alt.YOffset("fit:N", sort=[POST_HOC, FROZEN]),
        color=_fit_colour(legend=False, fits=[POST_HOC, FROZEN]),
    )
    target = (
        alt.Chart(pl.DataFrame({"x": [0.9]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    bottom = (
        alt.layer(
            target,
            base.mark_point(filled=True, size=40, opacity=1).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "coverage_90:Q",
                    title="Coverage of the 90% interval",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1]),
                )
            ),
        )
        .properties(width=265, height=170)
        .facet(
            column=alt.Column(
                "weeks:N",
                title=None,
                sort=["Weeks with no step", "Step weeks"],
                header=alt.Header(labelFontSize=11),
            ),
            data=per,
        )
    )
    return alt.vconcat(top, bottom).properties(
        title=_title(
            text=(
                "The 90% interval covers the known path in 6 to 7 weeks in 10, and in fewer step "
                "weeks"
            ),
            lines=[
                (
                    "Residual block bootstrap, whole path refitted, on 94 scoring units. Top: "
                    "coverage at 50%, 80%, and 90% against the dashed line of perfect calibration "
                    "(points above it would mean intervals too wide). Bottom: 90% coverage per "
                    "Primary; dashed line: 90%. The plan asks for 85% to 95% at weeks with no step "
                    "and at least 60% at step weeks."
                )
            ],
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 14: the real series
# ---------------------------------------------------------------------------------------------


def _real_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the weekly capacity rows of the two shown variants, and experiment 00040's anchors."""
    final = pl.read_parquet(common.DATA_DIR / "capacity_weekly.parquet").filter(
        pl.col("variant").is_in(["season_year_constant", "alt_post_hoc"])
    )
    final = final.with_columns(
        name=pl.col("time_series_id").replace_strict(SERIES_ID_TO_LABEL, return_dtype=pl.String),
        fit=pl.col("variant").replace(
            {"season_year_constant": SEASON_YEAR, "alt_post_hoc": POST_HOC}
        ),
    )
    anchors = (
        pl.read_parquet(common.DATA_DIR / "outside_checks.parquet")
        .filter(
            (pl.col("check") == "season_year_anchor")
            & (pl.col("variant") == "season_year_constant")
        )
        .select("primary_name", "season_year", value="outside_value")
        .with_columns(
            week_start=pl.date(pl.col("season_year"), 7, 1),
            name=pl.col("primary_name").replace(
                {common.PRIMARY_NAMES[s]: PRIMARY_LABEL[s] for s in common.PRIMARIES}
                | {"Wrangle T2": PRIMARY_LABEL["D18"]}
            ),
        )
    )
    return final, anchors


def figure_14() -> alt.VConcatChart:
    """Headline on real data: the weekly capacity of each Primary with its 90% band."""
    final, anchors = _real_frames()
    seen = final.filter(~pl.col("unobserved") | pl.col("capacity_mw").is_not_null())
    x = alt.X(
        "week_start:T",
        title=None,
        axis=alt.Axis(format="%Y", tickCount="year", labelOverlap=True),
        scale=alt.Scale(
            domain=[common.FIRST_WEEK_START.isoformat(), common.LAST_DATA_END.isoformat()]
        ),
    )
    colours = {SEASON_YEAR: ocf.DATA_GREEN, POST_HOC: ocf.DATA_BLUE}
    panels = [_key(colours=colours)]  # ty: ignore[invalid-argument-type]
    for site in common.PRIMARIES:
        label = PRIMARY_LABEL[site]
        mine = seen.filter(pl.col("name") == label)
        layers = []
        for fit, colour in colours.items():
            part = mine.filter(pl.col("fit") == fit)
            drawn = GREY if site == "D5" else colour
            layers.append(
                alt.Chart(part.filter(pl.col("low_mw").is_not_null()))  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.22, interpolate="step-after", color=drawn)
                .encode(x=x, y=alt.Y("low_mw:Q", title=None), y2="high_mw:Q")
            )
            layers.append(
                alt.Chart(part)  # ty: ignore[unresolved-attribute]
                .mark_line(strokeWidth=1.6, interpolate="step-after", color=drawn)
                .encode(x=x, y=alt.Y("capacity_mw:Q", title=None, scale=alt.Scale(zero=True)))
            )
        band, line = layers[0], layers[1]
        del band, line
        scenario = (
            alt.Chart(mine.filter(pl.col("fit") == SEASON_YEAR))  # ty: ignore[unresolved-attribute]
            .transform_fold(["scenario_dayL_mw", "scenario_sunload_mw"], as_=["scenario", "mw"])
            .mark_line(strokeWidth=1, strokeDash=[4, 3], color="#222222", interpolate="step-after")
            .encode(x=x, y="mw:Q", detail="scenario:N")
        )
        dots = (
            alt.Chart(anchors.filter(pl.col("name") == label))  # ty: ignore[unresolved-attribute]
            .mark_point(shape="diamond", filled=True, size=70, color="#222222", opacity=1)
            .encode(x="week_start:T", y="value:Q")
        )
        panels.append(
            alt.layer(*layers, scenario, dots).properties(
                width=WIDTH, height=70, title=alt.TitleParams(label, anchor="start", fontSize=11)
            )
        )
    return (
        alt.vconcat(*panels, spacing=4)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text=(
                    "The season-year constant and the post hoc path agree on level and "
                    "differ within "
                    ""
                    "each year"
                ),
                lines=[
                    (
                        "Weekly embedded-solar capacity in MW (the 99th percentile of half-hourly "
                        "AC output) behind each Primary substation. Green: season-year constant, "
                        "the recommended series. Blue: post hoc asymmetric fit (a=2, r=10), "
                        "offered for change within a year. Bands: 90% sampling interval. Dashed "
                        "black lines: the season-year constant under two systematic scenarios. "
                        "Black diamonds: experiment 00040's season-year estimate, at 1 July of the "
                        "year. Kirton (grey) is range-only; read it as a range. Weeks with fewer "
                        "than 3 usable days are bridged by widened intervals. Mount Bridge "
                        "Boston's season-year constant falls to about 0 MW from September 2023 to "
                        "August 2025, weeks that are all unobserved; do not read them."
                    )
                ],
            )
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 15: the gap between the post hoc and the symmetric path
# ---------------------------------------------------------------------------------------------


def _gap_paths() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the raw post hoc and symmetric paths per Primary, and the held falls.

    The post hoc path is the stored weekly capacity times its recovery ratio, and the symmetric
    path is refitted with `model.fit_path` at the post hoc pair's `a` and `r = 1`, as
    `e00045_outputs.gap_rows` does. A held fall is a run of 8 or more observed weeks 10% or more
    below the symmetric path's trailing 26-week maximum, in fewer than half of which the post hoc
    path is that far below its own.
    """
    final = pl.read_parquet(common.DATA_DIR / "capacity_weekly.parquet")
    inputs = inject.load_inputs()
    rows, falls = [], []
    spec = outputs.real.VARIANTS["alt_post_hoc"]
    for site in common.PRIMARIES:
        design = model.design_from_half_hourly(
            half_hourly=inputs.half_hourly, weeks=inputs.weeks, site=site
        )
        symmetric = model.fit_path(design=design, a=spec[1], r=1.0)
        part = final.filter(
            (pl.col("time_series_id") == common.PRIMARY_SERIES_IDS[site])
            & (pl.col("variant") == "alt_post_hoc")
        )
        index = np.array(
            [(d - common.FIRST_WEEK_START.date()).days // 7 for d in part["week_start"]]
        )
        asym = np.full(common.N_WEEKS, np.nan)
        observed = np.zeros(common.N_WEEKS, dtype=bool)
        asym[index] = part["capacity_mw"].fill_null(np.nan).to_numpy() * part["recovery_ratio"][0]
        observed[index] = ~part["unobserved"].to_numpy()
        rows.extend(
            {
                "name": PRIMARY_LABEL[site],
                "week_start": common.week_start(week=week).date(),
                "post hoc asymmetric": float(asym[week]),
                "symmetric": float(symmetric[week]),
            }
            for week in range(common.N_WEEKS)
            if observed[week]
        )
        for first, last in outputs.fall_episodes(path=symmetric, observed=observed):
            if outputs.below_level(path=asym, observed=observed, episode=(first, last)) < 0.5:
                falls.append(
                    {
                        "name": PRIMARY_LABEL[site],
                        "start": common.week_start(week=first).date(),
                        "end": common.week_start(week=last).date(),
                    }
                )
    return pl.DataFrame(rows), pl.DataFrame(
        falls, schema={"name": pl.String, "start": pl.Date, "end": pl.Date}
    )


def figure_15() -> alt.VConcatChart:
    """The gap between the post hoc and the symmetric paths on the real series."""
    paths, falls = _gap_paths()
    x = alt.X(
        "week_start:T",
        title=None,
        axis=alt.Axis(format="%Y", tickCount="year", labelOverlap=True),
        scale=alt.Scale(
            domain=[common.FIRST_WEEK_START.isoformat(), common.LAST_DATA_END.isoformat()]
        ),
    )
    colours = {"symmetric": ocf.DATA_PURPLE, "post hoc asymmetric": ocf.DATA_BLUE}
    panels = [_key(colours=colours)]  # ty: ignore[invalid-argument-type]
    for site in common.PRIMARIES:
        label = PRIMARY_LABEL[site]
        mine = paths.filter(pl.col("name") == label)
        shade = (
            alt.Chart(falls.filter(pl.col("name") == label))  # ty: ignore[unresolved-attribute]
            .mark_rect(color=ocf.BRAND_ORANGE, opacity=0.18)
            .encode(x="start:T", x2="end:T")
        )
        lines = [
            alt.Chart(mine)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.4, interpolate="step-after", color=colour)
            .encode(x=x, y=alt.Y(f"{column}:Q", title=None, scale=alt.Scale(zero=True)))
            for column, colour in colours.items()
        ]
        panels.append(
            alt.layer(shade, *lines).properties(
                width=WIDTH, height=70, title=alt.TitleParams(label, anchor="start", fontSize=11)
            )
        )
    return (
        alt.vconcat(*panels, spacing=4)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text=(
                    "Where the symmetric path falls for 8 weeks or more, the post hoc path often "
                    "holds its level"
                ),
                lines=[
                    (
                        "Fitted weekly capacity in MW before the recovery-ratio correction, "
                        "observed weeks only. Blue: post hoc asymmetric fit (a=2, r=10). Purple: "
                        "symmetric fit with the same a. Orange shading: a run of 8 or more weeks "
                        "in which the symmetric path is 10% or more below its own trailing 26-week "
                        "maximum and the post hoc path is not (a held fall). A held fall is not a "
                        "real fall, so check each shaded run against the switching log."
                    )
                ],
            )
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 16: the outside checks
# ---------------------------------------------------------------------------------------------

LEVEL_SOURCES: Final[dict[str, str]] = {
    "desnz_constituency_domestic": "DESNZ domestic installations by constituency",
    "desnz_local_authority": "DESNZ all sizes by local authority",
}


def figure_16() -> alt.VConcatChart:
    """The series against the DESNZ expected level and the register total."""
    checks = pl.read_parquet(common.DATA_DIR / "outside_checks.parquet").filter(
        pl.col("variant") == "season_year_constant"
    )
    names = {
        **{common.PRIMARY_NAMES[s]: PRIMARY_LABEL[s] for s in common.PRIMARIES},
        "Wrangle T1 + T2": "Wrangle T1 + T2",
    }
    order = [PRIMARY_LABEL[s] for s in common.PRIMARIES if s not in ("D15", "D18")] + [
        "Wrangle T1 + T2"
    ]
    level = checks.filter(pl.col("check") == "level").with_columns(
        name=pl.col("primary_name").replace(names),
        source=pl.col("source").replace(LEVEL_SOURCES),
    )
    band = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            pl.DataFrame({"low": [level["tolerance_low"][0]], "high": [level["tolerance_high"][0]]})
        )
        .mark_rect(color="#E3EBFF", opacity=0.8)
        .encode(x="low:Q", x2="high:Q")
    )
    base = alt.Chart(level).encode(
        y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=230)),
        yOffset=alt.YOffset("source:N", sort=list(LEVEL_SOURCES.values())),
        color=alt.Color(
            "source:N",
            title=None,
            sort=list(LEVEL_SOURCES.values()),
            scale=alt.Scale(
                domain=list(LEVEL_SOURCES.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(orient="top", columns=1, labelLimit=420),
        ),
    )
    dots = base.mark_point(filled=True, size=60, opacity=1).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "ratio:Q",
            title="Series over the outside figure (1 means they agree; log scale)",
            scale=alt.Scale(type="log", domain=[0.05, 6]),
            axis=alt.Axis(values=[0.1, 0.25, 0.5, 1, 2, 5]),
        ),
        shape=alt.Shape("result:N", title="Result", legend=alt.Legend(orient="top")),
    )
    one = (
        alt.Chart(pl.DataFrame({"x": [1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    top = alt.layer(band, one, dots).properties(
        width=WIDTH,
        height=190,
        title=(
            "Level: series against the outside estimate of the expected peak (blue band: tolerance)"
        ),
    )
    register = (
        checks.filter(pl.col("check") == "ecr")
        .with_columns(name=pl.col("primary_name").replace(names))
        .select("name", series="series_value", register="outside_value")
        .unpivot(index="name", variable_name="which", value_name="mw")
        .with_columns(
            which=pl.col("which").replace(
                {
                    "series": "Season-year constant, final fit",
                    "register": "Embedded generators of 1 MW or more in the register",
                }
            )
        )
    )
    reg_order = [
        "Season-year constant, final fit",
        "Embedded generators of 1 MW or more in the register",
    ]
    bottom = (
        alt.Chart(register)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60, opacity=1)
        .encode(
            x=alt.X("mw:Q", title="Capacity (MW)", scale=alt.Scale(zero=True)),
            y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=230)),
            color=alt.Color(
                "which:N",
                title=None,
                sort=reg_order,
                scale=alt.Scale(domain=reg_order, range=[ocf.DATA_GREEN, "#6B6B6B"]),
                legend=alt.Legend(orient="top", columns=1, labelLimit=420),
            ),
        )
        .properties(
            width=WIDTH,
            height=190,
            title="Register total: a floor, because the register is partial",
        )
    )
    return (
        alt.vconcat(top, bottom)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text=(
                    "The series agrees with DESNZ's domestic estimate at some Primaries "
                    "and exceeds "
                    ""
                    "the register floor at all"
                ),
                lines=[
                    (
                        "Season-year constant, final fit. Top: the series' level against two "
                        "DESNZ-based estimates of the expected peak, scaled to each Primary's "
                        "polygon; the all-sizes local-authority estimate is the less direct "
                        "comparison. Kirton is range-only and Wrangle T1 and T2 are summed. "
                        "Bottom: the series against the register of embedded generators, which is "
                        "partial, so the register is a floor."
                    )
                ],
            )
        )
    )


# ---------------------------------------------------------------------------------------------
# Figure 17: additivity
# ---------------------------------------------------------------------------------------------


def figure_17() -> alt.Chart:
    """Additivity: the error on the real series plus injection against the semi-synthetic error."""
    data = (
        _read(name="additivity")
        .with_columns(fit=pl.col("fit").replace(REPORT_FIT_LABELS))
        .with_columns(pl.col("semi_synthetic", "real_plus_injection").round(4))
    )
    data = data.filter(pl.col("fit").is_in([SEASON_YEAR, POST_HOC, FROZEN, SYMMETRIC, FREE]))
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, 0.5], "y": [0.0, 0.5]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")
    )
    points = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=14, opacity=0.5, clip=True)
        .encode(
            x=alt.X(
                "semi_synthetic:Q",
                title="Path error on the semi-synthetic baseline (share of mean capacity)",
                scale=alt.Scale(domain=[0, 0.5]),
            ),
            y=alt.Y(
                "real_plus_injection:Q",
                title="Path error, real series plus injection",
                scale=alt.Scale(domain=[0, 0.5]),
            ),
            color=_fit_colour(title=None),
        )
        .properties(width=WIDTH, height=300)
    )
    return alt.layer(diagonal, points).properties(  # ty: ignore[invalid-return-type]
        title=_title(
            text=(
                "Scored on the real series plus the injection, most paths have a smaller error "
                "than on the semi-synthetic baseline, and the two errors are only weakly related"
            ),
            lines=[
                (
                    "One dot per injected unit (Primary, Solar farm D, E, or F, path seed, and "
                    "case; constant paths excluded). x: path error when the injection is added to "
                    "the semi-synthetic baseline. y: path error when it is added to the real "
                    "series and the fit without the injection is subtracted. Dashed line: equal "
                    "errors. Smaller is better on both axes; dots below the line score better on "
                    "the real series."
                )
            ],
        )
    )


FIGURES: Final[dict[int, Callable[[], alt.TopLevelMixin]]] = {
    5: figure_05,
    6: figure_06,
    7: figure_07,
    8: figure_08,
    9: figure_09,
    10: figure_10,
    11: figure_11,
    12: figure_12,
    13: figure_13,
    14: figure_14,
    15: figure_15,
    16: figure_16,
    17: figure_17,
}


def main() -> None:
    """Draw the requested figures, save each as SVG and as a PNG preview, and optionally svgo."""
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", nargs="*", type=int, default=list(FIGURES))
    parser.add_argument("--optimise", action="store_true", help="run svgo on each SVG")
    args = parser.parse_args()
    common.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(exist_ok=True)
    for number in args.figures:
        path = common.DOCS_DIR / f"figure_{number:02d}.svg"
        chart = FIGURES[number]().configure_mark(aria=False)
        chart.save(path)
        if args.optimise:
            subprocess.run(
                ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
                check=True,
                capture_output=True,
            )
        size = path.stat().st_size
        assert size < MAX_SVG_BYTES, f"figure {number} is {size} bytes"
        png = vl_convert.svg_to_png(path.read_text(), scale=1.5)
        (PREVIEW_DIR / f"figure_{number:02d}.png").write_bytes(png)
        print(f"figure {number}: {size / 1000:.0f} KB")


if __name__ == "__main__":
    main()
