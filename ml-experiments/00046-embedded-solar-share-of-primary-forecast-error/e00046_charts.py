"""Draw experiment 00046's figures from the saved results, as SVG, raw data first.

1. The residual of the `d1l` median against the oracle shift, binned, by lead band.
2. The four planned contrasts with their intervals and the 0.35-point threshold.
3. The four contrasts by valid month (the sign check).
4. The absolute score of every arm.
5. One week of one series' forecast with and without the ENS shift.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00046-*/e00046_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00046_common as common
import plotting.ocf_theme as ocf
import polars as pl

CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 480
WRAP_CHARACTERS: Final[int] = 100
SCOPE: Final[str] = (
    "Five Primary substations, daylight half-hours of April to September, 73 forecast runs one "
    "every 5 days: July to September 2025 and April to June 2026."
)
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C3": "C3: ENS shift minus d1l",
    "C4": "C4: CAMS shift minus d1l",
    "C5": "C5: CAMS shift minus ENS shift",
    "C6": "C6: ENS shift minus d0",
}
CONTRAST_COLOURS: Final[dict[str, str]] = {
    "C3": ocf.BRAND_ORANGE,
    "C4": ocf.DATA_BLUE,
    "C5": ocf.DATA_PURPLE,
    "C6": ocf.DATA_GREEN,
}
ARM_LABELS: Final[dict[str, str]] = {
    "d0": "d0: best guess with radiation inputs",
    "d1l": "d1l: without radiation inputs",
    "d1l_ens_shift": "d1l + ENS shift",
    "d1l_cams_shift": "d1l + CAMS shift (oracle)",
    "d1l_ens_shift_headline": "d1l + ENS shift, headline capacity",
    "d1l_cams_shift_headline": "d1l + CAMS shift, headline capacity",
    "d1l_ens_mean_shift": "d1l + ENS ensemble-mean shift",
}
MONTH_NAMES: Final[dict[int, str]] = {
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
}
GREY: Final[str] = "#8A8A8A"


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _zero_rule(*, x: float = 0.0) -> alt.Chart:
    return alt.Chart(pl.DataFrame({"x": [x]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def figure_binned() -> alt.TopLevelMixin:
    """Mean `d1l` residual in 2-point bins of the oracle shift, one panel per lead band."""
    data = pl.read_parquet(common.RESULTS_DIR / "binned.parquet")
    diagnostic = pl.read_parquet(common.RESULTS_DIR / "diagnostic.parquet").filter(
        pl.col("split") == "lead band"
    )
    bands = [name for name, _low, _high in common.LEAD_BANDS]
    panels = []
    for band in bands:
        part = data.filter(pl.col("band") == band)
        slope = float(diagnostic.filter(pl.col("group") == band)["slope"][0])
        line = pl.DataFrame({"x": [-20.0, 30.0]}).with_columns(
            ideal=-pl.col("x"), fitted=slope * pl.col("x")
        )
        base = alt.Chart(line).encode(x="x:Q")
        panels.append(
            alt.layer(
                _zero_rule(),
                base.mark_line(color=GREY, strokeDash=[4, 3]).encode(y="ideal:Q"),  # ty: ignore[unresolved-attribute]
                base.mark_line(color=ocf.DATA_BLUE).encode(y="fitted:Q"),  # ty: ignore[unresolved-attribute]
                alt.Chart(part)
                .mark_point(filled=True, size=40, color=ocf.BRAND_ORANGE)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X("shift_points:Q", title="Oracle shift (points of capacity)"),
                    y=alt.Y("residual_points:Q", title="Mean residual (points)"),
                ),
            ).properties(
                width=PLOT_WIDTH,
                height=110,
                title=alt.TitleParams(f"Lead {band}; fitted slope {slope:.2f}", anchor="start"),
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The forecast without radiation inputs under-uses the solar anomaly",
            lines=[
                "Mean residual of the d1l median (actual minus forecast, points of capacity) in "
                "2-point bins of the oracle shift (fitted capacity times the CAMS solar anomaly "
                "over effective capacity). Orange dots: bins of 50 or more half-hours. Grey "
                "dashed line: slope -1, a forecast with no solar information beyond the "
                "climatology. Blue line: fitted slope. " + SCOPE
            ],
        )
    )


def figure_contrasts() -> alt.TopLevelMixin:
    """The four planned contrasts with their intervals and the threshold."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "contrasts.parquet")
        .filter(
            pl.col("contrast").is_in(list(CONTRAST_LABELS))
            & (pl.col("scope") == "verdict series")
            & (pl.col("block") == "month")
            & (pl.col("n_resamples") == 50_000)
        )
        .with_columns(
            label=pl.col("contrast").replace_strict(CONTRAST_LABELS),
            kind=pl.when(pl.col("contrast").is_in(["C3", "C4"]))
            .then(pl.lit("deciding, 97.5% interval"))
            .otherwise(pl.lit("exploratory, 95% interval")),
        )
    )
    order = list(CONTRAST_LABELS.values())
    base = alt.Chart(data).encode(y=alt.Y("label:N", sort=order, title=None))
    points = base.mark_point(filled=True, size=90).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Quantile score difference (points of capacity; more negative is better)",
        ),
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(domain=list(CONTRAST_COLOURS), range=list(CONTRAST_COLOURS.values())),
            legend=None,
        ),
    )
    bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x="lower:Q",
        x2="upper:Q",
        color=alt.Color("contrast:N", legend=None),
        strokeDash=alt.StrokeDash("kind:N", legend=alt.Legend(title=None, orient="top")),
    )
    return alt.layer(_zero_rule(), _zero_rule(x=-0.35), bars, points).properties(
        width=PLOT_WIDTH,
        height=150,
        title=_title(
            text="The planned contrasts of experiment 46",
            lines=[
                "Each row is the first arm's quantile score minus the second's, over the scored "
                "rows. Dot: estimate. Line: interval from re-drawing whole months of run "
                "initialisation and seeds (solid: 97.5%, Bonferroni across the two deciding "
                "contrasts; dashed: 95%, exploratory). Left rule at -0.35: the smallest gain "
                "worth porting. Right rule at 0: same score. " + SCOPE
            ],
        ),
    )


def figure_by_month() -> alt.TopLevelMixin:
    """The mean paired difference of the four contrasts in each valid month."""
    data = pl.read_parquet(common.RESULTS_DIR / "by_month.parquet").with_columns(
        month=pl.col("valid_month").replace_strict(MONTH_NAMES),
        label=pl.col("contrast").replace_strict(CONTRAST_LABELS),
    )
    order = list(MONTH_NAMES.values())
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:N", sort=order, title="Valid month (local)"),
            y=alt.Y(
                "difference:Q",
                title="Quantile score difference (points; more negative is better)",
            ),
            color=alt.Color(
                "contrast:N",
                scale=alt.Scale(
                    domain=list(CONTRAST_COLOURS), range=list(CONTRAST_COLOURS.values())
                ),
                legend=None,
            ),
        )
        .properties(width=PLOT_WIDTH, height=90)
        .facet(row=alt.Row("label:N", sort=list(CONTRAST_LABELS.values()), title=None))
        .properties(
            title=_title(
                text="Does the sign of each contrast hold in every month?",
                lines=[
                    "Mean quantile score difference of each contrast in each local valid month "
                    "(six months, so six blocks). A contrast counts only if every month has "
                    "the same sign. " + SCOPE
                ],
            )
        )
    )


def figure_absolute() -> alt.TopLevelMixin:
    """The mean quantile score of every arm with its 95% interval."""
    data = pl.read_parquet(common.RESULTS_DIR / "absolute.parquet").with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS)
    )
    order = list(ARM_LABELS.values())
    base = alt.Chart(data).encode(y=alt.Y("label:N", sort=order, title=None))
    return alt.layer(
        base.mark_rule(color=ocf.DATA_BLUE).encode(x="qs_lower:Q", x2="qs_upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=80, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "qs:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            )
        ),
    ).properties(
        width=PLOT_WIDTH,
        height=190,
        title=_title(
            text="Absolute score of every arm",
            lines=[
                "Quantile score (the CRPS estimate; 1 point is 1% of effective capacity), mean "
                "over three seeds. Dot: estimate. Line: 95% interval from re-drawing whole months "
                "and seeds; the arms rise and fall together, so these intervals are much wider "
                "than the paired contrasts'. " + SCOPE
            ],
        ),
    )


def figure_week() -> alt.TopLevelMixin:
    """One week of one series' forecast with and without the ENS shift."""
    pieces = []
    for arm, label in (("d1l", "d1l"), ("d1l_ens_shift", "d1l + ENS shift")):
        frame = (
            pl.read_parquet(common.ARMS_DIR / f"{arm}.parquet")
            .join(
                common.series_table().select("time_series_id", "site"),
                on="time_series_id",
            )
            .filter((pl.col("site") == "D10") & (pl.col("seed") == 0))
        )
        origin = frame.filter(
            pl.col("power_fcst_init_time") >= pl.datetime(2025, 7, 20, time_zone="UTC")
        )["power_fcst_init_time"].min()
        pieces.append(
            frame.filter(pl.col("power_fcst_init_time") == origin)
            .with_columns(
                day=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                / 1440.0,
                arm=pl.lit(label),
            )
            .filter(pl.col("day") <= 7.0)
        )
    data = pl.concat(pieces)
    actual = data.filter(pl.col("arm") == "d1l")
    base = alt.Chart(data).encode(
        x=alt.X(
            "day:Q", title="Days after the forecast origin", axis=alt.Axis(values=list(range(8)))
        ),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=["d1l", "d1l + ENS shift"], range=[GREY, ocf.BRAND_ORANGE]),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    return alt.layer(
        base.mark_area(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Power over effective capacity"), y2="q95:Q"
        ),
        base.mark_line().encode(y="q50:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(actual).mark_line(color="black", strokeWidth=1.2).encode(x="day:Q", y="actual:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=PLOT_WIDTH,
        height=220,
        title=_title(
            text="One week of one Primary's forecast, with and without the ENS shift",
            lines=[
                (
                    "Median (line) and 5% to 95% band of the corrected forecast, seed 0, one run "
                    "from July 2025. Black line: observed power. Nights are included; the shift "
                    "is zero where the climatological solar output is zero."
                )
            ],
        ),
    )


def main() -> None:
    """Write every figure."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_residual_against_shift": figure_binned,
        "figure_2_contrasts": figure_contrasts,
        "figure_3_by_month": figure_by_month,
        "figure_4_absolute": figure_absolute,
        "figure_5_week": figure_week,
    }
    for name, build in figures.items():
        build().save(CHART_DIR / f"{name}.svg")
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
