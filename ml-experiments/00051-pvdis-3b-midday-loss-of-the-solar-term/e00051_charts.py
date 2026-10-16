"""Draw experiment 51's figures from the saved results, as SVG, raw data first.

1. The error left after the oracle shift against the clear-sky index, by lead day.
2. The same by hour of day.
3. The same by Primary (anonymised labels).
4. The residual of the `d1l` median against the oracle shift, with the coefficient-1 line.
5. The out-of-fold repairs and their contrasts.
6. The share of actuals below the 0.05 quantile in sunny rows.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00051-*/e00051_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00051_data import WORK_DIR

RESULTS_DIR: Final[Path] = WORK_DIR / "results"
CHART_DIR: Final[Path] = WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 520
WRAP_CHARACTERS: Final[int] = 105
GREY: Final[str] = "#B5B5B5"
SCOPE: Final[str] = (
    "Five Primary substations, June to August, valid 09:00 to 15:00 UTC, 1 to 4 days ahead, "
    "forecasts from experiment 46."
)
SERIES_COLOURS: Final[dict[str, str]] = {
    "D9": ocf.BRAND_ORANGE,
    "D10": ocf.DATA_BLUE,
    "D11": ocf.DATA_SKY,
    "D14": ocf.DATA_PURPLE,
    "D16": ocf.DATA_GREEN,
}
HOUR_GROUPS: Final[dict[str, str]] = {
    "09-10 UTC": ocf.DATA_BLUE,
    "11-13 UTC": ocf.BRAND_ORANGE,
    "14-15 UTC": ocf.DATA_GREEN,
}
DOTS_PER_PANEL: Final[int] = 400
DOTS_IN_SCATTER: Final[int] = 1200
SAMPLE_SEED: Final[int] = 51
E_TITLE: Final[str] = "Error left after the oracle shift (points of capacity; 0 is exact)"


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _cases() -> pl.DataFrame:
    return pl.read_parquet(RESULTS_DIR / "cases.parquet").with_columns(
        csi_bin=((pl.col("clear_sky_index") * 10).floor().clip(0, 9) / 10 + 0.05),
        hour_group=pl.when(pl.col("hour") <= 10)
        .then(pl.lit("09-10 UTC"))
        .when(pl.col("hour") <= 13)
        .then(pl.lit("11-13 UTC"))
        .otherwise(pl.lit("14-15 UTC")),
    )


def _zero() -> alt.Chart:
    return alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#666666").encode(y="y:Q")  # ty: ignore[unresolved-attribute]


CSI_TITLE: Final[str] = "CAMS clear-sky index (0 overcast, 1 cloudless)"
E_AXIS: Final[alt.Y] = alt.Y("e:Q", title=E_TITLE)
CSI_AXIS: Final[alt.X] = alt.X(
    "clear_sky_index:Q",
    title=CSI_TITLE,
    scale=alt.Scale(domain=[0, 1.1]),
)


def figure_by_lead_day(*, cases: pl.DataFrame) -> alt.TopLevelMixin:
    """Per-case error and its binned mean, one panel per lead day."""
    panels = []
    for band in ("day+1", "day+2", "day+3"):
        part = cases.filter(pl.col("day_band") == band)
        raw = (
            alt.Chart(part.sample(n=DOTS_PER_PANEL, seed=SAMPLE_SEED))  # ty: ignore[unresolved-attribute]
            .mark_circle(size=8, opacity=0.25, color=GREY)
            .encode(x=CSI_AXIS, y=E_AXIS)
        )
        means = part.group_by("csi_bin").agg(e=pl.col("e").mean()).sort("csi_bin")
        line = (
            alt.Chart(means)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True, color=ocf.BRAND_ORANGE)
            .encode(x=alt.X("csi_bin:Q"), y=alt.Y("e:Q"))
        )
        panels.append(
            alt.layer(raw, line, _zero()).properties(width=PLOT_WIDTH, height=150, title=band)
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The oracle shift is too large, on cloudy and sunny rows, at every lead day",
            lines=[
                "Grey dots: a random sample of 400 of the about 1,200 forecast cases in each "
                "panel, each averaged over 3 seeds. Orange line: mean of all the panel's cases in "
                "bins of 0.1 clear-sky index. Above 0 the shift lowered the forecast too far; below 0 it raised it too far. "
                + SCOPE
            ],
        )
    )


def figure_by_hour(*, cases: pl.DataFrame) -> alt.TopLevelMixin:
    """Binned mean error for three groups of hours."""
    means = cases.group_by("hour_group", "csi_bin").agg(e=pl.col("e").mean()).sort("csi_bin")
    return (
        alt.layer(
            alt.Chart(means)
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("csi_bin:Q", title=CSI_TITLE, scale=alt.Scale(domain=[0, 1.1])),
                y=E_AXIS,
                color=alt.Color(
                    "hour_group:N",
                    scale=alt.Scale(domain=list(HOUR_GROUPS), range=list(HOUR_GROUPS.values())),
                    legend=alt.Legend(title="Valid time of day", orient="top"),
                ),
            ),
            _zero(),
        )
        .properties(
            width=PLOT_WIDTH,
            height=240,
            title=_title(
                text="The error after the oracle shift looks alike at every hour of the day",
                lines=[
                    "Mean in bins of 0.1 clear-sky index, for three groups of valid hours. If "
                    "clipping drove the over-correction, the hours around solar noon (11-13 UTC) "
                    "would sit above the others in the sunniest bins. " + SCOPE
                ],
            ),
        )
        .configure_axis(grid=True)
    )


def figure_by_series(*, cases: pl.DataFrame) -> alt.TopLevelMixin:
    """Binned mean error for each Primary."""
    means = cases.group_by("site", "csi_bin").agg(e=pl.col("e").mean()).sort("csi_bin")
    return alt.layer(
        alt.Chart(means)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("csi_bin:Q", title=CSI_TITLE, scale=alt.Scale(domain=[0, 1.1])),
            y=E_AXIS,
            color=alt.Color(
                "site:N",
                scale=alt.Scale(domain=list(SERIES_COLOURS), range=list(SERIES_COLOURS.values())),
                legend=alt.Legend(title="Primary (anonymised)", orient="top"),
            ),
        ),
        _zero(),
    ).properties(
        width=PLOT_WIDTH,
        height=240,
        title=_title(
            text="All five Primaries over-correct, with different amounts",
            lines=["Mean in bins of 0.1 clear-sky index, for each Primary. " + SCOPE],
        ),
    )


def figure_residual_against_shift(*, cases: pl.DataFrame) -> alt.TopLevelMixin:
    """Residual of the d1l median against the oracle shift, with the coefficient-1 line."""
    binned = (
        cases.with_columns(shift_bin=(pl.col("shift") / 4).floor() * 4 + 2)
        .group_by("shift_bin")
        .agg(r=pl.col("r").mean(), rows=pl.len())
        .filter(pl.col("rows") >= 20)
        .sort("shift_bin")
    )
    raw = (
        alt.Chart(cases.sample(n=DOTS_IN_SCATTER, seed=SAMPLE_SEED))
        .mark_circle(size=8, opacity=0.25, color=GREY)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("shift:Q", title="Oracle shift (points of capacity; positive means sunnier)"),
            y=alt.Y("r:Q", title="Actual minus d1l median (points of capacity)"),
        )
    )
    diagonal = pl.DataFrame({"shift": [-20.0, 20.0], "r": [20.0, -20.0]})
    full = (
        alt.Chart(diagonal)  # ty: ignore[unresolved-attribute]
        .mark_line(color="#666666", strokeDash=[4, 3])
        .encode(x="shift:Q", y="r:Q")
    )
    points = (
        alt.Chart(binned)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.BRAND_ORANGE)
        .encode(x="shift_bin:Q", y="r:Q")
    )
    return alt.layer(raw, full, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="Following the oracle shift fully would over-correct by a third or more",
            lines=[
                "Grey dots: a random sample of 1,200 of the 3,575 forecast cases. Orange line: "
                "mean of all cases in bins of 4 points of shift. "
                "Dashed line: the residual if the shift were exactly right (coefficient 1, slope "
                "-1). The orange line is flatter than the dashed one. " + SCOPE
            ],
        ),
    )


def figure_repairs() -> alt.TopLevelMixin:
    """Out-of-fold repairs and their contrasts, at 95%."""
    labels = {
        "A hinge - unrepaired": "Hinge (clipping) minus unrepaired",
        "L line - unrepaired": "Straight line minus unrepaired",
        "B coefficient - unrepaired": "Fitted coefficient minus unrepaired",
        "A hinge - L line": "Hinge minus straight line",
    }
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE]
    data = (
        pl.read_parquet(RESULTS_DIR / "repairs.parquet")
        .filter(
            pl.col("contrast").is_in(list(labels))
            & pl.col("scope").is_in(["all", "day+2 and day+3", "sunny (csi >= 0.7)"])
        )
        .with_columns(label=pl.col("contrast").replace_strict(labels))
    )
    base = alt.Chart(data).encode(
        y=alt.Y("scope:N", title=None, sort=["all", "day+2 and day+3", "sunny (csi >= 0.7)"]),
        yOffset=alt.YOffset("label:N", sort=list(labels.values())),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=list(labels.values()), range=colours),
            legend=alt.Legend(
                title=None, orient="top", columns=2, labelLimit=320, symbolType="circle"
            ),
        ),
    )
    return alt.layer(
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Mean absolute error of the median, difference (points; negative is better)",
            )
        ),
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#666666").encode(x="x:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=PLOT_WIDTH,
        height=220,
        title=_title(
            text="A fitted coefficient repairs the median as well as a hinge does",
            lines=[
                "Each repair is fitted on odd ISO weeks of the valid time and scored on even "
                "weeks, then the reverse. Dot: estimate. Line: 95% interval from resampling whole "
                "weeks of forecast start (13 weeks). Rule at 0: same error. " + SCOPE
            ],
        ),
    )


def figure_coverage() -> alt.TopLevelMixin:
    """Share of actuals below the 0.05 quantile in sunny rows, by arm and lead day."""
    labels = {
        "d1l": "No solar term (d1l)",
        "d1l_ens_shift": "ENS shift, with member spread",
        "d1l_ens_mean_shift": "ENS shift, ensemble mean only",
        "d1l_cams_shift": "Oracle shift (true sunshine)",
    }
    data = pl.read_parquet(RESULTS_DIR / "coverage.parquet").with_columns(
        label=pl.col("arm").replace_strict(labels)
    )
    colours = [GREY, ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("band:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("label:N", sort=list(labels.values())),
            y=alt.Y("below_q05:Q", title="Share of actuals below the 0.05 quantile"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=list(labels.values()), range=colours),
                legend=alt.Legend(
                    title=None, orient="top", columns=2, labelLimit=320, symbolType="circle"
                ),
            ),
        )
    )
    target = (
        alt.Chart(pl.DataFrame({"y": [0.05]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#666666", strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    return alt.layer(bars, target).properties(
        width=PLOT_WIDTH,
        height=240,
        title=_title(
            text="The lower tail of sunny rows is too wide even with the true sunshine",
            lines=[
                "Sunny rows have a clear-sky index of 0.7 or more. A well-calibrated 0.05 quantile "
                "has 0.05 of actuals below it (dashed line). Below the line the lower tail is too "
                "wide. " + SCOPE
            ],
        ),
    )


def main() -> None:
    """Write the SVG figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    figures = {
        "figure_1_by_lead_day": figure_by_lead_day(cases=cases),
        "figure_2_by_hour": figure_by_hour(cases=cases),
        "figure_3_by_primary": figure_by_series(cases=cases),
        "figure_4_residual_against_shift": figure_residual_against_shift(cases=cases),
        "figure_5_repairs": figure_repairs(),
        "figure_6_coverage": figure_coverage(),
    }
    for name, figure in figures.items():
        figure.save(CHART_DIR / f"{name}.svg")
        print("wrote", CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
