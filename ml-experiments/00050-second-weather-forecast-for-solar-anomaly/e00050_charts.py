"""Draw experiment 00050's figures from the saved results, as SVG.

1. The difference in correlation with the CAMS anomaly, GEFS and the blend minus ENS, by lead band,
   with 95% intervals (the headline).
2. Each arm's correlation with the CAMS anomaly by lead band (absolute skill).
3. The raw data: the CAMS anomaly against the ENS and GEFS anomalies, in bins.
4. The cross-validation error of GEFS and the blend as a ratio to ENS's, by held-out month.
5. The blend's weight on GEFS by lead band.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00050-*/e00050_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00050_common as common
import plotting.ocf_theme as ocf
import polars as pl

CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 440
WRAP_CHARACTERS: Final[int] = 100
SCOPE: Final[str] = (
    "Five Primary substations metered in MW, daylight half-hours of April to September, "
    "73 forecast runs one every 5 days: July to September 2025 and April to June 2026."
)
BANDS: Final[list[str]] = [name for name, _low, _high in common.LEAD_BANDS]
ARM_LABELS: Final[dict[str, str]] = {
    "ens": "ENS (ECMWF)",
    "gefs": "GEFS (NOAA)",
    "blend": "Fitted blend",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "ens": ocf.BRAND_ORANGE,
    "gefs": ocf.DATA_BLUE,
    "blend": ocf.DATA_GREEN,
}
MONTHS: Final[list[str]] = [
    "Apr 2024",
    "May 2024",
    "Jun 2024",
    "Jul 2024",
    "Aug 2024",
    "Sep 2024",
    "Apr 2025",
    "May 2025",
    "Jun 2025",
]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _rule(*, value: float) -> alt.Chart:
    return alt.Chart(pl.DataFrame({"x": [value]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def figure_headline() -> alt.TopLevelMixin:
    """The difference in correlation from ENS by lead band, with 95% intervals."""
    data = pl.read_parquet(common.RESULTS_DIR / "anomaly_intervals.parquet").filter(
        pl.col("band") != "all"
    )
    labels = {"gefs minus ens": "GEFS minus ENS", "blend minus ens": "Fitted blend minus ENS"}
    data = data.with_columns(label=pl.col("contrast").replace_strict(labels))
    order = list(labels.values())
    base = alt.Chart(data).encode(
        y=alt.Y("band:N", sort=BANDS, title="Lead time", scale=alt.Scale(paddingOuter=0.1)),
        yOffset=alt.YOffset("label:N", sort=order),
        color=alt.Color(
            "label:N",
            sort=order,
            scale=alt.Scale(domain=order, range=[ocf.DATA_BLUE, ocf.DATA_GREEN]),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    return alt.layer(
        _rule(value=0.0),
        base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Correlation with the CAMS anomaly minus ENS's (more positive is better)",
            )
        ),
    ).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="GEFS forecasts the sunshine anomaly worse than ENS, and the blend adds nothing",
            lines=[
                "Dot: estimate. Line: 95% interval from re-drawing the 7 months in which runs were "
                "started. Rule at 0: same correlation as ENS. " + SCOPE
            ],
        ),
    )


def figure_correlation() -> alt.TopLevelMixin:
    """Each arm's correlation with the CAMS anomaly by lead band."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "anomaly_scores.parquet")
        .filter(
            (pl.col("scope") == "pooled")
            & (pl.col("metric") == "corr")
            & pl.col("arm").is_in(list(ARM_LABELS))
            & (pl.col("band") != "all")
        )
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    order = list(ARM_LABELS.values())
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70))
        .encode(
            x=alt.X("band:N", sort=BANDS, title="Lead time", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "value:Q",
                title="Correlation with the CAMS anomaly (higher is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "label:N",
                sort=order,
                scale=alt.Scale(domain=order, range=list(ARM_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=260,
            title=_title(
                text="Skill falls from 0.8 inside a day to 0.3 at 6 to 10 days for every arm",
                lines=[
                    "Pearson correlation between each arm's calibrated anomaly and the CAMS "
                    "anomaly, by lead band. Each calibration is fitted per series and lead band on "
                    "the training window. " + SCOPE
                ],
            ),
        )
    )


def figure_raw() -> alt.TopLevelMixin:
    """The CAMS anomaly against the raw ENS and GEFS anomalies, as bin means, 0 to 48 hours."""
    scored = _scored_with_anomalies().filter(pl.col("lead_hours") <= 48.0)
    pieces = []
    for source, column in (("ENS (ECMWF)", "a_ens"), ("GEFS (NOAA)", "a_gefs")):
        pieces.append(
            scored.with_columns(bin=(pl.col(column) * 10).floor() / 10 + 0.05)
            .group_by("bin")
            .agg(cams=pl.col("anomaly_cams").mean(), rows=pl.len())
            .filter(pl.col("rows") >= 100)
            .with_columns(source=pl.lit(source))
        )
    data = pl.concat(pieces)
    line = pl.DataFrame({"x": [-0.5, 0.5], "y": [-0.5, 0.5]})
    return alt.layer(
        alt.Chart(line).mark_line(color="#999999", strokeDash=[4, 3]).encode(x="x:Q", y="y:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=50))
        .encode(
            x=alt.X("bin:Q", title="Forecast anomaly (raw, before calibration; unit PV shape)"),
            y=alt.Y("cams:Q", title="Mean CAMS anomaly in the bin"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=["ENS (ECMWF)", "GEFS (NOAA)"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        ),
    ).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="GEFS's raw anomaly is about 15% too large; ENS's is about right",
            lines=[
                "Mean CAMS anomaly in bins of 0.1 of each forecast's raw ensemble-mean anomaly, "
                "leads of 0 to 48 hours, bins with 100 or more rows. Dashed line: a perfect "
                "forecast. " + SCOPE
            ],
        ),
    )


def _scored_with_anomalies() -> pl.DataFrame:
    """Return the scored verdict rows with the CAMS anomaly and the raw ENS and GEFS anomalies."""
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    arrays = common.load_series_arrays(series=series)
    scored = (
        common.scored_rows(series=series, arrays=arrays)
        .filter(pl.col("scored") & pl.col("site").is_in(list(common.VERDICT_SITES)))
        .select("site", "power_fcst_init_time", "valid_time", "lead_hours", "anomaly_cams")
    )
    centrals = pl.read_parquet(common.SOLAR_DIR / "central_val.parquet").select(
        "site", "power_fcst_init_time", "valid_time", "a_ens", "a_gefs"
    )
    return scored.join(centrals, on=["site", "power_fcst_init_time", "valid_time"], how="inner")


def figure_cv() -> alt.TopLevelMixin:
    """The cross-validation error of GEFS and the blend as a ratio to ENS's, by held-out month."""
    errors = pl.read_parquet(common.RESULTS_DIR / "cv_errors.parquet")
    wide = (
        errors.pivot(on="arm", index="fold", values="sse")
        .filter(pl.col("ens") > 0)
        .sort("fold")
        .with_columns(month=pl.Series(MONTHS))
    )
    data = wide.select(
        "month",
        **{
            ARM_LABELS["gefs"]: pl.col("gefs") / pl.col("ens"),
            ARM_LABELS["blend"]: pl.col("blend") / pl.col("ens"),
        },
    ).unpivot(index="month", variable_name="label", value_name="ratio")
    order = [ARM_LABELS["gefs"], ARM_LABELS["blend"]]
    points = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X(
                "month:N",
                sort=MONTHS,
                title="Held-out month (valid times)",
                axis=alt.Axis(labelAngle=-40),
            ),
            y=alt.Y(
                "ratio:Q",
                title="Squared error relative to ENS (below 1 is better)",
                scale=alt.Scale(domain=[0.95, 1.2], zero=False, clamp=True),
            ),
            color=alt.Color(
                "label:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[ocf.DATA_BLUE, ocf.DATA_GREEN]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    rules = (
        alt.Chart(pl.DataFrame({"y": [1.0]})).mark_rule(color="#999999").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(rules, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="GEFS loses to ENS in all 9 months; the blend ties",
            lines=[
                (
                    "Squared error of each calibrated anomaly against the CAMS anomaly, divided by "
                    "ENS's, in each held-out month of the training window (cross-validation: all "
                    "calibrations fitted on the other months). Rule at 1: same as ENS. Daylight "
                    "April-to-September rows at leads of 0 to 240 hours, five Primary substations. "
                    "Pooled over the months: GEFS 1.109, blend 1.002; the blend needed 0.99."
                )
            ],
        ),
    )


def figure_weights() -> alt.TopLevelMixin:
    """The fitted blend's mean weights by lead band."""
    weights = (
        pl.read_parquet(common.RESULTS_DIR / "blend_weights.parquet")
        .filter(pl.col("site").is_in(list(common.VERDICT_SITES)))
        .group_by("band")
        .agg(ENS=pl.col("w_ens").mean(), GEFS=pl.col("w_gefs").mean())
        .sort("band")
        .with_columns(lead=pl.col("band").map_elements(BANDS.__getitem__, return_dtype=pl.String))
        .unpivot(index=["band", "lead"], variable_name="weight_on", value_name="weight")
    )
    return (
        alt.Chart(weights)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("lead:N", sort=BANDS, title="Lead time", axis=alt.Axis(labelAngle=0)),
            xOffset="weight_on:N",
            y=alt.Y("weight:Q", title="Mean fitted weight on the raw anomaly"),
            color=alt.Color(
                "weight_on:N",
                scale=alt.Scale(domain=["ENS", "GEFS"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=240,
            title=_title(
                text="The blend gives GEFS 0.1 weight inside 2 days and none to 10 days",
                lines=[
                    (
                        "Mean over the five verdict series of the least-squares weights of the "
                        "CAMS anomaly on the raw ENS and GEFS anomalies, fitted on the training "
                        "window."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write every figure."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_headline": figure_headline,
        "figure_2_correlation": figure_correlation,
        "figure_3_raw": figure_raw,
        "figure_4_cv": figure_cv,
        "figure_5_weights": figure_weights,
    }
    for name, build in figures.items():
        build().save(CHART_DIR / f"{name}.svg")
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
