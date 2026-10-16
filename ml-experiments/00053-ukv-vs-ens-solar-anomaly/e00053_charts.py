"""Draw experiment 00053's figures as SVG, raw data first.

1. Scatter of each arm's calibrated anomaly against the CAMS anomaly (leads under 24 h).
2. Correlation with the CAMS anomaly by lead band, with 95% intervals.
3. Mean `u` by hour of day for CAMS, ENS and the UKV forms (the label-convention check).

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00053-*/e00053_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00053_common as common
import plotting.ocf_theme as ocf
import polars as pl
from e00053_scores import scored_rows

CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
ARM_LABELS: Final[dict[str, str]] = {
    "ens": "ENS (calibrated mean)",
    "ukv": "UKV, native steps",
    "blend": "ENS + UKV blend",
    "ukv3h": "UKV at 3-hourly steps",
    "ukv3w": "UKV as 3-hour window means",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "ens": ocf.BRAND_ORANGE,
    "ukv": ocf.DATA_BLUE,
    "blend": ocf.DATA_GREEN,
    "ukv3h": ocf.DATA_PURPLE,
    "ukv3w": ocf.DATA_SKY,
}
SCOPE: Final[str] = (
    "Five Primary substations, daylight half-hours of April to September, forecast runs every "
    "5 days from July to September 2025 and March to June 2026."
)


def wrapped(*, text: str) -> list[str]:
    """Wrap a title or subtitle to the plot width.

    Args:
        text: The text.

    Returns:
        The lines.
    """
    return textwrap.wrap(text, WRAP)


def scatter() -> alt.FacetChart:
    """Return the scatter of calibrated anomaly against CAMS anomaly, one panel per arm."""
    rows = scored_rows().filter((pl.col("variant") == "t54") & (pl.col("lead_hours") <= 24))
    sample = rows.sample(n=min(600, rows.height), seed=1)
    long = pl.concat(
        [
            sample.select(arm=pl.lit(arm), forecast=pl.col(f"c_{arm}"), cams=pl.col("a_cams"))
            for arm in ARM_LABELS
        ]
    ).with_columns(label=pl.col("arm").replace(ARM_LABELS))
    base = (
        alt.Chart()
        .mark_circle(size=6, opacity=0.35, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "forecast:Q",
                title="Calibrated forecast anomaly of u",
                scale=alt.Scale(domain=[-0.8, 0.8]),
                axis=alt.Axis(values=[-0.5, 0, 0.5]),
            ),
            y=alt.Y(
                "cams:Q",
                title="CAMS anomaly of u",
                scale=alt.Scale(domain=[-0.8, 0.8]),
                axis=alt.Axis(values=[-0.5, 0, 0.5]),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
        )
    )
    line = (
        alt.Chart()
        .mark_line(color="#888888", strokeDash=[4, 3])
        .encode(  # ty: ignore[unresolved-attribute]
            x="forecast:Q", y="forecast:Q"
        )
    )
    panels = (
        alt.layer(base, line, data=long)
        .properties(width=190, height=190)
        .facet(facet=alt.Facet("label:N", sort=list(ARM_LABELS.values()), title=None), columns=3)
    )
    return panels.properties(
        title=alt.TitleParams(
            wrapped(text="Each forecast anomaly against the CAMS anomaly, lead under 24 h"),
            subtitle=wrapped(
                text=(
                    "u is the output of a generic solar fleet per MW of AC capacity; the anomaly "
                    "is u minus its month-by-half-hour mean. Dashed line: forecast equals CAMS. "
                    "600 "
                    f"randomly drawn rows of the validation year. {SCOPE}"
                )
            ),
        )
    )


def correlation_by_band() -> alt.Chart:
    """Return correlation with the CAMS anomaly by lead band, one row per band."""
    scores = pl.read_parquet(common.RESULTS_DIR / "scores.parquet")
    single = list(common.BAND_NAMES)
    rows = pl.concat(
        [
            scores.filter((pl.col("variant") == "t54") & pl.col("band").is_in(single[:5])),
            scores.filter((pl.col("variant") == "t120") & (pl.col("band") == "54-120 h")),
        ]
    ).filter(pl.col("arm").is_in(list(ARM_LABELS)))
    rows = rows.with_columns(label=pl.col("arm").replace(ARM_LABELS))
    base = alt.Chart(rows).encode(
        y=alt.Y("band:N", sort=list(common.BAND_NAMES), title="Lead from the forecast origin"),
        yOffset=alt.YOffset("label:N", sort=list(ARM_LABELS.values())),
        color=alt.Color(
            "label:N",
            sort=list(ARM_LABELS.values()),
            scale=alt.Scale(
                domain=list(ARM_LABELS.values()), range=[ARM_COLOURS[a] for a in ARM_LABELS]
            ),
            legend=alt.Legend(title=None, orient="top", columns=2),
        ),
    )
    interval = base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "correlation_low:Q",
            title="Correlation with the CAMS anomaly (higher is better)",
            scale=alt.Scale(domain=[0, 1]),
        ),
        x2="correlation_high:Q",
    )
    dot = base.mark_circle(size=40, opacity=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x="correlation:Q"
    )
    return (interval + dot).properties(
        width=WIDTH - 90,
        height=330,
        title=alt.TitleParams(
            wrapped(text="Correlation with the CAMS anomaly by lead band"),
            subtitle=wrapped(
                text=(
                    "Dot: correlation. Line: 95% interval from resampling whole calendar months of "
                    "the forecast start. Bands up to 48-54 h use the 06 UTC UKV run; 54-120 h uses "
                    f"the 03 UTC run with leads to 120 h. {SCOPE}"
                )
            ),
        ),
    )


def profile() -> alt.Chart:
    """Return the mean `u` by UTC hour for CAMS, ENS and the UKV forms."""
    table = pl.read_parquet(common.RESULTS_DIR / "profile.parquet")
    names = {"cams": "CAMS", **ARM_LABELS}
    colours = {"cams": ocf.BRAND_ORANGE}
    long = table.unpivot(
        on=["cams", "ens", "ukv", "ukv3h", "ukv3w"],
        index="hour",
        variable_name="source",
        value_name="u",
    )
    long = long.with_columns(label=pl.col("source").replace({"cams": "CAMS", **ARM_LABELS}))
    domain = ["CAMS", *ARM_LABELS.values()]
    range_ = ["#000000", ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_SKY]
    del names, colours
    chart = (
        alt.Chart(long)
        .mark_line(clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                title="Hour of day, UTC (daylight hours only)",
                scale=alt.Scale(domain=[5, 20]),
                axis=alt.Axis(values=[6, 9, 12, 15, 18]),
            ),
            y=alt.Y("u:Q", title="Mean u (output per MW of AC capacity)"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=domain, range=range_),
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
        )
    )
    return chart.properties(
        width=WIDTH - 90,
        height=240,
        title=alt.TitleParams(
            wrapped(text="Mean solar output shape by hour of day, leads under 24 h"),
            subtitle=wrapped(
                text=(
                    "Check of the time-label convention: all sources peak at 12 UTC, as CAMS does, "
                    "except the 3-hourly UKV values that were labelled as window means. "
                    f"{SCOPE}"
                )
            ),
        ),
    )


def main() -> None:
    """Write the three figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("e00053_scatter", scatter()),
        ("e00053_correlation_by_band", correlation_by_band()),
        ("e00053_hour_of_day", profile()),
    ):
        chart.save(CHART_DIR / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
