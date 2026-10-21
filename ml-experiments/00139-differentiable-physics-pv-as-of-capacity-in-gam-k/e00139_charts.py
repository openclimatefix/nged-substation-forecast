"""Draw the figures of experiment 139 as SVG into the docs folder, from the `work/` tables.

Only Primary substations appear, under their D-labels; the metered farms appear only as ratios.
Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00139-*/e00139_charts.py
"""

import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00139_common as common
import e00139_fits as fits_module
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00139-differentiable-physics-pv-as-of-capacity-in-gam-k"
)
PNG_DIR: Final[Path] = Path("/tmp/e00139_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 100
GREY: Final[str] = "#808080"
CONTRAST_LABELS: Final[dict[str, str]] = {
    "asof - pin": "As-of minus pin (the scored contrast)",
    "hindsight - pin": "Hindsight minus pin",
    "hindsight - asof": "Hindsight minus as-of",
    "asof_scaled - pin": "As-of x 1.15 minus pin (post hoc)",
}
SCOPE_LABELS: Final[dict[str, str]] = {
    "sunniest third": "Sunniest third of daylight half-hours",
    "darkest daylight third": "Darkest third of daylight half-hours",
    "all": "All half-hours",
}


def chart_title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title with the subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=text, subtitle=textwrap.wrap(subtitle, WRAP), anchor="start", subtitleColor="#444444"
    )


def save_chart(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write the chart as an SVG into the docs folder and as a PNG for checking."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def figure_contrasts(*, period: str) -> alt.TopLevelMixin:
    """Return the paired contrasts of the quantile score with 98.75% intervals, by scope."""
    table = (
        pl.read_parquet(common.WORK_DIR / f"contrasts_{period}.parquet")
        .filter(
            pl.col("metric") == "qs",
            pl.col("scope").is_in(list(SCOPE_LABELS)),
            pl.col("contrast").is_in(list(CONTRAST_LABELS)),
        )
        .with_columns(
            scope_label=pl.col("scope").replace(SCOPE_LABELS),
            contrast_label=pl.col("contrast").replace(CONTRAST_LABELS),
        )
    )
    colours = alt.Scale(
        domain=list(CONTRAST_LABELS.values()),
        range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE],
    )
    base = alt.Chart().encode(
        y=alt.Y(
            "contrast_label:N",
            sort=list(CONTRAST_LABELS.values()),
            title=None,
            axis=alt.Axis(labelLimit=260),
        ),
        color=alt.Color("contrast_label:N", scale=colours, legend=None),
    )
    interval = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q", title="Quantile score difference (points of capacity; negative is better)"
        ),
        x2="upper:Q",
    )
    dot = base.mark_point(filled=True, size=70).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart().mark_rule(color=GREY, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    panels = (
        alt.layer(zero, interval, dot, data=table)
        .properties(width=WIDTH - 280, height=120)
        .facet(
            row=alt.Row(
                "scope_label:N",
                sort=list(SCOPE_LABELS.values()),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12),
            )
        )
    )
    subtitle = (
        "Dot: estimate. Line: 98.75% interval from resampling whole months. Dashed rule: no "
        "difference. Rows: the five Primaries D5, D10, D11, D14, and D16, "
        + (
            "the scored year 2025-07 to 2026-06."
            if period == "scored"
            else "the even months of 2024-07 to 2025-06."
        )
    )
    return panels.properties(
        title=chart_title(
            text="The as-of capacity forecasts sunny half-hours worse than the pin",
            subtitle=subtitle,
        )
    )


def figure_capacity() -> alt.TopLevelMixin:
    """Return the capacity each arm subtracts at each monthly cut, per Primary, in MW."""
    cache = common.load_cache()
    rows = []
    for arm, label in (("pin", "Pin"), ("asof", "As-of"), ("hindsight", "Hindsight")):
        for site in common.DECISION_SITES:
            scale = cache.sites[site].table.effective_capacity_mw / 100.0
            for fit in fits_module.load_fits(arm=arm, site=site):
                assert fit.pinned is not None
                if fit.cut >= datetime(2025, 7, 1, tzinfo=UTC):
                    rows.append(
                        {
                            "site": site,
                            "arm": label,
                            "cut": fit.cut.replace(tzinfo=None),
                            "mw": fit.pinned.held_points * scale,
                        }
                    )
    table = pl.DataFrame(rows)
    domain = ["Pin", "As-of", "Hindsight"]
    base = (
        alt.Chart()
        .mark_line(point=True, strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "cut:T", title="Monthly fit cut in the scored year", axis=alt.Axis(format="%b %Y")
            ),
            y=alt.Y("mw:Q", title="MW", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "arm:N",
                sort=domain,
                scale=alt.Scale(
                    domain=domain, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    chart = (
        alt.layer(base, data=table)
        .properties(width=WIDTH - 120, height=95)
        .facet(
            row=alt.Row(
                "site:N",
                sort=list(common.DECISION_SITES),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12),
            )
        )
    )
    return chart.properties(
        title=chart_title(
            text="The pin grows through the scored year; the as-of capacity stays flat",
            subtitle=(
                "Solar capacity (MW; AC, 99th percentile) the fit of each monthly cut subtracts "
                "for rows after the cut. Pin: experiment 45's real-time vintage. As-of: experiment "
                "129's fit on data before 2025-07-01, held forward. Hindsight: experiment 129's "
                "fit on data to 2026-07-31, one value for the season-year."
            ),
        )
    )


def figure_recovery() -> alt.TopLevelMixin:
    """Return the known-answer recovery per Primary and plant."""
    table = (
        pl.read_parquet(common.WORK_DIR / "kat.parquet")
        .group_by("site", "plant")
        .agg(recovery=pl.col("recovery").median())
    )
    labels = {
        "matched": "Real farm",
        "east_west": "East-west plant",
        "south_west": "South-west plant",
    }
    table = table.with_columns(plant_label=pl.col("plant").replace(labels))
    band = (
        alt.Chart(pl.DataFrame({"low": [0.75], "high": [1.25]}))
        .mark_rect(opacity=0.12, color=GREY)
        .encode(y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    one = (
        alt.Chart(pl.DataFrame({"y": [1.0]}))
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    dots = (
        alt.Chart(table)
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "site:N", sort=list(common.DECISION_SITES), title=None, axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y(
                "recovery:Q",
                title="Injected output recovered (share)",
                scale=alt.Scale(domain=[0.6, 1.4]),
            ),
            color=alt.Color(
                "plant_label:N",
                sort=list(labels.values()),
                scale=alt.Scale(
                    domain=list(labels.values()),
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    return alt.layer(band, one, dots).properties(
        width=WIDTH - 60,
        height=200,
        title=chart_title(
            text="Known-answer test: a capacity raise removes 0.78 to 0.88 of the injection",
            subtitle=(
                "Output of a solar plant equal to half the as-of capacity was added to the "
                "Primary's power in the scored year. Each dot is the share of the injected output "
                "that "
                "raising "
                "the forecast's capacity by that amount removes on the sunniest third of "
                "daylight "
                "half-hours (median over five real farms for the real-farm dots). Shaded band: the "
                "pre-registered pass band of 0.75 to 1.25."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    save_chart(figure_contrasts(period="scored"), name="figure_01_contrasts_scored")
    save_chart(figure_contrasts(period="replicate"), name="figure_02_contrasts_replicate")
    save_chart(figure_capacity(), name="figure_03_capacity")
    save_chart(figure_recovery(), name="figure_04_recovery")


if __name__ == "__main__":
    main()
