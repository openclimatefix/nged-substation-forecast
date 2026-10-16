"""Draw experiment 00052's figures from the saved results, as SVG, into the docs folder.

1. Each blend minus the best guess, at lead days 1 and 2 and over all leads.
2. The primary blend minus the best guess for each wind farm.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00052-more-nwp-products-for-wind-farms/e00052_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00052_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00052-more-nwp-products-for-wind-farms"
)
PLOT_WIDTH: Final[int] = 380
WRAP: Final[int] = 80
GREY: Final[str] = "#999999"
SCOPE: Final[str] = "3 wind farms, 53 Tuesday runs, 3 seeds, July 2025 to June 2026."
ARM_LABELS: Final[dict[str, str]] = {
    "ens_icon_ifs": "Best guess + ICON-EU + IFS (primary)",
    "ens_icon": "Best guess + ICON-EU",
    "ens_ifs": "Best guess + IFS",
    "ens_gfs": "Best guess + GFS",
    "ens_icon_ifs_one": "Primary, one weight for all leads",
    "ens_icon_ifs_nospread": "Primary, no spread factor",
    "placebo": "Placebo (shuffled products)",
}
SCOPES: Final[dict[str, str]] = {"days 1-2": "Lead days 1 and 2", "all": "All leads"}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _layers(*, data: pl.DataFrame, y: alt.Y, colour: alt.Color, height: int) -> alt.TopLevelMixin:
    base = alt.Chart(data).encode(y=y, yOffset=alt.YOffset("scope_label:N"), color=colour)
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q", title="Blend minus best guess (points of capacity; negative is better)"
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    threshold = (
        alt.Chart(pl.DataFrame({"x": [-0.15]}))
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(zero, threshold, bars, points).properties(width=PLOT_WIDTH, height=height)


def _colour() -> alt.Color:
    return alt.Color(
        "scope_label:N",
        scale=alt.Scale(domain=list(SCOPES.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(title=None, orient="top"),
    )


def figure_1(*, results: dict) -> alt.TopLevelMixin:
    """Each arm minus the best guess at days 1 and 2, and over all leads."""
    rows = [
        e
        for e in results["exploratory"]
        if e["treatment"] in ARM_LABELS and e["band"] in SCOPES and e.get("series") is None
    ]
    data = pl.DataFrame(rows).with_columns(
        arm_label=pl.col("treatment").replace_strict(ARM_LABELS),
        scope_label=pl.col("band").replace_strict(SCOPES),
    )
    chart = _layers(
        data=data,
        y=alt.Y(
            "arm_label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=300)
        ),
        colour=_colour(),
        height=260,
    )
    return chart.properties(
        title=_title(
            text="No blend lowers the wind farms' error resolvably at 1 to 2 days",
            lines=[
                "Quantile score of each blend minus the best guess (smaller is better). Solid grey "
                "line: same as the best guess. Dashed line: the 0.15-point threshold. Dot: "
                "estimate. Line: 95% interval from re-drawing whole months. " + SCOPE
            ],
        )
    )


def figure_2(*, results: dict) -> alt.TopLevelMixin:
    """The primary blend minus the best guess, per wind farm."""
    rows = [
        e
        for e in results["exploratory"]
        if e["treatment"] == "ens_icon_ifs" and e["band"] in SCOPES and e.get("series") is not None
    ]
    data = pl.DataFrame(rows).with_columns(
        scope_label=pl.col("band").replace_strict(SCOPES),
    )
    chart = _layers(
        data=data,
        y=alt.Y("site:N", title="Wind farm"),
        colour=_colour(),
        height=140,
    )
    return chart.properties(
        title=_title(
            text="One wind farm gains at 1 to 2 days, one loses, and one is unresolved",
            lines=[
                "Quantile score of the primary blend minus the best guess, per wind farm (smaller "
                "is better). Solid grey line: same as the best guess. Dashed line: the 0.15-point "
                "threshold. Dot: estimate. Line: 95% interval from re-drawing whole months. "
                + SCOPE
            ],
        )
    )


def main() -> None:
    """Write both figures."""
    results = json.loads((common.WORK_DIR / "results" / "results.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1(results=results)),
        ("figure_2", figure_2(results=results)),
    ):
        chart.save(DOCS_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
