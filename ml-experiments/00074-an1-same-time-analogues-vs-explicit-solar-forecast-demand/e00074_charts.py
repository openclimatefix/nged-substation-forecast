"""Draw experiment 00074's figures from the saved results, as SVG, headline contrasts first.

1. The headline: each arm's absolute quantile score and the six planned contrasts with their
   99.17% intervals (seed 0, which decides the verdict).
2. Contrasts C1 to C4 by lead band (exploratory).
3. Contrasts C1 to C4 by valid month, the sign check (exploratory).
4. The planned contrasts from seed 0 and from seeds 0 to 2 together (post hoc).

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00074-*/e00074_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00074_common as common
import plotting.ocf_theme as ocf
import polars as pl

CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
SEED0_DIR: Final[Path] = common.RESULTS_DIR / "seed0"
SEEDS3_DIR: Final[Path] = common.RESULTS_DIR / "seeds012"
PLOT_WIDTH: Final[int] = 440
WRAP_CHARACTERS: Final[int] = 78
SCOPE: Final[str] = (
    "Five Primary substations, daylight half-hours of April to September, forecast runs one "
    "every 5 days: July to September 2025 and April to June 2026. Quantile score in points of "
    "capacity (1 point is 1% of a substation's capacity)."
)
GREY: Final[str] = "#8A8A8A"
ARM_LABELS: Final[dict[str, str]] = {
    "R0": "R0: monthly refit, no solar input",
    "E": "E: explicit solar forecast as offset",
    "E_twin": "E_twin: offset from another date",
    "A1": "A1: 1 analogue",
    "A3": "A3: 3 analogues",
    "A5": "A5: 5 analogues",
    "A3E": "A3E: 3 analogues and the offset",
    "A3S": "A3S: 3 random days",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "R0": GREY,
    "E": ocf.DATA_GREEN,
    "E_twin": ocf.DATA_GREEN_LIGHT,
    "A1": ocf.DATA_SKY,
    "A3": ocf.DATA_BLUE,
    "A5": ocf.DATA_BLUE_LIGHT,
    "A3E": ocf.DATA_PURPLE,
    "A3S": ocf.BRAND_ORANGE_LIGHT,
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1: offset minus reference (E - R0)",
    "C2": "C2: analogues minus reference (A3 - R0)",
    "C3": "C3: analogues minus offset (A3 - E)",
    "C4": "C4: both minus offset (A3E - E)",
    "C5": "C5: offset minus its twin (E - E_twin)",
    "C6": "C6: matched minus random days (A3 - A3S)",
}
CONTRAST_COLOURS: Final[dict[str, str]] = {
    "C1": ocf.DATA_GREEN,
    "C2": ocf.DATA_BLUE,
    "C3": ocf.BRAND_ORANGE,
    "C4": ocf.DATA_PURPLE,
    "C5": ocf.DATA_SKY,
    "C6": GREY,
}
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _low, _high in common.LEAD_BANDS)
MONTH_NAMES: Final[dict[int, str]] = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
ABSOLUTE_DOMAIN: Final[tuple[float, float]] = (4.1, 5.3)
CONTRAST_DOMAIN: Final[tuple[float, float]] = (-0.5, 0.4)
BAND_DOMAIN: Final[tuple[float, float]] = (-0.9, 0.6)
THRESHOLD: Final[float] = -0.05


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _rule(
    *,
    x: float,
    domain: tuple[float, float],
    dash: tuple[int, int] | None = None,
    colour: str = "#999999",
) -> alt.Chart:
    mark = alt.Chart(pl.DataFrame({"x": [x]})).mark_rule(color=colour, strokeDash=dash or [])
    return mark.encode(x=alt.X("x:Q", scale=alt.Scale(domain=list(domain))))  # ty: ignore[unresolved-attribute]


def _absolute_panel() -> alt.LayerChart | alt.FacetChart:
    """Each arm's absolute quantile score with its 95% interval."""
    absolute = pl.read_parquet(SEED0_DIR / "absolute.parquet")
    arms = list(ARM_LABELS)
    data = absolute.filter(pl.col("arm").is_in(arms)).with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS),
        colour=pl.col("arm").replace_strict(ARM_COLOURS),
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N",
            sort=[ARM_LABELS[arm] for arm in arms],
            title=None,
            axis=alt.Axis(labelLimit=240),
        ),
        color=alt.Color("colour:N", scale=None),
    )
    return alt.layer(
        _rule(
            x=absolute.filter(pl.col("arm") == "R0")["qs"].item(),
            domain=ABSOLUTE_DOMAIN,
            colour=GREY,
        ),
        base.mark_rule().encode(x="qs_lower:Q", x2="qs_upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "qs:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=list(ABSOLUTE_DOMAIN)),
            )
        ),
    ).properties(width=PLOT_WIDTH, height=200)


def _contrast_layer(
    *, data: pl.DataFrame, order: list[str], domain: tuple[float, float]
) -> alt.LayerChart | alt.FacetChart:
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color("colour:N", scale=None),
    )
    return alt.layer(
        _rule(x=0.0, domain=domain),
        _rule(x=THRESHOLD, domain=domain, dash=(4, 3), colour="black"),
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Quantile score difference (points of capacity; more negative is better)",
                scale=alt.Scale(domain=list(domain)),
            )
        ),
    )


def _planned_data(*, folder: Path) -> pl.DataFrame:
    return pl.read_parquet(folder / "planned.parquet").with_columns(
        label=pl.col("contrast").replace_strict(CONTRAST_LABELS),
        colour=pl.col("contrast").replace_strict(CONTRAST_COLOURS),
    )


def figure_headline() -> alt.TopLevelMixin:
    """Absolute scores above, the planned contrasts below."""
    order = list(CONTRAST_LABELS.values())
    contrasts = _contrast_layer(
        data=_planned_data(folder=SEED0_DIR), order=order, domain=CONTRAST_DOMAIN
    ).properties(width=PLOT_WIDTH, height=190)
    return (
        alt.vconcat(_absolute_panel(), contrasts)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Both the offset and the analogues lower the error; the data cannot rank them",
                lines=[
                    "Top: each arm's quantile score (dot) with its 95% interval from re-drawing "
                    "whole weeks of forecast start. Grey line: R0. Bottom: the six planned "
                    "contrasts with 99.17% intervals. Grey line at 0: same score. Dashed line, "
                    "-0.05: the gain the analogues must clear on top of the offset (C4). " + SCOPE
                ],
            )
        )
    )


def figure_by_lead_band() -> alt.TopLevelMixin:
    """Contrasts C1 to C4 in each lead band, at 95%."""
    keep = ["C1", "C2", "C3", "C4"]
    data = (
        pl.read_parquet(SEED0_DIR / "exploratory.parquet")
        .filter(pl.col("contrast").is_in([f"{c}_by_band" for c in keep]))
        .with_columns(
            band=pl.col("scope").str.replace("band ", ""),
            key=pl.col("contrast").str.slice(0, 2),
        )
        .with_columns(
            label=pl.col("key").replace_strict(CONTRAST_LABELS),
            colour=pl.col("key").replace_strict(CONTRAST_COLOURS),
        )
    )
    labels = [CONTRAST_LABELS[c] for c in keep]
    base = alt.Chart(data).encode(
        y=alt.Y("band:N", sort=list(BAND_NAMES), title="Lead time"),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N",
            sort=labels,
            scale=alt.Scale(domain=labels, range=[CONTRAST_COLOURS[c] for c in keep]),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=320),
        ),
    )
    return alt.layer(
        _rule(x=0.0, domain=BAND_DOMAIN),
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Quantile score difference (points of capacity; more negative is better)",
                scale=alt.Scale(domain=list(BAND_DOMAIN)),
            )
        ),
    ).properties(
        width=PLOT_WIDTH,
        height=330,
        title=_title(
            text="Both gains fade with lead time; the offset wins at the shortest leads",
            lines=[
                "Each contrast in each lead band. Dot: estimate. Line: 95% interval from "
                "re-drawing whole weeks of forecast start. Line at 0: same score. Exploratory. "
                + SCOPE
            ],
        ),
    )


def figure_by_month() -> alt.TopLevelMixin:
    """Contrasts C1 to C4 in each valid month: the mean difference, not an interval."""
    keep = ["C1", "C2", "C3", "C4"]
    data = (
        pl.read_parquet(SEED0_DIR / "exploratory.parquet")
        .filter(pl.col("contrast").is_in([f"{c}_by_month" for c in keep]))
        .with_columns(
            month=pl.col("scope")
            .str.replace("month ", "")
            .cast(pl.Int64)
            .replace_strict(MONTH_NAMES),
            key=pl.col("contrast").str.slice(0, 2),
        )
        .with_columns(label=pl.col("key").replace_strict(CONTRAST_LABELS))
    )
    labels = [CONTRAST_LABELS[c] for c in keep]
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "month:N", sort=list(MONTH_NAMES.values()), title=None, axis=alt.Axis(labelAngle=0)
            ),
            xOffset=alt.XOffset("label:N", sort=labels),
            y=alt.Y(
                "difference:Q",
                title="Mean difference (points; more negative is better)",
                scale=alt.Scale(domain=[-0.35, 0.2]),
            ),
            color=alt.Color(
                "label:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=[CONTRAST_COLOURS[c] for c in keep]),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=320),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=240,
            title=_title(
                text="The analogues match the offset only in August and September",
                lines=[
                    "Mean quantile score difference in each local valid month; below zero means "
                    "the first arm of the contrast scored better. Exploratory, and each month "
                    "holds 5 to 7 weeks of forecast start. " + SCOPE
                ],
            ),
        )
    )


def figure_seeds() -> alt.TopLevelMixin:
    """The planned contrasts from seed 0 and from seeds 0 to 2 (post hoc)."""
    both = pl.concat(
        [
            _planned_data(folder=SEED0_DIR).with_columns(seeds=pl.lit("seed 0 (decides)")),
            _planned_data(folder=SEEDS3_DIR).with_columns(seeds=pl.lit("seeds 0 to 2 (post hoc)")),
        ]
    )
    labels = list(CONTRAST_LABELS.values())
    kinds = ["seed 0 (decides)", "seeds 0 to 2 (post hoc)"]
    base = alt.Chart(both).encode(
        y=alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("seeds:N", sort=kinds),
        color=alt.Color("colour:N", scale=None),
    )
    return alt.layer(
        _rule(x=0.0, domain=CONTRAST_DOMAIN),
        _rule(x=THRESHOLD, domain=CONTRAST_DOMAIN, dash=(4, 3), colour="black"),
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Quantile score difference (points of capacity; more negative is better)",
                scale=alt.Scale(domain=list(CONTRAST_DOMAIN)),
            ),
            shape=alt.Shape(
                "seeds:N",
                sort=kinds,
                scale=alt.Scale(domain=kinds, range=["circle", "diamond"]),
                legend=alt.Legend(title=None, orient="top", symbolFillColor=GREY),
            ),
        ),
    ).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="Seeds 1 and 2 move no planned contrast by more than 0.01 points",
            lines=[
                "Planned contrasts with 99.17% intervals. Circle: seed 0, which decides the "
                "verdict. Diamond: seeds 0 to 2 together, a post hoc check. Line at 0: same "
                "score. Dashed line, -0.05: the porting threshold for C4. " + SCOPE
            ],
        ),
    )


def main() -> None:
    """Write every figure."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_headline": figure_headline,
        "figure_2_by_lead_band": figure_by_lead_band,
        "figure_3_by_month": figure_by_month,
        "figure_4_seeds": figure_seeds,
    }
    for name, build in figures.items():
        build().save(CHART_DIR / f"{name}.svg")
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
