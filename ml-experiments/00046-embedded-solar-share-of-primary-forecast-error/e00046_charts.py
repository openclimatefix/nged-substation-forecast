"""Draw experiment 00046's figures from the saved results, as SVG, raw data first.

1. The four planned contrasts, and the two fitted-coefficient reads, with their intervals and the
   0.35-point threshold.
2. Contrasts C3, C4 and C6 by lead band.
3. The share of actuals below the 0.05 and 0.95 quantiles, in sunny and cloudy rows.
4. The four contrasts by valid month (the sign check).

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
PLOT_WIDTH: Final[int] = 440
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
    "C3_fitted": "C3 fitted coefficient (exploratory)",
    "C4_fitted": "C4 fitted coefficient (exploratory)",
}
CONTRAST_COLOURS: Final[dict[str, str]] = {
    "C3": ocf.BRAND_ORANGE,
    "C4": ocf.DATA_BLUE,
    "C5": ocf.DATA_PURPLE,
    "C6": ocf.DATA_GREEN,
    "C3_fitted": ocf.BRAND_ORANGE,
    "C4_fitted": ocf.DATA_BLUE,
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


def figure_contrasts() -> alt.TopLevelMixin:
    """The four planned contrasts and two fitted-coefficient reads, with the threshold."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "contrasts.parquet")
        .filter(
            pl.col("contrast").is_in(list(CONTRAST_LABELS))
            & (pl.col("scope") == "verdict series")
            & (pl.col("block") == "month")
        )
        .sort("n_resamples", descending=True)
        .unique(subset=["contrast"], keep="first")
        .with_columns(
            label=pl.col("contrast").replace_strict(CONTRAST_LABELS),
            kind=pl.when(pl.col("contrast").is_in(["C3", "C4"]))
            .then(pl.lit("deciding, 97.5% interval"))
            .otherwise(pl.lit("exploratory, 95% interval")),
        )
    )
    order = list(CONTRAST_LABELS.values())
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    )
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
        height=200,
        title=_title(
            text="True sunshine helps by about 1 point; the ENS forecast of it does not",
            lines=[
                "Each row is the first arm's quantile score minus the second's, over the scored "
                "rows. Dot: estimate. Line: interval from re-drawing whole months of run "
                "initialisation and seeds (solid: 97.5%, Bonferroni across the two deciding "
                "contrasts; dashed: 95%, exploratory). Left rule at -0.35: the smallest gain "
                "worth porting. Right rule at 0: same score. " + SCOPE
            ],
        ),
    )


def figure_by_lead_band() -> alt.TopLevelMixin:
    """Contrasts C3, C4 and C6 in each lead band, at 95%."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "contrasts.parquet")
        .filter(
            pl.col("scope").str.starts_with("lead band ")
            & pl.col("contrast").is_in(["C3", "C4", "C6"])
        )
        .with_columns(
            band=pl.col("scope").str.replace("lead band ", ""),
            label=pl.col("contrast").replace_strict(CONTRAST_LABELS),
        )
    )
    bands = [name for name, _low, _high in common.LEAD_BANDS]
    labels = [CONTRAST_LABELS[name] for name in ("C3", "C4", "C6")]
    base = alt.Chart(data).encode(
        y=alt.Y("band:N", sort=bands, title="Lead time", scale=alt.Scale(paddingOuter=0.1)),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N",
            sort=labels,
            scale=alt.Scale(
                domain=labels,
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    return alt.layer(
        _zero_rule(),
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Quantile score difference (points of capacity; more negative is better)",
            )
        ),
    ).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="The ENS shift helps inside 2 days and not beyond",
            lines=[
                "Quantile score difference in each lead band. Dot: estimate. Line: 95% interval "
                "from re-drawing whole months of run initialisation and seeds, exploratory. "
                "Right rule at 0: same score. " + SCOPE
            ],
        ),
    )


def figure_coverage() -> alt.TopLevelMixin:
    """The share of actuals below the 0.05 and 0.95 quantiles of `d1l` and `d1l_ens_shift`."""
    arms = {"d1l": "d1l", "d1l_ens_shift": "d1l + ENS shift"}
    data = (
        pl.read_parquet(common.RESULTS_DIR / "coverage.parquet")
        .filter(pl.col("arm").is_in(list(arms)))
        .unpivot(
            on=["below_q05", "below_q95"],
            index=["arm", "csi_bin"],
            variable_name="quantile",
            value_name="share",
        )
        .with_columns(target=pl.when(pl.col("quantile") == "below_q05").then(0.05).otherwise(0.95))
        .with_columns(
            arm_label=pl.col("arm").replace_strict(arms),
            quantile=pl.col("quantile").replace_strict(
                {
                    "below_q05": "0.05 quantile (target 5%)",
                    "below_q95": "0.95 quantile (target 95%)",
                }
            ),
            sky=pl.col("csi_bin").replace_strict({"sunny": "Sunny rows", "cloudy": "Cloudy rows"}),
        )
    )
    panels = []
    for quantile in data["quantile"].unique(maintain_order=True):
        part = data.filter(pl.col("quantile") == quantile)
        bars = (
            alt.Chart(part)
            .mark_bar()
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("sky:N", sort=["Sunny rows", "Cloudy rows"], title=None),
                yOffset=alt.YOffset("arm_label:N", sort=list(arms.values())),
                x=alt.X(
                    "share:Q",
                    title="Share of actuals below the quantile",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1]),
                ),
                color=alt.Color(
                    "arm_label:N",
                    sort=list(arms.values()),
                    scale=alt.Scale(domain=list(arms.values()), range=[GREY, ocf.BRAND_ORANGE]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        rule = (
            alt.Chart(part).mark_rule(color="black", strokeDash=[4, 3]).encode(x="target:Q")  # ty: ignore[unresolved-attribute]
        )
        text = (
            alt.Chart(part)
            .mark_text(align="left", dx=3, fontSize=10)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("sky:N", sort=["Sunny rows", "Cloudy rows"]),
                yOffset=alt.YOffset("arm_label:N", sort=list(arms.values())),
                x="share:Q",
                text=alt.Text("share:Q", format=".1%"),
            )
        )
        panels.append(
            alt.layer(bars, rule, text).properties(
                width=PLOT_WIDTH, height=90, title=alt.TitleParams(quantile, anchor="start")
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The ENS shift moves the quantiles towards their targets in sunny and cloudy rows",
            lines=[
                "Share of scored actuals below the forecast's 0.05 and 0.95 quantiles, by sky. "
                "Sunny: CAMS clear-sky index 0.7 or more. Cloudy: under 0.3. Dashed line: the "
                "share a well-calibrated quantile would give. " + SCOPE
            ],
        )
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
            x=alt.X(
                "month:N", sort=order, title="Valid month (local)", axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("difference:Q", title=None),
            color=alt.Color(
                "contrast:N",
                scale=alt.Scale(
                    domain=list(CONTRAST_COLOURS), range=list(CONTRAST_COLOURS.values())
                ),
                legend=None,
            ),
        )
        .properties(width=PLOT_WIDTH, height=70)
        .facet(
            row=alt.Row(
                "label:N",
                sort=[CONTRAST_LABELS[name] for name in ("C3", "C4", "C5", "C6")],
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11),
            )
        )
        .properties(
            title=_title(
                text="The ENS shift helps in April, May and September and hurts in June to August",
                lines=[
                    "Mean quantile score difference of each contrast in each local valid month, "
                    "in points of capacity (more negative is better). A contrast counts only if "
                    "every month has the same sign. " + SCOPE
                ],
            )
        )
    )


def figure_midday() -> alt.TopLevelMixin:
    """Contrasts C3, C4 and C6 by lead day on summer midday rows, exploratory and post hoc."""
    data = pl.read_parquet(common.RESULTS_DIR / "midday_contrasts.parquet").with_columns(
        label=pl.col("contrast").replace_strict(CONTRAST_LABELS),
        panel=pl.col("row_set")
        + ", "
        + pl.col("series")
        + " ("
        + pl.col("blocks").cast(pl.String)
        + " blocks)",
        band=pl.col("band").str.replace(r" \(.*\)", ""),
    )
    labels = [CONTRAST_LABELS[name] for name in ("C3", "C4", "C6")]
    panels = []
    for panel in data["panel"].unique(maintain_order=True):
        base = alt.Chart(data.filter(pl.col("panel") == panel)).encode(
            y=alt.Y("band:N", title=None, sort=["day+1", "day+2", "day+3"]),
            yOffset=alt.YOffset("label:N", sort=labels),
            color=alt.Color(
                "label:N",
                sort=labels,
                scale=alt.Scale(
                    domain=labels, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        panels.append(
            alt.layer(
                _zero_rule(),
                base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
                base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "difference:Q",
                        title="Score difference (points of capacity; more negative is better)",
                        scale=alt.Scale(domain=[-2.2, 0.8]),
                    )
                ),
            ).properties(width=PLOT_WIDTH, height=90, title=alt.TitleParams(panel, anchor="start"))
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Exploratory: at summer midday the ENS shift helps most on day+1",
            lines=[
                (
                    "Contrasts by lead day from the forecast origin, on valid times 09:00 to 15:00 "
                    "UTC. Dot: estimate. Line: 95% interval from re-drawing whole months of run "
                    "initialisation and seeds; with 3 blocks the interval is rough. Right rule at "
                    "0: same score. Post hoc."
                )
            ],
        )
    )


def main() -> None:
    """Write every figure."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_contrasts": figure_contrasts,
        "figure_2_by_lead_band": figure_by_lead_band,
        "figure_3_coverage": figure_coverage,
        "figure_4_by_month": figure_by_month,
        "figure_5_midday": figure_midday,
    }
    for name, build in figures.items():
        build().save(CHART_DIR / f"{name}.svg")
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
