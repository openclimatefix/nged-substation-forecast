"""Draw the figures of experiment 173's write-up from `work/report/report_<arm>.json`.

uv run --no-sync python ml-experiments/00173-*/e00173_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00173_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GREY: Final[str] = "#808080"
YEAR_LABELS: Final[dict[int, str]] = {
    season: f"{season}-{str(season + 1)[2:]}" for season in common.SEASONS
}
ADEQUATE_DAYS: Final[int] = 180
SUBSET_LABEL: Final[str] = "Quantile score, 180+ days (post hoc)"
LOSS_ORDER: Final[list[str]] = ["Quantile score", "Absolute error of the mean", SUBSET_LABEL]
GROUP_ORDER: Final[tuple[str, ...]] = ("24-25 Dec", "26 Dec", "27-31 Dec", "1-2 Jan")


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _ratios(*, data: dict, **keys: object) -> pl.DataFrame:
    rows = [r for r in data["ratios"] if all(r[k] == v for k, v in keys.items()) and "ratio" in r]
    return pl.DataFrame(rows).filter(pl.col("ratio").is_not_nan())


def figure_headline(*, m12: dict, all_years: dict) -> alt.TopLevelMixin:
    """The pooled ratio per Christmas, both losses, with 95% intervals."""
    rows = []
    for metric, label in (("qs", "Quantile score"), ("abs_mean", "Absolute error of the mean")):
        frame = _ratios(data=m12, scope="all", metric=metric, site="all", group="christmas")
        for r in frame.filter(pl.col("season") != "all").iter_rows(named=True):
            rows.append(  # noqa: PERF401
                {
                    "season": YEAR_LABELS[int(r["season"])], "loss": label,
                    "ratio": r["ratio"], "low": r["low"], "high": r["high"],
                }
            )  # fmt: skip
    for r in m12["adequate_subset"]:
        if (
            r.get("min_days") == ADEQUATE_DAYS
            and r.get("metric") == "qs"
            and "low" in r
            and r["season"] != "all"
        ):
            rows.append(  # noqa: PERF401
                {
                    "season": YEAR_LABELS[int(r["season"])], "loss": SUBSET_LABEL,
                    "ratio": r["ratio"], "low": r["low"], "high": r["high"],
                }
            )  # fmt: skip
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("season:N", title="Christmas", sort=list(YEAR_LABELS.values())),
        yOffset=alt.YOffset("loss:N", sort=LOSS_ORDER),
        color=alt.Color(
            "loss:N",
            scale=alt.Scale(
                domain=LOSS_ORDER,
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN],
            ),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    rules = (
        alt.Chart(pl.DataFrame({"x": [1.0, 1.3]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    line = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "ratio:Q",
            title="Christmas error / ordinary-weekday error (1 = same; the 1.3 line is the limit)",
        )
    )
    return (rules + line + dot).properties(
        width=WIDTH - 120,
        height=330,
        title=_title(
            text="Christmas 2020 to 2023 broke the 1.3 limit; the last two Christmases did not",
            subtitle=(
                "Dot: mean error on 24 December to 2 January over the mean error on ordinary "
                "weekdays at the same half-hours, leads 0.5 to 168 hours, all Primary substations "
                "with a fitted model. Line: 95% interval from resampling days. Green: only fits "
                "with 180 or more days of training rows. Dashed lines at 1 (same) and 1.3 (the "
                "limit). "
                "Lower is better."
            ),
        ),
    )


def figure_sites(*, m12: dict) -> alt.TopLevelMixin:
    """Ratio per substation and Christmas as a heat map."""
    frame = _ratios(data=m12, scope="all", metric="qs", group="christmas").filter(
        (pl.col("season") != "all") & (pl.col("site") != "all")
    )
    frame = frame.with_columns(
        season=pl.col("season").map_elements(lambda s: YEAR_LABELS[int(s)], return_dtype=pl.String),
        text=pl.col("ratio").round(1).cast(pl.String),
    )
    base = alt.Chart(frame).encode(
        x=alt.X("season:N", title=None, sort=list(YEAR_LABELS.values())),
        y=alt.Y("site:N", title="Substation", sort=list(common.ALL_SITES)),
    )
    heat = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "ratio:Q",
            scale=alt.Scale(domain=[1.0, 1.3, 3.0], range=["#FFFFFF", "#FFC4AD", ocf.BRAND_ORANGE]),
            legend=alt.Legend(title="Ratio", orient="top", direction="horizontal"),
        )
    )
    label = base.mark_text(fontSize=9).encode(text="text:N")  # ty: ignore[unresolved-attribute]
    return (heat + label).properties(
        width=WIDTH - 120,
        height=330,
        title=_title(
            text="A few substations carry the bad Christmases",
            subtitle=(
                "Quantile-score ratio (Christmas over ordinary weekdays) per substation and "
                "Christmas; blank where the substation has no fitted model that year. Cream is "
                "1.0 or below, pale orange 1.3 (the limit), orange 3.0 or more. Lower is better."
            ),
        ),
    )


def figure_groups(*, m12: dict) -> alt.TopLevelMixin:
    """Ratio per day group, one line per Christmas."""
    frame = _ratios(data=m12, scope="all", metric="abs_mean", site="all").filter(
        pl.col("group").is_in(list(GROUP_ORDER)) & (pl.col("season") != "all")
    )
    frame = frame.with_columns(
        season=pl.col("season").map_elements(lambda s: YEAR_LABELS[int(s)], return_dtype=pl.String)
    )
    colours = [
        GREY, "#BBBBBB", ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE,
        ocf.DATA_GREEN,
    ]  # fmt: skip
    chart = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("group:N", title=None, sort=list(GROUP_ORDER)),
            y=alt.Y("ratio:Q", title="Error / ordinary-weekday error"),
            color=alt.Color(
                "season:N",
                scale=alt.Scale(domain=list(YEAR_LABELS.values()), range=colours),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
    )  # ty: ignore[unresolved-attribute]
    return chart.properties(
        width=WIDTH - 120,
        height=260,
        title=_title(
            text="The error is highest on 24 to 26 December in the bad years",
            subtitle=(
                "Ratio of the mean absolute error of the mean forecast on each group of days to "
                "the ordinary-weekday error at the same half-hours, all substations pooled, "
                "leads 0.5 to 168 hours. Lower is better."
            ),
        ),
    )


def figure_bias(*, m12: dict, all_years: dict) -> alt.TopLevelMixin:
    """Signed error on 25 and 26 December as a share of measured power."""
    rows = []
    for arm, data in (("12 months of history", m12), ("all years of history", all_years)):
        for b in data["bias"]:
            if b["site"] == "all" and b["group"] == "25-26 Dec" and b["season"] != "all":
                rows.append(  # noqa: PERF401
                    {
                        "arm": arm, "season": YEAR_LABELS[int(b["season"])],
                        "share": 100.0 * b["signed_share"],
                    }
                )  # fmt: skip
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("season:N", title="Christmas", sort=list(YEAR_LABELS.values())),
        yOffset=alt.YOffset("arm:N"),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=["12 months of history", "all years of history"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [-5.0, 0.0, 5.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    bars = base.mark_bar(size=8, clip=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "share:Q",
            scale=alt.Scale(domain=[-16, 20]),
            title="Measured minus forecast, % of measured power (0 is unbiased)",
        )
    )
    return (
        (zero + bars)
        .properties(
            width=WIDTH - 120,
            height=260,
            title=_title(
                text="The 12-month forecast for 25 and 26 December was within 5% from 2022-23",
                subtitle=(
                    "Summed measured minus mean forecast over 25 and 26 December, as a share of "
                    "summed measured power, all substations pooled, leads 0.5 to 168 hours. "
                    "Positive: demand above the forecast. Dashed lines at 0 and plus or minus 5%. "
                    "Christmas 2019-20 (+47%) is off the axis."
                ),
            ),
        )
        .configure_axisX(domain=True)
    )


def main() -> None:
    """Draw every figure."""
    folder = common.WORK_DIR / "report"
    m12 = json.loads((folder / "report_m12.json").read_text())
    all_years = json.loads((folder / "report_all.json").read_text())
    _save(figure_headline(m12=m12, all_years=all_years), name="figure_1_headline")
    _save(figure_sites(m12=m12), name="figure_2_sites")
    _save(figure_groups(m12=m12), name="figure_3_groups")
    _save(figure_bias(m12=m12, all_years=all_years), name="figure_4_bias")


if __name__ == "__main__":
    main()
