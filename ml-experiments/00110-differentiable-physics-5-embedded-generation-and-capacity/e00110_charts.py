"""Draw the figures of experiment 110 as SVG into the docs folder, from the report and predictions.

Only Primary substations appear, under their D-labels, so calendar dates may be shown. The real
generator case G-c appears in no figure here: it has no forecast arm, and its numbers are quoted in
the text under the neutral label. The tables come from `report.txt` (`e00110_report.py`), so a
figure and the page quote the same numbers. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00110-*/e00110_charts.py
"""

import io
import textwrap
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import e00110_report as report
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00110-differentiable-physics-5-embedded-generation-and-capacity"
)
REPORT_PATH: Final[Path] = DOCS_DIR / "report.txt"
PNG_DIR: Final[Path] = Path("/tmp/e00110_png")
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16"]
GREY: Final[str] = "#808080"
ARM_LABELS: Final[dict[str, str]] = {
    "b": "B pinned capacity",
    "b_d1": "B-D1 drift",
    "b_clip": "B-clip",
    "b_node": "B-node",
    "b_gen": "B-gen node + clip",
    "k_base": "K-base",
    "k_i": "K-i",
    "k_ii": "K-ii",
    "k_iii": "K-iii",
    "k_iiiii": "K-ii+iii",
}
FORECAST_WEEKS: Final[dict[str, date]] = {
    "Winter: 5 to 11 Jan 2026": date(2026, 1, 5),
    "Spring: 6 to 12 Apr 2026": date(2026, 4, 6),
    "Summer: 7 to 13 Jul 2025": date(2025, 7, 7),
    "Autumn: 6 to 12 Oct 2025": date(2025, 10, 6),
}
"""The first full Monday-to-Sunday week of January, April, July, and October in the scored year,
fixed by that rule before any score was read."""


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
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


def read_sections() -> dict[str, pl.DataFrame]:
    """Return each table of `report.txt` by the text of its `==` heading."""
    sections: dict[str, pl.DataFrame] = {}
    heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if heading is not None and lines:
            sections[heading] = pl.read_csv(io.StringIO("\n".join(lines)), infer_schema_length=None)

    for line in REPORT_PATH.read_text().splitlines():
        if line.startswith("== "):
            flush()
            heading, lines = line.strip("= ").strip(), []
        elif line.strip():
            lines.append(line)
    flush()
    return sections


def _section(*, sections: dict[str, pl.DataFrame], start: str) -> pl.DataFrame:
    return next(frame for name, frame in sections.items() if name.startswith(start))


def figure_1_headline(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The planned contrasts with intervals, and the replicate year beside each."""
    planned = _section(sections=sections, start="Planned contrasts (")
    rows = []
    for row in planned.iter_rows(named=True):
        if row["scored QS"] is None:
            continue
        name = str(row["contrast"])
        short = (
            name.split(" ", 1)[0]
            + ": "
            + name.split(" ", 1)[1]
            .replace(
                "DP against lagless XGBoost: B - x_best (both unfiltered)",
                "DP against XGBoost: B - x_best",
            )
            .replace(
                "generation side harms MW forecast: B-gen - B-D1", "generation side: B-gen - B-D1"
            )
        )
        rows.append(
            {
                "contrast": short,
                "estimate": row["scored QS"],
                "low": row["scored low"],
                "high": row["scored high"],
                "kind": "scored year (interval)",
                "label": f"{row['scored sites better']} sites better",
            }
        )
        rows.append(
            {
                "contrast": short,
                "estimate": row["replicate QS"],
                "low": None,
                "high": None,
                "kind": "replicate year",
                "label": f"{row['replicate sites better']} sites",
            }
        )
    table = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        label_x=pl.max_horizontal("estimate", "high")
    )
    order = list(dict.fromkeys(table["contrast"].to_list()))
    base = alt.Chart(table).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        yOffset=alt.YOffset("kind:N"),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["scored year (interval)", "replicate year"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(title="Year", orient="top", direction="horizontal"),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Difference in quantile score (points; below zero favours the first arm)",
        )
    )
    text = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=6, fontSize=10)
        .encode(
            y=alt.Y("contrast:N", sort=order),
            yOffset=alt.YOffset("kind:N"),
            x=alt.X("label_x:Q"),
            text="label:N",
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot + text).properties(
        width=WIDTH - 200,
        height=46 * len(order),
        title=_title(
            text="The four planned contrasts that ran, in two years",
            subtitle=(
                "Dot: difference in mean quantile score, filtered forecasts, rows outside the "
                "switching events (C1, C4 and C5 at the five decision substations; C2 and C3 at "
                "the 8 MVA substations). Orange: scored year, 1 July 2025 to 30 June 2026, with a "
                "99% interval from resampling whole calendar months. Blue: replicate year, "
                "estimate only. Dashed line: no difference. C4 did not run because no capacity "
                "arm was identified. C5 is the unfiltered DP against unfiltered lagless XGBoost."
            ),
        ),
    )


def figure_2_absolute(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Every site-level arm's mean quantile score at each decision substation, both years."""
    absolute = _section(sections=sections, start="Absolute quantile score")
    rows = []
    for row in absolute.iter_rows(named=True):
        if row["arm"] not in ARM_LABELS or not row["filtered"]:
            continue
        rows.extend(
            {
                "year": "Scored year" if row["year"] == "scored" else "Replicate year",
                "arm": ARM_LABELS[row["arm"]],
                "site": site,
                "score": row[site],
            }
            for site in SITE_ORDER
            if row[site] is not None
        )
    table = pl.DataFrame(rows)
    order = list(ARM_LABELS.values())
    base = alt.Chart(table).encode(
        x=alt.X("site:N", sort=SITE_ORDER, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("arm:N", sort=order, title=None),
    )
    heat = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "score:Q",
            scale=alt.Scale(domain=[1.6, 3.2], range=["#FFFFFF", ocf.DATA_BLUE], clamp=True),
            legend=alt.Legend(title="Score (points)", orient="top", direction="horizontal"),
        )
    )
    text = base.mark_text(fontSize=10).encode(text=alt.Text("score:Q", format=".2f"))  # ty: ignore[unresolved-attribute]
    return (
        (heat + text)
        .properties(width=WIDTH - 160, height=17 * len(order))
        .facet(column=alt.Column("year:N", sort=["Scored year", "Replicate year"], title=None))
        .properties(
            title=_title(
                text="Quantile score of every MW-substation arm, with the filter",
                subtitle=(
                    "Twice the mean pinball loss over 11 quantile levels, in points of effective "
                    "capacity (smaller is better), day-ahead forecasts with the recent-level "
                    "filter, on rows outside the switching events."
                ),
            )
        )
    )


def figure_3_capacity(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Fitted capacity of the three tested capacity arms against the vintage, scored cut."""
    capacity = _section(sections=sections, start="Capacity by site").filter(
        pl.col("cut") == "scored",
        pl.col("arm").is_in(["k_base", "k_i", "k_ii", "k_iii", "k_iiiii"]),
    )
    capacity = capacity.with_columns(
        arm_label=pl.col("arm").replace_strict(ARM_LABELS),
        vintage_low=0.8 * pl.col("vintage_mean"),
        vintage_high=1.2 * pl.col("vintage_mean"),
    )
    order = [ARM_LABELS[a] for a in ("k_base", "k_i", "k_ii", "k_iii", "k_iiiii")]
    base = alt.Chart(capacity).encode(
        y=alt.Y("arm_label:N", sort=order, title=None),
        color=alt.Color(
            "arm_label:N",
            sort=order,
            scale=alt.Scale(
                domain=order,
                range=[
                    ocf.BRAND_ORANGE,
                    ocf.DATA_BLUE,
                    ocf.DATA_SKY,
                    ocf.DATA_PURPLE,
                    ocf.DATA_GREEN,
                ],
            ),
            legend=None,
        ),
    )
    band = (
        alt.Chart(capacity)  # ty: ignore[unresolved-attribute]
        .mark_rect(color="#DDDDDD")
        .encode(x="vintage_low:Q", x2="vintage_high:Q")
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="range_low:Q", x2="range_high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("capacity:Q", title="Fitted PV capacity (points of effective capacity)")
    )
    return (
        (band + rule + dot)
        .properties(width=WIDTH - 140, height=90)
        .facet(row=alt.Row("site:N", sort=SITE_ORDER, title=None, header=alt.Header(labelAngle=0)))
        .properties(
            title=_title(
                text="Fitted PV capacity at the scored cut, by capacity arm",
                subtitle=(
                    "Dot: capacity of the last trained season-year. Line: 90% range from a "
                    "stratified month bootstrap, widened for the number of months. Grey band: "
                    "20% either side of the real-time vintage mean of experiment 45. No arm was "
                    "labelled identified."
                ),
            )
        )
    )


def figure_4_k_iii(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """K-iii over K-base against the target band."""
    ratio = _section(sections=sections, start="K-iii over K-base").rename(
        {"K-iii / K-base": "ratio", "range low": "low", "range high": "high"}
    )
    band = (
        alt.Chart(pl.DataFrame({"lo": [0.65], "hi": [0.87]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color="#DDDDDD")
        .encode(x="lo:Q", x2="hi:Q")
    )
    base = alt.Chart(ratio).encode(
        y=alt.Y("site:N", sort=SITE_ORDER, title=None),
        yOffset=alt.YOffset("cut:N"),
        color=alt.Color(
            "cut:N",
            sort=["scored", "replicate"],
            scale=alt.Scale(
                domain=["scored", "replicate"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title="Cut", orient="top", direction="horizontal"),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "ratio:Q",
            title="K-iii capacity divided by K-base capacity",
            scale=alt.Scale(domain=[0.5, 1.3]),
        )
    )
    return (band + rule + dot).properties(
        width=WIDTH - 60,
        height=150,
        title=_title(
            text="K-iii over K-base, against the range a real farm's anomaly slope predicts",
            subtitle=(
                "Dot: ratio of the fitted capacities. Line: 90% range from the same month draws. "
                "Grey band: 0.65 to 0.87, the slope with which CAMS irradiance anomalies track a "
                "real farm (experiment 63). K-iii failed gate tests 1 and 2, so it is not an "
                "identified arm and this ratio supports no claim about K-base."
            ),
        ),
    )


def figure_5_forecast_weeks() -> alt.TopLevelMixin:
    """Measured power and B-gen's median and bands in four calendar-rule weeks per site."""
    frame = report.load_side(
        label="scored", side=report.Side(arm="b_gen", filtered=True), drop_switching=False
    ).filter(pl.col("site_name").is_in(SITE_ORDER))
    parts = []
    for week, start in FORECAST_WEEKS.items():
        origin = pl.datetime(start.year, start.month, start.day, time_zone="UTC")
        parts.append(
            frame.filter(
                pl.col("valid_time") >= origin, pl.col("valid_time") < origin + pl.duration(days=7)
            ).with_columns(
                week=pl.lit(week),
                site=pl.col("site_name"),
                day=(pl.col("valid_time") - origin).dt.total_hours() / 24.0,
            )
        )
    table = pl.concat(parts).select(
        "site", "week", "day", "actual", "q05", "q10", "q50", "q90", "q95"
    )
    weeks = list(FORECAST_WEEKS)
    base = alt.Chart(table).encode(
        x=alt.X(
            "day:Q",
            title="Days from Monday",
            axis=alt.Axis(values=[0, 3, 7]),
            scale=alt.Scale(domain=[0, 7]),
        )
    )
    band90 = base.mark_area(opacity=0.2, color=ocf.DATA_BLUE).encode(y="q05:Q", y2="q95:Q")  # ty: ignore[unresolved-attribute]
    band80 = base.mark_area(opacity=0.3, color=ocf.DATA_BLUE).encode(y="q10:Q", y2="q90:Q")  # ty: ignore[unresolved-attribute]
    median = base.mark_line(strokeWidth=1, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q50:Q", title=None)
    )
    actual = base.mark_line(strokeWidth=1, color="#333333").encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    return (
        (band90 + band80 + median + actual)
        .properties(width=(WIDTH - 60) // 4, height=60)
        .facet(
            row=alt.Row(
                "site:N",
                sort=SITE_ORDER,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
            column=alt.Column("week:N", sort=weeks, title=None),
        )
        .properties(
            title=_title(
                text="Day-ahead forecasts and measured net power in four calendar-rule weeks",
                subtitle=(
                    "Net power as a percentage of each substation's effective capacity (negative "
                    "is export). Black: measured. Blue line: median of B-gen with the filter. "
                    "Darker band: 80% interval (q10 to q90). Lighter band: 90% interval (q05 to "
                    "q95). The weeks are the first full Monday-to-Sunday week of January, April, "
                    "July, and October in the scored year, chosen by a calendar rule that does not "
                    "depend on any score."
                ),
            )
        )
    )


def figure_6_tails(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Coverage at the four tail levels for the MW-substation arms, scored year, filtered."""
    tails = _section(sections=sections, start="Tail calibration").filter(
        pl.col("year") == "scored",
        pl.col("filtered"),
        pl.col("arm").is_in(list(ARM_LABELS)),
    )
    rows = []
    for row in tails.iter_rows(named=True):
        rows.extend(
            {"arm": ARM_LABELS[row["arm"]], "level": level, "coverage": row[f"{level} coverage"]}
            for level in ("q01", "q05", "q95", "q99")
        )
    table = pl.DataFrame(rows).with_columns(
        miss=(
            pl.col("coverage")
            - pl.col("level").replace_strict(
                {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}, return_dtype=pl.Float64
            )
        )
        * 100.0
    )
    order = list(ARM_LABELS.values())
    points = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=45)
        .encode(
            x=alt.X(
                "miss:Q",
                title="Miss (points)",
                scale=alt.Scale(domain=[-2, 2]),
                axis=alt.Axis(values=[-2, 0, 2]),
            ),
            y=alt.Y("arm:N", sort=order, title=None),
            color=alt.value(ocf.DATA_BLUE),
        )
    )
    zero = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x=alt.datum(0))
    )
    return (
        alt.layer(zero, points, data=table)
        .properties(width=(WIDTH - 200) // 2, height=12 * len(order))
        .facet(column=alt.Column("level:N", sort=["q01", "q05", "q95", "q99"], title=None))
        .properties(
            title=_title(
                text="Tail calibration of the MW-substation arms, scored year",
                subtitle=(
                    "Blue dot: the share of observed half-hours below the forecast quantile, "
                    "minus the nominal share (the miss, in percentage points). Orange line: "
                    "no miss. Filtered forecasts, rows outside the switching events, five decision "
                    "substations."
                ),
            )
        )
    )


def figure_7_node_depth(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The deepest outage the node reads at each decision substation."""
    node = _section(sections=sections, start="Node: deepest outage").filter(
        pl.col("arm") == "b_gen"
    )
    base = alt.Chart(node).encode(
        y=alt.Y("site:N", sort=["D5", "D9", "D10", "D11", "D14", "D16"], title=None),
        yOffset=alt.YOffset("cut:N"),
        color=alt.Color(
            "cut:N",
            sort=["scored", "replicate"],
            scale=alt.Scale(
                domain=["scored", "replicate"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title="Cut", orient="top", direction="horizontal"),
        ),
    )
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("evidence points:Q", title="Depth of the deepest outage the node reads (points)")
    )
    one = (
        alt.Chart(pl.DataFrame({"x": [1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (one + dot).properties(
        width=WIDTH - 80,
        height=150,
        title=_title(
            text="The node reads a generator at every decision substation",
            subtitle=(
                "Depth of the deepest dip with higher levels on both sides, in points of "
                "effective capacity, from B-gen. Dashed line: the pre-registered limit of 1 "
                "point, below which the node counts as off. No site is below it."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    sections = read_sections()
    _save(figure_1_headline(sections=sections), name="figure_01_headline")
    _save(figure_2_absolute(sections=sections), name="figure_02_absolute")
    _save(figure_3_capacity(sections=sections), name="figure_03_capacity")
    _save(figure_4_k_iii(sections=sections), name="figure_04_k_iii")
    _save(figure_5_forecast_weeks(), name="figure_05_forecast_weeks")
    _save(figure_6_tails(sections=sections), name="figure_06_tails")
    _save(figure_7_node_depth(sections=sections), name="figure_07_node_depth")


if __name__ == "__main__":
    main()
