"""Draw the figures of experiment 109 as SVG into the docs folder, from the report and predictions.

Only Primary substations appear, under their D-labels, so dates may be shown. The numbers come from
`report.txt` (`e00109_report.py`), `tuned.json`, and the gate files, so a figure and the page quote
the same values. Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline`.

    uv run python ml-experiments/00109-*/e00109_charts.py
"""

import io
import json
import textwrap
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import e00109_common as common
import e00109_report as report
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
REPORT_PATH: Final[Path] = DOCS_DIR / "report.txt"
PNG_DIR: Final[Path] = Path("/tmp/e00109_png")
WIDTH: Final[int] = 560
"""Plot-area width: the axis labels bring a chart to the 680 px text column."""
WRAP: Final[int] = 95
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16"]
GATE_NOTE: Final[str] = "Gate failed under the pre-registered definition; a post hoc one passes."
SITE_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
]
ARM_LABELS: Final[dict[str, str]] = {
    "g": "G per site",
    "n": "N pooled",
    "n_full": "N-full complete pooling",
    "g_4w": "G-4w",
    "n_4w": "N-4w",
    "n_full_4w": "N-full-4w",
    "x_4w": "X-4w (XGBoost)",
    "x_pool_4w": "X-pool-4w (XGBoost)",
    "g_3m": "G-3m",
    "n_3m": "N-3m",
    "x_3m": "X-3m (XGBoost)",
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    wrapped = textwrap.wrap(subtitle, WRAP)
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
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
        if heading is not None and lines and not heading.startswith("Decision"):
            try:
                sections[heading] = pl.read_csv(
                    io.StringIO("\n".join(lines)), infer_schema_length=None
                )
            except pl.exceptions.PolarsError:
                return

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


def figure_1_contrasts(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The five planned contrasts with 99% intervals, scored year and replicate year."""
    table = _section(sections=sections, start="Planned contrasts")
    rows = []
    for row in table.iter_rows(named=True):
        for year, label in (("scored", "Scored year"), ("replicate", "Replicate year")):
            rows.append(
                {
                    "contrast": row["contrast"],
                    "year": label,
                    "estimate": row[f"{year} QS"],
                    "low": row[f"{year} low"],
                    "high": row[f"{year} high"],
                    "better": f"{row[f'{year} sites better']} of 5",
                }
            )
    data = pl.DataFrame(rows)
    order = table["contrast"].to_list()
    colour = alt.Color(
        "year:N",
        scale=alt.Scale(
            domain=["Scored year", "Replicate year"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    y = alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    offset = alt.YOffset("year:N", sort=["Scored year", "Replicate year"])
    base = alt.Chart(data)
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q", title="Quantile score difference (points of capacity; negative is better)"
        ),
        x2="high:Q",
        y=y,
        yOffset=offset,
        color=colour,
    )
    dots = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x="estimate:Q", y=y, yOffset=offset, color=colour, tooltip=["contrast", "year", "better"]
    )
    return (rule + bars + dots).properties(
        width=WIDTH,
        height=230,
        title=_title(
            text="Pooling matches a model per site on full history and helps a new site",
            subtitle=(
                "Dot: estimate. Line: 99% interval from resampling whole calendar months. First "
                "arm minus second; 0 means equal. Five decision sites. " + GATE_NOTE
            ),
        ),
    )


def figure_2_absolute(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The quantile score of the headline arms at each decision site, scored year."""
    table = _section(sections=sections, start="Absolute").filter(pl.col("cut") == "scored")
    arms = [a for a in ARM_LABELS if a in table["arm"].to_list()]
    long = (
        table.filter(pl.col("arm").is_in(arms))
        .unpivot(on=SITE_ORDER, index=["arm"], variable_name="site", value_name="score")
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    colour = alt.Color(
        "site:N",
        sort=SITE_ORDER,
        scale=alt.Scale(domain=SITE_ORDER, range=SITE_COLOURS),
        legend=alt.Legend(orient="top", title=None),
    )
    return (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("score:Q", title="Quantile score (points of capacity; smaller is better)"),
            y=alt.Y("label:N", sort=[ARM_LABELS[a] for a in arms], title=None),
            color=colour,
        )
        .properties(
            width=WIDTH,
            height=26 * len(arms),
            title=_title(
                text="Absolute scores by arm and substation, scored year",
                subtitle="Each dot is one substation's mean score over 2025-07 to 2026-06.",
            ),
        )
    )


def figure_3_curve(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The pooled and the per-site score against the weeks of history, both years."""
    table = _section(sections=sections, start="Onboarding").unpivot(
        on=["N", "G"], index=["year", "weeks of history"], variable_name="model", value_name="score"
    )
    labelled = table.with_columns(
        model=pl.col("model").replace({"N": "N pooled", "G": "G per site"}),
        year=pl.col("year").replace({"scored": "Scored year", "replicate": "Replicate year"}),
    )
    colour = alt.Color(
        "model:N",
        scale=alt.Scale(domain=["N pooled", "G per site"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(orient="top", title=None),
    )
    base = alt.Chart(labelled).encode(
        x=alt.X(
            "weeks of history:Q",
            scale=alt.Scale(type="log", domain=[1.6, 32]),
            axis=alt.Axis(values=[2, 4, 8, 13, 26]),
            title="Weeks of history at the new substation (log scale)",
        ),
        y=alt.Y("score:Q", title="Score (points)", scale=alt.Scale(domain=[0, 10])),
        color=colour,
    )
    chart = (
        base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=50)  # ty: ignore[unsupported-operator]
    ).properties(width=WIDTH, height=150)
    return chart.facet(row=alt.Row("year:N", title=None)).properties(
        title=_title(
            text="The gain from pooling falls as the new substation's history grows",
            subtitle=(
                "Quantile score in points of capacity; smaller is better. Mean over the five "
                "decision sites; each takes a turn as the new substation."
            ),
        )
    )


def figure_4_lambda() -> alt.TopLevelMixin:
    """The tuning year's ratios against the deviation penalty."""
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    summary = pl.DataFrame(tuned["summary"]).filter(pl.col("lam") > 0)
    long = summary.unpivot(
        on=["ratio_4w", "ratio_3m"], index=["lam"], variable_name="history", value_name="ratio"
    ).with_columns(
        history=pl.col("history").replace({"ratio_4w": "4 weeks", "ratio_3m": "3 months"})
    )
    colour = alt.Color(
        "history:N",
        scale=alt.Scale(domain=["4 weeks", "3 months"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(orient="top", title=None),
    )
    base = alt.Chart(long).encode(
        x=alt.X(
            "lam:Q",
            scale=alt.Scale(type="log"),
            axis=alt.Axis(
                values=[100, 300, 1000, 3000, 10000, 30000, 100000, 1000000], format="~s"
            ),
            title="Deviation penalty (rows of evidence)",
        ),
        y=alt.Y(
            "ratio:Q",
            title="Score over the unpooled score (smaller is better)",
            scale=alt.Scale(zero=False),
        ),
        color=colour,
    )
    star = (
        alt.Chart(pl.DataFrame({"lam": [tuned["lambda_star"]]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="lam:Q")
    )
    return (
        base.mark_line(strokeWidth=2)  # ty: ignore[unsupported-operator]
        + base.mark_point(filled=True, size=50)
        + star
    ).properties(
        width=WIDTH,
        height=200,
        title=_title(
            text="The tuning year picks the penalty",
            subtitle=(
                "Mean score of the pooled model over the unpooled model on the same rows, with one "
                f"decision site held out at a time. Dashed: the chosen {tuned['lambda_star']:g}."
            ),
        ),
    )


def figure_5_gate() -> alt.TopLevelMixin:
    """The four gate runs, line by line."""
    table = pl.read_csv(DOCS_DIR / "gate_runs.csv")
    data = table.with_columns(
        status=pl.when(pl.col("passed")).then(pl.lit("passed")).otherwise(pl.lit("failed"))
    )
    base = alt.Chart(data).encode(
        x=alt.X("run:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(
            "line:N",
            sort=table["line"].unique(maintain_order=True).to_list(),
            title=None,
            axis=alt.Axis(labelLimit=320),
        ),
    )
    rects = base.mark_rect(stroke="white").encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["passed", "failed"], range=[ocf.DATA_GREEN, ocf.BRAND_ORANGE]),
            legend=alt.Legend(orient="top", title=None),
        )
    )
    return rects.properties(
        width=WIDTH,
        height=160,
        title=_title(
            text="The known-answer gate failed under its pre-registered definition",
            subtitle=(
                "Run 1 is the pre-registered definition; runs 2 to 5 are post hoc changes, each on "
                "a new seed. Only run 4 passes every line. Line (b) has little power, and line (d) "
                "passed on 3 of 5 seeds."
            ),
        ),
    )


def figure_6_weeks() -> alt.TopLevelMixin:
    """Actual power and the 4-week forecasts in a summer week and a winter week at each site."""
    weeks = {
        "Summer: 14 to 20 July 2025": date(2025, 7, 14),
        "Winter: 12 to 18 January 2026": date(2026, 1, 12),
    }
    labels = {"n_4w": "N-4w", "g_4w": "G-4w"}
    frames = []
    for week, start in weeks.items():
        for name, label in labels.items():
            frame = report.load_arm(label="scored", arm=name, drop_switching=False)
            part = frame.filter(
                pl.col("valid_time").dt.date() >= start,
                pl.col("valid_time").dt.date() < pl.lit(start) + pl.duration(days=7),
                pl.col("site_name").is_in(SITE_ORDER),
            ).with_columns(
                week=pl.lit(week),
                site=pl.col("site_name"),
                series=pl.lit(label),
                value=pl.col("q50"),
                day=(
                    pl.col("valid_time")
                    - pl.datetime(start.year, start.month, start.day, time_zone="UTC")
                ).dt.total_hours()
                / 24.0,
            )
            frames.append(part.select("site", "week", "series", "day", "value"))
            if name == "n_4w":
                frames.append(
                    part.select(
                        "site",
                        "week",
                        pl.lit("Measured").alias("series"),
                        "day",
                        pl.col("actual").alias("value"),
                    )
                )
    table = pl.concat(frames)
    order = ["Measured", "N-4w", "G-4w"]

    def week_chart(week: str) -> alt.FacetChart:
        return (
            alt.Chart(table.filter(pl.col("week") == week))  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X(
                    "day:Q",
                    title="Day",
                    axis=alt.Axis(values=list(range(8))),
                    scale=alt.Scale(domain=[0, 7]),
                ),
                y=alt.Y("value:Q", title="Net power (points)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "series:N",
                    sort=order,
                    scale=alt.Scale(
                        domain=order, range=["#404040", ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
            .properties(width=WIDTH // 5 - 10, height=110)
            .facet(column=alt.Column("site:N", sort=SITE_ORDER, title=week))
        )

    return alt.vconcat(*[week_chart(week) for week in weeks]).properties(
        title=_title(
            text="Four weeks of June history, one summer week and one winter week",
            subtitle=(
                "Median forecasts of the pooled and the per-site model that held only the 4 weeks "
                "before 2025-07-01, against the measured power. Day 0 is Monday 00:00 UTC."
            ),
        )
    )


def figure_7_tails(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Coverage at the four tail levels against the nominal level, scored year."""
    table = _section(sections=sections, start="Tail").filter(pl.col("year") == "scored")
    nominal = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
    rows = []
    for row in table.iter_rows(named=True):
        for level, value in nominal.items():
            rows.append(
                {
                    "arm": ARM_LABELS.get(row["arm"], row["arm"]),
                    "level": level,
                    "excess": row[f"{level} coverage"] - value,
                }
            )
    data = pl.DataFrame(rows)
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    dots = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X(
                "excess:Q", title="Share of actuals below the quantile, minus the nominal level"
            ),
            y=alt.Y("arm:N", title=None),
            color=alt.Color(
                "level:N",
                sort=list(nominal),
                scale=alt.Scale(
                    domain=list(nominal),
                    range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    return (rule + dots).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="Tail calibration at the 0.01, 0.05, 0.95, and 0.99 quantiles",
            subtitle="Zero means a quantile covers its nominal share exactly. Five decision sites.",
        ),
    )


def main() -> None:
    """Draw every figure."""
    sections = read_sections()
    _save(figure_1_contrasts(sections=sections), name="figure_01_contrasts")
    _save(figure_2_absolute(sections=sections), name="figure_02_absolute")
    _save(figure_3_curve(sections=sections), name="figure_03_onboarding")
    _save(figure_4_lambda(), name="figure_04_lambda")
    _save(figure_5_gate(), name="figure_05_gate")
    _save(figure_6_weeks(), name="figure_06_weeks")
    _save(figure_7_tails(sections=sections), name="figure_07_tails")


if __name__ == "__main__":
    main()
