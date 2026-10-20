"""Draw the figures of experiment 120 as SVG into the docs folder, from the report and fits.

Only Primary substations appear, under their D-labels, so dates may be shown. D18 is a generator
series: it appears on no D-label figure, and a shape that takes more than 30% of its weight from
D18 is left off the D-label shape figure. The numbers come from `report.txt`
(`e00120_report.py`), `tuned.json`, `gate_report.json`, and the stored fits, so a figure and the
page quote the same values. Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline`.

    uv run python ml-experiments/00120-*/e00120_charts.py
"""

import io
import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00120_common as common
import e00120_report as report
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
REPORT_PATH: Final[Path] = DOCS_DIR / "report.txt"
PNG_DIR: Final[Path] = Path("/tmp/e00120_png")
WIDTH: Final[int] = 560
"""Plot-area width: the axis labels bring a chart to the 680 px text column."""
WRAP: Final[int] = 95
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16"]
SITE_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
]
SHAPE_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    "#808080",
]
DAY_TYPE_NAMES: Final[list[str]] = ["Weekday", "Saturday", "Sunday", "Bank holiday"]
ARM_LABELS: Final[dict[str, str]] = {
    "g": "G per site",
    "n": "N fleet mean",
    "n_full": "N-full complete pooling",
    "sh": "SH library",
    "sh_k1": "SH-K1 one shape",
    "sh_g0": "SH-g0 no orthogonality",
    "sh_nodev": "SH-nodev no deviation",
    "sh_wd": "SH-wd weights per day type",
    "sh_all": "SH-all all years",
    "x_2y": "Lagless XGBoost",
    "x_2yamq": "Lagless XGBoost, solar offset",
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
}
D18_SHARE_LIMIT: Final[float] = 0.3
"""A shape taking more than this share of its weight from D18 is left off the D-label figure."""


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    wrapped = textwrap.wrap(subtitle, WRAP)
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    with (DOCS_DIR / f"{name}.svg").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def read_sections() -> dict[str, pl.DataFrame]:
    """Return each table of `report.txt` by the text of its `==` heading."""
    sections: dict[str, pl.DataFrame] = {}
    heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if heading is not None and lines:
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


def _zero_rule() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )


def figure_headline(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The three planned contrasts with 98.33% intervals, scored year and replicate year."""
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
    colour = alt.Color(
        "year:N",
        scale=alt.Scale(
            domain=["Scored year", "Replicate year"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    y = alt.Y("contrast:N", sort=table["contrast"].to_list(), title=None)
    offset = alt.YOffset("year:N", sort=["Scored year", "Replicate year"])
    base = alt.Chart(data)
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
    return (_zero_rule() + bars + dots).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="The library matches the fleet mean and trails the per-site model (scored year)",
            subtitle=(
                "Dot: estimate. Line: 98.33% interval from resampling whole calendar months. First "
                "arm minus second; 0 means equal. Five decision substations, headline seed."
            ),
        ),
    )


def figure_winter(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """C1 and C2 by season: all rows, winter, evening peak, winter nights, and cold-snap days."""
    table = _section(sections=sections, start="Winter breakdown").filter(
        pl.col("contrast").is_in(["C1 SH - N", "C2 SH - G"]) & pl.col("QS").is_not_nan()
    )
    colour = alt.Color(
        "year:N",
        scale=alt.Scale(domain=["scored", "replicate"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(orient="top", title="Year"),
    )
    y = alt.Y("season:N", sort=table["season"].unique(maintain_order=True).to_list(), title=None)
    offset = alt.YOffset("year:N", sort=["scored", "replicate"])
    base = alt.Chart()
    rule = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x=alt.datum(0))
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
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x="QS:Q", y=y, yOffset=offset, color=colour
    )
    return (
        alt.layer(rule, bars, dots)
        .properties(width=WIDTH, height=140)
        .facet(row=alt.Row("contrast:N", title=None), data=table)
        .properties(
            title=_title(
                text="In the scored year the library loses to the per-site model in winter",
                subtitle=(
                    "SH minus N (top) and SH minus G (bottom). Dot: estimate. Line: 98.33% "
                    "interval from resampling whole calendar months; the cold-snap interval is "
                    "degenerate because those days fall in few months. Winter is November to "
                    "March; a winter loss is a reason to hold the library back."
                ),
            )
        )
    )


def figure_absolute(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The quantile score of every arm at each decision substation, scored year."""
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
            x=alt.X(
                "score:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("label:N", sort=[ARM_LABELS[a] for a in arms], title=None),
            color=colour,
        )
        .properties(
            width=WIDTH,
            height=24 * len(arms),
            title=_title(
                text="Absolute scores by arm and substation, scored year",
                subtitle="Each dot is one substation's mean score over 2025-07 to 2026-06.",
            ),
        )
    )


def figure_gate() -> alt.TopLevelMixin:
    """Each gate line's pass rate with its Wilson 90% interval and the 75% threshold."""
    gate = json.loads((common.WORK_DIR / "gate_report.json").read_text())
    import e00120_gate as gate_module

    short = {
        "1": "1 shapes recovered",
        "2": "2 weights recovered",
        "3": "3 no duplicate shapes",
        "4": "4 deviation bump kept out",
        "5": "5 null fleet",
        "6": "6 K chosen",
        "7": "7 local optima",
        "8": "8 module tests",
        "9": "9 score power",
        "10": "10 Economy 7 step",
    }
    rows = []
    for name, (passed, n) in gate["lines"].items():
        low, high = gate_module.wilson(passed=passed, n=n)
        label = short[name.split(" ")[0]]
        rows.append({"line": label, "rate": passed / n, "low": low, "high": high, "n": n})
    rows.sort(key=lambda row: int(str(row["line"]).split(" ")[0]))
    data = pl.DataFrame(rows)
    base = alt.Chart(data)
    y = alt.Y("line:N", sort=data["line"].to_list(), title=None, axis=alt.Axis(labelLimit=330))
    threshold = (
        alt.Chart(pl.DataFrame({"x": [0.75]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    bars = base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Share of fleets passing", scale=alt.Scale(domain=[0, 1])),
        x2="high:Q",
        y=y,
    )
    dots = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x="rate:Q", y=y
    )
    return (threshold + bars + dots).properties(
        width=WIDTH,
        height=26 * len(data),
        title=_title(
            text="Known-answer gate: pass rate of each line",
            subtitle=(
                "Dot: share of the fleets that pass the line. Line: Wilson 90% interval. Dashed "
                "rule: the 75% a line needs to pass."
            ),
        ),
    )


def _headline_fit(*, label: str) -> dict[str, np.ndarray]:
    """Return the arrays of the lowest-objective stored fit of the `sh` arm at a cut."""
    fits = report.fit_files(label=label, arm="sh")
    return min(fits, key=lambda fit: fit[2]["objective"])[1]


def figure_shapes() -> alt.TopLevelMixin:
    """The library's shapes (scored-year headline fit) by day type and half-hour."""
    arrays = _headline_fit(label="scored")
    shapes, weights = arrays["shapes"], arrays["weights"]
    labels = [str(label) for label in arrays["labels"]]
    mass = weights.sum(axis=0)
    d18_share = (
        weights[labels.index("D18")] / mass if "D18" in labels else np.zeros(shapes.shape[0])
    )
    keep = [k for k in range(shapes.shape[0]) if d18_share[k] <= D18_SHARE_LIMIT]
    rows = [
        {
            "shape": f"Shape {k + 1}",
            "day type": name,
            "hour": half_hour / 2.0,
            "value": float(shapes[k, day_type * 48 + half_hour]),
        }
        for k in keep
        for day_type, name in enumerate(DAY_TYPE_NAMES)
        for half_hour in range(48)
    ]
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "hour:Q",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                scale=alt.Scale(domain=[0, 24]),
                title="Local half-hour of the day (hour)",
            ),
            y=alt.Y("value:Q", title="Shape value (unit norm)"),
            color=alt.Color(
                "shape:N",
                scale=alt.Scale(
                    domain=[f"Shape {k + 1}" for k in keep], range=[SHAPE_COLOURS[k] for k in keep]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH, height=110)
        .facet(row=alt.Row("day type:N", sort=DAY_TYPE_NAMES, title=None))
    )
    left_off = [k + 1 for k in range(shapes.shape[0]) if k not in keep]
    note = "" if not left_off else f" Shape {left_off} is left off (drawn on a separate figure)."
    return chart.properties(
        title=_title(
            text="The shared daily shapes, scored-year fit",
            subtitle=(
                "Each shape is centred within each hour band and scaled to unit length; its sign "
                "and scale are meaningless on their own. Weights and shapes are identified only up "
                "to the rotation the data fix." + note
            ),
        )
    )


def figure_mix() -> alt.TopLevelMixin:
    """Each decision substation's share of weight on each shape, with the range over the seeds."""
    table = report.mix_table().filter(pl.col("site").is_in([*SITE_ORDER, "D9"]))
    table = table.with_columns(shape=pl.format("Shape {}", pl.col("shape")))
    shapes = table["shape"].unique(maintain_order=True).to_list()
    colour = alt.Color(
        "shape:N",
        scale=alt.Scale(domain=shapes, range=SHAPE_COLOURS[: len(shapes)]),
        legend=alt.Legend(orient="top", title=None),
    )
    bars = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("share:Q", title="Share of the substation's weight", stack="zero"),
            y=alt.Y("site:N", sort=[*SITE_ORDER, "D9"], title=None),
            color=colour,
            tooltip=["site", "shape", "share", "low over seeds", "high over seeds"],
        )
    )
    return bars.properties(
        width=WIDTH,
        height=150,
        title=_title(
            text="Most substations sit on one or two shapes",
            subtitle=(
                "Share of each substation's total weight, scored-year headline fit. The ranges "
                "over the five seeds are in the report; the shares are identified only up to the "
                "rotation the data fix."
            ),
        ),
    )


def figure_stability(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Matched shape cosines across seeds and years, beside the null fleet's."""
    table = _section(sections=sections, start="Stability")
    gate = pl.DataFrame(
        [json.loads(path.read_text()) for path in (common.WORK_DIR / "gate").glob("null_*.json")]
    )
    null = float(np.median(gate["line5_median_seed_cosine"].to_numpy()))
    names = {
        "across seeds at tuning": "Seeds, tuning cut",
        "across seeds at replicate": "Seeds, replicate cut",
        "across seeds at scored": "Seeds, scored cut",
        "headline fits, tuning against replicate": "Years: tuning vs replicate",
        "headline fits, replicate against scored": "Years: replicate vs scored",
        "headline fits, tuning against scored (disjoint windows)": "Years: tuning vs scored",
    }
    data = pl.concat(
        [
            table.select(pl.col("comparison").replace(names), cosine=pl.col("median cosine")),
            pl.DataFrame({"comparison": ["Null fleet, two starts"], "cosine": [null]}),
        ]
    )
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "cosine:Q",
                title="Median matched cosine between shapes",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("comparison:N", sort=data["comparison"].to_list(), title=None),
        )
        .properties(
            width=WIDTH,
            height=24 * data.height,
            title=_title(
                text="How stable the shapes are across seeds and years",
                subtitle=(
                    "Shapes of two fits are matched one to one on the signed cosine; 1 means the "
                    "same shape. Years: the two fits share a year of data except tuning vs scored. "
                    "The bottom bar is the same measure on fleets of random profiles that share "
                    "nothing."
                ),
            ),
        )
    )


def figure_tuning() -> alt.TopLevelMixin:
    """The tuning year's score of every configuration (flat across the grid)."""
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    data = pl.DataFrame(tuned["configurations"]).with_columns(
        config=pl.format("K={}, λ={}, γ={}", "k", "lambda_dev", "gamma")
    )
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=40)
        .encode(
            x=alt.X(
                "score:Q",
                title="Tuning-year quantile score (points; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("k:N", title="Shapes K"),
            color=alt.Color(
                "lambda_dev:N",
                scale=alt.Scale(
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE]
                ),
                legend=alt.Legend(orient="top", title="Deviation weight (rows)"),
            ),
            tooltip=["config", "score"],
        )
        .properties(
            width=WIDTH,
            height=90,
            title=_title(
                text="The tuning year cannot tell the settings apart",
                subtitle=(
                    "Each dot is one (K, deviation weight, orthogonality weight) configuration, "
                    "the mean over three seeds and five substations, 2023-07 to 2024-06."
                ),
            ),
        )
    )


def figure_tails(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Coverage at the four tail levels for the headline arms, scored year."""
    table = _section(sections=sections, start="Tail calibration").filter(
        (pl.col("year") == "scored") & pl.col("arm").is_in(["g", "n", "sh"])
    )
    long = table.unpivot(
        on=["q01 coverage", "q05 coverage", "q95 coverage", "q99 coverage"],
        index=["arm", "season"],
        variable_name="level",
        value_name="coverage",
    ).with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS),
        target=pl.col("level").str.slice(1, 2).cast(pl.Float64) / 100.0,
    )
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(
            domain=[ARM_LABELS[a] for a in ("g", "n", "sh")],
            range=[ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_BLUE],
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    base = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X(
                "level:N",
                title="Quantile level",
                sort=["q01 coverage", "q05 coverage", "q95 coverage", "q99 coverage"],
            ),
            y=alt.Y("coverage:Q", title="Share of actuals below the quantile"),
            color=colour,
        )
        .properties(width=WIDTH, height=110)
    )
    return base.facet(row=alt.Row("season:N", title=None)).properties(
        title=_title(
            text="Tail calibration of the headline arms, scored year",
            subtitle=(
                "A calibrated quantile covers its own level (0.01, 0.05, 0.95, 0.99). The spread "
                "of every arm comes from one replica year."
            ),
        )
    )


def figure_step(*, sections: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The 01:00 step of the actual power and of SH's forecast on cold and mild nights."""
    table = _section(sections=sections, start="The 01:00 step on cold")
    data = table.filter(pl.col("year") == "scored").with_columns(
        series=pl.col("series").replace({"actual": "Actual", "mean": "SH mean forecast"})
    )
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=["Actual", "SH mean forecast"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("site:N", sort=SITE_ORDER, title=None),
            y=alt.Y("step at 01\\:00:Q", title=None),
            xOffset="series:N",
            color=colour,
        )
        .properties(width=WIDTH, height=110)
    )
    return chart.facet(row=alt.Row("nights:N", title=None)).properties(
        title=_title(
            text="Does the forecast reproduce the 01:00 step?",
            subtitle=(
                "Points of capacity: the half-hour ending 01:00 minus the one ending 00:30, local "
                "time, on cold nights (below 2 °C) and mild nights (above 10 °C), scored year. "
                "Fewer than 20 cold nights at four of the five substations."
            ),
        )
    )


def main() -> None:
    """Draw every figure from the report and the stored fits."""
    sections = read_sections()
    _save(figure_headline(sections=sections), name="figure_01_headline")
    _save(figure_winter(sections=sections), name="figure_02_winter")
    _save(figure_absolute(sections=sections), name="figure_03_absolute")
    _save(figure_step(sections=sections), name="figure_04_step")
    _save(figure_shapes(), name="figure_05_shapes")
    _save(figure_mix(), name="figure_06_mix")
    _save(figure_stability(sections=sections), name="figure_07_stability")
    _save(figure_gate(), name="figure_08_gate")
    _save(figure_tuning(), name="figure_09_tuning")
    _save(figure_tails(sections=sections), name="figure_10_tails")


if __name__ == "__main__":
    main()
