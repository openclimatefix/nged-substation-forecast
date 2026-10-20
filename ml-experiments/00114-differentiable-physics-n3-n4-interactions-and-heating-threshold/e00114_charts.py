"""Draw the figures of experiment 114 as SVG into the docs folder.

Only Primary substations appear, under their D-labels, so dates may be shown. The numbers come from
`work/numbers.json` (written by `e00114_report.py`), so a figure and the page quote the same
numbers; the gate figures read the gates' stored draws. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00114-*/e00114_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00114_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "scratch_ref" / "png"
WIDTH: Final[int] = 480
"""Plot-area width: the axis labels bring a chart to the 680 px text column."""
WRAP: Final[int] = 90
GREY: Final[str] = "#808080"
DARK: Final[str] = "#333333"
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16"]
THRESHOLD_POINTS: Final[float] = 0.03


def _numbers() -> dict[str, list[dict[str, Any]]]:
    return json.loads((common.WORK_DIR / "numbers.json").read_text())


def _table(*, start: str) -> pl.DataFrame:
    numbers = _numbers()
    key = next(k for k in numbers if k.startswith(start))
    return pl.DataFrame(numbers[key], infer_schema_length=None)


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


def figure_1_headline() -> alt.TopLevelMixin:
    """The planned contrasts with their intervals, the replicate year, and unfiltered versions."""
    planned = _table(start="planned contrasts")
    beside = _table(start="beside the planned")
    exploratory = _table(start="exploratory contrasts")
    labels = {
        "C1 (N3): T-all-K - B0-K": "N3: all interactions (T-all) after the filter",
        "C1 beside: T-all - B0 (unfiltered)": "N3: all interactions, before the filter",
        "C1 control: T-all-K with B0's filter - B0-K": "N3: after the filter, B0's filter settings",
        "C2 (N4): H-site-K - B0-K": "N4: per-site threshold after the filter",
        "C2 beside: H-site - B0 (unfiltered)": "N4: per-site threshold, before the filter",
        "T1-K - B0-K (heating by day type)": "T1: heating by day type, after the filter",
        "T2-K - B0-K (heating by half-band)": "T2: heating by half-band, after the filter",
        "T3-K - B0-K (lighting by day type)": "T3: lighting by day type, after the filter",
    }
    planned_rows = pl.concat([planned, beside], how="diagonal_relaxed").with_columns(
        level=pl.lit("97.5% interval")
    )
    exploratory_rows = exploratory.filter(
        ~pl.col("contrast").str.contains("unfiltered")
    ).with_columns(level=pl.lit("95% interval (exploratory)"))
    rows = pl.concat([planned_rows, exploratory_rows], how="diagonal_relaxed").filter(
        pl.col("contrast").is_in(list(labels))
    )
    rows = rows.with_columns(label=pl.col("contrast").replace_strict(labels))
    order = [labels[k] for k in labels if k in rows["contrast"].to_list()]
    base = alt.Chart(rows)
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    rules = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "low:Q",
            title="Quantile score minus B0-K's (points of capacity; more negative is better)",
            scale=alt.Scale(domain=[-0.08, 0.05]),
        ),
        x2="high:Q",
        color=alt.Color(
            "level:N",
            scale=alt.Scale(
                domain=["97.5% interval", "95% interval (exploratory)"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    dots = base.mark_point(filled=True, size=70, color=DARK).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="scored QS:Q", tooltip=["contrast", "scored QS"]
    )
    replicate = base.mark_point(filled=False, size=70, color=DARK, shape="diamond").encode(  # ty: ignore[unresolved-attribute]
        y=y, x="replicate QS:Q"
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    threshold = (
        alt.Chart(pl.DataFrame({"x": [-THRESHOLD_POINTS]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (
        (rules + dots + replicate + zero + threshold)
        .properties(
            width=WIDTH,
            height=34 * len(order),
            title=_title(
                text="Interaction terms help after the filter; a per-site threshold is not clear",
                subtitle=(
                    "Scored year 2025-07 to 2026-06, day ahead, five decision substations, past "
                    "weather. Dot: estimate. Line: interval from resampling whole calendar months. "
                    "Diamond: replicate year 2024-25. Solid rule: no difference from B0-K. Dashed "
                    "rule: "
                    "the 0.03-point gain that counts."
                ),
            ),
        )
        .configure_axis(labelFontSize=11)
    )


def figure_2_absolute() -> alt.TopLevelMixin:
    """Each arm's mean quantile score per substation, with the references."""
    table = _table(start="absolute quantile scores")
    scored = table.filter(pl.col("year") == "scored")
    labels = {
        "b0_k": "B0-K (GAM-K's model + filter)",
        "t_all_k": "T-all-K",
        "h_site_k": "H-site-K",
        "p4:wk": "Part 4: window + drift + filter",
        "p4:x_best_k": "Part 4: lagless XGBoost + filter",
        "p4:manual_heuristic": "Manual heuristic",
        "p4:climatology": "Climatology",
    }
    long = (
        scored.filter(pl.col("arm").is_in(list(labels)))
        .unpivot(
            on=[*SITE_ORDER, "pooled (5 sites)"],
            index="arm",
            variable_name="site",
            value_name="score",
        )
        .with_columns(label=pl.col("arm").replace_strict(labels))
    )
    colours = [
        ocf.BRAND_ORANGE,
        ocf.DATA_BLUE,
        ocf.DATA_SKY,
        DARK,
        ocf.DATA_PURPLE,
        GREY,
        ocf.DATA_GREEN,
    ]
    return (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            y=alt.Y("site:N", sort=[*SITE_ORDER, "pooled (5 sites)"], title=None),
            x=alt.X(
                "score:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(values=[1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0]),
            ),
            color=alt.Color(
                "label:N",
                sort=list(labels.values()),
                scale=alt.Scale(domain=list(labels.values()), range=colours),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
            ),
            shape=alt.Shape("label:N", legend=None),
        )
        .properties(
            width=WIDTH,
            height=210,
            title=_title(
                text="Filtered convex models score 2.2 to 2.3; XGBoost 2.5; manual heuristic 3.5",
                subtitle=(
                    "Scored year, day ahead, rows outside the hand-filtered switching events. "
                    "Quantile score: twice the mean pinball loss over 11 quantiles."
                ),
            ),
        )
    )


def figure_3_splits() -> alt.TopLevelMixin:
    """Where the N3 gain sits: seasons and hour bands, for T-all-K and T2-K."""
    planned = _table(start="planned contrasts")
    exploratory = _table(start="exploratory contrasts")
    wanted = {
        "C1 (N3): T-all-K - B0-K": "T-all-K minus B0-K",
        "T2-K - B0-K (heating by half-band)": "T2-K minus B0-K",
        "T1-K - B0-K (heating by day type)": "T1-K minus B0-K",
        "T3-K - B0-K (lighting by day type)": "T3-K minus B0-K",
    }
    rows = pl.concat([planned, exploratory], how="diagonal_relaxed").filter(
        pl.col("contrast").is_in(list(wanted))
    )
    splits = {
        "Nov-Mar": "November to March",
        "Apr-Oct": "April to October",
        "band night": "Night (00 to 06)",
        "band morning": "Morning (06 to 11)",
        "band midday": "Midday (11 to 16)",
        "band evening": "Evening (16 to 24)",
    }
    long = rows.unpivot(
        on=list(splits), index="contrast", variable_name="split", value_name="difference"
    ).with_columns(
        split=pl.col("split").replace_strict(splits),
        arm=pl.col("contrast").replace_strict(wanted),
    )
    return (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("split:N", sort=list(splits.values()), title=None),
            yOffset=alt.YOffset("arm:N", sort=list(wanted.values())),
            x=alt.X(
                "difference:Q",
                title="Quantile score minus B0-K's (points; more negative is better)",
            ),
            color=alt.Color(
                "arm:N",
                sort=list(wanted.values()),
                scale=alt.Scale(
                    domain=list(wanted.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
        )
        .properties(
            width=WIDTH,
            height=240,
            title=_title(
                text="The interaction gain sits in winter and in the morning and evening bands",
                subtitle=(
                    "Scored year, after the filter, five decision substations. Each bar is the "
                    "point estimate for the rows in that season or hour band (local time)."
                ),
            ),
        )
    )


def figure_4_profiles() -> alt.TopLevelMixin:
    """The loss profile over heating thresholds at each substation, scored cut."""
    records = json.loads((common.WORK_DIR / "profiles.json").read_text())
    rows = []
    for record in records:
        if record["label"] != "scored":
            continue
        for threshold, nats in zip(common.HINGE_GRID_C, record["nats"], strict=True):
            rows.append({"site": record["site"], "threshold": threshold, "nats": nats})
    frame = pl.DataFrame(rows)
    order = [*SITE_ORDER, "D9"]
    lines = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "threshold:Q",
                title="Heating threshold (degrees C)",
                axis=alt.Axis(values=[12, 14, 16, 18]),
                scale=alt.Scale(domain=[11.5, 19.5]),
            ),
            y=alt.Y(
                "nats:Q",
                title="Nats above the best",
            ),
        )
    )
    base_rule = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x=alt.datum(common.BASE_THRESHOLD_C))
    )
    two = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY)
        .encode(y=alt.datum(2.0))
    )
    return (
        alt.layer(lines, base_rule, two, data=frame)
        .properties(width=130, height=120)
        .facet(
            facet=alt.Facet("site:N", sort=order, title=None),
            columns=3,
            title=_title(
                text="Four of five substations prefer a threshold below 15.5; two sit at 12 C",
                subtitle=(
                    "Scored cut. Loss in effective nats (loss difference over 2 times the noise "
                    "variance times 21.4, gate 3's median variance inflation); gap of 2 or less "
                    "is not distinguishable. Orange dashed rule: B0's 15.5 degrees; grey rule: "
                    "2 nats. D9 has 9 months of history at the replicate cut."
                ),
            ),
        )
    )


def figure_5_gate3() -> alt.TopLevelMixin:
    """Gate 3: the recovered against the true threshold on synthetic substations."""
    frame = pl.read_parquet(common.WORK_DIR / "gate3_draws.parquet").filter(
        pl.col("kind") == "main"
    )
    gate = json.loads((common.WORK_DIR / "gate3.json").read_text())
    coverage = 100.0 * gate["a_coverage"]
    width = gate["b_median_width_c"]
    scale = alt.Scale(domain=[11.5, 19.5])
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [12.0, 19.0], "y": [12.0, 19.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color=GREY, strokeDash=[4, 3])
        .encode(x=alt.X("x:Q", scale=scale), y=alt.Y("y:Q", scale=scale))
    )
    bars = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(opacity=0.6, clip=True)
        .encode(
            x=alt.X("true_threshold_c:Q", title="True threshold (degrees C)", scale=scale),
            y=alt.Y("low_c:Q", title="Recovered threshold (degrees C)", scale=scale),
            y2="high_c:Q",
            color=alt.Color(
                "covers:N",
                scale=alt.Scale(domain=[True, False], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title="Interval covers the truth", orient="top"),
            ),
        )
    )
    dots = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=25, color=DARK, clip=True)
        .encode(
            x="true_threshold_c:Q",
            y=alt.Y("argmin_c:Q", scale=scale),
        )
    )
    return (diagonal + bars + dots).properties(
        width=WIDTH,
        height=260,
        title=_title(
            text=(
                f"The 80% interval covers the true threshold in {coverage:.0f} of 100 synthetic "
                f"substations, with a median width of {width:.1f} degrees C"
            ),
            subtitle=(
                "100 synthetic substations at real noise. Dot: threshold with the lowest training "
                "loss. Line: 80% interval from 50 month-block resamples, widened by 0.25 C. "
                "Dashed diagonal: perfect recovery."
            ),
        ),
    )


def figure_6_gate2() -> alt.TopLevelMixin:
    """Gate 2: T-all minus B0 per synthetic draw, with and without a real interaction."""
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    weight = float(decisions["lambda"]["t_all"])
    frame = pl.read_parquet(common.WORK_DIR / "gate2_draws.parquet").filter(
        pl.col("penalty_weight") == weight
    )
    wins = int((frame["i_t_all"] < frame["i_b0"]).sum())
    long = pl.concat(
        [
            frame.select(
                difference=pl.col("i_t_all") - pl.col("i_b0"),
                truth=pl.lit("Real interaction (truth I)"),
            ),
            frame.select(
                difference=pl.col("zero_t_all") - pl.col("zero_b0"),
                truth=pl.lit("No interaction (truth 0)"),
            ),
        ]
    )
    bars = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(
                "difference:Q",
                bin=alt.Bin(step=0.01),
                title="T-all minus B0, held-out synthetic year (points; negative: T-all better)",
            ),
            y=alt.Y("count():Q", title="Synthetic draws"),
            color=alt.Color(
                "truth:N",
                scale=alt.Scale(
                    domain=["Real interaction (truth I)", "No interaction (truth 0)"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (bars + zero).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text=(
                f"T-all beats B0 in {wins} of 100 draws with a real interaction, and loses a "
                "little without one"
            ),
            subtitle=(
                "Unfiltered models, 100 synthetic substations per truth at real noise. The "
                "interaction is sized so that the true-parameter forecast gains 0.03 points."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    for draw, name in (
        (figure_1_headline, "figure_1_headline"),
        (figure_2_absolute, "figure_2_absolute"),
        (figure_3_splits, "figure_3_splits"),
        (figure_4_profiles, "figure_4_profiles"),
        (figure_5_gate3, "figure_5_gate3"),
        (figure_6_gate2, "figure_6_gate2"),
    ):
        _save(draw(), name=name)


if __name__ == "__main__":
    main()
