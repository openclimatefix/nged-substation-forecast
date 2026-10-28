# ruff: noqa: E501
"""Draw experiment 363's figures from the stored scores.

uv run --no-sync python ml-experiments/00363-*/e00363_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00363-d8-segment-level-term-in-the-gam-stack"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 480
LABELS: Final[dict[str, str]] = {
    "seg_weak": "Segment term, weak penalty",
    "seg_mid": "Segment term, middle penalty",
    "seg_strong": "Segment term, strong penalty",
    "m": "Masking (experiment 361)",
}


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def _title(text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 85), subtitle=textwrap.wrap(subtitle, 110),
        anchor="start", fontSize=14, subtitleFontSize=11,
    )  # fmt: skip


def figure_1(*, scores: dict) -> None:
    """Contrast against the control for each arm, with 99% intervals."""
    rows = []
    for arm in LABELS:
        for flag, label in (("all", "All scored rows"), ("y2026", "2026 only")):
            point, low, high = scores["contrast"][f"c_{arm}__ctl"][flag]["ci99"]
            rows.append({"arm": LABELS[arm], "rows": label, "point": point, "low": low, "high": high})
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", title=None, sort=list(LABELS.values())),
        yOffset="rows:N",
        color=alt.Color("rows:N", title=None, scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE])),
    )  # fmt: skip
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3], color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title="Quantile score minus the control's (points of capacity; negative is better)"), x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=45, aria=False).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=230,
        title=_title(
            "A level term for each stretch between detected steps raises D8's quantile score",
            "Dot: estimate. Line: 99% interval from resampling whole months. Dashed line: same as the control (experiment 354's fit (b), 8.382).",
        ),
    )
    _save(chart, name="figure_1_headline")


def figure_2(*, scores: dict) -> None:
    """Absolute score by lead band and by days with and without a detected step."""
    slices = [("lead_le12", "Up to 12 h"), ("lead_12_36", "12 to 36 h"), ("lead_36_72", "36 to 72 h"), ("lead_gt72", "Over 72 h"), ("step_day", "Days with a detected step"), ("no_step_day", "Days without")]
    arms = {"ctl": "Control", "m": "Masking", "seg_mid": "Segment term, middle penalty"}
    rows = [
        {"slice": label, "arm": arms[arm], "score": scores["absolute"][flag][arm]}
        for flag, label in slices
        for arm in arms
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar(aria=False)
        .encode(
            y=alt.Y("slice:N", title=None, sort=[label for _, label in slices]),
            yOffset=alt.YOffset("arm:N", sort=list(arms.values())),
            x=alt.X("score:Q", title="Quantile score (points of capacity; smaller is better)"),
            color=alt.Color("arm:N", title=None, sort=list(arms.values()), scale=alt.Scale(domain=list(arms.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN])),
        )
        .properties(
            width=WIDTH,
            height=300,
            title=_title(
                "The segment term is worse than the control in every lead band and on days with and without steps",
                "D8, 96,986 scored rows. Days with a step are days that hold a half-hour inside a segment found with hindsight by the same rules.",
            ),
        )
    )  # fmt: skip
    _save(chart, name="figure_2_by_lead_and_step_day")


def figure_3(*, placebo: dict) -> None:
    """Score change at the placebo Primaries and D8."""
    rows = [
        {"site": site, "point": v["contrast"]["ci99"][0] / v["ctl"] * 100, "low": v["contrast"]["ci99"][1] / v["ctl"] * 100, "high": v["contrast"]["ci99"][2] / v["ctl"] * 100}
        for site, v in placebo.items()
    ]  # fmt: skip
    base = alt.Chart(pl.DataFrame(rows)).encode(y=alt.Y("site:N", title=None))
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3], color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    limit = alt.Chart(pl.DataFrame({"x": [3.0]})).mark_rule(strokeDash=[2, 2], color=ocf.BRAND_ORANGE).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        zero + limit
        + base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(x=alt.X("low:Q", title="Change in quantile score (% of the control's; negative is better)"), x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=45, color=ocf.DATA_BLUE, aria=False).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=110,
        title=_title(
            "The segment term raises the score at both Primaries that have no gas engine",
            "Middle penalty against the control at D2 and D4. Dot: estimate. Line: 99% interval from resampling whole months. Orange dashed line: the pre-registered +3% limit.",
        ),
    )  # fmt: skip
    _save(chart, name="figure_3_placebos")


def figure_4(*, weather: dict) -> None:
    """Relative change of each weather and calendar term's coefficients when the segment term is added."""
    rows = [{"site": site, "term": term, "change": value} for site, terms in weather.items() for term, value in terms.items()]
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar(aria=False)
        .encode(
            y=alt.Y("term:N", title=None, sort=list(next(iter(weather.values())))),
            yOffset="site:N",
            x=alt.X("change:Q", title="Relative change in the term's coefficients (norm of the difference over norm of the control's)"),
            color=alt.Color("site:N", title=None, scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE])),
        )
        .properties(
            width=WIDTH,
            height=260,
            title=_title(
                "Adding the segment term changes the fitted weather and calendar terms most at D8",
                "Middle penalty against the control, at the last monthly refit. D2 has no gas engine and a quarter of D8's stretches.",
            ),
        )
    )  # fmt: skip
    _save(chart, name="figure_4_weather_terms")


def figure_5() -> None:
    """Two weeks of actual demand and the median forecasts of the control and the segment term."""
    wide = pl.read_parquet(WORK / "scores_wide_D8.parquet").filter(
        (pl.col("lead_hours") > 12) & (pl.col("lead_hours") <= 36)
    )
    days = wide.group_by("day").agg(pl.col("step_day").first()).sort("day")
    roll = days.with_columns(roll=pl.col("step_day").cast(pl.Int32).rolling_sum(14))
    end_row = int(roll["roll"].arg_max())
    start, end = days["day"][end_row - 13], days["day"][end_row]
    part = wide.filter((pl.col("day") >= start) & (pl.col("day") <= end)).group_by("valid_time").agg(
        pl.col("actual").first(), pl.col("ctl__q50").mean(), pl.col("seg_mid__q50").mean(), pl.col("m__q50").mean()
    ).sort("valid_time")  # fmt: skip
    t0 = part["valid_time"][0]
    long = part.with_columns(day=(pl.col("valid_time") - t0).dt.total_hours() / 24 + 1).unpivot(
        index="day", on=["actual", "ctl__q50", "seg_mid__q50"], variable_name="series", value_name="points", 
    ).with_columns(pl.col("series").replace({"actual": "Reading", "ctl__q50": "Control median", "seg_mid__q50": "Segment term median"}))  # fmt: skip
    chart = (
        alt.Chart(long)
        .mark_line(clip=True, aria=False)
        .encode(
            x=alt.X("day:Q", title="Day of the 14-day window", axis=alt.Axis(values=[1, 4, 7, 10, 14])),
            y=alt.Y("points:Q", title="Power (points of capacity; 1 point is about 0.1 MW)"),
            color=alt.Color("series:N", title=None, sort=["Reading", "Control median", "Segment term median"], scale=alt.Scale(domain=["Reading", "Control median", "Segment term median"], range=["#333333", ocf.DATA_BLUE, ocf.BRAND_ORANGE])),
            strokeWidth=alt.condition(alt.datum.series == "Reading", alt.value(1), alt.value(1.5)),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                "The segment term moves the forecast level only slightly in a fortnight with detected steps",
                "D8, the fortnight of the scored year with the most days that hold a detected step. Forecasts 12 to 36 hours ahead, averaged over runs.",
            ),
        )
    )  # fmt: skip
    _save(chart, name="figure_5_fortnight")


def main() -> None:
    """Draw every figure."""
    scores = json.loads((WORK / "scores.json").read_text())
    figure_1(scores=scores)
    figure_2(scores=scores)
    figure_3(placebo=scores["placebo"])
    figure_4(weather=json.loads((WORK / "weather_terms.json").read_text()))
    figure_5()


if __name__ == "__main__":
    main()
