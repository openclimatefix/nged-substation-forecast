"""Draw the figures of experiment 281's write-up from `work/report/report.json` and friends.

uv run --no-sync python ml-experiments/00281-*/e00281_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00281_common as common
import e00281_report as rep
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
MASTER_266: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00266-summer-gap-diagnostics-solar-term" / "work" / "master.parquet"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GREY: Final[str] = "#808080"
PERIOD_COLOURS: Final[dict[str, str]] = {
    "replicate year": ocf.DATA_BLUE,
    "scored year": ocf.BRAND_ORANGE,
}
CONTRAST_NAMES: Final[dict[str, str]] = {
    "C1": "C1 A1 minus A0 (night levels)",
    "C2": "C2 A2 minus A1 (temperature step)",
    "C3": "C3 A2 minus A0 (both)",
    "P2": "P2 E2 minus A0 (post hoc: no offset)",
}
ROW_SETS: Final[dict[str, str]] = {
    rep.STEP_KEY: "Step hours: half-hours starting 00:30 to 02:00 UTC, December to February",
    "winter nights": "Winter nights: November to March, 16:00 to 06:00 local time",
    "all": "All rows, all leads",
}
HALF_HOUR_COLOURS: Final[dict[str, str]] = {
    "k0": GREY,
    "a0": ocf.BRAND_ORANGE,
    "a1": ocf.DATA_BLUE,
    "a2": ocf.DATA_PURPLE,
}
HALF_HOUR_LABELS: Final[dict[str, str]] = {
    "k0": "K0 no night term",
    "a0": "A0 winter night offset",
    "a1": "A1 offset + night levels",
    "a2": "A2 offset + levels + step",
}
SITE_COLOURS: Final[tuple[str, ...]] = (
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
)


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


def figure_headline(*, data: dict) -> alt.TopLevelMixin:
    """The planned contrasts on three row sets, scored and replicate year, with 99% intervals."""
    panels = []
    for key, label in ROW_SETS.items():
        rows = []
        for period, period_label in (
            ("replicate", "replicate year"),
            ("scored", "scored year"),
        ):
            for contrast, name in CONTRAST_NAMES.items():
                entry = data[period]["contrasts"].get(contrast, {}).get(key)
                if entry is None:
                    continue
                q = entry["qs"]
                rows.append(
                    {
                        "contrast": name,
                        "period": period_label,
                        "estimate": q["estimate"],
                        "low": q["lower"],
                        "high": q["upper"],
                    }
                )
        base = alt.Chart(pl.DataFrame(rows)).encode(
            y=alt.Y(
                "contrast:N",
                sort=list(CONTRAST_NAMES.values()),
                title=None,
                axis=alt.Axis(labelLimit=300),
            ),
            yOffset=alt.YOffset("period:N", sort=list(PERIOD_COLOURS)),
            color=alt.Color(
                "period:N",
                scale=alt.Scale(domain=list(PERIOD_COLOURS), range=list(PERIOD_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top", direction="horizontal")
                if key == rep.STEP_KEY
                else None,
            ),
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=GREY, strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "estimate:Q", title="Difference in quantile score (points; below zero is better)"
            )
        )
        panels.append(
            (zero + rule + dot).properties(
                width=WIDTH - 280,
                height=130,
                title=alt.TitleParams(text=label, anchor="start", fontSize=12),
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text=(
                "Night levels fitted inside the model beat the offset in both years; "
                "the temperature step adds little"
            ),
            subtitle=(
                "Dot: paired difference in mean quantile score, five decision substations "
                "(D5, D10, D11, D14, D16), outside load shifts. Line: 99% interval from "
                "resampling whole calendar months. Zero: same score as the second arm."
            ),
        )
    )


def figure_amplitude(*, amplitude: dict) -> alt.TopLevelMixin:
    """The fitted step amplitude against the 24-hour mean temperature, per substation."""
    rows = [
        {"site": site, "temperature": float(t), "amplitude": value}
        for site, entry in amplitude.items()
        if site != "summary"
        for t, value in entry["amplitude_by_temperature_c"].items()
    ]
    sites = [s for s in amplitude if s != "summary"]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("temperature:Q", title="24-hour mean temperature (degrees C)"),
            y=alt.Y("amplitude:Q", title="Fitted step amplitude (points of rated demand)"),
            color=alt.Color(
                "site:N",
                sort=sites,
                scale=alt.Scale(domain=sites, range=list(SITE_COLOURS)),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
        .properties(
            width=WIDTH - 100,
            height=240,
            title=_title(
                text="The fitted night step grows as the day gets colder, at all five substations",
                subtitle=(
                    "A2's step amplitude from the fit at 2026-06-01: the sum of four non-negative "
                    "hinges (knots 3, 7, 11, and 15.5 C), valid 00:30 to 08:00 UTC."
                ),
            ),
        )
    )


def figure_half_hour(*, data: dict) -> alt.TopLevelMixin:
    """The mean score by UTC half-hour on winter nights, scored year."""
    rows = [
        {
            "arm": HALF_HOUR_LABELS[row["arm"]],
            "hour": row["minute"] / 60.0 if row["minute"] >= 21 * 60 else row["minute"] / 60.0 + 24,
            "score": row["qs"],
            "key": row["arm"],
        }
        for row in data["scored"]["winter_night_by_half_hour"]
        if row["arm"] in HALF_HOUR_LABELS
    ]
    labels = list(HALF_HOUR_LABELS.values())
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "hour:Q",
                title="UTC hour at which the half-hour ends (24 is midnight; 33 is 09:00)",
                axis=alt.Axis(values=[21, 24, 27, 30, 33]),
            ),
            y=alt.Y("score:Q", title="Mean quantile score (points)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "arm:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=list(HALF_HOUR_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
        )
        .properties(
            width=WIDTH - 100,
            height=240,
            title=_title(
                text=(
                    "The night corrections remove the error peak after 01:00 UTC; "
                    "the three corrected arms nearly coincide"
                ),
                subtitle=(
                    "Scored year, November to March, five decision substations, all leads, "
                    "half-hours ending 21:00 to 09:00 UTC."
                ),
            ),
        )
    )


def figure_d10_night() -> alt.TopLevelMixin:
    """Experiment 266's night residuals at D10 in April to June, by UTC half-hour."""
    master = pl.read_parquet(MASTER_266).filter(
        (pl.col("site") == "D10")
        & pl.col("valid_time").dt.month().is_in([4, 5, 6])
        & ~pl.col("daylight")
    )
    start = pl.col("valid_time") - pl.duration(minutes=30)
    table = (
        master.with_columns(hour=(start.dt.hour() + start.dt.minute() / 60.0))
        .filter((pl.col("hour") >= 21) | (pl.col("hour") < 9))
        .with_columns(hour=pl.when(pl.col("hour") < 12).then(pl.col("hour") + 24).otherwise("hour"))
        .group_by("hour")
        .agg(
            (pl.col("actual") - pl.col("med_arm1")).mean().alias("Best-guess stack"),
            (pl.col("actual") - pl.col("med_bg2")).mean().alias("BG2"),
        )
        .unpivot(index="hour", variable_name="forecast", value_name="residual")
    )
    labels = ["Best-guess stack", "BG2"]
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    lines = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "hour:Q",
                title="UTC hour at which the half-hour starts (24 is midnight; 33 is 09:00)",
                axis=alt.Axis(values=[21, 24, 27, 30, 33]),
                scale=alt.Scale(domain=[21, 33]),
            ),
            y=alt.Y(
                "residual:Q", title="Mean actual minus forecast (points; negative is too high)"
            ),
            color=alt.Color(
                "forecast:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=[ocf.BRAND_ORANGE, ocf.DATA_GREEN]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
    )
    return (zero + lines).properties(
        width=WIDTH - 100,
        height=240,
        title=_title(
            text="At D10 the April to June night error is a level error, not a step at 00:30 UTC",
            subtitle=(
                "Scored year, April to June, night half-hours, all leads, D10 (an anonymised "
                "primary substation). The stack forecast runs about 2 points too high until "
                "07:30 UTC and 4 to 6 points too high after."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    data = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    amplitude = json.loads((common.WORK_DIR / "amplitude.json").read_text())
    _save(figure_headline(data=data), name="figure_1_headline")
    _save(figure_amplitude(amplitude=amplitude), name="figure_2_step_amplitude")
    _save(figure_half_hour(data=data), name="figure_3_half_hour")
    _save(figure_d10_night(), name="figure_4_d10_night")


if __name__ == "__main__":
    main()
