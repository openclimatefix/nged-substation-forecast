"""Draw experiment 71's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00071-*/e00071_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00071_common import WORK_DIR
from e00071_report import BENCHMARK, contrast, pooled, populations

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00071-sw2-18-refit-offset-and-gap-bridging"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 95
LABELS: Final[dict[str, str]] = {
    "PF14_48": "PF14 on the stored residual (benchmark)",
    "X_48": "X: two models, refits half a month apart",
    "G_48": "G: gaps of a day or more bridged",
    "XG_48": "XG: both changes",
    "L_48": "L: suppression lifted (control)",
}
COLOURS: Final[list[str]] = ["#888888", ocf.BRAND_ORANGE, ocf.DATA_BLUE, "#7A3E9D", ocf.DATA_GREEN]


def _title(*, text: str, body: str) -> alt.TitleParams:
    wrapped = textwrap.wrap(body, WRAP)
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def _forest(
    *, rows: list[dict], order: list[str], domain: list[float], title: str, colours: list[str]
) -> alt.Chart:
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.Color("arm:N", legend=None, scale=alt.Scale(domain=order, range=colours)),
    )
    rules = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title=title, scale=alt.Scale(domain=domain)), x2="high:Q"
    )
    points = base.mark_point(filled=True, size=70).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    return rules + points


def figure_recall(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall of each arm on the two fresh populations, with intervals."""
    panels = []
    for population, label in (
        ("base", "Headline band: 615 transfers per shape"),
        ("long", "Long, 15.6 to 60 days: 220 per shape"),
    ):
        frames = pooled(data=data, population=population, parity="even")
        rows = []
        for arm, name in LABELS.items():
            point, low, high = contrast(frames=frames, arm=arm)
            rows.append({"arm": name, "point": point, "low": low, "high": high})
        panels.append(
            _forest(
                rows=rows,
                order=list(LABELS.values()),
                domain=[0, 1],
                title="Share of the moved MW found (larger is better)",
                colours=COLOURS,
            ).properties(width=WIDTH, height=120, title=label)
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Removing the refit blind spot finds about 5% more of the moved MW",
            body=(
                "MW-weighted recall at 80 odd-month unexplained alarms, injected transfers "
                "that start in even months, shapes S2, S3, and S4 pooled. Dot: estimate. "
                "Line: 98.33% interval from resampling whole calendar months."
            ),
        )
    )


def figure_contrasts(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """The three planned contrasts and the descriptive ones."""
    base = pooled(data=data, population="base", parity="even")
    long = pooled(data=data, population="long", parity="even")
    specs = [
        ("C1: X minus PF14", base, "X_48", BENCHMARK),
        ("C2: G minus PF14", base, "G_48", BENCHMARK),
        ("C3: XG minus PF14", base, "XG_48", BENCHMARK),
        ("Control L minus PF14", base, "L_48", BENCHMARK),
        ("X minus L (what the second model adds)", base, "X_48", "L_48"),
        ("Long transfers: X minus PF14", long, "X_48", BENCHMARK),
    ]
    rows = []
    for name, frames, arm, ref in specs:
        point, low, high = contrast(frames=frames, arm=arm, base=ref)
        rows.append({"arm": name, "point": point, "low": low, "high": high})
    order = [str(r["arm"]) for r in rows]
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#444444")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = _forest(
        rows=rows,
        order=order,
        domain=[-0.1, 0.2],
        title="Change in share of the moved MW found (larger is better; 0 = same as PF14)",
        colours=[ocf.BRAND_ORANGE] * len(order),
    )
    return (chart + rule).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="The gain is in the planned contrast; the second model adds nothing to it",
            body=(
                "Paired difference in MW-weighted recall, even months, shapes S2, S3, and S4 "
                "pooled, 98.33% intervals from resampling whole calendar months. Gap bridging "
                "(G) changes no outcome."
            ),
        ),
    )


def figure_zones(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """C1 split by whether a transfer starts or ends within 24 hours of a refit."""
    rows = []
    for population, zone, label in (
        ("base", True, "Headline band, in a zone"),
        ("base", False, "Headline band, outside the zones"),
        ("long", True, "Long transfers, in a zone"),
        ("long", False, "Long transfers, outside the zones"),
    ):
        frames = pooled(data=data, population=population, parity="even", zone=zone)
        point, low, high = contrast(frames=frames, arm="X_48", base=BENCHMARK)
        rows.append(
            {
                "arm": f"{label} ({frames[0].height} per shape)",
                "point": point,
                "low": low,
                "high": high,
            }
        )
    frames = pooled(data=data, population="zone", parity="even")
    point, low, high = contrast(frames=frames, arm="X_48", base=BENCHMARK)
    rows.append(
        {
            "arm": f"Onsets forced into a zone ({frames[0].height} per shape)",
            "point": point,
            "low": low,
            "high": high,
        }
    )
    order = [str(r["arm"]) for r in rows]
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#444444")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = _forest(
        rows=rows,
        order=order,
        domain=[-0.2, 1.0],
        title="X minus PF14, share of the moved MW found (0 = same as PF14)",
        colours=[ocf.BRAND_ORANGE] * len(order),
    )
    return (chart + rule).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="The whole gain sits in transfers that touch a refit",
            body=(
                "X minus PF14 in MW-weighted recall, even months, shapes S2, S3, and S4 "
                "pooled. A zone is the 24 hours either side of a refit of the model refitted "
                "on the 1st. Intervals are 98.33%, from resampling whole calendar months."
            ),
        ),
    )


def figure_refit_profile() -> alt.TopLevelMixin:
    """How far the two models' forecasts differ around each model's refit."""
    profile = pl.read_parquet(WORK_DIR / "refit_profile.parquet")
    summary = (
        profile.group_by("refit", "offset_half_hours")
        .agg(pl.col("mean_abs_difference_share").median())
        .with_columns(
            hours=pl.col("offset_half_hours") / 2,
            refit=pl.when(pl.col("refit") == "1st")
            .then(pl.lit("Refit on the 1st"))
            .otherwise(pl.lit("Refit on the 16th")),
        )
    )
    order = ["Refit on the 1st", "Refit on the 16th"]
    return (
        alt.Chart(summary)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hours:Q",
                title="Hours from the refit (negative = before)",
                axis=alt.Axis(values=[-72, -48, -24, 0, 24, 48, 72]),
            ),
            y=alt.Y("mean_abs_difference_share:Q", title="Difference as a share of median power"),
            color=alt.Color(
                "refit:N",
                title=None,
                scale=alt.Scale(domain=order, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="The two models differ by about 2% of power, with no jump at a refit",
                body=(
                    "Median over 16 Primary substations of the mean absolute difference between "
                    "the forecast of the model refitted on the 1st and of the model refitted on "
                    "the 16th, as a share of the substation's median absolute power."
                ),
            ),
        )
    )


def main() -> None:
    """Write the figures."""
    data = populations(out=WORK_DIR)
    _save(figure_recall(data=data), name="figure_1_recall_by_arm.svg")
    _save(figure_contrasts(data=data), name="figure_2_contrasts.svg")
    _save(figure_zones(data=data), name="figure_3_zone_split.svg")
    _save(figure_refit_profile(), name="figure_4_refit_profile.svg")


if __name__ == "__main__":
    main()
