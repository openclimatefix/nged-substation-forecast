"""Draw experiment 59's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00059-*/e00059_charts.py
"""

from __future__ import annotations

import itertools
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00059_common import WORK_DIR
from e00059_inject import NON_MATCHING_SHAPES
from e00059_report import STEP_EDGES, event_data, headline, share, share_interval

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00059-sw2-13-several-windows-in-one-detector"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
LABELS: Final[dict[str, str]] = {
    "W8": "4 hours",
    "W24": "12 hours",
    "W48": "24 hours",
    "W96": "48 hours",
    "MAX": "Maximum of the four",
    "SUM": "Sum of the four",
}
SINGLE_ORDER: Final[list[str]] = ["4 hours", "12 hours", "24 hours", "48 hours"]
ORDER: Final[list[str]] = [*SINGLE_ORDER, "Maximum of the four", "Sum of the four"]
REASONS: Final[dict[str, str]] = {
    "found": "Found",
    "end_outside_tolerance": "Onset found, end outside tolerance",
    "absorbed": "Alarm absorbed by a larger peak",
    "onset_outside_tolerance": "Onset outside tolerance",
    "no_alarm": "No alarm",
}
REASON_COLOURS: Final[list[str]] = [
    ocf.DATA_GREEN,
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE,
    ocf.DATA_PURPLE,
    ocf.DATA_SKY,
]
DETECTOR_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_share_found(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted share found by detector, with 95% intervals."""
    rows = headline(data=data, even=True, shapes=NON_MATCHING_SHAPES)
    table = []
    for name, label in LABELS.items():
        point, low, high = share_interval(rows=rows.filter(pl.col("detector") == name))
        table.append({"detector": label, "share": point, "low": low, "high": high})
    base = alt.Chart(pl.DataFrame(table)).encode(
        y=alt.Y("detector:N", sort=ORDER, title="Window of the detector")
    )
    bars = base.mark_bar(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "share:Q",
            title="Share of the moved MW found (larger is better)",
            scale=alt.Scale(domain=[0, 0.6]),
        )
    )
    bars_error = base.mark_rule(color="black").encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    return (bars + bars_error).properties(
        width=WIDTH,
        height=190,
        title=_title(
            text="Combining windows finds no more MW than the 24-hour window",
            lines=[
                (
                    "Bar: share of the MW moved by injected transfers that the detector finds. "
                    "Line: 95% interval from resampling whole calendar months. Transfers of 25% to "
                    "40% of the receiving load lasting over 1 day that start in even months; "
                    "constant-MW and receiver-profile shapes pooled; every detector raises the "
                    "same number of false alarms on odd months."
                )
            ],
        ),
    )


def figure_reasons(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Why large transfers with a metered receiver are or are not found, by detector."""
    rows = headline(data=data, even=True, shapes=NON_MATCHING_SHAPES).filter(
        pl.col("metered_receivers") > 0, pl.col("donor_step_z") >= 4.0
    )
    table = (
        rows.group_by("detector", "reason")
        .len()
        .with_columns(
            detector=pl.col("detector").replace_strict(LABELS),
            reason=pl.col("reason").replace_strict(REASONS),
            share=pl.col("len") / pl.col("len").sum().over("detector"),
        )
    )
    return (
        alt.Chart(table)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("detector:N", sort=ORDER, title="Window of the detector"),
            x=alt.X("share:Q", title="Share of transfers", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "reason:N",
                title=None,
                sort=list(REASONS.values()),
                scale=alt.Scale(domain=list(REASONS.values()), range=REASON_COLOURS),
                legend=alt.Legend(orient="top", columns=2, labelLimit=300),
            ),
            order=alt.Order("reason_order:Q"),
        )
        .transform_calculate(
            reason_order="indexof(" + str(list(REASONS.values())) + ", datum.reason)"
        )
        .properties(
            width=WIDTH,
            height=230,
            title=_title(
                text="Most missed large transfers have a found onset and a wrong end",
                lines=[
                    (
                        "Transfers with a metered receiver and a donor step of 4.0 noise units or "
                        "more of the 24-hour statistic, starting in even months; constant-MW and "
                        "receiver-profile shapes pooled."
                    )
                ],
            ),
        )
    )


def figure_by_step(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Share found by donor step for the 24-hour window and the two combined detectors."""
    rows = headline(data=data, even=True, shapes=NON_MATCHING_SHAPES).filter(
        pl.col("detector").is_in(["W48", "MAX", "SUM"])
    )
    labels = [
        f"{low:g}-{high:g}" if high < 1000 else f"{low:g}+"
        for low, high in itertools.pairwise(STEP_EDGES)
    ]
    table = []
    for name in ("W48", "MAX", "SUM"):
        for metered in (True, False):
            for (low, high), label in zip(itertools.pairwise(STEP_EDGES), labels, strict=True):
                part = rows.filter(
                    pl.col("detector") == name,
                    (pl.col("metered_receivers") > 0) == metered,
                    pl.col("donor_step_z").is_between(low, high, closed="left"),
                )
                if part.height:
                    table.append(
                        {
                            "detector": LABELS[name],
                            "group": "Metered receiver" if metered else "No metered receiver",
                            "step": label,
                            "share": share(rows=part, weighted=False),
                            "n": part.height,
                        }
                    )
    return (
        alt.Chart(pl.DataFrame(table))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "step:N", sort=labels, title="Donor step (noise units)", axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("share:Q", title="Share found", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "detector:N",
                title=None,
                sort=[LABELS["W48"], LABELS["MAX"], LABELS["SUM"]],
                scale=alt.Scale(
                    domain=[LABELS["W48"], LABELS["MAX"], LABELS["SUM"]], range=DETECTOR_COLOURS
                ),
                legend=alt.Legend(orient="top"),
            ),
            xOffset="detector:N",
            row=alt.Row("group:N", title=None, sort=["Metered receiver", "No metered receiver"]),
        )
        .properties(
            width=WIDTH,
            height=130,
            title=_title(
                text="Combining windows does not help at any donor step",
                lines=[
                    (
                        "Unweighted share of injected transfers found, by the MW the donor loses "
                        "over 24 hours in noise units of the 24-hour statistic. Even months, "
                        "constant-MW and receiver-profile shapes pooled."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    data = event_data(out=WORK_DIR)
    _save(figure_share_found(data=data), name="figure_1_share_found.svg")
    _save(figure_reasons(data=data), name="figure_2_miss_reasons.svg")
    _save(figure_by_step(data=data), name="figure_3_share_by_step.svg")


if __name__ == "__main__":
    main()
