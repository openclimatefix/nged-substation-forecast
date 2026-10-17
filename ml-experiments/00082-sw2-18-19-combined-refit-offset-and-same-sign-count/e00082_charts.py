"""Draw experiment 82's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00070-*/e00082_charts.py
"""

from __future__ import annotations

import itertools
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00082_common import WORK_DIR
from e00082_report import STEP_EDGES, contrast, pooled, populations

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00082-sw2-18-19-combined-refit-offset-and-same-sign-count"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
LABELS: Final[dict[str, str]] = {
    "PF14": "PF14: the single-series detector (benchmark)",
    "X": "X: refit-offset step statistic (experiment 71)",
    "LOGIT_SAME": "LOGIT_SAME: same-sign neighbour count (experiment 70)",
    "XL": "XL: both together",
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_PURPLE]
POPULATIONS: Final[dict[str, str]] = {
    "base": "Fresh headline-band transfers (25% or more of the receiving load, over 1 day)",
    "long": "Fresh long transfers (15.6 to 60 days)",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_recall(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall of each arm on the two fresh populations, with intervals."""
    rows = []
    for population, label in POPULATIONS.items():
        frames = pooled(data=data, population=population, parity="even")
        for arm, name in LABELS.items():
            point, low, high = contrast(frames=frames, arm=arm)
            rows.append(
                {"population": label, "arm": name, "point": point, "low": low, "high": high}
            )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=list(LABELS.values()), title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color(
            "arm:N", legend=None, scale=alt.Scale(domain=list(LABELS.values()), range=COLOURS)
        ),
    )
    chart = (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0, 1])),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=70).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=110)
    return (
        chart.facet(
            row=alt.Row(
                "population:N",
                title=None,
                sort=list(POPULATIONS.values()),
                header=alt.Header(
                    labelOrient="top", labelAngle=0, labelLimit=520, labelFontSize=12
                ),
            )
        )
        .properties(
            title=_title(
                text="Using both fixes finds 0.770 of the moved MW, against 0.656 for PF14",
                lines=[
                    (
                        "MW-weighted recall at 80 odd-month unexplained alarms, shapes S2, S3, and "
                        "S4 averaged, transfers starting in even months, first-return end rule. "
                        "Dot: estimate. Line: 98.33% interval, resampling whole calendar months."
                    )
                ],
            )
        )
        .resolve_scale(y="independent")
    )


def figure_by_step(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Share of headline-band transfers found, by donor step and metered receiver."""
    rows = []
    for flag, label in ((True, "Receiver metered"), (False, "No metered receiver")):
        for low, high in itertools.pairwise(STEP_EDGES):
            if low == 0.0:
                continue
            n = 0
            found = dict.fromkeys(LABELS, 0)
            for frame in pooled(data=data, population="base", parity="even"):
                part = frame.filter(
                    (pl.col("metered_receivers") > 0) == flag,
                    pl.col("donor_step_z").abs().is_between(low, high, "left"),
                )
                n += part.height
                for arm in LABELS:
                    found[arm] += int(part[arm].sum())
            if n == 0:
                continue
            step = f"{low:g} to {high:g}" if high != float("inf") else f"{low:g} or more"
            for arm, name in LABELS.items():
                rows.append(
                    {
                        "receiver": label,
                        "step": step,
                        "arm": name,
                        "share": found[arm] / n,
                        "n": n,
                    }
                )
    steps = [str(step) for step in dict.fromkeys(r["step"] for r in rows)]
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "step:N", sort=steps, title="Donor step (noise units)", axis=alt.Axis(labelAngle=0)
            ),
            xOffset=alt.XOffset("arm:N", sort=list(LABELS.values())),
            y=alt.Y("share:Q", title="Share of transfers found", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "arm:N",
                sort=list(LABELS.values()),
                legend=alt.Legend(title=None, orient="bottom", labelLimit=400),
                scale=alt.Scale(domain=list(LABELS.values()), range=COLOURS),
            ),
        )
        .properties(width=WIDTH, height=130)
    )
    return (
        chart.facet(
            row=alt.Row(
                "receiver:N",
                title=None,
                header=alt.Header(labelOrient="top", labelAngle=0, labelFontSize=12),
            )
        )
        .properties(
            title=_title(
                text="The gain is mostly in unmetered-receiver transfers with a step of 2 to 4",
                lines=[
                    (
                        "Share of fresh headline-band transfers found, by the size of the donor "
                        "substation's step, shapes S2, S3, and S4 summed, even months."
                    )
                ],
            )
        )
        .resolve_scale(x="independent")
    )


def main() -> None:
    """Write the figures."""
    data = populations(out=WORK_DIR)
    _save(figure_recall(data=data), name="figure_1_recall_by_arm.svg")
    _save(figure_by_step(data=data), name="figure_2_share_found_by_step.svg")


if __name__ == "__main__":
    main()
