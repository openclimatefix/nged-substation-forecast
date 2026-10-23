"""Draw experiment 00262's figures from `work/score.json` and `work/arms/weights.json`.

1. Planned and descriptive contrasts of the quantile score, with intervals.
2. Quantile score of each arm by lead band.
3. Weights of the ECMWF members, the AIFS members, and climatology by lead day (arm M4).

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00262-*/e00262_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00262_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00262-wind-aifs-members-and-climatology-by-lead"
)
PLOT_WIDTH: Final[int] = 380
WRAP: Final[int] = 80
SCOPE: Final[str] = (
    "3 wind farms (W1 to W3), Tuesday runs from October 2025 to June 2026, rows kept by the "
    "idle-row mask, offset table applied to every arm."
)
BANDS: Final[tuple[str, ...]] = ("days 0-2", "days 3-6", "days 7+")
ARM_LABELS: Final[dict[str, str]] = {
    "M0+T": "M0: ECMWF members",
    "M2+T": "M2: AIFS members",
    "M3+T": "M3: ECMWF + AIFS",
    "M4+T": "M4: ECMWF + AIFS + climatology",
    "M5+T": "M5: gate",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "M0+T": ocf.DATA_BLUE,
    "M2+T": ocf.DATA_PURPLE,
    "M3+T": ocf.BRAND_ORANGE,
    "M4+T": ocf.DATA_GREEN,
    "M5+T": ocf.DATA_SKY,
}
SOURCE_LABELS: Final[tuple[str, ...]] = ("ECMWF members", "AIFS members", "Climatology")
SOURCE_COLOURS: Final[tuple[str, ...]] = (ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN)
CONTRASTS: Final[tuple[tuple[str, str, str, str, bool], ...]] = (
    ("C1", "days 7+", "C1: climatology in, days 7+", "decisive", True),
    ("C2", "days 3-6", "C2: AIFS in, days 3-6", "decisive", True),
    ("C3", "all leads", "C3: gate over weights by lead", "decisive", True),
    ("D4", "days 1-2", "AIFS alone, days 1-2", "descriptive", False),
    ("D4", "days 3-6", "AIFS alone, days 3-6", "descriptive", False),
    ("D4", "days 7+", "AIFS alone, days 7+", "descriptive", False),
    ("D5", "all leads", "Offset table: with minus without", "descriptive", False),
    ("D6", "all leads", "Offset table: scored months minus hindcast", "descriptive", False),
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1(*, report: dict) -> alt.TopLevelMixin:
    """Dots and intervals of the contrasts on the quantile score."""
    rows = []
    for label, band, name, kind, decisive in CONTRASTS:
        values = report["contrasts"][label][f"kept rows | {band} | qs"]
        rows.append(
            {
                "contrast": name,
                "kind": kind,
                "difference": values[0],
                "low": values[3] if decisive else values[1],
                "high": values[4] if decisive else values[2],
            }
        )
    frame = pl.DataFrame(rows)
    order = [name for _, _, name, _, _ in CONTRASTS]
    base = alt.Chart(frame).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["decisive", "descriptive"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    ranges = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Quantile score difference (points of capacity; below 0 is better)"),
        x2="high:Q",
    )
    dots = base.mark_point(filled=True, size=70).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#666666", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(ranges, dots, zero).properties(
        width=PLOT_WIDTH,
        height=190,
        title=_title(
            text="AIFS lowers the score at days 3 to 6; climatology and the gate do not",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 98.33% interval for decisive contrasts, 95% for the "
                    "rest, from resampling whole months and seeds. Dashed line: no difference."
                ),
            ],
        ),
    )


def figure_2(*, report: dict) -> alt.TopLevelMixin:
    """Dots of each arm's quantile score by lead band."""
    rows = [
        {
            "arm": ARM_LABELS[arm],
            "band": band,
            "score": report["scores"][arm][f"kept rows | {band}"]["qs"],
        }
        for arm in ARM_LABELS
        for band in BANDS
    ]
    order = list(ARM_LABELS.values())
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "score:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[6, 14.5]),
            ),
            y=alt.Y("band:N", sort=list(BANDS), title=None),
            yOffset=alt.YOffset("arm:N", sort=order),
            color=alt.Color(
                "arm:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[ARM_COLOURS[a] for a in ARM_LABELS]),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=260),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=170,
            title=_title(
                text="The quantile score of every arm rises with lead",
                lines=[
                    SCOPE,
                    "Lead bands are calendar days from the run's date to the valid time.",
                ],
            ),
        )
    )


def figure_3(*, weights: dict) -> alt.TopLevelMixin:
    """Lines of the mean weight of each source by lead day, arm M4."""
    rows = []
    for day in range(16):
        values = np.mean(
            [
                weights[f"seed{seed}_M4"][f"{month} day {day}"]
                for seed in (0, 1, 2)
                for month in ("2025-10", "2026-01", "2026-04", "2026-06")
            ],
            axis=0,
        )
        rows.extend(
            {"lead_day": day, "source": label, "weight": float(w)}
            for label, w in zip(SOURCE_LABELS, values, strict=True)
        )
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lead_day:Q",
                title="Lead (calendar days from the run's date to the valid time)",
                scale=alt.Scale(domain=[0, 15]),
            ),
            y=alt.Y("weight:Q", title="Mean weight in the pool", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "source:N",
                sort=list(SOURCE_LABELS),
                scale=alt.Scale(domain=list(SOURCE_LABELS), range=list(SOURCE_COLOURS)),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=170,
            title=_title(
                text="The fitted weights move about from one lead day to the next",
                lines=[
                    (
                        "Arm M4, mean of the weights refitted in October 2025, January 2026, April "
                        "2026, and June 2026, over three seeds. Each lead day has about 7,000 rows "
                        "from 53 runs, so neighbouring lead days differ by more than a smooth "
                        "weighting would."
                    ),
                ],
            ),
        )
    )


def main() -> None:
    """Write the figures as SVG."""
    report = json.loads((common.WORK_DIR / "score.json").read_text())
    weights = json.loads((common.WORK_DIR / "arms" / "weights.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {1: figure_1(report=report), 2: figure_2(report=report), 3: figure_3(weights=weights)}
    for number, chart in figures.items():
        chart.save(str(DOCS_DIR / f"figure_{number}.svg"))
        print(f"figure_{number}.svg")


if __name__ == "__main__":
    main()
