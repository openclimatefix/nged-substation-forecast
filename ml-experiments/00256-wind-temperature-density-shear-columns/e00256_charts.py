"""Draw experiment 00256's figures from `work/score.json`, as SVG.

1. The contrasts on the rows kept by experiment 210's mask, over all leads, with intervals.
2. The three contrasts that carry the finding, split by night, day, winter, and summer.
3. Each arm's absolute quantile score by lead band.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00256-*/e00256_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00256_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00256-wind-temperature-density-shear-columns"
)
PLOT_WIDTH: Final[int] = 380
WRAP: Final[int] = 80
GREY: Final[str] = "#999999"
SCOPE: Final[str] = "3 wind farms, 53 Tuesday runs, 3 seeds, July 2025 to June 2026."
DECISIVE: Final[tuple[str, ...]] = ("C1", "C3", "C4")
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1 temperature (T1) minus six wind columns (T0)",
    "C4": "C4 placebo temperature (P1) minus T0",
    "C3": "C3 plus density, shear, elevation (T3) minus T1",
    "D6": "Density with measured pressure (T2p) minus T1",
    "D8": "Post hoc: plus shear (T4) minus T1",
    "D9": "Post hoc: plus solar elevation (T5) minus T1",
    "D10": "Post hoc: plus hour and day of year (T6) minus T1",
}
ARM_LABELS: Final[dict[str, str]] = {
    "T0": "T0 six wind columns",
    "T1": "T1 plus temperature",
    "T3": "T3 plus density, shear, elevation",
    "T6": "T6 plus hour and day of year",
    "CLIM": "Climatology",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "T0 six wind columns": ocf.DATA_BLUE,
    "T1 plus temperature": ocf.BRAND_ORANGE,
    "T3 plus density, shear, elevation": ocf.DATA_PURPLE,
    "T6 plus hour and day of year": ocf.DATA_GREEN,
    "Climatology": GREY,
}
BANDS: Final[tuple[str, ...]] = ("days 1-2", "days 3-6", "days 7+", "all leads")
SPLITS: Final[tuple[tuple[str, str], ...]] = (
    ("kept night", "Night"),
    ("kept day", "Day"),
    ("kept winter", "Winter"),
    ("kept summer", "Summer"),
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _interval(*, entry: list[float], label: str) -> tuple[float, float]:
    """Return the 98.75% interval for a decisive contrast and the 95% interval otherwise."""
    return (entry[3], entry[4]) if label in DECISIVE else (entry[1], entry[2])


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Each contrast on the kept rows over all leads, with its interval."""
    rows = []
    for label, text in CONTRAST_LABELS.items():
        entry = score["contrasts"][label]["kept rows | all leads | qs"]
        lower, upper = _interval(entry=entry, label=label)
        rows.append(
            {
                "contrast": text,
                "difference": entry[0],
                "lower": lower,
                "upper": upper,
                "kind": "Planned (98.75%)" if label in DECISIVE else "Descriptive (95%)",
            }
        )
    base = alt.Chart(pl.DataFrame(rows)).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "contrast:N",
            sort=list(CONTRAST_LABELS.values()),
            title=None,
            axis=alt.Axis(labelLimit=320),
        ),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["Planned (98.75%)", "Descriptive (95%)"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q", title="First arm minus second (points of capacity; negative is better)"
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=230,
        title=_title(
            text="Temperature helps slightly; hour and day of year help more than physics",
            lines=[
                "Quantile score of the first arm minus the second, on the rows experiment 210's "
                "mask keeps (smaller is better). Grey line: no difference. Dot: estimate. Line: "
                "interval from re-drawing whole months and seeds. " + SCOPE
            ],
        ),
    )


def figure_2(*, score: dict) -> alt.TopLevelMixin:
    """C1, C3, and D10 by night, day, winter, and summer, with 95% intervals."""
    rows = []
    for label in ("C1", "C3", "D10"):
        for key, split in SPLITS:
            entry = score["contrasts"][label][f"{key} | all leads | qs"]
            rows.append(
                {
                    "contrast": CONTRAST_LABELS[label],
                    "split": split,
                    "difference": entry[0],
                    "lower": entry[1],
                    "upper": entry[2],
                }
            )
    base = alt.Chart().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("split:N", sort=[s for _, s in SPLITS], title=None),
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(
                domain=[CONTRAST_LABELS[k] for k in ("C1", "C3", "D10")],
                range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_GREEN],
            ),
            legend=None,
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="First arm minus second (points of capacity)")
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart().mark_rule(color=GREY).encode(x=alt.datum(0.0))  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(zero, bars, points)
        .properties(width=PLOT_WIDTH, height=110)
        .facet(
            row=alt.Row(
                "contrast:N",
                sort=[CONTRAST_LABELS[k] for k in ("C1", "C3", "D10")],
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelLimit=380),
            ),
            data=pl.DataFrame(rows),
        )
        .properties(
            title=_title(
                text="The gain from the extra columns sits in daytime and summer",
                lines=[
                    "Night is solar elevation below zero. Intervals are 95%, resampling whole "
                    "months and seeds, over all leads on the kept rows. " + SCOPE
                ],
            )
        )
    )


def figure_3(*, score: dict) -> alt.TopLevelMixin:
    """Absolute quantile score of each arm, one panel per lead band, kept rows."""
    rows = [
        {"arm": label, "band": band, "score": score["scores"][arm][f"kept rows | {band}"]["qs"]}
        for arm, label in ARM_LABELS.items()
        for band in BANDS
    ]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "score:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("arm:N", sort=list(ARM_LABELS.values()), title=None),
            row=alt.Row(
                "band:N",
                sort=list(BANDS),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
        )
        .resolve_scale(x="independent")
        .properties(
            width=PLOT_WIDTH,
            height=100,
            title=_title(
                text="All four XGBoost arms beat climatology up to day 6",
                lines=[
                    "Absolute quantile score by lead band, on the kept rows. Each panel has its "
                    "own horizontal range. " + SCOPE
                ],
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1(score=score)),
        ("figure_2", figure_2(score=score)),
        ("figure_3", figure_3(score=score)),
    ):
        chart.save(DOCS_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
