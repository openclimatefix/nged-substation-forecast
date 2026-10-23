"""Draw experiment 00230's figures from `work/score.json` and `work/step0.json`, as SVG.

1. The planned and exploratory contrasts, on the rows kept by experiment 210's mask.
2. Each arm's absolute quantile score by lead band, on the same rows.
3. Step 0: the ERA5-input ceiling against a binned curve, on three row sets.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00230-wind-physics-core-part-1/e00230_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00230_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00230-wind-physics-core-part-1"
)
PLOT_WIDTH: Final[int] = 380
WRAP: Final[int] = 80
GREY: Final[str] = "#999999"
SCOPE: Final[str] = "3 wind farms, 53 Tuesday runs, 3 seeds, July 2025 to June 2026."
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "Physics core (B1) minus best guess (A0)",
    "E1": "Network, no physics (B6) minus A0",
    "E2": "A0 plus the core's columns (B14) minus A0",
    "C2": "Core + output adapter (B9) minus B1",
    "C3": "Core on ERA5 (B1) minus on ENS control (B2)",
    "C4": "Core with idle layer (B1) minus without (B8)",
}
ARM_LABELS: Final[dict[str, str]] = {
    "A0": "A0 best guess",
    "B1": "B1 physics core",
    "B9": "B9 core + output adapter",
    "B5": "B5 core + XGBoost stage 2",
    "B6": "B6 network, no physics",
    "B14": "B14 A0 + core columns",
    "B2": "B2 core, ENS-control trained",
    "CLIM": "Climatology",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "A0 best guess": ocf.DATA_BLUE,
    "B1 physics core": ocf.BRAND_ORANGE,
    "B9 core + output adapter": ocf.DATA_PURPLE,
    "B5 core + XGBoost stage 2": ocf.DATA_SKY,
    "B6 network, no physics": ocf.DATA_GREEN,
    "B14 A0 + core columns": ocf.DATA_BLUE_LIGHT,
    "B2 core, ENS-control trained": ocf.BRAND_ORANGE_LIGHT,
    "Climatology": GREY,
}
BANDS: Final[tuple[str, ...]] = ("days 1-2", "days 3-6", "days 7+", "all leads")


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Each contrast, kept rows, at lead days 1 and 2 and over all leads, with 95% intervals."""
    rows = []
    for label, text in CONTRAST_LABELS.items():
        for band, band_label in (("days 1-2", "Lead days 1 and 2"), ("all leads", "All leads")):
            entry = score["contrasts"].get(label, {}).get(f"kept rows | {band} | qs")
            if entry:
                rows.append(
                    {
                        "contrast": text,
                        "scope": band_label,
                        "difference": entry[0],
                        "lower": entry[1],
                        "upper": entry[2],
                    }
                )
    data = pl.DataFrame(rows)
    scopes = ["Lead days 1 and 2", "All leads"]
    colour = alt.Color(
        "scope:N",
        scale=alt.Scale(domain=scopes, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            "contrast:N",
            sort=list(CONTRAST_LABELS.values()),
            title=None,
            axis=alt.Axis(labelLimit=300),
        ),
        yOffset=alt.YOffset("scope:N"),
        color=colour,
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
        height=260,
        title=_title(
            text="The physics core scores 0.1 to 0.3 worse than the XGBoost best guess",
            lines=[
                "Quantile score of the first arm minus the second, on the rows experiment 210's "
                "mask keeps (smaller is better). Grey line: no difference. Dot: estimate. Line: "
                "95% interval from re-drawing whole months and seeds. " + SCOPE
            ],
        ),
    )


def figure_2(*, score: dict) -> alt.TopLevelMixin:
    """Absolute quantile score of each arm, one panel per lead band, kept rows."""
    rows = [
        {"arm": label, "band": band, "score": score["scores"][arm][f"kept rows | {band}"]["qs"]}
        for arm, label in ARM_LABELS.items()
        if arm in score["scores"]
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
            height=110,
            title=_title(
                text="Trained arms beat climatology up to day 6 and lose to it from day 7",
                lines=[
                    "Absolute quantile score by lead band, on the rows experiment 210's mask "
                    "keeps. Each panel has its own horizontal range. " + SCOPE
                ],
            ),
        )
    )


def figure_3(*, step0: dict) -> alt.TopLevelMixin:
    """Step 0: ERA5 as the weather, three arms, three row sets."""
    names = {"P": "Physics core", "S": "Binned power curve", "X": "Flexible XGBoost"}
    rows = [
        {"arm": names[arm], "rows": scope, "score": value}
        for scope, scores in step0["scores"].items()
        for arm, value in scores.items()
    ]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Quantile score (points of capacity; smaller is better)"),
            y=alt.Y("arm:N", sort=list(names.values()), title=None, axis=None),
            row=alt.Row("rows:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(names.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE_LIGHT, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=50,
            title=_title(
                text="With ERA5 as the weather, XGBoost beats the physics core by 0.3",
                lines=[
                    "Step 0: models fitted on ERA5 before June 2025, scored with ERA5 as the "
                    "forecast. " + SCOPE
                ],
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    step0 = json.loads((common.WORK_DIR / "step0.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1(score=score)),
        ("figure_2", figure_2(score=score)),
        ("figure_3", figure_3(step0=step0)),
    ):
        chart.save(DOCS_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
