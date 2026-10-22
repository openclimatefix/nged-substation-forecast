"""Draw experiment 00259's figures from `work/score.json`, as SVG.

1. The planned contrasts C1 to C4 and the tail score T on the rows kept by experiment 210's mask.
2. Each arm's absolute quantile score by lead band.
3. The pinball loss at the 0.99 quantile by lead band, for R0, R1, and H5c.
4. R1's quantile score by quarter, on the hindcast and on the scored year.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00259-*/e00259_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00259_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00259-wind-plant-plus-quantile-offset-head"
)
PLOT_WIDTH: Final[int] = 380
WRAP: Final[int] = 80
GREY: Final[str] = "#999999"
SCOPE: Final[str] = "3 wind farms, 53 Tuesday runs, 5 seeds, July 2025 to June 2026."
DECISIVE: Final[tuple[str, ...]] = ("C1", "C2", "C3", "C4")
BANDS: Final[tuple[str, ...]] = ("days 1-2", "days 3-6", "days 7+", "all leads")
ARM_LABELS: Final[dict[str, str]] = {
    "R0": "R0 XGBoost",
    "R1": "R1 physics core",
    "H5c": "H5c physics core plus head",
    "CLIM": "Climatology",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "R0 XGBoost": ocf.DATA_BLUE,
    "R1 physics core": ocf.DATA_PURPLE,
    "H5c physics core plus head": ocf.BRAND_ORANGE,
    "Climatology": GREY,
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _contrast_rows(*, score: dict) -> list[dict]:
    """Return one chart row per planned contrast, plus the tail score of H5c against R0."""
    names = {
        "C1": "C1 head (H5c) minus physics core (R1)",
        "C2": "C2 head (H5c) minus XGBoost (R0)",
        "C3": "C3 head (H5c) minus the better non-neural arm",
        "C4": "C4 head (H5c) minus calibration-only head (K5)",
    }
    rows = []
    for label, text in names.items():
        if label not in score["contrasts"]:
            continue
        entry = score["contrasts"][label]["kept rows | all leads | qs"]
        rows.append(
            {
                "contrast": text,
                "difference": entry[0],
                "lower": entry[3],
                "upper": entry[4],
                "kind": "Planned (98.75%)",
            }
        )
    if "C2" in score["contrasts"]:
        entry = score["contrasts"]["C2"]["kept rows | all leads | tail_t"]
        rows.append(
            {
                "contrast": "T tail score, H5c minus R0",
                "difference": entry[0],
                "lower": entry[3],
                "upper": entry[4],
                "kind": "Planned (98.75%)",
            }
        )
    for label, text in (
        ("C5", "C5 daily-run head minus Tuesday-run head"),
        ("D1", "H5c minus blind head (BL5)"),
        ("D3", "H5c minus offset table (T5)"),
    ):
        if label in score["contrasts"]:
            entry = score["contrasts"][label]["kept rows | all leads | qs"]
            rows.append(
                {
                    "contrast": text,
                    "difference": entry[0],
                    "lower": entry[1],
                    "upper": entry[2],
                    "kind": "Descriptive (95%)",
                }
            )
    return rows


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Each contrast on the kept rows over all leads, with its interval."""
    rows = _contrast_rows(score=score)
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y(
            "contrast:N",
            sort=[r["contrast"] for r in rows],
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
            text="Planned contrasts for the neural head",
            lines=[
                "Quantile score of the first arm minus the second, on the rows experiment 210's "
                "mask keeps (smaller is better); the tail row is in the same units. Grey line: no "
                "difference. Dot: estimate. Line: interval from re-drawing whole months and seeds. "
                + SCOPE
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
            height=100,
            title=_title(
                text="Absolute quantile score by lead band",
                lines=["On the kept rows. Each panel has its own horizontal range. " + SCOPE],
            ),
        )
    )


def figure_3(*, score: dict) -> alt.TopLevelMixin:
    """Pinball loss at the 0.99 quantile of R0, R1, and H5c by lead band, kept rows."""
    rows = [
        {
            "arm": label,
            "band": band,
            "loss": score["scores"][arm][f"kept rows | {band}"]["pinball_q99"],
        }
        for arm, label in ARM_LABELS.items()
        if arm in score["scores"] and arm != "CLIM"
        for band in BANDS
    ]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "loss:Q",
                title="Pinball loss at the 0.99 quantile (points of capacity)",
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
            height=80,
            title=_title(
                text="Upper-tail pinball loss by lead band",
                lines=["On the kept rows. Each panel has its own horizontal range. " + SCOPE],
            ),
        )
    )


def figure_4(*, score: dict) -> alt.TopLevelMixin:
    """R1's quantile score by quarter of valid time, hindcast against the scored year."""
    rows = [
        {"quarter": quarter, "score": value, "period": "Hindcast (Apr 2024 to Jun 2025)"}
        for quarter, value in score["quarters"]["R1 on the hindcast"].items()
    ] + [
        {"quarter": quarter, "score": value, "period": "Scored year (Jul 2025 to Jun 2026)"}
        for quarter, value in score["quarters"]["R1"].items()
    ]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X("quarter:N", title=None, sort=sorted({r["quarter"] for r in rows})),
            y=alt.Y(
                "score:Q",
                title="R1 quantile score (points of capacity)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "period:N",
                scale=alt.Scale(
                    domain=[
                        "Hindcast (Apr 2024 to Jun 2025)",
                        "Scored year (Jul 2025 to Jun 2026)",
                    ],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=180,
            title=_title(
                text="The physics core's error by quarter",
                lines=[
                    (
                        "R1 is fitted on ERA5 before April 2024, so its error on later quarters "
                        "shows whether it drifts. Kept rows, all leads."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write the figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1(score=score)),
        ("figure_2", figure_2(score=score)),
        ("figure_3", figure_3(score=score)),
        ("figure_4", figure_4(score=score)),
    ):
        chart.save(DOCS_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
