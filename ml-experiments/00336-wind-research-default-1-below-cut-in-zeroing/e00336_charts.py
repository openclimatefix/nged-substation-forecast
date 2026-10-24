"""Draw experiment 00336's figures as SVG. Reads `work/score.json`.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00336-*/e00336_charts.py   (PYTHONPATH as for the fit)
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00336_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00336-wind-research-default-1-below-cut-in-zeroing"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
LEVEL: Final[str] = "0.9967"
SCOPE: Final[str] = "3 wind farms, 3 seeds, scored at 1 to 10 days."
MAIN: Final[dict[str, str]] = {
    "x1": "x1: forecast set to zero below cut-in",
    "x2": "x2: training rows below cut-in dropped",
    "x3": "x3: training target zeroed below cut-in",
}
VARIANTS: Final[dict[str, str]] = {
    "x1": "x1 (mean speed, cut-in from the power curve)",
    "x1lo": "x1, cut-in minus 1 m/s",
    "x1hi": "x1, cut-in plus 1 m/s",
    "x1g3": "x1, cut-in 3.0 m/s",
    "x1a": "x1, all members below cut-in",
    "x1m": "x1m, each member zeroed on its own speed",
    "x1pl": "placebo for x1 (random rows)",
    "x2": "x2",
    "x2lo": "x2, cut-in minus 1 m/s",
    "x2hi": "x2, cut-in plus 1 m/s",
    "x2g3": "x2, cut-in 3.0 m/s",
    "x2e": "x2e, also drop the experiment 210 rows",
    "x2pl": "placebo for x2 (random rows)",
    "x3": "x3",
    "x3pl": "placebo for x3 (random rows)",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _panel(
    *, score: dict, period: str, subset: str, labels: dict[str, str], title: alt.TitleParams
) -> alt.TopLevelMixin:
    rows = []
    for key, label in labels.items():
        entry = score["contrasts"][period].get(key)
        if entry is None:
            continue
        found = entry[f"{subset} | qs"]
        low, high = found[LEVEL]
        placebo = key.endswith("pl")
        rows.append(
            {
                "arm": label,
                "estimate": found["mean"],
                "lower": low,
                "upper": high,
                "placebo": placebo,
            }
        )
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=list(data["arm"]), title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.condition("datum.placebo", alt.value(GREY), alt.value(ocf.DATA_BLUE)),
    )
    points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q", title="Score minus the default's (points of capacity; lower is better)"
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH, height=26 * data.height, title=title
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """The three pre-registered arms, kept rows and all half-hours, scored year and replicate."""
    panels = []
    for period, subset, text, note in (
        (
            "scored",
            "kept",
            "Dropping low-wind training rows (x2) gains on the kept rows",
            "Kept rows, scored year.",
        ),
        (
            "scored",
            "all",
            "On all half-hours the gain of x2 is smaller",
            "All half-hours, scored year.",
        ),
        (
            "replicate",
            "kept",
            "The replicate period points the same way for x2",
            "Kept rows, replicate period (January to June 2025).",
        ),
    ):
        panels.append(
            _panel(
                score=score,
                period=period,
                subset=subset,
                labels=MAIN,
                title=_title(
                    text=text,
                    lines=[
                        f"{note} Quantile score minus Wind farms research default 1's. Dot: "
                        "estimate. Line: 99.67% interval from re-drawing whole months and seeds. "
                        + SCOPE
                    ],
                ),
            )
        )
    return alt.vconcat(*panels).resolve_scale(x="shared")  # ty: ignore[invalid-argument-type]


def figure_2(*, score: dict) -> alt.TopLevelMixin:
    """Variants, cut-in sensitivity, and shuffled-mask placebos on the kept rows."""
    return _panel(
        score=score,
        period="scored",
        subset="kept",
        labels=VARIANTS,
        title=_title(
            text="A higher cut-in raises the gain of x2, and random rows gain nothing",
            lines=[
                "Kept rows, scored year, quantile score minus the default's. Grey: placebos "
                "that mask the same share of rows at random within farm and month. Line: 99.67% "
                "interval. " + SCOPE
            ],
        ),
    )


def figure_3(*, score: dict) -> alt.TopLevelMixin:
    """The default's score and the share of rows in each row class."""
    names = {
        "era5_masked": "ERA5 speed below cut-in",
        "mean_masked": "ensemble-mean speed below cut-in",
        "all_masked": "every member below cut-in",
        "idle210": "experiment 210 idle rows",
        "kept": "kept rows",
    }
    rows = [
        {
            "class": label,
            "share": 100 * score["row_classes"]["scored"]["share"][key],
            "default": score["scores"]["scored"]["default"][key]["qs"],
            "x1": score["scores"]["scored"]["x1"][key]["qs"],
            "x2": score["scores"]["scored"]["x2"][key]["qs"],
        }
        for key, label in names.items()
    ]
    data = pl.DataFrame(rows)
    order = list(names.values())
    share = (
        alt.Chart(data)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("class:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("share:Q", title="Share of scored half-hours (%)"),
        )
        .properties(
            width=WIDTH,
            height=120,
            title=_title(
                text="The speed masks cover few rows, and ERA5's lies inside experiment 210's",
                lines=["Share of the 107,121 scored half-hours in each row class. " + SCOPE],
            ),
        )
    )
    long = data.unpivot(
        index="class", on=["default", "x1", "x2"], variable_name="arm", value_name="score"
    )
    score_chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("class:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            yOffset="arm:N",
            x=alt.X("score:Q", title="Quantile score (points of capacity; lower is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["default", "x1", "x2"],
                    range=[GREY, ocf.DATA_GREEN, ocf.DATA_BLUE],
                ),
                title=None,
            ),
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                text="Zeroing the forecast (x1) and dropping rows (x2) make the masked rows worse",
                lines=["Quantile score of each arm within each row class, scored year."],
            ),
        )
    )
    return alt.vconcat(share, score_chart).resolve_scale(y="independent")


def main() -> None:
    """Write the three figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_1(score=score).save(DOCS_DIR / "figure_1.svg")
    figure_2(score=score).save(DOCS_DIR / "figure_2.svg")
    figure_3(score=score).save(DOCS_DIR / "figure_3.svg")


if __name__ == "__main__":
    main()
