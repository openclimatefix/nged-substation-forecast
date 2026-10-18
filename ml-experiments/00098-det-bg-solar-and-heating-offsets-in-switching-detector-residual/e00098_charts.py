"""Draw experiment 98's figures as SVG from `results/compare.json` and `results/noise.parquet`.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00098-*/e00098_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00098_compare import RESULTS_PATH

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00098-det-bg-solar-and-heating-offsets-in-switching-detector-residual"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
ARM_LABELS: Final[dict[str, str]] = {
    "h05_stored": "Stored H0.5 (experiment 82)",
    "ctrl": "ctrl: H0.5 refitted here",
    "era5_solar": "ERA5 solar offset",
    "cams_solar": "CAMS solar offset",
    "heat": "CAMS solar + heating offset",
}
ARM_COLOURS: Final[list[str]] = [
    "#7A7F87",
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
]
BAND_ORDER: Final[list[str]] = [
    "night 00-06",
    "morning 06-10",
    "midday 10-14",
    "afternoon 14-18",
    "evening 18-24",
]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def _dot_and_line(
    *, rows: pl.DataFrame, y: str, order: list[str], colours: list[str] | None
) -> alt.Chart:
    base = alt.Chart(rows).encode(
        y=alt.Y(f"{y}:N", sort=order, title=None, axis=alt.Axis(labelLimit=320))
    )
    if colours is not None:
        base = base.encode(
            color=alt.Color(f"{y}:N", legend=None, scale=alt.Scale(domain=order, range=colours))
        )
    return base.mark_rule().encode(
        x=alt.X("low:Q", title="Share of moved MW found"), x2="high:Q"
    ) + (  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    )


def figure_recall(*, results: dict) -> alt.TopLevelMixin:
    """RO-NC and PF14 recall by arm, with intervals."""
    rows = []
    for detector, label in (
        ("XL", "RO-NC (the headline detector)"),
        ("PF14", "PF14 (older benchmark)"),
    ):
        for arm, name in ARM_LABELS.items():
            point, low, high = results["recall"][detector][arm]
            rows.append({"detector": label, "arm": name, "point": point, "low": low, "high": high})
    frame = pl.DataFrame(rows)
    chart = _dot_and_line(rows=frame, y="arm", order=list(ARM_LABELS.values()), colours=ARM_COLOURS)
    chart = chart.properties(width=WIDTH, height=130)
    return (
        chart.facet(
            row=alt.Row(
                "detector:N",
                title=None,
                sort=["RO-NC (the headline detector)", "PF14 (older benchmark)"],
                header=alt.Header(
                    labelOrient="top", labelAngle=0, labelLimit=520, labelFontSize=12
                ),
            )
        )
        .properties(
            title=_title(
                text="No offset changes the share of moved MW the detector finds",
                lines=[
                    (
                        "MW-weighted recall at 80 odd-month unexplained alarms, injection "
                        "shapes S2, S3, and S4 averaged, transfers starting in even "
                        "months, 16 Primary substations. Dot: estimate. Line: 98.75% "
                        "interval, resampling whole calendar months."
                    )
                ],
            )
        )
        .resolve_scale(y="independent")
    )


def figure_contrasts(*, results: dict) -> alt.TopLevelMixin:
    """The planned contrasts of RO-NC, with a zero rule and the +0.02 threshold."""
    names = {
        "C1 era5_solar - ctrl": "C1: ERA5 solar offset minus ctrl",
        "C2 cams_solar - ctrl": "C2: CAMS solar offset minus ctrl",
        "C3 heat - ctrl": "C3: solar + heating offset minus ctrl",
        "C4 heat - solar parent": "C4: solar + heating minus CAMS solar",
    }
    frame = pl.DataFrame(
        [
            {"contrast": names[key], "point": v[0], "low": v[1], "high": v[2]}
            for key, v in results["contrasts"].items()
            if key in names
        ]
    )
    marks = pl.DataFrame({"x": [0.0, 0.02], "what": ["no change", "threshold +0.02"]})
    rules = (
        alt.Chart(marks)
        .mark_rule(strokeDash=[4, 3])
        .encode(
            x="x:Q",
            color=alt.Color(
                "what:N", scale=alt.Scale(range=["#7A7F87", ocf.BRAND_ORANGE]), title=None
            ),
        )
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("contrast:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=320))
    )
    chart = (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Change in share of moved MW found (RO-NC)"), x2="high:Q"
        )
        + base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
        + rules
    ).properties(width=WIDTH, height=150)
    return chart.properties(
        title=_title(
            text="Every planned contrast straddles zero",
            lines=[
                (
                    "Paired difference in MW-weighted recall of RO-NC, same transfers "
                    "in every arm. Dot: estimate. Line: 98.75% interval from 20,000 "
                    "resamples of whole calendar months."
                )
            ],
        )
    )


def figure_noise() -> alt.TopLevelMixin:
    """Median absolute residual of the solar series in summer, by hour band and arm."""
    noise = pl.read_parquet(RESULTS_PATH.with_name("noise.parquet")).filter(
        pl.col("group") == "solar series", pl.col("season") == "summer"
    )
    noise = noise.with_columns(arm=pl.col("arm").replace(ARM_LABELS))
    order = [v for k, v in ARM_LABELS.items() if k != "h05_stored"]
    colours = [c for k, c in zip(ARM_LABELS, ARM_COLOURS, strict=True) if k != "h05_stored"]
    return (
        alt.Chart(noise)
        .mark_line(point=True)
        .encode(
            x=alt.X("band:N", sort=BAND_ORDER, title="UTC hour band", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("median_abs:Q", title="Median absolute residual (points of capacity)"),
            color=alt.Color(
                "arm:N", sort=order, scale=alt.Scale(domain=order, range=colours), title=None
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="The CAMS solar offset lowers the summer midday residual by about 12%, the ERA5 offset does not",
                lines=[
                    (
                        "Median absolute difference between measured and forecast demand, "
                        "June to August 2022 to 2026, the 5 Primary substations that have "
                        "a solar capacity. Smaller is better; 1 point is 1% of a "
                        "substation's effective capacity."
                    )
                ],
            ),
        )
    )


def figure_donors(*, results: dict) -> alt.TopLevelMixin:
    """RO-NC recall when the donor has a solar term and when it does not."""
    rows = []
    for group, label in (
        ("solar donors", "Donor is one of the 5 solar series"),
        ("other donors", "Donor is one of the 11 others"),
    ):
        for arm, name in ARM_LABELS.items():
            point, low, high = results["donor_split"][group][arm]
            rows.append({"group": label, "arm": name, "point": point, "low": low, "high": high})
    chart = _dot_and_line(
        rows=pl.DataFrame(rows), y="arm", order=list(ARM_LABELS.values()), colours=ARM_COLOURS
    ).properties(width=WIDTH, height=130)
    return (
        chart.facet(
            row=alt.Row(
                "group:N",
                title=None,
                sort=["Donor is one of the 5 solar series", "Donor is one of the 11 others"],
                header=alt.Header(
                    labelOrient="top", labelAngle=0, labelLimit=520, labelFontSize=12
                ),
            )
        )
        .properties(
            title=_title(
                text="Transfers from the solar series are the hard ones, and no arm helps them",
                lines=[
                    (
                        "Exploratory split of RO-NC's MW-weighted recall by the "
                        "substation that gives up load. Dot: estimate. Line: 98.75% "
                        "interval, resampling whole calendar months."
                    )
                ],
            )
        )
        .resolve_scale(y="independent")
    )


def main() -> None:
    """Write the figures."""
    results = json.loads(RESULTS_PATH.read_text())
    _save(figure_recall(results=results), name="figure_1_recall_by_arm.svg")
    _save(figure_contrasts(results=results), name="figure_2_contrasts.svg")
    _save(figure_noise(), name="figure_3_summer_residual.svg")
    _save(figure_donors(results=results), name="figure_4_recall_by_donor.svg")


if __name__ == "__main__":
    main()
