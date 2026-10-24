# ruff: noqa: E501
"""Draw experiment 327's figures from the stored result files.

Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00327-lsc-label-free-detect-mask-train-pipeline/e00327_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf  # noqa: F401  (registers and enables the theme)
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00327-lsc-label-free-detect-mask-train-pipeline"
)
ARM_LABELS: Final[dict[str, str]] = {
    "A": "A: unusualness, mask + retrain",
    "B": "B: 24 h residual, mask + retrain",
    "E": "E: arm A, two passes",
    "AF": "AF: arm A, filter mask only",
    "D": "D: logged shifts masked",
    "PERFECT": "D: true spans masked",
}
COLOURS: Final[dict[str, str]] = {
    "A": "#FF4901",
    "B": "#306BFF",
    "E": "#B701FF",
    "AF": "#10C5F7",
    "D": "#17E58F",
    "PERFECT": "#17E58F",
}
WIDTH: Final[int] = 560


def contrast_rows() -> pl.DataFrame:
    """Dots and intervals of arm minus no-mask arm, for the headline panels."""
    inj = json.loads((WORK / "results" / "inj_contrasts.json").read_text())["contrast"]
    real = json.loads((WORK / "results" / "real_contrasts_without_D9.json").read_text())["contrast"]
    rows = []
    panels = [
        (
            "Injected load shifts: rows during and after a shift (95% pinball loss)",
            inj,
            "DA_during_after|even",
            ("A", "B", "PERFECT"),
            "pin95",
            "-K0|",
        ),
        (
            "Real record: ordinary rows (mean quantile score)",
            real,
            "ordinary|winter=None",
            ("A", "B", "E", "AF", "D"),
            "qs",
            "-C|",
        ),
        (
            "Real record: 48 hours after a shift (95% pinball loss)",
            real,
            "after48|winter=None",
            ("A", "B", "E", "AF", "D"),
            "pin95",
            "-C|",
        ),
    ]
    for title, src, group, arms, metric, sep in panels:
        for arm in arms:
            v = src[f"{arm}{sep}{group}"][metric]
            rows.append(
                {
                    "panel": title,
                    "arm": ARM_LABELS[arm],
                    "key": arm,
                    "diff": v["diff"],
                    "lo": v["lo95"],
                    "hi": v["hi95"],
                }
            )
    return pl.DataFrame(rows)


def figure_headline() -> alt.TopLevelMixin:
    """Difference from no mask, with 95% intervals, in three stacked panels."""
    data = contrast_rows()
    charts = []
    for panel in dict.fromkeys(data["panel"].to_list()):
        part = data.filter(pl.col("panel") == panel)
        domain = part["arm"].to_list()
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=domain, range=[COLOURS[k] for k in part["key"].to_list()]),
            legend=None,
        )
        y = alt.Y("arm:N", title=None, sort=domain, axis=alt.Axis(labelLimit=260))
        base = alt.Chart(part)
        rule = base.mark_rule(color="#888888").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
        bars = base.mark_rule(size=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Arm minus no mask, points of capacity (below zero is better)"),
            x2="hi:Q",
            y=y,
            color=colour,
        )
        dots = base.mark_point(filled=True, size=90).encode(x="diff:Q", y=y, color=colour)  # ty: ignore[unresolved-attribute]
        charts.append(
            alt.layer(rule, bars, dots).properties(
                width=WIDTH - 250,
                height=30 * len(part) + 10,
                title=alt.TitleParams(panel, anchor="start", fontSize=12),
            )
        )
    return alt.vconcat(*charts).resolve_scale(x="independent", color="independent")


def figure_per_series() -> alt.TopLevelMixin:
    """Ordinary-row quantile score of arm C and arm A, one dot per Primary."""
    table = pl.read_parquet(WORK / "results" / "real_units.parquet")
    per = (
        table.filter((pl.col("group") == "ordinary") & pl.col("arm").is_in(["C", "A", "B"]))
        .group_by("series", "arm")
        .agg(pl.col("n").sum(), pl.col("qs").sum())
        .with_columns(
            score=pl.col("qs") / pl.col("n"),
            arm=pl.col("arm").replace(ARM_LABELS | {"C": "C: no mask"}),
        )
    )
    order = ["C: no mask", ARM_LABELS["A"], ARM_LABELS["B"]]
    return (
        alt.Chart(per.with_columns(series=pl.col("series").cast(pl.String)))
        .mark_point(filled=True, size=80)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("series:N", title="Primary substation (series id)"),
            y=alt.Y(
                "score:Q",
                title="Mean quantile score, ordinary rows (points of capacity; smaller is better)",
                scale=alt.Scale(type="symlog"),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=order, range=["#888888", "#FF4901", "#306BFF"]),
                legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=400),
            ),
        )
        .properties(width=WIDTH, height=240)
    )


def figure_short_history() -> alt.TopLevelMixin:
    """Monthly mean absolute error of the demand model at the short-history Primary, arm C and arm A."""
    series = 7
    power = np.load(WORK / "power" / f"power_{series}.npy")
    rows = []
    for arm in ("C", "A"):
        forecast = np.load(WORK / "forecast" / arm / f"forecast_{series}.npy")
        for m in range(len(power) // (48 * 30)):
            sl = slice(m * 48 * 30, (m + 1) * 48 * 30)
            if np.isfinite(forecast[sl]).sum() < 100:
                continue
            rows.append(
                {
                    "month": m + 1,
                    "arm": "C: no mask" if arm == "C" else ARM_LABELS["A"],
                    "mae": float(np.nanmean(np.abs(power[sl] - forecast[sl]))),
                }
            )
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "month:Q",
                title="Month of record (30-day blocks from January 2022)",
                axis=alt.Axis(values=[24, 30, 36, 42, 48, 54]),
            ),
            y=alt.Y(
                "mae:Q", title="Mean absolute error of the demand model (MW; smaller is better)"
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["C: no mask", ARM_LABELS["A"]], range=["#888888", "#FF4901"]
                ),
                legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=400),
            ),
        )
        .properties(width=WIDTH, height=220)
    )


def figure_detector() -> alt.TopLevelMixin:
    """Real load shifts found (of 35) by each detector, plain and with the neighbour filter."""
    d = json.loads((WORK / "results" / "detector_alone.json").read_text())
    names = {
        "A_plain": "A, no neighbour filter",
        "A": "A, neighbour filter",
        "B_plain": "B, no neighbour filter",
        "B": "B, neighbour filter",
        "E_plain": "E (pass 2), no neighbour filter",
        "E": "E (pass 2), neighbour filter",
    }
    rows = [
        {
            "arm": names[k],
            "found": d[k]["matched"],
            "lo": d[k]["matched_lo"],
            "hi": d[k]["matched_hi"],
            "false": d[k]["false_alarms"],
        }
        for k in names
    ]
    data = pl.DataFrame(rows)
    order = list(names.values())
    base = alt.Chart(data)
    bars = base.mark_rule(size=3, color="#306BFF").encode(
        x=alt.X(
            "lo:Q",
            title="Real load shifts found, of 35 (dot: count; line: 95% interval; number: false alarms)",
            scale=alt.Scale(domain=[0, 35]),
        ),
        x2="hi:Q",
        y=alt.Y("arm:N", sort=order, title=None),
    )  # ty: ignore[unresolved-attribute]
    dots = base.mark_point(filled=True, size=90, color="#306BFF").encode(
        x="found:Q", y=alt.Y("arm:N", sort=order)
    )  # ty: ignore[unresolved-attribute]
    text = base.mark_text(align="left", dx=8, color="#555555").encode(
        x="hi:Q", y=alt.Y("arm:N", sort=order), text=alt.Text("false:Q", format="d")
    )  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, dots, text).properties(width=WIDTH - 240, height=170)


def main() -> None:
    """Write the figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    figure_headline().save(OUT / "figure_1_difference_from_no_mask.svg")
    figure_per_series().save(OUT / "figure_2_ordinary_score_per_primary.svg")
    figure_short_history().save(OUT / "figure_3_short_history_primary.svg")
    figure_detector().save(OUT / "figure_4_detectors_alone.svg")


if __name__ == "__main__":
    main()
