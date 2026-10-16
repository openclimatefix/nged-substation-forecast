"""Draw the SW2-0b figures: real labelled events (raw data first) and the labelled set's counts.

Series are Primary substations (named). The SVGs are afterwards optimised with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00041-sw2-0a-labeller-gate/e00041_charts_real.py
"""

from __future__ import annotations

import json
import pickle
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00041_common import WORK_DIR
from e00041_panel import INDEX, excess, load_panel

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / "00041-sw2-0a-labeller-gate"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 100
REAL: Final[Path] = WORK_DIR / "real"
KIND_NAMES: Final[dict[str, str]] = {
    "real_operation": "Logged operations",
    "real_edge_close": "Start of closures over 60 days",
    "real_edge_open": "End of closures over 60 days",
    "real_short": "Closures under 2 hours",
    "alarm": "Detector alarms",
    "alarm_placebo": "Quiet windows (alarm placebos)",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_3() -> alt.TopLevelMixin:
    """Excess power around four labelled real operations whose labels are conserved."""
    panel = load_panel()
    labels = pl.read_parquet(REAL / "labelled_set.parquet")
    specs = pickle.loads((REAL / "features.pkl").read_bytes())["specs"]
    chosen = (
        labels.filter(
            pl.col("kind") == "real_operation",
            pl.col("visibility") == "conserved",
            pl.col("confidence") == "high",
        )
        .sort("moved_mw_p50", descending=True)
        .head(60)
        .sample(4, seed=3)
    )
    charts = []
    for row in chosen.iter_rows(named=True):
        spec = specs[row["card_id"]]
        sides = [s for s in (spec.side_a, spec.side_b) if s in INDEX]
        frames = []
        for sid in sides:
            lo, hi = spec.first - 96, spec.last + 97
            exc = excess(panel=panel, row=INDEX[sid], lo=lo, hi=hi)
            frames.append(
                pl.DataFrame(
                    {
                        "hours": (np.arange(lo, hi) - spec.first) / 2,
                        "excess": exc,
                        "series": f"Series {sid}",
                    }
                )
            )
        data = pl.concat(frames)
        shade = (
            alt.Chart(pl.DataFrame({"a": [0.0], "b": [(spec.last - spec.first + 1) / 2]}))
            .mark_rect(opacity=0.12, color="#888888")
            .encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
        )
        line = (
            alt.Chart(data)
            .mark_line(strokeWidth=1.2, clip=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("hours:Q", title="Hours from the logged close"),
                y=alt.Y("excess:Q", title="Excess (MW)"),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=[f"Series {s}" for s in sides],
                        range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE][: len(sides)],
                    ),
                ),
            )
        )
        charts.append(alt.layer(shade, line).properties(width=WIDTH - 130, height=120))
    return (
        alt.vconcat(*charts)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Real logged operations that the labellers call conserved transfers",
                lines=[
                    (
                        "Four of the highest-confidence conserved operations, drawn at random. "
                        "Shaded: the logged closure. Lines: half-hourly excess (observed power "
                        "minus the median of the same weekday and clock time over the 4 weeks "
                        "before and after) of the two substations on the sheet, in MW."
                    )
                ],
            )
        )
    )


def figure_4() -> alt.TopLevelMixin:
    """Final labels by card kind, as counts of cards."""
    meta = json.loads((REAL / "meta.json").read_text())
    final = json.loads((REAL / "labels_final.json").read_text())
    rows = []
    for card_id, info in meta.items():
        if info["kind"] in KIND_NAMES:
            rows.append({"kind": KIND_NAMES[info["kind"]], "label": final[card_id]["visibility"]})
    data = pl.DataFrame(rows).group_by("kind", "label").len()
    order = list(KIND_NAMES.values())
    label_names = {
        "conserved": "Conserved (two series step)",
        "one_sided": "One-sided (one series steps)",
        "not_visible": "Not visible",
        "unassessable": "Cannot be assessed",
    }
    data = data.with_columns(pl.col("label").replace(label_names))
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("kind:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("len:Q", title="Cards", stack="zero"),
            color=alt.Color(
                "label:N",
                title=None,
                sort=list(label_names.values()),
                scale=alt.Scale(
                    domain=list(label_names.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_BLUE_LIGHT, "#BBBBBB"],
                ),
                legend=alt.Legend(orient="top", columns=2, labelLimit=300, symbolLimit=0),
            ),
        )
        .properties(
            width=WIDTH - 280,
            height=200,
            title=_title(
                text="One in eight logged operations shows a visible transfer",
                lines=[
                    (
                        "Number of cards by what they describe and the final label (Sonnet, with "
                        "Opus settling disagreements). Hidden known-answer cards are not shown."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write both figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_3_real_examples", figure_3()),
        ("figure_4_labelled_counts", figure_4()),
    ):
        chart.save(OUT_DIR / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
