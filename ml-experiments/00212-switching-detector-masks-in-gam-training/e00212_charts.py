# ruff: noqa: E501
"""Draw experiment 212's figures as SVG from the stored report tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python e00212_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00212_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00212-switching-detector-masks-in-gam-training"
)
WIDTH: Final[int] = 520
LABELS: Final[dict[tuple[str, str], str]] = {
    ("CLEAN", "A - B"): "Clean rows: no mask minus hand labels",
    ("CLEAN", "C - B"): "Clean rows: detector minus hand labels",
    ("CLEAN", "A - C"): "Clean rows: no mask minus detector",
    ("POST-OLD", "A - B"): "After a transfer: no mask minus hand labels",
    ("POST-OLD", "A - C"): "After a transfer: no mask minus detector",
    ("POST-OLD", "C - B"): "After a transfer: detector minus hand labels",
    ("worlds", "A-B net of real"): "Injected worlds: no mask minus true mask",
    ("worlds", "A-C net of real"): "Injected worlds: no mask minus detector",
    ("worlds", "C-B net of real"): "Injected worlds: detector minus true mask",
}


def main() -> None:
    """Draw the headline figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in json.loads((WORK_DIR / "report.json").read_text()):
        label = LABELS.get((r["set"], r["contrast"]))
        if label is not None:
            rows.append({**r, "row": label, "group": label.split(":")[0]})
    data = pl.DataFrame(rows)
    title = alt.TitleParams(
        text=textwrap.wrap("No training mask costs at most 0.01 points of score, far below the 0.05 bar", 70),
        subtitle=textwrap.wrap(
            "Difference in quantile score (points of effective capacity; positive means the first arm is worse), with 95% intervals. Dashed lines: 0 and the 0.05 bar.",
            85,
        ),
        anchor="start", fontSize=14, subtitleFontSize=11,
    )
    chart = alt.layer(
        alt.Chart(data).mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Score difference (points)", scale=alt.Scale(domain=[-0.05, 0.07])),
            x2="high:Q",
            y=alt.Y("row:N", title=None, sort=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color("group:N", legend=None, scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN])),
        ),
        alt.Chart(data).mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x="mean:Q", y=alt.Y("row:N", sort=None), color=alt.Color("group:N", legend=None)
        ),
        alt.Chart(pl.DataFrame({"v": [0.0]})).mark_rule(strokeDash=[4, 3], color="#888").encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(pl.DataFrame({"v": [0.05]})).mark_rule(strokeDash=[2, 2], color="#444").encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=260, title=title)
    chart.save(OUT_DIR / "figure_1_contrasts.svg")


if __name__ == "__main__":
    main()
