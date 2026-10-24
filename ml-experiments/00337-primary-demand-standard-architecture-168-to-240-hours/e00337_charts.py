# ruff: noqa: E501
"""Draw the figures of experiment 337 from `work/report/report_scored.json`.

The drawing helpers are experiment 330's, pointed at this experiment's page folder.

    source /tmp/pp337.sh   # or the PYTHONPATH in e00337_common.py
    uv run --no-sync python ml-experiments/00337-*/e00337_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt

import e00311_charts as base  # isort: skip  # ty: ignore[unresolved-import]
import e00311_common as common  # isort: skip  # ty: ignore[unresolved-import]
import e00330_charts as c330  # isort: skip  # ty: ignore[unresolved-import]
import e00337_common as own  # isort: skip
import polars as pl

DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"{own.EXPERIMENT_ID:05d}-{own.SLUG}"  # fmt: skip
c330.DOCS_DIR = DOCS_DIR
c330.PNG_DIR = own.WORK_DIR / "png"
c330.LABELS.update({"climatology": "Climatology"})
c330.COLOURS.update({"climatology": "#B0B0B0"})
BANDS: Final[tuple[str, ...]] = tuple(f"lead {int(lo)}-{int(hi)} h" for lo, hi in own.LEAD_BANDS)
GROUPS: Final[tuple[str, ...]] = ("all rows", "all rows (without D10)", "ordinary rows", "cold winter nights", "sunny daylight", "Economy 7 night (00:30-07:30)")  # fmt: skip
ARMS: Final[tuple[str, ...]] = ("STD", "STDA")
SHOWN: Final[tuple[str, ...]] = ("default", "STD", "STDA", "manual_heuristic", "climatology")


def main() -> None:
    """Draw every figure."""
    rep = json.loads((own.REPORT_DIR / "report_scored.json").read_text())
    c = rep["contrasts"]
    labels = c330.LABELS
    long = pl.DataFrame([{"band": b.removeprefix("lead ").removesuffix(" h"), "arm": labels[a], "score": rep["by_mask"][a][b]["qs"]} for b in BANDS for a in SHOWN])  # fmt: skip
    order = [b.removeprefix("lead ").removesuffix(" h") for b in BANDS]
    colour = alt.Color("arm:N", scale=alt.Scale(domain=[labels[a] for a in SHOWN], range=[c330.COLOURS[a] for a in SHOWN]), legend=alt.Legend(orient="top", title=None, columns=2))  # fmt: skip
    lines = c330.chart(long).mark_line(point=True).encode(x=alt.X("band:N", sort=order, title="Lead time (hours)"), y=alt.Y("score:Q", title="Quantile score (points of capacity; lower is better)", scale=alt.Scale(zero=False)), color=colour)  # fmt: skip
    c330.save(lines.properties(width=c330.WIDTH - 60, height=300, title=base.title(text="The standard architecture scores below the default at every lead from 168 to 240 hours", lines=["Quantile score by 24-hour lead band, six Primaries, scored year 2025-07-02 to 2026-06-30. Lower is better."])), name="figure_1_score_by_lead")  # fmt: skip
    rows = [{"label": g, "arm": a, **{k: c[a]["groups"][g][k] for k in ("est", "lo", "hi")}} for g in GROUPS for a in ARMS]  # fmt: skip
    c330.ARMS = ARMS
    c330.forest(rows=rows, name="figure_2_gain_by_group", head="The gain is spread over groups, with wide intervals on the hard ones", lines=["Quantile score minus Primary demand research default 3, six Primaries, leads 168 to 240 hours.", "Dot: estimate. Line: 95% interval from resampling whole months. Left of zero is better."])  # fmt: skip
    rows = [{"label": f"all rows, leaving out {s}" if s != "none" else "all rows", "arm": a, **{k: (c[a]["gain_95"] if s == "none" else c[a]["leave_one_site_out"][s]["ci99"])[k] for k in ("est", "lo", "hi")}} for s in ("none", *common.SCORED_SITES) for a in ARMS]  # fmt: skip
    c330.forest(rows=rows, name="figure_3_leave_one_primary_out", head="Leaving out D10 halves the gain of the standard architecture", lines=["Quantile score minus the default with one Primary left out, scored year. Dot: estimate.", "Line: 99% interval (the top row, with nothing left out, shows 95%). Left of zero is better."])  # fmt: skip


if __name__ == "__main__":
    main()
