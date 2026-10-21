"""Draw experiment 132's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are the two transformers of one Primary substation, not NGED generators.

    uv run --no-sync python ml-experiments/00132-*/e00132_charts.py
"""

from __future__ import annotations

import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00132_common import BLOCKED_SERIES, GRID_START, RESTORE_PAD, WORK_DIR, load_panel
from e00132_report import FEATURES, count_text, real_table, wilson
from e00132_restore import NGED_DIR
from e00132_run import all_scales
from e00132_variants import load_offset_panel

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00132-ro-nc-rescored-with-p12-and-p13-residuals-in-2025"
)
WIDTH: Final[int] = 600
WRAP: Final[int] = 95
OLD: Final[str] = "Experiment 101's panel (readings flagged, no residual before 2026)"
NEW: Final[str] = "Restored readings and blocked residuals"


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_found(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Share of transfers found by RO-NC with and without the restoration."""
    long = table.filter(pl.col("hours") > 24)
    groups = {
        "6 at P12 and P13": long.filter(pl.col("series") == "12,13"),
        "29 elsewhere": long.filter(pl.col("series") != "12,13"),
        "all 35": long,
    }
    rows = []
    for group, frame in groups.items():
        for label, column in ((OLD, "RO-NC|old panel"), (NEW, "RO-NC|half_life")):
            k, n = int(frame[column].sum()), frame.height
            low, high = wilson(k=k, n=n)
            rows.append(
                {
                    "group": group,
                    "panel": label,
                    "share": k / n,
                    "low": low,
                    "high": high,
                    "text": f"{k} of {n}",
                }
            )
    data = pl.DataFrame(rows)
    colour = alt.Color(
        "panel:N",
        scale=alt.Scale(domain=[OLD, NEW], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=450),
    )
    base = alt.Chart(data).encode(
        y=alt.Y("group:N", sort=list(groups), title=None),
        yOffset=alt.YOffset("panel:N", sort=[OLD, NEW]),
        color=colour,
    )
    bars = base.mark_bar().encode(
        x=alt.X(
            "share:Q",
            title="Share of the transfers found (95% Wilson interval; larger is better)",
            scale=alt.Scale(domain=[0, 1]),
        )
    )
    rules = base.mark_rule(color="black").encode(x="low:Q", x2="high:Q")
    labels = base.mark_text(align="left", dx=4, fontSize=11, color="black").encode(
        x="high:Q", text="text:N"
    )
    return (bars + rules + labels).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="RO-NC finds 4 of the 6 transfers at P12 and P13 with readings restored",
            lines=[
                (
                    "Real load transfers over 1 day, labelled in experiment 41. The refit-offset "
                    "neighbour-count detector (RO-NC) and every threshold are experiment 82's. "
                    "Bars: share found; line: 95% Wilson interval; label: found of scored."
                )
            ],
        ),
    )


def figure_readings(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Raw readings of P12 and P13 around each of the six transfers, restored half-hours marked."""
    raw = (
        pl.scan_delta(str(NGED_DIR / "power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(BLOCKED_SERIES))
        .collect()
    )
    restored = pl.read_parquet(WORK_DIR / "restored.parquet")
    events = table.filter(pl.col("series") == "12,13", pl.col("hours") > 24)
    pieces = []
    for row in events.iter_rows(named=True):
        start = GRID_START + timedelta(minutes=30 * row["onset"])
        end = GRID_START + timedelta(minutes=30 * row["end"])
        window = raw.filter(
            pl.col("time") >= start - RESTORE_PAD, pl.col("time") < end + RESTORE_PAD
        ).with_columns(
            event=pl.lit(row["event"]),
            hour=(pl.col("time") - start).dt.total_minutes() / 60,
            series=pl.format("P{}", pl.col("time_series_id")),
            end_hour=pl.lit((end - start).total_seconds() / 3600),
            restored=pl.col("time").is_in(restored["time"].to_list()) & (pl.col("power") == 0),
        )
        pieces.append(window)
    data = pl.concat(pieces).select("event", "hour", "series", "power", "end_hour", "restored")
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(domain=["P12", "P13"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(title=None, orient="top", direction="horizontal"),
    )
    line = (
        alt.Chart()
        .mark_line(strokeWidth=1.2, clip=True)
        .encode(
            x=alt.X("hour:Q", title="Hours since the labelled start of the transfer"),
            y=alt.Y("power:Q", title="Power (MW)"),
            color=colour,
        )
    )
    points = (
        alt.Chart()
        .mark_point(shape="cross", size=22, filled=False, strokeWidth=1.2)
        .encode(x="hour:Q", y="power:Q", color=colour)
    )
    panels = []
    for event in events["event"]:
        part = data.filter(pl.col("event") == event)
        end_hour = float(part["end_hour"][0])
        shade = (
            alt.Chart(pl.DataFrame({"start": [0.0], "end": [end_hour]}))
            .mark_rect(opacity=0.12, color="grey")
            .encode(x="start:Q", x2="end:Q")
        )
        panels.append(
            alt.layer(
                shade,
                line.properties(data=part),
                points.properties(data=part.filter(pl.col("restored"))),
            ).properties(width=WIDTH, height=90, title=alt.TitleParams(event, anchor="start"))
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="In 5 of the 6 transfers one transformer reads 0 MW, which cleaning flagged",
            lines=[
                (
                    "Raw half-hourly readings of the two transformers of one Primary substation, "
                    "2 days either side of each labelled transfer (shaded between its start and "
                    "end). Crosses: zero readings that `substation_zero` flagged and this "
                    "experiment re-read from the raw store. E59 has no raw readings at P12."
                )
            ],
        )
    )


def figure_noise() -> alt.TopLevelMixin:
    """The detector's noise scale of each series as a share of its capacity."""
    panel = load_panel()
    offset = load_offset_panel(panel=panel)
    scales = all_scales(panel=panel, offset_panel=offset)["PF14"]
    capacity = (
        pl.scan_parquet(FEATURES)
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").median())
        .collect()
    )
    rows = [
        {
            "series": f"P{series:02d}",
            "share": scale
            / capacity.filter(pl.col("time_series_id") == series)["effective_capacity_mw"].item(),
            "group": "P12 and P13 (blocked fits before 2026)"
            if series in BLOCKED_SERIES
            else "Other Primaries",
        }
        for series, scale in sorted(scales.items())
    ]
    data = pl.DataFrame(rows)
    median = float(np.median([r["share"] for r in rows if r["group"] == "Other Primaries"]))
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y("series:N", sort=None, title=None),
            x=alt.X(
                "share:Q",
                title="Noise unit as a share of the series' capacity (smaller is quieter)",
            ),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["Other Primaries", "P12 and P13 (blocked fits before 2026)"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=450),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [median]}))
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (bars + rule).properties(
        width=WIDTH,
        height=300,
        title=_title(
            text="The noise unit at P12 is about 5 times the other Primaries' share of capacity",
            lines=[
                (
                    "A noise unit is the usual size of the detector's 24-hour step on odd-numbered "
                    "months, from the new panel. Dashed line: median of the other 14 Primaries."
                )
            ],
        ),
    )


def main() -> None:
    """Draw all three figures."""
    table = real_table()
    print(count_text(k=int(table.filter(pl.col("hours") > 24)["RO-NC|half_life"].sum()), n=35))
    _save(figure_found(table=table), name="figure_1_found_by_panel.svg")
    _save(figure_readings(table=table), name="figure_2_readings_of_the_six_transfers.svg")
    _save(figure_noise(), name="figure_3_noise_by_series.svg")


if __name__ == "__main__":
    main()
