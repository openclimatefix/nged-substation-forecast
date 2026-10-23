# ruff: noqa: E501
"""Draw the figures of experiment 309 as SVG beside the write-up.

uv run --no-sync python e00309_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00309_data as dat
import e00309_real_report as rr
import e00309_state as st
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00309-lsc-two-forecast-gap-choosing-without-truth"
)
W: Final[int] = CONTENT_WIDTH_PX - 160
COLOURS: Final[dict[str, str]] = {
    "No lags (GAM_ONLY)": ocf.DATA_BLUE,
    "With lags (K0)": ocf.BRAND_ORANGE,
    "Band head (BAND_HEAD)": ocf.DATA_SKY,
    "Rule on 24 h residual": ocf.DATA_GREEN,
    "Rule on gap": ocf.DATA_PURPLE,
    "Knows the true state": "#1B1B1B",
}


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    chart.configure_view(stroke=None).save(OUT / f"{name}.svg")


def dots(
    *,
    rows: list[dict],
    x_title: str,
    title: list[str],
    subtitle: list[str],
    order: list[str],
    name: str,
    height: int,
) -> None:
    """A dot plot: one row per row group, one coloured dot per arm."""
    t = pl.DataFrame(rows)
    chart = (
        alt.Chart(t)
        .mark_point(filled=True, size=110, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("score:Q", title=x_title),
            y=alt.Y("group:N", sort=order, title=None),
            yOffset=alt.YOffset("arm:N", scale=alt.Scale(domain=list(COLOURS))),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(title=None, orient="top", columns=3),
            ),
        )
        .properties(
            width=W, height=height, title=alt.TitleParams(title, subtitle=subtitle, anchor="start")
        )
    )
    save(chart, name)


def fig_injected() -> None:
    """Injected worlds: threshold-weighted CRPS against observed power, even months, by row group."""
    inj = {
        s: json.loads((dat.WORK_DIR / "results" / f"injected_{s}.json").read_text())
        for s in st.THRESHOLDS
    }
    best = {s: f"RULE_{s}_{inj[s]['selected_k']}" for s in inj}
    names = {
        "GAM_ONLY": "No lags (GAM_ONLY)",
        "K0": "With lags (K0)",
        "BAND_HEAD": "Band head (BAND_HEAD)",
        best["r24"]: "Rule on 24 h residual",
        best["d"]: "Rule on gap",
        "ORACLE_STATE": "Knows the true state",
    }
    groups = {
        "G1 ordinary": "Ordinary rows",
        "G2 before alarm": "Before the alarm",
        "G3 during": "During a shift",
        "G4 after": "48 hours after",
        "G5 near": "Within 7 days",
    }
    rows = []
    rows.extend(
        {"arm": names[r["arm"]], "group": groups[r["group"]], "score": r["tw95_obs"]}
        for r in inj["r24"]["absolute_even"]
        if r["arm"] in names and r["group"] in groups
    )
    rows.extend(
        {"arm": names[r["arm"]], "group": groups[r["group"]], "score": r["tw95_obs"]}
        for r in inj["d"]["absolute_even"]
        if r["arm"] == best["d"] and r["group"] in groups
    )
    dots(
        rows=rows, x_title="Threshold-weighted CRPS at the 95th percentile, observed power (smaller is better)",
        title=["In the injected worlds, the rules score like the forecast", "without lags, not like the forecast with lags"],
        subtitle=["Injected load shifts on six Primary substations, ensemble weather, even months only.", "Dot: mean over rows; shift rows weighted 1, ordinary rows 4."],
        order=list(groups.values()), name="figure_1_injected_worlds_by_row_group", height=360,
    )  # fmt: skip


def fig_real() -> None:
    """Real record: pinball loss at the 95th percentile against observed power, even months, by row group."""
    table = pl.read_parquet(dat.WORK_DIR / "results" / "real_units.parquet")
    real = json.loads((dat.WORK_DIR / "results" / "real.json").read_text())
    best = {s: f"RULE_{s}_{real[s]['best_k']}" for s in st.THRESHOLDS}
    names = {
        "GAM_ONLY": "No lags (GAM_ONLY)",
        "K0": "With lags (K0)",
        best["r24"]: "Rule on 24 h residual",
        best["d"]: "Rule on gap",
        "ORACLE_STATE": "Knows the true state",
    }
    groups = {"ordinary": "Ordinary rows", "during": "During a shift", "after": "72 hours after"}
    rows = [
        {"arm": names[a], "group": g_name, "score": rr.pooled(table=table, arm=a, group=g, odd=False, metric="pin95")}
        for a in names for g, g_name in groups.items()
    ]  # fmt: skip
    dots(
        rows=rows, x_title="Pinball loss at the 95th percentile, observed power (points of capacity; smaller is better)",
        title=["On real shifts, the forecast without lags is better only in the", "72 hours after a shift ends"],
        subtitle=["35 real load shifts over 1 day, 15 Primary substations, past weather, even months only.", "Dot: mean over rows. The no-lag forecast is the better fixed choice only after a shift."],
        order=list(groups.values()), name="figure_2_real_record_by_row_group", height=260,
    )  # fmt: skip


if __name__ == "__main__":
    fig_injected()
    fig_real()
