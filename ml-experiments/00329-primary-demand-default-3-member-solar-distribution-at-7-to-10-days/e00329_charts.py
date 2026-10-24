# ruff: noqa: E501
"""Draw the figures of experiment 329 from `work/report/stage0.json`.

PYTHONPATH=the folders of the earlier experiments uv run --no-sync python ml-experiments/00329-*/e00329_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00329_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
NAMES: Final[dict[str, str]] = {
    "default": "Primary demand research default 3",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#7A7A7A"]


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _colour() -> Any:
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(NAMES.values()), range=COLOURS),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
    )


def stages(*, report: dict) -> None:
    """Score on realised-sunny daylight by stage of the stack, at 168 to 240 hours."""
    rows = [{"stage": k, "score": v["score"]} for k, v in report["sunny_by_stage"].items()] + [
        {"stage": "Climatology", "score": report["realised-sunny daylight"]["climatology"]["score"]}
    ]
    names = {
        "chain (experiment 317)": "Convex chain (experiment 317)",
        "default 2 (experiment 319 x1)": "Primary demand research default 2",
        "default 3": "Primary demand research default 3",
        "manual_heuristic": "Manual heuristic",
    }
    data = pl.DataFrame(
        [{"stage": names.get(r["stage"], r["stage"]), "score": r["score"]} for r in rows]
    )
    order = [*list(names.values()), "Climatology"]
    base: Any = alt.Chart(data)
    chart = (
        base.mark_bar(color=ocf.DATA_BLUE).encode(
            y=alt.Y("stage:N", sort=order, title=None),
            x=alt.X(
                "score:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[0, 5.5]),
            ),
        )
        + base.mark_text(align="left", dx=3).encode(
            y=alt.Y("stage:N", sort=order), x="score:Q", text=alt.Text("score:Q", format=".2f")
        )
    ).properties(
        width=WIDTH,
        height=150,
        title=_title(
            title="Realised-sunny daylight at 7 to 10 days: the stack now beats the manual heuristic",
            subtitle="Sunniest third of each Primary's daylight half-hours by realised sunshine, leads 168 to 240 h, six Primaries, scored year 2025-07 to 2026-06.",
        ),
    )
    _save(chart, name="sunny_by_stage")


def by_site(*, report: dict) -> None:
    """Score on realised-sunny daylight by Primary."""
    rows = [
        {"site": s, "arm": NAMES[a], "score": v[a]["score"]}
        for s, v in report["realised-sunny daylight"]["by_site"].items()
        for a in NAMES
    ]
    chart: Any = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("site:N", title=None, sort=list(common.SITES)),
            x=alt.X(
                "score:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=_colour(),
        )
        .properties(
            width=WIDTH,
            height=190,
            title=_title(
                title="Only one Primary (D5, coastal) is still worse than the manual heuristic on sunny daylight",
                subtitle="Realised-sunny daylight half-hours, leads 168 to 240 h, scored year. Primaries are anonymised labels.",
            ),
        )
    )
    _save(chart, name="sunny_by_primary")


def pit(*, report: dict) -> None:
    """Probability-integral-transform histogram on realised-sunny daylight."""
    rows = [
        {"bin": i, "arm": NAMES[a], "share": 100 * x}
        for a, v in report["pit_sunny"].items()
        for i, x in enumerate(v)
    ]
    chart: Any = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "bin:O",
                title="Number of the 11 quantiles that lie below the realised demand (0 = realised demand below all)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title="Share of half-hours (%)"),
            color=_colour(),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                title="The heuristic's lower tail is too thin on sunny half-hours; the stack's is nearly right",
                subtitle="Realised-sunny daylight half-hours, leads 168 to 240 h, scored year. Bin 0 is realised demand under q01 (1% if calibrated).",
            ),
        )
    )
    _save(chart, name="pit_sunny")


if __name__ == "__main__":
    rep = json.loads((common.REPORT_DIR / "stage0.json").read_text())
    stages(report=rep)
    by_site(report=rep)
    pit(report=rep)
