# ruff: noqa: E501
"""Draw the figures of experiment 323 from `work/report/report.json` and the stored fits.

PYTHONPATH=<folders of experiments 184, 323> uv run --no-sync python ml-experiments/00323-*/e00323_charts.py
"""

import json
import pickle
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00323_basis as sun
import e00323_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 640
BANDS: Final[tuple[str, ...]] = tuple(f"{a}-{a + 24} h" for a in range(0, 240, 24))
CENTRES: Final[dict[str, int]] = {b: i * 24 + 12 for i, b in enumerate(BANDS)}
NAMES: Final[dict[str, str]] = {
    "default": "Default (Primary demand research default 3)",
    "f1": "Default plus F1 (one response per Primary)",
    "f2": "Default plus F2 (one shape, per-Primary scale)",
    "BG2": "BG2 (to 168 hours)",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
COLOURS: Final[dict[str, str]] = {
    NAMES["default"]: ocf.DATA_BLUE,
    NAMES["f1"]: ocf.DATA_GREEN,
    NAMES["f2"]: ocf.DATA_PURPLE,
    NAMES["BG2"]: ocf.DATA_SKY,
    NAMES["manual_heuristic"]: ocf.BRAND_ORANGE,
    NAMES["climatology"]: "#B0B0B0",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


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
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def headline(*, report: dict) -> None:
    """Quantile score against lead for every forecast, with 95% intervals."""
    keys = ("default", "f1", "f2", "BG2", "manual_heuristic", "climatology")
    data = pl.DataFrame(
        [
            {
                "arm": NAMES[k],
                "lead": CENTRES[b],
                **{f: report["by_band"][b][k][f] for f in ("est", "lo", "hi")},
            }
            for b in BANDS
            for k in keys
            if k in report["by_band"][b]
        ]
    )
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=[NAMES[k] for k in keys], range=[COLOURS[NAMES[k]] for k in keys]),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420, symbolType="stroke"),
    )
    yt = "Mean quantile score (points of capacity; smaller is better)"
    x = alt.X(
        "lead:Q",
        title="Lead time (hours; each point is a 24-hour band, at its centre)",
        scale=alt.Scale(domain=[0, 240]),
        axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
    )
    base = _chart(data)
    chart = (
        base.mark_errorbar(thickness=1.5).encode(
            x=x, y=alt.Y("lo:Q", title=yt, scale=alt.Scale(zero=False)), y2="hi:Q", color=colour
        )
        + base.mark_line(strokeWidth=2).encode(x=x, y=alt.Y("est:Q", title=yt), color=colour)
    ).properties(
        width=WIDTH,
        height=320,
        title=_title(
            title="The flexible sunshine response raises the score below 168 hours and changes it by under 0.01 points above",
            subtitle="Six Primaries, 2025-07-02 to 2026-06-30; 95% intervals resample the 12 initialisation months.",
        ),
    )
    _save(chart, name="headline_score_by_lead")


def gain_by_group(*, report: dict) -> None:
    """Difference from the default by group at 0.5 to 240 hours, with 95% intervals."""
    groups = [
        "all rows", "sunny daylight", "all daylight", "all night", "cold winter nights",
        "D10 and D16 (placebo)", "April to June daylight, D5 and D9",
    ]  # fmt: skip
    rows = []
    for g in groups:
        for arm in ("f1", "f2"):
            e = report["by_group"][g]["0-240 h"][f"{arm}_vs_default"]
            rows.append(
                {
                    "group": g,
                    "arm": NAMES[arm],
                    "diff": e["diff"],
                    "lo": e["diff_lo_95"],
                    "hi": e["diff_hi_95"],
                }
            )
    base = _chart(pl.DataFrame(rows))
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=[NAMES["f1"], NAMES["f2"]], range=[ocf.DATA_GREEN, ocf.DATA_PURPLE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y("group:N", sort=groups, title=None)
    chart = (
        base.mark_rule(strokeDash=[3, 3], color="#888").encode(x=alt.datum(0))
        + base.mark_errorbar(thickness=2).encode(
            y=y,
            x=alt.X(
                "lo:Q",
                title="Score minus the default's score, 0.5 to 240 hours (points; below zero is better)",
            ),
            x2="hi:Q",
            color=colour,
            yOffset="arm:N",
        )
        + base.mark_point(filled=True, size=50).encode(
            y=y, x="diff:Q", color=colour, yOffset="arm:N"
        )
    ).properties(
        width=WIDTH,
        height=280,
        title=_title(
            title="The response helps sunny daylight and costs about 0.03 points on other daylight",
            subtitle="Differences from Primary demand research default 3; April-to-June interval resamples ISO weeks.",
        ),
    )
    _save(chart, name="gain_by_group")


def response_surface() -> None:
    """The fitted F1 response against clear-sky index at four sun elevations, per Primary (cut 2026-06-01)."""
    fitted = pickle.loads((common.FITS_DIR / "2026-06-01.pkl").read_bytes())
    rows = []
    for site, entry in fitted["f1"].items():
        grid = entry["gamma"].reshape(sun.N_ELEVATION, sun.N_CSI)
        for k, elevation in enumerate(common.ELEVATION_KNOTS_DEG[1:]):
            if elevation not in (15.0, 30.0, 45.0, 65.0):
                continue
            for c, csi in enumerate(common.CSI_KNOTS):
                rows.append(
                    {
                        "Primary": site,
                        "elevation": f"{elevation:.0f} degrees",
                        "csi": float(csi),
                        "response": float(grid[k, c]),
                    }
                )
    chart = (
        _chart(pl.DataFrame(rows))
        .mark_line(point=True, strokeWidth=1.8)
        .encode(
            x=alt.X("csi:Q", title="Clear-sky index (irradiance over clear-sky irradiance)"),
            y=alt.Y("response:Q", title="Added response (points of capacity)"),
            color=alt.Color(
                "elevation:N",
                title="Sun elevation",
                sort=["15 degrees", "30 degrees", "45 degrees", "65 degrees"],
            ),
            facet=alt.Facet("Primary:N", columns=3),
        )
        .properties(
            width=190,
            height=140,
            title=_title(
                title="The fitted response raises net demand as sunshine rises, at every Primary, so the physics term removes too much solar output",
                subtitle="Fit at the 2026-06-01 cut; rows with the sun under 5 degrees carry little weight. Values at knots, not smoothed.",
            ),
        )
    )
    _save(chart, name="response_surface")


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.REPORT_DIR / "report.json").read_text())
    headline(report=report)
    gain_by_group(report=report)
    response_surface()


if __name__ == "__main__":
    main()
