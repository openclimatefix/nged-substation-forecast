# ruff: noqa: E501
"""Draw the figures of experiment 325 from `work/report/report.json` and the stored fits.

source /tmp/env325.sh && uv run --no-sync python ml-experiments/00325-*/e00325_charts.py
"""

import json
import pickle
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00323_basis as sun  # ty: ignore[unresolved-import]
import e00323_common as common323  # ty: ignore[unresolved-import]
import e00325_common as common
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
    "default": "Default (rebuilt, mean of 6 network seed sets)",
    "f1_as_run": "F1 as run in experiment 323",
    "f1r": "F1r (unrestricted response, stack refitted)",
    "f1s": "F1s (restricted response, refitted; post hoc)",
    "f1sel": "F1sel (F1s switched per Primary; post hoc)",
    "BG2": "BG2 (to 168 hours)",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
COLOURS: Final[dict[str, str]] = {
    NAMES["default"]: ocf.DATA_BLUE,
    NAMES["f1_as_run"]: "#B0B0B0",
    NAMES["f1r"]: ocf.DATA_GREEN,
    NAMES["f1s"]: ocf.DATA_PURPLE,
    NAMES["f1sel"]: ocf.BRAND_ORANGE,
    NAMES["BG2"]: ocf.DATA_SKY,
    NAMES["manual_heuristic"]: "#555555",
    NAMES["climatology"]: "#DDDDDD",
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
    keys = ("default", "f1r", "f1s", "f1sel", "BG2", "manual_heuristic", "climatology")
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
            title="Refitting the stack around the sunshine response changes the score by under 0.02 points at every lead",
            subtitle="Six Primaries, 2025-07-02 to 2026-06-30; 95% intervals resample the 12 initialisation months. F1s and F1sel are post hoc.",
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
        for arm in ("f1r", "f1s", "f1sel"):
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
        scale=alt.Scale(
            domain=[NAMES[k] for k in ("f1r", "f1s", "f1sel")],
            range=[ocf.DATA_GREEN, ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
        ),
        legend=alt.Legend(title=None, orient="top", labelLimit=420, columns=1),
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
            title="Where the refitted response helps and where it still costs score, by group",
            subtitle="Differences from Primary demand research default 3; April-to-June interval resamples ISO weeks.",
        ),
    )
    _save(chart, name="gain_by_group")


def response_surface() -> None:
    """The fitted F1 and F1s response against clear-sky index at four sun elevations (cut 2026-06-01)."""
    rows = []
    for site in common.SITES:
        stored = pickle.loads((common.FITS_DIR / site / "2026-06-01.pkl").read_bytes())
        for variant in common.RESPONSE_VARIANTS:
            grid = stored[variant]["gamma"].reshape(sun.N_ELEVATION, sun.N_CSI)
            for k, elevation in enumerate(common323.ELEVATION_KNOTS_DEG[1:]):
                if elevation not in (15.0, 30.0, 45.0, 65.0):
                    continue
                for c, csi in enumerate(common323.CSI_KNOTS):
                    rows.append(
                        {
                            "Primary": site,
                            "variant": "F1 (unrestricted)"
                            if variant == "f1"
                            else "F1s (restricted)",
                            "elevation": f"{elevation:.0f} degrees",
                            "csi": float(csi),
                            "response": float(grid[k, c]),
                        }
                    )
    chart = (
        _chart(pl.DataFrame(rows).filter(pl.col("elevation").is_in(["30 degrees", "45 degrees"])))
        .mark_line(point=True, strokeWidth=1.8)
        .encode(
            x=alt.X("csi:Q", title="Clear-sky index (irradiance over clear-sky irradiance)"),
            y=alt.Y("response:Q", title="Added response (points of capacity)"),
            color=alt.Color(
                "variant:N", title=None, scale=alt.Scale(range=[ocf.DATA_GREEN, ocf.DATA_PURPLE])
            ),
            strokeDash=alt.StrokeDash("elevation:N", title="Sun elevation"),
            facet=alt.Facet("Primary:N", columns=3),
        )
        .properties(
            width=190,
            height=140,
            title=_title(
                title="Restricting the response removes its rise at low clear-sky index; above it the two fits agree",
                subtitle="Fit at the 2026-06-01 cut, response at the spline knots, sun elevation 30 and 45 degrees.",
            ),
        )
    )
    _save(chart, name="response_surface")


def stage_zero() -> None:
    """Experiment 323's F1 minus default score difference by ensemble clear-sky index and elevation."""
    report = json.loads((common.REPORT_DIR / "stage0.json").read_text())
    rows = []
    for key, label in (
        ("by_csi_all_daylight", "All daylight"),
        ("by_csi_sunny_daylight", "Sunny daylight"),
        ("by_csi_placebo_D10_D16", "D10 and D16 daylight (placebo)"),
    ):
        for band, entry in report[key].items():
            rows.append(
                {
                    "group": label,
                    "band": band,
                    "mean_difference": entry["mean_difference"],
                    "n": entry["n"],
                }
            )
    data = pl.DataFrame(rows).filter(pl.col("n") >= 100)
    order = ["0-0.3", "0.3-0.6", "0.6-0.9", ">0.9"]
    chart = (
        _chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "band:N", sort=order, title="Clear-sky index of the ensemble-mean shortwave flux"
            ),
            y=alt.Y(
                "mean_difference:Q", title="F1 minus default score (points; above zero is worse)"
            ),
            color=alt.Color(
                "group:N",
                title=None,
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.BRAND_ORANGE]),
            ),
            xOffset="group:N",
        )
        .properties(
            width=WIDTH,
            height=240,
            title=_title(
                title="Experiment 323's F1 harmed daylight half-hours mostly at clear-sky index under 0.3 and over 0.9, not at 0.3 to 0.9",
                subtitle="Daylight half-hours at 0.5 to 168 hours, six Primaries, 2025-07-02 to 2026-06-30; the flux is the raw ensemble mean. Bands under 100 rows are omitted.",
            ),
        )
    )
    _save(chart, name="stage0_by_clear_sky_index")


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.REPORT_DIR / "report.json").read_text())
    headline(report=report)
    gain_by_group(report=report)
    response_surface()
    stage_zero()


if __name__ == "__main__":
    main()
