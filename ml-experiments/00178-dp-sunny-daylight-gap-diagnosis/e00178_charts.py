"""Draw the figures of experiment 178's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00178-*/e00178_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00178_diagnose as diagnose
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{diagnose.EXPERIMENT_ID:05d}-{diagnose.SLUG}"
)
REPORT: Final[Path] = diagnose.WORK_DIR / "report" / "report.json"
PNG_DIR: Final[Path] = diagnose.WORK_DIR / "png"
WIDTH: Final[int] = 640
SCOPE: Final[str] = "all_leads_six"


def bars(
    *, rows: list[dict], key: str, order: list[str], title: str, x_title: str
) -> alt.LayerChart:
    """Return bars of the mean GAM-minus-BG2 absolute error per level, with the gap share."""
    frame = pl.DataFrame(
        {
            "level": [str(r[key]) for r in rows],
            "gap": [r["gam_mae"] - r["bg2_mae"] for r in rows],
            "anchor": [max(r["gam_mae"] - r["bg2_mae"], 0.0) for r in rows],
            "label": [f"{r['gap_share']:.0%} of gap, {r['row_share']:.0%} of rows" for r in rows],
        }
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("level:N", sort=order, title=x_title),
        x=alt.X("gap:Q", title="GAM minus BG2 mean absolute error (MW); above zero = GAM worse"),
    )
    layer = base.mark_bar(color=ocf.BRAND_ORANGE)
    text = base.mark_text(align="left", dx=4, fontSize=10).encode(  # ty: ignore[unresolved-attribute]
        x="anchor:Q", text="label:N"
    )
    return (layer + text).properties(width=WIDTH, height=22 * len(rows) + 40, title=title)


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG beside the page and as PNG in the private folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def main() -> None:
    """Draw the five figures."""
    report = json.loads(REPORT.read_text())[SCOPE]
    hours = [str(r["hour_bin"]) for r in report["hour_bin"]]
    save(
        chart=bars(
            rows=report["hour_bin"],
            key="hour_bin",
            order=hours,
            title="No 2-hour bin of the day holds more than a third of the gap",
            x_title="Local hour of day (start of 2-hour bin)",
        ),
        name="figure_01_hour",
    )
    save(
        chart=bars(
            rows=report["csi_band"],
            key="csi_band",
            order=list(diagnose.CSI_LABELS),
            title="The GAM loses on partly cloudy half-hours and is level or ahead on clear ones",
            x_title="Clear-sky index of the ensemble-mean shortwave radiation",
        ),
        name="figure_02_clear_sky_band",
    )
    save(
        chart=bars(
            rows=report["lead_band"],
            key="lead_band",
            order=list(diagnose.LEAD_LABELS),
            title="Most of the gap is at 72 to 168 hours, which also holds most of the rows",
            x_title="Lead band",
        ),
        name="figure_03_lead",
    )
    sites = ["D5", "D10", "D11", "D14", "D16", "D9"]
    save(
        chart=bars(
            rows=report["site"],
            key="site",
            order=sites,
            title="One substation, D5, holds most of the gap",
            x_title="Substation",
        ),
        name="figure_04_site",
    )
    save(
        chart=bars(
            rows=report["not_d5_csi_band"],
            key="csi_band",
            order=list(diagnose.CSI_LABELS),
            title="Without D5 the partly cloudy loss remains and clear half-hours favour the GAM",
            x_title="Clear-sky index of the ensemble-mean shortwave radiation, five substations",
        ),
        name="figure_05_clear_sky_band_without_d5",
    )


if __name__ == "__main__":
    main()
