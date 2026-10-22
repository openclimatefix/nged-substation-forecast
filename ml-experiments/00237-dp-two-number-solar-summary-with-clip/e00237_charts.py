"""Draw the figures of experiment 237's write-up from `work/report/scores.json`.

uv run --no-sync python ml-experiments/00237-*/e00237_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00237-dp-two-number-solar-summary-with-clip"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 600


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG beside the page and as PNG in the private folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "png").mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(DATA_DIR / "png" / f"{name}.png", scale_factor=1.5)


def interval_chart(
    *, frame: pl.DataFrame, x_title: str, title: list[str], zero_label: str, height: int
) -> alt.LayerChart:
    """Return dots with interval rules, one row per `row`, and a labelled zero rule."""
    base = alt.Chart(frame).encode(
        y=alt.Y("row:N", sort=None, title=None),
        color=alt.Color("colour:N", scale=None, legend=None),
    )
    rules = base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title=x_title), x2="high:Q")  # ty: ignore[unresolved-attribute]
    dots = base.mark_point(filled=True, size=70, opacity=1.0).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#444444")
        .encode(x="x:Q")
    )
    label = (
        alt.Chart(pl.DataFrame({"x": [0.0], "t": [zero_label]}))  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=4, dy=-6, baseline="bottom", fontSize=10, color="#444444")
        .encode(x="x:Q", y=alt.value(0), text="t:N")
    )
    return (rules + dots + zero + label).properties(width=WIDTH, height=height, title=title)


def rows_for(*, cells: dict, keys: list[tuple[str, str, str, str]]) -> pl.DataFrame:
    """Return one chart row per (label, cell key, field, colour)."""
    rows = []
    for label, cell_key, field, colour in keys:
        mean, low, high = cells[cell_key][field]
        rows.append({"row": label, "colour": colour, "mean": mean, "low": low, "high": high})
    return pl.DataFrame(rows)


def gains(*, scores: dict) -> alt.LayerChart:
    """Figure 1: the two-number gain on daylight half-hours, with and without the clip."""
    scored = scores["scored"]
    keys = []
    for site in ("D5", "D9"):
        keys += [
            (f"{site}, no clip", f"{site}|daylight", "gain_noclip_q2", ocf.DATA_BLUE),
            (f"{site}, clip", f"{site}|daylight", "gain_clip_q2", ocf.BRAND_ORANGE),
        ]
    return interval_chart(
        frame=rows_for(cells=scored, keys=keys),
        x_title="Quantile score change from arm A (%; more negative is better)",
        title=[
            "Two-number summary (arm Q2), daylight half-hours, leads to 168 hours, scored year.",
            "Dot: estimate. Line: 95% interval from resampling whole months.",
        ],
        zero_label="same as arm A",
        height=200,
    )


def retained(*, scores: dict) -> alt.LayerChart:
    """Figure 2: the share of the gain that survives the clip."""
    keys = []
    for period in ("scored", "replicate"):
        for site in ("D5", "D9"):
            for mask in ("daylight", "sunny"):
                colour = ocf.DATA_BLUE if mask == "daylight" else ocf.BRAND_ORANGE
                keys.append(
                    (f"{site} {mask}, {period}", f"{period}|{site}|{mask}", "retained_q2", colour)
                )
    flat = {f"{p}|{k}": v for p, cells in scores.items() for k, v in cells.items()}
    return interval_chart(
        frame=rows_for(cells=flat, keys=keys),
        x_title="Gain with the clip divided by gain without it (1 = all of the gain kept)",
        title=[
            "Share of the gain kept with the clip. Blue: daylight. Orange: sunniest third.",
            "Dot: estimate. Line: 95% interval from resampling whole months.",
        ],
        zero_label="none kept",
        height=300,
    )


def median_error(*, scores: dict) -> alt.LayerChart:
    """Figure 3: the change in the median's absolute error from the two numbers."""
    scored = scores["scored"]
    keys = []
    for site, mask in (("D5", "daylight"), ("D5", "sunny"), ("D9", "daylight")):
        keys += [
            (f"{site} {mask}, no clip", f"{site}|{mask}", "mae_change_noclip_q2", ocf.DATA_BLUE),
            (f"{site} {mask}, clip", f"{site}|{mask}", "mae_change_clip_q2", ocf.BRAND_ORANGE),
        ]
    return interval_chart(
        frame=rows_for(cells=scored, keys=keys),
        x_title="Mean absolute error of the median, change from arm A (%; negative is better)",
        title=[
            "Two-number summary (arm Q2), scored year.",
            "Dot: estimate. Line: 95% interval from resampling whole months.",
        ],
        zero_label="same as arm A",
        height=300,
    )


def main() -> None:
    """Write the three figures."""
    scores = json.loads((DATA_DIR / "work" / "report" / "scores.json").read_text())
    save(chart=gains(scores=scores), name="figure_01_gain_with_and_without_clip")
    save(chart=retained(scores=scores), name="figure_02_share_kept")
    save(chart=median_error(scores=scores), name="figure_03_median_error")


if __name__ == "__main__":
    main()
