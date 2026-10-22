"""Draw the figures of experiment 232's write-up from `scores.json`.

uv run --no-sync python ml-experiments/00232-*/e00232_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00232-dp-pv-clip-with-correction-heads"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 600
LABELS: Final[dict[str, str]] = {
    "C": "C: arm D, no clip, no head",
    "K": "K: 60% clip",
    "H2": "H: head of experiment 201",
    "KH2": "KH: clip and head",
}
COLOURS: Final[dict[str, str]] = {
    "C": ocf.DATA_BLUE,
    "K": ocf.BRAND_ORANGE,
    "H2": ocf.DATA_SKY,
    "KH2": ocf.DATA_BLUE,
}


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


def headline(*, scores: dict) -> alt.LayerChart:
    """Figure 1: D5's sunny-daylight gap to BG2 by arm."""
    rows = []
    for arm, label in LABELS.items():
        mean, low, high = scores["D5|sunny"][f"{arm}_gap_to_BG2"]
        rows.append({"row": label, "colour": COLOURS[arm], "mean": mean, "low": low, "high": high})
    return interval_chart(
        frame=pl.DataFrame(rows),
        x_title="Mean absolute error minus BG2's (MW; smaller is better)",
        title=[
            "D5, sunniest third of daylight half-hours, all leads to 168 hours, scored year.",
            "Dot: estimate. Line: 98.75% interval from resampling whole months.",
        ],
        zero_label="same as BG2",
        height=150,
    )


def by_site(*, scores: dict) -> alt.LayerChart:
    """Figure 2: each arm's change against the control at D5 and the two placebos."""
    rows = []
    for site in ("D5", "D10", "D16"):
        for arm in ("K", "H2", "KH2"):
            mean, low, high = scores[f"{site}|sunny"][f"{arm}_minus_C"]
            rows.append(
                {
                    "row": f"{LABELS[arm].split(':')[0]} at {site}",
                    "colour": COLOURS[arm],
                    "mean": mean,
                    "low": low,
                    "high": high,
                }
            )
    return interval_chart(
        frame=pl.DataFrame(rows),
        x_title="Mean absolute error minus the control's (MW; more negative is better)",
        title=[
            "Sunniest third of daylight half-hours, all leads, scored year.",
            "Dot: estimate. Line: 98.75% interval from resampling whole months.",
        ],
        zero_label="same as the control",
        height=260,
    )


def additivity(*, scores: dict) -> alt.LayerChart:
    """Figure 3: the clip's effect with and without the head, and the sum of the two effects."""
    cell = scores["D5|sunny"]
    rows = []
    for row, key, colour in (
        ("Clip alone (K minus C)", "K_minus_C", ocf.BRAND_ORANGE),
        ("Head alone (H minus C)", "H2_minus_C", ocf.DATA_SKY),
        ("Clip and head (KH minus C)", "KH2_minus_C", ocf.DATA_BLUE),
        ("Overlap: KH minus K minus H plus C", "additivity_H2", "#444444"),
    ):
        mean, low, high = cell[key]
        rows.append({"row": row, "colour": colour, "mean": mean, "low": low, "high": high})
    return interval_chart(
        frame=pl.DataFrame(rows),
        x_title="Change in mean absolute error at D5 (MW; negative is better)",
        title=[
            "D5, sunniest third of daylight half-hours. If the effects added, the overlap is zero.",
            "Dot: estimate. Line: 98.75% interval from resampling whole months.",
        ],
        zero_label="zero",
        height=150,
    )


def main() -> None:
    """Write the three figures."""
    scores = json.loads((DATA_DIR / "scores.json").read_text())
    save(chart=headline(scores=scores), name="figure_01_headline")
    save(chart=by_site(scores=scores), name="figure_02_by_site")
    save(chart=additivity(scores=scores), name="figure_03_overlap")


if __name__ == "__main__":
    main()
