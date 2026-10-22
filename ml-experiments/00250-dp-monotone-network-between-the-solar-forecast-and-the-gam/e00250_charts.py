"""Draw the figures of experiment 250's write-up from `scores.json`.

uv run --no-sync python ml-experiments/00232-*/e00232_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00250-dp-monotone-network-between-the-solar-forecast-and-the-gam"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 600
LABELS: Final[dict[str, str]] = {
    "p": "Learned cap (247)",
    "n": "Network, unclipped",
    "n2": "Network, unclipped, two numbers",
    "nc": "Network and 60% clip",
}
COLOURS: Final[dict[str, str]] = {
    "p": ocf.DATA_SKY,
    "n": ocf.DATA_BLUE,
    "n2": ocf.BRAND_ORANGE,
    "nc": ocf.DATA_BLUE,
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


def minus_clip_figure(
    *, scores: dict, site: str, key: str, arms: tuple[str, ...], title: str, height: int
) -> alt.LayerChart:
    """Return one site's change in mean absolute error from the 60% clip, by arm."""
    rows = []
    for arm in arms:
        mean, low, high = scores[f"{site}|{arm}"][key]
        rows.append(
            {"row": LABELS[arm], "colour": COLOURS[arm], "mean": mean, "low": low, "high": high}
        )
    return interval_chart(
        frame=pl.DataFrame(rows),
        x_title="Mean absolute error minus the 60% clip's (MW; smaller is better)",
        title=[
            f"{site}, {title}.",
            "Dot: estimate. Line: 95% interval from resampling whole months.",
        ],
        zero_label="same as the 60% clip",
        height=height,
    )


def main() -> None:
    """Write the two figures."""
    scores = json.loads((DATA_DIR / "work" / "scores.json").read_text())
    sunny = "sunniest third of daylight half-hours, leads to 168 hours, scored year"
    arms = ("p", "n", "n2", "nc")
    save(
        chart=alt.vconcat(
            *[
                minus_clip_figure(
                    scores=scores,
                    site=s,
                    key="sunny_minus_clip",
                    arms=arms,
                    title=sunny,
                    height=150,
                )
                for s in ("D5", "D9")
            ]
        ).resolve_scale(x="shared"),
        name="sunny_error_change",
    )
    placebo = ("n", "nc")
    save(
        chart=alt.vconcat(
            *[
                minus_clip_figure(
                    scores=scores,
                    site=s,
                    key="all_minus_clip",
                    arms=placebo,
                    title="all half-hours, leads to 168 hours, scored year",
                    height=110,
                )
                for s in ("D10", "D16")
            ]
        ),
        name="placebos",
    )


if __name__ == "__main__":
    main()
