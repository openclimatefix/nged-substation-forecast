"""Draw the figures of experiment 247's write-up from `scores.json`.

uv run --no-sync python ml-experiments/00232-*/e00232_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00247-dp-learned-export-cap-for-the-solar-clip"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 600
LABELS: Final[dict[str, str]] = {
    "k60": "60% clip (hand-set)",
    "sat": "60% clip, softer edge",
    "p": "Learned: profile fit",
    "e": "Learned: clear-sky envelope",
    "as_of": "As-of choice (experiment 214)",
}
COLOURS: Final[dict[str, str]] = {
    "k60": ocf.BRAND_ORANGE,
    "sat": ocf.BRAND_ORANGE,
    "p": ocf.DATA_BLUE,
    "e": ocf.DATA_SKY,
    "as_of": "#444444",
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


def contrast_figure(
    *, scores: dict, site: str, arms: tuple[str, ...], height: int
) -> alt.LayerChart:
    """Return one site's change in mean absolute error from the unclipped control, by arm."""
    rows = []
    for arm in arms:
        if arm == "as_of":
            mean, low, high = scores["as_of_214"][site]
        else:
            mean, low, high = scores[f"{site}|{arm}"]["sunny"]["mae_minus_control"]
        rows.append(
            {"row": LABELS[arm], "colour": COLOURS[arm], "mean": mean, "low": low, "high": high}
        )
    return interval_chart(
        frame=pl.DataFrame(rows),
        x_title="Mean absolute error minus the unclipped fit's (MW; smaller is better)",
        title=[
            f"{site}, sunniest third of daylight half-hours, all leads to 168 hours, scored year.",
            "Dot: estimate. Line: 95% interval from resampling whole months.",
        ],
        zero_label="same as no clip",
        height=height,
    )


def shares_figure(*, caps: dict) -> alt.HConcatChart:
    """Return the ceiling share each learned arm picked at each monthly cut at D5 and D9."""
    panels = []
    for site in ("D5", "D9"):
        rows = [
            {"arm": LABELS[arm], "cut": cut, "share": share}
            for arm in ("p", "e")
            for cut, share in caps[site][arm].items()
        ]
        frame = pl.DataFrame(rows).with_columns(pl.col("cut").str.to_date())
        lines = (
            alt.Chart(frame)
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("cut:T", title="Monthly cut"),
                y=alt.Y(
                    "share:Q",
                    title="Ceiling as a share of held capacity",
                    scale=alt.Scale(domain=[0.3, 1.0]),
                ),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(
                        domain=[LABELS["p"], LABELS["e"]], range=[ocf.DATA_BLUE, ocf.DATA_SKY]
                    ),
                    title=None,
                    legend=alt.Legend(orient="bottom", labelLimit=300),
                ),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"y": [0.6]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color=ocf.BRAND_ORANGE)
            .encode(y="y:Q")
        )
        panels.append(
            alt.layer(lines, rule).properties(width=WIDTH // 2 - 60, height=200, title=site)
        )
    return alt.hconcat(*panels)


def main() -> None:
    """Write the three figures."""
    scores = json.loads((DATA_DIR / "work" / "scores.json").read_text())
    caps = json.loads((DATA_DIR / "work" / "learned_caps.json").read_text())
    arms = ("k60", "sat", "p", "e", "as_of")
    save(
        chart=alt.vconcat(
            contrast_figure(scores=scores, site="D5", arms=arms, height=170),
            contrast_figure(scores=scores, site="D9", arms=arms, height=170),
        ).resolve_scale(x="shared"),
        name="sunny_error_change",
    )
    save(chart=shares_figure(caps=caps), name="learned_shares")
    placebo = ("k60", "sat", "p", "e")
    save(
        chart=alt.vconcat(
            contrast_figure(scores=scores, site="D10", arms=placebo, height=130),
            contrast_figure(scores=scores, site="D16", arms=placebo, height=130),
        ),
        name="placebos",
    )


if __name__ == "__main__":
    main()
