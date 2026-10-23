"""Draw the figures of experiment 279's write-up from `work/numbers.json`.

uv run --no-sync python ml-experiments/00279-*/e00279_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00279-dp-heating-curve-monotone-lagged-temperature-and-wind"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 600
ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "A1: smooth curve", ocf.DATA_SKY),
    ("C2", "A2: A1 plus lagged temperatures", ocf.DATA_BLUE),
    ("C3", "A3: network (pre-registered)", ocf.BRAND_ORANGE),
    ("C7", "A3b: network (post hoc)", ocf.DATA_PURPLE),
)
SUBSETS: Final[tuple[tuple[str, str], ...]] = (
    ("cold_night", "Cold nights"),
    ("nov_mar", "November to March"),
    ("all_rows", "All rows"),
)


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG beside the page and as PNG in the private folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "png").mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(DATA_DIR / "png" / f"{name}.png", scale_factor=1.5)


def interval_layers(*, frame: pl.DataFrame, x_title: str, height: int) -> alt.LayerChart:
    """Return dots with interval rules and a labelled zero rule."""
    base = alt.Chart(frame).encode(
        y=alt.Y("row:N", sort=None, title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.Color("colour:N", scale=None, legend=None),
    )
    rules = base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title=x_title), x2="high:Q")  # ty: ignore[unresolved-attribute]
    dots = base.mark_point(filled=True, size=70, opacity=1.0).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#444444")
        .encode(x="x:Q")
    )
    return (rules + dots + zero).properties(width=WIDTH, height=height)


def headline(*, numbers: dict) -> alt.TopLevelMixin:
    """Return the cold-night contrasts of each arm against T-all-K, with 99% intervals."""
    rows = []
    for key, label, colour in ROWS:
        cold = numbers["contrasts"][key]["cold_night"]
        rows.append(
            {
                "row": label,
                "colour": colour,
                "mean": cold["estimate"],
                "low": cold["low"],
                "high": cold["high"],
            }
        )
    return interval_layers(
        frame=pl.DataFrame(rows),
        x_title="Quantile score minus T-all-K's (points of capacity; more negative is better)",
        height=170,
    ).properties(
        title=[
            "Cold nights (22:00 to 08:00 on days with a mean of at most 3 C), five Primaries,",
            "scored year. Dot: estimate. Line: 99% interval from resampling whole months.",
            "Dashed line: same score as T-all-K.",
        ]
    )


def by_subset(*, numbers: dict) -> alt.TopLevelMixin:
    """Return one panel per row subset for A1, A2, and the two networks."""
    panels = []
    for subset, title in SUBSETS:
        rows = []
        for key, label, colour in ROWS:
            entry = numbers["contrasts"][key][subset]
            rows.append(
                {
                    "row": label,
                    "colour": colour,
                    "mean": entry["estimate"],
                    "low": entry["low"],
                    "high": entry["high"],
                }
            )
        panels.append(
            interval_layers(
                frame=pl.DataFrame(rows),
                x_title="Quantile score minus T-all-K's (points; more negative is better)",
                height=120,
            ).properties(title=f"{title}, scored year, 99% interval")
        )
    return alt.vconcat(*panels).resolve_scale(x="shared")


def curves(*, numbers: dict) -> alt.TopLevelMixin:
    """Return the post hoc network's fitted heating curve at each site, with a straight line."""
    rows = []
    grid = [-4.0 + i for i in range(len(next(iter(numbers["curves"].values()))["curve"]))]
    for key, entry in numbers["curves"].items():
        variant, site = key.split("_", 1)
        if variant != "a3b":
            continue
        for temperature, power in zip(grid, entry["curve"], strict=True):
            rows.append(
                {"site": site, "temperature": temperature, "power": power, "kind": "network"}
            )
        cold_end = entry["curve"][0]
        rows.extend(
            {
                "site": site,
                "temperature": temperature,
                "power": cold_end * (15.5 - temperature) / 19.5,
                "kind": "straight line",
            }
            for temperature in (-4.0, 15.5)
        )
    frame = pl.DataFrame(rows)
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("temperature:Q", title="Temperature at every lag (C)"),
            y=alt.Y("power:Q", title="Heating (points)"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["network", "straight line"], range=[ocf.DATA_PURPLE, "#888888"]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash("kind:N", legend=None),
        )
        .properties(width=WIDTH // 3 - 20, height=130)
        .facet(facet=alt.Facet("site:N", title=None), columns=3)
        .properties(
            title=[
                "Heating curve the post hoc network learned (average of 16 hour-band groups,",
                "mean wind), against the straight line through its coldest point. Scored-year fit.",
            ]
        )
    )


def main() -> None:
    """Write the figures."""
    numbers = json.loads((DATA_DIR / "work" / "numbers.json").read_text())
    save(chart=headline(numbers=numbers), name="cold_night_contrasts")
    save(chart=by_subset(numbers=numbers), name="contrasts_by_rows")
    save(chart=curves(numbers=numbers), name="network_curves")


if __name__ == "__main__":
    main()
