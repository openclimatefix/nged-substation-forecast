"""Draw the figures of experiment 214's write-up from `work/asof.json` and `work/stage1.json`.

uv run --no-sync python ml-experiments/00214-*/e00214_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00214_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 600
GREY: Final[str] = "#444444"


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG beside the page and as PNG in the private folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def zero_rule(*, label: str) -> alt.LayerChart:
    """Return a dashed vertical rule at zero with a text label."""
    data = pl.DataFrame({"x": [0.0], "t": [label]})
    rule = alt.Chart(data).mark_rule(strokeDash=[4, 3], color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    text = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=4, dy=-6, baseline="bottom", fontSize=10, color=GREY)
        .encode(x="x:Q", y=alt.value(0), text="t:N")
    )
    return rule + text


def headline(*, asof: dict) -> alt.LayerChart:
    """Figure 1: sunny-daylight error against the control, as-of and fixed 60%, by Primary."""
    scored = asof["scored"]
    rows = []

    def add(*, label: str, stats: dict, colour: str) -> None:
        mean, low, high = stats["sunny"]["mae_minus_control"]
        rows.append({"row": label, "colour": colour, "mean": mean, "low": low, "high": high})

    add(label="D5, ceiling chosen as-of", stats=scored["D5"]["as_of"], colour=ocf.BRAND_ORANGE)
    add(label="D5, fixed 60%", stats=asof["fixed"]["D5"], colour=ocf.BRAND_ORANGE)
    add(label="D9, ceiling chosen as-of", stats=scored["D9"]["as_of"], colour=ocf.DATA_PURPLE)
    add(label="D9, fixed 60%", stats=asof["fixed"]["D9"], colour=ocf.DATA_PURPLE)
    for site in ("D10", "D11", "D14", "D16"):
        add(label=f"{site}, fixed 60%", stats=asof["fixed"][site], colour=ocf.DATA_BLUE)
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("row:N", sort=None, title=None),
        color=alt.Color("colour:N", scale=None, legend=None),
    )
    rules = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q", title="Mean absolute error minus the control's (MW; more negative is better)"
        ),
        x2="high:Q",
    )
    dots = base.mark_point(filled=True, size=70, opacity=1.0).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    return (rules + dots + zero_rule(label="same as the control")).properties(
        width=WIDTH,
        height=250,
        title=[
            "Sunniest third of daylight half-hours, leads to 168 hours, scored year.",
            "Dot: estimate. Line: 98.33% interval from resampling whole months.",
        ],
    )


def sweep(*, asof: dict) -> alt.LayerChart:
    """Figure 2: error against the control for each clip ceiling, at D5 and D9, scored year."""
    rows = []
    for site, colour, shift in (("D5", ocf.BRAND_ORANGE, -0.8), ("D9", ocf.DATA_PURPLE, 0.8)):
        for ceiling, (mean, low, high) in asof["sweep"][f"{site}|scored"].items():
            rows.append(
                {
                    "site": site,
                    "colour": colour,
                    "ceiling": float(ceiling) + shift,
                    "mean": mean,
                    "low": low,
                    "high": high,
                }
            )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        x=alt.X(
            "ceiling:Q",
            title="Clip ceiling (% of the capacity held at each refit; 100 is no clip)",
            scale=alt.Scale(domain=[45, 95]),
            axis=alt.Axis(values=[50, 60, 70, 80, 90]),
        ),
        color=alt.Color("colour:N", scale=None, legend=None),
    )
    rules = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "low:Q", title="Mean absolute error minus the control's (MW; more negative is better)"
        ),
        y2="high:Q",
    )
    dots = base.mark_point(filled=True, size=70, opacity=1.0).encode(y="mean:Q")  # ty: ignore[unresolved-attribute]
    labels = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            pl.DataFrame(
                {"x": [91.0, 91.0], "y": [-0.12, -0.03], "t": ["D5 (orange)", "D9 (purple)"]}
            )
        )
        .mark_text(align="left", fontSize=11, color=GREY)
        .encode(x="x:Q", y="y:Q", text="t:N")
    )
    return (rules + dots + labels).properties(
        width=WIDTH,
        height=220,
        title=[
            "Sunniest third of daylight half-hours, all leads to 168 hours, scored year.",
            "Dot: estimate. Line: 98.33% interval from resampling whole months.",
        ],
    )


def screen(*, stage1: dict) -> alt.Chart:
    """Figure 3: implied solar output per unit of the fleet shape, by decile, per Primary."""
    rows = []
    for site in ("D5", "D9", "D10", "D11", "D14", "D16"):
        for summer, colour in (
            ("2022", ocf.BRAND_ORANGE),
            ("2023", ocf.DATA_BLUE),
            ("2024", ocf.DATA_SKY),
        ):
            cell = stage1[f"{site}|{summer}"]
            for decile, value in enumerate(cell.get("slope", []), start=1):
                if decile < 3:
                    continue
                rows.append(
                    {
                        "site": site,
                        "summer": summer,
                        "colour": colour,
                        "decile": decile,
                        "slope": value,
                    }
                )
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "decile:Q",
                title="Shape decile (8 = top)",
                axis=alt.Axis(values=[3, 4, 5, 6, 7, 8]),
            ),
            y=alt.Y("slope:Q", title="Points per unit shape"),
            color=alt.Color(
                "summer:N",
                scale=alt.Scale(
                    domain=["2022", "2023", "2024"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY],
                ),
                legend=alt.Legend(title="Summer", orient="top"),
            ),
        )
        .properties(width=170, height=120)
        .facet(
            facet=alt.Facet("site:N", title=None, sort=["D5", "D9", "D10", "D11", "D14", "D16"]),
            columns=3,
        )
        .resolve_scale(y="independent")
        .properties(
            title=[
                "Clear May to August half-hours before 2025-07-01. A fall towards the top deciles",
                "means the substation's net power flattens as sun rises, as an export limit does.",
            ]
        )
    )


def main() -> None:
    """Write the figures."""
    asof = json.loads((common.WORK_DIR / "asof.json").read_text())
    stage1 = json.loads((common.WORK_DIR / "stage1.json").read_text())
    save(chart=headline(asof=asof), name="figure_01_headline")
    save(chart=sweep(asof=asof), name="figure_02_ceiling_sweep")
    save(chart=screen(stage1=stage1), name="figure_03_screen")


if __name__ == "__main__":
    main()
