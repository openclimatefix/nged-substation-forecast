"""Draw the figure of experiment 223's write-up from `work/rowa.json`.

uv run --no-sync python ml-experiments/00223-*/e00223_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00223_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
LABELS: Final[dict[str, str]] = {
    "G1": "Wind G1 (D13)",
    "G2": "Wind G2 (D13, MVA)",
    "G3": "Wind G3 (D17, MVA)",
    "G4": "Biomass G4 (D17)",
    "G5": "Battery G5 (D17)",
    "solar_D13": "Solar group, D13 (3 farms)",
    "solar_D17": "Solar group, D17 (3 farms)",
}
BSP_COLOURS: Final[dict[str, str]] = {"D13": ocf.DATA_BLUE, "D17": ocf.BRAND_ORANGE}


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_coefficients(*, data: dict) -> alt.TopLevelMixin:
    """The remainder's coefficient on each generator, with the 99.3% interval and the placebo."""
    rows = []
    for name, unit in data["units"].items():
        placebo = data["known_answer"][name]["planted_behind_primary"]["slope"]
        rows.append(
            {
                "unit": LABELS[name],
                "bsp": unit["bsp"],
                "slope": unit["slope"],
                "low": unit["low"],
                "high": unit["high"],
                "placebo": placebo,
            }
        )
    frame = pl.DataFrame(rows)
    order = [LABELS[n] for n in data["units"]]
    base = alt.Chart(frame).encode(y=alt.Y("unit:N", sort=order, title=None))
    bars = base.mark_rule(size=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Coefficient of the remainder on the generator's output"),
        x2="high:Q",
        color=alt.Color(
            "bsp:N",
            scale=alt.Scale(domain=list(BSP_COLOURS), range=list(BSP_COLOURS.values())),
            title="Bulk supply point",
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x="slope:Q", color="bsp:N"
    )
    placebo = base.mark_point(shape="cross", size=70, color="#808080").encode(  # ty: ignore[unresolved-attribute]
        x="placebo:Q"
    )
    guides = (
        alt.Chart(pl.DataFrame({"x": [-1.0, 0.0]}))
        .mark_rule(strokeDash=[4, 4], color="#808080")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (guides + bars + points + placebo).properties(
        width=520,
        height=200,
        title=alt.TitleParams(
            text="No generator's coefficient sits at 0, but few sit at -1",
            subtitle=textwrap.wrap(
                "Dots: coefficient. Bars: 99.3% interval over resampled days. Grey crosses: the "
                "same regression on a copy of the generator's output shifted by 364 days. A "
                "coefficient of -1 means the bulk supply point carries the generator; 0 means a "
                "Primary does.",
                95,
            ),
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        ),
    )


def main() -> None:
    """Draw the figure from `work/rowa.json`."""
    data = json.loads((common.WORK_DIR / "rowa.json").read_text())
    _save(figure_coefficients(data=data), name="figure_1_coefficients")


if __name__ == "__main__":
    main()
