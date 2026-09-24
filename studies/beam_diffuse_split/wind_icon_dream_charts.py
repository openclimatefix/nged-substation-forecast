"""Draw the leaderboard and planned-contrasts charts for the ICON-DREAM-EU wind study.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, in
`wind_product_charts.py`'s style. **Every number a chart shares with the page is read from
`wind_icon_dream.py`'s report**, so a chart cannot disagree with the page.

Generators appear only as `W1` to `W3`, and no chart carries a calendar date. Every mark is drawn
with `aria=False`, including point overlays, so Vega does not write each point's value into the
SVG's ARIA labels.

**Do not run this script until `wind_icon_dream.py` has fitted every arm and written its report.**

Run it with `uv run python studies/beam_diffuse_split/wind_icon_dream_charts.py`, after
`wind_icon_dream.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`
before committing it.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from studies.charts import (
    figure,
    interval_panel,
    leaderboard_panel,
    report_contrasts,
    report_errors,
)
from wind_icon_dream import DECIDING_CONTRASTS, OUTPUT_DIR

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

NAMES: Final[dict[str, str]] = {
    "era5": "ERA5",
    "ukv": "UKV",
    "icon_d2": "ICON-D2",
    "icon_eu": "ICON-EU",
    "icon_global": "ICON global",
    "icon_dream_eu": "ICON-DREAM-EU",
}
"""Every product's public name, as the page writes it."""

FAMILIES: Final[dict[str, str]] = {
    "era5": "reanalysis",
    "ukv": "weather model",
    "icon_d2": "weather model",
    "icon_eu": "weather model",
    "icon_global": "weather model",
    "icon_dream_eu": "reanalysis",
}
"""Every product's family, which sets its colour in `studies.charts`."""

SECTION_DECIDING: Final[str] = "Deciding contrasts, named before the run"
"""The report heading `wind_icon_dream.py` writes above the two planned contrasts."""

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "Three wind farms in Lincolnshire, August 2024 to August 2026."
LEADERBOARD_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
LEADERBOARD_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest product's error the leaderboard's x domain extends."""


def _leaderboard(*, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw every product's own mean absolute error, best first, with its 95% interval.

    Args:
        errors: Each product's pooled mean absolute error, read from the report's first table.

    Returns:
        Figure 1.
    """
    order = sorted(errors, key=errors.__getitem__)
    domain = (
        min(errors.values()) - LEADERBOARD_MARGIN,
        max(errors.values()) + LEADERBOARD_MARGIN,
    )
    rows = pl.DataFrame(
        [
            {"label": NAMES[product], "family": FAMILIES[product], "value": errors[product]}
            for product in order
        ]
    )
    panel = leaderboard_panel(rows=rows, x_domain=domain, x_title=LEADERBOARD_X_TITLE)
    return figure(
        panels=[panel],
        number=1,
        figure_planning=None,
        title="ICON-DREAM-EU's own mean absolute error, against the five products already scored",
        subtitle=[
            "Each product's own mean absolute error, sorted best first.",
            DOTS,
            CAPACITY,
            SCOPE,
        ],
    )


def _planned_contrasts(*, report_path: Path) -> alt.VConcatChart:
    """Draw the two planned contrasts, ICON-DREAM-EU against ERA5 and against ICON-EU.

    Args:
        report_path: The `report.md` `wind_icon_dream.py` wrote.

    Returns:
        Figure 2.
    """
    contrasts = report_contrasts(report_path=report_path)
    selected = contrasts.filter(
        pl.col("section") == SECTION_DECIDING,
        pl.col("scope") == "all",
        pl.col("treatment").is_in([treatment for treatment, _ in DECIDING_CONTRASTS]),
        pl.col("reference").is_in([reference for _, reference in DECIDING_CONTRASTS]),
    ).sort(
        pl.col("treatment").replace_strict(
            {treatment: index for index, (treatment, _) in enumerate(DECIDING_CONTRASTS)}
        )
    )
    labels = [
        f"{NAMES['icon_dream_eu']} − {NAMES[reference.removesuffix('_wind')]}"
        for _, reference in DECIDING_CONTRASTS
    ]
    rows = selected.select(
        "treatment", "reference", "difference", "lower_95", "upper_95"
    ).with_columns(label=pl.Series(labels), family=pl.lit("reanalysis"), planned=pl.lit(value=True))
    panel = interval_panel(
        rows=rows,
        x_domain=(-1.0, 0.6),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="ICON-DREAM-EU better",
        panel_title="The two planned contrasts",
        figure_planning="planned",
    )
    return figure(
        panels=[panel],
        number=2,
        figure_planning="planned",
        title="ICON-DREAM-EU against ERA5 and against ICON-EU",
        subtitle=[
            "The two contrasts named in the plan before any result existed.",
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def main() -> int:
    """Read the report and write the two SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = OUTPUT_DIR / "report.md"
    errors = report_errors(report_path=report_path, column="All sites")
    errors = {product: errors[product] for product in NAMES if product in errors}
    charts = {
        "wind_icon_dream_leaderboard": _leaderboard(errors=errors),
        "wind_icon_dream_planned_contrasts": _planned_contrasts(report_path=report_path),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
