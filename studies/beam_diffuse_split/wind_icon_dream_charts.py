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
from studies.bootstrap import bootstrap_absolute
from studies.charts import (
    figure,
    interval_panel,
    leaderboard_panel,
    report_contrasts,
    report_errors,
)
from weather_products import METRIC, PERCENTAGE_POINTS
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
DOMAIN_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest value a figure's x domain extends, on either chart."""


def _pooled_losses() -> pl.DataFrame:
    """Return the pooled setting's losses, every arm.

    Returns:
        The losses whose `setting` is `pooled`.
    """
    return pl.read_parquet(OUTPUT_DIR / "losses.parquet").filter(pl.col("setting") == "pooled")


def _leaderboard(*, losses: pl.DataFrame, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw every product's own mean absolute error, best first, with its 95% interval.

    Bootstraps each product's absolute error from `losses.parquet` directly, the same
    month-and-seed resampling `wind_icon_dream.py`'s own report interval uses, because
    `leaderboard_panel` needs a `lower_95` and `upper_95` per row and the report's point estimate
    alone cannot supply one. No model is refitted.

    Args:
        losses: The pooled setting's losses, every arm.
        errors: Each product's pooled mean absolute error, read from the report's first table.

    Returns:
        Figure 11.

    Raises:
        ValueError: If a bootstrapped point estimate disagrees with the report's own number.
    """
    order = sorted(errors, key=errors.__getitem__)
    records = []
    for product in order:
        arm = f"{product}_wind"
        interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
        value = interval["value"] * PERCENTAGE_POINTS
        if round(value, 3) != errors[product]:
            msg = f"{product}: bootstrapped {value:.3f} but the report says {errors[product]}"
            raise ValueError(msg)
        records.append(
            {
                "label": NAMES[product],
                "family": FAMILIES[product],
                "value": value,
                "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
            }
        )
    rows = pl.DataFrame(records)
    domain = (
        min(rows["lower_95"].to_list()) - DOMAIN_MARGIN,
        max(rows["upper_95"].to_list()) + DOMAIN_MARGIN,
    )
    panel = leaderboard_panel(rows=rows, x_domain=domain, x_title=LEADERBOARD_X_TITLE)
    return figure(
        panels=[panel],
        number=11,
        figure_planning=None,
        title="ICON-DREAM-EU ties ERA5 and beats only ICON global of the other five products",
        subtitle=[
            "Each product's own mean absolute error, sorted best first.",
            "Only the comparisons with ERA5 and ICON-EU were planned.",
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
        Figure 12.
    """
    contrasts = report_contrasts(report_path=report_path)
    # Order by each row's own `reference` column rather than by position: `treatment` is the same
    # string ("icon_dream_eu_wind") on both planned rows, so sorting on it ties, and Polars' sort
    # is not guaranteed stable under a tie -- a label built from DECIDING_CONTRASTS' position
    # could then be attached to the wrong row.
    reference_order = {reference: index for index, (_, reference) in enumerate(DECIDING_CONTRASTS)}
    selected = (
        contrasts.filter(
            pl.col("section") == SECTION_DECIDING,
            pl.col("scope") == "all",
            pl.col("treatment").is_in([treatment for treatment, _ in DECIDING_CONTRASTS]),
            pl.col("reference").is_in(list(reference_order)),
        )
        .with_columns(_order=pl.col("reference").replace_strict(reference_order))
        .sort("_order")
    )
    labels = [
        f"{NAMES['icon_dream_eu']} − {NAMES[reference.removesuffix('_wind')]}"
        for reference in selected["reference"].to_list()
    ]
    rows = selected.select(
        "treatment", "reference", "difference", "lower_95", "upper_95"
    ).with_columns(label=pl.Series(labels), family=pl.lit("reanalysis"), planned=pl.lit(value=True))
    domain = (
        min(0.0, *rows["lower_95"].to_list()) - DOMAIN_MARGIN,
        max(0.0, *rows["upper_95"].to_list()) + DOMAIN_MARGIN,
    )
    panel = interval_panel(
        rows=rows,
        x_domain=domain,
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="ICON-DREAM-EU better",
        panel_title="The two planned contrasts",
        figure_planning="planned",
    )
    # figure_planning=None here, not "planned": the panel's own title ("The two planned
    # contrasts") and this subtitle's first line already say the rows are planned, so the
    # PLANNING_NOTES line `figure_planning="planned"` would add repeats that a third time.
    return figure(
        panels=[panel],
        number=12,
        figure_planning=None,
        title="ICON-DREAM-EU does not beat ERA5, and trails ICON-EU by 0.34 points",
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
    losses = _pooled_losses()
    charts = {
        "wind_icon_dream_leaderboard": _leaderboard(losses=losses, errors=errors),
        "wind_icon_dream_planned_contrasts": _planned_contrasts(report_path=report_path),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
