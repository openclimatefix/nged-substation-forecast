"""Draw the leaderboard and planned-contrasts charts for the ENS past-solar section.

One-off throwaway script for the charts of the ECMWF ENS addition to
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, in
`weather_product_charts.py`'s style. **Every number a chart shares with the page is read from
`ens_past_solar.py`'s report**, so a chart cannot disagree with the page.

Generators appear only as `A` to `F`, and no chart carries a calendar date. Every mark is drawn
with `aria=False`.

**Do not run this script until `ens_past_solar.py` has fitted every arm and written its report.**

Run it with `uv run python studies/beam_diffuse_split/ens_past_solar_charts.py`, after
`ens_past_solar.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`
before committing it.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from ens_past_solar import DECIDING_CONTRASTS, MEAN_ARM, OUTPUT_DIR
from studies.bootstrap import bootstrap_absolute
from studies.charts import (
    figure,
    interval_panel,
    leaderboard_panel,
    report_contrasts,
    report_errors,
)
from weather_products import METRIC, PERCENTAGE_POINTS

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

NAMES: Final[dict[str, str]] = {
    "ens_mean_t3": "ECMWF ENS (T+3 band)",
    "era5_global": "ERA5",
    "cams_global": "CAMS",
}
"""Every arm's public name, as the page writes it."""

FAMILIES: Final[dict[str, str]] = {
    "ens_mean_t3": "weather model",
    "era5_global": "reanalysis",
    "cams_global": "satellite",
}
"""Every arm's family, which sets its colour in `studies.charts`."""

SECTION_DECIDING: Final[str] = "Deciding contrasts, named before the run"
"""The report heading `ens_past_solar.py` writes above the two planned contrasts."""

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, April 2024 to September 2026."
LEADERBOARD_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
DOMAIN_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest value a figure's x domain extends, on either chart."""

FIGURE_LEADERBOARD: Final[int] = 17
FIGURE_CONTRASTS: Final[int] = 18


def _pooled_losses() -> pl.DataFrame:
    """Return every arm's losses.

    Returns:
        `ens_past_solar.py`'s saved losses, every arm at the `pooled` setting.
    """
    return pl.read_parquet(OUTPUT_DIR / "losses.parquet").filter(pl.col("setting") == "pooled")


def _leaderboard(*, losses: pl.DataFrame, errors: dict[str, float]) -> alt.VConcatChart:
    """Draw the three headline arms' own mean absolute error, best first, with its 95% interval.

    Bootstraps each arm's absolute error from `losses.parquet` directly, the same month-and-seed
    resampling `ens_past_solar.py`'s own report interval uses, because `leaderboard_panel` needs a
    `lower_95` and `upper_95` per row and the report's point estimate alone cannot supply one. No
    model is refitted.

    Args:
        losses: Every arm's losses, at the `pooled` setting.
        errors: Each arm's pooled mean absolute error, read from the report's first table.

    Returns:
        The figure.

    Raises:
        ValueError: If a bootstrapped point estimate disagrees with the report's own number.
    """
    order = sorted(NAMES, key=errors.__getitem__)
    records = []
    for arm in order:
        interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
        value = interval["value"] * PERCENTAGE_POINTS
        if round(value, 3) != errors[arm]:
            msg = f"{arm}: bootstrapped {value:.3f} but the report says {errors[arm]}"
            raise ValueError(msg)
        records.append(
            {
                "label": NAMES[arm],
                "family": FAMILIES[arm],
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
        number=FIGURE_LEADERBOARD,
        figure_planning=None,
        title="ENS's own forecast trails CAMS by far, and beats ERA5",
        subtitle=[
            "ECMWF ENS's T+3 band is a genuine forecast; ERA5 and CAMS are near-zero-lead",
            "descriptions of the same hours, refit on ENS's shorter row set.",
            DOTS,
            CAPACITY,
            SCOPE,
        ],
    )


def _planned_contrasts(*, report_path: Path) -> alt.VConcatChart:
    """Draw the two planned contrasts, ENS against ERA5 and against CAMS.

    Args:
        report_path: The `report.md` `ens_past_solar.py` wrote.

    Returns:
        The figure.
    """
    contrasts = report_contrasts(report_path=report_path)
    reference_order = {reference: index for index, (_, reference) in enumerate(DECIDING_CONTRASTS)}
    selected = (
        contrasts.filter(
            pl.col("section") == SECTION_DECIDING,
            pl.col("scope") == "all",
            pl.col("treatment") == MEAN_ARM,
            pl.col("reference").is_in(list(reference_order)),
        )
        .with_columns(_order=pl.col("reference").replace_strict(reference_order))
        .sort("_order")
    )
    labels = [
        f"{NAMES[MEAN_ARM]} − {NAMES[reference]}" for reference in selected["reference"].to_list()
    ]
    rows = selected.select(
        "treatment", "reference", "difference", "lower_95", "upper_95"
    ).with_columns(
        label=pl.Series(labels), family=pl.lit("weather model"), planned=pl.lit(value=True)
    )
    domain = (
        min(0.0, *rows["lower_95"].to_list()) - DOMAIN_MARGIN,
        max(0.0, *rows["upper_95"].to_list()) + DOMAIN_MARGIN,
    )
    panel = interval_panel(
        rows=rows,
        x_domain=domain,
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="ENS better",
        panel_title="The two planned contrasts",
        figure_planning="planned",
    )
    return figure(
        panels=[panel],
        number=FIGURE_CONTRASTS,
        figure_planning=None,
        title="ENS beats ERA5 but trails CAMS by more than three points",
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
    errors = {arm: errors[arm] for arm in NAMES if arm in errors}
    losses = _pooled_losses()
    charts = {
        "ens_past_solar_leaderboard": _leaderboard(losses=losses, errors=errors),
        "ens_past_solar_planned_contrasts": _planned_contrasts(report_path=report_path),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
