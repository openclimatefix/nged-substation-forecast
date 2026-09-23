"""Draw the anonymised charts for the write-up on blending CAMS and SARAH-3.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/860>.

**Every interval and every error a chart shares with the page is read from the report
`blend_satellites.py` wrote**, so a chart cannot disagree with the page. The leaderboard also draws
each arm's own absolute error, bootstrapped straight from `losses.parquet` with no refit, the same
month-and-seed resampling the report's contrasts use.

Generators appear only as `A` to `F`, and every output is a fraction of the generator's own
capacity. Follow the style of `blend_product_charts.py`.

Run it with `uv run python studies/beam_diffuse_split/blend_satellites_charts.py`, after
`blend_satellites.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from blend_satellites import (
    DERIVED_ARMS,
    METRIC,
    NEGATIVE_CONTROL,
    OUTPUT_DIR,
    PERCENTAGE_POINTS,
    PLANNED_CONTRASTS,
)
from studies.bootstrap import bootstrap_absolute
from studies.charts import (
    ContrastKey,
    figure,
    interval_panel,
    leaderboard_panel,
    planning,
    report_contrasts,
    select_contrasts,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

NAMES: Final[dict[str, str]] = {
    "cams": "CAMS",
    "sarah3": "SARAH-3",
    "cams_sarah3_xgb": "CAMS + SARAH-3 (XGBoost)",
    "cams_sarah3_control": "CAMS + SARAH-3 (climatology control)",
    "cams_sarah3_mean": "CAMS + SARAH-3 (mean)",
    "cams_sarah3_stack": "CAMS + SARAH-3 (stack)",
    "cams_sarah3_equal": "CAMS + SARAH-3 (equal weight)",
    "cams_cams_noise_xgb": "CAMS + noised CAMS (negative control)",
}
"""Each arm's name as the page writes it. Every blend names both products it reads."""

SECTION_PLANNED: Final[str] = "Planned contrasts"
SECTION_SENSITIVITY: Final[str] = "Planned contrasts at the second hyperparameter setting"
SECTION_NEGATIVE_CONTROL: Final[str] = (
    "Negative control: CAMS plus a noised copy of itself, against CAMS alone"
)
SECTION_METHODS: Final[str] = "Exploratory: the mean, stack and equal blends against CAMS"

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months and a seed."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, January 2021 to August 2026."
LEADERBOARD_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
CONTRAST_X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"

LEADERBOARD_DOMAIN: Final[tuple[float, float]] = (5.0, 10.0)
"""The x range of the leaderboard, covering every arm's 95% interval with a small margin."""

CONTRAST_DOMAIN: Final[tuple[float, float]] = (-0.6, 0.6)
"""The x range shared by the contrast panels."""


def _leaderboard_rows(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Bootstrap every arm's own absolute error, best first.

    Args:
        losses: Every arm's rows from `losses.parquet`, primary setting.

    Returns:
        One row per arm, in `NAMES`' order, with `label`, `family`, `value`, `lower_95`, `upper_95`.
    """
    records = []
    for arm in NAMES:
        interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
        records.append(
            {
                "label": NAMES[arm],
                "family": "satellite",
                "value": interval["value"] * PERCENTAGE_POINTS,
                "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
            }
        )
    return pl.DataFrame(records).sort("value")


def _leaderboard(*, losses: pl.DataFrame) -> alt.VConcatChart:
    """Draw every arm's own mean absolute error, best first, with its 95% interval.

    Args:
        losses: Every arm's rows from `losses.parquet`, primary setting.

    Returns:
        Figure 1.
    """
    rows = _leaderboard_rows(losses=losses)
    panel = leaderboard_panel(rows=rows, x_domain=LEADERBOARD_DOMAIN, x_title=LEADERBOARD_X_TITLE)
    return figure(
        panels=[panel],
        number=1,
        figure_planning=None,
        title="Every arm's own mean absolute error, CAMS and SARAH-3 and their blends",
        subtitle=[
            "Each arm's own mean absolute error, sorted best first.",
            DOTS,
            (
                "The intervals are wide mainly because every arm's error swings together from "
                "month to month, a swing the contrasts in Figures 2 and 3 cancel."
            ),
            CAPACITY,
            SCOPE,
        ],
    )


def _headline(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw the two planned contrasts at both hyperparameter settings.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 2.
    """
    primary = select_contrasts(
        contrasts=contrasts,
        wanted=[ContrastKey(SECTION_PLANNED, "all", t, r) for t, r in PLANNED_CONTRASTS],
    )
    sensitivity = select_contrasts(
        contrasts=contrasts,
        wanted=[ContrastKey(SECTION_SENSITIVITY, "all", t, r) for t, r in PLANNED_CONTRASTS],
    )
    labels = [f"{NAMES[t]} − {NAMES[r]}" for t, r in PLANNED_CONTRASTS]
    rows = pl.concat(
        [
            frame.select("difference", "lower_95", "upper_95").with_columns(
                label=pl.Series(labels),
                family=pl.lit("satellite"),
                planned=pl.lit(value=True),
                condition=pl.lit(condition),
            )
            for frame, condition in ((primary, "Primary setting"), (sensitivity, "Second setting"))
        ]
    )
    panel = interval_panel(
        rows=rows,
        x_domain=CONTRAST_DOMAIN,
        x_title=CONTRAST_X_TITLE,
        zero_label="same as the reference arm",
        better_label="lower error than the reference arm",
        conditions=("Primary setting", "Second setting"),
        condition_title="Hyperparameter setting",
        figure_planning=planning(rows=[rows]),
    )
    return figure(
        panels=[panel],
        number=2,
        figure_planning=planning(rows=[rows]),
        title="Does the blend beat CAMS, and does the gain survive a climatology control?",
        subtitle=[DOTS, CAPACITY, SCOPE],
    )


def _exploratory(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw the mean, stack and equal blends and the negative control, all against CAMS.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 3.
    """
    method_pairs = [(arm, "cams") for arm in ("cams_sarah3_mean", *DERIVED_ARMS)]
    methods = select_contrasts(
        contrasts=contrasts,
        wanted=[ContrastKey(SECTION_METHODS, "all", t, r) for t, r in method_pairs],
    )
    control = select_contrasts(
        contrasts=contrasts,
        wanted=[ContrastKey(SECTION_NEGATIVE_CONTROL, "all", *NEGATIVE_CONTROL)],
    )
    labels = [f"{NAMES[t]} − {NAMES[r]}" for t, r in method_pairs] + [
        f"{NAMES[NEGATIVE_CONTROL[0]]} − {NAMES[NEGATIVE_CONTROL[1]]}"
    ]
    rows = (
        pl.concat([methods, control])
        .select("difference", "lower_95", "upper_95")
        .with_columns(
            label=pl.Series(labels), family=pl.lit("satellite"), planned=pl.lit(value=False)
        )
    )
    panel = interval_panel(
        rows=rows,
        x_domain=CONTRAST_DOMAIN,
        x_title=CONTRAST_X_TITLE,
        zero_label="same as CAMS",
        better_label="lower error than CAMS",
        figure_planning="exploratory",
    )
    return figure(
        panels=[panel],
        number=3,
        figure_planning="exploratory",
        title="A simple mean or a linear stack against CAMS, and the negative control",
        subtitle=[
            DOTS,
            (
                "The negative control adds a noised copy of CAMS's own column, which should carry "
                "almost no independent information."
            ),
            CAPACITY,
            SCOPE,
        ],
    )


def main() -> int:
    """Read the report and the losses, and write the three SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = OUTPUT_DIR / "report.md"
    contrasts = report_contrasts(report_path=report_path)
    losses = pl.read_parquet(OUTPUT_DIR / "losses.parquet").filter(pl.col("setting") == "pooled")
    charts = {
        "satellite_blend_leaderboard": _leaderboard(losses=losses),
        "satellite_blend_headline": _headline(contrasts=contrasts),
        "satellite_blend_exploratory": _exploratory(contrasts=contrasts),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
