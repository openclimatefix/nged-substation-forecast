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
    EXPLORATORY_METHOD_CONTRASTS,
    METRIC,
    NEGATIVE_CONTROLS,
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
    "cams_split": "CAMS (split)",
    "cams_split_sarah3_xgb": "CAMS (split) + SARAH-3 (XGBoost)",
    "cams_split_sarah3_control": "CAMS (split) + SARAH-3 (climatology control)",
    "cams_sarah3_xgb": "CAMS + SARAH-3, both global (XGBoost)",
    "cams_sarah3_control": "CAMS + SARAH-3, both global (climatology control)",
    "cams_sarah3_mean": "CAMS + SARAH-3 (mean)",
    "cams_sarah3_stack": "CAMS + SARAH-3 (stack)",
    "cams_sarah3_equal": "CAMS + SARAH-3 (equal weight)",
    "cams_cams_noise_xgb": "CAMS + noised CAMS (negative control)",
    "cams_cams_noise5_xgb": "CAMS + noised CAMS, 5 W/m2 (negative control)",
}
"""Each arm's name as the page writes it. Every blend names both products it reads."""

SECTION_PLANNED: Final[str] = "Planned contrasts"
SECTION_SENSITIVITY: Final[str] = "Planned contrasts at the second hyperparameter setting"
SECTION_NEGATIVE_CONTROL: Final[str] = (
    "Negative controls: CAMS plus a noised copy of itself, against CAMS alone"
)
SECTION_METHODS: Final[str] = (
    "Exploratory: the all-global blend, and the mean, stack and equal blends, against CAMS"
)

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months and a seed."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, January 2021 to August 2026."
LEADERBOARD_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
CONTRAST_X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"

LEADERBOARD_DOMAIN_MARGIN_FRACTION: Final[float] = 0.1
"""The margin added on each side of the leaderboard's data-derived x domain, as a fraction of the
span every arm's 95% interval covers, so no dot or line touches the axis edge."""

CONTRAST_DOMAIN_MARGIN_FRACTION: Final[float] = 0.1
"""The margin added on each side of a contrast panel's data-derived x domain, as a fraction of the
largest interval bound, so no dot or line touches the axis edge."""


def _contrast_domain(*, rows: pl.DataFrame) -> tuple[float, float]:
    """Return a symmetric x domain covering every row's 95% interval, with a margin either side.

    Derived from the data rather than fixed by hand, so a change to the contrasts fitted, or to
    their intervals, can never clip a dot or an interval line off the edge of the axis.

    Args:
        rows: The contrast rows, carrying `lower_95` and `upper_95`.

    Returns:
        A domain symmetric about zero, so "same as the reference arm" sits at the axis centre.
    """
    largest = float(
        rows.select(
            pl.max_horizontal(pl.col("lower_95").abs(), pl.col("upper_95").abs()).max()
        ).item()
    )
    bound = largest * (1 + CONTRAST_DOMAIN_MARGIN_FRACTION)
    return -bound, bound


def _leaderboard_domain(*, rows: pl.DataFrame) -> tuple[float, float]:
    """Return an x domain covering every row's 95% interval, with a margin either side.

    Derived from the data rather than fixed by hand, so a change to the arms fitted, or to their
    errors, can never clip a dot or an interval line off the edge of the axis.

    Args:
        rows: The leaderboard rows, carrying `lower_95` and `upper_95`.

    Returns:
        The domain, its lower bound never below 0 since this axis is an absolute error.
    """
    lower = float(rows.select(pl.col("lower_95").min()).item())
    upper = float(rows.select(pl.col("upper_95").max()).item())
    margin = (upper - lower) * LEADERBOARD_DOMAIN_MARGIN_FRACTION
    return max(0.0, lower - margin), upper + margin


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
    panel = leaderboard_panel(
        rows=rows, x_domain=_leaderboard_domain(rows=rows), x_title=LEADERBOARD_X_TITLE
    )
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
        x_domain=_contrast_domain(rows=rows),
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
    """Draw the all-global blend, the mean, stack and equal blends, and both negative controls.

    All against CAMS.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 3.
    """
    methods = select_contrasts(
        contrasts=contrasts,
        wanted=[ContrastKey(SECTION_METHODS, "all", t, r) for t, r in EXPLORATORY_METHOD_CONTRASTS],
    )
    controls = select_contrasts(
        contrasts=contrasts,
        wanted=[ContrastKey(SECTION_NEGATIVE_CONTROL, "all", t, r) for t, r in NEGATIVE_CONTROLS],
    )
    labels = [
        f"{NAMES[t]} − {NAMES[r]}" for t, r in (*EXPLORATORY_METHOD_CONTRASTS, *NEGATIVE_CONTROLS)
    ]
    rows = (
        pl.concat([methods, controls])
        .select("difference", "lower_95", "upper_95")
        .with_columns(
            label=pl.Series(labels), family=pl.lit("satellite"), planned=pl.lit(value=False)
        )
    )
    panel = interval_panel(
        rows=rows,
        x_domain=_contrast_domain(rows=rows),
        x_title=CONTRAST_X_TITLE,
        zero_label="same as CAMS",
        better_label="lower error than CAMS",
        figure_planning="exploratory",
    )
    return figure(
        panels=[panel],
        number=3,
        figure_planning="exploratory",
        title="The all-global blend, a simple mean or a linear stack, and both negative controls",
        subtitle=[
            DOTS,
            (
                "Each negative control adds a noised copy of CAMS's own column, which should "
                "carry almost no independent information."
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
