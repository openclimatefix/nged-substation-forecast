"""Draw the anonymised charts for the write-up on blending CAMS and SARAH-3.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/860>.

**Every interval and every error a chart shares with the page is read from the report
`blend_satellites.py` wrote**, so a chart cannot disagree with the page. The leaderboard also draws
each arm's own absolute error, bootstrapped straight from `losses.parquet` with no refit, the same
month-and-seed resampling the report's contrasts use.

Generators appear only as `A` to `F`, and every output is a fraction of the generator's own
capacity. Follow the style of `blend_product_charts.py`. Figures are numbered 12 to 14, continuing
`docs/studies/blending-weather-products.md`'s own Figures 1 to 11.

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
    ENRICHED_CONTRASTS,
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
    "cams_split_sarah3_control": "CAMS (split) + SARAH-3 (control)",
    "cams_sarah3_xgb": "CAMS + SARAH-3 (global, XGBoost)",
    "cams_sarah3_control": "CAMS + SARAH-3 (global, control)",
    "cams_sarah3_mean": "CAMS + SARAH-3 (mean)",
    "cams_sarah3_stack": "CAMS + SARAH-3 (stack)",
    "cams_sarah3_equal": "CAMS + SARAH-3 (equal weight)",
    "cams_cams_noise_xgb": "CAMS + noised CAMS (negative control)",
    "cams_cams_noise5_xgb": "CAMS + noised CAMS, 5 W/m2 (negative control)",
    "cams_rich": "CAMS (split + neighbouring hours)",
    "cams_rich_sarah3_xgb": "CAMS (split + neighbouring hours) + SARAH-3 (XGBoost)",
    "cams_rich_sarah3_control": "CAMS (split + neighbouring hours) + SARAH-3 (control)",
}
"""Each arm's name as the page writes it. Every blend names both products it reads."""

SECTION_PLANNED: Final[str] = "Planned contrasts"
SECTION_SENSITIVITY: Final[str] = "Planned contrasts at the second hyperparameter setting"
SECTION_ENRICHED: Final[str] = (
    "Post hoc: CAMS with its own neighbouring hours, against the enriched reference"
)
SECTION_ENRICHED_SENSITIVITY: Final[str] = (
    "Post hoc, enriched reference, at the second hyperparameter setting"
)
SECTION_NEGATIVE_CONTROL: Final[str] = (
    "Negative controls: CAMS plus a noised copy of itself, against CAMS alone"
)
SECTION_METHODS: Final[str] = (
    "Exploratory: the all-global blend, the mean, stack and equal blends, and the same-reference "
    "comparisons the lead needs"
)

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months and a seed."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, January 2021 to August 2026."
LEADERBOARD_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
CONTRAST_X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
ZERO_LABEL: Final[str] = "same as the reference arm"
BETTER_LABEL: Final[str] = "lower error than the reference arm"

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
        Figure 12.
    """
    rows = _leaderboard_rows(losses=losses)
    panel = leaderboard_panel(
        rows=rows, x_domain=_leaderboard_domain(rows=rows), x_title=LEADERBOARD_X_TITLE
    )
    return figure(
        panels=[panel],
        number=12,
        figure_planning=None,
        title="Every arm's own mean absolute error, CAMS and SARAH-3 and their blends",
        subtitle=[
            "Each arm's own mean absolute error, sorted best first.",
            DOTS,
            (
                "The intervals are wide mainly because every arm's error swings together from "
                "month to month, a swing the contrasts in Figures 13 and 14 cancel."
            ),
            CAPACITY,
            SCOPE,
        ],
    )


def _headline(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw the enriched-reference and plain-reference contrasts, at both hyperparameter settings.

    Leads with the post hoc `cams_rich` reference the first science review asked for, and keeps
    the two originally planned contrasts against plain `cams_split` beside it, each row's label
    naming its own reference so the two are never confused.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 13.
    """
    sections = (
        (SECTION_ENRICHED, SECTION_ENRICHED_SENSITIVITY, ENRICHED_CONTRASTS, False),
        (SECTION_PLANNED, SECTION_SENSITIVITY, PLANNED_CONTRASTS, True),
    )
    frames = []
    for primary_section, sensitivity_section, contrast_pairs, planned in sections:
        labels = [f"{NAMES[t]} − {NAMES[r]}" for t, r in contrast_pairs]
        primary = select_contrasts(
            contrasts=contrasts,
            wanted=[ContrastKey(primary_section, "all", t, r) for t, r in contrast_pairs],
        )
        sensitivity = select_contrasts(
            contrasts=contrasts,
            wanted=[ContrastKey(sensitivity_section, "all", t, r) for t, r in contrast_pairs],
        )
        frames += [
            frame.select("difference", "lower_95", "upper_95").with_columns(
                label=pl.Series(labels),
                family=pl.lit("satellite"),
                planned=pl.lit(value=planned),
                condition=pl.lit(condition),
            )
            for frame, condition in ((primary, "Primary setting"), (sensitivity, "Second setting"))
        ]
    rows = pl.concat(frames)
    panel = interval_panel(
        rows=rows,
        x_domain=_contrast_domain(rows=rows),
        x_title=CONTRAST_X_TITLE,
        zero_label=ZERO_LABEL,
        better_label=BETTER_LABEL,
        conditions=("Primary setting", "Second setting"),
        condition_title="Hyperparameter setting",
        figure_planning=planning(rows=[rows]),
    )
    return figure(
        panels=[panel],
        number=13,
        figure_planning=planning(rows=[rows]),
        title=(
            "Does CAMS's split plus SARAH-3 beat CAMS's split, with and without CAMS's own "
            "neighbouring hours, and does the gain survive a climatology control?"
        ),
        subtitle=[
            (
                "Rows against plain CAMS (split) are the planned comparisons. Rows against CAMS "
                "(split + neighbouring hours) are post hoc, added after the first science review."
            ),
            DOTS,
            CAPACITY,
            SCOPE,
        ],
    )


def _exploratory(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw the all-global blend, the mean/stack/equal blends, and both negative controls.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 14.
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
        zero_label=ZERO_LABEL,
        better_label=BETTER_LABEL,
        figure_planning="exploratory",
    )
    return figure(
        panels=[panel],
        number=14,
        figure_planning="exploratory",
        title=(
            "The all-global blend, a simple mean or a linear stack, both negative controls, and "
            "the same-reference comparisons the lead needs"
        ),
        subtitle=[
            "Every row's label names its own treatment and reference arm.",
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
