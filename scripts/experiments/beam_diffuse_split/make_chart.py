"""Draw the anonymised result chart for the beam/diffuse experiment.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

The chart shows, for each arm and each site, how much the arm's mean absolute error differs from
the global-irradiance-only arm's, as a percentage of that arm's error, with the monthly block
bootstrap's 95% interval. The zero rule is the whole point: an arm whose interval crosses it has
not been shown to carry information the tree can use.

Sites are labelled `A`–`F` and no identifier reaches the chart, because a metered generator's output
is commercially sensitive and this repo is public.

Run it with `uv run --with vl-convert-python python
scripts/experiments/beam_diffuse_split/make_chart.py`.
"""

import logging
import sys
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme  # noqa: F401  (importing registers and enables the OCF theme)
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("make_chart")

RESULTS_DIR: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/ERA5/beam_diffuse_results_cds"
)
"""The Copernicus run is what the published chart shows; the mirror's run is reported as text."""

OUTPUT_PATH: Final[Path] = RESULTS_DIR / "beam_diffuse_split_result.svg"

ARM_LABELS: Final[dict[str, str]] = {
    "B_erbs": "B — Erbs separation model",
    "B_disc": "B-DISC — DISC separation model",
    "C_era5_split": "C — ERA5's own beam/diffuse split",
    "D_direct_fraction": "D — ERA5 direct fraction",
}
"""Arm keys to the labels a reader sees. Arm A is the reference and has no row of its own."""

SCOPE_LABELS: Final[dict[str, str]] = {"all_sites": "All six sites"}
"""Scope keys needing a label other than the site letter itself."""

SUBTITLE: Final[tuple[str, ...]] = (
    "Six PV sites in one 34 km box in Lincolnshire, hourly daylight rows, 2019-2026.",
    "Negative is better. Bars are 95% monthly block bootstrap intervals;",
    "an arm whose bar crosses zero has not been shown to help.",
)


def _reference_mean_absolute_errors(*, summary: pl.DataFrame, setting: str) -> pl.DataFrame:
    """Return the reference arm's mean absolute error for every scope, in MW.

    The pooled figure is weighted by row count rather than by site, so it matches the pooled
    bootstrap, which resamples rows through whole months.

    Args:
        summary: The per-site summary `run_experiment.py` wrote.
        setting: Which hyperparameter setting to read.

    Returns:
        One row per scope with `reference_mae_mw`.
    """
    reference = summary.filter((pl.col("setting") == setting) & (pl.col("arm") == "A_global_only"))
    per_site = reference.select(scope=pl.col("site"), reference_mae_mw=pl.col("mae_mw"))
    weighted_total = float((reference["mae_mw"] * reference["n_rows"]).sum())
    row_total = float(reference["n_rows"].sum())
    pooled = pl.DataFrame(
        {"scope": ["all_sites"], "reference_mae_mw": [weighted_total / row_total]}
    )
    return pl.concat([pooled, per_site])


def _relative_differences(*, setting: str) -> pl.DataFrame:
    """Express every bootstrap interval as a percentage of the reference arm's error.

    Dividing an interval by a constant is still an interval for the scaled quantity, so the
    percentage interval needs no second bootstrap.

    Args:
        setting: Which hyperparameter setting to read.

    Returns:
        One row per (arm, scope) ready to plot.
    """
    intervals = pl.read_parquet(RESULTS_DIR / "bootstrap_intervals.parquet").filter(
        (pl.col("setting") == setting) & (pl.col("metric") == "absolute_error_mw")
    )
    summary = pl.read_parquet(RESULTS_DIR / "per_site_summary.parquet")
    reference = _reference_mean_absolute_errors(summary=summary, setting=setting)

    percent = 100.0 / pl.col("reference_mae_mw")
    return (
        intervals.join(reference, on="scope")
        .with_columns(
            difference_percent=pl.col("difference") * percent,
            lower_95_percent=pl.col("lower_95") * percent,
            upper_95_percent=pl.col("upper_95") * percent,
            arm_label=pl.col("arm").replace_strict(ARM_LABELS),
            scope_label=pl.col("scope").replace(SCOPE_LABELS),
        )
        .filter(pl.col("arm").is_in(list(ARM_LABELS)))
    )


def _chart(*, differences: pl.DataFrame) -> alt.LayerChart:
    """Build the layered point-and-interval chart.

    Args:
        differences: The frame `_relative_differences` returned.

    Returns:
        The chart, ready to save.
    """
    scope_order = ["All six sites", *sorted(set(differences["scope_label"]) - {"All six sites"})]
    arm_order = list(ARM_LABELS.values())
    base = alt.Chart(differences).encode(
        y=alt.Y("scope_label:N", title=None, sort=scope_order),
        yOffset=alt.YOffset("arm_label:N", sort=arm_order),
        color=alt.Color("arm_label:N", title="Arm", sort=arm_order),
    )
    interval = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]  # astral-sh/ty#2520
        x=alt.X(
            "lower_95_percent:Q",
            title="Change in mean absolute error vs. global irradiance alone (%)",
        ),
        x2=alt.X2("upper_95_percent:Q"),
    )
    estimate = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference_percent:Q")
    )
    zero = (
        alt.Chart(differences).mark_rule(strokeDash=[4, 4], color="#292B2B").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    )
    return (
        (zero + interval + estimate)
        .properties(
            width=520,
            height=alt.Step(13),
            title=alt.TitleParams(
                text="Does ERA5's beam/diffuse split help an XGBoost PV forecast?",
                subtitle=SUBTITLE,
            ),
        )
        .resolve_scale(color="shared")
    )


def main() -> int:
    """Write the chart as an SVG."""
    differences = _relative_differences(setting="primary")
    _chart(differences=differences).save(OUTPUT_PATH)
    _LOG.info("wrote %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
