"""Draw the two charts for the write-up on whether CERRA's wind levels beat its 100 m wind alone.

One-off throwaway script for the charts of
<https://github.com/openclimatefix/nged-substation-forecast/issues/957>. **Every number a chart
shares with the page is read from `intervals.parquet` and `absolute.parquet`, and checked against
the `report.md` beside each**, so a chart cannot disagree with the page.

Generators appear only as `W1` to `W3`, and no chart carries a calendar date. Every mark is drawn
with `aria=False`.

**Do not run this script until `cerra_wind_levels.py` and `cerra_wind_levels_shear.py` have
written their reports.**

Run it with `uv run python studies/beam_diffuse_split/cerra_wind_levels_charts.py`. Optimise each
SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing it.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from cerra_wind_levels import OUTPUT_DIR as MAIN_DIR
from cerra_wind_levels import PRIMARY_SETTING, SENSITIVITY_SETTING
from cerra_wind_levels_shear import OUTPUT_DIR as SHEAR_DIR
from studies.charts import (
    POST_HOC_SUFFIX,
    figure,
    interval_panel,
    leaderboard_panel,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

NAMES: Final[dict[str, str]] = {
    "speed_10m": "10 m speed",
    "speed_100m": "100 m speed",
    "speed_10m_100m": "10 m and 100 m speeds",
    "mean_near_100m": "Mean of 75, 100, and 150 m speeds",
    "levels_50_to_150": "50, 75, 100, and 150 m speeds",
    "levels_all": "All five heights",
    "speed_100m_noise": "Negative control: 100 m speed plus 4 shuffled columns",
}
"""Each arm's public name, as the page writes it."""

FAMILY: Final = "reanalysis"
"""Every arm reads CERRA, so every row takes the reanalysis colour."""

CONTRAST_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("speed_100m", "speed_10m", "planned"),
    ("mean_near_100m", "speed_100m", "planned"),
    ("levels_50_to_150", "speed_100m", "planned"),
    ("levels_all", "levels_50_to_150", "planned"),
    ("speed_10m_100m", "speed_100m", "exploratory"),
    ("levels_50_to_150", "speed_10m_100m", "post hoc"),
    ("levels_all", "speed_10m_100m", "post hoc"),
    ("speed_10m_100m", "speed_10m", "post hoc"),
    ("levels_50_to_150", "mean_near_100m", "post hoc"),
    ("speed_100m_noise", "speed_100m", "control"),
)
"""The contrasts Figure 2 draws, each with how the page labels it."""

DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
CAPACITY: Final[str] = "Errors are a fraction of each wind farm's capacity."
SCOPE: Final[str] = "Three wind farms, September 2019 to June 2026, on CERRA's 3-hourly analysis."
ERROR_X_TITLE: Final[str] = "Mean absolute error (points of capacity; smaller is better)"
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
FIGURE_1: Final[int] = 1
FIGURE_2: Final[int] = 2
FIGURE_3: Final[int] = 3
BINS: Final[int] = 20
"""The number of equal-width bins on each axis of Figure 3."""
FOUR_HEIGHTS: Final[str] = "levels_50_to_150"
DOMAIN_MARGIN: Final[float] = 0.1


def _check_printed(*, report: str, texts: list[str]) -> None:
    """Stop unless every formatted number appears in the report as printed.

    Args:
        report: The report's text.
        texts: The table-row fragments to look for.

    Raises:
        ValueError: If any fragment is missing.
    """
    missing = [text for text in texts if text not in report]
    if missing:
        msg = f"{len(missing)} numbers are not in report.md as printed, such as {missing[:3]}"
        raise ValueError(msg)


def _contrast_fragment(*, row: dict[str, float | str]) -> str:
    """Format the leading cells of a contrast row as the reports print them.

    Args:
        row: A row of `intervals.parquet`.

    Returns:
        The report's text from the contrast to the unadjusted interval.
    """
    return (
        f"| {row['treatment']} − {row['reference']} | {row['treatment_mae_pp']:.3f} "
        f"| {row['reference_mae_pp']:.3f} | {row['difference_pp']:+.3f} "
        f"| [{row['lower_95_pp']:+.3f}, {row['upper_95_pp']:+.3f}] |"
    )


def _leaderboard(*, absolute: pl.DataFrame, report: str) -> alt.VConcatChart:
    """Draw every arm's own error, best first.

    Args:
        absolute: `absolute.parquet`.
        report: `cerra_wind_levels.py`'s `report.md`.

    Returns:
        Figure 1.
    """
    scored = absolute.filter(
        pl.col("setting") == PRIMARY_SETTING, pl.col("scope") == "all", pl.col("arm").is_in(NAMES)
    ).sort("mae_pp")
    _check_printed(
        report=report,
        texts=[
            f"| {row['arm']} | {row['mae_pp']:.3f} "
            f"| [{row['lower_95_pp']:.3f}, {row['upper_95_pp']:.3f}] |"
            for row in scored.iter_rows(named=True)
        ],
    )
    rows = scored.select(
        label=pl.col("arm").replace_strict(NAMES),
        family=pl.lit(FAMILY),
        value="mae_pp",
        lower_95="lower_95_pp",
        upper_95="upper_95_pp",
    )
    panel = leaderboard_panel(
        rows=rows,
        x_domain=(7.0, 9.0),
        x_title=ERROR_X_TITLE,
        keys=False,
    )
    return figure(
        panels=[panel],
        number=FIGURE_1,
        figure_planning=None,
        title="The seven sets of CERRA wind columns span 0.5 points of pooled error",
        subtitle=[
            "Mean absolute error of each XGBoost model's estimate of a wind farm's hourly power.",
            f"{DOTS} {CAPACITY}",
            (
                "The intervals overlap because months of weather dominate them, which does not "
                "mean the sets are equal. Figure 2 compares the sets on the same months."
            ),
            SCOPE,
        ],
    )


def _contrasts(*, shear: pl.DataFrame, reports: str) -> alt.VConcatChart:
    """Draw the paired contrasts, planned, exploratory, and post hoc, at both settings.

    Args:
        shear: `cerra_wind_levels_shear.py`'s `intervals.parquet`, which holds both scripts'
            intervals.
        reports: `cerra_wind_levels_shear.py`'s `report.md`, which holds both reports.

    Returns:
        Figure 2.
    """
    both = shear.filter(
        pl.col("scope") == "all", pl.col("setting").is_in([PRIMARY_SETTING, SENSITIVITY_SETTING])
    )
    rows = []
    for treatment, reference, kind in CONTRAST_ROWS:
        pair = both.filter(pl.col("treatment") == treatment, pl.col("reference") == reference)
        primary = pair.filter(pl.col("setting") == PRIMARY_SETTING).row(0, named=True)
        second = pair.filter(pl.col("setting") == SENSITIVITY_SETTING).row(0, named=True)
        _check_printed(
            report=reports,
            texts=[_contrast_fragment(row=primary), _contrast_fragment(row=second)],
        )
        suffix = POST_HOC_SUFFIX if kind == "post hoc" else ""
        rows.append(
            {
                "label": f"{NAMES[treatment]} − {NAMES[reference]}{suffix}",
                "treatment": treatment,
                "reference": reference,
                "difference": primary["difference_pp"],
                "lower_95": primary["lower_95_pp"],
                "upper_95": primary["upper_95_pp"],
                "second_difference": second["difference_pp"],
                "family": FAMILY,
                "planned": kind == "planned",
            }
        )
    frame = pl.DataFrame(rows)
    domain = (
        min(0.0, *frame["lower_95"].to_list()) - DOMAIN_MARGIN,
        max(0.0, *frame["upper_95"].to_list()) + DOMAIN_MARGIN,
    )
    panel = interval_panel(
        rows=frame,
        x_domain=domain,
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first set better",
        panel_title="Paired differences, primary setting",
        figure_planning="mixed",
    )
    return figure(
        panels=[panel],
        number=FIGURE_2,
        figure_planning="mixed",
        post_hoc=True,
        title="A second height gives most of the gain from using more CERRA wind heights",
        subtitle=[
            "Each row: the first set of columns minus the second set, on the same rows.",
            f"{DOTS} Hollow triangle: the second hyperparameter setting. {CAPACITY}",
            SCOPE,
        ],
    )


def _predictions(*, main_dir: Path) -> alt.VConcatChart:
    """Draw where the four-height XGBoost model's out-of-fold estimates fall against measured power.

    Each estimate is the measured power plus the saved signed error, at the first seed and the
    primary setting. Both axes are fractions of the farm's capacity, counted in equal bins.

    Args:
        main_dir: The folder holding `cerra_wind_levels.py`'s `rows.parquet` and `losses.parquet`.

    Returns:
        Figure 3.
    """
    measured = pl.read_parquet(main_dir / "rows.parquet").select("site", "time", "power_mw")
    joined = (
        pl.read_parquet(main_dir / "losses.parquet")
        .filter(
            pl.col("arm") == FOUR_HEIGHTS,
            pl.col("setting") == PRIMARY_SETTING,
            pl.col("target") == "power_mw",
            pl.col("seed") == 0,
        )
        .join(measured, on=["site", "time"], validate="1:1")
        .with_columns(
            measured=pl.col("power_mw") / pl.col("effective_capacity_mw"),
            estimated=(pl.col("power_mw") + pl.col("signed_error_mw"))
            / pl.col("effective_capacity_mw"),
        )
    )
    bins = (
        joined.with_columns(
            measured_bin=(pl.col("measured").clip(0.0, 1.0) * BINS).floor().clip(0, BINS - 1),
            estimated_bin=(pl.col("estimated").clip(0.0, 1.0) * BINS).floor().clip(0, BINS - 1),
        )
        .group_by("site", "measured_bin", "estimated_bin")
        .agg(count=pl.len())
        .with_columns(share=pl.col("count") / pl.col("count").sum().over("site"))
        .with_columns(
            measured_low=pl.col("measured_bin") / BINS,
            measured_high=(pl.col("measured_bin") + 1) / BINS,
            estimated_low=pl.col("estimated_bin") / BINS,
            estimated_high=(pl.col("estimated_bin") + 1) / BINS,
        )
    )
    facets = [
        alt.Chart(bins.filter(pl.col("site") == farm))
        .mark_rect(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("measured_low:Q", title="Measured power (fraction of capacity)").scale(
                domain=[0, 1]
            ),
            x2="measured_high:Q",
            y=alt.Y("estimated_low:Q", title="Estimated power (fraction of capacity)").scale(
                domain=[0, 1]
            ),
            y2="estimated_high:Q",
            color=alt.Color("share:Q", legend=None).scale(
                type="sqrt", range=["#ffffff", ocf.DATA_BLUE]
            ),
        )
        .properties(width=190, height=190, title=farm)
        for farm in sorted(bins["site"].unique().to_list())
    ]
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(aria=False, color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    panel = alt.hconcat(*(facet + diagonal for facet in facets), spacing=16)
    return figure(
        panels=[panel],
        number=FIGURE_3,
        figure_planning=None,
        title="The four-height XGBoost model's estimates follow measured power at each farm",
        subtitle=[
            (
                "Each cell counts hours in which an XGBoost model, given the 50, 75, 100, and "
                "150 m speeds and not trained on that hour's month, made that estimate. Darker "
                "cells hold more hours. Dashed line: estimate equals measurement."
            ),
            "Primary setting, first fitting seed. Power is a fraction of each farm's capacity.",
            SCOPE,
        ],
    )


def main() -> int:
    """Check both reports against the saved intervals, then write the two SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    main_report = (MAIN_DIR / "report.md").read_text()
    shear_report = (SHEAR_DIR / "report.md").read_text()
    charts = {
        "cerra_wind_levels_leaderboard": _leaderboard(
            absolute=pl.read_parquet(MAIN_DIR / "absolute.parquet"), report=main_report
        ),
        "cerra_wind_levels_predictions": _predictions(main_dir=MAIN_DIR),
        "cerra_wind_levels_contrasts": _contrasts(
            shear=pl.read_parquet(SHEAR_DIR / "intervals.parquet"), reports=shear_report
        ),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
