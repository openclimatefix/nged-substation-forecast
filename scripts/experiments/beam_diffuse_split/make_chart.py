"""Draw the anonymised result chart for the beam/diffuse experiment.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

The chart carries the two findings the experiment has to keep apart. The left panel measures every
arm against the arm shown global horizontal irradiance alone, which is what *having* a split buys.
The right panel measures the arm shown the product's own published beam against the arm shown a
separation model's estimate of the same beam, which is what the *published field* buys on top of
what global irradiance already implies. Both panels are drawn in the same units, a percentage of
the global-only arm's mean absolute error, so the two effects can be read against each other.

Each row of panels is one instrument, because a tree and a fitted physical model disagree about how
much of the split they can use, and that disagreement is part of the answer.

Sites are pooled here and no identifier reaches the chart, because a metered generator's output is
commercially sensitive and this repo is public.

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

REPO_DATA_DIR: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data")

OUTPUT_PATH: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse_split_result.svg"

ALIGNMENT: Final[str] = "shifted"
"""Which stamp alignment the chart draws.

The two alignments agree on every contrast the chart shows, so drawing both would double the rows
and say nothing. The tables report both.
"""

SOURCE_LABELS: Final[dict[str, str]] = {
    "open-meteo": "ERA5 (31 km reanalysis)",
    "cams": "CAMS (5 km satellite retrieval)",
}
"""Source keys to the labels a reader sees."""

INSTRUMENT_LABELS: Final[dict[str, str]] = {
    "xgboost": "XGBoost",
    "physics": "Fitted physical PV model",
}
"""Instrument keys to the labels a reader sees."""

REFERENCE_ARMS: Final[dict[str, str]] = {
    "xgboost": "A_global_only",
    "physics": "P_A_global_only",
}
"""The arm each instrument's percentages are measured against."""

CONTRAST_LABELS: Final[dict[str, str]] = {
    "B_erbs|A_global_only": "Erbs split",
    "B_disc|A_global_only": "DISC split",
    "C_era5_split|A_global_only": "The product's own split",
    "D_direct_fraction|A_global_only": "The product's direct fraction",
    "C_era5_split|B_erbs": "The product's split vs. Erbs",
    "P_B_erbs|P_A_global_only": "Erbs split",
    "P_B_disc|P_A_global_only": "DISC split",
    "P_C_source_split|P_A_global_only": "The product's own split",
    "P_C_source_split|P_B_erbs": "The product's split vs. Erbs",
}
"""Each plotted contrast, keyed by `treatment|reference`, to the label a reader sees."""

PANEL_TITLES: Final[tuple[str, str]] = (
    "Against global irradiance alone",
    "The product's split against Erbs's",
)
"""The two panels, in the order they are drawn."""

ROW_ORDER: Final[tuple[str, ...]] = (
    "Erbs split",
    "DISC split",
    "The product's own split",
    "The product's direct fraction",
    "The product's split vs. Erbs",
)
"""The order the contrasts are stacked in, best-known to least-known."""

SUBTITLE: Final[tuple[str, ...]] = (
    "Six PV sites in one 34 km box in Lincolnshire, hourly daylight rows, 2019-2026.",
    "Negative is better. Bars are 95% monthly block bootstrap intervals;",
    "a bar crossing zero has not been shown to help. Reanalysis and satellite",
    "retrieval, not forecasts, so this is information content, not forecast skill.",
)

PERCENTAGE_POINTS: Final[float] = 100.0


def _reference_mean_absolute_error(*, instrument: str, source: str) -> float:
    """Return the global-only arm's pooled mean absolute error in MW, weighted by row count."""
    stem = "results" if instrument == "xgboost" else "physics"
    summary = pl.read_parquet(
        REPO_DATA_DIR
        / "ERA5"
        / f"beam_diffuse_{stem}_{source}_{ALIGNMENT}"
        / "per_site_summary.parquet"
    ).filter((pl.col("setting") == "primary") & (pl.col("arm") == REFERENCE_ARMS[instrument]))
    return float((summary["mae_mw"] * summary["n_rows"]).sum()) / float(summary["n_rows"].sum())


def _differences() -> pl.DataFrame:
    """Collect every plotted contrast from every instrument and source.

    Dividing an interval by a constant is still an interval for the scaled quantity, so expressing
    each difference as a percentage of the reference arm's error needs no second bootstrap.

    Returns:
        One row per (instrument, source, contrast) ready to plot.
    """
    frames: list[pl.DataFrame] = []
    for instrument in INSTRUMENT_LABELS:
        stem = "results" if instrument == "xgboost" else "physics"
        for source in SOURCE_LABELS:
            results_dir = REPO_DATA_DIR / "ERA5" / f"beam_diffuse_{stem}_{source}_{ALIGNMENT}"
            if not results_dir.exists():
                continue
            reference_mae = _reference_mean_absolute_error(instrument=instrument, source=source)
            percent = PERCENTAGE_POINTS / reference_mae
            intervals = pl.read_parquet(results_dir / "bootstrap_intervals.parquet").filter(
                (pl.col("setting") == "primary")
                & (pl.col("metric") == "absolute_error_mw")
                & (pl.col("scope") == "all_sites")
            )
            frames.append(
                intervals.with_columns(
                    key=pl.col("treatment") + pl.lit("|") + pl.col("reference"),
                    instrument_label=pl.lit(INSTRUMENT_LABELS[instrument]),
                    source_label=pl.lit(SOURCE_LABELS[source]),
                    difference_percent=pl.col("difference") * percent,
                    lower_95_percent=pl.col("lower_95") * percent,
                    upper_95_percent=pl.col("upper_95") * percent,
                )
                .filter(pl.col("key").is_in(list(CONTRAST_LABELS)))
                .select(
                    "instrument_label",
                    "source_label",
                    "difference_percent",
                    "lower_95_percent",
                    "upper_95_percent",
                    contrast_label=pl.col("key").replace_strict(CONTRAST_LABELS),
                )
            )
    combined = pl.concat(frames)
    return combined.with_columns(
        panel=pl.when(pl.col("contrast_label") == ROW_ORDER[-1])
        .then(pl.lit(PANEL_TITLES[1]))
        .otherwise(pl.lit(PANEL_TITLES[0]))
    )


def _chart(*, differences: pl.DataFrame) -> alt.FacetChart:
    """Build the faceted point-and-interval chart."""
    base = alt.Chart(differences).encode(
        y=alt.Y("contrast_label:N", title=None, sort=list(ROW_ORDER)),
        yOffset=alt.YOffset("source_label:N", sort=list(SOURCE_LABELS.values())),
        color=alt.Color(
            "source_label:N", title="Irradiance source", sort=list(SOURCE_LABELS.values())
        ),
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
        alt.Chart(differences)
        .mark_rule(strokeDash=[4, 4], color="#292B2B")
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.datum(0)
        )
    )
    return (
        (zero + interval + estimate)
        .properties(width=330, height=alt.Step(14))
        .facet(
            column=alt.Column("panel:N", title=None, sort=list(PANEL_TITLES)),
            row=alt.Row("instrument_label:N", title=None, sort=list(INSTRUMENT_LABELS.values())),
        )
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                text="Does a weather product's own beam/diffuse split help a PV model?",
                subtitle=SUBTITLE,
            )
        )
    )


def main() -> int:
    """Write the chart as an SVG."""
    differences = _differences()
    _LOG.info("plotting %d contrasts", differences.height)
    _chart(differences=differences).save(OUTPUT_PATH)
    _LOG.info("wrote %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
