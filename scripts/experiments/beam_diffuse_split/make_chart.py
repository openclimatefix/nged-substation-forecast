"""Draw the anonymised result chart for the beam/diffuse experiment.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

The chart carries the two findings the experiment has to keep apart. The left panel measures every
arm against the arm shown global horizontal irradiance alone, which is what *having* a split buys.
The right panel measures the arm shown the product's own published beam against the arm shown a
separation model's estimate of the same beam, which is what the *published field* buys on top of
what global irradiance already implies. Both panels are drawn in the same units, a percentage of
the global-only arm's mean absolute error, but **each panel carries its own x scale**, because the
right panel's effects are an order of magnitude smaller than the left panel's and a shared scale
would flatten them to nothing. Compare bars within a panel, and read the numbers rather than the
lengths across panels.

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
from sources import REPO_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("make_chart")


OUTPUT_PATH: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse_split_result.svg"

ALIGNMENT: Final[str] = "piecewise"
"""Which stamp alignment the chart draws.

NGED corrected the half-hourly power stamps at 2026-03-26 08:30 UTC, so readings before that
instant are 30 minutes late and readings from it are correct. Only the piecewise reading matches
the feed, and every figure and every headline number here is drawn under it.

Getting the alignment right matters more for the fitted physical model than for the tree. Under
the uncorrected stamps the physical model's headline contrast on the reanalysis changes sign,
because a 30-minute error is absorbed into the fitted azimuth and the arms then differ in geometry
as well as in beam field. The tree's contrasts hold their sign under every reading.
"""

SOURCE_LABELS: Final[dict[str, str]] = {
    "open-meteo": "ERA5 (31 km reanalysis)",
    "ukv": "UKV (2 km model analysis)",
    "icon-d2": "ICON-D2 (2 km model analysis)",
    "cams": "CAMS (5 km satellite retrieval)",
}
"""Source keys to the labels a reader sees, the model sources before the retrieval.

**A source missing from this mapping is drawn nowhere**, which is why
`_raise_on_unlabelled_sources` stops on a results directory named by neither this mapping nor
`UNDRAWN_SOURCES`, rather than letting the chart come out looking complete with an arm silently
absent from it.
"""

UNDRAWN_SOURCES: Final[tuple[str, ...]] = ("cds",)
"""Sources with results on disk that the chart deliberately leaves out.

`cds` is the Copernicus route to the same ERA5 fields `open-meteo` serves, checked against each
other by `verify_era5_sources.py`, so drawing both would put one reanalysis on the chart twice
under two names. Naming the exclusion here is what lets the guard below tell a deliberate omission
from a forgotten omission.
"""

INSTRUMENT_LABELS: Final[dict[str, str]] = {
    "xgboost": "XGBoost",
    "physics": "Fitted physical PV model",
}
"""Instrument keys to the labels a reader sees."""

CONTRAST_LABELS: Final[dict[str, str]] = {
    "B_erbs|A_global_only": "Erbs beam/diffuse split",
    "B_disc|A_global_only": "DISC beam/diffuse split",
    "B_learned|A_global_only": "Learned beam/diffuse split",
    "C_era5_split|A_global_only": "Weather product's beam/diffuse split",
    "D_direct_fraction|A_global_only": "Weather product's direct-beam fraction",
    "C_era5_split|B_erbs": "Weather product's split vs. Erbs's",
    "C_era5_split|B_learned": "Weather product's split vs. the learned split",
    "P_B_erbs|P_A_global_only": "Erbs beam/diffuse split",
    "P_B_disc|P_A_global_only": "DISC beam/diffuse split",
    "P_C_source_split|P_A_global_only": "Weather product's beam/diffuse split",
    "P_C_source_split|P_B_erbs": "Weather product's split vs. Erbs's",
}
"""Each plotted contrast, keyed by `treatment|reference`, to the label a reader sees."""

PANEL_TITLES: Final[tuple[str, str]] = (
    "Against total irradiance alone",
    "The weather product's split against Erbs's",
)
"""The two panels, in the order they are drawn."""

ROW_ORDER: Final[tuple[str, ...]] = (
    "Erbs beam/diffuse split",
    "DISC beam/diffuse split",
    "Learned beam/diffuse split",
    "Weather product's beam/diffuse split",
    "Weather product's direct-beam fraction",
    "Weather product's split vs. Erbs's",
    "Weather product's split vs. the learned split",
)
"""The order the contrasts are stacked in, best-known to least-known."""

RIGHT_PANEL_LABELS: Final[tuple[str, ...]] = (
    "Weather product's split vs. Erbs's",
    "Weather product's split vs. the learned split",
)
"""The contrasts drawn in the right panel, which compares two ways of getting a split."""

SUBTITLE: Final[tuple[str, ...]] = (
    "Six PV sites in one 25 km by 23 km box in Lincolnshire, hourly daylight rows.",
    "ERA5 and CAMS cover 2019-2026; UKV's archive starts in 2022.",
    "The beam/diffuse split is how total sunlight divides between the direct beam",
    "from the sun's disc and the light scattered across the rest of the sky.",
    "Change in mean absolute error against a model given total irradiance alone (%).",
    "Negative is better. Bars are 95% monthly block bootstrap intervals; a bar",
    "crossing zero has not been shown to help. Each panel has its own x scale.",
    "Every source is valid at the hour rather than forecast for it, so this is",
    "information content, not forecast skill. A half-hour error in the stamps flips",
    "the physical model's contrasts on the reanalysis; the tree's hold.",
)

CHART_PADDING: Final[dict[str, int]] = {"left": 120, "top": 5, "right": 5, "bottom": 5}
"""Outer padding in pixels, left-heavy so the y-axis labels have room.

Vega lays a faceted chart's shared y axis out inside a row-header group of zero width, so
an unlimited label runs off the left edge of the canvas instead of widening it. Padding the
whole chart is what actually reserves the space."""

PERCENTAGE_POINTS: Final[float] = 100.0


def _raise_on_unlabelled_sources(*, stem: str) -> None:
    """Raise if a results directory exists for a source `SOURCE_LABELS` does not name.

    **The failure this exists for is a chart that looks finished with an arm missing from it.**
    Drawing iterates the label mapping rather than the directories on disk, so a source added to the
    experiment and not to the mapping is dropped with no error, no warning, and no gap in the chart
    for a reader to notice. This is R&D code, so it stops rather than degrading.

    Two kinds of directory are left out on purpose and must not stop the run. A `--suffix` variant
    build is named `{source}{suffix}`, so it begins with a source one of the two tables names, and
    the chart draws each source's main build rather than its variants. And a source in
    `UNDRAWN_SOURCES` is excluded by a decision recorded there.

    Args:
        stem: `results` for the tree's runs or `physics` for the fitted model's.

    Raises:
        ValueError: If any results directory names a source the mapping does not.
    """
    prefix = f"beam_diffuse_{stem}_"
    suffix = f"_{ALIGNMENT}"
    found = {
        path.name[len(prefix) : -len(suffix)]
        for path in (REPO_DATA_DIR / "ERA5").glob(f"{prefix}*{suffix}")
        if path.is_dir()
    }
    known = (*SOURCE_LABELS, *UNDRAWN_SOURCES)
    unlabelled = sorted(
        name for name in found if not any(name.startswith(source) for source in known)
    )
    if unlabelled:
        msg = (
            f"{stem} results exist for {unlabelled}, which neither SOURCE_LABELS nor "
            "UNDRAWN_SOURCES names, so they would be left out of the chart without saying so. Add "
            "each one to whichever it belongs in."
        )
        raise ValueError(msg)


def _arm_mean_absolute_errors(*, instrument: str, source: str) -> dict[str, float]:
    """Return each arm's pooled mean absolute error as a fraction of P99 output.

    Weighted by row count, and on the same capped metric, the same unit and the same weighting as
    the contrast table in `report_results.py`, so the relative change the chart draws and the
    relative change the table prints are one statistic rather than two.

    Args:
        instrument: Which instrument's run to read, `xgboost` or `physics`.
        source: Which irradiance source's run to read.

    Returns:
        One pooled mean absolute error per arm, keyed by arm name.
    """
    stem = "results" if instrument == "xgboost" else "physics"
    summary = pl.read_parquet(
        REPO_DATA_DIR
        / "ERA5"
        / f"beam_diffuse_{stem}_{source}_{ALIGNMENT}"
        / "per_site_summary.parquet"
    ).filter(pl.col("setting") == "primary")
    pooled = summary.group_by("arm").agg(
        mae=(pl.col("mae_capped_fraction_of_capacity") * pl.col("n_rows")).sum()
        / pl.col("n_rows").sum()
    )
    return dict(zip(pooled["arm"], pooled["mae"], strict=True))


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
        _raise_on_unlabelled_sources(stem=stem)
        for source in SOURCE_LABELS:
            results_dir = REPO_DATA_DIR / "ERA5" / f"beam_diffuse_{stem}_{source}_{ALIGNMENT}"
            if not results_dir.exists():
                continue
            arm_mae = _arm_mean_absolute_errors(instrument=instrument, source=source)
            intervals = pl.read_parquet(results_dir / "bootstrap_intervals.parquet").filter(
                (pl.col("setting") == "primary")
                & (pl.col("metric") == "absolute_error_capped_fraction_of_capacity")
                & (pl.col("scope") == "all_sites")
            )
            # Each contrast is scaled by the arm it is against, so a bar reads as "this much
            # better than the arm named after the minus sign".
            percent = PERCENTAGE_POINTS / pl.col("reference").replace_strict(arm_mae)
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
        panel=pl.when(pl.col("contrast_label").is_in(RIGHT_PANEL_LABELS))
        .then(pl.lit(PANEL_TITLES[1]))
        .otherwise(pl.lit(PANEL_TITLES[0]))
    )


def _chart(*, differences: pl.DataFrame) -> alt.FacetChart:
    """Build the faceted point-and-interval chart."""
    base = alt.Chart(differences).encode(
        y=alt.Y(
            "contrast_label:N",
            title=None,
            sort=list(ROW_ORDER),
            # Vega truncates a category label at 180 pixels by default, which cut the
            # longest contrast to an ellipsis. Lifting the limit needs the chart padding
            # below, because a faceted chart draws its shared y axis inside a zero-width
            # row-header group and will otherwise run the text off the left edge.
            axis=alt.Axis(labelLimit=0),
        ),
        yOffset=alt.YOffset("source_label:N", sort=list(SOURCE_LABELS.values())),
        color=alt.Color(
            "source_label:N",
            title="Irradiance source",
            sort=list(SOURCE_LABELS.values()),
            legend=alt.Legend(orient="bottom", direction="horizontal", titleLimit=0, labelLimit=0),
        ),
    )
    # The axis title lives in the subtitle instead: four facets each drawing their own would
    # collide, and every panel measures the same quantity in the same units.
    interval = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]  # astral-sh/ty#2520
        x=alt.X("lower_95_percent:Q", title=None),
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
            row=alt.Row(
                "instrument_label:N",
                title=None,
                sort=list(INSTRUMENT_LABELS.values()),
                # Default is the left, where the row header collides with the longer
                # contrast labels.
                header=alt.Header(orient="right"),
            ),
        )
        # Only the x scale is resolved per panel. Resolving y as well would drop the three rows the
        # physical model has no arm for, but it also detaches the column headers from the columns
        # they label, which is a chart that misleads rather than one with a gap in it.
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                text="Does a weather product's own beam/diffuse split help a PV model?",
                subtitle=SUBTITLE,
            ),
            padding=CHART_PADDING,
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
