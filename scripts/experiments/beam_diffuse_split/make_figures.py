"""Draw the figures the write-up needs beyond the headline contrast chart.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

Three kinds of figure, all anonymised:

- **Predicted against measured power, over three weeks chosen for their sky.** A contrast between
  two arms says nothing about whether either arm works, and a reader has no reason to believe a
  0.09-point difference from a pipeline they have not seen produce a sane forecast. One week is the
  clearest in the record, one the dullest, and one the most variable within days, so a reader sees
  the easy case, the hard case and the interesting case rather than a week that flattered the model.
- **Mean absolute error per site, for each source and instrument.** The levels the contrasts sit
  on top of.
- **The headline contrast split by sky condition**, which is what `sky_conditions.py` computed.

**Power is drawn as a percentage of each site's own 99th-percentile output, never in MW.** These
are metered generators whose output is commercially sensitive, the site labels are already a
shuffled relabelling, and a per-site figure in MW would combine with the published generation
register to shortlist candidates inside a 34 km box. Normalising removes that, and costs the reader
nothing the contrasts do not already express in the same unit.

Run it with `uv run --no-project --with polars --with altair --with vl-convert-python
--with numpy python scripts/experiments/beam_diffuse_split/make_figures.py`.
"""

import logging
import sys
from pathlib import Path
from typing import Final

import altair as alt

# Importing the theme module registers and enables the OCF Altair theme as a side effect.
import plotting.ocf_theme as ocf
import polars as pl
from run_experiment import dataset_path_for, results_dir_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("make_figures")

REPO_DATA_DIR: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data")
FIGURES_DIR: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse_figures"

ALIGNMENT: Final[str] = "shifted"
"""The stamp alignment every figure is drawn under, the one three independent tests support."""

MEASURED_COLOUR: Final[str] = "#3A3A3A"
"""Measured power is drawn in ink rather than in a series colour, because it is the reference."""

SETUPS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("open-meteo", "xgboost", "C_era5_split", "ERA5 → XGBoost"),
    ("cams", "xgboost", "C_era5_split", "CAMS → XGBoost"),
    ("cams", "physics", "P_C_source_split", "CAMS → fitted physical model"),
)
"""Each drawn setup as (source, instrument, arm, label), in legend order.

Every one is the arm given the product's own split, which is the best each setup has to offer, so
the figure shows what the pipeline can do rather than what a deliberately weakened arm can do.
"""

SETUP_COLOURS: Final[tuple[str, ...]] = (ocf.ORANGE_RED, ocf.BLUE, ocf.DARK_GREEN)
"""One hue per setup, in `SETUPS` order.

Checked for colour-vision deficiency rather than chosen by eye: the worst adjacent pair separates
by 25 units of perceptual distance under deuteranopia, against a floor of 8.
"""

MAE_SETUPS: Final[tuple[tuple[str, str, str], ...]] = (
    ("open-meteo", "xgboost", "ERA5 → XGBoost"),
    ("cams", "xgboost", "CAMS → XGBoost"),
    ("open-meteo", "physics", "ERA5 → fitted physical model"),
    ("cams", "physics", "CAMS → fitted physical model"),
)
"""Every (source, instrument) pairing the per-site error figure compares."""

SOURCE_LABELS: Final[dict[str, str]] = {
    "open-meteo": "ERA5 (31 km reanalysis)",
    "cams": "CAMS (5 km satellite retrieval)",
}
"""Source keys to the label a reader sees, matching the headline contrast chart."""

INSTRUMENT_LABELS: Final[dict[str, str]] = {
    "xgboost": "XGBoost",
    "physics": "Fitted physical PV model",
}
"""Instrument keys to the label a reader sees."""

BEST_ARM: Final[dict[str, str]] = {
    "xgboost": "C_era5_split",
    "physics": "P_C_source_split",
}
"""The arm each instrument is scored on in the per-site error figure."""

WEEK_SEASON_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""Weeks are chosen from these months only.

A midwinter week has so few daylight hours that every model looks alike on it, which would waste
one of the three panels a reader actually studies.
"""

PERCENT: Final[float] = 100.0


def _predictions(*, source: str, instrument: str, arm: str) -> pl.DataFrame:
    """Reconstruct one setup's out-of-fold predictions.

    The runners store the signed error rather than the prediction, so the prediction comes back by
    adding the error to the measured value. Predictions are averaged over the seeds, because a
    figure showing three near-identical lines per setup would say nothing a single line does not.

    Args:
        source: The irradiance source the setup used.
        instrument: `xgboost` or `physics`.
        arm: Which arm's predictions to take.

    Returns:
        One row per (site, time) with `predicted_mw`.
    """
    stem = "results" if instrument == "xgboost" else "physics"
    path = REPO_DATA_DIR / "ERA5" / f"beam_diffuse_{stem}_{source}_{ALIGNMENT}"
    losses = pl.read_parquet(path / "per_row_losses.parquet").filter(
        (pl.col("setting") == "primary") & (pl.col("target") == "power_mw") & (pl.col("arm") == arm)
    )
    return (
        losses.group_by("site", "time")
        .agg(signed_error_mw=pl.col("signed_error_mw").mean())
        .sort("site", "time")
    )


def _measured() -> pl.DataFrame:
    """Return measured power and the normalising denominator for every row of the CAMS build."""
    return pl.read_parquet(dataset_path_for(source="cams", alignment=ALIGNMENT)).select(
        "site",
        "time",
        "power_mw",
        "effective_capacity_mw",
        "ghi_w_m2",
        "extraterrestrial_horizontal_w_m2",
    )


def _chosen_weeks(*, measured: pl.DataFrame) -> pl.DataFrame:
    """Pick the clearest, the dullest and the most variable week in the record.

    Args:
        measured: Rows carrying global and extraterrestrial irradiance.

    Returns:
        One row per chosen week, with the label the figure gives it.
    """
    hourly = measured.filter(
        (pl.col("extraterrestrial_horizontal_w_m2") > 0)
        & pl.col("time").dt.month().is_in(WEEK_SEASON_MONTHS)
    ).with_columns(
        clearness=pl.col("ghi_w_m2") / pl.col("extraterrestrial_horizontal_w_m2"),
        week=pl.col("time").dt.truncate("1w"),
    )
    complete = (
        hourly.group_by("week")
        .agg(
            sites=pl.col("site").n_unique(),
            hours=pl.len(),
            mean_clearness=pl.col("clearness").mean(),
            daily_spread=pl.col("clearness").std(),
        )
        .filter((pl.col("sites") == pl.col("sites").max()) & (pl.col("hours") > 400))
    )
    clearest = complete.sort("mean_clearness", descending=True).head(1)
    dullest = complete.sort("mean_clearness").head(1)
    most_variable = complete.sort("daily_spread", descending=True).head(1)
    chosen = pl.concat(
        [
            clearest.with_columns(sky=pl.lit("Clearest week in the record")),
            most_variable.with_columns(sky=pl.lit("Most variable week")),
            dullest.with_columns(sky=pl.lit("Dullest week")),
        ]
    )
    _LOG.info("weeks chosen: %s", chosen.select("week", "sky", "mean_clearness").to_dicts())
    return chosen.select("week", "sky", "mean_clearness")


def _timeseries_frame(*, measured: pl.DataFrame, weeks: pl.DataFrame) -> pl.DataFrame:
    """Join every setup's predictions to the measured power, restricted to the chosen weeks."""
    scoped = measured.with_columns(week=pl.col("time").dt.truncate("1w")).join(
        weeks, on="week", how="inner"
    )
    frames = [
        scoped.select(
            "site",
            "time",
            "sky",
            percent_of_p99=(
                pl.col("power_mw").cast(pl.Float64) / pl.col("effective_capacity_mw") * PERCENT
            ),
        ).with_columns(series=pl.lit("Measured"))
    ]
    for source, instrument, arm, label in SETUPS:
        predicted = scoped.join(
            _predictions(source=source, instrument=instrument, arm=arm),
            on=["site", "time"],
            how="inner",
        )
        frames.append(
            predicted.select(
                "site",
                "time",
                "sky",
                percent_of_p99=(pl.col("power_mw") + pl.col("signed_error_mw"))
                / pl.col("effective_capacity_mw")
                * PERCENT,
            ).with_columns(series=pl.lit(label))
        )
    return pl.concat(frames)


def _timeseries_chart(*, frame: pl.DataFrame, site: str) -> alt.FacetChart:
    """Draw one site's measured and predicted power across the three chosen weeks."""
    rows = frame.filter(pl.col("site") == site).with_columns(
        day=pl.col("time").dt.strftime("%Y-%m-%d")
    )
    order = ["Measured", *(label for *_, label in SETUPS)]
    lines = (
        alt.Chart(rows)
        .mark_line(strokeWidth=1.6, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%a %d %b")),
            # Only daylight hours are modelled, so one line per day stops each evening joining to
            # the next morning across a night the experiment never looked at.
            detail=alt.Detail("day:N"),
            y=alt.Y("percent_of_p99:Q", title=None, scale=alt.Scale(zero=True)),
            color=alt.Color(
                "series:N",
                title=None,
                sort=order,
                scale=alt.Scale(domain=order, range=[MEASURED_COLOUR, *SETUP_COLOURS]),
                legend=alt.Legend(orient="bottom", direction="horizontal", labelLimit=0),
            ),
            strokeDash=alt.StrokeDash("series:N", sort=order, legend=None),
        )
    )
    return (
        lines.properties(width=620, height=150)
        .facet(row=alt.Row("sky:N", title=None, sort=list(_SKY_ORDER)))
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                text=f"Site {site}: predicted against measured PV power",
                subtitle=(
                    (
                        "Power as a percentage of the site's own 99th-percentile output,"
                        " which keeps a commercially sensitive meter anonymous."
                    ),
                    (
                        "Out-of-fold predictions, each given the product's own split."
                        " Hourly, daylight hours only."
                    ),
                ),
            )
        )
    )


_SKY_ORDER: Final[tuple[str, ...]] = (
    "Clearest week in the record",
    "Most variable week",
    "Dullest week",
)
"""Panel order, from the easiest sky to the hardest."""


def _per_site_error() -> pl.DataFrame:
    """Collect every setup's per-site mean absolute error, as a percentage of P99 output."""
    records: list[dict[str, object]] = []
    for source, instrument, label in MAE_SETUPS:
        stem = "results" if instrument == "xgboost" else "physics"
        path = REPO_DATA_DIR / "ERA5" / f"beam_diffuse_{stem}_{source}_{ALIGNMENT}"
        summary = pl.read_parquet(path / "per_site_summary.parquet").filter(
            (pl.col("setting") == "primary") & (pl.col("arm") == BEST_ARM[instrument])
        )
        records.extend(
            {
                "setup": label,
                "source_label": SOURCE_LABELS[source],
                "instrument_label": INSTRUMENT_LABELS[instrument],
                "site": row["site"],
                "mae_percent": row["mae_fraction_of_capacity"] * PERCENT,
                "hours": row["n_rows"],
            }
            for row in summary.iter_rows(named=True)
        )
    return pl.DataFrame(records)


def _per_site_chart(*, frame: pl.DataFrame) -> alt.FacetChart:
    """Draw per-site mean absolute error, one panel per instrument and one hue per source.

    Colour carries the source alone rather than the whole (source, instrument) pairing. Four hues
    would need a fourth that separates from the other three under colour-vision deficiency, and the
    two that read most naturally against this palette do not; splitting the instrument out into its
    own panel keeps two hues, and they are the same two the headline chart uses.

    Args:
        frame: One row per (setup, site).

    Returns:
        The faceted bar chart.
    """
    sources = list(SOURCE_LABELS.values())
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("site:N", title="Site"),
            x=alt.X("mae_percent:Q", title="Mean absolute error (% of the site's P99 output)"),
            yOffset=alt.YOffset("source_label:N", sort=sources),
            color=alt.Color(
                "source_label:N",
                title=None,
                sort=sources,
                scale=alt.Scale(domain=sources, range=[ocf.ORANGE_RED, ocf.BLUE]),
                legend=alt.Legend(orient="bottom", direction="horizontal", labelLimit=0),
            ),
        )
        .properties(width=330, height=alt.Step(15))
    )
    return bars.facet(
        column=alt.Column("instrument_label:N", title=None, sort=list(INSTRUMENT_LABELS.values()))
    ).properties(
        title=alt.TitleParams(
            text="Error level of each setup, per site",
            subtitle=(
                "Each setup given the product's own split. Lower is better.",
                (
                    "Site labels are shuffled and the error normalised, because these"
                    " meters are commercially sensitive."
                ),
            ),
        ),
    )


def _sky_chart(*, source: str) -> alt.Chart:
    """Draw the headline contrast alone, split by sky condition.

    Only the contrast against Erbs is drawn. The contrast against the learned split runs within a
    few thousandths of it in every bin, so plotting both would put two indistinguishable intervals
    on each row and invite a reader to hunt for a difference that is not there; the table carries
    the second contrast instead.

    Args:
        source: Which irradiance source's breakdown to draw.

    Returns:
        The point-and-interval chart.
    """
    intervals = pl.read_parquet(
        results_dir_for(source=source, alignment=ALIGNMENT) / "sky_intervals.parquet"
    ).filter((pl.col("treatment") == "C_era5_split") & (pl.col("reference") == "B_erbs"))
    # The bin's clearness range belongs in the table, not in an axis label long enough to run off
    # the left of the figure.
    intervals = intervals.with_columns(
        sky_short=pl.col("sky").str.replace(r" \(.*\)$", ""),
    )
    order = list(intervals["sky_short"].unique(maintain_order=True))
    base = alt.Chart(intervals).encode(
        y=alt.Y("sky_short:N", title=None, sort=order),
    )
    rule = base.mark_rule(strokeWidth=2, color=ocf.BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower_95:Q", title="Change in mean absolute error (pp of P99 output)"),
        x2=alt.X2("upper_95:Q"),
    )
    point = base.mark_point(filled=True, size=80, color=ocf.BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q")
    )
    zero = (
        alt.Chart(intervals).mark_rule(strokeDash=[4, 4], color="#292B2B").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    )
    return (zero + rule + point).properties(
        width=430,
        height=alt.Step(34),
        title=alt.TitleParams(
            text="Where the published beam field earns its advantage",
            subtitle=(
                "The product's own split against the Erbs split. Negative is better.",
                "Bars are 95% monthly block bootstrap intervals.",
                "Under a clear sky a separation model already knows the answer.",
            ),
        ),
    )


def main() -> int:
    """Write every figure as an SVG."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    measured = _measured()
    weeks = _chosen_weeks(measured=measured)
    frame = _timeseries_frame(measured=measured, weeks=weeks)
    for site in sorted(frame["site"].unique().to_list()):
        path = FIGURES_DIR / f"power_timeseries_site_{site.lower()}.svg"
        _timeseries_chart(frame=frame, site=site).save(path)
        _LOG.info("wrote %s", path)

    errors = _per_site_error()
    errors.write_parquet(FIGURES_DIR / "per_site_error.parquet")
    _per_site_chart(frame=errors).save(FIGURES_DIR / "per_site_error.svg")
    _LOG.info("wrote %s", FIGURES_DIR / "per_site_error.svg")

    _sky_chart(source="cams").save(FIGURES_DIR / "sky_conditions.svg")
    _LOG.info("wrote %s", FIGURES_DIR / "sky_conditions.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
