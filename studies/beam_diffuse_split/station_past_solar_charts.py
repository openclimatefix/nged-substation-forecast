"""Draw the station-arm charts for the past-solar study.

One-off throwaway script for the charts of the weather-station addition to
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, in
`weather_product_charts.py`'s style. **Every number a chart shares with the page is read from
`station_past_solar.py`'s report**, and `_check_report` first regenerates the whole report from the
saved losses and the rebuilt row set and refuses to draw unless the saved file matches it
character for character, so a chart cannot disagree with the page.

Generators appear only as `A` to `F`, and no chart carries a calendar date. Every mark is drawn
with `aria=False`. No chart prints a station, a station-to-generator mapping, or a per-generator
distance; the nearest-station distance is quoted as the pooled range the report prints.

**Do not run this script until `station_past_solar.py` has fitted every arm and written its
report.**

Run it with `uv run python studies/beam_diffuse_split/station_past_solar_charts.py`, after
`station_past_solar.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from build_dataset import _pv_sites
from station_past_solar import (
    BLEND_ARM,
    BLEND_CONTROL_ARM,
    OUTPUT_DIR,
    PLANNED_CONTRASTS,
    STATION_ARM,
    Selection,
    _fingerprint,
    _report,
    build_rows,
    jobs,
)
from studies.bootstrap import bootstrap_absolute
from studies.charts import (
    FAMILY_COLOURS,
    ProductFamily,
    figure,
    interval_panel,
    leaderboard_panel,
    report_contrasts,
    report_errors,
)
from weather_product_charts import (
    MODELS_WORK_MIN_DAYLIGHT_HOURS,
    MODELS_WORK_MONTHS,
    MODELS_WORK_SITES,
    SOLAR_WEEK_CRITERIA,
    SOLAR_WEEK_DISPLAY_ORDER,
    _models_work_long_frame,
    _models_work_timeseries,
    _pick_weeks,
    _reconstruct_predicted,
)
from weather_products import METRIC, PERCENTAGE_POINTS

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

NAMES: Final[dict[str, str]] = {
    "cams_global": "CAMS",
    "era5_global": "ERA5",
    STATION_ARM: "Nearest station",
    "station_mean3": "Mean of the 3 nearest stations",
    "station_rank2": "Second-nearest station",
    "station_rank3": "Third-nearest station",
    BLEND_ARM: "CAMS and the nearest station",
    BLEND_CONTROL_ARM: "CAMS and a shuffled station column",
    "station_ghi_era5temp": "Nearest station's irradiance with ERA5's temperature",
    "station_era5_xgb": "ERA5 and the nearest station",
    "station_era5_control": "ERA5 and a shuffled station column",
}
"""Every arm's public name, as the page writes it."""

FAMILIES: Final[dict[str, ProductFamily]] = {
    "cams_global": "satellite",
    "era5_global": "reanalysis",
    STATION_ARM: "station observations",
    "station_mean3": "station observations",
    "station_rank2": "station observations",
    "station_rank3": "station observations",
    BLEND_ARM: "station observations",
    BLEND_CONTROL_ARM: "satellite",
    "station_ghi_era5temp": "station observations",
    "station_era5_xgb": "station observations",
    "station_era5_control": "reanalysis",
}
"""Every arm's family, which sets its colour in `studies.charts`."""

LEADERBOARD_ARMS: Final[tuple[str, ...]] = (
    BLEND_ARM,
    "cams_global",
    "station_mean3",
    STATION_ARM,
    "station_rank2",
    "era5_global",
    "station_rank3",
)
"""The arms the leaderboard draws: every real input, and neither padded control."""

TEMPERATURE_KEYS: Final[tuple[tuple[str, str], ...]] = ((STATION_ARM, "station_ghi_era5temp"),)
"""The contrast that swaps the station's air temperature for ERA5's."""

PADDED_KEYS: Final[tuple[tuple[str, str], ...]] = (
    (BLEND_CONTROL_ARM, "cams_global"),
    ("station_era5_control", "era5_global"),
)
"""The contrasts that pad a plain product with a shuffled copy of the station's irradiance."""

SECTION_PLANNED: Final[str] = "Planned contrasts"
SECTION_PER_GENERATOR: Final[str] = "The planned contrasts, per generator (exploratory)"
SECTION_EXPLORATORY: Final[str] = "Exploratory contrasts"

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "Six solar farms in Lincolnshire, December 2022 to December 2025."
ONE_STATION: Final[str] = (
    "All six generators take the same nearest radiation station, {low} to {high} km away, so the "
    "station rows rest on one pyranometer."
)
LEADERBOARD_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
DOMAIN_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest value a figure's x domain extends."""

FIGURE_LEADERBOARD: Final[int] = 21
FIGURE_PLANNED: Final[int] = 22
FIGURE_MODELS_WORK: Final[int] = 20
FIGURE_PER_GENERATOR: Final[int] = 23
FIGURE_STATIONS: Final[int] = 25
FIGURE_CONTROLS: Final[int] = 24


def _check_report(
    *,
    report: str,
    frame: pl.DataFrame,
    all_losses: pl.DataFrame,
    selection: Selection,
    repairs: dict[str, int],
    candidates: int,
) -> None:
    """Stop unless the saved report is exactly what the saved losses and row set now produce.

    This is the guard that every number a chart or the page quotes is one the committed script
    printed: the report is regenerated from the losses and the rebuilt row set, so a stale or
    hand-edited report fails here.

    Args:
        report: The saved `report.md`.
        frame: The rebuilt row set.
        all_losses: Every arm's saved losses, at both settings.
        selection: The in-memory station choice `build_rows` returned.
        repairs: The reader's repair counts `build_rows` returned.
        candidates: The candidate row count `build_rows` returned.

    Raises:
        ValueError: If the regenerated report differs from the saved one.
    """
    regenerated = _report(
        frame=frame,
        losses=all_losses,
        sites=_pv_sites(),
        job_list=jobs(),
        selection=selection,
        repairs=repairs,
        candidates=candidates,
    )
    if regenerated != report:
        first = next(
            (
                (a, b)
                for a, b in zip(regenerated.splitlines(), report.splitlines(), strict=False)
                if a != b
            ),
            "the end: one report has extra lines",
        )
        msg = f"report.md differs from what the saved losses now produce, first at {first}"
        raise ValueError(msg)


def _row_count(*, report: str) -> int:
    """Read the number of common site-hours from the report's heading."""
    match = re.search(r"on ([\d,]+) common site-hours", report)
    if match is None:
        msg = "report.md has no 'on N common site-hours' heading"
        raise ValueError(msg)
    return int(match[1].replace(",", ""))


def _nearest_range(*, report: str) -> tuple[str, str]:
    """Read the pooled distance range of the nearest radiation station from the report."""
    match = re.search(r"The nearest radiation station is (\d+) to (\d+) km", report)
    if match is None:
        msg = "report.md has no nearest-station distance line"
        raise ValueError(msg)
    return match[1], match[2]


def _one_station_line(*, report: str) -> str:
    low, high = _nearest_range(report=report)
    return ONE_STATION.format(low=low, high=high)


def _contrast_rows(*, contrasts: pl.DataFrame, section: str, scope: str = "all") -> pl.DataFrame:
    """Return one section's contrast rows at one scope, in report order."""
    return contrasts.filter(pl.col("section") == section, pl.col("scope") == scope)


def _labelled(*, rows: pl.DataFrame, planned: bool) -> pl.DataFrame:
    """Add a label, a family and a `planned` flag to contrast rows, in the order given."""
    return rows.select(
        "treatment",
        "reference",
        "difference",
        "lower_95",
        "upper_95",
        label=pl.col("treatment").replace_strict(NAMES)
        + pl.lit(" − ")
        + pl.col("reference").replace_strict(NAMES),
        family=pl.col("treatment").replace_strict(FAMILIES),
        planned=pl.lit(value=planned),
    )


def _domain(*, rows: pl.DataFrame) -> tuple[float, float]:
    return (
        min(0.0, *rows["lower_95"].to_list()) - DOMAIN_MARGIN,
        max(0.0, *rows["upper_95"].to_list()) + DOMAIN_MARGIN,
    )


def _leaderboard(
    *, losses: pl.DataFrame, errors: dict[str, float], report: str
) -> alt.VConcatChart:
    """Draw every real input's own mean absolute error, best first, with its 95% interval.

    Args:
        losses: Every arm's losses at the `pooled` setting.
        errors: Each arm's pooled mean absolute error, read from the report's first table.
        report: The report's text.

    Returns:
        The figure.

    Raises:
        ValueError: If a bootstrapped point estimate disagrees with the report's own number.
    """
    order = sorted(LEADERBOARD_ARMS, key=errors.__getitem__)
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
        title="The nearest station beats ERA5 but not CAMS, and adds to CAMS",
        subtitle=[
            (
                f"Every input scored on the same {_row_count(report=report):,} site-hours, "
                "each through its own XGBoost model."
            ),
            (
                "The station rows also read the station's own air temperature. The top row is "
                "CAMS with the station's irradiance added."
            ),
            _one_station_line(report=report),
            DOTS,
            CAPACITY,
            SCOPE,
        ],
    )


def _planned_contrasts(*, contrasts: pl.DataFrame, report: str) -> alt.VConcatChart:
    """Draw the three planned contrasts.

    Args:
        contrasts: Every contrast table the report holds.
        report: The report's text.

    Returns:
        The figure.

    Raises:
        ValueError: If the report does not hold exactly the three planned contrasts.
    """
    selected = _contrast_rows(contrasts=contrasts, section=SECTION_PLANNED)
    if selected.height != len(PLANNED_CONTRASTS):
        msg = f"expected {len(PLANNED_CONTRASTS)} planned contrasts, found {selected.height}"
        raise ValueError(msg)
    rows = _labelled(rows=selected, planned=True)
    difference = {
        (t, r): d
        for t, r, d in zip(rows["treatment"], rows["reference"], rows["difference"], strict=True)
    }
    panel = interval_panel(
        rows=rows,
        x_domain=_domain(rows=rows),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first input better",
        panel_title="The three planned contrasts",
        figure_planning="planned",
    )
    against_cams = difference[PLANNED_CONTRASTS[0]]
    against_era5 = -difference[PLANNED_CONTRASTS[1]]
    added = -difference[PLANNED_CONTRASTS[2]]
    return figure(
        panels=[panel],
        number=FIGURE_PLANNED,
        figure_planning=None,
        title=(
            f"The nearest station trails CAMS by {against_cams:.3f} points, beats ERA5 by "
            f"{against_era5:.3f}, and lowers CAMS's error by {added:.3f}"
        ),
        subtitle=[
            (
                "All rows are planned: written into the study plan before any station model was "
                "fitted. Each contrast holds at the second hyperparameter setting."
            ),
            (
                "The last row pairs CAMS and the station with CAMS and a shuffled copy of the "
                "station's irradiance, which carries the same number of columns."
            ),
            _one_station_line(report=report),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _per_generator(*, contrasts: pl.DataFrame, report: str) -> alt.VConcatChart:
    """Draw each planned contrast at each generator, one panel per contrast.

    Args:
        contrasts: Every contrast table the report holds.
        report: The report's text.

    Returns:
        The figure.

    Raises:
        ValueError: If a contrast lacks a row for a generator.
    """
    per_site = contrasts.filter(pl.col("section") == SECTION_PER_GENERATOR)
    panels = []
    frames = []
    for treatment, reference in PLANNED_CONTRASTS:
        selected = per_site.filter(
            pl.col("treatment") == treatment, pl.col("reference") == reference
        )
        if selected.height != len(MODELS_WORK_SITES):
            msg = f"{treatment} − {reference}: expected one row per generator"
            raise ValueError(msg)
        frames.append(
            selected.select(
                "difference",
                "lower_95",
                "upper_95",
                label=pl.col("scope").str.replace("site ", "Generator "),
                family=pl.lit(FAMILIES[treatment]),
                planned=pl.lit(value=False),
            )
        )
    domain = _domain(rows=pl.concat(frames))
    for (treatment, reference), rows in zip(PLANNED_CONTRASTS, frames, strict=True):
        panels.append(
            interval_panel(
                rows=rows,
                x_domain=domain,
                x_title=X_TITLE,
                zero_label="no difference",
                better_label="first input better",
                panel_title=f"{NAMES[treatment]} − {NAMES[reference]}",
                figure_planning="exploratory",
                family_key=False,
            )
        )
    return figure(
        panels=panels,
        number=FIGURE_PER_GENERATOR,
        figure_planning="exploratory",
        title="Each planned contrast has the same sign at all six generators",
        subtitle=[
            (
                "Each generator's paired difference, with the same month-resampling. The six "
                "generators share one nearest station and two ERA5 grid cells, so they are not "
                "six independent replications."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPE,
        ],
    )


def _bound(*, contrasts: pl.DataFrame, keys: tuple[tuple[str, str], ...]) -> float:
    """Return the largest absolute interval end over the exploratory contrasts named.

    Args:
        contrasts: Every contrast table the report holds.
        keys: The (treatment, reference) pairs.

    Returns:
        The bound, in points of capacity.
    """
    table = _contrast_rows(contrasts=contrasts, section=SECTION_EXPLORATORY)
    ends = [
        abs(end)
        for treatment, reference in keys
        for end in table.filter(pl.col("treatment") == treatment, pl.col("reference") == reference)
        .select("lower_95", "upper_95")
        .row(0)
    ]
    return max(ends)


def _difference(*, contrasts: pl.DataFrame, key: tuple[str, str]) -> float:
    """Return one exploratory contrast's difference, in points of capacity."""
    table = _contrast_rows(contrasts=contrasts, section=SECTION_EXPLORATORY)
    treatment, reference = key
    row = table.filter(pl.col("treatment") == treatment, pl.col("reference") == reference)
    return float(row["difference"].item())


def _exploratory(
    *,
    contrasts: pl.DataFrame,
    keys: list[tuple[str, str]],
    number: int,
    title: str,
    subtitle: list[str],
    panel_title: str,
) -> alt.VConcatChart:
    """Draw a chosen set of the report's exploratory contrasts, in the order given.

    Args:
        contrasts: Every contrast table the report holds.
        keys: The (treatment, reference) pairs to draw, top to bottom.
        number: The figure's number.
        title: The finding the figure shows.
        subtitle: Short lines for the caption.
        panel_title: The panel's title.

    Returns:
        The figure.

    Raises:
        ValueError: If a pair is missing from the report.
    """
    table = _contrast_rows(contrasts=contrasts, section=SECTION_EXPLORATORY)
    records = []
    for treatment, reference in keys:
        row = table.filter(pl.col("treatment") == treatment, pl.col("reference") == reference)
        if row.height != 1:
            msg = f"{treatment} − {reference} is not exactly once in the exploratory table"
            raise ValueError(msg)
        records.append(row)
    rows = _labelled(rows=pl.concat(records), planned=False)
    panel = interval_panel(
        rows=rows,
        x_domain=_domain(rows=rows),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first input better",
        panel_title=panel_title,
        figure_planning="exploratory",
    )
    return figure(
        panels=[panel],
        number=number,
        figure_planning="exploratory",
        title=title,
        subtitle=[*subtitle, f"{DOTS} {CAPACITY}", SCOPE],
    )


def _models_work(*, frame: pl.DataFrame, losses: pl.DataFrame) -> alt.VConcatChart:
    """Draw out-of-fold power against measured power, for the station and CAMS inputs.

    The weeks come from measured power alone, by the rule `weather_product_charts.py` uses, so no
    input's values enter the choice. The x axis counts days 1 to 7 and every mark has
    `aria=False`.

    Args:
        frame: This section's row set, carrying measured power.
        losses: Every arm's losses at the `pooled` setting.

    Returns:
        The figure.
    """
    measured = frame.select(
        "site", "time", "power_mw", "effective_capacity_mw", "extraterrestrial_horizontal_w_m2"
    )
    labels = {
        STATION_ARM: "XGBoost model given the nearest station",
        "cams_global": "XGBoost model given CAMS",
    }
    order = ("Measured", *labels.values())
    predicted = tuple(
        (_reconstruct_predicted(losses=losses, measured=measured, arm=arm), label)
        for arm, label in labels.items()
    )
    hourly = measured.filter(
        (pl.col("extraterrestrial_horizontal_w_m2") > 0)
        & pl.col("time").dt.month().is_in(MODELS_WORK_MONTHS)
    ).with_columns(
        output_frac=pl.col("power_mw").cast(pl.Float64) / pl.col("effective_capacity_mw")
    )
    weeks = _pick_weeks(
        hourly=hourly,
        min_hours=MODELS_WORK_MIN_DAYLIGHT_HOURS,
        agg="sum",
        criteria=SOLAR_WEEK_CRITERIA,
    )
    long_frame = (
        _models_work_long_frame(measured=measured, predicted=predicted)
        .with_columns(week=pl.col("time").dt.truncate("1w"))
        .join(weeks, on="week", how="inner")
    )
    return _models_work_timeseries(
        long_frame=long_frame,
        sites=MODELS_WORK_SITES,
        week_order=SOLAR_WEEK_DISPLAY_ORDER,
        order=order,
        colours=(
            ocf.TEXT,
            FAMILY_COLOURS[FAMILIES[STATION_ARM]],
            FAMILY_COLOURS[FAMILIES["cams_global"]],
        ),
        number=FIGURE_MODELS_WORK,
        title=(
            "An XGBoost model given the nearest station follows measured power at every "
            "generator, across a clear, a variable, and a dull week"
        ),
        subtitle=[
            (
                "Out-of-fold power as a percentage of the generator's own capacity, each "
                "prediction held to the export cap as the scores are."
            ),
            (
                "Weeks are picked from measured power alone, pooled over the six generators, "
                "April to September: the clearest has the most output, the dullest the least, "
                "and the most variable the largest swing in daily output."
            ),
            CAPACITY,
            SCOPE,
        ],
    )


def main() -> int:
    """Check the report, then write the six SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = OUTPUT_DIR / "report.md"
    report = report_path.read_text()
    frame, selection, repairs, candidates = build_rows()
    all_losses = pl.read_parquet(OUTPUT_DIR / "losses.parquet")
    saved_fingerprint = (OUTPUT_DIR / "losses.fingerprint").read_text().strip()
    if _fingerprint(frame=frame, job_list=jobs()) != saved_fingerprint:
        msg = (
            "losses.parquet was fitted on a different row set or job list than this code builds; "
            "re-run station_past_solar.py"
        )
        raise ValueError(msg)
    _check_report(
        report=report,
        frame=frame,
        all_losses=all_losses,
        selection=selection,
        repairs=repairs,
        candidates=candidates,
    )
    contrasts = report_contrasts(report_path=report_path)
    errors = report_errors(report_path=report_path, column="All sites")
    losses = all_losses.filter(pl.col("setting") == "pooled")
    era5_gain = -_difference(contrasts=contrasts, key=("station_era5_xgb", "station_era5_control"))
    cams_gain = -_difference(contrasts=contrasts, key=(BLEND_ARM, "cams_global"))
    null_bound = max(
        _bound(contrasts=contrasts, keys=PADDED_KEYS),
        _bound(contrasts=contrasts, keys=TEMPERATURE_KEYS),
    )
    charts = {
        "station_past_solar_leaderboard": _leaderboard(losses=losses, errors=errors, report=report),
        "station_past_solar_planned_contrasts": _planned_contrasts(
            contrasts=contrasts, report=report
        ),
        "station_past_solar_models_work": _models_work(frame=frame, losses=losses),
        "station_past_solar_per_generator": _per_generator(contrasts=contrasts, report=report),
        "station_past_solar_stations": _exploratory(
            contrasts=contrasts,
            keys=[
                ("station_mean3", STATION_ARM),
                ("station_rank2", STATION_ARM),
                ("station_rank3", STATION_ARM),
                ("station_mean3", "cams_global"),
                ("station_mean3", "era5_global"),
                ("station_rank2", "era5_global"),
                ("station_rank3", "era5_global"),
            ],
            number=FIGURE_STATIONS,
            title=(
                "Averaging three stations beats the nearest station alone, and the third-nearest "
                "station scores no better than ERA5"
            ),
            subtitle=[
                (
                    "The second- and third-nearest stations differ from the nearest in place, "
                    "instrument, and record, as well as distance, so the gaps are not a distance "
                    "effect alone."
                ),
                _one_station_line(report=report),
            ],
            panel_title="Exploratory contrasts: more stations, and further stations",
        ),
        "station_past_solar_controls": _exploratory(
            contrasts=contrasts,
            keys=[
                *TEMPERATURE_KEYS,
                *PADDED_KEYS,
                ("station_era5_xgb", "station_era5_control"),
                (BLEND_ARM, "cams_global"),
            ],
            number=FIGURE_CONTROLS,
            title=(
                f"The real station column lowers ERA5's error by {era5_gain:.3f} points and "
                f"CAMS's by {cams_gain:.3f}; a shuffled one, or the station's own temperature, "
                f"moves no error by more than {null_bound:.3f}"
            ),
            subtitle=[
                (
                    "The first row swaps the station's air temperature for ERA5's; the next two "
                    "pad a product with a shuffled station column, one more column than the plain "
                    "product; the last two add the real station column, and the last row's two "
                    "arms differ by one column."
                ),
            ],
            panel_title="Exploratory contrasts: controls, and what a station adds to a product",
        ),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
