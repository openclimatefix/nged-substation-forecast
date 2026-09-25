"""Draw the station-arm charts for the past-solar study, and the per-generator chart.

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
import polars as pl
from build_dataset import _pv_sites
from ens_past_solar import DECIDING_CONTRASTS
from ens_past_solar import OUTPUT_DIR as ENS_OUTPUT_DIR
from ens_past_solar_charts import NAMES as ENS_NAMES
from ens_past_solar_charts import per_generator_rows
from figure_numbers import FIGURE_NUMBERS, FigureKey
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
from studies.charts import (
    ProductFamily,
    figure,
    interval_panel,
    report_contrasts,
)
from weather_product_charts import (
    MODELS_WORK_SITES,
)

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
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
DOMAIN_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest value a figure's x domain extends."""


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


def _per_generator(*, contrasts: pl.DataFrame, ens_rows: list[pl.DataFrame]) -> alt.VConcatChart:
    """Draw each planned contrast at each generator, one panel per contrast: ENS's, then stations'.

    Args:
        contrasts: Every contrast table the station report holds.
        ens_rows: `ens_past_solar_charts.per_generator_rows`'s output, one frame per planned ENS
            contrast, in `ens_past_solar.DECIDING_CONTRASTS` order.

    Returns:
        The figure.

    Raises:
        ValueError: If a station contrast lacks a row for a generator.
    """
    per_site = contrasts.filter(pl.col("section") == SECTION_PER_GENERATOR)
    panel_rows = [
        (f"ENS rows: {ENS_NAMES[t]} − {ENS_NAMES[r]}", "weather model", rows)
        for (t, r), rows in zip(DECIDING_CONTRASTS, ens_rows, strict=True)
    ]
    for treatment, reference in PLANNED_CONTRASTS:
        selected = per_site.filter(
            pl.col("treatment") == treatment, pl.col("reference") == reference
        )
        if selected.height != len(MODELS_WORK_SITES):
            msg = f"{treatment} − {reference}: expected one row per generator"
            raise ValueError(msg)
        panel_rows.append(
            (
                f"Station rows: {NAMES[treatment]} − {NAMES[reference]}",
                FAMILIES[treatment],
                selected.select(
                    "difference",
                    "lower_95",
                    "upper_95",
                    label=pl.col("scope").str.replace("site ", "Generator "),
                    family=pl.lit(FAMILIES[treatment]),
                    planned=pl.lit(value=False),
                ),
            )
        )
    domain = _domain(rows=pl.concat([rows.select(_DOMAIN_COLUMNS) for _, _, rows in panel_rows]))
    panels = [
        interval_panel(
            rows=rows,
            x_domain=domain,
            x_title=X_TITLE if index == len(panel_rows) - 1 else "",
            zero_label="no difference",
            better_label="first input better",
            panel_title=title,
            figure_planning="exploratory",
            family_key=False,
        )
        for index, (title, _, rows) in enumerate(panel_rows)
    ]
    return figure(
        panels=panels,
        number=FIGURE_NUMBERS["per_generator"],
        figure_planning="exploratory",
        title="Each planned contrast has the same sign at all six generators",
        subtitle=[
            (
                "Each generator's paired difference, with the same month-resampling. The six "
                "generators share one nearest station and two ERA5 grid cells, so they are not "
                "six independent replications."
            ),
            f"{DOTS} {CAPACITY}",
            (
                "Six solar farms in Lincolnshire. The ENS and station panels are scored on "
                "different rows."
            ),
        ],
    )


_DOMAIN_COLUMNS: Final[tuple[str, str]] = ("lower_95", "upper_95")
"""The columns a shared x range must cover."""


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
    number: FigureKey,
    title: str,
    subtitle: list[str],
    panel_title: str,
) -> alt.VConcatChart:
    """Draw a chosen set of the report's exploratory contrasts, in the order given.

    Args:
        contrasts: Every contrast table the report holds.
        keys: The (treatment, reference) pairs to draw, top to bottom.
        number: The figure's key in `FIGURE_NUMBERS`.
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
        number=FIGURE_NUMBERS[number],
        figure_planning="exploratory",
        title=title,
        subtitle=[*subtitle, f"{DOTS} {CAPACITY}", SCOPE],
    )


def main() -> int:
    """Check the report, then write the station SVGs."""
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
    era5_gain = -_difference(contrasts=contrasts, key=("station_era5_xgb", "station_era5_control"))
    cams_gain = -_difference(contrasts=contrasts, key=(BLEND_ARM, "cams_global"))
    null_bound = max(
        _bound(contrasts=contrasts, keys=PADDED_KEYS),
        _bound(contrasts=contrasts, keys=TEMPERATURE_KEYS),
    )
    charts = {
        "station_past_solar_per_generator": _per_generator(
            contrasts=contrasts,
            ens_rows=per_generator_rows(report_path=ENS_OUTPUT_DIR / "report.md"),
        ),
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
            number="station_stations",
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
            number="station_controls",
            title=(
                f"The real station column lowers ERA5's error by {era5_gain:.3f} points against "
                f"a shuffled column, and CAMS's error by {cams_gain:.3f} points against plain "
                f"CAMS; a shuffled station column, or the station's own temperature, moves no "
                f"error by more than {null_bound:.3f}"
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
