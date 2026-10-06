"""The model-free comparison of UKV from CEDA with UKV from Open-Meteo, and its diagnostics.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1051>. It answers the first
question of the study: how far apart are the two archives' values, with no XGBoost model between
them. It reads `model_free_hours.parquet`, which `ukv_ceda_vs_openmeteo_build.py` wrote, and
describes the difference CEDA minus Open-Meteo in temperature (K), 10 m wind speed (m/s), 10 m wind
direction (the circular difference, degrees), and global horizontal irradiance (W m⁻²).

**Only lead-0 hours separate a product difference from a lead difference.** Open-Meteo serves each
hour's freshest analysis, and CEDA's 6-hourly archive gives leads 0 to 5 hours, so at leads 1 to 5
the two archives are different forecasts of one hour. Every table therefore reports the lead-0
rows beside all rows. The study's months are 2024-09 to 2025-12 (era 0) and 2026-02 to 2026-08
(era 1, after the PS47 upgrade of 2026-01-21), and a before-and-after row at 2025-05 reads PS46, a
move to new computers.

**Statistics.** Each split reports the mean difference, the mean absolute difference, the 99th
percentile of the absolute difference, and the correlation. The two means carry 95% intervals that
resample whole calendar months (`studies.bootstrap.bootstrap_row_difference`), and the percentile
and the correlation are point values. The irradiance rows compare Open-Meteo with CEDA's snapshot
rebuilt as Open-Meteo builds its hourly value, and with the raw snapshot.

**The irradiance ratio by era and hour.** At lead 0 both archives are the same UKV snapshot, so the
median ratio of CEDA's irradiance to Open-Meteo's, by era and UTC hour of day, shows whether
Open-Meteo builds its hourly value as CEDA's snapshot rebuilt with the zenith-cosine ratio. It does
before the PS47 upgrade of 2026-01-21 (ratios near 1) and does not after it, so the solar power
contrasts are planned on era 0 only and the irradiance rows of era 1 carry that caveat.

**Diagnostics, reporting only.** The cell-match diagnostic counts, at lead-0 hours, how often the
nearest of the nine CEDA cells around a site has the value closest to Open-Meteo's, which says
whether the archives read the same cell. The elevation diagnostic reports each site's mean
temperature offset at lead 0, which shows whether Open-Meteo adjusts temperature to the elevation
of the point. Neither chooses a cell: the study always reads the nearest.

Nothing here prints a generator's name, identifier or coordinates.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_openmeteo_compare.py`. A fresh run
stops (`refuse_to_overwrite`) while an output exists.
"""

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple, TypedDict

import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_row_difference
from studies.grid_sampling import distance_matrix_km
from studies.guards import refuse_to_overwrite
from ukv_ceda_vs_era5_build import (
    KELVIN,
    UKV_TEMPERATURE_VARIABLE,
    UkvStores,
    open_ukv_stores,
    ukv_at,
)
from ukv_ceda_vs_openmeteo_build import (
    MODEL_FREE_NAME,
    OUTPUT_DIR,
    era_1_irradiance_note,
    generator_roster,
    irradiance_ratios_by_elevation,
    irradiance_ratios_by_era_hour,
    lead_zero,
)

_LOG: Final[logging.Logger] = logging.getLogger("ukv_ceda_vs_openmeteo_compare")

REPORT_NAME: Final[str] = "direct_report.md"
DIFFERENCES_NAME: Final[str] = "direct_differences.parquet"
RATIOS_NAME: Final[str] = "direct_irradiance_ratios.parquet"

PS46_FIRST_MONTH: Final[str] = "2025-05"
"""The first month after the Met Office's move to new computers (PS46)."""

NEIGHBOURHOOD_CELLS: Final[int] = 9
"""How many CEDA cells nearest a site the cell-match diagnostic considers."""

MIN_ROWS_FOR_INTERVAL: Final[int] = 200
"""A split with fewer rows gets point values only."""

MIN_MONTHS_FOR_INTERVAL: Final[int] = 6
"""A split spanning fewer months gets point values only."""


TABLE_HEADER: Final[str] = (
    "| Split | Rows | Mean difference | Mean absolute difference | 99th percentile | Correlation |"
)


class Variable(NamedTuple):
    """One compared quantity."""

    name: str
    unit: str
    ceda: str
    om: str
    circular: bool = False
    daylight_only: bool = False
    wind_step: str = "ignore"
    """`exclude` drops the hours of Open-Meteo's wind-step spans, `only` keeps them alone."""
    min_speed_m_s: float = 0.0
    """Rows where CEDA's 10 m speed is at or below this are left out, (no direction when calm)."""


VARIABLES: Final[tuple[Variable, ...]] = (
    Variable("air temperature", "K", "ceda_temp_c", "om_temp_c"),
    Variable(
        "10 m wind speed", "m/s", "ceda_speed_10m_m_s", "om_speed_10m_m_s", wind_step="exclude"
    ),
    Variable(
        "10 m wind speed, inside Open-Meteo's step spans",
        "m/s",
        "ceda_speed_10m_m_s",
        "om_speed_10m_m_s",
        wind_step="only",
    ),
    Variable(
        "10 m wind direction (hours above 2 m/s)",
        "degrees",
        "ceda_direction_10m_deg",
        "om_direction_10m_deg",
        circular=True,
        wind_step="exclude",
        min_speed_m_s=2.0,
    ),
    Variable("global irradiance (rebuilt)", "W/m2", "ceda_ghi", "om_ghi", daylight_only=True),
    Variable(
        "global irradiance (raw snapshot)",
        "W/m2",
        "ceda_ghi_snapshot",
        "om_ghi",
        daylight_only=True,
    ),
)
"""The compared quantities: CEDA's column, Open-Meteo's column, and how to difference them."""


class DifferenceRecord(TypedDict):
    """One split of one variable, CEDA minus Open-Meteo."""

    variable: str
    unit: str
    split: str
    label: str
    n_rows: int
    n_months: int
    mean_difference: float
    mean_difference_lower_95: float
    mean_difference_upper_95: float
    mean_absolute_difference: float
    mean_absolute_difference_lower_95: float
    mean_absolute_difference_upper_95: float
    p99_absolute_difference: float
    correlation: float


def circular_difference_degrees(*, ceda: pl.Expr, om: pl.Expr) -> pl.Expr:
    """Return the signed difference of two directions, in (-180, 180] degrees.

    Args:
        ceda: CEDA's direction in degrees.
        om: Open-Meteo's direction in degrees.

    Returns:
        CEDA minus Open-Meteo, wrapped so that 359 against 1 is -2 degrees, not 358.
    """
    return ((ceda - om + 180.0) % 360.0) - 180.0


def with_difference(*, frame: pl.DataFrame, variable: Variable) -> pl.DataFrame:
    """Keep a variable's comparable rows and add its signed difference.

    Args:
        frame: `model_free_hours`'s result.
        variable: The quantity.

    Returns:
        The rows where both archives hold a value (and, for irradiance, where either is above
        zero), with `difference`.
    """
    rows = frame.drop_nulls([variable.ceda, variable.om])
    if variable.wind_step == "exclude":
        rows = rows.filter(~pl.col("om_wind_step"))
    elif variable.wind_step == "only":
        rows = rows.filter(pl.col("om_wind_step"))
    if variable.min_speed_m_s > 0.0:
        rows = rows.filter(pl.col("ceda_speed_10m_m_s") > variable.min_speed_m_s)
    if variable.daylight_only:
        rows = rows.filter((pl.col(variable.ceda) > 0.0) | (pl.col(variable.om) > 0.0))
    difference = (
        circular_difference_degrees(ceda=pl.col(variable.ceda), om=pl.col(variable.om))
        if variable.circular
        else pl.col(variable.ceda) - pl.col(variable.om)
    )
    return rows.with_columns(difference=difference)


def difference_record(
    *, rows: pl.DataFrame, variable: Variable, split: str, label: str
) -> DifferenceRecord:
    """Summarise one split of one variable.

    Args:
        rows: `with_difference`'s rows restricted to the split.
        variable: The quantity.
        split: The kind of split, such as `lead` or `era`.
        label: The split's value, such as `lead 3 h`.

    Returns:
        The record. A split with too few rows or months has null interval bounds.
    """
    difference = rows["difference"].to_numpy()
    months = rows["month"].to_numpy()
    n_months = len(np.unique(months)) if rows.height else 0
    enough = rows.height >= MIN_ROWS_FOR_INTERVAL and n_months >= MIN_MONTHS_FOR_INTERVAL
    nan = float("nan")
    if enough:
        mean_interval = bootstrap_row_difference(values=difference, months=months)
        mad_interval = bootstrap_row_difference(values=np.abs(difference), months=months)
        bounds = (
            mean_interval["lower_95"],
            mean_interval["upper_95"],
            mad_interval["lower_95"],
            mad_interval["upper_95"],
        )
    else:
        bounds = (nan, nan, nan, nan)
    if rows.height == 0:
        return {
            "variable": variable.name,
            "unit": variable.unit,
            "split": split,
            "label": label,
            "n_rows": 0,
            "n_months": 0,
            "mean_difference": nan,
            "mean_difference_lower_95": nan,
            "mean_difference_upper_95": nan,
            "mean_absolute_difference": nan,
            "mean_absolute_difference_lower_95": nan,
            "mean_absolute_difference_upper_95": nan,
            "p99_absolute_difference": nan,
            "correlation": nan,
        }
    correlation = (
        float(np.corrcoef(rows[variable.ceda].to_numpy(), rows[variable.om].to_numpy())[0, 1])
        if rows.height > 2 and not variable.circular
        else nan
    )
    return {
        "variable": variable.name,
        "unit": variable.unit,
        "split": split,
        "label": label,
        "n_rows": rows.height,
        "n_months": n_months,
        "mean_difference": float(difference.mean()),
        "mean_difference_lower_95": bounds[0],
        "mean_difference_upper_95": bounds[1],
        "mean_absolute_difference": float(np.abs(difference).mean()),
        "mean_absolute_difference_lower_95": bounds[2],
        "mean_absolute_difference_upper_95": bounds[3],
        "p99_absolute_difference": float(np.quantile(np.abs(difference), 0.99)),
        "correlation": correlation,
    }


def splits_of(*, rows: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    """List the splits of one variable's rows, as (split, label, rows).

    Args:
        rows: `with_difference`'s rows.

    Returns:
        All rows and the lead-0 rows; each CEDA lead; each hour of day; each calendar month; each
        era, all hours and lead 0; the PS46 before-and-after at lead 0; and each site at lead 0.
    """
    zero = lead_zero(frame=rows)
    splits: list[tuple[str, str, pl.DataFrame]] = [
        ("all", "all hours", rows),
        ("all", "lead 0 only", zero),
    ]
    splits += [
        ("lead", f"lead {lead} h", rows.filter(pl.col("lead_hours") == lead))
        for lead in sorted(rows["lead_hours"].unique().to_list())
    ]
    splits += [
        ("hour", f"UTC hour {hour:02d}", rows.filter(pl.col("hour_of_day") == hour))
        for hour in sorted(rows["hour_of_day"].unique().to_list())
    ]
    splits += [
        ("month", month, rows.filter(pl.col("month") == month))
        for month in sorted(rows["month"].unique().to_list())
    ]
    for era in (0, 1):
        splits.append(("era", f"era {era}, all hours", rows.filter(pl.col("era_code") == era)))
        splits.append(("era", f"era {era}, lead 0", zero.filter(pl.col("era_code") == era)))
    era_zero = zero.filter(pl.col("era_code") == 0)
    splits.append(
        (
            "PS46",
            f"before {PS46_FIRST_MONTH}, lead 0 (not an isolated PS46 effect)",
            era_zero.filter(pl.col("month") < PS46_FIRST_MONTH),
        )
    )
    splits.append(
        (
            "PS46",
            f"from {PS46_FIRST_MONTH}, lead 0 (not an isolated PS46 effect)",
            era_zero.filter(pl.col("month") >= PS46_FIRST_MONTH),
        )
    )
    splits += [
        ("site", f"site {site}, lead 0", zero.filter(pl.col("site") == site))
        for site in sorted(rows["site"].unique().to_list())
    ]
    return splits


def difference_records(*, frame: pl.DataFrame) -> list[DifferenceRecord]:
    """Summarise every split of every variable.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        One record per (variable, split).
    """
    records: list[DifferenceRecord] = []
    for variable in VARIABLES:
        rows = with_difference(frame=frame, variable=variable)
        for split, label, part in splits_of(rows=rows):
            _LOG.info("%s / %s", variable.name, label)
            records.append(
                difference_record(rows=part, variable=variable, split=split, label=label)
            )
    return records


# --- Diagnostics ----------------------------------------------------------------------------------


def temperature_offsets(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return each site's mean temperature offset (CEDA minus Open-Meteo) at lead 0.

    A near-constant offset that differs by site shows Open-Meteo adjusting temperature to the
    elevation of the point.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        One row per site with `mean_offset_k`, `sd_offset_k` and `n`.
    """
    return (
        lead_zero(frame=frame)
        .drop_nulls(["ceda_temp_c", "om_temp_c"])
        .group_by("site")
        .agg(
            mean_offset_k=(pl.col("ceda_temp_c") - pl.col("om_temp_c")).mean(),
            sd_offset_k=(pl.col("ceda_temp_c") - pl.col("om_temp_c")).std(),
            n=pl.len(),
        )
        .sort("site")
    )


def nearest_cell_wins(*, ceda: np.ndarray, om: np.ndarray) -> float:
    """Return the share of hours at which the first CEDA cell has the value closest to Open-Meteo's.

    Args:
        ceda: CEDA's values, one row per hour and one column per cell, the nearest cell first.
        om: Open-Meteo's value per hour.

    Returns:
        The share of hours whose smallest absolute difference is at column 0. Ties go to column 0.
    """
    error = np.abs(ceda - om[:, None])
    return float(np.mean(error.argmin(axis=1) == 0))


def cell_match_lines(*, ukv: UkvStores, frame: pl.DataFrame) -> list[str]:
    """Report how often the nearest CEDA cell is the one closest to Open-Meteo's value.

    Args:
        ukv: The opened stores.
        frame: `model_free_hours`'s result.

    Returns:
        Markdown lines, one per variable, pooled over sites.
    """
    cells = pl.DataFrame(
        {
            "cell_id": np.arange(len(ukv.latitude)),
            "latitude": ukv.latitude,
            "longitude": ukv.longitude,
        }
    )
    roster = generator_roster()
    distance = distance_matrix_km(sites=roster, cells=cells)
    zero = lead_zero(frame=frame)
    wins: dict[str, list[float]] = {"air temperature": [], "10 m wind speed": []}
    for index, site in enumerate(roster["site"]):
        nearest = np.argsort(distance[index])[:NEIGHBOURHOOD_CELLS]
        rows = zero.filter(pl.col("site") == site).drop_nulls(["om_temp_c", "om_speed_10m_m_s"])
        if rows.is_empty():
            continue
        _, means, _ = ukv_at(
            ukv=ukv,
            hours=rows["time"],
            offsets_hours=(0,),
            variables=(UKV_TEMPERATURE_VARIABLE, "wind_speed_10m"),
            cells=nearest,
            read_values=True,
        )
        wins["air temperature"].append(
            nearest_cell_wins(
                ceda=means[UKV_TEMPERATURE_VARIABLE] - KELVIN, om=rows["om_temp_c"].to_numpy()
            )
        )
        wins["10 m wind speed"].append(
            nearest_cell_wins(ceda=means["wind_speed_10m"], om=rows["om_speed_10m_m_s"].to_numpy())
        )
    chance = 1.0 / NEIGHBOURHOOD_CELLS
    return [
        (
            f"- {name}: the nearest of the {NEIGHBOURHOOD_CELLS} CEDA cells has the value closest "
            f"to Open-Meteo's at {np.mean(shares):.0%} of lead-0 hours on average over sites "
            f"(range {min(shares):.0%} to {max(shares):.0%}; chance is {chance:.0%})."
        )
        for name, shares in wins.items()
        if shares
    ]


# --- Report ---------------------------------------------------------------------------------------


def _row(*, record: DifferenceRecord) -> str:
    mean_interval = _interval(
        lower=record["mean_difference_lower_95"],
        upper=record["mean_difference_upper_95"],
        sign=True,
    )
    absolute_interval = _interval(
        lower=record["mean_absolute_difference_lower_95"],
        upper=record["mean_absolute_difference_upper_95"],
        sign=False,
    )
    return (
        f"| {record['label']} | {record['n_rows']:,} | {record['mean_difference']:+.3f} "
        f"{mean_interval} | {record['mean_absolute_difference']:.3f} {absolute_interval} | "
        f"{record['p99_absolute_difference']:.3f} | {record['correlation']:.4f} |"
    )


def _interval(*, lower: float, upper: float, sign: bool) -> str:
    """Format an interval, or a dash where the split has too few months for one."""
    if lower != lower or upper != upper:  # noqa: PLR0124 - NaN marks a split with no interval
        return "(no interval)"
    return f"[{lower:+.3f}, {upper:+.3f}]" if sign else f"[{lower:.3f}, {upper:.3f}]"


def report_text(
    *,
    records: Sequence[DifferenceRecord],
    offsets: pl.DataFrame,
    cell_lines: Sequence[str],
    ratios: pl.DataFrame,
    elevation_ratios: pl.DataFrame,
    era_1_note: str,
    levels: Mapping[str, tuple[float, float]],
) -> str:
    """Render every table the page quotes from the model-free comparison.

    Args:
        records: `difference_records`'s result.
        offsets: `temperature_offsets`'s result.
        cell_lines: `cell_match_lines`'s result.
        ratios: `irradiance_ratios_by_era_hour`'s result.
        elevation_ratios: `irradiance_ratios_by_elevation`'s result.
        era_1_note: `era_1_irradiance_note`'s result, empty where every hour matches.
        levels: Each variable's mean level at lead 0 in CEDA and in Open-Meteo, so a difference
            reads as a share.

    Returns:
        The report, in Markdown.
    """
    lines = [
        "# UKV from CEDA against UKV from Open-Meteo: the model-free comparison",
        "",
        (
            "Every difference is CEDA minus Open-Meteo. Means carry 95% intervals that resample "
            "whole months. The percentile and the correlation are point values."
        ),
        "",
    ]
    for variable in VARIABLES:
        lines += [f"## {variable.name} ({variable.unit})", ""]
        if variable.name in levels:
            ceda_level, om_level = levels[variable.name]
            lines += [
                (
                    f"Mean level at lead 0: CEDA {ceda_level:.3f}, Open-Meteo {om_level:.3f} "
                    f"{variable.unit}."
                ),
                "",
            ]
        for split in ("all", "lead", "era", "PS46", "site", "month", "hour"):
            chosen = [
                record
                for record in records
                if record["variable"] == variable.name and record["split"] == split
            ]
            if not chosen:
                continue
            lines += [
                f"### By {split}",
                "",
                TABLE_HEADER,
                "|---|---|---|---|---|---|",
                *(_row(record=record) for record in chosen),
                "",
            ]
    lines += [
        "## Irradiance construction, by era and hour",
        "",
        (
            "The median of CEDA's irradiance over Open-Meteo's at lead-0 hours where Open-Meteo "
            "exceeds 50 W/m2. `Rebuilt` is CEDA's snapshot scaled by the zenith-cosine ratio that "
            "Open-Meteo applied before PS47, and `raw` is the snapshot itself."
        ),
        f"After the upgrade: {era_1_note or 'every hour matches'}.",
        "",
        "| Era | UTC hour | Rebuilt ratio | Raw ratio | Rows |",
        "|---|---|---|---|---|",
        *(
            f"| {row['era_code']} | {row['hour_of_day']:02d} | {row['rebuilt_ratio']:.3f} | "
            f"{row['raw_ratio']:.3f} | {row['n']:,} |"
            for row in ratios.iter_rows(named=True)
        ),
        "",
        "### By sun elevation",
        "",
        (
            "The same ratio, without the 50 W/m2 cut, by the sun's elevation at the label, with "
            "the 10th and 90th percentiles of the ratio and the mean absolute difference. A "
            "rebuild from a snapshot at the label does not reproduce Open-Meteo's value when the "
            "sun is within a few degrees of the horizon, and after PS47 the spread is wide above "
            "10 degrees too, though the median is not."
        ),
        "",
        (
            "| Era | Sun elevation | Median ratio | 10th to 90th percentile | "
            "Mean abs. diff. (W/m2) | Rows |"
        ),
        "|---|---|---|---|---|---|",
        *(
            f"| {row['era_code']} | {row['bin']} | {row['rebuilt_ratio']:.3f} | "
            f"{row['p10']:.3f} to {row['p90']:.3f} | {row['mean_abs_diff']:.1f} | {row['n']:,} |"
            for row in elevation_ratios.iter_rows(named=True)
        ),
        "",
        "## Diagnostics",
        "",
        "### Cell match",
        "",
        (
            "Open-Meteo does not serve the nearest CEDA cell's value: a share far below 100% means "
            "it interpolates or reads a different grid, so the two archives are not the same "
            "analysis sampled at one cell, even at lead 0."
        ),
        "",
        *cell_lines,
        "",
        "### Temperature offset at lead 0, by site (K)",
        "",
        "| Site | Mean offset | Standard deviation | Rows |",
        "|---|---|---|---|",
        *(
            (
                f"| {row['site']} | {row['mean_offset_k']:+.3f} | {row['sd_offset_k']:.3f} | "
                f"{row['n']:,} |"
            )
            for row in offsets.iter_rows(named=True)
        ),
        "",
    ]
    return "\n".join(lines)


def lead_zero_levels(*, frame: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Return each variable's mean level at lead 0 in CEDA and in Open-Meteo.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        The two means, by variable name, over the rows `with_difference` keeps at lead 0.
    """
    levels: dict[str, tuple[float, float]] = {}
    for variable in VARIABLES:
        if variable.ceda not in frame.columns:
            continue
        rows = lead_zero(frame=with_difference(frame=frame, variable=variable))
        if rows.is_empty():
            continue
        levels[variable.name] = (
            float(np.mean(rows[variable.ceda].to_numpy())),
            float(np.mean(rows[variable.om].to_numpy())),
        )
    return levels


def main() -> int:
    """Read the model-free hours, describe the difference, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    paths = (
        arguments.output_dir / REPORT_NAME,
        arguments.output_dir / DIFFERENCES_NAME,
        arguments.output_dir / RATIOS_NAME,
    )
    refuse_to_overwrite(paths=paths)

    frame = pl.read_parquet(arguments.frames_dir / MODEL_FREE_NAME)
    records = difference_records(frame=frame)
    text = report_text(
        records=records,
        offsets=temperature_offsets(frame=frame),
        cell_lines=cell_match_lines(ukv=open_ukv_stores(), frame=frame),
        ratios=irradiance_ratios_by_era_hour(frame=frame),
        elevation_ratios=irradiance_ratios_by_elevation(frame=frame),
        era_1_note=era_1_irradiance_note(frame=frame),
        levels=lead_zero_levels(frame=frame),
    )
    paths[0].write_text(text)
    pl.DataFrame(records).write_parquet(paths[1])
    irradiance_ratios_by_era_hour(frame=frame).write_parquet(paths[2])
    sys.stdout.write(f"Wrote {len(records)} records to {arguments.output_dir}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
