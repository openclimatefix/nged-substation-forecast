"""Outside checks of the weekly capacity series (experiment 00045, unit U5b).

Every outside photovoltaic source is partial, so every check here is descriptive and none can
decide the verdict. The script runs in two stages, in this order, so that the tolerances in the
plan's section "Outside checks and tolerances" are committed before any comparison is run:

- `--stage growth` reads only the outside sources and the dates (never the values) of the weekly
  series. It writes `outside_growth.parquet`, one row per polygon, source, and window, giving the
  source's fractional growth per year, and prints the table.
- `--stage compare` reads `outside_growth.parquet` and `capacity_weekly.parquet` and writes
  `outside_checks.parquet` and `outside_report.txt`.

The fractional growth per year of a curve over a window is its ordinary-least-squares slope in MW
per year, divided by its mean level over the window, with the curve sampled at every Monday of the
window (a source that holds only year-end values is interpolated linearly between them).

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_outside.py \
        --stage growth
"""

import argparse
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import e00045_common as common
import e00045_fetch_lsoa as fetch
import numpy as np
import polars as pl
from studies.name_matching import normalise
from studies.sources import REPO_DATA_DIR

from studies import ecr_reader as reader

OUTSIDE_DIR: Final[Path] = common.DATA_DIR / "outside"
GROWTH_PATH: Final[Path] = common.DATA_DIR / "outside_growth.parquet"
CHECKS_PATH: Final[Path] = common.DATA_DIR / "outside_checks.parquet"
REPORT_PATH: Final[Path] = common.DATA_DIR / "outside_report.txt"
WEEKLY_PATH: Final[Path] = common.DATA_DIR / "capacity_weekly.parquet"
METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
EXPERIMENT_00040_SCORES_PATH: Final[Path] = (
    REPO_DATA_DIR
    / "ml-experiments"
    / "00040-pv-capacity-error-bars"
    / "work"
    / "real_scores.parquet"
)

DESNZ_CONSTITUENCY_WORKBOOK: Final[Path] = OUTSIDE_DIR / "desnz" / "ET_6.3_SOLAR_PV_SEP_26.xlsx"
DESNZ_LOCAL_AUTHORITY_WORKBOOK: Final[Path] = (
    OUTSIDE_DIR / "desnz" / "Renewable_electricity_by_local_authority_2014_2025.xlsx"
)
LOCAL_AUTHORITY_YEARS: Final[range] = range(2014, 2026)
"""The year-end snapshots of the DESNZ local-authority workbook."""
CONSTITUENCY_SNAPSHOTS: Final[dict[str, tuple[str, date]]] = {
    "Table_3_Jul": ("Table_3_Jul", date(2026, 7, 31)),
    "Table_3_dom_by_PC": ("Table_3_dom_by_PC", date(2026, 8, 31)),
}
"""The two constituency tables the workbook holds, and the end of the month each describes."""

FIT_SCHEME_START: Final[date] = date(2010, 4, 1)
FIT_REPORT_END: Final[date] = date(2021, 3, 31)
FIT_FLOOR_FROM: Final[date] = date(2019, 4, 1)
"""The FiT closed to new applicants on 31 March 2019; later commissioning dates are late
accreditations and extensions, so the register is a floor from this date."""
FIT_PRE_2019_WINDOW: Final[tuple[date, date]] = (date(2015, 1, 1), date(2019, 3, 31))
SMALL_INSTALLATION_KW: Final[float] = 50.0
MIN_WINDOW_YEARS: Final[float] = 1.0
"""A growth rate over a window shorter than this is reported but never tested."""

SMALL_SHEET: Final[str] = "Register Part 1 50kW - <1MW"

CHECKED_STATUSES: Final[frozenset[str]] = frozenset({"clean", "low-power"})
"""The statuses of `capacity_weekly.parquet` the checks score; the others are reported only."""
VARIANTS: Final[tuple[str, ...]] = ("alt_post_hoc", "season_year_constant")


@dataclass(frozen=True)
class Polygon:
    """One Primary polygon and the weekly-series Primaries that sit on it."""

    name: str
    polygon_id: int
    series_ids: tuple[int, ...]


POLYGONS: Final[tuple[Polygon, ...]] = (
    Polygon(name="Spilsby", polygon_id=10, series_ids=(10,)),
    Polygon(name="Horncastle", polygon_id=8, series_ids=(8,)),
    Polygon(name="Mount Bridge Boston", polygon_id=16, series_ids=(16,)),
    Polygon(name="Donington", polygon_id=18, series_ids=(18,)),
    Polygon(name="Trusthorpe", polygon_id=7, series_ids=(7,)),
    Polygon(name="Wrangle T1 + T2", polygon_id=12, series_ids=(12, 13)),
    Polygon(name="Kirton", polygon_id=17, series_ids=(17,)),
)
"""Wrangle T1 and T2 share one polygon, so they are summed against it."""


@dataclass(frozen=True)
class Curve:
    """A source's cumulative capacity for one polygon, as points in time."""

    days: np.ndarray
    megawatts: np.ndarray

    def at(self, *, days: np.ndarray) -> np.ndarray:
        """Return the capacity at the given day numbers, interpolated between the points."""
        return np.interp(days, self.days, self.megawatts)


def day_number(*, day: date) -> int:
    """Return the number of days from 1970-01-01 to `day`."""
    return (day - date(1970, 1, 1)).days


def mondays(*, start: date, end: date) -> np.ndarray:
    """Return the day number of every Monday from `start` to `end`, inclusive."""
    first = start + timedelta(days=(-start.weekday()) % 7)
    return np.arange(day_number(day=first), day_number(day=end) + 1, 7)


def growth_per_year(*, days: np.ndarray, megawatts: np.ndarray) -> tuple[float, float, float]:
    """Return the fractional growth per year, the slope in MW per year, and the mean level.

    Args:
        days: Day numbers of the samples.
        megawatts: Capacity at each sample.

    Returns:
        The slope divided by the mean level (NaN where the mean level is not positive), the
        ordinary-least-squares slope in MW per year, and the mean level in MW.
    """
    if days.size < 2:
        return float("nan"), float("nan"), float("nan")
    years = (days - days.mean()) / 365.25
    slope = float(np.polyfit(years, megawatts, deg=1)[0])
    level = float(megawatts.mean())
    return (slope / level if level > 0 else float("nan")), slope, level


def series_spans() -> dict[str, tuple[date, date]]:
    """Return the first and last week of each polygon's weekly series, from the dates alone."""
    weekly = pl.read_parquet(WEEKLY_PATH, columns=["week_start", "time_series_id"])
    spans = {}
    for polygon in POLYGONS:
        rows = weekly.filter(pl.col("time_series_id").is_in(polygon.series_ids))
        weeks: list[date] = rows["week_start"].to_list()
        spans[polygon.name] = (min(weeks), max(weeks) + timedelta(6))
    return spans


def fit_rows() -> pl.DataFrame:
    """Return the FiT photovoltaic rows, each split over the polygons by district share.

    Returns:
        One row per FiT row and polygon it touches, with `polygon_id`, `share`,
        `declared_net_mw` (the row's declared net capacity times the share, in MW),
        `small` (installed capacity under 50 kW), `commissioning_date`, and `local_authority`.
    """
    fit = pl.read_parquet(OUTSIDE_DIR / "fit_solar_pv_installations.parquet").select(
        "postcode",
        "installed_capacity_kw",
        "declared_net_capacity_kw",
        "commissioning_date",
        "local_authority",
    )
    shares = pl.read_parquet(OUTSIDE_DIR / "fit_district_polygon_shares.parquet").select(
        "district", pl.col("time_series_id").alias("polygon_id"), "share"
    )
    return fit.join(shares, left_on="postcode", right_on="district", how="inner").select(
        "polygon_id",
        "share",
        "commissioning_date",
        "local_authority",
        declared_net_mw=pl.col("declared_net_capacity_kw") * pl.col("share") / 1000.0,
        small=pl.col("installed_capacity_kw") < SMALL_INSTALLATION_KW,
    )


def fit_curve(*, rows: pl.DataFrame, polygon: Polygon) -> Curve:
    """Return the cumulative declared net capacity of the FiT register under one polygon."""
    own = rows.filter(pl.col("polygon_id") == polygon.polygon_id).sort("commissioning_date")
    commissioned = own["commissioning_date"].dt.epoch("d").to_numpy()
    cumulative = own["declared_net_mw"].cum_sum().to_numpy()
    days = np.arange(day_number(day=FIT_SCHEME_START), day_number(day=FIT_REPORT_END) + 1)
    index = np.searchsorted(commissioned, days, side="right")
    levels = np.where(index > 0, cumulative[np.maximum(index - 1, 0)], 0.0)
    return Curve(days=days, megawatts=levels)


def read_local_authority_year(*, year: int) -> dict[str, float]:
    """Return DESNZ's solar PV capacity in MW by local-authority name at the end of `year`."""
    rows = fetch.read_workbook_rows(
        path=DESNZ_LOCAL_AUTHORITY_WORKBOOK, sheet_name=f"LA - Capacity, {year}"
    )
    header_index = next(
        i for i, row in enumerate(rows) if row and (row[0] or "").startswith("Local Authority Code")
    )
    solar_column = rows[header_index].index("Solar PV")
    return {
        str(row[1]).strip(): float(row[solar_column] or 0.0)
        for row in rows[header_index + 1 :]
        if len(row) > solar_column and row[0] is not None and re.fullmatch(r"[EWSN]\d{8}", row[0])
    }


def local_authority_fractions(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return, per polygon and local authority, the share of the authority's FiT capacity inside.

    DESNZ publishes no boundary-based allocation of an authority to a polygon, and this project
    holds no local-authority boundaries. The share of the authority's FiT declared net capacity
    that lies inside the polygon (through the postcode-district shares) stands in for the share of
    the authority's solar capacity that does.
    """
    inside = rows.group_by("polygon_id", "local_authority").agg(
        inside_mw=pl.col("declared_net_mw").sum()
    )
    fit = pl.read_parquet(OUTSIDE_DIR / "fit_solar_pv_installations.parquet")
    total = fit.group_by("local_authority").agg(
        total_mw=pl.col("declared_net_capacity_kw").sum() / 1000.0
    )
    return inside.join(total, on="local_authority", how="left").with_columns(
        fraction=pl.col("inside_mw") / pl.col("total_mw")
    )


def local_authority_curve(
    *, polygon: Polygon, fractions: pl.DataFrame, by_year: dict[int, dict[str, float]]
) -> tuple[Curve, list[str]]:
    """Return the polygon's DESNZ local-authority capacity at each year end, and unmatched names."""
    own = fractions.filter(pl.col("polygon_id") == polygon.polygon_id)
    unmatched = [
        name
        for name in own["local_authority"].to_list()
        if name not in by_year[LOCAL_AUTHORITY_YEARS[-1]]
    ]
    levels = [
        sum(
            row["fraction"] * by_year[year].get(row["local_authority"], 0.0)
            for row in own.iter_rows(named=True)
        )
        for year in LOCAL_AUTHORITY_YEARS
    ]
    days = np.array([day_number(day=date(year, 12, 31)) for year in LOCAL_AUTHORITY_YEARS])
    return Curve(days=days, megawatts=np.array(levels)), unmatched


def read_constituency_table(*, sheet_name: str) -> dict[str, float]:
    """Return DESNZ's domestic installed solar capacity in MW by constituency code."""
    rows = fetch.read_workbook_rows(path=DESNZ_CONSTITUENCY_WORKBOOK, sheet_name=sheet_name)
    return {
        str(row[0]): float(row[3] or 0.0)
        for row in rows
        if len(row) >= 5
        and row[0] is not None
        and re.fullmatch(fetch.CONSTITUENCY_CODE_PATTERN, row[0])
    }


def constituency_membership() -> pl.DataFrame:
    """Return the share of each constituency inside each polygon."""
    return pl.read_parquet(OUTSIDE_DIR / "constituency_primary_membership.parquet").select(
        "time_series_id", "PCON24CD", "share_of_feature_inside_polygon"
    )


def constituency_level(*, polygon: Polygon, capacities: dict[str, float]) -> float:
    """Return the domestic capacity in MW inside a polygon, each constituency scaled by area."""
    own = constituency_membership().filter(pl.col("time_series_id") == polygon.polygon_id)
    return sum(
        row["share_of_feature_inside_polygon"] * capacities.get(row["PCON24CD"], 0.0)
        for row in own.iter_rows(named=True)
    )


def constituency_curve(*, polygon: Polygon) -> Curve:
    """Return the domestic capacity inside a polygon at the two month ends the workbook holds."""
    levels = [
        constituency_level(polygon=polygon, capacities=read_constituency_table(sheet_name=sheet))
        for sheet, _ in CONSTITUENCY_SNAPSHOTS.values()
    ]
    days = [day_number(day=day) for _, day in CONSTITUENCY_SNAPSHOTS.values()]
    return Curve(days=np.array(days), megawatts=np.array(levels))


def growth_row(
    *,
    polygon: Polygon,
    source: str,
    window_name: str,
    curve: Curve,
    start: date,
    end: date,
    series_span: tuple[date, date],
    status: str,
    limits: str,
) -> dict[str, object]:
    """Return one row of the growth table for a source's curve over a window."""
    days = mondays(start=start, end=end)
    base: dict[str, object] = {
        "primary_name": polygon.name,
        "polygon_id": polygon.polygon_id,
        "source": source,
        "window": window_name,
        "window_start": start,
        "window_end": end,
        "window_years": (end - start).days / 365.25,
        "series_first_week": series_span[0],
        "series_last_day": series_span[1],
        "source_status": status,
        "limits": limits,
    }
    if days.size < 2:
        return base | {"growth_per_year": None, "slope_mw_per_year": None, "mean_level_mw": None}
    fractional, slope, level = growth_per_year(days=days, megawatts=curve.at(days=days))
    return base | {
        "growth_per_year": None if np.isnan(fractional) else fractional,
        "slope_mw_per_year": slope,
        "mean_level_mw": level,
    }


FIT_LIMITS: Final[str] = (
    "Declared net (AC) capacity of FiT installations of every size up to 5 MW, commissioned "
    "2010-04-01 to 2021-03-24. The scheme closed to new applicants on 2019-03-31, so later rows "
    "are late accreditations and extensions: a floor from then on. Allocation to the polygon by "
    "the share of each postcode district's units inside it. Excludes every installation that "
    "never took the tariff."
)
LA_LIMITS: Final[str] = (
    "DESNZ solar PV of every size at each year end 2014 to 2025 for the whole local authority, "
    "scaled to the polygon by the share of the authority's FiT capacity inside it (no authority "
    "boundaries are held). Includes the farms of 1 MW or more that NGED meters and the "
    "Disaggregated Demand series does not contain, so its growth can exceed the series' growth. "
    "DESNZ revises recent years upward."
)
CONSTITUENCY_LIMITS: Final[str] = (
    "DESNZ domestic installations only, area-scaled to the polygon. The workbook holds a "
    "snapshot for August 2026 and the previous release for July 2026, no longer history, and "
    "DESNZ revises recent months upward as late notifications arrive. One month of growth is "
    "not a growth rate."
)
ECR_LIMITS: Final[str] = (
    "Connected solar of 50 kW to under 1 MW. The register's only date column, `Last Updated`, "
    "holds the release date of the register for every row, not a connection date, so no growth "
    "rate can be read from it. Misses systems under 50 kW."
)


def growth_table() -> pl.DataFrame:
    """Build the growth table from the outside sources and the dates of the weekly series."""
    spans = series_spans()
    rows = fit_rows()
    fractions = local_authority_fractions(rows=rows)
    by_year = {year: read_local_authority_year(year=year) for year in LOCAL_AUTHORITY_YEARS}
    out: list[dict[str, object]] = []
    for polygon in POLYGONS:
        span = spans[polygon.name]
        sources: list[tuple[str, Curve, date, date, str, str, list[tuple[str, date, date]]]] = []
        fit = fit_curve(rows=rows, polygon=polygon)
        sources.append(
            (
                "fit",
                fit,
                FIT_SCHEME_START,
                FIT_REPORT_END,
                "partial; a floor from 2019-04",
                FIT_LIMITS,
                [("pre_2019", *FIT_PRE_2019_WINDOW)],
            )
        )
        authority, _ = local_authority_curve(polygon=polygon, fractions=fractions, by_year=by_year)
        sources.append(
            (
                "desnz_local_authority",
                authority,
                date(LOCAL_AUTHORITY_YEARS[0], 12, 31),
                date(LOCAL_AUTHORITY_YEARS[-1], 12, 31),
                "all sizes; includes metered farms of 1 MW or more",
                LA_LIMITS,
                [],
            )
        )
        constituency = constituency_curve(polygon=polygon)
        sources.append(
            (
                "desnz_constituency_domestic",
                constituency,
                date(2026, 7, 31),
                date(2026, 8, 31),
                "domestic only; two monthly snapshots",
                CONSTITUENCY_LIMITS,
                [],
            )
        )
        for source, curve, first, last, status, limits, extra in sources:
            windows = [("source_coverage", first, last), *extra]
            overlap = (max(span[0], first), min(span[1], last))
            if overlap[0] < overlap[1]:
                windows.append(("series_span", *overlap))
            else:
                windows.append(("series_span", span[0], span[0]))
            for name, start, end in windows:
                out.append(
                    growth_row(
                        polygon=polygon,
                        source=source,
                        window_name=name,
                        curve=curve,
                        start=start,
                        end=end,
                        series_span=span,
                        status=status,
                        limits=limits,
                    )
                )
        out.append(
            {
                "primary_name": polygon.name,
                "polygon_id": polygon.polygon_id,
                "source": "ecr",
                "window": "none",
                "window_start": None,
                "window_end": None,
                "window_years": None,
                "series_first_week": span[0],
                "series_last_day": span[1],
                "growth_per_year": None,
                "slope_mw_per_year": None,
                "mean_level_mw": None,
                "source_status": "partial; no connection date",
                "limits": ECR_LIMITS,
            }
        )
    return pl.DataFrame(out, infer_schema_length=None)


def print_ecr_dates() -> None:
    """Print what the register's date columns record, before any ECR comparison."""
    ecr = reader.read_ecr()
    print("ECR date columns: the only date-like column is `last_updated`.")
    print(ecr["last_updated"].value_counts())
    serial = int(ecr["last_updated"].drop_nulls().unique().to_list()[0])
    print(
        f"ECR `last_updated` holds one value for all {ecr.height} rows: Excel serial {serial} = "
        f"{date(1899, 12, 30) + timedelta(days=serial)}, the register's release date, "
        "not a connection date."
    )


def run_growth() -> None:
    """Write `outside_growth.parquet` and print it."""
    print_ecr_dates()
    table = growth_table()
    table.write_parquet(GROWTH_PATH)
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=40, tbl_width_chars=250):
        print(
            table.select(
                "primary_name",
                "source",
                "window",
                "window_start",
                "window_end",
                pl.col("window_years").round(2),
                pl.col("mean_level_mw").round(2),
                pl.col("slope_mw_per_year").round(3),
                pl.col("growth_per_year").round(3),
                "series_first_week",
            )
        )
    print(f"Wrote {GROWTH_PATH}")


GROWTH_FACTOR: Final[float] = 2.0
LEVEL_RATIO_RANGE: Final[tuple[float, float]] = (0.67, 1.5)
AC_TO_DC: Final[float] = 0.70
AC_TO_DC_RANGE: Final[tuple[float, float]] = (0.60, 0.80)
MIN_LEVEL_WEEKS: Final[int] = 26
MIN_SEASON_YEAR_WEEKS: Final[int] = 40
CONSTITUENCY_LEVEL_END: Final[date] = date(2026, 7, 31)
LOCAL_AUTHORITY_LEVEL_END: Final[date] = date(2025, 12, 31)
FIT_BASELINE_DATE: Final[date] = date(2019, 3, 31)
BASELINE_WEEKS: Final[int] = 8
ANCHOR_PASS_SHARE: Final[float] = 0.8
ECR_PASS_COUNT: Final[int] = 5
REPORT_ONLY: Final[frozenset[str]] = frozenset({"Kirton"})
"""Kirton has status range-only and is reported without a pass or fail."""

CHECK_COLUMNS: Final[dict[str, type[pl.DataType]]] = {
    "check": pl.String,
    "variant": pl.String,
    "primary_name": pl.String,
    "source": pl.String,
    "season_year": pl.Int64,
    "series_value": pl.Float64,
    "outside_value": pl.Float64,
    "ratio": pl.Float64,
    "tolerance_low": pl.Float64,
    "tolerance_high": pl.Float64,
    "result": pl.String,
    "note": pl.String,
}


def check_row(
    *,
    check: str,
    variant: str,
    primary_name: str,
    source: str,
    result: str,
    note: str,
    season_year: int | None = None,
    series_value: float | None = None,
    outside_value: float | None = None,
    tolerance: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Return one row of the checks table; `ratio` is the series value over the outside value."""
    ratio = (
        series_value / outside_value
        if series_value is not None and outside_value is not None and abs(outside_value) > 1e-9
        else None
    )
    return {
        "check": check,
        "variant": variant,
        "primary_name": primary_name,
        "source": source,
        "season_year": season_year,
        "series_value": series_value,
        "outside_value": outside_value,
        "ratio": ratio,
        "tolerance_low": None if tolerance is None else tolerance[0],
        "tolerance_high": None if tolerance is None else tolerance[1],
        "result": "report only"
        if primary_name in REPORT_ONLY and result in ("pass", "fail")
        else result,
        "note": note,
    }


def weekly_series(*, weekly: pl.DataFrame, variant: str, polygon: Polygon) -> pl.DataFrame:
    """Return the polygon's weekly capacity, summed over its series, for weeks all of them hold."""
    return (
        weekly.filter(
            (pl.col("variant") == variant) & pl.col("time_series_id").is_in(polygon.series_ids)
        )
        .group_by("week_start")
        .agg(
            n=pl.len(),
            capacity_mw=pl.col("capacity_mw").sum(),
            low_mw=pl.col("low_mw").sum(),
            high_mw=pl.col("high_mw").sum(),
        )
        .filter(pl.col("n") == len(polygon.series_ids))
        .sort("week_start")
        .drop("n")
    )


def window_values(
    *, series: pl.DataFrame, start: date, end: date, column: str = "capacity_mw"
) -> pl.DataFrame:
    """Return the series' weeks that start in [`start`, `end`]."""
    return series.filter((pl.col("week_start") >= start) & (pl.col("week_start") <= end))


def growth_result(*, series_growth: float, outside_growth: float, source: str) -> tuple[str, str]:
    """Apply the growth tolerance of the plan to one source and return the result and its reason."""
    ratio = series_growth / outside_growth if outside_growth > 0 else float("nan")
    if source != "desnz_local_authority" or not outside_growth > 0:
        return "not testable", "the plan states that this source cannot be tested for growth"
    if ratio > GROWTH_FACTOR:
        return "fail", "the series grows more than twice as fast as the source"
    if series_growth > 0 and ratio >= 1 / GROWTH_FACTOR:
        return "pass", "the series' growth is within a factor of 2 of the source's"
    reason = (
        "the series grows less than half as fast as the source, or falls; the source includes "
        "the metered farms of 1 MW or more that the series does not hold"
    )
    return "inconclusive", reason


def growth_checks(*, weekly: pl.DataFrame, growth: pl.DataFrame) -> list[dict[str, object]]:
    """Compare the series' growth per year with each outside source over the series' window."""
    rows = []
    for variant in VARIANTS:
        for polygon in POLYGONS:
            series = weekly_series(weekly=weekly, variant=variant, polygon=polygon)
            for source in ("fit", "desnz_local_authority", "desnz_constituency_domestic"):
                outside = growth.filter(
                    (pl.col("primary_name") == polygon.name)
                    & (pl.col("source") == source)
                    & (pl.col("window") == "series_span")
                ).row(0, named=True)
                window_years = outside["window_years"] or 0.0
                if window_years < MIN_WINDOW_YEARS or outside["growth_per_year"] is None:
                    rows.append(
                        check_row(
                            check="growth",
                            variant=variant,
                            primary_name=polygon.name,
                            source=source,
                            result="not testable",
                            note=f"window of {window_years:.2f} years is under 1 year",
                            outside_value=outside["growth_per_year"],
                        )
                    )
                    continue
                inside = window_values(
                    series=series, start=outside["window_start"], end=outside["window_end"]
                )
                days = inside["week_start"].dt.epoch("d").to_numpy()
                series_growth, _, _ = growth_per_year(
                    days=days, megawatts=inside["capacity_mw"].to_numpy()
                )
                result, reason = growth_result(
                    series_growth=series_growth,
                    outside_growth=outside["growth_per_year"],
                    source=source,
                )
                rows.append(
                    check_row(
                        check="growth",
                        variant=variant,
                        primary_name=polygon.name,
                        source=source,
                        result=result,
                        note=(
                            f"window {outside['window_start']} to {outside['window_end']}; {reason}"
                        ),
                        series_value=series_growth,
                        outside_value=outside["growth_per_year"],
                        tolerance=(1 / GROWTH_FACTOR, GROWTH_FACTOR),
                    )
                )
    return rows


def fit_baseline_checks(*, weekly: pl.DataFrame) -> list[dict[str, object]]:
    """Report the FiT baseline against the series' first weeks and the share under 50 kW.

    Args:
        weekly: The weekly capacity table.

    Returns:
        Report-only rows of the checks table.
    """
    rows = fit_rows()
    out = []
    for variant in VARIANTS:
        for polygon in POLYGONS:
            series = weekly_series(weekly=weekly, variant=variant, polygon=polygon)
            curve = fit_curve(rows=rows, polygon=polygon)
            baseline = float(curve.at(days=np.array([day_number(day=FIT_BASELINE_DATE)]))[0])
            first = float(series["capacity_mw"].head(BASELINE_WEEKS).to_numpy().mean())
            gap = (min(series["week_start"].to_list()) - FIT_BASELINE_DATE).days / 365.25
            out.append(
                check_row(
                    check="fit_baseline",
                    variant=variant,
                    primary_name=polygon.name,
                    source="fit",
                    result="report only",
                    note=(
                        f"FiT declared net MW commissioned to {FIT_BASELINE_DATE} against the "
                        f"series' first {BASELINE_WEEKS} weeks, which start {gap:.1f} years later; "
                        "the FiT is a partial count (under 5 MW, FiT-accredited), "
                        "so a ratio above 1 is expected and a ratio far below 1 is not"
                    ),
                    series_value=first,
                    outside_value=baseline,
                )
            )
    for polygon in POLYGONS:
        before = rows.filter(
            (pl.col("polygon_id") == polygon.polygon_id)
            & (pl.col("commissioning_date") < FIT_FLOOR_FROM)
        )
        total = float(before["declared_net_mw"].sum())
        small = float(before.filter(pl.col("small"))["declared_net_mw"].sum())
        out.append(
            check_row(
                check="fit_small_share",
                variant="all",
                primary_name=polygon.name,
                source="fit",
                result="report only",
                note=(
                    "share of FiT declared net capacity commissioned before 2019-04 that sits in "
                    "installations under 50 kW installed, which the Embedded Capacity Register "
                    f"misses; FiT total {total:.2f} MW"
                ),
                outside_value=small / total if total > 0 else None,
            )
        )
    return out


def level_checks(*, weekly: pl.DataFrame) -> list[dict[str, object]]:
    """Compare the series' 12-month mean with DESNZ capacity times an AC-to-DC ratio of 0.70."""
    constituency = read_constituency_table(
        sheet_name=CONSTITUENCY_SNAPSHOTS["Table_3_dom_by_PC"][0]
    )
    rows = fit_rows()
    fractions = local_authority_fractions(rows=rows)
    by_year = {LOCAL_AUTHORITY_YEARS[-1]: read_local_authority_year(year=LOCAL_AUTHORITY_YEARS[-1])}
    out = []
    for variant in VARIANTS:
        for polygon in POLYGONS:
            series = weekly_series(weekly=weekly, variant=variant, polygon=polygon)
            authority, _ = local_authority_curve(
                polygon=polygon,
                fractions=fractions,
                by_year=dict.fromkeys(LOCAL_AUTHORITY_YEARS, by_year[LOCAL_AUTHORITY_YEARS[-1]]),
            )
            cases = (
                (
                    "desnz_constituency_domestic",
                    constituency_level(polygon=polygon, capacities=constituency),
                    CONSTITUENCY_LEVEL_END,
                    "domestic installations only, area-scaled",
                ),
                (
                    "desnz_local_authority",
                    float(authority.megawatts[-1]),
                    LOCAL_AUTHORITY_LEVEL_END,
                    "all sizes, scaled by FiT share; includes metered farms of 1 MW or more",
                ),
            )
            for source, dc_mw, end, description in cases:
                start = end - timedelta(days=364)
                inside = window_values(series=series, start=start, end=end)
                if inside.height < MIN_LEVEL_WEEKS:
                    out.append(
                        check_row(
                            check="level",
                            variant=variant,
                            primary_name=polygon.name,
                            source=source,
                            result="not testable",
                            note=f"{inside.height} weeks of series in the 12 months to {end}",
                            outside_value=dc_mw * AC_TO_DC,
                        )
                    )
                    continue
                mean = float(inside["capacity_mw"].to_numpy().mean())
                ratio = mean / (dc_mw * AC_TO_DC)
                low, high = (mean / (dc_mw * r) for r in AC_TO_DC_RANGE)
                passed = LEVEL_RATIO_RANGE[0] <= ratio <= LEVEL_RATIO_RANGE[1]
                out.append(
                    check_row(
                        check="level",
                        variant=variant,
                        primary_name=polygon.name,
                        source=source,
                        result="pass" if passed else "fail",
                        note=(
                            f"{description}; DESNZ DC {dc_mw:.2f} MW; series mean of "
                            f"{inside.height} weeks to {end}; ratio at AC-to-DC 0.80 is "
                            f"{low:.2f} and at 0.60 is {high:.2f}"
                            + (
                                "; Wrangle T1 is not-clean"
                                if polygon.name.startswith("Wrangle")
                                else ""
                            )
                            + (
                                "; the table is a partial count, so a ratio above 1 is the less "
                                "alarming direction"
                                if not passed and ratio > 1 and source != "desnz_local_authority"
                                else ""
                            )
                            + (
                                "; the authority total includes metered farms of 1 MW or more, so "
                                "a ratio below 1 is expected and is not evidence against the series"
                                if not passed and ratio < 1 and source == "desnz_local_authority"
                                else ""
                            )
                        ),
                        series_value=mean,
                        outside_value=dc_mw * AC_TO_DC,
                        tolerance=LEVEL_RATIO_RANGE,
                    )
                )
    return out


def ecr_totals() -> dict[str, tuple[float, int]]:
    """Return, per polygon name, the register's connected 50 kW to 1 MW solar in MW and the rows."""
    ecr = reader.read_ecr().filter(
        (pl.col("connection_status") == "Connected")
        & (pl.col("energy_source_1") == "Solar")
        & (pl.col("sheet") == SMALL_SHEET)
        & pl.col("connected_registered_capacity_mw").is_not_null()
    )
    ecr = ecr.with_columns(
        primary_key=pl.col("primary_substation")
        .fill_null("")
        .map_elements(normalise, return_dtype=pl.String)
    )
    names = pl.read_parquet(METADATA_PATH).select("time_series_id", "time_series_name")
    out = {}
    for polygon in POLYGONS:
        name = names.filter(pl.col("time_series_id") == polygon.series_ids[0])["time_series_name"][
            0
        ]
        key = normalise(name.removesuffix(" T1").removesuffix(" T2"))
        matched = ecr.filter(pl.col("primary_key") == key)
        out[polygon.name] = (
            float(matched["connected_registered_capacity_mw"].sum()),
            matched.height,
        )
    return out


def ecr_checks(*, weekly: pl.DataFrame) -> list[dict[str, object]]:
    """Check the register's total stays under the series' upper 90% bound at 5 of 6 Primaries."""
    totals = ecr_totals()
    out = []
    for variant in VARIANTS:
        passes = 0
        scored = 0
        for polygon in POLYGONS:
            series = weekly_series(weekly=weekly, variant=variant, polygon=polygon)
            last = series.tail(52)
            upper = float(last["high_mw"].to_numpy().mean())
            mean = float(last["capacity_mw"].to_numpy().mean())
            total, matched = totals[polygon.name]
            if matched == 0:
                result = "pass"
                note = (
                    "the register holds no connected solar row of 50 kW to under 1 MW for the "
                    "Primary, so its total of 0 MW cannot exceed the bound"
                )
            else:
                within = total <= upper
                result = "pass" if within else "fail"
                note = (
                    f"{matched} connected solar rows of 50 kW to under 1 MW; compared with the "
                    f"upper 90% bound averaged over the last {last.height} weeks; the series "
                    f"mean is {mean:.2f} MW; the register's date column holds only its release date"
                    + ("; Wrangle T1 is not-clean" if polygon.name.startswith("Wrangle") else "")
                )
            if polygon.name not in REPORT_ONLY and result in ("pass", "fail"):
                scored += 1
                passes += result == "pass"
            out.append(
                check_row(
                    check="ecr",
                    variant=variant,
                    primary_name=polygon.name,
                    source="ecr",
                    result=result,
                    note=note,
                    series_value=upper,
                    outside_value=total,
                    tolerance=None,
                )
            )
        out.append(
            check_row(
                check="ecr_overall",
                variant=variant,
                primary_name="all scored",
                source="ecr",
                result="pass" if passes >= ECR_PASS_COUNT else "fail",
                note=f"{passes} of {scored} comparisons have the register at or below the bound",
                series_value=float(passes),
                outside_value=float(ECR_PASS_COUNT),
            )
        )
    return out


def anchor_checks(*, weekly: pl.DataFrame) -> list[dict[str, object]]:
    """Check the season-year mean of the series lies inside experiment 00040's 95% interval."""
    scores = pl.read_parquet(EXPERIMENT_00040_SCORES_PATH)
    site_of_id = {series_id: site for site, series_id in common.PRIMARY_SERIES_IDS.items()}
    out = []
    for variant in VARIANTS:
        inside_count = 0
        total = 0
        clean = weekly.filter((pl.col("variant") == variant) & (pl.col("status") == "clean"))
        for series_id in sorted(clean["time_series_id"].unique().to_list()):
            site = site_of_id[series_id]
            name = common.PRIMARY_NAMES[site]
            own = clean.filter(pl.col("time_series_id") == series_id).with_columns(
                season_year=pl.col("week_start").dt.year()
                - (pl.col("week_start").dt.month() < 9).cast(pl.Int32)
            )
            for year, group in own.group_by("season_year", maintain_order=True):
                reference = scores.filter((pl.col("site") == site) & (pl.col("year") == year[0]))
                if group.height < MIN_SEASON_YEAR_WEEKS or reference.is_empty():
                    continue
                row = reference.row(0, named=True)
                mean = float(group["capacity_mw"].to_numpy().mean())
                inside = row["headline_low"] <= mean <= row["headline_high"]
                total += 1
                inside_count += inside
                out.append(
                    check_row(
                        check="season_year_anchor",
                        variant=variant,
                        primary_name=name,
                        source="experiment_00040",
                        season_year=int(year[0]),
                        result="pass" if inside else "fail",
                        note=(
                            f"{group.height} weeks; 00040 interval {row['headline_low']:.2f} to "
                            f"{row['headline_high']:.2f} MW, six-arm C {row['C']:.2f}, shape range "
                            f"{row['range_low']:.2f} to {row['range_high']:.2f}"
                        ),
                        series_value=mean,
                        outside_value=float(row["headline"]),
                        tolerance=(float(row["headline_low"]), float(row["headline_high"])),
                    )
                )
        share = inside_count / total if total else float("nan")
        out.append(
            check_row(
                check="season_year_anchor_overall",
                variant=variant,
                primary_name="all clean",
                source="experiment_00040",
                result="pass" if share >= ANCHOR_PASS_SHARE else "fail",
                note=f"{inside_count} of {total} Primary-years inside the interval",
                series_value=share,
                outside_value=ANCHOR_PASS_SHARE,
            )
        )
    return out


def run_compare() -> None:
    """Run every comparison, write `outside_checks.parquet`, and write and print the report."""
    weekly = pl.read_parquet(WEEKLY_PATH).filter(pl.col("variant").is_in(VARIANTS))
    growth = pl.read_parquet(GROWTH_PATH)
    rows = [
        *growth_checks(weekly=weekly, growth=growth),
        *fit_baseline_checks(weekly=weekly),
        *level_checks(weekly=weekly),
        *ecr_checks(weekly=weekly),
        *anchor_checks(weekly=weekly),
    ]
    checks = pl.DataFrame(rows, schema=CHECK_COLUMNS)
    checks.write_parquet(CHECKS_PATH)
    lines = [
        "Outside checks of the weekly capacity series (experiment 00045, U5b).",
        "Every outside source is partial, so every check is descriptive and decides nothing.",
        "Series: variants alt_post_hoc and season_year_constant.",
        "Kirton (range-only) is report only;",
        "Wrangle T1 (not-clean) is summed with Wrangle T2 against the shared polygon.",
        "",
    ]
    with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=260):
        for name in checks["check"].unique(maintain_order=True):
            part = checks.filter(pl.col("check") == name)
            lines.append(f"== {name} ==")
            lines.append(
                str(
                    part.select(
                        "variant",
                        "primary_name",
                        "source",
                        "season_year",
                        pl.col("series_value").round(3),
                        pl.col("outside_value").round(3),
                        pl.col("ratio").round(2),
                        "result",
                    )
                )
            )
            lines.append("")
        lines.append("== result counts ==")
        lines.append(
            str(
                checks.group_by("check", "variant", "result")
                .len()
                .sort("check", "variant", "result")
            )
        )
        lines.append("")
        lines.append("== notes (first row of each check and source) ==")
        firsts = checks.unique(subset=["check", "source"], keep="first", maintain_order=True)
        lines.extend(
            f"{row['check']} / {row['source']}: {row['note']}"
            for row in firsts.iter_rows(named=True)
        )
    report = "\n".join(lines)
    REPORT_PATH.write_text(report + "\n")
    print(report)
    print(f"Wrote {CHECKS_PATH} and {REPORT_PATH}")


def main() -> None:
    """Run the requested stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("growth", "compare"), required=True)
    stage = parser.parse_args().stage
    if stage == "growth":
        run_growth()
    else:
        run_compare()


if __name__ == "__main__":
    main()
