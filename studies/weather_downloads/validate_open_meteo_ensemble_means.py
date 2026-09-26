"""Validate the combined parquet of each Open-Meteo ensemble-mean product.

One-off throwaway script for `fetch_open_meteo_ensemble_means.py`. For each product's combined file
it checks the nine site labels, duplicate `(site, time)` keys, hourly spacing and gaps per site,
that a series is either entirely null (listed in the product's `always_null`) or has no nulls at
all, physical ranges, and that every spread is non-negative. It prints one PASS or FAIL line per
check and the null fraction of every series, prints no coordinate, and exits non-zero on a failure.

Run it with `uv run python studies/weather_downloads/validate_open_meteo_ensemble_means.py`.
"""

import logging
import sys
from datetime import date, datetime, time, timedelta
from typing import Final

import polars as pl
from fetch_open_meteo_ensemble_means import (
    FIRST_DATE,
    PRODUCTS,
    SERIES,
    EnsembleMeanProduct,
    _combined_path,
)
from studies.anonymise import SITE_LABELS, WIND_SITE_LABELS

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("validate_open_meteo_ensemble_means")

EXPECTED_SITES: Final[frozenset[str]] = frozenset(SITE_LABELS + WIND_SITE_LABELS)

RANGES: Final[dict[str, tuple[float, float]]] = {
    "shortwave_radiation": (0.0, 1400.0),
    "direct_radiation": (0.0, 1400.0),
    "diffuse_radiation": (0.0, 1400.0),
    "temperature_2m": (-40.0, 50.0),
    "wind_speed_10m": (0.0, 200.0),
    "wind_speed_100m": (0.0, 250.0),
    "wind_direction_100m": (0.0, 360.0),
    "cloud_cover": (0.0, 100.0),
}
"""Physical bounds of each mean. A value outside them is a fault, not an extreme."""


def _window_failures(
    *, frame: pl.DataFrame, expected_window: tuple[date, date] | None
) -> list[str]:
    """Return the failures of the shared-bounds and expected-window checks."""
    bounds = frame.group_by("site").agg(
        pl.col("time").min().alias("first"), pl.col("time").max().alias("last")
    )
    if bounds["first"].n_unique() != 1 or bounds["last"].n_unique() != 1:
        return ["sites do not share the same first and last time"]
    if expected_window is None:
        return []
    first = datetime.combine(expected_window[0], time(0))
    last = datetime.combine(expected_window[1], time(23))
    if (bounds["first"][0], bounds["last"][0]) != (first, last):
        return [f"time range is not {first}..{last}"]
    return []


def warn_on_shared_series(*, frame: pl.DataFrame, product: EnsembleMeanProduct) -> list[str]:
    """Return one sentence per constant or cross-site duplicated series. Warnings, not failures.

    Two sites in one grid cell legitimately share a series, and a coarse product can be constant
    over a short window, so neither is a failure.

    Args:
        frame: One product's combined frame.
        product: The product, whose `always_null` series are skipped.

    Returns:
        The warnings, none of which contains a coordinate.
    """
    warnings = []
    for name in SERIES:
        if name in product.always_null:
            continue
        wide = frame.pivot(on="site", index="time", values=name).sort("time").drop("time")
        constant = [site for site in wide.columns if wide[site].n_unique() <= 1]
        if constant:
            warnings.append(f"{name} is constant at sites {sorted(constant)}")
        sites = wide.columns
        warnings.extend(
            f"{name} is identical at sites {first} and {second}"
            for position, first in enumerate(sites)
            for second in sites[position + 1 :]
            if wide[first].equals(wide[second])
        )
    return warnings


def validate_frame(
    *, frame: pl.DataFrame, product: EnsembleMeanProduct, expected_window: tuple[date, date] | None
) -> list[str]:
    """Return one sentence per failed check, empty if the frame passes.

    Args:
        frame: One product's combined frame, with `site`, `time`, and every entry of `SERIES`.
        product: The product the frame belongs to, for its `always_null` series.
        expected_window: The first and last date the frame must cover (24 hours on each), or
            `None` to check only that every site shares one first and last time.

    Returns:
        The failures, none of which contains a coordinate.
    """
    failures = _window_failures(frame=frame, expected_window=expected_window)
    sites = set(frame["site"].unique())
    if sites != EXPECTED_SITES:
        failures.append(f"site labels are {sorted(sites)}, expected {sorted(EXPECTED_SITES)}")
    if frame.select(pl.struct("site", "time").is_duplicated().any()).item():
        failures.append("duplicate (site, time) keys")
    per_site = frame.group_by("site").agg(
        pl.col("time").min().alias("first"),
        pl.col("time").max().alias("last"),
        pl.col("time").n_unique().alias("n"),
        pl.col("time").sort().diff().drop_nulls().unique().alias("steps"),
    )
    for row in per_site.iter_rows(named=True):
        expected = int((row["last"] - row["first"]).total_seconds() // 3600) + 1
        if row["n"] != expected or set(row["steps"]) != {timedelta(hours=1)}:
            failures.append(f"site {row['site']} has gaps or a step other than one hour")
    for name in SERIES:
        nulls = frame[name].null_count()
        expect_null = name in product.always_null
        if expect_null and nulls != frame.height:
            failures.append(f"{name} is expected all null but has values")
        if not expect_null and nulls:
            failures.append(f"{name} has {nulls / frame.height:.3f} null fraction")
        if name.endswith("_spread") and (frame[name].drop_nulls() < 0).any():
            failures.append(f"{name} has negative values")
        low, high = RANGES.get(name, (None, None))
        if low is not None and high is not None:
            outside = frame.filter((pl.col(name) < low) | (pl.col(name) > high))
            if not outside.is_empty():
                failures.append(f"{name} has values outside {low}..{high}")
    return failures


def main() -> int:
    """Validate every product's combined file that exists, and return a process exit code."""
    failed = False
    for product in PRODUCTS.values():
        path = _combined_path(product=product)
        if not path.exists():
            _LOG.info("SKIP %s: %s not written yet", product.output_dir, path.name)
            continue
        frame = pl.read_parquet(path)
        expected_window = (FIRST_DATE, frame.select(pl.col("time").dt.date().max()).item())
        failures = validate_frame(frame=frame, product=product, expected_window=expected_window)
        _LOG.info(
            "%s %s (%d rows)", "FAIL" if failures else "PASS", product.output_dir, frame.height
        )
        for failure in failures:
            _LOG.info("  %s", failure)
        for warning in warn_on_shared_series(frame=frame, product=product):
            _LOG.info("  WARN %s", warning)
        nulls = {name: round(frame[name].null_count() / frame.height, 3) for name in SERIES}
        _LOG.info("  null fractions: %s", nulls)
        failed = failed or bool(failures)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
