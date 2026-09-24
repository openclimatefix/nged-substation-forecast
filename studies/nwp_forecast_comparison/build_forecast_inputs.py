"""Build the per-technology weather-arm frames the matched-lead comparison fits on.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. Reads:

- **Power, the site list, and the shared rows**: `ens_forecast_horizons.base_frame`, the same
  hourly rows (commissioning ramp, zero half-hours and outages already dropped) the ENS-horizons
  page builds its own inputs from, made public in that script for this reuse.
- **Previous Runs products**: each product's `previous_runs/combined.parquet`, already carrying the
  anonymised `site` label. UKV's radiation is rebuilt from its two label-adjacent snapshots
  (`studies.hourly_means.hourly_from_snapshots`), because V3 reads UKV as an instantaneous snapshot
  rather than an hour-ending mean; every other product's radiation is used as served.
- **ECMWF ENS**: `ens_forecast_horizons.build_inputs` and `.main_frame`, at the upsampling
  combination the ENS horizons page's rule chose before this study existed: `clear_sky` for solar
  (radiation through the clear-sky index, temperature by straight line) and `speed_components` for
  wind. This study reads that choice from `UPSAMPLING_METHODS` rather than re-running the rule.
- **NOAA GEFS**: gated on `data/studies/weather/GEFS/_month_cache/` holding every month from
  2024-11 to the month before today, or on `--gefs-window-dir` for a `GEFS_window_*` test extract
  during development. Neither is complete yet, so this pass writes no GEFS columns.

Every output row carries only the anonymised `site` label; no generator name, id or coordinate is
read from the private roster in this script, except inside `studies.grid_sampling` (GEFS's
nearest-cell match), which never prints what it reads.

Run it with `uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --output-dir
<dir>`.
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import polars as pl
from contracts.settings import PROJECT_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_previous_runs_leads import PRODUCT_DIRS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beam_diffuse_split"))
import ens_forecast_horizons as efh
from studies.hourly_means import hourly_from_snapshots

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DomainType = Literal["solar", "wind"]

DEFAULT_OUTPUT_DIR_NAME: Final[str] = "nwp_forecast_comparison"
"""Under `data/studies/`, the default place this script writes to."""

GEFS_FIRST_MONTH: Final[str] = "2024-11"
"""The first month the study reads GEFS from (the plan reads GEFS from 2024-11-30 onwards)."""

UPSAMPLING_METHODS: Final[dict[DomainType, str]] = {
    "solar": "clear_sky",
    "wind": "speed_components",
}
"""The ENS upsampling combination per technology, as the ENS horizons page's report records its rule
choosing them."""

ENS_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3)
"""The bands this study reads ENS at (days 5, 7, 10 and 14 are the horizons page's territory)."""

PRODUCT_SLUGS: Final[dict[str, str]] = {
    "UKV": "ukv",
    "ICON-D2": "icon_d2",
    "ICON-EU": "icon_eu",
    "ICON global": "icon_global",
    "IFS 0.25°": "ifs025",
    "GFS": "gfs",
    "ARPEGE Europe": "arpege",
    "AROME France": "arome",
    "KNMI HARMONIE-AROME": "knmi_harmonie",
    "DMI HARMONIE-AROME": "dmi_harmonie",
}
"""Each Previous Runs product's name (a `PRODUCT_DIRS` key) to the slug its output columns use."""

PRODUCT_DAY_OFFSETS: Final[dict[str, tuple[int, ...]]] = {
    "UKV": (1,),
    "ICON-D2": (1,),
    "ICON-EU": (1, 2, 3),
    "ICON global": (1, 2, 3),
    "IFS 0.25°": (1, 2, 3),
    "GFS": (1, 2, 3),
    "ARPEGE Europe": (1, 2, 3),
    "AROME France": (1,),
    "KNMI HARMONIE-AROME": (1,),
    "DMI HARMONIE-AROME": (1,),
}
"""Each product's `previous_dayN` offsets this study reads, from the plan's product table."""

SOLAR_ONLY_PRODUCTS: Final[frozenset[str]] = frozenset({"ARPEGE Europe", "AROME France"})
"""Products the plan scores for solar only: their 100 m wind offsets are missing on most rows."""

SNAPSHOT_RADIATION_PRODUCTS: Final[frozenset[str]] = frozenset({"UKV"})
"""Products whose Previous Runs radiation is an instantaneous snapshot (V3), rebuilt through
`hourly_from_snapshots` rather than used as served."""


def _repo_data_dir() -> Path:
    """Return the shared `data/` directory, resolving a linked worktree to the main checkout.

    See `verify_previous_runs_leads._repo_data_dir` for the reasoning; duplicated here because
    study scripts in different directories cannot import one another's private helpers.

    Returns:
        The directory holding `studies/`, `NGED/` and the rest of the shared downloads.
    """
    root = PROJECT_ROOT
    marker = root / ".git"
    if marker.is_file():
        pointer = marker.read_text().removeprefix("gitdir:").strip()
        git_dir = Path(pointer)
        if git_dir.parent.name == "worktrees":
            root = git_dir.parent.parent.parent
    return Path(os.environ.get("DATA_PATH_INTERNAL") or root / "data")


def _weather_dir() -> Path:
    """Return `data/studies/weather/`, where every downloaded weather product lives."""
    return _repo_data_dir() / "studies" / "weather"


def _solar_columns(*, frame: pl.DataFrame, day: int, snapshot: bool) -> pl.DataFrame:
    """Return one product's day-N radiation and temperature, on the rows both are present.

    Args:
        frame: The product's `previous_runs/combined.parquet`.
        day: The `previous_dayN` offset to read.
        snapshot: Whether the radiation column is an instantaneous snapshot that needs rebuilding
            through `hourly_from_snapshots` (UKV) rather than being used as served.

    Returns:
        `site`, `time`, `ghi`, `temp`.
    """
    ghi_column = f"shortwave_radiation_previous_day{day}"
    temp_column = f"temperature_2m_previous_day{day}"
    if snapshot:
        snapshots = frame.select(
            key=pl.col("site"), time="time", value=pl.col(ghi_column)
        ).drop_nulls()
        ghi = hourly_from_snapshots(
            frame=snapshots, value_columns=["value"], slot_offsets_minutes=(-60, 0)
        ).select(site=pl.col("key"), time="time", ghi=pl.col("value"))
    else:
        ghi = frame.select("site", "time", ghi=pl.col(ghi_column)).drop_nulls()
    temp = frame.select("site", "time", temp=pl.col(temp_column)).drop_nulls()
    return ghi.join(temp, on=["site", "time"], how="inner")


def _wind_columns(*, frame: pl.DataFrame, day: int) -> pl.DataFrame:
    """Return one product's day-N hub-height wind and 10 m wind, on the rows all three are present.

    Args:
        frame: The product's `previous_runs/combined.parquet`.
        day: The `previous_dayN` offset to read.

    Returns:
        `site`, `time`, `speed_100m`, `sin_100m`, `cos_100m`, `speed_10m`.
    """
    speed_100m_column = f"wind_speed_100m_previous_day{day}"
    direction_100m_column = f"wind_direction_100m_previous_day{day}"
    speed_10m_column = f"wind_speed_10m_previous_day{day}"
    return (
        frame.select(
            "site",
            "time",
            speed_100m=pl.col(speed_100m_column),
            direction_100m=pl.col(direction_100m_column),
            speed_10m=pl.col(speed_10m_column),
        )
        .drop_nulls()
        .with_columns(
            sin_100m=pl.col("direction_100m").radians().sin(),
            cos_100m=pl.col("direction_100m").radians().cos(),
        )
        .drop("direction_100m")
    )


def _previous_runs_frame(*, keys: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Join every Previous Runs product's arm columns onto the shared rows.

    Args:
        keys: `site`, `time` for every row the study might score.
        domain: `solar` or `wind`.

    Returns:
        `keys` with `<slug>_day<N>_<field>` for every product and offset that applies to `domain`,
        left-joined (a row missing a product's offset carries a null, not a dropped row: the row
        set is shared across the planned arms only, which `nwp_forecast_comparison.py` enforces).
    """
    frame = keys
    for product, dir_name in PRODUCT_DIRS.items():
        if domain == "wind" and product in SOLAR_ONLY_PRODUCTS:
            continue
        path = _weather_dir() / dir_name / "previous_runs" / "combined.parquet"
        if not path.exists():
            _LOG.warning("%s: %s not found, skipping", product, path)
            continue
        slug = PRODUCT_SLUGS[product]
        combined = pl.read_parquet(path)
        for day in PRODUCT_DAY_OFFSETS[product]:
            if domain == "solar":
                ghi_column = f"shortwave_radiation_previous_day{day}"
                if ghi_column not in combined.columns:
                    continue
                columns = _solar_columns(
                    frame=combined, day=day, snapshot=product in SNAPSHOT_RADIATION_PRODUCTS
                )
                rename = {"ghi": f"{slug}_day{day}_ghi", "temp": f"{slug}_day{day}_temp"}
            else:
                speed_column = f"wind_speed_100m_previous_day{day}"
                if speed_column not in combined.columns:
                    continue
                columns = _wind_columns(frame=combined, day=day)
                rename = {
                    "speed_100m": f"{slug}_day{day}_speed_100m",
                    "sin_100m": f"{slug}_day{day}_sin_100m",
                    "cos_100m": f"{slug}_day{day}_cos_100m",
                    "speed_10m": f"{slug}_day{day}_speed_10m",
                }
            frame = frame.join(columns.rename(rename), on=["site", "time"], how="left")
    return frame


def _ens_frame(*, keys: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Join ECMWF ENS's mean and control-member columns at `ENS_DAYS` onto the shared rows.

    Reuses `ens_forecast_horizons.build_inputs` and `.main_frame` rather than re-deriving the
    upsampling: both are public functions of that script, unchanged in behaviour, and already do
    the per-member upsampling, the clear-sky and component techniques, and the ensemble mean and
    control-member reduction this study also needs.

    Args:
        keys: `site`, `time` for every row the study might score.
        domain: `solar` or `wind`.

    Returns:
        `keys` with `ens_mean_day<N>_<field>` and `ens_control_day<N>_<field>` for every `N` in
        `ENS_DAYS`, left-joined, plus `persistence_day<N>` and `diurnal_persistence_day<N>` (the
        no-weather baselines' own inputs), which `build_inputs`'s `_with_baselines` step already
        computes on the way to the ENS columns.
    """
    inputs = efh.build_inputs(domain=domain)
    main = efh.main_frame(inputs=inputs, method=UPSAMPLING_METHODS[domain], domain=domain)
    wanted = [
        column
        for day in ENS_DAYS
        for way in ("mean", "control")
        for column in efh.ens_columns(arm=efh.ens_arm(way=way, day=day), domain=domain)
    ]
    baseline_columns = [
        efh.baseline_arm(name=name, day=day)
        for day in ENS_DAYS
        for name in ("persistence", "diurnal_persistence")
    ]
    return keys.join(
        main.frame.select("site", "time", *wanted, *baseline_columns),
        on=["site", "time"],
        how="left",
    )


def _gefs_months_available() -> list[str]:
    """Return the calendar months `data/studies/weather/GEFS/_month_cache/` holds.

    Returns:
        Each cached month's `%Y-%m` label, sorted.
    """
    cache_dir = _weather_dir() / "GEFS" / "_month_cache"
    if not cache_dir.exists():
        return []
    return sorted(path.stem for path in cache_dir.glob("*.parquet"))


def _gefs_span_complete(*, last_month: str) -> bool:
    """Whether the GEFS month cache covers every month from `GEFS_FIRST_MONTH` to `last_month`.

    Args:
        last_month: The last month the study needs, `%Y-%m`.

    Returns:
        Whether every month in that span is cached, with no gap.
    """
    available = set(_gefs_months_available())
    first_year, first_month = (int(part) for part in GEFS_FIRST_MONTH.split("-"))
    last_year, last_month_number = (int(part) for part in last_month.split("-"))
    expected = pl.date_range(
        pl.date(first_year, first_month, 1),
        pl.date(last_year, last_month_number, 1),
        interval="1mo",
        eager=True,
    )
    return all(f"{d.year:04d}-{d.month:02d}" in available for d in expected)


def _gefs_frame(*, keys: pl.DataFrame, domain: DomainType, window_dir: Path | None) -> pl.DataFrame:
    """Read NOAA GEFS's mean columns, gated on a complete month cache or a test window extract.

    The full build (nearest-cell selection per site through `studies.grid_sampling`, the ensemble
    mean, and `studies.resample.gefs_step_means` before any band slicing) is not implemented in
    this pass: the month cache does not yet reach 2026-09 (see the module docstring), so there is
    nothing to build against beyond a `GEFS_window_*` test extract, and the plan gates any fit that
    reads GEFS on the study coordinator confirming the download.

    Args:
        keys: `site`, `time` for every row the study might score.
        domain: `solar` or `wind`.
        window_dir: A `GEFS_window_*` test extract, for development only. `None` in production.

    Returns:
        `keys` unchanged: no GEFS columns are added while the gate does not pass.
    """
    del domain  # not read until the full build is implemented
    if window_dir is not None:
        _LOG.warning(
            "GEFS: --gefs-window-dir given (%s), but the member-mean build is not implemented in "
            "this pass; no GEFS columns are written. See the module docstring.",
            window_dir,
        )
        return keys
    months = _gefs_months_available()
    if not months:
        _LOG.info(
            "GEFS: data/studies/weather/GEFS/_month_cache/ is empty, no GEFS columns written."
        )
        return keys
    last_needed = f"{datetime.now(tz=UTC).year:04d}-{datetime.now(tz=UTC).month - 1:02d}"
    if not _gefs_span_complete(last_month=last_needed):
        _LOG.info(
            "GEFS: month cache holds %d months but does not yet cover %s to %s, no GEFS columns "
            "written.",
            len(months),
            GEFS_FIRST_MONTH,
            last_needed,
        )
        return keys
    _LOG.warning(
        "GEFS: month cache covers %s to %s, but the member-mean build is not implemented in this "
        "pass. No GEFS columns written; see the module docstring.",
        GEFS_FIRST_MONTH,
        last_needed,
    )
    return keys


def build_domain(*, domain: DomainType, output_dir: Path, gefs_window_dir: Path | None) -> Path:
    """Build one technology's arm-input parquet and write it under `output_dir`.

    Args:
        domain: `solar` or `wind`.
        output_dir: Where `<domain>_forecast_inputs.parquet` is written.
        gefs_window_dir: A `GEFS_window_*` test extract, or `None`.

    Returns:
        The written file's path.
    """
    base = efh.base_frame(domain=domain)
    reference_columns = (
        ["ghi_era5", "ghi_cams", "temp_c", "solar_elevation_deg", "solar_azimuth_deg"]
        if domain == "solar"
        else ["speed_hub_era5", "speed_10m_era5"]
    )
    keys = base.select(
        "site",
        "time",
        "power_mw",
        "cap_mw",
        "constrained",
        "effective_capacity_mw",
        "hour_of_day",
        "day_of_year",
        "month",
        *[column for column in reference_columns if column in base.columns],
    )
    frame = _previous_runs_frame(keys=keys, domain=domain)
    frame = _ens_frame(keys=frame, domain=domain)
    frame = _gefs_frame(keys=frame, domain=domain, window_dir=gefs_window_dir)
    output_path = output_dir / f"{domain}_forecast_inputs.parquet"
    frame.write_parquet(output_path)
    _LOG.info("%s: wrote %d rows, %d columns to %s", domain, frame.height, frame.width, output_path)
    return output_path


def main() -> int:
    """Build the solar and wind arm-input frames and write them under `--output-dir`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_data_dir() / "studies" / DEFAULT_OUTPUT_DIR_NAME,
        help="Directory the two arm-input parquets are written to.",
    )
    parser.add_argument(
        "--gefs-window-dir",
        type=Path,
        default=None,
        help="A GEFS_window_* test extract, for development only; production reads the month "
        "cache once it is complete.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for domain in ("solar", "wind"):
        build_domain(
            domain=domain, output_dir=args.output_dir, gefs_window_dir=args.gefs_window_dir
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
