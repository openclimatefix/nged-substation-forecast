"""Check the readings the AIFS arms rest on, from the data on disk, before any fit.

One-off throwaway script for the AIFS arms of
<https://github.com/openclimatefix/nged-substation-forecast/issues/923>. It reads and never fits.
It writes Markdown under `<output-dir>/verification/` and exits non-zero if any check fails.

Before `build_forecast_inputs.py --aifs` has run, `aifs_steps.md` holds:

1. **Radiation window.** AIFS Single's radiation is read as a 6-hour mean ending at the lead. For
   each candidate end offset from -6 to +6 h, the mean absolute difference between AIFS Single's
   radiation at each valid time and ERA5's mean over the six hours ending at that time plus the
   offset, pooled over the solar sites at leads 6 to 78 h. The check fails unless the minimum is at
   offset 0, both over every valid time and over the 06 UTC valid times alone. ERA5 is read for all
   24 hours, and every offset is scored on the same rows: the valid times at which every hour any
   offset's window needs is present.
2. **Wind is instantaneous.** For offsets from -3 to +3 h, the correlation of AIFS Single's 100 m
   speed with ERA5's 100 m speed at the valid time plus the offset, and with ERA5's 6-hour
   mean ending at the valid time. The check fails unless offset 0 correlates best and beats the
   6-hour mean.
3. **Units.** AIFS temperature in degrees Celsius, wind in m/s, radiation between 0 and 1,100 W/m2.
4. **Grid orientation.** `_grid_cells.parquet`'s latitude rises with `lat_index` and its longitude
   with `lon_index`. For each non-central cell of the block shared with GEFS's crop, the correlation
   of its 2 m temperature anomaly (the cell's value minus the mean of the 9 shared cells at the same
   run and lead) with GEFS's control-member anomaly at the cell's latitude and longitude must exceed
   the correlation of each mirrored or transposed cell's anomaly with the same GEFS anomaly.

`NaN` after the run filter is checked by `aifs_members_frame`, which raises.

With `--wiring`, after the build has run, `aifs_wiring.md` holds:

1. **Default ENS path unchanged.** `ens_member_arms(six_hourly=False)` for days 1 and 2 reproduces
   the published inputs' ENS mean and control columns exactly.
2. **The 6-hour path is wired.** For wind, ENS's control member on 6-hourly steps equals the
   published control at hours whose UTC hour is a multiple of 6 and differs at hours that are 3
   mod 6; for solar, the 6-hourly ENS mean differs from the published mean.
3. **AIFS time and lead wiring.** At wind hours whose UTC hour is a multiple of 6, the nearest-cell
   arm equals the speed of the raw store's row at the same cell, the init `time.date() - 1` 00 UTC
   and the lead `24 + hour`.

Only pooled statistics and the anonymised `site` label are printed; no coordinate or cell id.

Run it with `uv run python studies/nwp_forecast_comparison/verify_aifs_steps.py --published-dir
PUBLISHED --output-dir DIR`, and again with `--wiring` after the build.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, cast

import polars as pl
from build_forecast_inputs import (
    AIFS_DAYS,
    AIFS_SINGLE_DIR_NAME,
    AIFS_SINGLE_FIRST_INIT,
    GEFS_WINDOW_DIR_NAME,
    UPSAMPLING_METHODS,
    DomainType,
    aifs_members_frame,
    aifs_site_weights,
    ens_member_arms,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beam_diffuse_split"))
import ens_forecast_horizons as efh
from build_dataset import _pv_sites, nearest_era5_cell, read_era5
from sources import WEATHER_DATA_DIR
from studies.guards import refuse_to_overwrite

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

MAX_LEAD_HOURS: Final[int] = 24 * max(AIFS_DAYS) + 30
"""The longest AIFS lead the arms read."""

RADIATION_OFFSETS: Final[range] = range(-6, 7)
WIND_OFFSETS: Final[range] = range(-3, 4)
WINDOW_HOURS: Final[int] = 6
"""AIFS Single's radiation window, in hours."""

ORIENTATION_MONTH: Final[str] = "2026-03"
"""The month of 00 UTC runs whose 2 m temperature anomalies the orientation check correlates."""

ORIENTATION_LEADS: Final[tuple[int, ...]] = (24, 48)
"""The leads, in hours, that the orientation check pairs with GEFS (both products carry them)."""

CROP_SIZE: Final[int] = 3
"""The side of the block of cells AIFS shares with the GEFS crop."""

KM_PER_HOUR_PER_M_PER_S: Final[float] = 3.6
"""ERA5's `wind_speed_100m` is in km/h; dividing by this gives m/s, AIFS's unit."""

MIN_WIRING_ROWS: Final[int] = 1000
"""The fewest rows a wiring comparison may rest on."""

DIFFERENT_SHARE: Final[float] = 0.9
"""The share of rows on which the 6-hourly ENS columns must differ from the published ones where
the two must differ."""

SPEED_TOLERANCE: Final[float] = 1e-5
"""The relative difference between the nearest-cell arm and the raw store's speed that the
Float32 store allows."""


def era5_column(*, domain: DomainType, sites: list[str]) -> pl.DataFrame:
    """Return ERA5's own value at each generator for every hour of the day.

    The published inputs hold only the daylight hours of solar, so the radiation window check
    needs these full series: a 6-hour window at every end offset from -6 to +6 h has to be complete
    for one common set of valid times.

    Args:
        domain: `solar` (hour-ending global irradiance at the site's nearest ERA5 cell, from
            `beam_diffuse_open_meteo.parquet`) or `wind` (100 m speed at the instant, from
            `wind_era5.parquet`).
        sites: The anonymised site labels to keep.

    Returns:
        `site`, `time` and `era5`, with one row per hour.
    """
    if domain == "wind":
        return (
            pl.read_parquet(WEATHER_DATA_DIR / "ERA5" / "wind_era5.parquet")
            .filter(pl.col("site").is_in(sites))
            .select("site", "time", era5=pl.col("wind_speed_100m") / KM_PER_HOUR_PER_M_PER_S)
            .drop_nulls()
        )
    gridded = read_era5(source="open-meteo")
    cells = nearest_era5_cell(sites=_pv_sites().filter(pl.col("site").is_in(sites)), era5=gridded)
    return (
        cells.join(
            gridded,
            left_on=["cell_latitude", "cell_longitude"],
            right_on=["latitude", "longitude"],
        )
        .select("site", "time", era5="ghi_w_m2")
        .drop_nulls()
    )


def aifs_valid_rows(*, weather_dir: Path, published_dir: Path, domain: DomainType) -> pl.DataFrame:
    """Return AIFS Single's H3-read values at every valid time of the bands' leads.

    Args:
        weather_dir: The folder holding the AIFS downloads.
        published_dir: The folder holding the published inputs, whose sites are read.
        domain: `solar` or `wind`.

    Returns:
        `aifs_members_frame`'s result, with `valid_time`.
    """
    sites = sorted(
        pl.read_parquet(published_dir / f"{domain}_forecast_inputs.parquet", columns=["site"])[
            "site"
        ]
        .unique()
        .to_list()
    )
    path = weather_dir / AIFS_SINGLE_DIR_NAME
    extract = aifs_members_frame(
        store=path / f"{AIFS_SINGLE_DIR_NAME}.parquet",
        weights=aifs_site_weights(path=path, domain=domain, sites=sites, spatial="h3"),
        ensemble=False,
        first_init=AIFS_SINGLE_FIRST_INIT,
        max_lead_hours=MAX_LEAD_HOURS,
    )
    return extract.with_columns(
        valid_time=pl.col("init_time") + pl.duration(hours=pl.col("lead_hours"))
    )


def radiation_window_table(*, aifs: pl.DataFrame, era5: pl.DataFrame) -> pl.DataFrame:
    """Return the mean absolute radiation difference for each candidate window end offset.

    Every offset is scored on one common set of rows: the valid times at which ERA5 has every hour
    that any offset's window needs. Offsets therefore differ only in where the window ends.

    Args:
        aifs: `aifs_valid_rows`'s result for solar.
        era5: `era5_column`'s result for solar, holding all 24 hours.

    Returns:
        `offset`, `valid_hours` (`all` or `06`), `mae` and `n`, with the same `n` at every offset.
    """
    shifts = range(min(RADIATION_OFFSETS) - WINDOW_HOURS + 1, max(RADIATION_OFFSETS) + 1)
    window = aifs.filter(pl.col("lead_hours") > 0).select(
        "site", "valid_time", ghi=pl.col("ghi_w_m2").cast(pl.Float64)
    )
    for shift in shifts:
        window = window.join(
            era5.select(
                "site", end=pl.col("time") - pl.duration(hours=shift), **{f"era5_{shift}": "era5"}
            ),
            left_on=["site", "valid_time"],
            right_on=["site", "end"],
            how="left",
        )
    common = window.drop_nulls([f"era5_{shift}" for shift in shifts])
    records = []
    for offset in RADIATION_OFFSETS:
        window_hours = [f"era5_{offset - k}" for k in range(WINDOW_HOURS)]
        scored = common.with_columns(error=(pl.col("ghi") - pl.mean_horizontal(window_hours)).abs())
        for label, subset in (
            ("all", scored),
            ("06", scored.filter(pl.col("valid_time").dt.hour() == WINDOW_HOURS)),
        ):
            records.append(
                {
                    "offset": offset,
                    "valid_hours": label,
                    "mae": cast("float | None", subset["error"].mean()),
                    "n": subset.height,
                }
            )
    return pl.DataFrame(records)


def radiation_verdict(*, table: pl.DataFrame) -> list[str]:
    """Return the failures of the radiation window check, empty when it passes.

    Offset 0 must have a strictly smaller error than every other offset, so a tie (as at night)
    is a failure rather than evidence. A subset with no rows is a failure too.
    """
    failures = []
    for label in ("all", "06"):
        subset = table.filter(pl.col("valid_hours") == label)
        if subset["n"].min() == 0:
            failures.append(f"radiation check has no rows ({label})")
            continue
        at_zero = subset.filter(pl.col("offset") == 0)["mae"].item()
        if at_zero >= subset.filter(pl.col("offset") != 0)["mae"].min():
            failures.append(f"radiation error is not smallest at offset 0 ({label})")
    return failures


def wind_table(*, aifs: pl.DataFrame, era5: pl.DataFrame) -> pl.DataFrame:
    """Return AIFS 100 m speed's correlation with ERA5's at each offset and with its 6-hour mean.

    Each offset is an inner join, so `n` differs by a few rows near the end of ERA5's series. That
    difference does not change a correlation materially.

    Args:
        aifs: `aifs_valid_rows`'s result for wind.
        era5: `era5_column`'s result for wind in m/s, holding all 24 hours.

    Returns:
        `reading` (an offset, or `6-hour mean`), `correlation` and `n`.
    """
    base = aifs.select("site", "valid_time", speed=pl.col("speed_100m"))
    records = []
    for offset in WIND_OFFSETS:
        joined = base.join(
            era5.select("site", end=pl.col("time") - pl.duration(hours=offset), era5="era5"),
            left_on=["site", "valid_time"],
            right_on=["site", "end"],
        )
        records.append(
            {
                "reading": f"instant at {offset:+d} h",
                "correlation": float(joined.select(pl.corr("speed", "era5")).item()),
                "n": joined.height,
            }
        )
    window = base
    hours = []
    for k in range(WINDOW_HOURS):
        name = f"era5_{k}"
        hours.append(name)
        window = window.join(
            era5.select("site", end=pl.col("time") + pl.duration(hours=k), **{name: "era5"}),
            left_on=["site", "valid_time"],
            right_on=["site", "end"],
            how="left",
        )
    mean = window.drop_nulls(hours).with_columns(era5=pl.mean_horizontal(hours))
    records.append(
        {
            "reading": "6-hour mean ending at the valid time",
            "correlation": float(mean.select(pl.corr("speed", "era5")).item()),
            "n": mean.height,
        }
    )
    return pl.DataFrame(records)


def wind_verdict(*, table: pl.DataFrame) -> list[str]:
    """Return the failures of the wind check, empty when it passes."""
    best = table.sort("correlation", descending=True).row(0, named=True)["reading"]
    return [] if best == "instant at +0 h" else [f"wind correlates best with {best!r}"]


def units_lines(*, solar: pl.DataFrame, wind: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Return the units table and the failures of the units check.

    Args:
        solar: `aifs_valid_rows`'s result for solar.
        wind: `aifs_valid_rows`'s result for wind.

    Returns:
        Markdown lines, and the failures (empty when every quantity is in its expected range).
    """
    ghi = solar.filter(pl.col("lead_hours") > 0)["ghi_w_m2"]
    stats = {
        "temperature mean (deg C)": (cast("float", solar["temp_c"].mean()), -30.0, 50.0),
        "10 m speed mean (m/s)": (cast("float", wind["speed_10m"].mean()), 0.0, 15.0),
        "100 m speed mean (m/s)": (cast("float", wind["speed_100m"].mean()), 0.0, 40.0),
        "radiation minimum (W/m2)": (cast("float", ghi.min()), 0.0, 1100.0),
        "radiation maximum (W/m2)": (cast("float", ghi.max()), 0.0, 1100.0),
    }
    failures = [
        f"{name} = {value:.2f}"
        for name, (value, low, high) in stats.items()
        if not low <= value <= high
    ]
    if cast("float", wind["speed_100m"].mean()) <= cast("float", wind["speed_10m"].mean()):
        failures.append("100 m speed is not above 10 m speed")
    lines = [
        "| Quantity | Value | Expected range |",
        "|---|---|---|",
        *(
            f"| {name} | {value:.2f} | {low:g} to {high:g} |"
            for name, (value, low, high) in stats.items()
        ),
    ]
    return lines, failures


def orientation_table(*, weather_dir: Path) -> tuple[pl.DataFrame, list[str]]:
    """Check the crop's index order, then each cell's anomaly against GEFS's at its coordinates.

    Args:
        weather_dir: The folder holding the AIFS Single and GEFS downloads.

    Returns:
        A table of, for each non-central cell, the correlation of its anomaly and of each mirrored
        or transposed cell's anomaly with GEFS's anomaly at the cell's coordinates, and the
        failures.
    """
    aifs_dir = weather_dir / AIFS_SINGLE_DIR_NAME
    grid = pl.read_parquet(aifs_dir / "_grid_cells.parquet")
    failures = []
    for index, coordinate in (("lat_index", "latitude"), ("lon_index", "longitude")):
        ordered = grid.group_by(index).agg(pl.col(coordinate).mean()).sort(index)[coordinate]
        if not ordered.diff().drop_nulls().gt(0).all():
            failures.append(f"{coordinate} does not rise strictly with {index}")
    key = [pl.col("latitude").round(4).alias("lat"), pl.col("longitude").round(4).alias("lon")]
    first = datetime.strptime(ORIENTATION_MONTH, "%Y-%m")  # noqa: DTZ007
    last = first + timedelta(days=31)
    hours = [timedelta(hours=lead) for lead in ORIENTATION_LEADS]

    def anomalies(*, store: Path, cells: pl.DataFrame, control: bool) -> pl.DataFrame:
        scan = pl.scan_parquet(store).filter(
            pl.col("init_time").dt.hour() == 0,
            pl.col("init_time").is_between(first, last, closed="left"),
            pl.col("lead_time").is_in(hours),
        )
        if control:
            scan = scan.filter(pl.col("ensemble_member") == 0)
        joined = scan.join(cells.lazy(), on=["lat_index", "lon_index"]).select(
            "init_time", "lead_time", "lat_index", "lon_index", "lat", "lon", "temperature_2m"
        )
        return joined.with_columns(
            anomaly=pl.col("temperature_2m")
            - pl.col("temperature_2m").mean().over("init_time", "lead_time")
        ).collect()

    gefs_dir = weather_dir / GEFS_WINDOW_DIR_NAME
    gefs_cells = pl.read_parquet(gefs_dir / "_grid_cells.parquet").select(
        "lat_index", "lon_index", *key
    )
    aifs_cells = grid.select("lat_index", "lon_index", *key).join(
        gefs_cells.select("lat", "lon"), on=["lat", "lon"], how="semi"
    )
    shared = set(aifs_cells.select("lat_index", "lon_index").iter_rows())
    if shared != {(i, j) for i in range(CROP_SIZE) for j in range(CROP_SIZE)}:
        failures.append("the AIFS cells shared with GEFS are not the 3 by 3 block indexed 0 to 2")
        return pl.DataFrame(), failures
    aifs = anomalies(
        store=aifs_dir / f"{AIFS_SINGLE_DIR_NAME}.parquet", cells=aifs_cells, control=False
    )
    gefs = anomalies(
        store=gefs_dir / "_month_cache" / f"{ORIENTATION_MONTH}.parquet",
        cells=gefs_cells,
        control=True,
    )
    if gefs["lat_index"].n_unique() * gefs["lon_index"].n_unique() != CROP_SIZE**2:
        failures.append("the GEFS crop does not have 9 cells")
    records = []
    centre = CROP_SIZE // 2
    for row in aifs_cells.iter_rows(named=True):
        i, j = row["lat_index"], row["lon_index"]
        if (i, j) == (centre, centre):
            continue
        reference = gefs.filter((pl.col("lat") == row["lat"]) & (pl.col("lon") == row["lon"]))
        if reference.is_empty():
            failures.append("a crop cell's coordinates are not in the GEFS crop")
            continue
        alternatives = {
            (CROP_SIZE - 1 - i, j),
            (i, CROP_SIZE - 1 - j),
            (CROP_SIZE - 1 - i, CROP_SIZE - 1 - j),
            (j, i),
        } - {(i, j)}

        def correlation(cell: tuple[int, int], reference: pl.DataFrame = reference) -> float:
            own = aifs.filter((pl.col("lat_index") == cell[0]) & (pl.col("lon_index") == cell[1]))
            paired = own.join(reference, on=["init_time", "lead_time"], suffix="_gefs")
            return float(paired.select(pl.corr("anomaly", "anomaly_gefs")).item())

        correct = correlation((i, j))
        worst = max(correlation(cell) for cell in alternatives)
        records.append({"cell": f"({i}, {j})", "correct": correct, "best_alternative": worst})
        if correct <= worst:
            failures.append(f"cell ({i}, {j}) correlates better with a mirrored cell")
    return pl.DataFrame(records), failures


def steps_report(*, published_dir: Path, weather_dir: Path) -> tuple[list[str], list[str]]:
    """Run the four checks that precede the build.

    Args:
        published_dir: The folder holding the published inputs.
        weather_dir: The folder holding the AIFS and GEFS downloads.

    Returns:
        The report's Markdown lines and every failure.
    """
    solar = aifs_valid_rows(weather_dir=weather_dir, published_dir=published_dir, domain="solar")
    wind = aifs_valid_rows(weather_dir=weather_dir, published_dir=published_dir, domain="wind")
    radiation = radiation_window_table(
        aifs=solar, era5=era5_column(domain="solar", sites=solar["site"].unique().to_list())
    )
    wind_corr = wind_table(
        aifs=wind, era5=era5_column(domain="wind", sites=wind["site"].unique().to_list())
    )
    units, unit_failures = units_lines(solar=solar, wind=wind)
    orientation, orientation_failures = orientation_table(weather_dir=weather_dir)
    failures = [
        *radiation_verdict(table=radiation),
        *wind_verdict(table=wind_corr),
        *unit_failures,
        *orientation_failures,
    ]
    lines = [
        "# AIFS Single: radiation window, wind reading, units and grid orientation",
        "",
        "## Radiation: mean absolute difference from ERA5's 6-hour mean, by window end offset",
        "",
        "| Offset (h) | Valid times | Mean absolute difference (W/m2) | Rows |",
        "|---|---|---|---|",
        *(
            f"| {r['offset']:+d} | {r['valid_hours']} | {r['mae']:.2f} | {r['n']} |"
            for r in radiation.iter_rows(named=True)
        ),
        "",
        "## Wind: correlation of AIFS Single's 100 m speed with ERA5's 100 m speed",
        "",
        "| ERA5 reading | Correlation | Rows |",
        "|---|---|---|",
        *(
            f"| {r['reading']} | {r['correlation']:.4f} | {r['n']} |"
            for r in wind_corr.iter_rows(named=True)
        ),
        "",
        "## Units",
        "",
        *units,
        "",
        "## Grid orientation: correlation of 2 m temperature anomalies with GEFS's at the cell",
        "",
        "| Crop cell (lat_index, lon_index) | Correct cell | Best mirrored or transposed cell |",
        "|---|---|---|",
        *(
            f"| {r['cell']} | {r['correct']:.4f} | {r['best_alternative']:.4f} |"
            for r in orientation.iter_rows(named=True)
        ),
        "",
        "**Verdict:** "
        + ("every check passes." if not failures else "FAILED: " + "; ".join(failures)),
        "",
    ]
    return lines, failures


def default_path_check(*, published_dir: Path, domain: DomainType) -> tuple[str, list[str]]:
    """Check that `ens_member_arms` on its default path reproduces the published ENS columns.

    Args:
        published_dir: The folder holding the published inputs.
        domain: `solar` or `wind`.

    Returns:
        A report line and the failures.
    """
    published = pl.read_parquet(published_dir / f"{domain}_forecast_inputs.parquet")
    sites = sorted(published["site"].unique().to_list())
    arms = ens_member_arms(
        extract=efh.members(sites=sites),
        domain=domain,
        days=AIFS_DAYS,
        method=UPSAMPLING_METHODS[domain],
        ensemble_size=efh.ENSEMBLE_SIZE,
        arm_name=lambda way, day: efh.ens_arm(way=way, day=day),
    )
    largest = 0.0
    failures = []
    for arm_frame in arms:
        joined = published.select("site", "time", *arm_frame.columns[2:]).join(
            arm_frame, on=["site", "time"], suffix="_rebuilt"
        )
        for column in arm_frame.columns[2:]:
            both = joined.filter(
                pl.col(column).is_not_null() & pl.col(f"{column}_rebuilt").is_not_null()
            )
            largest = max(
                largest, cast("float", (both[column] - both[f"{column}_rebuilt"]).abs().max())
            )
    if largest != 0.0:
        failures.append(f"{domain}: the default ENS path differs from the published columns")
    return f"| {domain} | default ENS path against published columns | {largest:g} |", failures


def six_hourly_check(
    *, published_dir: Path, aifs_dir: Path, domain: DomainType
) -> tuple[list[str], list[str]]:
    """Check that the 6-hourly ENS columns differ from the published ones where they must.

    Args:
        published_dir: The folder holding the published inputs.
        aifs_dir: The folder holding the AIFS inputs.
        domain: `solar` or `wind`.

    Returns:
        Report lines and the failures.
    """
    column = "speed_100m" if domain == "wind" else "ghi"
    arm = "ens_control" if domain == "wind" else "ens_mean"
    joined = (
        pl.read_parquet(
            published_dir / f"{domain}_forecast_inputs.parquet",
            columns=["site", "time", f"{arm}_day1_{column}"],
        )
        .join(
            pl.read_parquet(
                aifs_dir / f"{domain}_aifs_inputs.parquet",
                columns=["site", "time", f"{arm}6_day1_{column}"],
            ),
            on=["site", "time"],
        )
        .drop_nulls()
    )
    equal = pl.col(f"{arm}_day1_{column}") == pl.col(f"{arm}6_day1_{column}")
    lines, failures = [], []
    if domain == "wind":
        on_grid = joined.filter(pl.col("time").dt.hour() % 6 == 0)
        off_grid = joined.filter(pl.col("time").dt.hour() % 6 == 3)
        same = float(on_grid.select(equal.mean()).item())
        different = float(off_grid.select((~equal).mean()).item())
        lines.append(f"| wind | 6-hourly control equals published at hours 0 mod 6 | {same:.4f} |")
        lines.append(f"| wind | 6-hourly control differs at hours 3 mod 6 | {different:.4f} |")
        if (
            same != 1.0
            or different < DIFFERENT_SHARE
            or min(on_grid.height, off_grid.height) < MIN_WIRING_ROWS
        ):
            failures.append("wind: the 6-hourly ENS control is not wired as 6-hourly steps")
    else:
        different = float(joined.select((~equal).mean()).item())
        lines.append(f"| solar | 6-hourly mean differs from published mean | {different:.4f} |")
        if different < DIFFERENT_SHARE or joined.height < MIN_WIRING_ROWS:
            failures.append("solar: the 6-hourly ENS mean is not wired as 6-hourly steps")
    return lines, failures


def lead_wiring_check(
    *, aifs_dir: Path, weather_dir: Path, published_dir: Path
) -> tuple[str, list[str]]:
    """Check the nearest-cell arm against the raw store at the run and lead it should read.

    Args:
        aifs_dir: The folder holding the AIFS inputs.
        weather_dir: The folder holding the AIFS downloads.
        published_dir: The folder holding the published inputs, whose sites are read.

    Returns:
        A report line and the failures.
    """
    column = "aifs_single_nearest_day1_speed_100m"
    built = pl.read_parquet(aifs_dir / "wind_aifs_inputs.parquet", columns=["site", "time", column])
    built = built.drop_nulls().filter(pl.col("time").dt.hour() % 6 == 0)
    sites = sorted(built["site"].unique().to_list())
    path = weather_dir / AIFS_SINGLE_DIR_NAME
    nearest = aifs_site_weights(path=path, domain="wind", sites=sites, spatial="nearest")
    expected = (
        pl.scan_parquet(path / f"{AIFS_SINGLE_DIR_NAME}.parquet")
        .filter(pl.col("init_time").dt.hour() == 0)
        .join(nearest.lazy(), on=["lat_index", "lon_index"])
        .select(
            "site",
            init_date=pl.col("init_time").dt.date(),
            lead_hours=pl.col("lead_time").dt.total_hours(),
            expected=(
                pl.col("wind_u_100m").cast(pl.Float64) ** 2
                + pl.col("wind_v_100m").cast(pl.Float64) ** 2
            ).sqrt(),
        )
        .collect()
    )
    compared = built.with_columns(
        init_date=pl.col("time").dt.date() - pl.duration(days=1),
        lead_hours=(24 + pl.col("time").dt.hour()).cast(pl.Int64),
    ).join(expected, on=["site", "init_date", "lead_hours"])
    worst = cast(
        "float", ((compared[column] - compared["expected"]).abs() / compared["expected"]).max()
    )
    failures = []
    if compared.height < MIN_WIRING_ROWS or worst > SPEED_TOLERANCE:
        failures.append("wind: the nearest-cell arm is not the raw store's row at the day-1 run")
    return (
        f"| wind | nearest-cell arm against the raw store, {compared.height} rows | {worst:.2e} |",
        failures,
    )


def wiring_report(
    *, published_dir: Path, aifs_dir: Path, weather_dir: Path
) -> tuple[list[str], list[str]]:
    """Run the three checks that follow the build.

    Args:
        published_dir: The folder holding the published inputs.
        aifs_dir: The folder holding the AIFS inputs.
        weather_dir: The folder holding the AIFS downloads.

    Returns:
        The report's Markdown lines and every failure.
    """
    rows = []
    failures = []
    for domain in ("solar", "wind"):
        line, problems = default_path_check(published_dir=published_dir, domain=domain)
        rows.append(line)
        failures += problems
        lines, problems = six_hourly_check(
            published_dir=published_dir, aifs_dir=aifs_dir, domain=domain
        )
        rows += lines
        failures += problems
    line, problems = lead_wiring_check(
        aifs_dir=aifs_dir, weather_dir=weather_dir, published_dir=published_dir
    )
    rows.append(line)
    failures += problems
    return [
        "# AIFS inputs: wiring checks",
        "",
        "| Technology | Check | Largest difference, or share of rows |",
        "|---|---|---|",
        *rows,
        "",
        "**Verdict:** "
        + ("every check passes." if not failures else "FAILED: " + "; ".join(failures)),
        "",
    ], failures


def main() -> int:
    """Run the checks, write the report, and return 1 if any check failed."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weather-dir", type=Path, default=WEATHER_DATA_DIR)
    parser.add_argument(
        "--wiring", action="store_true", help="Run the checks that follow the build."
    )
    args = parser.parse_args()
    if args.output_dir.resolve() == args.published_dir.resolve():
        msg = "the verification output must not be the published folder"
        raise ValueError(msg)
    verification = args.output_dir / "verification"
    name = "aifs_wiring.md" if args.wiring else "aifs_steps.md"
    refuse_to_overwrite(paths=[verification / name])
    verification.mkdir(parents=True, exist_ok=True)
    if args.wiring:
        lines, failures = wiring_report(
            published_dir=args.published_dir, aifs_dir=args.output_dir, weather_dir=args.weather_dir
        )
    else:
        lines, failures = steps_report(
            published_dir=args.published_dir, weather_dir=args.weather_dir
        )
    (verification / name).write_text("\n".join(lines))
    _LOG.info("wrote %s", verification / name)
    if failures:
        _LOG.error("failed: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
