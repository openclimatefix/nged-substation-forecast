"""Load the inputs that every script of the unmetered-battery-capacity study shares.

The study year is September 2025 to August 2026 on a half-hourly grid labelled by each half-hour's
end time in UTC, as NGED's telemetry and Elexon's B1610 label it. The price files label a period by
its start, so a price is looked up at the end time minus 30 minutes. A *block* is one of the four
3-month spans (September to November, December to February, March to May, June to August).

Sign convention: an aggregate `y` is positive for import. A battery exports when it discharges, so
it enters the model as `- a * template`.

Nothing that identifies a generator is printed, logged, or written: NGED's primaries are labelled
S1 to S8, and the pieces of NGED's network that are not
primaries are labelled BSP1, BSP2, GSP1, and GSP2. NGED battery A is found by a search of series
names, and only aggregates over it are reported.

This module also runs the data-validation checks on the inputs and writes `report_inputs.md`.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_inputs.py`.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from studies.power import scan_power
from studies.pv_physics import SunAndSky, half_hour_sun_and_sky
from studies.pv_separation import solar_basis
from studies.sources import (
    AGILE_EAST_MIDLANDS_DIR,
    MARKET_DOWNLOADS_DIR,
    PRIVATE_DIR,
    REANALYSIS_DOWNLOADS_DIR,
    REPO_DATA_DIR,
    SOLAR_BMU_CENSUS_INPUTS_DIR,
    SOLAR_BMU_DISAGGREGATION_DIR,
    UNMETERED_BATTERY_CAPACITY_DIR,
)

OUTPUT_DIR: Final = UNMETERED_BATTERY_CAPACITY_DIR
WINDOW_START: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
WINDOW_END: Final[datetime] = datetime(2026, 9, 1, tzinfo=UTC)
HALF_HOURS_PER_DAY: Final[int] = 48
MINUTES_PER_HALF_HOUR: Final[int] = 30
BLOCK_MONTHS: Final[tuple[tuple[int, ...], ...]] = ((9, 10, 11), (12, 1, 2), (3, 4, 5), (6, 7, 8))
"""The calendar months of each block."""
BLOCK_NAMES: Final[tuple[str, ...]] = ("Sep-Nov", "Dec-Feb", "Mar-May", "Jun-Aug")
DEMAND_LABELS: Final[tuple[str, ...]] = (
    "S1", "S2", "S3", "S5", "S6", "S7", "S8", "BSP2", "GSP1"
)  # fmt: skip
"""The nine demand-like series of rungs 1 to 3 and the nulls: seven primaries and one bulk and one
grid supply point. BSP1 is the bulk supply point that holds NGED battery A, and S4 and GSP2 are net
exporters with large gaps (`report_inputs.md`)."""
B1610_DIR: Final = SOLAR_BMU_CENSUS_INPUTS_DIR / "b1610"
B1610_SUFFIX: Final[str] = "_20250901_20260901.parquet"
CAMS_PATH: Final = REANALYSIS_DOWNLOADS_DIR / "CAMS_public_points" / "cams_public_points.parquet"
MIN_CAMS_RELIABILITY: Final[float] = 0.9
REFERENCE_LATITUDE: Final[float] = 53.0
REFERENCE_LONGITUDE: Final[float] = -1.5
"""The point whose sun position stands for Great Britain, with the mean of the CAMS `gb_` points."""
BATTERY_A_NAME_PATTERN: Final[str] = "(?i)battery|bess|storage"
REGISTER_PATH: Final = PRIVATE_DIR / "embedded_capacity_register_storage.parquet"
STUCK_RUN_REPORT_LENGTH: Final[int] = 6


def window_half_hours() -> pl.Series:
    """Return the end time of every half-hour of the study year (UTC).

    Returns:
        17,520 end times, from 00:30 on 1 September 2025 to 00:00 on 1 September 2026.
    """
    return pl.datetime_range(
        WINDOW_START + timedelta(minutes=MINUTES_PER_HALF_HOUR),
        WINDOW_END,
        interval="30m",
        time_unit="us",
        time_zone="UTC",
        eager=True,
    ).alias("half_hour_end_time")


def block_slices() -> list[slice]:
    """Return each block's slice of the window grid.

    A half-hour belongs to the block of the UTC month in which it starts.

    Returns:
        Four slices, in `BLOCK_NAMES` order.
    """
    start_month = window_half_hours().dt.offset_by("-30m").dt.month().to_numpy()
    slices = []
    for months in BLOCK_MONTHS:
        indices = np.flatnonzero(np.isin(start_month, months))
        slices.append(slice(int(indices[0]), int(indices[-1]) + 1))
    return slices


def _metadata() -> pl.DataFrame:
    return pl.read_parquet(REPO_DATA_DIR / "NGED" / "metadata.parquet")


def _on_grid(*, readings: pl.DataFrame, series_ids: list[int]) -> dict[int, np.ndarray]:
    """Place each series' readings onto the window grid, NaN where there is no reading.

    Args:
        readings: Columns `time_series_id`, `time` (the half-hour's end, UTC), and `power`.
        series_ids: The series to place.

    Returns:
        The values by series id.
    """
    grid = pl.DataFrame({"time": window_half_hours()})
    out = {}
    for series_id in series_ids:
        one = (
            readings.filter(pl.col("time_series_id") == series_id)
            .select(pl.col("time").dt.cast_time_unit("us"), "power")
            .unique(subset="time", keep="first")
        )
        out[series_id] = (
            grid.join(one, on="time", how="left")["power"].cast(pl.Float64).to_numpy().copy()
        )
    return out


def nged_series() -> dict[str, np.ndarray]:
    """Return NGED's primaries, bulk and grid supply points, and the battery, on the window grid.

    The primaries are the 8 series in MW with time-series type `Disaggregated Demand` (a primary
    in MVA has no sign, so it is out of scope). BSP and GSP series are the raw-flow series in MW.

    Returns:
        Arrays in MW by label: `S1` to `S8`, `BSP1`, `BSP2`, `GSP1`, `GSP2` (import-positive), and
        `battery_A` (export-positive, as metered).

    Raises:
        ValueError: If the search for NGED battery A does not match exactly one series, or if the
            primaries are not 8 in number.
    """
    metadata = _metadata()
    primaries = metadata.filter(
        (pl.col("units") == "MW")
        & (pl.col("substation_type") == "Primary")
        & (pl.col("time_series_type") == "Disaggregated Demand")
    ).sort("time_series_id")
    if primaries.height != 8:
        raise ValueError(f"Expected 8 primaries metered in MW, found {primaries.height}")
    flows = metadata.filter(
        (pl.col("units") == "MW")
        & (pl.col("time_series_type") == "Raw Flow")
        & pl.col("substation_type").is_in(["BSP", "GSP"])
    ).sort("substation_type", "time_series_id")
    battery = metadata.filter(pl.col("time_series_name").str.contains(BATTERY_A_NAME_PATTERN))
    if battery.height != 1:
        raise ValueError(f"The search for battery A matched {battery.height} series, not 1")
    labels: dict[int, str] = {}
    for position, series_id in enumerate(primaries["time_series_id"].to_list(), start=1):
        labels[series_id] = f"S{position}"
    for kind in ("BSP", "GSP"):
        ids = flows.filter(pl.col("substation_type") == kind)["time_series_id"].to_list()
        for position, series_id in enumerate(ids, start=1):
            labels[series_id] = f"{kind}{position}"
    labels[battery["time_series_id"][0]] = "battery_A"
    ids = list(labels)
    readings = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(ids))
        .filter((pl.col("time") > WINDOW_START) & (pl.col("time") <= WINDOW_END))
        .select("time_series_id", "time", pl.col("power"))
        .collect()
    )
    placed = _on_grid(readings=readings, series_ids=ids)
    return {labels[series_id]: placed[series_id] for series_id in ids}


def demand_series() -> dict[str, np.ndarray]:
    """Return the nine demand-like series of rungs 1 to 3 and the nulls (`DEMAND_LABELS`).

    Returns:
        Arrays in MW, import-positive, on the window grid.
    """
    nged = nged_series()
    return {label: nged[label] for label in DEMAND_LABELS}


def agile_prices() -> pl.DataFrame:
    """Return the East Midlands Agile prices (half-hour start, UTC).

    Returns:
        Columns `time` and `price_inc_vat_p_per_kwh`.
    """
    return pl.read_parquet(AGILE_EAST_MIDLANDS_DIR / "octopus_agile_east_midlands.parquet").select(
        "time", "price_inc_vat_p_per_kwh"
    )


def day_ahead_on_grid() -> np.ndarray:
    """Return the N2EX day-ahead price at every half-hour of the window, NaN where there is none.

    The hourly price applies to both half-hours of its hour.

    Returns:
        Pounds per megawatt-hour, one value per half-hour of the grid, which starts at the first
        half-hour of a UTC day.
    """
    n2ex = pl.read_parquet(
        MARKET_DOWNLOADS_DIR / "neso_n2ex_day_ahead" / "neso_n2ex_day_ahead.parquet"
    ).select(hour=pl.col("time").dt.cast_time_unit("us"), price=pl.col("price_gbp_per_mwh"))
    grid = pl.DataFrame({"half_hour_end_time": window_half_hours()}).with_columns(
        hour=pl.col("half_hour_end_time").dt.offset_by("-30m").dt.truncate("1h")
    )
    return grid.join(n2ex, on="hour", how="left")["price"].to_numpy().copy()


def regional_sky() -> SunAndSky:
    """Return the sun and sky of Great Britain: the mean CAMS irradiance of the `gb_` points.

    Returns:
        The sun and sky at each half-hour that CAMS covers.
    """
    cams = pl.read_parquet(CAMS_PATH).filter(
        pl.col("point_id").str.starts_with("gb_")
        & (pl.col("reliability") >= MIN_CAMS_RELIABILITY)
        & (pl.col("time") > WINDOW_START)
        & (pl.col("time") <= WINDOW_END)
    )
    hourly = (
        cams.group_by("time")
        .agg(ghi_w_m2=pl.col("ghi_w_m2").mean())
        .sort("time")
        .with_columns(pl.col("time").dt.cast_time_unit("us"))
    )
    return half_hour_sun_and_sky(
        hourly=hourly, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE
    )


def dc_ac_ratio() -> float:
    """Return the median DC:AC ratio of the solar study's free-orientation fits to all the data."""
    fits = pl.read_parquet(SOLAR_BMU_DISAGGREGATION_DIR / "stage1_fits.parquet").filter(
        (pl.col("arm") == "free") & (pl.col("fold") == -1)
    )
    return float(fits["dc_ac_ratio"].median())  # ty: ignore[invalid-argument-type]


def solar_columns() -> np.ndarray:
    """Return the four fleet curves on the window grid, NaN where CAMS has no reliable hour.

    Returns:
        Shape (17,520, 4): one megawatt of AC capacity per orientation.
    """
    sky = regional_sky()
    basis = solar_basis(sky=sky, dc_ac_ratio=dc_ac_ratio())
    grid = window_half_hours().dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    stamps = sky.half_hour_end_time.astype("datetime64[us]")
    position = {stamp: row for row, stamp in enumerate(stamps)}
    out = np.full((len(grid), basis.shape[1]), np.nan)
    for row, stamp in enumerate(grid):
        found = position.get(stamp)
        if found is not None:
            out[row] = basis[found]
    return out


def storage_presence_by_primary() -> dict[str, dict[str, int]]:
    """Return, for each primary, how many register entries at its substation are storage.

    The register (NGED's Embedded Capacity Register, August 2026, entries of 50 kW and above) lists
    no storage capacity in MWh and no duration for any storage entry, so it can show presence and a
    size class only. Entries are matched to a primary by the primary substation's name with the
    voltage and transformer suffixes removed. No name, capacity, or customer is returned.

    Returns:
        By primary label, the counts of `connected` and `accepted` storage entries.
    """
    metadata = _metadata().filter(
        (pl.col("units") == "MW")
        & (pl.col("substation_type") == "Primary")
        & (pl.col("time_series_type") == "Disaggregated Demand")
    )
    register = pl.read_parquet(REGISTER_PATH).with_columns(
        key=pl.col("primary_substation")
        .str.to_uppercase()
        .str.replace(r"\s+33\s*11\s*K?V.*$", "")
        .str.replace(r"\s+T\d$", "")
        .str.strip_chars(),
        is_storage=pl.any_horizontal(
            pl.col("energy_source_1", "energy_source_2", "energy_source_3").str.contains("(?i)stor")
        ),
    )
    out = {}
    for position, row in enumerate(metadata.sort("time_series_id").iter_rows(named=True), start=1):
        key = (
            row["time_series_name"].upper()
            .replace("33 11KV S STN", "")
            .replace(" T1", "").replace(" T2", "")
            .strip()
        )  # fmt: skip
        entries = register.filter((pl.col("key") == key) & pl.col("is_storage"))
        out[f"S{position}"] = {
            "register_entries_at_the_substation": register.filter(pl.col("key") == key).height,
            "connected_storage": entries.filter(pl.col("connection_status") == "Connected").height,
            "accepted_storage": entries.filter(
                pl.col("connection_status") == "Accepted to connect"
            ).height,
        }
    return out


def change_correlations(*, series: dict[str, np.ndarray], reference: str) -> dict[str, float]:
    """Return the correlation of each series' half-hour changes with a reference's.

    Args:
        series: Arrays on the window grid.
        reference: The label of the reference series.

    Returns:
        The correlation by label, over half-hours where both changes exist.
    """
    ref = np.diff(series[reference])
    out = {}
    for label, values in series.items():
        if label == reference:
            continue
        change = np.diff(values)
        both = np.isfinite(ref) & np.isfinite(change)
        out[label] = float(np.corrcoef(ref[both], change[both])[0, 1])
    return out


def lagged_correlation(*, first: np.ndarray, second: np.ndarray, lag: int) -> float:
    """Return the correlation of `first`'s changes with `second`'s changes `lag` half-hours later.

    Args:
        first: One series on the grid.
        second: The other series.
        lag: Half-hours; positive means `second` lags `first`.

    Returns:
        The correlation over half-hours where both changes exist.
    """
    a, b = np.diff(first), np.diff(second)
    a, b = (a[: len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: len(b) + lag])
    both = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[both], b[both])[0, 1])


def robust_sd(values: np.ndarray) -> float:
    """Return `1.4826 * MAD` of the finite values."""
    finite = values[np.isfinite(values)]
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


def longest_run(values: np.ndarray) -> tuple[int, float]:
    """Return the length and value of the longest run of identical finite values."""
    finite = np.isfinite(values)
    best_length, best_value, length = 0, float("nan"), 0
    previous = np.nan
    for value, ok in zip(values, finite, strict=True):
        if ok and length > 0 and value == previous:
            length += 1
        elif ok:
            length = 1
        else:
            length = 0
        previous = value
        if length > best_length:
            best_length, best_value = length, float(value)
    return best_length, best_value


def _finite_mean(values: np.ndarray) -> float:
    """Return the mean of the finite values, NaN if there are none."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def validate_series(*, label: str, values: np.ndarray) -> dict[str, object]:
    """Run the data-validation checks that apply to one half-hourly power series.

    Args:
        label: The series' label.
        values: The series on the window grid, in MW.

    Returns:
        One row of findings.
    """
    finite = values[np.isfinite(values)]
    grid = window_half_hours()
    local = grid.dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
    half_hour = (local.dt.hour() * 2 + local.dt.minute() // 30).to_numpy()
    profile = np.array(
        [_finite_mean(values[half_hour == slot]) for slot in range(HALF_HOURS_PER_DAY)]
    )
    months = grid.dt.offset_by("-30m").dt.month().to_numpy()
    monthly = np.array([_finite_mean(values[months == month]) for month in range(1, 13)])
    peak = int(np.nanargmax(profile))
    run_length, run_value = longest_run(values)
    p99 = float(np.quantile(np.abs(finite), 0.99))
    return {
        "series": label,
        "missing_half_hours": int(np.isnan(values).sum()),
        "missing_by_block": "/".join(str(int(np.isnan(values[b]).sum())) for b in block_slices()),
        "p01_mw": float(np.quantile(finite, 0.01)),
        "median_mw": float(np.median(finite)),
        "p99_abs_mw": p99,
        "beyond_three_p99": int((np.abs(finite) > 3 * p99).sum()),
        "longest_identical_run": f"{run_length} half-hours at {run_value:.3f} MW",
        "peak_half_hour_uk_clock": f"{peak // 2:02d}:{peak % 2 * 30:02d}",
        "profile_max_mw": float(np.nanmax(profile)),
        "profile_min_mw": float(np.nanmin(profile)),
        "monthly_mean_lowest_mw": float(np.nanmin(monthly)),
        "monthly_mean_highest_mw": float(np.nanmax(monthly)),
        "sigma_step_raw_mw": robust_sd(np.diff(values)),
    }


def main() -> None:
    """Validate every input and write `report_inputs.md`."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nged = nged_series()
    metadata = _metadata()
    lines = ["# Inputs report", ""]
    grid = window_half_hours()
    lines += [
        f"Window grid: {len(grid)} half-hours, {grid[0]} to {grid[-1]} (UTC, end-stamped).",
        "Blocks (half-hours): "
        + str({n: b.stop - b.start for n, b in zip(BLOCK_NAMES, block_slices(), strict=True)}),
        "",
        "## NGED series scope",
        "",
    ]
    counts = metadata.group_by("units", "substation_type").len().sort("units", "substation_type")
    lines.append(f"Series by unit and substation type: {counts.rows()}")
    n_mva = metadata.filter(
        (pl.col("units") == "MVA") & (pl.col("substation_type") == "Primary")
    ).height
    lines.append(f"Primaries metered in MVA are out of scope (MVA has no sign): {n_mva} series.")
    labelled = metadata.filter(
        (pl.col("units") == "MW")
        & (pl.col("substation_type") == "Primary")
        & (pl.col("time_series_type") == "Disaggregated Demand")
    )
    named = labelled.filter(
        pl.col("information").is_not_null()
        & pl.col("information").str.contains("(?i)generation disaggregated")
    ).height
    lines += [
        "",
        '## The meaning of "Disaggregated Demand"',
        "",
        (
            "NGED's data portal publishes no definition of the label in the dataset descriptions "
            "that were searched (the substation-loading and live-primary-data descriptions). The "
            f"metadata's information field names generators as disaggregated for {named} of the 8 "
            "primaries, which says that some metered generation was separated from the flow, not "
            "which way. The label is therefore unverified. A primary whose median is negative is "
            "a net exporter at the median, so its series cannot be pure demand."
        ),
        "",
        "## Per-series checks (import-positive MW)",
        "",
    ]
    rows = [validate_series(label=k, values=v) for k, v in nged.items() if k != "battery_A"]
    lines.append(pl.DataFrame(rows).write_csv(separator="|"))
    lines += ["", "## Is a bulk supply point's flow connected to NGED battery A?", ""]
    correlation = change_correlations(series=nged, reference="battery_A")
    lines.append("Correlation of half-hour changes with NGED battery A's:")
    lines += [f"- {label}: {value:.3f}" for label, value in sorted(correlation.items())]
    best = min(
        (label for label in correlation if label.startswith("BSP")), key=correlation.__getitem__
    )
    lines.append(f"The strongest bulk supply point is {best}.")
    lines.append("Lagged correlation (positive lag: the flow lags the battery):")
    for lag in (-2, -1, 0, 1, 2):
        value = lagged_correlation(first=nged["battery_A"], second=nged[best], lag=lag)
        lines.append(f"- lag {lag:+d}: {value:.3f}")
    lines += ["", "## Register (storage entries of 50 kW and above) by primary", ""]
    for label, found in storage_presence_by_primary().items():
        lines.append(f"- {label}: {found}")
    sky_basis = solar_columns()
    lines += [
        "",
        "## Solar columns and prices",
        "",
        f"Half-hours with no reliable CAMS hour: {int(np.isnan(sky_basis[:, 0]).sum())}",
        f"Largest basis value per orientation: {np.nanmax(sky_basis, axis=0).round(3).tolist()}",
        f"Half-hours with no N2EX price: {int(np.isnan(day_ahead_on_grid()).sum())}",
    ]
    agile = agile_prices()
    lines.append(f"Agile rows: {agile.height}; first {agile['time'][0]}, last {agile['time'][-1]}.")
    (OUTPUT_DIR / "report_inputs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
