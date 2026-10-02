"""Verify the UKV-CEDA inputs `build_ukv_ceda_inputs.py` wrote, before any fit reads them.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1016>. It reads the built
`<domain>_ukv_ceda_inputs.parquet`, the published inputs, and the `UKV-CEDA-T120` store, and writes
only `verify.json` into the build's folder. It exits non-zero if a check fails, and
`fit_ukv_ceda_blends.py` runs only after a passing run, which it confirms from `verify.json`.

**Values are recomputed in plain Python.** A stratified sample of built values is recomputed from
the store with `math` and no array code, so a slip in the build's array arithmetic does not repeat
here. The strata hit leads up to 48 hours (native hourly), the step from 48 to 54 hours (the first
rebuilt leads), and leads of 57 hours and above, and wind hours 0 to 2 UTC, where the run is the
previous day's. A rebuilt radiation value is recomputed only where both 3-hourly anchors either side
are in daylight, because the hold-flat rule for night anchors is not worth reimplementing.

**Alignment.** The gate is the median over the solar generators, and has two parts. (a) The raw
native day-1 snapshots, which are instants, must track the sun best within
`RAW_PEAK_TOLERANCE_MINUTES` of 0, which catches any lead or slot error. (b) The rebuilt day-1
column, a mean of two snapshots an hour apart, must peak `MEAN_OF_TWO_OFFSET_MINUTES` plus or minus
`MEAN_PEAK_TOLERANCE_MINUTES` from (a)'s median, which confirms that the build averages the
snapshots at `L - 1` and `L`. The snapshots are not reweighted to land on -30 minutes. UKV-CEDA's
radiation behaves like the sun about 10 minutes after its stamp, a property of the archive that no
construction choice can remove, so the rebuilt column is centred about 20 minutes before the label,
not 30. Days 2 to 4 are printed only, because the clear-sky multiplication sets the diurnal shape
whatever run the anchor came from, so the check cannot test the lead there. A scan of the
correlation between UKV-CEDA's day-1 10 m wind speed and the wind power, with the speed shifted by
-3 to +3 hours, is printed.

**Skill.** The correlation of `ukv_ceda_day<N>_ghi` with CAMS's irradiance, and of `_speed_10m` with
ERA5's 10 m speed, taken at each lead day on the rows all four lead days hold, is printed beside
Open-Meteo UKV day 1's on the same rows. The check fails unless day 1 is within `DAY1_TOLERANCE` of
Open-Meteo UKV's correlation and the correlation never rises from one lead day to the next.

**Steps.** Each month's mean UKV-CEDA irradiance and 10 m wind speed over ENS's, per generator, is
printed where a month differs from the one before by 15% or more. This is a screen, and it does not
fail the run.

**The verify stamp.** The script writes `verify.json` into the build's folder, holding whether
every gating check passed and the SHA-256 of each inputs file it read. `fit_ukv_ceda_blends.py`
refuses to fit unless that stamp passed and its hashes are the inputs' current ones. A later run
replaces the stamp, because it describes the latest verification and not an output of the study.

Run it with `uv run python studies/ukv_ceda_blends/verify_ukv_ceda_inputs.py`.
"""

import argparse
import itertools
import json
import math
import random
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import polars as pl

_STUDIES_DIR: Final[Path] = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STUDIES_DIR / "ukv_ceda_blends"))
sys.path.insert(0, str(_STUDIES_DIR / "nwp_forecast_comparison"))
sys.path.insert(0, str(_STUDIES_DIR / "beam_diffuse_split"))
sys.path.insert(0, str(_STUDIES_DIR / "weather_downloads"))

import build_ukv_ceda_inputs as build  # noqa: E402
import ens_forecast_horizons as efh  # noqa: E402
from nwp_forecast_comparison import DomainType  # noqa: E402
from paths import REPO_DATA_DIR  # noqa: E402
from studies.grid_sampling import nearest_cells  # noqa: E402
from studies.ifs_single_runs import served_lead_hours  # noqa: E402
from studies.solar import zenith  # noqa: E402
from studies.timestamp_checks import best_offset_minutes, correlation_by_offset  # noqa: E402

SAMPLE_PER_STRATUM: Final[int] = 6
"""How many rows are recomputed per technology, lead day, and stratum."""

SAMPLE_SEED: Final[int] = 20261002

RELATIVE_TOLERANCE: Final[float] = 1e-6
ABSOLUTE_TOLERANCE: Final[float] = 1e-6
"""A built value and its recomputation agree within these (the build is float64 throughout)."""

DAY1_TOLERANCE: Final[float] = 0.05
"""How far below Open-Meteo UKV's day-1 correlation UKV-CEDA's day-1 correlation may sit."""

RAW_PEAK_TOLERANCE_MINUTES: Final[int] = 15
"""How far the median raw day-1 snapshot peak may sit from 0 minutes, where an instant peaks."""

MEAN_OF_TWO_OFFSET_MINUTES: Final[int] = -30
"""Where a mean of two snapshots an hour apart peaks, relative to the snapshots' own peak."""

MEAN_PEAK_TOLERANCE_MINUTES: Final[int] = 10
"""How far the rebuilt column's median peak may sit from `MEAN_OF_TWO_OFFSET_MINUTES` after (a)."""

DAY1_SNAPSHOT_LEADS: Final[range] = range(21, 46)
"""The native leads whose snapshots the day-1 rows average, in hours."""

STEP_RATIO_THRESHOLD: Final[float] = 1.15
"""A month-to-month change in a generator's UKV-CEDA to ENS ratio beyond this factor is flagged."""

DAYLIGHT_FLOOR_W_M2: Final[float] = 50.0
"""The least clear-sky irradiance at an anchor for a rebuilt value to be recomputed here."""

SHIFT_HOURS: Final[tuple[int, ...]] = tuple(range(-3, 4))

STRATA: Final[tuple[str, ...]] = ("native", "first rebuilt", "late")
WIND_EARLY_HOURS: Final[str] = "wind hours 0 to 2"


class Mismatch(NamedTuple):
    """One built value that its recomputation does not reproduce."""

    domain: str
    day: int
    site: str
    time: datetime
    column: str
    built: float
    recomputed: float


# --- Plain-Python recomputation -------------------------------------------------------------------


def haurwitz(*, zenith_deg: float) -> float:
    """Return the Haurwitz clear-sky irradiance in W/m2 at a solar zenith angle in degrees."""
    cosine = math.cos(math.radians(zenith_deg))
    if cosine <= 0.0:
        return 0.0
    return 1098.0 * cosine * math.exp(-0.059 / cosine)


def run_of_row(*, time: datetime, day: int, domain: DomainType) -> tuple[datetime, int, int]:
    """Return the 03 UTC run a row reads, its slot in the store, and the lead of the row's label.

    Args:
        time: The hour's label.
        day: The lead day.
        domain: `solar` or `wind`.

    Returns:
        The run's start, its slot on the store's 12-hourly grid, and the lead in hours.
    """
    instant = time - timedelta(hours=1) if domain == "solar" else time
    midnight = datetime(instant.year, instant.month, instant.day, tzinfo=UTC)
    init = midnight - timedelta(days=day) + timedelta(hours=build.RUN_HOUR)
    slot = int((init - build.T120_PROFILE.slot_epoch) / timedelta(hours=build.SLOT_HOURS))
    lead = int((time - init) / timedelta(hours=1))
    return init, slot, lead


def is_native(*, lead: int) -> bool:
    """Return whether the store holds a value at a lead."""
    return lead in set(build.NATIVE_LEADS.tolist())


def bracket(*, lead: int) -> tuple[int, int]:
    """Return the 3-hourly anchors either side of a lead the store lacks."""
    before = build.HOURLY_LAST_LEAD + 3 * ((lead - build.HOURLY_LAST_LEAD) // 3)
    return before, before + 3


def clear_sky_at(*, init: datetime, lead: int, latitude: float, longitude: float) -> float:
    """Return the instantaneous Haurwitz clear-sky irradiance at a run's lead."""
    stamp = pl.Series([init + timedelta(hours=lead)], dtype=pl.Datetime("us", "UTC"))
    angle = float(zenith(stamps=stamp, latitude=latitude, longitude=longitude)[0])
    return haurwitz(zenith_deg=angle)


def radiation_snapshot(
    *, series: Sequence[float], init: datetime, lead: int, latitude: float, longitude: float
) -> float | None:
    """Return the radiation snapshot at a lead, or `None` if it cannot be recomputed here.

    A native lead is the store's value. A rebuilt lead is the straight-line clear-sky index between
    its two anchors times the clear sky at its own instant, and is returned only if both anchors are
    in daylight and finite.

    Args:
        series: The run's `shortwave_down` at every lead, 0 to 120.
        init: The run's start.
        lead: The lead wanted.
        latitude: The site's latitude.
        longitude: The site's longitude.

    Returns:
        The snapshot in W/m2, or `None`.
    """
    if is_native(lead=lead):
        return series[lead] if math.isfinite(series[lead]) else None
    before, after = bracket(lead=lead)
    clear = {
        anchor: clear_sky_at(init=init, lead=anchor, latitude=latitude, longitude=longitude)
        for anchor in (before, after)
    }
    if min(clear.values()) < DAYLIGHT_FLOOR_W_M2:
        return None
    if not (math.isfinite(series[before]) and math.isfinite(series[after])):
        return None
    weight = (lead - before) / (after - before)
    index = (1 - weight) * series[before] / clear[before] + weight * series[after] / clear[after]
    return index * clear_sky_at(init=init, lead=lead, latitude=latitude, longitude=longitude)


def wind_at(
    *, speed: Sequence[float], direction: Sequence[float], lead: int
) -> tuple[float, float] | None:
    """Return a wind's speed and direction (from, degrees) at a lead, rebuilt by its components.

    Args:
        speed: The run's wind speed at every lead.
        direction: The run's wind direction at every lead, in degrees.
        lead: The lead wanted.

    Returns:
        The speed and direction, or `None` if a value needed is missing.
    """
    leads = [lead] if is_native(lead=lead) else [*bracket(lead=lead)]
    if not all(math.isfinite(speed[a]) and math.isfinite(direction[a]) for a in leads):
        return None
    if len(leads) == 1:
        return speed[lead], direction[lead]
    east = [-speed[a] * math.sin(math.radians(direction[a])) for a in leads]
    north = [-speed[a] * math.cos(math.radians(direction[a])) for a in leads]
    weight = (lead - leads[0]) / (leads[1] - leads[0])
    u = (1 - weight) * east[0] + weight * east[1]
    v = (1 - weight) * north[0] + weight * north[1]
    return math.hypot(u, v), math.degrees(math.atan2(-u, -v)) % 360.0


def agree(*, built: float, recomputed: float) -> bool:
    """Return whether a built value and its recomputation agree within the tolerances."""
    return math.isclose(built, recomputed, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)


def stratum_of(*, lead: int) -> str:
    """Return a row's stratum from the lead of its label."""
    if lead <= build.HOURLY_LAST_LEAD:
        return STRATA[0]
    return STRATA[1] if lead <= 54 else STRATA[2]


# --- Reading the store and the rows ---------------------------------------------------------------


class SiteCell(NamedTuple):
    """One site's nearest store cell and its private coordinates, which are never printed."""

    cell: int
    latitude: float
    longitude: float


def site_cells(
    *, store: build.StoreRead, domain: DomainType, sites: Sequence[str]
) -> dict[str, SiteCell]:
    """Return each site's nearest cell and coordinates.

    Args:
        store: The opened store.
        domain: `solar` or `wind`.
        sites: The site labels.

    Returns:
        Each label to its `SiteCell`.
    """
    roster = efh.site_roster(domain=domain).filter(pl.col("site").is_in(list(sites))).sort("site")
    latitude = np.asarray(build.zarr_array(group=store.group, name="cell_latitude")[:])
    longitude = np.asarray(build.zarr_array(group=store.group, name="cell_longitude")[:])
    cells = pl.DataFrame(
        {"cell_id": np.arange(len(latitude)), "latitude": latitude, "longitude": longitude}
    )
    nearest = nearest_cells(sites=roster, cells=cells)
    return {
        site: SiteCell(cell=int(cell), latitude=float(lat), longitude=float(lon))
        for (site, cell), (_, lat, lon) in zip(
            nearest.select("site", "cell_id").iter_rows(),
            roster.iter_rows(),
            strict=True,
        )
    }


def read_run(*, store: build.StoreRead, variable: str, slot: int, cell: int) -> list[float]:
    """Return one variable's value at every lead of one run and one cell."""
    values = build.zarr_array(group=store.group, name=variable)[slot, :, cell]
    return [float(value) for value in np.asarray(values)]


def sample_rows(
    *, built: pl.DataFrame, domain: DomainType, day: int
) -> dict[str, list[dict[str, Any]]]:
    """Shuffle the rows with values for one lead day into strata, ready to be sampled from.

    Args:
        built: The technology's built inputs.
        domain: `solar` or `wind`.
        day: The lead day.

    Returns:
        Each stratum of `STRATA`, and for wind `WIND_EARLY_HOURS`, to its rows as dictionaries of
        `site`, `time`, and the day's columns, in a seeded random order.
    """
    columns = [f"ukv_ceda_day{day}_{field}" for field in build.WEATHER_FIELDS[domain]]
    present = built.filter(pl.col(columns[0]).is_not_null()).select("site", "time", *columns)
    marked = present.with_columns(
        lead=served_lead_hours(
            time=pl.col("time"), day=day, domain=domain, run_hour=build.RUN_HOUR
        ),
        hour=pl.col("time").dt.hour(),
    )
    last_hourly, last_first_rebuilt = build.HOURLY_LAST_LEAD, 54
    filters = {
        STRATA[0]: pl.col("lead") <= last_hourly,
        STRATA[1]: (pl.col("lead") > last_hourly) & (pl.col("lead") <= last_first_rebuilt),
        STRATA[2]: pl.col("lead") > last_first_rebuilt,
    }
    if domain == "wind":
        filters[WIND_EARLY_HOURS] = pl.col("hour") <= 2
    generator = random.Random(SAMPLE_SEED + day)
    strata: dict[str, list[dict[str, Any]]] = {}
    for name, condition in filters.items():
        records = marked.filter(condition).to_dicts()
        generator.shuffle(records)
        strata[name] = records
    return strata


def recompute_solar(
    *, store: build.StoreRead, init: datetime, slot: int, lead: int, cell: SiteCell, prefix: str
) -> dict[str, float] | None:
    """Recompute a solar row's two columns, or return `None` if a snapshot cannot be recomputed."""
    ghi_series = read_run(store=store, variable="shortwave_down", slot=slot, cell=cell.cell)
    leads = (lead - 1, lead)
    snapshots = [
        radiation_snapshot(
            series=ghi_series,
            init=init,
            lead=snapshot,
            latitude=cell.latitude,
            longitude=cell.longitude,
        )
        for snapshot in leads
    ]
    if any(snapshot is None for snapshot in snapshots):
        return None
    kelvin = read_run(store=store, variable="temperature_1p5m", slot=slot, cell=cell.cell)
    temperature = [
        kelvin[snapshot] if is_native(lead=snapshot) else _linear_fill(values=kelvin, lead=snapshot)
        for snapshot in leads
    ]
    return {
        f"{prefix}ghi": sum(value for value in snapshots if value is not None) / len(leads),
        f"{prefix}temp": sum(temperature) / len(leads) - build.KELVIN,
    }


def recompute_wind(
    *, store: build.StoreRead, slot: int, lead: int, cell: SiteCell, prefix: str
) -> dict[str, float] | None:
    """Recompute a wind row's four columns, or return `None` if a value needed is missing."""

    def series(variable: str) -> list[float]:
        return read_run(store=store, variable=variable, slot=slot, cell=cell.cell)

    near = wind_at(
        speed=series("wind_speed_10m"), direction=series("wind_direction_10m"), lead=lead
    )
    far = wind_at(
        speed=series("wind_speed_925hpa"), direction=series("wind_direction_925hpa"), lead=lead
    )
    if near is None or far is None:
        return None
    return {
        f"{prefix}speed_10m": near[0],
        f"{prefix}sin_10m": math.sin(math.radians(near[1])),
        f"{prefix}cos_10m": math.cos(math.radians(near[1])),
        f"{prefix}speed_925hpa": far[0],
    }


def check_row(
    *,
    store: build.StoreRead,
    domain: DomainType,
    day: int,
    row: dict[str, Any],
    cell: SiteCell,
) -> tuple[int, list[Mismatch]]:
    """Recompute one row's columns from the store and compare them with the built values.

    Args:
        store: The opened store.
        domain: `solar` or `wind`.
        day: The lead day.
        row: A sampled row, with `site`, `time` and the day's columns.
        cell: The site's cell.

    Returns:
        How many columns were recomputed (0 if the row was not checkable), and the mismatches.
    """
    init, slot, lead = run_of_row(time=row["time"], day=day, domain=domain)
    prefix = f"ukv_ceda_day{day}_"
    if domain == "solar":
        recomputed = recompute_solar(
            store=store, init=init, slot=slot, lead=lead, cell=cell, prefix=prefix
        )
    else:
        recomputed = recompute_wind(store=store, slot=slot, lead=lead, cell=cell, prefix=prefix)
    if recomputed is None:
        return 0, []
    mismatches = [
        Mismatch(domain, day, row["site"], row["time"], column, row[column], value)
        for column, value in recomputed.items()
        if not agree(built=row[column], recomputed=value)
    ]
    return len(recomputed), mismatches


def _linear_fill(*, values: Sequence[float], lead: int) -> float:
    """Return a value at a lead the store lacks, straight between its two anchors."""
    before, after = bracket(lead=lead)
    weight = (lead - before) / (after - before)
    return (1 - weight) * values[before] + weight * values[after]


def check_values(
    *, store: build.StoreRead, domain: DomainType, built: pl.DataFrame
) -> tuple[int, list[Mismatch]]:
    """Recompute a stratified sample of one technology at every lead day.

    Args:
        store: The opened store.
        domain: `solar` or `wind`.
        built: The technology's built inputs.

    Returns:
        How many values were recomputed, and every mismatch.
    """
    cells = site_cells(store=store, domain=domain, sites=built["site"].unique().to_list())
    checked = 0
    found: list[Mismatch] = []
    for day in build.LEAD_DAYS:
        recomputed_rows: dict[str, int] = {}
        for name, records in sample_rows(built=built, domain=domain, day=day).items():
            done = 0
            for row in records:
                if done >= SAMPLE_PER_STRATUM:
                    break
                count, mismatches = check_row(
                    store=store, domain=domain, day=day, row=row, cell=cells[row["site"]]
                )
                if count:
                    done += 1
                    checked += count
                    found += mismatches
            recomputed_rows[name] = done
        sys.stdout.write(f"{domain} day {day}: rows recomputed per stratum {recomputed_rows}\n")
    return checked, found


# --- Alignment, skill, and steps ------------------------------------------------------------------


def correlation(*, first: pl.Series, second: pl.Series) -> float:
    """Return the Pearson correlation of two series over the rows where both hold a value."""
    frame = pl.DataFrame({"a": first, "b": second}).drop_nulls().drop_nans()
    return float(np.corrcoef(frame["a"].to_numpy(), frame["b"].to_numpy())[0, 1])


def non_increasing(*, values: Sequence[float]) -> bool:
    """Return whether a sequence never rises from one element to the next."""
    return all(later <= earlier for earlier, later in itertools.pairwise(values))


def skill_gate(
    *, ukv_ceda: Sequence[float], open_meteo_day1: float, day1_ceda_same_rows: float
) -> list[str]:
    """Return why a technology's skill check fails, if it does.

    Args:
        ukv_ceda: UKV-CEDA's correlation with the reference at each lead day, day 1 first.
        open_meteo_day1: Open-Meteo UKV's day-1 correlation on the rows both products hold.
        day1_ceda_same_rows: UKV-CEDA's day-1 correlation on those same rows.

    Returns:
        The reasons, empty if the check passes.
    """
    reasons = []
    if day1_ceda_same_rows < open_meteo_day1 - DAY1_TOLERANCE:
        reasons.append(
            f"day-1 correlation {day1_ceda_same_rows:.3f} is more than {DAY1_TOLERANCE} below "
            f"Open-Meteo UKV's {open_meteo_day1:.3f}"
        )
    if not non_increasing(values=ukv_ceda):
        reasons.append(
            f"the correlation rises with the lead day: {[round(v, 3) for v in ukv_ceda]}"
        )
    return reasons


def alignment_gate(*, raw_peaks: Sequence[int], rebuilt_peaks: Sequence[int]) -> list[str]:
    """Return why the day-1 radiation timing fails, if it does, from the median over generators.

    Args:
        raw_peaks: Each generator's peak offset of the raw day-1 snapshots, in minutes.
        rebuilt_peaks: Each generator's peak offset of the rebuilt day-1 column, in minutes.

    Returns:
        The reasons, empty if both parts of the gate pass.
    """
    raw = float(np.median(raw_peaks))
    rebuilt = float(np.median(rebuilt_peaks))
    reasons = []
    if abs(raw) > RAW_PEAK_TOLERANCE_MINUTES:
        reasons.append(
            f"the raw day-1 snapshots peak at {raw:+.0f} minutes, more than "
            f"{RAW_PEAK_TOLERANCE_MINUTES} from 0, so a lead or slot is misread"
        )
    if abs(rebuilt - raw - MEAN_OF_TWO_OFFSET_MINUTES) > MEAN_PEAK_TOLERANCE_MINUTES:
        reasons.append(
            f"the rebuilt day-1 column peaks {rebuilt - raw:+.0f} minutes from the raw snapshots, "
            f"not {MEAN_OF_TWO_OFFSET_MINUTES} plus or minus {MEAN_PEAK_TOLERANCE_MINUTES}, so the "
            "build is not averaging the snapshots at L - 1 and L"
        )
    return reasons


def raw_day1_peak(*, store: build.StoreRead, cell: SiteCell, init_times: Sequence[datetime]) -> int:
    """Return where a site's raw day-1 radiation snapshots track the sun best, in minutes.

    Args:
        store: The opened store.
        cell: The site's cell.
        init_times: The runs the site's day-1 rows read.

    Returns:
        The offset of the correlation's peak.
    """
    times: list[datetime] = []
    values: list[float] = []
    for init in init_times:
        slot = int((init - build.T120_PROFILE.slot_epoch) / timedelta(hours=build.SLOT_HOURS))
        series = read_run(store=store, variable="shortwave_down", slot=slot, cell=cell.cell)
        for lead in DAY1_SNAPSHOT_LEADS:
            if math.isfinite(series[lead]):
                times.append(init + timedelta(hours=lead))
                values.append(series[lead])
    correlations = correlation_by_offset(
        times=pl.Series(times, dtype=pl.Datetime("us", "UTC")),
        ghi=np.array(values),
        latitude=cell.latitude,
        longitude=cell.longitude,
    )
    return best_offset_minutes(correlations=correlations)


def alignment_lines(
    *,
    domain: DomainType,
    joined: pl.DataFrame,
    cells: dict[str, SiteCell],
    store: build.StoreRead,
) -> tuple[list[str], bool]:
    """Run the radiation peak-offset check at every lead day, gating day 1 on the median.

    Args:
        domain: `solar` or `wind`; only solar is checked.
        joined: Built inputs with `site` and `time`.
        cells: Each site's coordinates.
        store: The opened store, read for the raw day-1 snapshots.

    Returns:
        The printed lines, and whether the day-1 gate passed.
    """
    if domain != "solar":
        return [], True
    lines: list[str] = []
    raw_peaks: list[int] = []
    rebuilt_peaks: list[int] = []
    for day in build.LEAD_DAYS:
        for site, cell in sorted(cells.items()):
            frame = joined.filter(
                pl.col("site") == site, pl.col(f"ukv_ceda_day{day}_ghi").is_not_null()
            )
            correlations = correlation_by_offset(
                times=frame["time"],
                ghi=frame[f"ukv_ceda_day{day}_ghi"].to_numpy(),
                latitude=cell.latitude,
                longitude=cell.longitude,
            )
            best = best_offset_minutes(correlations=correlations)
            lines.append(
                f"solar day {day} site {site}: radiation tracks the sun best at {best:+d} minutes"
            )
            if day == 1:
                rebuilt_peaks.append(best)
                raw = raw_day1_peak(
                    store=store,
                    cell=cell,
                    init_times=frame["ukv_ceda_day1_init_time"].unique().to_list(),
                )
                raw_peaks.append(raw)
                lines.append(f"solar day 1 site {site}: raw snapshots peak at {raw:+d} minutes")
    reasons = alignment_gate(raw_peaks=raw_peaks, rebuilt_peaks=rebuilt_peaks)
    lines.append(
        f"solar day 1 median over {len(raw_peaks)} sites: raw snapshots "
        f"{np.median(raw_peaks):+.0f} minutes, rebuilt column {np.median(rebuilt_peaks):+.0f}"
    )
    lines.extend(f"FAIL {reason}" for reason in reasons)
    return lines, not reasons


def wind_offset_lines(*, joined: pl.DataFrame) -> list[str]:
    """Print the correlation of day-1 10 m wind speed with wind power at each shift in hours."""
    lines = ["wind day 1: correlation of UKV-CEDA 10 m speed with power, speed shifted by hours"]
    power = joined.select("site", "time", "power_mw")
    for shift in SHIFT_HOURS:
        moved = joined.select(
            "site",
            time=pl.col("time").dt.offset_by(f"{shift}h"),
            speed=pl.col("ukv_ceda_day1_speed_10m"),
        )
        paired = power.join(moved, on=["site", "time"]).drop_nulls()
        value = correlation(first=paired["speed"], second=paired["power_mw"])
        lines.append(f"  shift {shift:+d} h: {value:.3f}")
    return lines


def step_lines(*, joined: pl.DataFrame, domain: DomainType) -> list[str]:
    """Flag a month whose UKV-CEDA to ENS mean ratio differs from the month before by 15% or more.

    Args:
        joined: Built inputs joined with ENS's day-1 mean, with `site` and `time`.
        domain: `solar` or `wind`.

    Returns:
        One line per flagged generator and month.
    """
    column, ens = (
        ("ukv_ceda_day1_ghi", "ens_mean_day1_ghi")
        if domain == "solar"
        else ("ukv_ceda_day1_speed_10m", "ens_mean_day1_speed_10m")
    )
    monthly = (
        joined.drop_nulls([column, ens])
        .with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
        .group_by("site", "month")
        .agg(ratio=pl.col(column).mean() / pl.col(ens).mean())
        .sort("site", "month")
    )
    lines = []
    for site in sorted(monthly["site"].unique().to_list()):
        rows = monthly.filter(pl.col("site") == site)
        months, ratios = rows["month"].to_list(), rows["ratio"].to_list()
        for previous, current, month in zip(ratios, ratios[1:], months[1:], strict=False):
            if max(current / previous, previous / current) >= STEP_RATIO_THRESHOLD:
                lines.append(f"{domain} {site} {month}: ratio {previous:.3f} -> {current:.3f}")
    return lines or [f"{domain}: no month-to-month change of {STEP_RATIO_THRESHOLD}x or more"]


def skill_lines(*, domain: DomainType, joined: pl.DataFrame) -> tuple[list[str], bool]:
    """Print UKV-CEDA's skill at each lead day beside Open-Meteo UKV's day 1, and gate it.

    Args:
        domain: `solar` or `wind`.
        joined: Built inputs joined with the published reference columns.

    Returns:
        The printed lines and whether the skill check passed.
    """
    truth, open_meteo, field = (
        ("ghi_cams", "ukv_day1_ghi", "ghi")
        if domain == "solar"
        else ("speed_10m_era5", "ukv_day1_speed_10m", "speed_10m")
    )
    day_columns = [f"ukv_ceda_day{day}_{field}" for day in build.LEAD_DAYS]
    shared = joined.drop_nulls([truth, *day_columns]).drop_nans([truth, *day_columns])
    per_day = [correlation(first=shared[column], second=shared[truth]) for column in day_columns]
    both = joined.drop_nulls([open_meteo, "ukv_ceda_day1_" + field, truth])
    open_meteo_day1 = correlation(first=both[open_meteo], second=both[truth])
    same_rows = correlation(first=both["ukv_ceda_day1_" + field], second=both[truth])
    reasons = skill_gate(
        ukv_ceda=per_day, open_meteo_day1=open_meteo_day1, day1_ceda_same_rows=same_rows
    )
    lines = [
        (
            f"{domain}: correlation of UKV-CEDA {field} with {truth} by lead day, on the "
            f"{shared.height} rows all four days hold, "
            f"{[round(v, 3) for v in per_day]}; on the rows Open-Meteo UKV day 1 also holds, "
            f"UKV-CEDA {same_rows:.3f} and Open-Meteo UKV {open_meteo_day1:.3f}"
        ),
        *(f"FAIL {domain}: {reason}" for reason in reasons),
    ]
    return lines, not reasons


def write_verify_stamp(*, output_dir: Path, passed: bool) -> None:
    """Record whether verification passed and which inputs it read, replacing any earlier stamp.

    Args:
        output_dir: The build's folder, holding the inputs files.
        passed: Whether every gating check passed.
    """
    stamp = {
        "passed": passed,
        "inputs_sha256": {
            domain: build.sha256_of(path=output_dir / f"{domain}_ukv_ceda_inputs.parquet")
            for domain in build.DOMAINS
        },
    }
    (output_dir / build.VERIFY_STAMP_NAME).write_text(json.dumps(stamp, indent=2) + "\n")


def verify(*, published_dir: Path, day4_dir: Path, store_dir: Path, output_dir: Path) -> bool:
    """Run every check and print the result of each.

    Args:
        published_dir: The folder holding the published inputs.
        day4_dir: The folder holding ENS's day-4 mean.
        store_dir: The `UKV-CEDA-T120` folder.
        output_dir: The folder holding the build's outputs.

    Returns:
        Whether every gating check passed.
    """
    store = build.open_store(store_dir=store_dir)
    ok = True
    for domain in build.DOMAINS:
        built = pl.read_parquet(output_dir / f"{domain}_ukv_ceda_inputs.parquet")
        published = build.published_rows(
            published_dir=published_dir, day4_dir=day4_dir, domain=domain
        )
        joined = published.join(built, on=["site", "time"], how="left")
        checked, mismatches = check_values(store=store, domain=domain, built=built)
        sys.stdout.write(f"{domain}: {checked} built values recomputed, {len(mismatches)} differ\n")
        for mismatch in mismatches[:10]:
            sys.stdout.write(f"FAIL {mismatch}\n")
        ok = ok and checked > 0 and not mismatches
        lines, skill_ok = skill_lines(domain=domain, joined=joined)
        sys.stdout.write("\n".join(lines) + "\n")
        cells = site_cells(store=store, domain=domain, sites=built["site"].unique().to_list())
        aligned_lines, aligned = alignment_lines(
            domain=domain, joined=joined, cells=cells, store=store
        )
        sys.stdout.write("\n".join(aligned_lines) + ("\n" if aligned_lines else ""))
        if domain == "wind":
            sys.stdout.write("\n".join(wind_offset_lines(joined=joined)) + "\n")
        sys.stdout.write("\n".join(step_lines(joined=joined, domain=domain)) + "\n")
        ok = ok and skill_ok and aligned
    write_verify_stamp(output_dir=output_dir, passed=ok)
    sys.stdout.write(f"VERIFY {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> int:
    """Verify the built inputs and return 0 if every gating check passes."""
    studies_dir = REPO_DATA_DIR / "studies"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--published-dir", type=Path, default=studies_dir / build.PUBLISHED_DIR_NAME
    )
    parser.add_argument("--day4-dir", type=Path, default=studies_dir / build.DAY4_DIR_NAME)
    parser.add_argument(
        "--store-dir", type=Path, default=studies_dir / "weather" / build.STORE_DIR_NAME
    )
    parser.add_argument("--output-dir", type=Path, default=studies_dir / build.OUTPUT_DIR_NAME)
    args = parser.parse_args()
    ok = verify(
        published_dir=args.published_dir,
        day4_dir=args.day4_dir,
        store_dir=args.store_dir,
        output_dir=args.output_dir,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
