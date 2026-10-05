"""Build the UKV-CEDA weather columns for the published rows, at lead days 1 to 4.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1016>. It reads the 03 UTC runs
of the `UKV-CEDA-T120` Icechunk store (`studies/weather_downloads/fetch_ukv_ceda.py --product
ukv-ceda-t120`) and writes, for each technology, `<domain>_ukv_ceda_inputs.parquet` on the published
`(site, time)` rows of `nwp_forecast_comparison`, plus `build.json` and `README.md`, into a new
write-once `--output-dir`. It never writes to the published folder or the store.

**One run serves each lead day.** An hour on day `D` at lead day `N` reads the 03 UTC run of day
`D - N`, which a service running at 09:00 UTC could have read. The lead is `24 * N + h - 3`
(`studies.ifs_single_runs.served_lead_hours` with `run_hour=3`), from 21 to 117 hours over days 1 to
4. Day 5 cannot be built: its lead exceeds the store's 120 hours for every hour after 03:00 UTC.

**Each site reads its nearest 2 km cell.** The match uses the private roster
(`ens_forecast_horizons.site_roster`) and `studies.grid_sampling.nearest_cells`, and no coordinate
is printed or written.

**The store holds hourly steps to lead 48 hours, then 51 and 54, then every third hour to 120.**
Every lead the store lacks is rebuilt from the 3-hourly steps either side of it (`fill_radiation`,
`fill_linear`, and `fill_wind`), and every native lead passes through unchanged. A rebuilt value is
missing unless both steps either side of it are finite.

**Columns.** Solar: `ukv_ceda_day<N>_ghi` is the mean of the two snapshots of `shortwave_down` at
leads `L - 1` and `L`, where `L` is the lead of the hour's label (`hourly_from_snapshots` with
offsets of -60 and 0 minutes). `ukv_ceda_day<N>_temp` is `temperature_1p5m` in degrees Celsius,
averaged the same way. Wind is an instantaneous value at the label: `_speed_10m`, the sine and
cosine of the 10 m direction (`_sin_10m`, `_cos_10m`), and `_speed_925hpa`. Each day also carries
`ukv_ceda_day<N>_init_time`, the run read, and `ukv_ceda_day<N>_cause`, null where the day's columns
are present and otherwise the reason they are not.

**Rows lost.** Every candidate row is counted under the first cause that applies: target absent,
ENS absent, run not listed by CEDA, run missing, run partial, or lead beyond the store. A run with
status `complete` that still lacks a needed value raises, which is how a 925 hPa level below the
ground would show.

**The build refuses to run unless the download has covered the window.** The oldest archived run
must be at or before the first run any row reads, and every run slot in the window that the store
marks as never attempted must be named in `--unlisted-days`, after the fetcher has been re-run once
over those days.

**The older-run build (`--older-run`) is post hoc.** It reads the 15 UTC run of the day before the
ENS run's own day, which starts 9 hours before ENS's 00 UTC run, for lead days 1 to 3. See
`OLDER_RUN` for the lead mapping. It writes `<domain>_ukv_ceda_run15_inputs.parquet`, `build.json`,
and `README.md` into its own write-once folder, applies the same coverage guard and the same
stamp checks, and counts a gap in the 15 UTC run under the same run-gap causes.

`--dry-run` builds one month (`--dry-run-month`), prints the same tables on the runs the store
holds, and writes nothing.

Run it with `uv run python studies/nwp_forecast_comparison/build_ukv_ceda_inputs.py`.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple

import ens_forecast_horizons as efh
import icechunk
import numpy as np
import polars as pl
import zarr
from nwp_forecast_comparison import (
    DomainType,
    candidate_rows,
    rows,
)
from studies.baselines import haurwitz_w_m2
from studies.grid_sampling import nearest_cells
from studies.guards import refuse_to_overwrite
from studies.hourly_means import hourly_from_snapshots
from studies.ifs_single_runs import served_init_time, served_lead_hours
from studies.resample import (
    clear_sky_index_resample,
    interpolate_linear,
    wind_components,
    wind_polar,
)
from studies.solar import zenith
from studies.sources import (
    NFC_DAY4_SHARED_DIR,
    NFC_DIR,
    UKV_CEDA_BLENDS_DIR,
    UKV_CEDA_BLENDS_RUN15_DIR,
    UKV_CEDA_T120_PRODUCT_DIR,
)
from studies.ukv_ceda_profiles import (
    PLAIN_LAST_STEP,
    STATUS_COMPLETE,
    STATUS_MISSING,
    STATUS_PARTIAL,
    T54_STEPS,
    T120_PROFILE,
    T120_STEPS,
)
from studies.wind_direction import sine_cosine

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DOMAINS: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

STORE_DIR: Final[Path] = UKV_CEDA_T120_PRODUCT_DIR
"""The store this script reads."""

PUBLISHED_DIR: Final[Path] = NFC_DIR
DAY4_DIR: Final[Path] = NFC_DAY4_SHARED_DIR
"""The folders holding ENS's mean at days 1 to 3 and at day 4."""


class RunSpec(NamedTuple):
    """Which UKV-CEDA run a build reads for each ENS lead day, and where it writes.

    An hour on day `D` at ENS lead day `N` reads the run that starts at `run_hour` UTC on day
    `D - N - extra_days`, at lead `24 * N + h - run_hour + 24 * extra_days`, where `h` is the hour
    of day of the hour's instant (plus 1 for a solar label, which names the hour ending at it).
    """

    run_hour: int
    extra_days: int
    column_prefix: str
    lead_days: tuple[int, ...]
    output_dir: Path
    description: str


PLANNED_RUN: Final[RunSpec] = RunSpec(
    run_hour=T120_PROFILE.run_hours[0],
    extra_days=0,
    column_prefix="ukv_ceda",
    lead_days=(1, 2, 3, 4),
    output_dir=UKV_CEDA_BLENDS_DIR,
    description="the 03 UTC run of the ENS run's own day, which a 09:00 UTC service could read",
)
"""The planned run: 03 UTC of day `D - N`, at lead `24 * N + h - 3` hours, 3 hours after ENS's 00
UTC run starts. Day 5 would need leads beyond the store's last step."""

OLDER_RUN: Final[RunSpec] = RunSpec(
    run_hour=T120_PROFILE.run_hours[1],
    extra_days=1,
    column_prefix="ukv_ceda_run15",
    lead_days=(1, 2, 3),
    output_dir=UKV_CEDA_BLENDS_RUN15_DIR,
    description="the 15 UTC run of the day before the ENS run's own day, 9 hours before ENS's run",
)
"""The post hoc older run: 15 UTC of day `D - N - 1`, at lead `24 * N + h + 9` hours, which starts 9
hours before ENS's 00 UTC run of day `D - N` and leads 12 hours longer than the planned run's. Day 4
would need leads up to 129 hours, beyond the store's 120, for every hour after 14:00 UTC."""

OUTPUT_DIR: Final[Path] = PLANNED_RUN.output_dir
"""The folder the planned run writes to."""

RUN_HOUR: Final[int] = PLANNED_RUN.run_hour
"""The UTC hour of the run every lead day reads: 03 UTC, readable before 09:00 UTC."""

LEAD_DAYS: Final[tuple[int, ...]] = PLANNED_RUN.lead_days
"""The lead days built. Day 5 would need leads beyond the store's last step."""

N_LEADS: Final[int] = T120_PROFILE.n_steps
"""The length of a run's lead axis: hourly leads 0 to 120."""

LAST_LEAD_HOURS: Final[int] = T120_PROFILE.max_step_hours

HOURLY_LAST_LEAD: Final[int] = 48
"""The last lead of the hourly stretch; every later native lead is 3 hours after the one before."""

NATIVE_LEADS: Final[np.ndarray] = np.array(
    sorted({*range(PLAIN_LAST_STEP + 1), *T54_STEPS, *T120_STEPS})
)
"""Every lead the store holds a value at. Hourly to 48, then 51, 54, and every third hour to 120."""

ANCHOR_LEADS: Final[np.ndarray] = np.array([HOURLY_LAST_LEAD, *NATIVE_LEADS[NATIVE_LEADS > 48]])
"""The 3-hourly steps from lead 48 on, which anchor every rebuilt lead."""

FILLED_LEADS: Final[np.ndarray] = np.array(sorted(set(range(N_LEADS)) - set(NATIVE_LEADS.tolist())))
"""The leads the store lacks, which are rebuilt from the anchors either side."""

_LEFT_ANCHOR: Final[np.ndarray] = (FILLED_LEADS - HOURLY_LAST_LEAD) // 3
"""For each rebuilt lead, the index in `ANCHOR_LEADS` of the anchor before it."""

KELVIN: Final[float] = 273.15
CELL_DISTANCE_LIMIT_KM: Final[float] = 2.0
"""A site farther than this from its nearest cell lies outside the store's cells."""

STORE_VARIABLES: Final[Mapping[DomainType, tuple[str, ...]]] = {
    "solar": ("shortwave_down", "temperature_1p5m"),
    "wind": (
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_speed_925hpa",
        "wind_direction_925hpa",
    ),
}
"""The store arrays each technology reads."""

WEATHER_FIELDS: Final[Mapping[DomainType, tuple[str, ...]]] = {
    "solar": ("ghi", "temp"),
    "wind": ("speed_10m", "sin_10m", "cos_10m", "speed_925hpa"),
}
"""The columns each technology gets per lead day, after `ukv_ceda_day<N>_`."""

CAUSES: Final[tuple[str, ...]] = (
    "target absent",
    "ENS absent",
    "run not listed by CEDA",
    "run missing",
    "run partial",
    "lead beyond the store",
)
"""Why a candidate row lacks a day's columns, in the order a row is attributed to a cause."""

SNAPSHOT_OFFSETS_MINUTES: Final[tuple[int, int]] = (-60, 0)
"""The snapshots, relative to the label, whose mean is the hour ending at the label."""

README_NAME: Final[str] = "README.md"
STAMP_NAME: Final[str] = "build.json"

VERIFY_STAMP_NAME: Final[str] = "verify.json"
"""The stamp `verify_ukv_ceda_inputs.py` writes into the build's folder."""

SLOT_HOURS: Final[int] = T120_PROFILE.cycle_hours


# --- Rebuilding the leads the store lacks ---------------------------------------------------------


def bracketing_anchors_finite(*, values: np.ndarray) -> np.ndarray:
    """Return, for each rebuilt lead, whether both 3-hourly steps either side of it are finite.

    Args:
        values: Shape (n_runs, `N_LEADS`), each run's value at every lead, NaN where missing.

    Returns:
        Shape (n_runs, `len(FILLED_LEADS)`), true where the anchor before and the anchor after a
        rebuilt lead both hold a value.
    """
    anchors = values[:, ANCHOR_LEADS]
    return np.isfinite(anchors[:, _LEFT_ANCHOR]) & np.isfinite(anchors[:, _LEFT_ANCHOR + 1])


def fill_radiation(
    *, snapshots: np.ndarray, clear_sky: np.ndarray, run_hour: int = RUN_HOUR
) -> np.ndarray:
    """Rebuild the radiation at the leads the store lacks, through the clear-sky index.

    The value at each anchor is an instantaneous snapshot, so five conditions hold. The clear sky is
    the instantaneous Haurwitz value at each instant, never a mean over an hour. The positions are
    the instants' leads themselves. Whether an anchor is before noon comes from the UTC hour of its
    instant. Only the leads the store lacks are replaced, because below the daylight floor the
    resampler returns a neighbour's index times the clear sky instead of the native value. A
    rebuilt lead is missing unless both anchors either side of it are finite, because a missing
    anchor in daylight would otherwise be filled silently from a neighbour.

    Args:
        snapshots: Shape (n_runs, `N_LEADS`), `shortwave_down` at every lead, NaN where the store
            has no value.
        clear_sky: Shape (n_runs, `N_LEADS`), the instantaneous clear-sky irradiance at each lead's
            instant, in W/m2.
        run_hour: The UTC hour at which every run starts, which sets whether an anchor is before
            noon.

    Returns:
        A copy of `snapshots` with every lead in `FILLED_LEADS` rebuilt.
    """
    morning = (run_hour + ANCHOR_LEADS) % 24 < 12
    rebuilt = clear_sky_index_resample(
        values=snapshots[:, ANCHOR_LEADS],
        step_clear_sky=clear_sky[:, ANCHOR_LEADS],
        step_midpoints=ANCHOR_LEADS.astype(np.float64),
        morning=morning,
        target_clear_sky=clear_sky[:, FILLED_LEADS],
        target_midpoints=FILLED_LEADS.astype(np.float64),
    )
    output = snapshots.copy()
    output[:, FILLED_LEADS] = np.where(bracketing_anchors_finite(values=snapshots), rebuilt, np.nan)
    return output


def fill_linear(*, values: np.ndarray) -> np.ndarray:
    """Rebuild an instantaneous value at every lead the store lacks by a straight line.

    Args:
        values: Shape (n_runs, `N_LEADS`), the value at every lead, NaN where the store has none.

    Returns:
        A copy of `values` with every lead in `FILLED_LEADS` rebuilt, NaN where either anchor is.
    """
    rebuilt = interpolate_linear(
        values=values[:, ANCHOR_LEADS],
        x=ANCHOR_LEADS.astype(np.float64),
        targets=FILLED_LEADS.astype(np.float64),
    )
    output = values.copy()
    output[:, FILLED_LEADS] = np.where(bracketing_anchors_finite(values=values), rebuilt, np.nan)
    return output


def fill_wind(*, speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild a wind at the leads the store lacks, from its eastward and northward parts.

    Args:
        speed: Shape (n_runs, `N_LEADS`), the wind speed at every lead, NaN where missing.
        direction_deg: The bearing the wind blows from, the same shape, in degrees.

    Returns:
        The speed and direction with every lead in `FILLED_LEADS` rebuilt, NaN where either anchor
        is missing.
    """
    east, north = wind_components(speed=speed, direction_deg=direction_deg)
    finite = bracketing_anchors_finite(values=speed) & bracketing_anchors_finite(
        values=direction_deg
    )
    targets = FILLED_LEADS.astype(np.float64)
    x = ANCHOR_LEADS.astype(np.float64)
    rebuilt_east = interpolate_linear(values=east[:, ANCHOR_LEADS], x=x, targets=targets)
    rebuilt_north = interpolate_linear(values=north[:, ANCHOR_LEADS], x=x, targets=targets)
    rebuilt_speed, rebuilt_direction = wind_polar(u=rebuilt_east, v=rebuilt_north)
    out_speed = speed.copy()
    out_direction = direction_deg.copy()
    out_speed[:, FILLED_LEADS] = np.where(finite, rebuilt_speed, np.nan)
    out_direction[:, FILLED_LEADS] = np.where(finite, rebuilt_direction, np.nan)
    return out_speed, out_direction


# --- Reading the store ----------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreRead:
    """The store, opened read-only at one snapshot."""

    group: zarr.Group
    snapshot_id: str
    statuses: np.ndarray
    """Each slot's status: 0 never archived, else a `STATUS_*` code."""


def zarr_array(*, group: zarr.Group, name: str) -> zarr.Array:
    """Return a member of the group that the store's layout guarantees is an array."""
    member = group[name]
    if not isinstance(member, zarr.Array):
        msg = f"{name} is not an array"
        raise TypeError(msg)
    return member


def open_store(*, store_dir: Path) -> StoreRead:
    """Open the Icechunk store read-only on its main branch.

    Args:
        store_dir: The product's folder, holding `store`.

    Returns:
        The group, the snapshot read, and every slot's status.
    """
    repository = icechunk.Repository.open(
        storage=icechunk.local_filesystem_storage(str(store_dir / "store"))
    )
    session = repository.readonly_session(branch="main")
    group = zarr.open_group(session.store, mode="r")
    product = group.attrs.get("product")
    if product != T120_PROFILE.product_name:
        msg = f"the store holds {product!r}, not {T120_PROFILE.product_name!r}"
        raise ValueError(msg)
    return StoreRead(
        group=group,
        snapshot_id=session.snapshot_id,
        statuses=np.asarray(zarr_array(group=group, name="status")[:], dtype=np.int8),
    )


def slot_of(*, init_time: pl.Expr) -> pl.Expr:
    """Return the store's slot index of a 03 UTC run's start, on its 12-hourly slot grid."""
    return (init_time - pl.lit(T120_PROFILE.slot_epoch)).dt.total_hours() // SLOT_HOURS


def slot_init_time(*, slot: int) -> datetime:
    """Return the start of the run in a slot."""
    return T120_PROFILE.slot_epoch + timedelta(hours=SLOT_HOURS * slot)


def status_at(*, statuses: np.ndarray, slot: int) -> int:
    """Return a slot's status, 0 for a slot beyond the store's last."""
    return int(statuses[slot]) if 0 <= slot < len(statuses) else 0


def check_slot_times(*, store: StoreRead, slots: Sequence[int]) -> None:
    """Raise unless the store's own `init_time` of each slot is the run time the build assumes.

    The build maps a run to its slot by arithmetic on the 12-hourly grid, so this compares that
    arithmetic with the store's `init_time` coordinate. A slot beyond the coordinate's length was
    never archived and is not compared.

    Args:
        store: The opened store.
        slots: The slots to be read.

    Raises:
        ValueError: Naming the slots whose stored time is not the slot grid's.
    """
    stored = np.asarray(zarr_array(group=store.group, name="init_time")[:], dtype=np.int64)
    wrong = [
        slot
        for slot in slots
        if slot < len(stored) and int(stored[slot]) != int(slot_init_time(slot=slot).timestamp())
    ]
    if wrong:
        msg = f"the store's init_time is not the slot grid's for slots {wrong[:5]}"
        raise ValueError(msg)


def read_cells(
    *, store: StoreRead, slots: Sequence[int], variables: Sequence[str], cells: np.ndarray
) -> dict[str, np.ndarray]:
    """Read each variable's lead series at the given cells for every readable slot.

    Args:
        store: The opened store.
        slots: The slots to read.
        variables: Store array names.
        cells: The flat cell indices to keep.

    Returns:
        Each variable to a `float64` array of shape (`len(slots)`, `N_LEADS`, `len(cells)`), NaN for
        a slot whose status is neither complete nor partial.
    """
    output = {
        variable: np.full((len(slots), N_LEADS, len(cells)), np.nan) for variable in variables
    }
    for index, slot in enumerate(slots):
        if status_at(statuses=store.statuses, slot=slot) not in (STATUS_COMPLETE, STATUS_PARTIAL):
            continue
        for variable in variables:
            output[variable][index] = np.asarray(
                zarr_array(group=store.group, name=variable)[slot]
            )[:, cells]
    return output


def nearest_cell_indices(*, store: StoreRead, roster: pl.DataFrame) -> dict[str, int]:
    """Return each site's nearest cell, as an index into the store's flat cell axis.

    Args:
        store: The opened store, whose private cell centres are read and never printed.
        roster: The sites, carrying `site`, `latitude`, and `longitude`.

    Returns:
        Each site label to its cell index.

    Raises:
        ValueError: If a site lies farther than `CELL_DISTANCE_LIMIT_KM` from every cell.
    """
    latitude = np.asarray(zarr_array(group=store.group, name="cell_latitude")[:])
    longitude = np.asarray(zarr_array(group=store.group, name="cell_longitude")[:])
    cells = pl.DataFrame(
        {"cell_id": np.arange(len(latitude)), "latitude": latitude, "longitude": longitude}
    )
    nearest = nearest_cells(sites=roster, cells=cells)
    too_far = nearest.filter(pl.col("distance_km") > CELL_DISTANCE_LIMIT_KM)
    if too_far.height:
        msg = f"sites {too_far['site'].to_list()} lie outside the store's cells"
        raise ValueError(msg)
    return dict(zip(nearest["site"].to_list(), nearest["cell_id"].to_list(), strict=True))


# --- Hourly series per site -----------------------------------------------------------------------


def lead_times(*, init_times: Sequence[datetime]) -> pl.Series:
    """Return the instant of every (run, lead), run-major, as a UTC datetime series."""
    return pl.Series(
        [init + timedelta(hours=int(lead)) for init in init_times for lead in range(N_LEADS)],
        dtype=pl.Datetime("us", "UTC"),
    )


def clear_sky_by_lead(
    *, init_times: Sequence[datetime], latitude: float, longitude: float
) -> np.ndarray:
    """Return the instantaneous Haurwitz clear-sky irradiance at every (run, lead).

    Args:
        init_times: The runs' starts.
        latitude: The site's latitude in degrees north.
        longitude: The site's longitude in degrees east.

    Returns:
        Shape (`len(init_times)`, `N_LEADS`), in W/m2.
    """
    stamps = lead_times(init_times=init_times)
    angle = zenith(stamps=stamps, latitude=latitude, longitude=longitude)
    return haurwitz_w_m2(apparent_zenith_deg=angle).reshape(len(init_times), N_LEADS)


def long_frame(
    *,
    site: str,
    slots: Sequence[int],
    init_times: Sequence[datetime],
    arrays: Mapping[str, np.ndarray],
) -> pl.DataFrame:
    """Stack a site's per-lead arrays into one row per (run, lead), keyed by site and slot.

    Args:
        site: The anonymised site label.
        slots: The runs' slots, in the arrays' first-axis order.
        init_times: The runs' starts, in the same order.
        arrays: Each column to an array of shape (`len(slots)`, `N_LEADS`).

    Returns:
        `key` (`<site>|<slot>`), `time` (the instant of the lead), and one column per array.
    """
    keys = [f"{site}|{slot}" for slot in slots for _ in range(N_LEADS)]
    return pl.DataFrame(
        {
            "key": keys,
            "time": lead_times(init_times=init_times),
            **{name: values.reshape(-1) for name, values in arrays.items()},
        }
    )


def solar_hourly(
    *,
    slots: Sequence[int],
    series: Mapping[str, np.ndarray],
    sites: Sequence[str],
    coordinates: Mapping[str, tuple[float, float]],
    run_hour: int = RUN_HOUR,
) -> pl.DataFrame:
    """Return the hour-ending radiation and temperature of every run at every site.

    Args:
        slots: The runs' slots.
        series: Each store variable to its array from `read_cells`, cells in `sites` order.
        sites: The site labels, in the cells' order.
        coordinates: Each site's latitude and longitude, read from the private roster.
        run_hour: The UTC hour at which every run starts.

    Returns:
        `key`, `time` (the label), `ghi` in W/m2 and `temp` in degrees Celsius, each the mean of the
        two snapshots `SNAPSHOT_OFFSETS_MINUTES` names. An hour missing either snapshot is absent.
    """
    init_times = [slot_init_time(slot=slot) for slot in slots]
    frames = []
    for index, site in enumerate(sites):
        latitude, longitude = coordinates[site]
        radiation = fill_radiation(
            snapshots=series["shortwave_down"][:, :, index],
            clear_sky=clear_sky_by_lead(
                init_times=init_times, latitude=latitude, longitude=longitude
            ),
            run_hour=run_hour,
        )
        temperature = fill_linear(values=series["temperature_1p5m"][:, :, index] - KELVIN)
        frames.append(
            long_frame(
                site=site,
                slots=slots,
                init_times=init_times,
                arrays={"ghi": radiation, "temp": temperature},
            )
        )
    return hourly_from_snapshots(
        frame=pl.concat(frames),
        value_columns=["ghi", "temp"],
        slot_offsets_minutes=SNAPSHOT_OFFSETS_MINUTES,
    )


def wind_instants(
    *, slots: Sequence[int], series: Mapping[str, np.ndarray], sites: Sequence[str]
) -> pl.DataFrame:
    """Return the wind columns of every run at every site and every lead.

    Args:
        slots: The runs' slots.
        series: Each store variable to its array from `read_cells`, cells in `sites` order.
        sites: The site labels, in the cells' order.

    Returns:
        `key`, `time` (the instant, which is the label of a wind hour), and the four wind columns of
        `WEATHER_FIELDS`.
    """
    init_times = [slot_init_time(slot=slot) for slot in slots]
    frames = []
    for index, site in enumerate(sites):
        speed_10m, direction_10m = fill_wind(
            speed=series["wind_speed_10m"][:, :, index],
            direction_deg=series["wind_direction_10m"][:, :, index],
        )
        speed_925, _ = fill_wind(
            speed=series["wind_speed_925hpa"][:, :, index],
            direction_deg=series["wind_direction_925hpa"][:, :, index],
        )
        frames.append(
            long_frame(
                site=site,
                slots=slots,
                init_times=init_times,
                arrays={
                    "speed_10m": speed_10m,
                    "direction_10m": direction_10m,
                    "speed_925hpa": speed_925,
                },
            )
        )
    sine, cosine = sine_cosine(direction_deg=pl.col("direction_10m"))
    return (
        pl.concat(frames)
        .with_columns(sin_10m=sine, cos_10m=cosine)
        .select("key", "time", *WEATHER_FIELDS["wind"])
    )


# --- Assigning rows to runs and causes ------------------------------------------------------------


def with_run(
    *, frame: pl.DataFrame, day: int, domain: DomainType, spec: RunSpec = PLANNED_RUN
) -> pl.DataFrame:
    """Add the run, its slot, its lead, and its key that a lead day reads for each row.

    **The lead mapping.** The run is the one that starts at `spec.run_hour` UTC on the day
    `day + spec.extra_days` days before the hour's own day. The planned run (`PLANNED_RUN`) is the
    03 UTC run of day `D - N`, and the older run (`OLDER_RUN`) is the 15 UTC run of day `D - N - 1`.

    Args:
        frame: Rows carrying `site` and `time`.
        day: The ENS lead day.
        domain: `solar` or `wind`.
        spec: Which run is read.

    Returns:
        `frame` with `init_time`, `slot`, `lead_hours`, and `key`.

    Raises:
        ValueError: If a run does not fall on the store's 12-hourly slot grid.
    """
    found = frame.with_columns(
        init_time=served_init_time(
            time=pl.col("time"), day=day + spec.extra_days, domain=domain, run_hour=spec.run_hour
        ),
        lead_hours=served_lead_hours(
            time=pl.col("time"), day=day + spec.extra_days, domain=domain, run_hour=spec.run_hour
        ),
    ).with_columns(slot=slot_of(init_time=pl.col("init_time")))
    off_grid = found.filter(
        (pl.col("init_time") - pl.lit(T120_PROFILE.slot_epoch)).dt.total_hours() % SLOT_HOURS != 0
    )
    if off_grid.height:
        msg = f"{off_grid.height} runs fall off the store's slot grid"
        raise ValueError(msg)
    return found.with_columns(key=pl.col("site") + "|" + pl.col("slot").cast(pl.String))


def attribute_causes(
    *, frame: pl.DataFrame, statuses: np.ndarray, columns: Sequence[str]
) -> pl.DataFrame:
    """Label each row of one lead day with why its UKV-CEDA columns are absent, or null if present.

    Args:
        frame: Rows from `with_run`, holding `slot`, `lead_hours`, and `columns`, where the columns
            are null or NaN if the run did not supply them.
        statuses: The store's status of every slot.
        columns: The UKV-CEDA columns to test for presence.

    Returns:
        `frame` with `cause`: `lead beyond the store`, `run not listed by CEDA`, `run missing`,
        `run partial`, or null where every column is present. A row whose run is complete and still
        lacks a value is labelled `complete run lacks a value`.
    """
    slots = frame["slot"].to_numpy()
    inside = (slots >= 0) & (slots < len(statuses))
    status = np.where(inside, statuses[np.clip(slots, 0, max(len(statuses) - 1, 0))], 0)
    # The wind arrays mark a missing slot with NaN, not null, so both count as absent.
    absent = pl.any_horizontal(pl.col(column).fill_nan(None).is_null() for column in columns)
    return frame.with_columns(status=pl.Series(status, dtype=pl.Int64)).with_columns(
        cause=pl.when(~absent)
        .then(None)
        .when(pl.col("lead_hours") > LAST_LEAD_HOURS)
        .then(pl.lit("lead beyond the store"))
        .when(pl.col("status") == 0)
        .then(pl.lit("run not listed by CEDA"))
        .when(pl.col("status") == STATUS_MISSING)
        .then(pl.lit("run missing"))
        .when(pl.col("status") == STATUS_PARTIAL)
        .then(pl.lit("run partial"))
        .otherwise(pl.lit("complete run lacks a value"))
    )


def day_columns(
    *, joined: pl.DataFrame, day: int, domain: DomainType, spec: RunSpec = PLANNED_RUN
) -> pl.DataFrame:
    """Rename one lead day's joined columns to `<prefix>_day<N>_*` and keep the stamp and cause.

    Args:
        joined: Rows from `attribute_causes` joined with the hourly series.
        day: The lead day.
        domain: `solar` or `wind`.
        spec: Which run was read, which names the columns' prefix.

    Returns:
        `site`, `time`, the day's weather columns as `Float64`, `ukv_ceda_day<N>_init_time`, and
        `ukv_ceda_day<N>_cause`.
    """
    prefix = f"{spec.column_prefix}_day{day}"
    return joined.select(
        "site",
        "time",
        *(
            pl.col(field).cast(pl.Float64).fill_nan(None).alias(f"{prefix}_{field}")
            for field in WEATHER_FIELDS[domain]
        ),
        pl.col("init_time").alias(f"{prefix}_init_time"),
        pl.col("cause").alias(f"{prefix}_cause"),
    )


def build_day(
    *,
    keys: pl.DataFrame,
    day: int,
    domain: DomainType,
    hourly: pl.DataFrame,
    statuses: np.ndarray,
    spec: RunSpec = PLANNED_RUN,
) -> pl.DataFrame:
    """Join one lead day's UKV-CEDA columns onto the candidate rows.

    Args:
        keys: The candidate rows' `site` and `time`.
        day: The lead day.
        domain: `solar` or `wind`.
        hourly: `solar_hourly` or `wind_instants`' result.
        statuses: The store's status of every slot.
        spec: Which run is read.

    Returns:
        `day_columns`' result for every row of `keys`.
    """
    runs = with_run(frame=keys, day=day, domain=domain, spec=spec)
    joined = runs.join(hourly, on=["key", "time"], how="left")
    labelled = attribute_causes(frame=joined, statuses=statuses, columns=WEATHER_FIELDS[domain])
    return day_columns(joined=labelled, day=day, domain=domain, spec=spec)


def check_complete_runs_hold_values(
    *, columns: pl.DataFrame, domain: DomainType, spec: RunSpec = PLANNED_RUN
) -> None:
    """Raise if a run the store marks complete lacks a value a row needs.

    A 925 hPa level below the ground is stored as a missing value, and dropping the rows it affects
    would select rows on a weather value, so the build stops instead.

    Args:
        columns: The joined `build_day` results of every day, keyed by `site` and `time`.
        domain: `solar` or `wind`.
        spec: Which run was read.

    Raises:
        ValueError: Naming each day and how many rows lack a value under a complete run.
    """
    problems = {
        day: columns.filter(
            pl.col(f"{spec.column_prefix}_day{day}_cause") == "complete run lacks a value"
        ).height
        for day in spec.lead_days
    }
    problems = {day: count for day, count in problems.items() if count}
    if problems:
        msg = (
            f"{domain}: rows whose complete run lacks a value, by lead day: {problems}. "
            "Dropping them would select rows on a weather value; stop and revisit the plan"
        )
        raise ValueError(msg)


# --- The candidate rows and the loss table --------------------------------------------------------


def published_rows(*, published_dir: Path, day4_dir: Path, domain: DomainType) -> pl.DataFrame:
    """Return the candidate rows with ENS's mean at days 1 to 4 and the target.

    Args:
        published_dir: The folder holding `<domain>_forecast_inputs.parquet`.
        day4_dir: The folder holding `<domain>_extra_lead_inputs.parquet`, with ENS's mean at day 4.
        domain: `solar` or `wind`.

    Returns:
        `nwp_forecast_comparison.candidate_rows` joined with the day-4 columns on `site` and `time`.

    Raises:
        ValueError: If a candidate row has no row in the day-4 inputs.
    """
    candidates = candidate_rows(input_dir=published_dir, domain=domain)
    extra = pl.read_parquet(day4_dir / f"{domain}_extra_lead_inputs.parquet")
    day4 = [column for column in extra.columns if column.startswith("ens_mean_day4_")]
    keys = ["site", "time"]
    if candidates.select(keys).join(extra.select(keys), on=keys, how="anti").height:
        msg = f"{domain}: candidate rows are missing from the day-4 inputs"
        raise ValueError(msg)
    return candidates.join(extra.select(*keys, *day4), on=keys, how="left")


def ens_columns(*, domain: DomainType, day: int) -> list[str]:
    """Return ENS's mean columns at one lead day."""
    fields = (
        ("ghi", "temp")
        if domain == "solar"
        else ("speed_100m", "sin_100m", "cos_100m", "speed_10m")
    )
    return [f"ens_mean_day{day}_{field}" for field in fields]


def loss_table(
    *,
    candidates: pl.DataFrame,
    built: pl.DataFrame,
    domain: DomainType,
    spec: RunSpec = PLANNED_RUN,
) -> pl.DataFrame:
    """Count the candidate rows lost to each cause, per lead day.

    Each row is counted under the first cause in `CAUSES` that applies.

    Args:
        candidates: `published_rows`' result.
        built: Every day's `build_day` columns, keyed by `site` and `time`.
        domain: `solar` or `wind`.
        spec: Which run was read.

    Returns:
        One row per lead day with `candidates`, one column per cause, and `kept`.
    """
    joined = candidates.join(built, on=["site", "time"], how="left")
    records = []
    for day in spec.lead_days:
        ens_absent = pl.any_horizontal(
            pl.col(column).is_null() for column in ens_columns(domain=domain, day=day)
        )
        cause = (
            pl.when(pl.col("power_mw").is_null())
            .then(pl.lit("target absent"))
            .when(ens_absent)
            .then(pl.lit("ENS absent"))
            .otherwise(pl.col(f"{spec.column_prefix}_day{day}_cause"))
        )
        counts = dict(joined.select(cause.alias("cause")).group_by("cause").len().iter_rows())
        records.append(
            {
                "day": day,
                "candidates": joined.height,
                **{name: counts.get(name, 0) for name in CAUSES},
                "complete run lacks a value": counts.get("complete run lacks a value", 0),
                "kept": counts.get(None, 0),
            }
        )
    return pl.DataFrame(records)


def beyond_day_share(
    *, candidates: pl.DataFrame, domain: DomainType, spec: RunSpec = PLANNED_RUN
) -> float:
    """Return the share of candidate rows whose lead at the first unbuilt day is inside the store.

    Args:
        candidates: Rows carrying `site` and `time`.
        domain: `solar` or `wind`.
        spec: Which run is read. The first unbuilt day is the one after `spec.lead_days`' last.

    Returns:
        The share, from 0 to 1, of rows whose lead at that day is within the store's 120 hours.
    """
    lead = served_lead_hours(
        time=pl.col("time"),
        day=spec.lead_days[-1] + 1 + spec.extra_days,
        domain=domain,
        run_hour=spec.run_hour,
    )
    return float(candidates.select((lead <= LAST_LEAD_HOURS).mean()).item())


def table_lines(
    *,
    table: pl.DataFrame,
    domain: DomainType,
    shared_kept: int,
    share: float,
    spec: RunSpec = PLANNED_RUN,
) -> list[str]:
    """Format one technology's loss table as Markdown lines.

    Args:
        table: `loss_table`'s result.
        domain: `solar` or `wind`.
        shared_kept: How many rows `nwp_forecast_comparison.rows` keeps, for comparison.
        share: `beyond_day_share`'s result.
        spec: Which run was read.

    Returns:
        The lines.
    """
    names = [*CAUSES, "complete run lacks a value"]
    lines = [
        f"### {domain}",
        "",
        "| Day | Candidate rows | " + " | ".join(names) + " | Kept |",
        "|---|---|" + "---|" * (len(names) + 1),
    ]
    lines += [
        f"| {row['day']} | {row['candidates']} | "
        + " | ".join(str(row[name]) for name in names)
        + f" | {row['kept']} |"
        for row in table.iter_rows(named=True)
    ]
    lines += [
        "",
        (
            f"`nwp_forecast_comparison.rows` would keep {shared_kept} rows. The "
            f"{spec.run_hour:02d} UTC run read here reaches lead day {spec.lead_days[-1] + 1} for "
            f"{share:.1%} of candidate rows, so day {spec.lead_days[-1] + 1} is not built."
        ),
        "",
    ]
    return lines


# --- The coverage guard ---------------------------------------------------------------------------


def check_window_covered(
    *, statuses: np.ndarray, needed_slots: Sequence[int], unlisted_days: Sequence[date]
) -> list[date]:
    """Raise unless the download covers every run the rows read, and return the unlisted days.

    Args:
        statuses: The store's status of every slot.
        needed_slots: The slots any row reads.
        unlisted_days: Dates whose runs the operator confirms CEDA does not list, after one retry.

    Returns:
        The dates of the needed slots that were never archived, which the build then counts as
        `run not listed by CEDA`.

    Raises:
        ValueError: If the oldest archived run is after the first run a row reads, or a needed slot
            that was never archived is on a date outside `unlisted_days`.
    """
    archived = np.flatnonzero(statuses != 0)
    first_needed = min(needed_slots)
    if archived.size == 0 or int(archived.min()) > first_needed:
        oldest = slot_init_time(slot=int(archived.min())) if archived.size else None
        msg = (
            f"the fetcher has not reached the window: the oldest archived run is {oldest}, and "
            f"rows read the run of {slot_init_time(slot=first_needed)}"
        )
        raise ValueError(msg)
    never = sorted(
        {
            slot_init_time(slot=slot).date()
            for slot in needed_slots
            if status_at(statuses=statuses, slot=slot) == 0
        }
    )
    unconfirmed = [day for day in never if day not in set(unlisted_days)]
    if unconfirmed:
        msg = (
            f"runs on {[str(day) for day in unconfirmed]} were never archived. Re-run "
            "fetch_ukv_ceda.py --product ukv-ceda-t120 over those days, then name any day CEDA "
            "still does not list in --unlisted-days"
        )
        raise ValueError(msg)
    return never


# --- Writing --------------------------------------------------------------------------------------


def sha256_of(*, path: Path) -> str:
    """Return a file's SHA-256 as a hex string."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readme_text(
    *, stamp: Mapping[str, object], tables: Sequence[str], spec: RunSpec = PLANNED_RUN
) -> str:
    """Return the folder's `README.md`.

    Args:
        stamp: The contents of `build.json`.
        tables: The loss tables' Markdown lines.
        spec: Which run was read.

    Returns:
        The README text.
    """
    return "\n".join(
        [
            f"# UKV-CEDA inputs for the blends study, {spec.run_hour:02d} UTC run (write-once)",
            "",
            (
                "Built by `studies/nwp_forecast_comparison/build_ukv_ceda_inputs.py` for "
                "`docs/studies/forecasts/ukv-ceda-blends.md`. Never overwrite a file here."
            ),
            "",
            (
                f"- `<domain>_{spec.column_prefix}_inputs.parquet` holds, on the published "
                f"`(site, time)` rows, each UKV-CEDA column at lead days {spec.lead_days[0]} to "
                f"{spec.lead_days[-1]} (`{spec.column_prefix}_day<N>_*`), the run read "
                f"(`{spec.column_prefix}_day<N>_init_time`), and why a day's columns are absent "
                f"(`{spec.column_prefix}_day<N>_cause`). Every row carries only the anonymised "
                "`site` label."
            ),
            (
                "- `build.json` names the Icechunk snapshot read, the window's coverage guard, and "
                "the SHA-256 of each input and output."
            ),
            (
                "- `<domain>_day<N>_*` files are written by `fit_ukv_ceda_blends.py`: per-row "
                "losses, predictions, and a stamp for each technology and lead day, and the "
                "report."
            ),
            "",
            "## Build",
            "",
            f"- Icechunk snapshot: `{stamp['snapshot_id']}`",
            (
                f"- Runs read: {spec.description}, `UKV-CEDA-T120` only. For an hour on day `D` at "
                f"lead day `N`, the run starts at {spec.run_hour:02d} UTC on day "
                f"`D - N - {spec.extra_days}`."
            ),
            f"- Days never archived and named in `--unlisted-days`: {stamp['unlisted_days']}",
            "",
            "## Rows lost by cause",
            "",
            *tables,
        ]
    )


def build_domain(
    *,
    store: StoreRead,
    domain: DomainType,
    candidates: pl.DataFrame,
    spec: RunSpec = PLANNED_RUN,
) -> pl.DataFrame:
    """Build every lead day's UKV-CEDA columns for one technology.

    Args:
        store: The opened store.
        domain: `solar` or `wind`.
        candidates: `published_rows`' result.
        spec: Which run is read.

    Returns:
        `site`, `time`, and every day's columns, stamp, and cause.
    """
    keys = candidates.select("site", "time")
    sites = sorted(keys["site"].unique().to_list())
    roster = efh.site_roster(domain=domain).filter(pl.col("site").is_in(sites)).sort("site")
    if roster["site"].to_list() != sites:
        msg = f"{domain}: the roster lacks sites {sorted(set(sites) - set(roster['site']))}"
        raise ValueError(msg)
    cells = nearest_cell_indices(store=store, roster=roster)
    coordinates = {site: (latitude, longitude) for site, latitude, longitude in roster.iter_rows()}
    slots = sorted(
        {
            int(slot)
            for day in spec.lead_days
            for slot in with_run(frame=keys, day=day, domain=domain, spec=spec)["slot"]
            .unique()
            .to_list()
        }
    )
    check_slot_times(store=store, slots=slots)
    started = time.monotonic()
    series = read_cells(
        store=store,
        slots=slots,
        variables=STORE_VARIABLES[domain],
        cells=np.array([cells[site] for site in sites]),
    )
    _LOG.info("%s: read %d runs in %.0f s", domain, len(slots), time.monotonic() - started)
    if domain == "solar":
        hourly = solar_hourly(
            slots=slots,
            series=series,
            sites=sites,
            coordinates=coordinates,
            run_hour=spec.run_hour,
        )
    else:
        hourly = wind_instants(slots=slots, series=series, sites=sites)
    days = [
        build_day(
            keys=keys, day=day, domain=domain, hourly=hourly, statuses=store.statuses, spec=spec
        )
        for day in spec.lead_days
    ]
    built = days[0]
    for other in days[1:]:
        built = built.join(other, on=["site", "time"], how="left")
    check_complete_runs_hold_values(columns=built, domain=domain, spec=spec)
    return built


def run_build(
    *,
    published_dir: Path,
    day4_dir: Path,
    store_dir: Path,
    output_dir: Path | None,
    unlisted_days: Sequence[date],
    dry_run_month: str | None,
    spec: RunSpec = PLANNED_RUN,
) -> int:
    """Build both technologies' inputs, print the loss tables, and write them unless a dry run.

    Args:
        published_dir: The folder holding the published inputs.
        day4_dir: The folder holding ENS's day-4 mean.
        store_dir: The `UKV-CEDA-T120` folder.
        output_dir: The write-once folder, or `None` for a dry run.
        unlisted_days: Dates CEDA does not list, confirmed after one retry.
        dry_run_month: A `%Y-%m` month to build alone without the coverage guard, or `None`.
        spec: Which run is read.

    Returns:
        0.
    """
    store = open_store(store_dir=store_dir)
    outputs: dict[DomainType, pl.DataFrame] = {}
    tables: list[str] = []
    accepted: list[date] = []
    for domain in DOMAINS:
        candidates = published_rows(published_dir=published_dir, day4_dir=day4_dir, domain=domain)
        if dry_run_month is not None:
            candidates = candidates.filter(pl.col("month") == dry_run_month)
        else:
            needed = {
                int(slot)
                for day in spec.lead_days
                for slot in with_run(
                    frame=candidates.select("site", "time"), day=day, domain=domain, spec=spec
                )["slot"].unique()
            }
            accepted = sorted(
                {
                    *accepted,
                    *check_window_covered(
                        statuses=store.statuses,
                        needed_slots=sorted(needed),
                        unlisted_days=unlisted_days,
                    ),
                }
            )
        started = time.monotonic()
        built = build_domain(store=store, domain=domain, candidates=candidates, spec=spec)
        sys.stdout.write(f"{domain}: built in {time.monotonic() - started:.0f} s\n")
        table = loss_table(candidates=candidates, built=built, domain=domain, spec=spec)
        shared = rows(input_dir=published_dir, domain=domain).height
        lines = table_lines(
            table=table,
            domain=domain,
            shared_kept=shared,
            share=beyond_day_share(candidates=candidates, domain=domain, spec=spec),
            spec=spec,
        )
        sys.stdout.write("\n".join(lines) + "\n")
        tables += lines
        outputs[domain] = built
    if output_dir is None:
        return 0
    write_outputs(
        output_dir=output_dir,
        outputs=outputs,
        tables=tables,
        store=store,
        published_dir=published_dir,
        day4_dir=day4_dir,
        unlisted_days=accepted,
        spec=spec,
    )
    return 0


def write_outputs(
    *,
    output_dir: Path,
    outputs: Mapping[DomainType, pl.DataFrame],
    tables: Sequence[str],
    store: StoreRead,
    published_dir: Path,
    day4_dir: Path,
    unlisted_days: Sequence[date],
    spec: RunSpec = PLANNED_RUN,
) -> None:
    """Write each technology's inputs, `build.json`, and `README.md`, refusing to overwrite.

    Args:
        output_dir: The write-once folder.
        outputs: Each technology's built columns.
        tables: The loss tables' Markdown lines.
        store: The opened store, for its snapshot.
        published_dir: The folder holding the published inputs.
        day4_dir: The folder holding ENS's day-4 mean.
        unlisted_days: The never-archived days the build accepted.
        spec: Which run was read.
    """
    targets = [
        *(output_dir / f"{domain}_{spec.column_prefix}_inputs.parquet" for domain in DOMAINS),
        output_dir / STAMP_NAME,
        output_dir / README_NAME,
    ]
    refuse_to_overwrite(paths=targets)
    output_dir.mkdir(exist_ok=True)
    for domain in DOMAINS:
        outputs[domain].sort("site", "time").write_parquet(
            output_dir / f"{domain}_{spec.column_prefix}_inputs.parquet"
        )
    stamp = {
        "snapshot_id": store.snapshot_id,
        "store": T120_PROFILE.product_name,
        "run_hour": spec.run_hour,
        "extra_days": spec.extra_days,
        "coverage_guard_passed": True,
        "unlisted_days": [str(day) for day in unlisted_days],
        "published_sha256": {
            domain: sha256_of(path=published_dir / f"{domain}_forecast_inputs.parquet")
            for domain in DOMAINS
        },
        "day4_sha256": {
            domain: sha256_of(path=day4_dir / f"{domain}_extra_lead_inputs.parquet")
            for domain in DOMAINS
        },
        "inputs_sha256": {
            domain: sha256_of(path=output_dir / f"{domain}_{spec.column_prefix}_inputs.parquet")
            for domain in DOMAINS
        },
    }
    (output_dir / STAMP_NAME).write_text(json.dumps(stamp, indent=2))
    (output_dir / README_NAME).write_text(readme_text(stamp=stamp, tables=tables, spec=spec))


def check_output_dir(
    *, output_dir: Path, read_only: Sequence[Path], spec: RunSpec = PLANNED_RUN
) -> None:
    """Raise unless `output_dir` is the one folder this script may write to for a run.

    Args:
        output_dir: Where the build would write.
        read_only: The folders the script reads.
        spec: Which run is read, whose `output_dir` is the only folder the run may write to.

    Raises:
        ValueError: If `output_dir` is a folder the script reads, or is not `spec.output_dir`.
    """
    if output_dir.resolve() in {folder.resolve() for folder in read_only} or (
        output_dir.resolve() != spec.output_dir.resolve()
    ):
        msg = f"this run writes only to {spec.output_dir}, not {output_dir}"
        raise ValueError(msg)


def parse_days(*, text: str) -> date:
    """Parse one `YYYY-MM-DD` date for the command line."""
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC).date()


def main() -> int:
    """Build the UKV-CEDA inputs, or time one month without writing (`--dry-run`)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-dir", type=Path, default=PUBLISHED_DIR)
    parser.add_argument("--day4-dir", type=Path, default=DAY4_DIR)
    parser.add_argument("--store-dir", type=Path, default=STORE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--older-run",
        action="store_true",
        help="Post hoc: read the 15 UTC run of the day before ENS's run, for lead days 1 to 3.",
    )
    parser.add_argument(
        "--unlisted-days",
        nargs="*",
        default=[],
        type=lambda text: parse_days(text=text),
        help="Dates CEDA lists no run for, after the fetcher was re-run over them once.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build one month; write nothing.")
    parser.add_argument("--dry-run-month", default="2026-03", help="The month `--dry-run` builds.")
    args = parser.parse_args()
    spec = OLDER_RUN if args.older_run else PLANNED_RUN
    output_dir = args.output_dir or spec.output_dir
    check_output_dir(
        output_dir=output_dir,
        read_only=[args.published_dir, args.day4_dir, args.store_dir],
        spec=spec,
    )
    return run_build(
        published_dir=args.published_dir,
        day4_dir=args.day4_dir,
        store_dir=args.store_dir,
        output_dir=None if args.dry_run else output_dir,
        unlisted_days=args.unlisted_days,
        dry_run_month=args.dry_run_month if args.dry_run else None,
        spec=spec,
    )


if __name__ == "__main__":
    sys.exit(main())
