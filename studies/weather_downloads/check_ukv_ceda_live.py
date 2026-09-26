"""Sanity-check the CEDA UKV Icechunk stores while the downloads are still running.

One-off throwaway script for the forecast study. It opens each store read-only, so it is safe to run
next to the single writer of that store, and it runs two checks:

`recent`: for the newest `--n-runs` complete runs of each store, the values of the main variables
sit inside their physical ranges, no variable is all NaN or one constant, each variable is finite at
the leads its files serve and NaN at every other lead, and the `init_time` axis is a gap-free
6-hourly grid across the range archived so far.

`era5`: the UKV analysis fields against ERA5 on disk, at the ERA5 sites' nearest UKV cells. Two
ERA5 files are used: the 2 m temperature (`temp_c`) of `beam_diffuse_open_meteo.parquet`, on the
public 0.25 degree grid, compared at the PV sites A to F; and the native 10 m wind (`u10`, `v10`) of
`wind_native_cds.parquet`, compared at the wind sites W1 to W3. About `--samples-per-year` runs are
sampled per year, spread across the archived range, and the `LEADS_PER_RUN` hourly leads from lead 0
(else from the shortest lead with data) of each are used, where ERA5 has the same valid time. Per
site label the script prints the correlation, the mean bias (UKV minus ERA5), and the root mean
square difference. A negative control repeats each comparison with the UKV cell moved 10 rows
(20 km) south, and the check requires the mean correlation to fall. A spatial check block-averages
UKV temperature onto the ERA5 grid cells that the UKV crop covers completely, and compares the
spatial pattern with ERA5's. A coverage check reports whether the UKV crop contains the ERA5 cell of
every site, and by how many kilometres. Correlations and the negative control are judged only from
`MIN_PAIRS_TO_JUDGE` pairs upward; below that they are printed.

**The script prints only site labels, correlations, biases, and margins in kilometres.** It derives
the nearest cells from the private site and cell coordinates in memory, and never prints or writes
a coordinate, a cell index, or a cell count. It exits with 1 when any check fails.

Run it with `uv run python studies/weather_downloads/check_ukv_ceda_live.py`.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import icechunk
import numpy as np
import polars as pl
import zarr
from fetch_open_meteo_previous_runs import _pv_sites, _wind_sites
from fetch_ukv_ceda import (
    CYCLE_HOURS,
    FIELDS,
    GRID_SPACING_M,
    N_STEPS,
    SLOT_EPOCH,
    STATUS_COMPLETE,
    _array,
    _repository_config,
)
from paths import WEATHER_DOWNLOADS_DIR
from validate_ukv_ceda import NAN_ALLOWED, VALUE_RANGES, check_run_spacing, expected_leads

CheckKindType = Literal["recent", "era5"]

STORE_NAMES: Final[tuple[str, ...]] = ("UKV-CEDA", "UKV-CEDA-part2", "UKV-CEDA-part3")
CHECK_KINDS: Final[tuple[CheckKindType, ...]] = ("recent", "era5")
RECENT_VARIABLES: Final[tuple[str, ...]] = (
    "temperature_1p5m",
    "relative_humidity_1p5m",
    "wind_speed_10m",
    "wind_direction_10m",
    "gust_10m",
    "pressure_msl",
    "shortwave_down",
    "cloud_total",
    "cloud_low",
    "cloud_medium",
    "cloud_high",
)
"""The variables the `recent` check reads: those that the study's models use, with ranges in
`validate_ukv_ceda.VALUE_RANGES`."""

KELVIN: Final[float] = 273.15
ERA5_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "ERA5"
ERA5_GRID_STEP_DEGREES: Final[float] = 0.25
KM_PER_DEGREE: Final[float] = 111.2
LEADS_PER_RUN: Final[int] = 6
"""How many consecutive hourly leads of a sampled run are compared, starting at the shortest lead
that has data. Runs are 6 hours apart, so the valid times never repeat, and a store holding a few
days of runs still yields enough pairs for a correlation."""
SHIFT_ROWS: Final[int] = 10
"""How many rows the negative control moves the UKV cell: 10 cells, which is 20 km."""

MIN_PAIRS: Final[int] = 10
"""Fewest paired times for a label to be compared at all; a label with fewer is skipped."""
MIN_PAIRS_TO_JUDGE: Final[int] = 150
"""Fewest paired times for a correlation, or the negative control, to be judged PASS or FAIL. The
paired times of a store holding a few days are strongly autocorrelated, so a correlation there is
printed but not judged (a calm 3-day window gave a correlation of 0.44 for a wind site whose bias
and spread were right). The bias is always judged."""
TEMPERATURE_MIN_CORRELATION: Final[float] = 0.9
TEMPERATURE_MAX_ABS_BIAS_C: Final[float] = 2.0
WIND_MIN_CORRELATION: Final[float] = 0.7
WIND_MAX_ABS_BIAS_M_S: Final[float] = 2.0
SPATIAL_MIN_CORRELATION: Final[float] = 0.5
MIN_COVERED_ERA5_CELLS: Final[int] = 4
BLOCK_COVERAGE_FRACTION: Final[float] = 0.9
"""A 0.25 degree ERA5 cell counts as covered when UKV has at least this fraction of the cells that
would fill it."""


@dataclass
class Verdicts:
    """The named PASS or FAIL outcomes of one store's checks, in print order."""

    outcomes: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def record(self, name: str, *, passed: bool, note: str | None = None) -> None:
        """Record one check, with a note naming the variables at fault when it failed."""
        self.outcomes[name] = passed
        if note and not passed:
            self.notes.append(f"{name}: {note}")

    @property
    def all_passed(self) -> bool:
        """Whether every recorded check passed."""
        return all(self.outcomes.values())

    def summary(self) -> str:
        """One `name=PASS` or `name=FAIL` token per check."""
        return " ".join(f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in self.outcomes.items())


def open_group(*, store_dir: Path) -> zarr.Group:
    """Open a store read-only, never creating it, on the branch head at this moment."""
    repository = icechunk.Repository.open(
        storage=icechunk.local_filesystem_storage(str(store_dir / "store")),
        config=_repository_config(),
    )
    return zarr.open_group(repository.readonly_session(branch="main").store, mode="r")


def slot_time(slot: int) -> datetime:
    """The initialisation time of a slot."""
    return SLOT_EPOCH + timedelta(hours=CYCLE_HOURS * int(slot))


def check_recent(group: zarr.Group, *, n_runs: int) -> tuple[str, Verdicts]:
    """Check the newest `n_runs` complete runs of one store.

    Args:
        group: The store, opened read-only.
        n_runs: How many of the newest complete runs to read.

    Returns:
        The newest complete init time as text, and the verdicts.
    """
    verdicts = Verdicts()
    statuses = np.asarray(_array(group, "status")[:], dtype=np.int8)
    complete = np.flatnonzero(statuses == STATUS_COMPLETE)
    spacing_ok, _ = check_run_spacing(group)
    visited = np.flatnonzero(statuses != 0)
    never_visited = int((statuses[visited[0] : visited[-1] + 1] == 0).sum()) if visited.size else 0
    verdicts.record(
        "init_times",
        passed=spacing_ok and never_visited == 0,
        note="init_time is off the 6-hourly grid, or a slot inside the range was never visited",
    )
    if complete.size == 0:
        for name in ("ranges", "not_all_nan", "not_constant", "lead_layout"):
            verdicts.record(name, passed=False, note="no complete run")
        return "none", verdicts
    newest = complete[-n_runs:]
    faults: dict[str, set[str]] = {name: set() for name in ("ranges", "nan", "const", "leads")}
    by_name = {spec.variable: spec for spec in FIELDS}
    for variable in RECENT_VARIABLES:
        spec = by_name[variable]
        low, high = VALUE_RANGES[variable]
        served = np.zeros(N_STEPS, dtype=bool)
        served[sorted(expected_leads(spec))] = True
        for slot in newest:
            data = np.asarray(_array(group, variable)[int(slot)])
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                faults["nan"].add(variable)
                continue
            if finite.min() < low or finite.max() > high:
                faults["ranges"].add(variable)
            if finite.max() == finite.min():
                faults["const"].add(variable)
            present = np.isfinite(data)
            if present[~served].any() or (
                variable not in NAN_ALLOWED and not present[served].all()
            ):
                faults["leads"].add(variable)
    for name, key in (
        ("ranges", "ranges"),
        ("not_all_nan", "nan"),
        ("not_constant", "const"),
        ("lead_layout", "leads"),
    ):
        verdicts.record(name, passed=not faults[key], note=", ".join(sorted(faults[key])))
    return f"{slot_time(int(newest[-1])):%Y-%m-%dT%HZ}", verdicts


@dataclass(frozen=True)
class SiteCell:
    """One labelled site, held only in memory: its coordinates are private and never printed."""

    label: str
    latitude: float
    longitude: float


def nearest_cell(
    *, latitude: float, longitude: float, cell_latitude: np.ndarray, cell_longitude: np.ndarray
) -> tuple[int, float]:
    """The index of the UKV cell nearest a point, and its distance in kilometres."""
    d_north = (cell_latitude - latitude) * KM_PER_DEGREE
    d_east = (cell_longitude - longitude) * KM_PER_DEGREE * np.cos(np.radians(latitude))
    distance = np.hypot(d_north, d_east)
    index = int(np.argmin(distance))
    return index, float(distance[index])


@dataclass(frozen=True)
class Crop:
    """The private geometry of one store, used in memory only."""

    latitude: np.ndarray
    longitude: np.ndarray
    row: np.ndarray
    column: np.ndarray
    n_columns: int

    @classmethod
    def read(cls, group: zarr.Group) -> Crop:
        """Read the cell coordinates and the rectangle's shape from a store."""
        shape = group.attrs["rect_shape"]
        return cls(
            latitude=np.asarray(_array(group, "cell_latitude")[:]),
            longitude=np.asarray(_array(group, "cell_longitude")[:]),
            row=np.asarray(_array(group, "cell_row")[:]),
            column=np.asarray(_array(group, "cell_column")[:]),
            n_columns=int(shape[1]),  # ty: ignore[invalid-argument-type, not-subscriptable]
        )

    def shifted(self, index: int) -> int:
        """The cell `SHIFT_ROWS` rows south of `index`, or north of it at the southern edge."""
        south = index + SHIFT_ROWS * self.n_columns
        return south if south < self.latitude.size else index - SHIFT_ROWS * self.n_columns

    def margin_km(self, *, latitude: float, longitude: float) -> float:
        """How far inside the crop a point sits, in km; negative when the point is outside."""
        index, distance = nearest_cell(
            latitude=latitude,
            longitude=longitude,
            cell_latitude=self.latitude,
            cell_longitude=self.longitude,
        )
        spacing_km = GRID_SPACING_M / 1000
        if distance > spacing_km * 0.75:
            return -distance
        row, column = int(self.row[index]), int(self.column[index])
        cells_inside = min(
            row - int(self.row.min()),
            int(self.row.max()) - row,
            column - int(self.column.min()),
            int(self.column.max()) - column,
        )
        return cells_inside * spacing_km


def era5_cell_corners(*, latitude: float, longitude: float) -> list[tuple[float, float]]:
    """The centre of the ERA5 cell containing a point, and its four corners."""
    centre_lat = round(latitude / ERA5_GRID_STEP_DEGREES) * ERA5_GRID_STEP_DEGREES
    centre_lon = round(longitude / ERA5_GRID_STEP_DEGREES) * ERA5_GRID_STEP_DEGREES
    half = ERA5_GRID_STEP_DEGREES / 2
    return [
        (centre_lat + sign_lat * half, centre_lon + sign_lon * half)
        for sign_lat in (-1, 1)
        for sign_lon in (-1, 1)
    ]


def era5_cell_margin_km(crop: Crop, *, site: SiteCell) -> float:
    """The smallest crop margin over the four corners of the site's ERA5 cell, in km."""
    return min(
        crop.margin_km(latitude=lat, longitude=lon)
        for lat, lon in era5_cell_corners(latitude=site.latitude, longitude=site.longitude)
    )


def load_sites(frame: pl.DataFrame) -> list[SiteCell]:
    """Turn a `_pv_sites()` or `_wind_sites()` frame into labelled sites, ordered by label."""
    return [
        SiteCell(label=label, latitude=latitude, longitude=longitude)
        for label, latitude, longitude in frame.select("site", "latitude", "longitude")
        .sort("site")
        .iter_rows()
    ]


@dataclass
class Era5Data:
    """The ERA5 frames that the comparison reads, with naive UTC times."""

    temperature: pl.DataFrame
    wind: pl.DataFrame

    @classmethod
    def read(cls) -> Era5Data:
        """Read `temp_c` on the 0.25 degree grid, and the native 10 m wind speed by site."""
        temperature = pl.read_parquet(
            ERA5_DIR / "beam_diffuse_open_meteo.parquet",
            columns=["time", "latitude", "longitude", "temp_c"],
        ).with_columns(pl.col("time").dt.replace_time_zone(None).dt.cast_time_unit("us"))
        wind = (
            pl.read_parquet(ERA5_DIR / "wind_native_cds.parquet")
            .filter((pl.col("dy") == 0) & (pl.col("dx") == 0))
            .select(
                "site",
                pl.col("time").dt.cast_time_unit("us"),
                speed=(pl.col("u10") ** 2 + pl.col("v10") ** 2).sqrt().cast(pl.Float64),
            )
        )
        return cls(temperature=temperature, wind=wind)


@dataclass
class Samples:
    """The UKV values gathered at the sampled analysis times of one store."""

    times: list[datetime] = field(default_factory=list)
    point: dict[str, list[float]] = field(default_factory=dict)
    control: dict[str, list[float]] = field(default_factory=dict)
    blocks: list[np.ndarray] = field(default_factory=list)
    blocks_control: list[np.ndarray] = field(default_factory=list)
    temperature_labels: list[str] = field(default_factory=list)
    wind_labels: list[str] = field(default_factory=list)
    block_times: list[datetime] = field(default_factory=list)


def sample_slots_by_year(
    complete: np.ndarray, *, per_year: int, years: Sequence[int], last_time: datetime
) -> np.ndarray:
    """Pick up to `per_year` complete slots per year, spread evenly across that year's slots."""
    chosen: list[int] = []
    slot_years = np.array([slot_time(int(slot)).year for slot in complete])
    for year in sorted(set(slot_years.tolist())):
        if years and year not in years:
            continue
        in_year = [
            int(slot) for slot in complete[slot_years == year] if slot_time(int(slot)) <= last_time
        ]
        if not in_year:
            continue
        positions = np.unique(np.linspace(0, len(in_year) - 1, min(per_year, len(in_year))))
        chosen.extend(in_year[int(p)] for p in positions.round())
    return np.array(sorted(set(chosen)), dtype=np.int64)


def analysis_leads(group: zarr.Group, *, variable: str, slot: int) -> tuple[np.ndarray, list[int]]:
    """Read one run's variable, and the `LEADS_PER_RUN` leads from its shortest finite lead."""
    data = np.asarray(_array(group, variable)[int(slot)])
    finite_leads = np.flatnonzero(np.isfinite(data).all(axis=1))
    if finite_leads.size == 0:
        return data, []
    first = int(finite_leads[0])
    return data, [lead for lead in range(first, first + LEADS_PER_RUN) if lead in finite_leads]


def era5_block_cells(crop: Crop, *, centres: list[tuple[float, float]]) -> list[np.ndarray]:
    """For each 0.25 degree ERA5 cell, the UKV cells inside it, or an empty array if not covered."""
    half = ERA5_GRID_STEP_DEGREES / 2
    blocks: list[np.ndarray] = []
    for centre_lat, centre_lon in centres:
        inside = np.flatnonzero(
            (np.abs(crop.latitude - centre_lat) <= half)
            & (np.abs(crop.longitude - centre_lon) <= half)
        )
        area_km2 = (
            ERA5_GRID_STEP_DEGREES
            * KM_PER_DEGREE
            * ERA5_GRID_STEP_DEGREES
            * KM_PER_DEGREE
            * np.cos(np.radians(centre_lat))
        )
        full = area_km2 / (GRID_SPACING_M / 1000) ** 2
        blocks.append(inside if inside.size >= BLOCK_COVERAGE_FRACTION * full else inside[:0])
    return blocks


def gather_samples(
    group: zarr.Group,
    *,
    crop: Crop,
    slots: np.ndarray,
    temperature_cells: dict[str, int],
    wind_cells: dict[str, int],
    blocks: list[np.ndarray],
) -> Samples:
    """Read the sampled runs and collect the values at each site's nearest cell.

    Args:
        group: The store, opened read-only.
        crop: The store's cell geometry.
        slots: The sampled slots.
        temperature_cells: The nearest UKV cell of each PV site label.
        wind_cells: The nearest UKV cell of each wind site label.
        blocks: The UKV cells of each fully covered ERA5 cell, as in `era5_block_cells`.

    Returns:
        The values at the sites in degrees Celsius and metres per second, and the block means.
    """
    samples = Samples(temperature_labels=list(temperature_cells), wind_labels=list(wind_cells))
    for label in (*temperature_cells, *wind_cells):
        samples.point[label] = []
        samples.control[label] = []
    shifted_blocks = [np.array([crop.shifted(int(i)) for i in block]) for block in blocks]
    for slot in slots:
        temperature, temperature_leads = analysis_leads(
            group, variable="temperature_1p5m", slot=int(slot)
        )
        wind, wind_leads = analysis_leads(group, variable="wind_speed_10m", slot=int(slot))
        init = slot_time(int(slot))
        for lead in sorted(set(temperature_leads) & set(wind_leads)):
            temperature_c = temperature[lead].astype(np.float64) - KELVIN
            speed = wind[lead].astype(np.float64)
            samples.times.append(init + timedelta(hours=lead))
            for label, cell in temperature_cells.items():
                samples.point[label].append(float(temperature_c[cell]))
                samples.control[label].append(float(temperature_c[crop.shifted(cell)]))
            for label, cell in wind_cells.items():
                samples.point[label].append(float(speed[cell]))
                samples.control[label].append(float(speed[crop.shifted(cell)]))
            samples.blocks.append(
                np.array([temperature_c[b].mean() if b.size else np.nan for b in blocks])
            )
            samples.blocks_control.append(
                np.array([temperature_c[b].mean() if b.size else np.nan for b in shifted_blocks])
            )
    return samples


def summarise(ukv: np.ndarray, era5: np.ndarray) -> tuple[float, float, float]:
    """The correlation, mean bias (UKV minus ERA5) and root mean square difference."""
    difference = ukv - era5
    return (
        float(np.corrcoef(ukv, era5)[0, 1]),
        float(difference.mean()),
        float(np.sqrt((difference**2).mean())),
    )


def compare_point(
    *,
    samples: Samples,
    era5: pl.DataFrame,
    label: str,
    value_column: str,
) -> tuple[int, tuple[float, float, float], float] | None:
    """Pair one label's UKV values with ERA5 at the same valid time.

    Args:
        samples: The UKV values at the sampled times.
        era5: The ERA5 frame of this label, with `time` and `value_column`.
        label: The site label.
        value_column: The ERA5 column to compare against.

    Returns:
        The number of pairs, the true statistics, and the shifted-cell correlation; `None` when
        fewer than `MIN_PAIRS` times have an ERA5 value.
    """
    ukv = pl.DataFrame(
        {
            "time": samples.times,
            "ukv": samples.point[label],
            "control": samples.control[label],
        }
    ).with_columns(pl.col("time").dt.replace_time_zone(None).dt.cast_time_unit("us"))
    paired = ukv.join(era5.select("time", era5=value_column), on="time", how="inner").drop_nulls()
    if paired.height < MIN_PAIRS:
        return None
    era5_values = paired["era5"].to_numpy()
    true_stats = summarise(paired["ukv"].to_numpy(), era5_values)
    control_correlation = summarise(paired["control"].to_numpy(), era5_values)[0]
    return paired.height, true_stats, control_correlation


def report_points(
    *,
    verdicts: Verdicts,
    kind: str,
    unit: str,
    min_correlation: float,
    max_abs_bias: float,
    results: dict[str, tuple[int, tuple[float, float, float], float] | None],
) -> None:
    """Print one line per label and record the verdicts of one variable."""
    trues: list[float] = []
    controls: list[float] = []
    for label, result in results.items():
        if result is None:
            print(f"    {kind} {label}: skipped, fewer than {MIN_PAIRS} paired times")
            continue
        n_pairs, (correlation, bias, rms), control = result
        judged = n_pairs >= MIN_PAIRS_TO_JUDGE
        if judged:
            trues.append(correlation)
            controls.append(control)
        ok = (correlation >= min_correlation or not judged) and abs(bias) <= max_abs_bias
        verdicts.record(
            f"{kind}_{label}", passed=ok, note=f"corr={correlation:.3f} bias={bias:+.2f}"
        )
        print(
            f"    {kind} {label}: n={n_pairs} corr={correlation:.3f} bias={bias:+.2f} {unit} "
            f"rms={rms:.2f} {unit}; cell moved 10 cells: corr={control:.3f}"
            f"{'' if judged else ' (correlation not judged: few pairs)'}"
        )
    if trues:
        dropped = float(np.mean(controls)) < float(np.mean(trues))
        verdicts.record(
            f"{kind}_negative_control",
            passed=dropped,
            note="moving the UKV cell did not lower the correlation",
        )


def report_spatial(
    *,
    verdicts: Verdicts,
    samples: Samples,
    era5_temperature: pl.DataFrame,
    centres: list[tuple[float, float]],
    covered: np.ndarray,
) -> None:
    """Compare the spatial pattern of block-averaged UKV temperature with ERA5's."""
    if int(covered.sum()) < MIN_COVERED_ERA5_CELLS:
        print("    spatial pattern: skipped, the UKV crop fully covers too few ERA5 cells")
        return
    sampled = [time.replace(tzinfo=None) for time in samples.times]
    era5_by_time = {
        time: frame
        for (time,), frame in era5_temperature.filter(pl.col("time").is_in(sampled))
        .partition_by("time", as_dict=True)
        .items()
    }
    order = sorted(range(len(centres)), key=lambda i: centres[i])
    keep = [i for i in order if covered[i]]
    true_anomalies: list[np.ndarray] = []
    control_anomalies: list[np.ndarray] = []
    era5_anomalies: list[np.ndarray] = []
    biases: list[float] = []
    for time, ukv_blocks, control_blocks in zip(
        samples.times, samples.blocks, samples.blocks_control, strict=True
    ):
        frame = era5_by_time.get(time.replace(tzinfo=None))
        if frame is None:
            continue
        lookup = {
            (lat, lon): t
            for lat, lon, t in frame.select("latitude", "longitude", "temp_c").iter_rows()
        }
        era5_values = np.array([lookup[centres[i]] for i in keep])
        ukv_values = ukv_blocks[keep]
        biases.append(float((ukv_values - era5_values).mean()))
        true_anomalies.append(ukv_values - ukv_values.mean())
        control_anomalies.append(control_blocks[keep] - control_blocks[keep].mean())
        era5_anomalies.append(era5_values - era5_values.mean())
    if len(era5_anomalies) < MIN_PAIRS:
        print("    spatial pattern: skipped, too few sampled times with ERA5 values")
        return
    reference = np.concatenate(era5_anomalies)
    true_correlation = float(np.corrcoef(np.concatenate(true_anomalies), reference)[0, 1])
    control_correlation = float(np.corrcoef(np.concatenate(control_anomalies), reference)[0, 1])
    judged = len(era5_anomalies) >= MIN_PAIRS_TO_JUDGE
    ok = true_correlation >= SPATIAL_MIN_CORRELATION and true_correlation > control_correlation
    verdicts.record(
        "spatial_pattern",
        passed=ok or not judged,
        note=f"corr={true_correlation:.3f} vs {control_correlation:.3f} with the cells moved",
    )
    print(
        f"    spatial pattern of temperature (UKV averaged to 0.25 degree, spatial mean removed): "
        f"n_times={len(era5_anomalies)} corr={true_correlation:.3f} "
        f"mean bias={float(np.mean(biases)):+.2f} C; cells moved 10 cells: "
        f"corr={control_correlation:.3f}{'' if judged else ' (not judged: few times)'}"
    )


def check_era5(
    group: zarr.Group,
    *,
    era5: Era5Data,
    pv_sites: list[SiteCell],
    wind_sites: list[SiteCell],
    samples_per_year: int,
    years: Sequence[int],
) -> Verdicts:
    """Compare one store's UKV analysis fields with ERA5, and check the crop covers ERA5's cells.

    Args:
        group: The store, opened read-only.
        era5: The ERA5 frames.
        pv_sites: The PV sites, which supply the temperature comparison.
        wind_sites: The wind sites, which supply the wind speed comparison.
        samples_per_year: How many analysis times to sample from each year of the store.
        years: The years to keep; empty for all.

    Returns:
        The verdicts.
    """
    verdicts = Verdicts()
    crop = Crop.read(group)
    statuses = np.asarray(_array(group, "status")[:], dtype=np.int8)
    last_time = era5.temperature.select(pl.col("time").max()).item().replace(tzinfo=UTC)
    slots = sample_slots_by_year(
        np.flatnonzero(statuses == STATUS_COMPLETE),
        per_year=samples_per_year,
        years=years,
        last_time=last_time,
    )
    grid_cells = era5.temperature.select("latitude", "longitude").unique()
    centres = [
        (float(lat), float(lon))
        for lat, lon in grid_cells.sort("latitude", "longitude").iter_rows()
    ]
    print("  coverage of the ERA5 cell of each site by the UKV crop (km inside the crop edge):")
    for site in (*pv_sites, *wind_sites):
        margin = era5_cell_margin_km(crop, site=site)
        verdicts.record(
            f"crop_contains_{site.label}", passed=margin >= 0, note=f"margin={margin:.0f} km"
        )
        print(
            f"    {site.label}: {'contained' if margin >= 0 else 'NOT contained'}, "
            f"margin {margin:.0f} km"
        )
    if slots.size == 0:
        print(
            "  no complete run in the requested years with an ERA5 valid time: comparison skipped"
        )
        return verdicts
    temperature_cells, wind_cells = {}, {}
    for site in pv_sites:
        temperature_cells[site.label] = nearest_cell(
            latitude=site.latitude,
            longitude=site.longitude,
            cell_latitude=crop.latitude,
            cell_longitude=crop.longitude,
        )[0]
    for site in wind_sites:
        wind_cells[site.label] = nearest_cell(
            latitude=site.latitude,
            longitude=site.longitude,
            cell_latitude=crop.latitude,
            cell_longitude=crop.longitude,
        )[0]
    blocks = era5_block_cells(crop, centres=centres)
    samples = gather_samples(
        group,
        crop=crop,
        slots=slots,
        temperature_cells=temperature_cells,
        wind_cells=wind_cells,
        blocks=blocks,
    )
    sampled_years = sorted({time.year for time in samples.times})
    print(f"  {len(samples.times)} analysis times sampled, years {sampled_years}")
    temperature_results = {}
    for site in pv_sites:
        centre_lat, centre_lon = (
            round(site.latitude / ERA5_GRID_STEP_DEGREES) * ERA5_GRID_STEP_DEGREES,
            round(site.longitude / ERA5_GRID_STEP_DEGREES) * ERA5_GRID_STEP_DEGREES,
        )
        cell_frame = era5.temperature.filter(
            (pl.col("latitude") == centre_lat) & (pl.col("longitude") == centre_lon)
        )
        temperature_results[site.label] = compare_point(
            samples=samples, era5=cell_frame, label=site.label, value_column="temp_c"
        )
    report_points(
        verdicts=verdicts,
        kind="temperature",
        unit="C",
        min_correlation=TEMPERATURE_MIN_CORRELATION,
        max_abs_bias=TEMPERATURE_MAX_ABS_BIAS_C,
        results=temperature_results,
    )
    wind_results = {
        site.label: compare_point(
            samples=samples,
            era5=era5.wind.filter(pl.col("site") == site.label),
            label=site.label,
            value_column="speed",
        )
        for site in wind_sites
    }
    report_points(
        verdicts=verdicts,
        kind="wind_speed_10m",
        unit="m/s",
        min_correlation=WIND_MIN_CORRELATION,
        max_abs_bias=WIND_MAX_ABS_BIAS_M_S,
        results=wind_results,
    )
    covered = np.array([block.size > 0 for block in blocks])
    report_spatial(
        verdicts=verdicts,
        samples=samples,
        era5_temperature=era5.temperature,
        centres=centres,
        covered=covered,
    )
    return verdicts


def build_parser() -> argparse.ArgumentParser:
    """The command-line options."""
    parser = argparse.ArgumentParser(description="Sanity-check the CEDA UKV stores.")
    parser.add_argument("--store", action="append", choices=STORE_NAMES, dest="stores")
    parser.add_argument("--check", action="append", choices=CHECK_KINDS, dest="checks")
    parser.add_argument("--years", type=int, nargs="+", default=[])
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--samples-per-year", type=int, default=30)
    return parser


def main() -> int:
    """Run the requested checks on each store, and print one block per store.

    Returns:
        0 if every check passed, else 1.
    """
    args = build_parser().parse_args()
    stores: list[str] = args.stores or list(STORE_NAMES)
    checks: list[str] = args.checks or list(CHECK_KINDS)
    all_passed = True
    if "era5" in checks:
        era5 = Era5Data.read()
        pv_sites = load_sites(_pv_sites())
        wind_sites = load_sites(_wind_sites())
    for name in stores:
        group = open_group(store_dir=WEATHER_DOWNLOADS_DIR / name)
        if "recent" in checks:
            newest, verdicts = check_recent(group, n_runs=args.n_runs)
            print(f"{name} recent: newest={newest} {verdicts.summary()}")
            for note in verdicts.notes:
                print(f"  {note}")
            all_passed &= verdicts.all_passed
        if "era5" in checks:
            print(f"{name} era5:")
            verdicts = check_era5(
                group,
                era5=era5,
                pv_sites=pv_sites,
                wind_sites=wind_sites,
                samples_per_year=args.samples_per_year,
                years=args.years,
            )
            print(f"  {verdicts.summary()}")
            all_passed &= verdicts.all_passed
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
