"""The series of each bulk supply point (BSP) for experiment 152's falsifier F1.

Copied from experiment 124's `e00124_common.py`, so this folder imports only itself. A BSP is
labelled by its position (0 or 1) in the hierarchy file; its membership is read at run time and
never printed.
"""

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths
from studies.power import scan_power
from studies.sources import REPO_DATA_DIR

HIERARCHY_DIR: Final[Path] = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"
HELD_OUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No row on or after this instant is read."""

GENERATOR_CASE_SITES: Final[frozenset[str]] = frozenset({"D15", "D18"})
"""Series whose data are generator data: they enter the generator sum "G-c" and nothing else."""
BATTERY_OR_BIOFUEL_SITES: Final[frozenset[str]] = frozenset({"D7", "D20"})
GENERATION_TYPES: Final[frozenset[str]] = frozenset(
    {"PV", "Wind", "Biofuel", "BESS", "Other Generation"}
)
"""NGED's `TimeSeriesType` values of the metered generation connected beneath a BSP."""
HEATING_THRESHOLD_C: Final[float] = 15.0
HEATING_SCALE_C: Final[float] = 10.0


@dataclass(frozen=True)
class BspSeries:
    """One BSP's series on the cache grid, in MW.

    Attributes:
        index: The BSP's position (0 or 1) in the hierarchy file; the only label the report uses.
        reading: The BSP's import-positive power, `NaN` where missing.
        mw_list: The metered MW Primaries' series (excluding the generator-case series).
        mva_list: The metered MVA Primaries' apparent-power series.
        generator_list: All metered generation series in MW, the generator-case series (G-c)
            included, export-positive.
        gc_list: The generator-case series alone (G-c).
        mw_labels: The cache label of each series in `mw_list`; private, never printed.
        mva_labels: The cache label of each series in `mva_list`; private, never printed.
        temperature_c: The mean of the members' ERA5 temperatures.
        pv_shape: The mean of the members' unit PV shapes.
        effective_capacity_mw: The BSP's effective capacity, which converts MW to points.
        n_primaries: The number of MW and MVA Primaries beneath it.
        n_generators: The number of generation series beneath it, G-c included.
        primary_series: Each metered MW Primary's own series, keyed by an anonymous number, for the
            injections.
    """

    index: int
    reading: np.ndarray
    mw_list: list[np.ndarray]
    mva_list: list[np.ndarray]
    generator_list: list[np.ndarray]
    gc_list: list[np.ndarray]
    mw_labels: list[str]
    mva_labels: list[str]
    temperature_c: np.ndarray
    pv_shape: np.ndarray
    effective_capacity_mw: float
    n_primaries: int
    n_generators: int
    primary_series: dict[str, np.ndarray]


def _read_hierarchy() -> tuple[dict[str, str], list[dict[str, str]]]:
    """Return substation number -> BSP number, and the monitored series rows."""
    with (HIERARCHY_DIR / "Substations.csv").open() as handle:
        substations = {
            r["substation_number"]: r["bsp_substation_number"] for r in csv.DictReader(handle)
        }
    with (HIERARCHY_DIR / "Monitor Direction.csv").open() as handle:
        series = list(csv.DictReader(handle))
    return substations, series


def _grid_series(*, cache: FeatureCache, power: pl.DataFrame, series_id: int) -> np.ndarray:
    """Return one series' power on the cache grid, `NaN` where there is no reading."""
    out = np.full(len(cache.end_times), np.nan)
    frame = power.filter(pl.col("time_series_id") == series_id)
    if frame.height == 0:
        return out
    stamps = frame["time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    positions = np.searchsorted(cache.end_times, stamps)
    keep = (positions < len(out)) & (cache.end_times[np.minimum(positions, len(out) - 1)] == stamps)
    out[positions[keep]] = frame["power"].to_numpy()[keep]
    return out


def sum_all(*, arrays: list[np.ndarray], n: int) -> np.ndarray:
    """Sum arrays elementwise, `NaN` where any one is `NaN`; no arrays sum to zero everywhere."""
    return np.sum(np.stack(arrays), axis=0) if arrays else np.zeros(n)


def load_bsps(*, cache: FeatureCache) -> list[BspSeries]:
    """Build each BSP's series from the hierarchy files, the cleaned power table, and the cache.

    Args:
        cache: The feature cache that holds the series.

    Returns:
        The two BSPs in the order of the hierarchy file.
    """
    substations, series = _read_hierarchy()
    table = pl.read_parquet(default_input_paths().series_table)
    label_of = dict(zip(table["time_series_id"].to_list(), table["site"].to_list(), strict=True))
    capacity_of = dict(
        zip(
            table["time_series_id"].to_list(), table["effective_capacity_mw"].to_list(), strict=True
        )
    )
    ids = [int(r["TimeSeriesID"]) for r in series]
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(ids))
        .filter(pl.col("time") < pl.lit(HELD_OUT))
        .collect()
    )
    bsp_ids = [int(r["TimeSeriesID"]) for r in series if r["SubstationType"] == "BSP"]
    out = []
    for position, bsp_id in enumerate(bsp_ids):
        number = next(r["SubstationNumber"] for r in series if int(r["TimeSeriesID"]) == bsp_id)
        members = [
            r
            for r in series
            if r["SubstationType"] != "BSP"
            and r["SubstationType"] != "GSP"
            and substations.get(r["SubstationNumber"]) == number
        ]
        mw, mva, gen, gc = [], [], [], []
        mw_labels, mva_labels = [], []
        keys: dict[str, np.ndarray] = {}
        temperature, shape = [], []
        for r in members:
            ts_id = int(r["TimeSeriesID"])
            label = label_of.get(ts_id)
            values = _grid_series(cache=cache, power=power, series_id=ts_id)
            if label is not None and label in cache.sites:
                temperature.append(cache.sites[label].temperature_c)
                shape.append(cache.sites[label].pv_unit_shape)
            is_generator = (
                label in GENERATOR_CASE_SITES
                or label in BATTERY_OR_BIOFUEL_SITES
                or r["TimeSeriesType"] in GENERATION_TYPES
            )
            if is_generator:
                if r["Units"] == "MVA":
                    continue
                gen.append(values)
                if label in GENERATOR_CASE_SITES:
                    gc.append(values)
            elif r["Units"] == "MVA":
                mva.append(values)
                mva_labels.append(label)
            else:
                mw.append(values)
                mw_labels.append(label)
                keys[str(len(keys))] = values
        out.append(
            BspSeries(
                index=position,
                reading=_grid_series(cache=cache, power=power, series_id=bsp_id),
                mw_list=mw,
                mva_list=mva,
                generator_list=gen,
                gc_list=gc,
                mw_labels=mw_labels,
                mva_labels=mva_labels,
                temperature_c=np.mean(np.stack(temperature), axis=0),
                pv_shape=np.mean(np.stack(shape), axis=0),
                effective_capacity_mw=float(capacity_of[bsp_id]),
                n_primaries=len(mw) + len(mva),
                n_generators=len(gen),
                primary_series=keys,
            )
        )
    return out
