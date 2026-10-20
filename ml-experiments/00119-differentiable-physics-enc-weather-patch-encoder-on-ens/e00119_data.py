"""Build experiment 119's row table and patch tensors, and the periods and folds.

A **row** is one (site, run, half-hour) of GAM-K's stored forecast (K0) outside the hand-filtered
switching events. Its weather input is the patch of the last 3-hourly ENS step at or before its
valid time (causal, no interpolation). A **sample** is one (site, run, step) patch; rows index into
the samples. Patches are 37 cells, centre first, then ring by ring.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import e00119_common as common
import e00119_gamk as gamk
import numpy as np
import polars as pl

N_CHANNELS: Final[int] = 2 * len(common.FIELDS)
"""Member mean and member standard deviation of each of the 5 fields."""
LEAD_BANDS: Final[tuple[float, ...]] = (24.0, 48.0, 144.0)
"""Upper edges of lead bands 0 to 2 in hours after the origin; band 3 is the rest."""
TRAIN_F1: Final[tuple[datetime, datetime]] = (
    datetime(2024, 4, 1, tzinfo=UTC),
    datetime(2025, 7, 1, tzinfo=UTC),
)
SCORED_F1: Final[tuple[datetime, datetime]] = (
    datetime(2025, 7, 1, tzinfo=UTC),
    datetime(2026, 7, 1, tzinfo=UTC),
)
TRAIN_F2: Final[tuple[datetime, datetime]] = SCORED_F1
SCORED_F2: Final[tuple[datetime, datetime]] = (
    datetime(2024, 7, 1, tzinfo=UTC),
    datetime(2025, 7, 1, tzinfo=UTC),
)
STEP_ARRAY: Final[np.ndarray] = np.array(common.STEP_HOURS, dtype=np.float64)


@dataclass(frozen=True)
class Dataset:
    """The rows and patch tensors of every scored site.

    Attributes:
        rows: One row per (site, run, half-hour), with `site_index`, `sample`, `step`, `month`
            (year * 12 + month - 1 of the run date), `run_date`, K0's 11 quantiles, `mean`,
            `actual`, `lead_hours`, `hod_sin`, `hod_cos`, and `band`.
        patch: Float32 array (sites, runs, steps, 37, channels), NaN where absent.
        present: Bool array (sites, 37) of cells inside the stored grid.
        geometry: Float32 array (sites, 37, 3) of east offset, north offset, and ring.
        sites: The site labels in `site_index` order.
        run_dates: The 821 run dates.
    """

    rows: pl.DataFrame
    patch: np.ndarray
    present: np.ndarray
    geometry: np.ndarray
    sites: tuple[str, ...]
    run_dates: list[datetime]


def load_patches() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the patch tensor, the cell-present mask, and the cell geometry of the scored sites."""
    frames = np.load(common.frames_path())
    layout = json.loads(common.patch_path().read_text())["patches"]
    union = [int(c) for c in frames["cells"]]
    position = {cell: i for i, cell in enumerate(union)}
    stacked = np.concatenate([frames["mean"], frames["std"]], axis=-1)
    n_runs, _, n_steps, _ = frames["mean"].shape
    patch = np.full(
        (len(common.SCORED_SITES), n_runs, n_steps, common.N_PATCH_CELLS, N_CHANNELS),
        np.nan,
        dtype=np.float32,
    )
    present = np.zeros((len(common.SCORED_SITES), common.N_PATCH_CELLS), dtype=bool)
    geometry = np.zeros((len(common.SCORED_SITES), common.N_PATCH_CELLS, 3), dtype=np.float32)
    for i, site in enumerate(common.SCORED_SITES):
        info = layout[site]
        for slot, (cell, ok) in enumerate(zip(info["cells"], info["present"], strict=True)):
            present[i, slot] = ok
            if ok:
                patch[i, :, :, slot, :] = stacked[:, position[cell], :, :]
        geometry[i, :, 0] = info["dx"]
        geometry[i, :, 1] = info["dy"]
        geometry[i, :, 2] = info["ring"]
    return patch, present, geometry


def build_rows() -> pl.DataFrame:
    """Return K0's rows of every site with the keys the encoder needs, outside switching events."""
    dates = [d for d in common.run_dates() if d < common.HELD_OUT_START]
    run_date = pl.Series([d.replace(tzinfo=None) for d in dates]).dt.replace_time_zone("UTC")
    run_table = pl.DataFrame(
        {
            "run": np.arange(len(dates), dtype=np.int32),
            "run_date": run_date,
            "month": np.array([d.year * 12 + d.month - 1 for d in dates], dtype=np.int32),
        }
    )
    pieces = []
    for index, site in enumerate(common.SCORED_SITES):
        k0 = pl.read_parquet(gamk.k0_path(site=site)).filter(~pl.col("switching_mask"))
        pieces.append(k0.with_columns(site_index=pl.lit(index, dtype=pl.Int8)))
    rows = pl.concat(pieces).join(run_table, on="run", how="left")
    run_lead = (pl.col("valid_time") - pl.col("run_date")).dt.total_seconds() / 3600.0
    step = np.searchsorted(STEP_ARRAY, run_lead_array(rows), side="right") - 1
    hours = rows["valid_time"].dt.hour() + rows["valid_time"].dt.minute() / 60.0
    angle = hours.to_numpy() * (2.0 * np.pi / 24.0)
    band = np.searchsorted(np.array(LEAD_BANDS), rows["lead_hours"].to_numpy(), side="right")
    del run_lead
    return rows.with_columns(
        step=pl.Series(step.astype(np.int16)),
        hod_sin=pl.Series(np.sin(angle).astype(np.float32)),
        hod_cos=pl.Series(np.cos(angle).astype(np.float32)),
        band=pl.Series(band.astype(np.int8)),
    ).sort("site_index", "run", "valid_time")


def run_lead_array(rows: pl.DataFrame) -> np.ndarray:
    """Return each row's valid time in hours after its run's 00:00 UTC initialisation."""
    delta = rows["valid_time"] - rows["run_date"]
    return (delta.dt.total_seconds().to_numpy() / 3600.0).astype(np.float64)


def load_dataset() -> Dataset:
    """Return the dataset of every scored site."""
    patch, present, geometry = load_patches()
    return Dataset(
        rows=build_rows(),
        patch=patch,
        present=present,
        geometry=geometry,
        sites=common.SCORED_SITES,
        run_dates=[d for d in common.run_dates() if d < common.HELD_OUT_START],
    )
