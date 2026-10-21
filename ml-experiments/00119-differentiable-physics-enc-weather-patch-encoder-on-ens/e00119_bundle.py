"""Move experiment 119's rows and patches onto the GPU, with the placebo inputs.

The placebos change only the weather the encoder reads. `distant` reads the patch of the decision
site farthest from the row's site (no shared cells) at the same run and step. `foreign_date` reads
the same site's patch from another run of the same year, within 15 days either side of the same day
of year and at least 2 days away, at the same step (seeded).
"""

from datetime import UTC, datetime
from typing import Final

import e00119_common as common
import e00119_data as data
import e00119_model as model
import e00119_sensitivities as sensitivities
import numpy as np
import polars as pl
import torch

PLACEBO_SEED: Final[int] = 20261020
MAX_OFFSET_DAYS: Final[int] = 15
MIN_OFFSET_DAYS: Final[int] = 2
YEAR_BOUNDARY: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
ROW_COLUMNS: Final[tuple[str, ...]] = (
    "site_index",
    "run",
    "step",
    "band",
    "lead_hours",
    "hod_sin",
    "hod_cos",
    "mean",
    "actual",
    "month",
)


def rows_with_sensitivities(*, dataset: data.Dataset) -> pl.DataFrame:
    """Return the row table with `sens_t` and `sens_g`, raising if any row lacks them."""
    pieces = []
    for index, site in enumerate(dataset.sites):
        frame = pl.read_parquet(sensitivities.sensitivities_path(site=site)).with_columns(
            site_index=pl.lit(index, dtype=pl.Int8)
        )
        pieces.append(frame)
    joined = dataset.rows.join(
        pl.concat(pieces), on=["site_index", "run", "valid_time"], how="left"
    )
    if joined["sens_t"].null_count() or joined["sens_g"].null_count():
        msg = "some rows lack sensitivities"
        raise ValueError(msg)
    return joined


def distant_sites(*, dataset: data.Dataset) -> list[int]:
    """Return, for each site, the index of the farthest other decision site.

    The six patches lie in one region and overlap by 12 to 30 of their 37 cells, so no decision
    site shares no cell with another; `write_decisions` records the shared-cell counts.
    """
    cache = common.load_cache()
    coords = {
        site: (cache.sites[site].table.latitude, cache.sites[site].table.longitude)
        for site in dataset.sites
    }
    out = []
    for site in dataset.sites:
        lat, lon = coords[site]
        options = []
        for other in common.DECISION_SITES:
            if other == site:
                continue
            d_lat, d_lon = (
                coords[other][0] - lat,
                (coords[other][1] - lon) * np.cos(np.radians(lat)),
            )
            options.append((d_lat**2 + d_lon**2, other))
        out.append(dataset.sites.index(max(options)[1]))
    return out


def foreign_runs(*, dataset: data.Dataset) -> np.ndarray:
    """Return an array (sites, runs) giving each (site, run)'s placebo run, drawn with a seed."""
    dates = dataset.run_dates
    n_runs = len(dates)
    year = np.array([0 if d < YEAR_BOUNDARY else 1 for d in dates])
    offsets = [o for o in range(-MAX_OFFSET_DAYS, MAX_OFFSET_DAYS + 1) if abs(o) >= MIN_OFFSET_DAYS]
    out = np.zeros((len(dataset.sites), n_runs), dtype=np.int64)
    for site_index in range(len(dataset.sites)):
        rng = np.random.default_rng(PLACEBO_SEED + site_index)
        for run in range(n_runs):
            valid = [
                run + o for o in offsets if 0 <= run + o < n_runs and year[run + o] == year[run]
            ]
            out[site_index, run] = rng.choice(valid)
    return out


def make_tensors(*, dataset: data.Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the patch, the cell-has-value mask, and the scaled geometry on the device."""
    patch = torch.from_numpy(dataset.patch)
    ok = ~torch.isnan(patch[..., 0])
    patch = torch.nan_to_num(patch, nan=0.0)
    scale = torch.tensor(model.GEOMETRY_SCALE)
    geometry = torch.from_numpy(dataset.geometry) / scale
    return patch.to(model.DEVICE), ok.to(model.DEVICE), geometry.to(model.DEVICE)


def make_bundle(
    *,
    rows: pl.DataFrame,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    arm: model.Arm,
    distant: list[int],
    foreign: np.ndarray,
    actual: np.ndarray | None = None,
) -> model.Bundle:
    """Build a device bundle from a row table for one arm.

    Args:
        rows: Rows from `rows_with_sensitivities`.
        tensors: The output of `make_tensors`.
        arm: The arm, which sets the placebo source of the weather.
        distant: The farthest decision site of each site.
        foreign: Each (site, run)'s placebo run.
        actual: A replacement for the `actual` column, used by the known-answer gate.

    Returns:
        The bundle.
    """
    patch, ok, geometry = tensors
    site = rows["site_index"].to_numpy().astype(np.int64)
    run = rows["run"].to_numpy().astype(np.int64)
    src_site, src_run = site.copy(), run.copy()
    if arm.mode == "distant":
        src_site = np.array(distant)[site]
    elif arm.mode == "foreign_date":
        src_run = foreign[site, run]
    q = rows.select(common.QUANTILE_COLUMNS).to_numpy().astype(np.float32)
    values = {
        "site": site,
        "run": run,
        "step": rows["step"].to_numpy().astype(np.int64),
        "band": rows["band"].to_numpy().astype(np.int64),
        "lead": rows["lead_hours"].to_numpy().astype(np.float32),
        "hod_sin": rows["hod_sin"].to_numpy().astype(np.float32),
        "hod_cos": rows["hod_cos"].to_numpy().astype(np.float32),
        "mean": rows["mean"].to_numpy().astype(np.float32),
        "actual": (rows["actual"].to_numpy() if actual is None else actual).astype(np.float32),
        "sens_t": rows["sens_t"].to_numpy().astype(np.float32),
        "sens_g": rows["sens_g"].to_numpy().astype(np.float32),
        "src_site": src_site,
        "src_run": src_run,
        "month": rows["month"].to_numpy().astype(np.int64),
    }
    columns = {k: torch.from_numpy(v).to(model.DEVICE) for k, v in values.items()}
    columns["q"] = torch.from_numpy(q).to(model.DEVICE)
    return model.Bundle(columns, patch, ok, geometry)
