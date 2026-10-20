"""Extract the 37-cell ENS patch of every Primary for experiment 119.

For each daily 00:00 UTC run from 2024-04-01 to 2026-06-30 the script reads only the 5 fields the
encoder uses, at the H3 resolution-5 cells in the union of the sites' patches (a site's cell plus
three hexagonal rings), and keeps for every cell and step the member mean, the member standard
deviation, and the control member (member 0) value. It checks that every run has 51 members and
reports any short run. The patch layout (cell order and each cell's offset in degrees from the
site's centre cell) is written to a private file and never printed.

    uv run python ml-experiments/00119-*/e00119_extract.py [--workers 6]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Final

import e00119_common as common
import h3
import numpy as np
import polars as pl

FIELD_COLUMNS: Final[list[str]] = list(common.FIELDS)
_CELLS: dict[str, object] = {}


def site_patches() -> dict[str, dict[str, list]]:
    """Return, per site, the 37 cells (centre first, then ring by ring) and their offsets.

    The offsets are the cell centre's longitude and latitude minus the site's centre cell's, in
    degrees, and the ring number. A cell outside the stored grid is kept in the list with
    `present` false, so every site has 37 slots.
    """
    cache = common.load_cache()
    stored = set(
        pl.scan_parquet(next(next(common.NWP_DIR.iterdir()).glob("*.parquet")))
        .select("h3_index")
        .unique()
        .collect()["h3_index"]
        .to_list()
    )
    patches: dict[str, dict[str, list]] = {}
    for site in common.SCORED_SITES:
        table = cache.sites[site].table
        centre = h3.latlng_to_cell(table.latitude, table.longitude, 5)
        centre_lat, centre_lon = h3.cell_to_latlng(centre)
        cells, rings, dx, dy, present = [], [], [], [], []
        for ring in range(common.N_RINGS + 1):
            for cell in sorted(h3.grid_ring(centre, ring) if ring else [centre]):
                lat, lon = h3.cell_to_latlng(cell)
                number = h3.str_to_int(cell)
                cells.append(number)
                rings.append(ring)
                dx.append(lon - centre_lon)
                dy.append(lat - centre_lat)
                present.append(number in stored)
        patches[site] = {"cells": cells, "ring": rings, "dx": dx, "dy": dy, "present": present}
    return patches


def read_run(run: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return one run's mean, standard deviation, and control value, and its member count.

    Returns:
        Three float32 arrays of shape (cells, steps, fields), NaN where a step or cell is absent,
        and the number of members found (the minimum over cells).
    """
    cells = _CELLS["cells"]
    index = _CELLS["index"]
    folder = common.NWP_DIR / f"init_time={run:%Y-%m-%d}%2000%3A00%3A00.000000"
    frame = (
        pl.scan_parquet(list(folder.glob("*.parquet")))
        .filter(pl.col("h3_index").is_in(cells))
        .select("h3_index", "ensemble_member", "valid_time", *FIELD_COLUMNS)
        .collect()
    )
    shape = (len(cells), common.N_STEPS, len(common.FIELDS))
    mean = np.full(shape, np.nan, dtype=np.float32)
    std = np.full(shape, np.nan, dtype=np.float32)
    control = np.full(shape, np.nan, dtype=np.float32)
    step_of = {hours: position for position, hours in enumerate(common.STEP_HOURS)}
    lead = ((pl.col("valid_time") - pl.lit(run)).dt.total_hours()).alias("lead")
    frame = frame.with_columns(lead).filter(pl.col("lead").is_in(list(step_of)))
    members = frame.group_by("h3_index").agg(pl.col("ensemble_member").n_unique().alias("n"))
    grouped = frame.group_by("h3_index", "lead").agg(
        *[pl.col(c).mean().alias(f"{c}_mean") for c in FIELD_COLUMNS],
        *[pl.col(c).std().alias(f"{c}_std") for c in FIELD_COLUMNS],
        *[
            pl.col(c).filter(pl.col("ensemble_member") == 0).first().alias(f"{c}_ctl")
            for c in FIELD_COLUMNS
        ],
    )
    for row in grouped.iter_rows(named=True):
        i, j = index[row["h3_index"]], step_of[row["lead"]]
        for k, column in enumerate(FIELD_COLUMNS):
            mean[i, j, k] = row[f"{column}_mean"]
            std[i, j, k] = row[f"{column}_std"]
            control[i, j, k] = row[f"{column}_ctl"]
    return mean, std, control, int(members["n"].min()) if members.height else 0


def _initialise(cells: list[int]) -> None:
    _CELLS["cells"] = cells
    _CELLS["index"] = {cell: position for position, cell in enumerate(cells)}


def main() -> None:
    """Extract every run and write the private frames file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    patches = site_patches()
    union = sorted(
        {c for p in patches.values() for c, ok in zip(p["cells"], p["present"], strict=True) if ok}
    )
    common.write_json_atomic(path=common.patch_path(), content={"patches": patches, "union": union})
    runs = [r for r in common.run_dates() if r < common.HELD_OUT_START]
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_initialise, initargs=(union,)
    ) as pool:
        results = list(pool.map(read_run, runs, chunksize=4))
    short = [
        f"{run:%Y-%m-%d}" for run, r in zip(runs, results, strict=True) if r[3] != common.N_MEMBERS
    ]
    print(f"runs {len(runs)}, union cells {len(union)}, short or missing runs: {short}")
    target = common.frames_path()
    temporary = target.with_name(f"{target.stem}.tmp.npz")
    np.savez(
        temporary,
        runs=np.array([np.datetime64(r.replace(tzinfo=None), "us") for r in runs]),
        cells=np.array(union, dtype=np.int64),
        mean=np.stack([r[0] for r in results]),
        std=np.stack([r[1] for r in results]),
        control=np.stack([r[2] for r in results]),
        members=np.array([r[3] for r in results]),
    )
    temporary.rename(target)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
