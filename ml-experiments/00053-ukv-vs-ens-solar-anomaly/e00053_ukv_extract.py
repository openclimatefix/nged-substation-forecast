"""Read the CEDA UKV `shortwave_down` at the cell holding each site, for every run needed.

The cell holding a site is the nearest cell centre (2 km grid). Only the largest distance between a
site and its cell is printed, because the coordinates are private. The values are instantaneous
fluxes at `init_time + step`. Output: `work/ukv/<product>.parquet` with columns `product`
(`t54` for the 00/06/12/18 UTC runs, `t120` for the 03/15 UTC runs), `init_time`, `site`, `step`,
and `ghi`.

    uv run python ml-experiments/00053-*/e00053_ukv_extract.py
"""

import multiprocessing
from datetime import UTC, datetime
from typing import Final

import e00053_common as common
import icechunk
import numpy as np
import polars as pl
import zarr

FIRST_INIT: Final[datetime] = datetime(2024, 3, 25, tzinfo=UTC)
LAST_INIT: Final[datetime] = datetime(2026, 7, 2, tzinfo=UTC)
STORES: Final[tuple[tuple[str, str], ...]] = (
    ("t54", "UKV-CEDA-part2"),
    ("t54", "UKV-CEDA-part3"),
    ("t120", "UKV-CEDA-T120"),
)


def _array(group: zarr.Group, name: str) -> zarr.Array:
    """Return an array the store layout guarantees exists.

    Args:
        group: The opened store root.
        name: The array name.

    Returns:
        The array.

    Raises:
        TypeError: If the member is not an array.
    """
    member = group[name]
    if not isinstance(member, zarr.Array):
        message = f"{name} is not an array"
        raise TypeError(message)
    return member


def read_store(job: tuple[str, str]) -> pl.DataFrame:
    """Return one store's complete runs in the window at every site's cell.

    Args:
        job: The product name and the store folder name.

    Returns:
        Columns `product`, `init_time`, `site`, `step`, `ghi`.
    """
    product, folder = job
    series = common.series_table()
    path = common.STUDY_WEATHER_DIR / folder / "store"
    repository = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))
    group = zarr.open_group(repository.readonly_session(branch="main").store, mode="r")
    latitude = np.asarray(_array(group, "cell_latitude")[:])
    longitude = np.asarray(_array(group, "cell_longitude")[:])
    cells = []
    worst_km = 0.0
    for row in series.iter_rows(named=True):
        dx = (longitude - row["longitude"]) * np.cos(np.radians(row["latitude"])) * 111.32
        dy = (latitude - row["latitude"]) * 110.57
        distance = np.hypot(dx, dy)
        cells.append(int(distance.argmin()))
        worst_km = max(worst_km, float(distance.min()))
    print(f"{folder}: largest site-to-cell distance {worst_km:.2f} km", flush=True)
    init = np.asarray(_array(group, "init_time")[:])
    status = np.asarray(_array(group, "status")[:])
    steps = np.asarray(_array(group, "step")[:])
    first, last = int(FIRST_INIT.timestamp()), int(LAST_INIT.timestamp())
    slots = np.flatnonzero((status == 1) & (init >= first) & (init < last))
    array = _array(group, "shortwave_down")
    pieces = []
    for slot in slots:
        block = np.asarray(array[int(slot), :, :])[:, cells]
        pieces.append(
            pl.DataFrame(
                {
                    "init_time": np.full(block.size, init[slot] * 10**6, dtype=np.int64),
                    "site": np.tile(np.array(series["site"].to_list()), len(steps)),
                    "step": np.repeat(steps, len(cells)).astype(np.int32),
                    "ghi": block.ravel().astype(np.float32),
                }
            )
        )
    frame = pl.concat(pieces).with_columns(
        init_time=pl.col("init_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
        product=pl.lit(product),
    )
    print(f"{folder}: {len(slots)} complete runs", flush=True)
    return frame.filter(pl.col("ghi").is_not_nan()).select(
        "product", "init_time", "site", "step", "ghi"
    )


def main() -> None:
    """Extract all three stores in parallel and write one file per product."""
    with multiprocessing.get_context("spawn").Pool(len(STORES)) as pool:
        frames = pool.map(read_store, STORES)
    combined = pl.concat(frames)
    for product in ("t54", "t120"):
        own = combined.filter(pl.col("product") == product)
        common.write_atomically(frame=own, target=common.WORK_DIR / "ukv" / f"{product}.parquet")
        print(product, own.height, own["init_time"].n_unique(), "runs")


if __name__ == "__main__":
    main()
