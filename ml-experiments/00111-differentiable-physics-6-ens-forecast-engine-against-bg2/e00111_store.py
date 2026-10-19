"""Cut the ENS frames into one compact array file per substation, for the DP scripts.

For each of the 7 MW sites the file holds every daily run from 2024-04-01 to 2026-06-30: the run's
origin, and the 51 members' temperature and shortwave flux at the half-hours after the origin. The
runs come from three folders of frames (`e00111_common.FRAME_FOLDERS`). The arrays are float32,
which is what the frames hold.

The script checks, before it writes, that every site has every run, that every run holds the 51
members on consecutive half-hours, and that no run's file holds a repeated key.

    uv run python ml-experiments/00111-*/e00111_store.py [--workers 3]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00111_common as common
import numpy as np
import polars as pl

COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "ensemble_member",
    "valid_time",
    "temperature_2m",
    "shortwave",
)
N_MEMBERS: Final[int] = 51
N_TARGETS: Final[int] = 702
"""Half-hours in a frame: 351 hours after the origin."""


def store_path(*, site: str) -> Path:
    """Return the array file of a site."""
    return common.STORE_DIR / f"{site}.npz"


def read_run(*, run: datetime, ids: dict[int, str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return one run's temperature and shortwave per site, shape (members, targets) each.

    Args:
        run: The run's 00:00 UTC initialisation time.
        ids: The site label of each `time_series_id`.

    Returns:
        A map from site label to the two arrays.

    Raises:
        ValueError: If a site lacks a member or a half-hour, or a key repeats.
    """
    frame = pl.read_parquet(common.frame_path(run=run), columns=list(COLUMNS)).filter(
        pl.col("time_series_id").is_in(list(ids))
    )
    if frame.select("time_series_id", "ensemble_member", "valid_time").is_duplicated().any():
        msg = f"the {run:%Y-%m-%d} frame repeats a key"
        raise ValueError(msg)
    out = {}
    for series_id, label in ids.items():
        rows = frame.filter(pl.col("time_series_id") == series_id).sort(
            "ensemble_member", "valid_time"
        )
        if rows.height != N_MEMBERS * N_TARGETS:
            msg = f"{label} on {run:%Y-%m-%d} has {rows.height} rows, not {N_MEMBERS * N_TARGETS}"
            raise ValueError(msg)
        out[label] = (
            rows["temperature_2m"].to_numpy().reshape(N_MEMBERS, N_TARGETS),
            rows["shortwave"].to_numpy().reshape(N_MEMBERS, N_TARGETS),
        )
    return out


_IDS: dict[int, str] = {}


def _initialise_worker() -> None:
    """Look up each site's series identifier once per worker process."""
    cache = common.load_cache()
    _IDS.update({cache.sites[site].table.time_series_id: site for site in common.ALL_SITES})


def _read(run: datetime) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return read_run(run=run, ids=_IDS)


def main() -> None:
    """Read every run and write one array file per site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    runs = common.run_dates()
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialise_worker) as pool:
        results = list(pool.map(_read, runs, chunksize=8))
    common.STORE_DIR.mkdir(parents=True, exist_ok=True)
    for site in common.ALL_SITES:
        temperature = np.stack([result[site][0] for result in results])
        shortwave = np.stack([result[site][1] for result in results])
        origins = np.array(
            [np.datetime64(common.origin_of(run=run).replace(tzinfo=None), "us") for run in runs]
        )
        target = store_path(site=site)
        temporary = target.with_name(f"{target.stem}.tmp.npz")
        np.savez(temporary, origins=origins, temperature=temperature, shortwave=shortwave)
        temporary.rename(target)
        print(f"{site}: {temperature.shape}, nulls {int(np.isnan(shortwave).sum())}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
