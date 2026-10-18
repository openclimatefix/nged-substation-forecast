"""Assemble the arms of experiment 00094 into one prediction table per arm.

An arm is one configuration's monthly refit predictions: `ref` is experiment 00085's stored `pm0`,
and `ht` and `mono` are this experiment's. Every table holds the 365 daily runs of the validation
year, keyed by series, run, valid time, and seed, with the nine quantiles (spread factor 1.15
applied), the observed power over capacity (`actual`), the series label (`site`), and the group.

    uv run python ml-experiments/00094-*/e00094_assemble.py
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Final

import e00094_common as common
import e00094_fit as fitting
import e00094_regimes as regimes
import numpy as np
import polars as pl

ARMS: Final[dict[str, tuple[Path, str, str]]] = {
    "ref": (common.E85_ANCHOR_PREDS_DIR, "pm0", "monthly"),
    "ht": (common.ANCHOR_PREDS_DIR, "ht", "monthly"),
    "mono": (common.ANCHOR_PREDS_DIR, "mono", "monthly"),
}
"""Each arm: the folder of refit predictions, the configuration folder, and the regime."""


def shared_origins() -> list[datetime]:
    """Return the `power_fcst_init_time` of the 365 daily runs, sorted."""
    return sorted(common.all_origins())


def arm_rows(
    *, root: Path, config: str, regime: str, site: str, origins: list[datetime]
) -> pl.DataFrame:
    """Return one series' quantiles for an arm: each run from the refit the regime assigns it.

    Args:
        root: The folder of refit predictions.
        config: The configuration folder under `root`.
        regime: A key of `regimes.REGIMES`.
        site: The series label.
        origins: The `power_fcst_init_time` of the runs to cover.

    Returns:
        The quantiles keyed by series, run, valid time, and seed.

    Raises:
        ValueError: If a refit file lacks a run assigned to it.
        FileNotFoundError: If a refit file is missing.
    """
    pieces = []
    for refit, chosen in regimes.served(name=regime, origins=origins).items():
        path = fitting.job_path(root=root, config=config, site=site, refit=refit)
        frame = pl.read_parquet(path).filter(pl.col("power_fcst_init_time").is_in(chosen))
        if frame["power_fcst_init_time"].n_unique() != len(chosen):
            msg = f"{config} {site} refit {refit}: the file lacks some assigned runs"
            raise ValueError(msg)
        pieces.append(frame)
    return pl.concat(pieces)


def main() -> None:
    """Write every arm whose refit files exist."""
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    arrays = common.load_series_arrays(series=series)
    origins = shared_origins()
    print(f"shared run set: {len(origins)} runs", flush=True)
    for name, (root, config, regime) in ARMS.items():
        target = common.ARMS_DIR / f"{name}.parquet"
        started = time.monotonic()
        pieces = []
        try:
            for row in series.iter_rows(named=True):
                raw = arm_rows(
                    root=root, config=config, regime=regime, site=row["site"], origins=origins
                )
                table = common.stack(frame=raw, correct=False, spread=True)
                index = np.asarray(common.grid_index(times=table["valid_time"]))
                actual = common.take(values=arrays[row["site"]].power, index=index)
                pieces.append(
                    table.with_columns(
                        site=pl.lit(row["site"]),
                        group=pl.lit("Primary"),
                        actual=pl.Series(actual),
                    )
                )
        except (FileNotFoundError, ValueError) as error:
            print(f"{name}: skipped ({error})", flush=True)
            continue
        frame = pl.concat(pieces).sort(*common.KEYS, "seed")
        common.atomic_write(frame=frame, target=target)
        print(f"{name}: {frame.height:,} rows, {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
