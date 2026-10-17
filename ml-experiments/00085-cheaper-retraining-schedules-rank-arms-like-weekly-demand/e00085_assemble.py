"""Assemble the arms of experiment 00085 into one prediction table per arm.

An arm is a retraining schedule (which refit forecasts which daily run) applied to one
configuration's refit predictions. The stored arms read experiment 00048's predictions (`w`, `m`,
`pm`) and experiment 00080's (`w0`); the others read this experiment's `work/anchor_preds/`. Every
table holds the runs of the shared run set only: the daily-gap runs whose latest weekly refit is
one of experiment 00080's 28 anchors. Each table is keyed by series, run, valid time, and seed, and
holds the nine quantiles with the spread factor 1.15 applied, the observed power over capacity
(`actual`), the series label (`site`), and the group (`Primary`).

    uv run python ml-experiments/00085-*/e00085_assemble.py
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Final

import e00085_common as common
import e00085_fit as fitting
import e00085_regimes as regimes
import numpy as np
import polars as pl

ARMS: Final[dict[str, tuple[Path, str, str]]] = {
    "w": (common.E48_ANCHOR_PREDS_DIR, "n1h", "weekly"),
    "m": (common.E48_ANCHOR_PREDS_DIR, "n1h", "monthly"),
    "pm": (common.E48_ANCHOR_PREDS_DIR, "roll12", "monthly"),
    "w0": (common.E80_ANCHOR_PREDS_DIR, "n1h_m0", "weekly"),
    **{
        name: (common.ANCHOR_PREDS_DIR, name, regime)
        for name, regime in (
            ("pm_rep", "monthly"),
            ("pm0", "monthly"),
            ("pm0_rep", "monthly"),
            ("pq", "quarterly"),
            ("pq0", "quarterly"),
            ("pw", "weekly"),
            ("pw0", "weekly"),
            ("fw", "weekly"),
            ("fw0", "weekly"),
            ("fm", "monthly"),
            ("m0", "monthly"),
        )
    },
}
"""Each arm: the folder of refit predictions, the configuration folder, and the regime."""


def shared_origins() -> list[datetime]:
    """Return the `power_fcst_init_time` of the shared run set, sorted.

    Returns:
        The gap runs whose latest weekly refit is one of experiment 00080's 28 anchors.
    """
    anchors = regimes.e80_anchors()
    gap = common.origins_of_subset(subset="gap")
    chosen = regimes.served(name="weekly", origins=gap)
    return sorted(o for refit, origins in chosen.items() if refit in anchors for o in origins)


def arm_rows(
    *, root: Path, config: str, regime: str, site: str, origins: list[datetime]
) -> pl.DataFrame:
    """Return one series' quantiles for an arm: each run from the refit the regime assigns it.

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
