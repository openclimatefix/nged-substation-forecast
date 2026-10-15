"""Assemble the reused demand arms of experiment 00033, with and without the 14-day correction.

Reads the raw (uncorrected) fits and stored baselines of experiments 00013, 00022, 00024, and
00030, adds the 14-day correction (`_c`) and the correction plus the spread factor (`_cs`) with
experiment 00019's code, joins the observed power, and writes one table per arm under
`work/arms/`. Each table holds the run keys, `seed`, the nine quantile columns, and `actual`.
The `n1_cs` table is checked against experiment 00019's stored `stack_minus_h` set.

    uv run python ml-experiments/00033-*/e00033_assemble_demand.py
"""

from pathlib import Path
from typing import Final

import e00033_common as common
import numpy as np
import polars as pl

from studies import experiment_store

DATA: Final[Path] = common.ML_EXPERIMENTS_DATA_DIR
RAW_FITS: Final[dict[str, tuple[Path, str]]] = {
    "n1": (common.ARMS_DIR_00013, "n1"),
    "a7": (DATA / "00030-lagged-power-feature-sets-primary-demand" / "work" / "arms", "a7"),
    "type_id": (DATA / "00024-pooling-groupings-demand-xgboost" / "work" / "arms", "type_id"),
    "own": (DATA / "00024-pooling-groupings-demand-xgboost" / "work" / "arms", "own"),
    **{
        f"w22_{name}": (
            DATA / "00022-era5-weather-ceiling-primary-demand" / "work" / "arms",
            name,
        )
        for name in (
            "ens",
            "era5_all",
            "era5_shortwave",
            "era5_temperature_2m",
            "era5_wind_speed_10m",
        )
    },
}
STORED: Final[tuple[str, ...]] = ("production", "manual_heuristic", "climatology")
NEW_FITS: Final[tuple[str, ...]] = ("n1_refit", "era5_retrained", "cams_shortwave", "monthly")
"""Arms fitted by `e00033_fit_demand.py`; `monthly` holds seed 0 only."""


def main() -> None:
    """Write every reused demand arm in its three forms."""
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    arrays = common.load_series_arrays(series=series)
    actual = common.actual_table(series=series, arrays=arrays)
    raw: dict[str, pl.DataFrame] = {
        name: common.read_seeded(folder=folder, name=file)
        for name, (folder, file) in RAW_FITS.items()
    }
    for name in NEW_FITS:
        seeds = sorted(
            int(p.stem.rsplit("seed", 1)[1]) for p in common.ARMS_DIR.glob(f"{name}_seed*.parquet")
        )
        if not seeds:
            print(f"{name}: not fitted yet, skipped", flush=True)
            continue
        raw[name] = pl.concat(
            [
                pl.read_parquet(common.ARMS_DIR / f"{name}_seed{s}.parquet").select(
                    *common.KEYS, "seed", *common.LEVEL_COLUMNS
                )
                for s in seeds
            ]
        ).with_columns(pl.col("seed").cast(pl.Int8))
    for name in STORED:
        stored = pl.read_parquet(common.ARMS_DIR_00013 / f"{name}.parquet").select(
            *common.KEYS, *common.LEVEL_COLUMNS
        )
        raw[name] = common.repeat_seeds(frame=stored)
    for name, frame in raw.items():
        target = common.ARMS_DIR / f"{name}.parquet"
        if target.exists():
            continue
        corrected = common.with_corrections(base=frame, series=series, arrays=arrays)
        forms = {
            name: frame,
            f"{name}_c": common.stack(frame=corrected, correct=True, spread=False),
            f"{name}_cs": common.stack(frame=corrected, correct=True, spread=True),
        }
        for form, table in forms.items():
            joined = table.join(actual, on=list(common.KEYS), how="left", validate="m:1")
            joined.write_parquet((common.ARMS_DIR / f"{form}.parquet").with_suffix(".tmp"))
            (common.ARMS_DIR / f"{form}.parquet").with_suffix(".tmp").rename(
                common.ARMS_DIR / f"{form}.parquet"
            )
        print(f"{name}: wrote 3 forms", flush=True)
    check = experiment_store.read_predictions(
        experiment_id=19, name="stack_minus_h", row_set_id=common.ROW_SET_ID
    ).sort(*common.KEYS, "seed")
    mine = (
        pl.read_parquet(common.ARMS_DIR / "n1_cs.parquet")
        .filter(pl.col("valid_time") < common.HOLDOUT)
        .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS)
        .sort(*common.KEYS, "seed")
    )
    both = check.join(mine, on=[*common.KEYS, "seed"], suffix="_mine")
    if both.height != check.height or both.height != mine.height:
        msg = "the rebuilt best guess and 00019's stored set hold different cases"
        raise ValueError(msg)
    worst = max(
        float(np.abs((both[c] - both[f"{c}_mine"]).to_numpy()).max()) for c in common.LEVEL_COLUMNS
    )
    print(f"n1_cs vs 00019 stack_minus_h: {both.height} rows, max difference {worst:.2e}")
    if worst > 1e-4:
        msg = "the rebuilt best guess differs from experiment 00019's stored set"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
