"""Fit the child forecasts that BSP and GSP models read, and write their medians.

A child is a Primary substation (experiment 00013's N1) or a metered generator (experiment 00020's
best-guess recipe: ERA5-trained, with the power-curve margin for wind). Per seed this writes:

- `children/oof_seed<s>.parquet`: the median of each child on every control-member training row,
  from a model trained without the row's cross-fitting block and the 15 days either side of it.
  Keyed by run and valid time, one `c<id>` column per child in MW.
- `children/val_seed<s>.parquet`: the median of each child, member by member, on the 73 validation
  runs, from the full-window model. Keyed by run, valid time, and member, in MW.

    uv run python ml-experiments/00036-*/e00036_children.py [--smoke]
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00036_common as common
import e00036_gen_common as gen
import numpy as np
import polars as pl
import xgboost as xgb
from studies.power import scan_power

CHILDREN_DIR: Final[Path] = common.WORK_DIR / "children"
RUN_KEYS: Final[tuple[str, ...]] = ("power_fcst_init_time", "valid_time")
MEMBER_KEYS: Final[tuple[str, ...]] = (*RUN_KEYS, "ensemble_member")
GENERATOR_DIR: Final[Path] = common.WORK_DIR / "ens_generators"


def child_column(*, series_id: int) -> str:
    """Return the name of a child's column."""
    return f"c{series_id}"


class FittedQuantiles:
    """One generator's quantile model, with the wind power curve when a base margin is used."""

    def __init__(
        self, *, rows: pl.DataFrame, columns: tuple[str, ...], seed: int, margin: bool
    ) -> None:
        """Fit the model; `margin` makes the power curve the base margin of all 9 levels."""
        self.columns = columns
        self.curve: tuple[np.ndarray, np.ndarray] | None = None
        features = rows.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()
        target = rows["target"].to_numpy()
        base_margin = None
        if margin:
            speed = rows["wind_speed_100m"].to_numpy()
            self.curve = gen.power_curve(speed=speed, power_fraction=target)
            base_margin = self._margin(speed=speed)
        self.booster = xgb.train(
            {**gen.XGBOOST_PARAMETERS, "seed": seed},
            xgb.DMatrix(features, label=target, base_margin=base_margin),
            num_boost_round=gen.N_ROUNDS,
        )

    def _margin(self, *, speed: np.ndarray) -> np.ndarray:
        if self.curve is None:
            msg = "no power curve"
            raise ValueError(msg)
        centres, curve = self.curve
        return np.tile(np.interp(speed, centres, curve)[:, None], (1, len(common.LEVELS)))

    def median(self, *, frame: pl.DataFrame) -> np.ndarray:
        """Return the median (a fraction of capacity) of the sorted quantiles of every row."""
        margin = (
            None if self.curve is None else self._margin(speed=frame["wind_speed_100m"].to_numpy())
        )
        matrix = xgb.DMatrix(
            frame.select([pl.col(c).cast(pl.Float32) for c in self.columns]).to_numpy(),
            base_margin=margin,
        )
        return np.sort(self.booster.predict(matrix), axis=1)[:, common.MEDIAN_INDEX]


def generator_training_rows(
    *, row: dict, analysis: pl.DataFrame, observed: pl.DataFrame
) -> pl.DataFrame:
    """Return one generator's ERA5 rows with their target, before `gen.TRAIN_END`."""
    series = int(row["time_series_id"])
    return (
        analysis.filter(pl.col("time_series_id") == series)
        .join(
            observed.filter(pl.col("time_series_id") == series),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="1:1",
        )
        .filter((pl.col("valid_time") >= gen.ERA5_START) & (pl.col("valid_time") < gen.TRAIN_END))
    )


def generator_columns(*, row: dict) -> tuple[str, ...]:
    """Return a generator's input columns."""
    return gen.SOLAR_FEATURES if row["technology"] == "solar" else gen.WIND_FEATURES


def primary_inputs(*, row: dict, arrays: common.SeriesArrays, frame: pl.DataFrame) -> pl.DataFrame:
    """Return a Primary's rows with derived columns, `lead_hours`, and the three origin lags."""
    own = common.with_derived(
        frame=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
        latitude=row["latitude"],
        longitude=row["longitude"],
    )
    lags = common.origin_lags(
        power=arrays.power,
        target=np.asarray(common.grid_index(times=own["valid_time"])),
        origin=np.asarray(common.grid_index(times=own["power_fcst_init_time"])),
    )
    return own.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    ).with_columns(pl.DataFrame(lags, schema=list(common.LAG_COLUMNS), orient="row"))


def primary_columns() -> list[str]:
    """Return N1's input columns in matrix order."""
    return [
        *common.feature_columns(calendar=True),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
    ]


def fit_primary(*, row: dict, rows: pl.DataFrame, seed: int) -> xgb.Booster:
    """Fit one Primary's N1 on rows that carry finite power."""
    fit_rows = rows.filter(pl.col("power").is_finite())
    age_days = (common.LAST_TRAINING_VALID - fit_rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": seed},
        xgb.DMatrix(
            fit_rows.select(primary_columns()).to_numpy().astype(np.float32),
            label=(fit_rows["power"] / row["effective_capacity_mw"]).to_numpy(),
            weight=0.5 ** (age_days.to_numpy() / common.RECENCY_HALF_LIFE_DAYS),
        ),
        num_boost_round=common.N_ROUNDS,
    )


def median_mw(*, booster: xgb.Booster, matrix: np.ndarray, capacity: float) -> np.ndarray:
    """Return the median of the sorted quantiles of an N1 booster, in MW."""
    raw = np.sort(booster.predict(xgb.DMatrix(matrix)), axis=1)
    return raw[:, common.MEDIAN_INDEX] * capacity


def primary_oof(
    *, row: dict, arrays: common.SeriesArrays, train: pl.DataFrame, seed: int
) -> pl.DataFrame:
    """Return a Primary's out-of-block medians on every control-member training row."""
    rows = primary_inputs(row=row, arrays=arrays, frame=train)
    blocks = common.block_of(valid_time=rows["valid_time"])
    out = np.full(rows.height, np.nan, dtype=np.float32)
    for block in range(len(common.BLOCKS)):
        keep = common.outside_block(valid_time=rows["valid_time"], block=block)
        booster = fit_primary(row=row, rows=rows.filter(pl.Series(keep)), seed=seed)
        mask = blocks == block
        matrix = rows.filter(pl.Series(mask)).select(primary_columns()).to_numpy()
        out[mask] = median_mw(
            booster=booster, matrix=matrix.astype(np.float32), capacity=row["effective_capacity_mw"]
        )
    return rows.select(RUN_KEYS).with_columns(
        pl.Series(child_column(series_id=row["time_series_id"]), out)
    )


def primary_val(
    *, row: dict, arrays: common.SeriesArrays, train: pl.DataFrame, seed: int, runs: list[Path]
) -> pl.DataFrame:
    """Return a Primary's member-by-member medians on the validation runs from the full model."""
    rows = primary_inputs(row=row, arrays=arrays, frame=train)
    booster = fit_primary(row=row, rows=rows, seed=seed)
    column = child_column(series_id=row["time_series_id"])
    pieces = []
    for path in runs:
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
        run = primary_inputs(row=row, arrays=arrays, frame=run).sort(*MEMBER_KEYS)
        values = median_mw(
            booster=booster,
            matrix=run.select(primary_columns()).to_numpy().astype(np.float32),
            capacity=row["effective_capacity_mw"],
        )
        pieces.append(run.select(MEMBER_KEYS).with_columns(pl.Series(column, values)))
    return pl.concat(pieces)


def generator_oof_and_val(
    *,
    row: dict,
    era5: pl.DataFrame,
    control: pl.DataFrame,
    runs: list[Path],
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return a generator's out-of-block control medians and its member medians on validation."""
    columns = generator_columns(row=row)
    margin = row["technology"] == "wind"
    capacity = row["effective_capacity_mw"]
    column = child_column(series_id=row["time_series_id"])
    own = control.filter(pl.col("time_series_id") == row["time_series_id"]).sort(*RUN_KEYS)
    blocks = common.block_of(valid_time=own["valid_time"])
    out = np.full(own.height, np.nan, dtype=np.float32)
    for block in range(len(common.BLOCKS)):
        keep = common.outside_block(valid_time=era5["valid_time"], block=block)
        model = FittedQuantiles(
            rows=era5.filter(pl.Series(keep)), columns=columns, seed=seed, margin=margin
        )
        mask = blocks == block
        out[mask] = model.median(frame=own.filter(pl.Series(mask))) * capacity
    oof = own.select(RUN_KEYS).with_columns(pl.Series(column, out))
    full = FittedQuantiles(rows=era5, columns=columns, seed=seed, margin=margin)
    pieces = []
    for path in runs:
        run = (
            pl.read_parquet(path)
            .filter(pl.col("time_series_id") == row["time_series_id"])
            .sort(*MEMBER_KEYS)
        )
        pieces.append(
            run.select(MEMBER_KEYS).with_columns(
                pl.Series(column, full.median(frame=run) * capacity)
            )
        )
    return oof, pl.concat(pieces)


def join_wide(*, pieces: list[pl.DataFrame], keys: tuple[str, ...]) -> pl.DataFrame:
    """Join per-child tables that share `keys` into one wide table, checking row counts agree."""
    wide = pieces[0]
    for piece in pieces[1:]:
        if piece.height != wide.height:
            msg = "child tables do not hold the same rows"
            raise ValueError(msg)
        wide = wide.join(piece, on=list(keys), how="left", validate="1:1")
    return wide


def observed_mw(*, series: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Return every child's observed power in MW by valid time, one `c<id>` column per child."""
    observed = pl.read_parquet(
        common.OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"]
    )
    primary_ids = series.filter(pl.col("substation_type") == "Primary")["time_series_id"].to_list()
    generators = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
    )
    both = pl.concat(
        [
            observed.filter(pl.col("time_series_id").is_in(primary_ids)),
            generators.cast(observed.schema),
        ]
    ).filter(pl.col("power").is_finite())
    return both.pivot(on="time_series_id", index="valid_time", values="power").rename(
        lambda name: name if name == "valid_time" else f"c{name}"
    )


def main() -> None:
    """Fit and write the children's tables for every seed, skipping tables that exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="one seed, 2 children, 3 runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(common.SEEDS))
    args = parser.parse_args()
    series = common.series_table()
    sites = gen.site_table()
    arrays = common.load_series_arrays(series=series)
    primaries = series.filter(pl.col("substation_type") == "Primary")
    generators = sites
    train = pl.read_parquet(common.ENS_TRAIN_PATH).filter(
        pl.col("valid_time") <= common.LAST_TRAINING_VALID
    )
    val_runs = sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
    gen_runs = sorted(GENERATOR_DIR.glob("init=*.parquet"))
    if args.smoke:
        primaries, generators = primaries.head(1), generators.head(1)
        val_runs, gen_runs = val_runs[:3], gen_runs[:3]
    analysis = pl.read_parquet(gen.ANALYSIS_PATH)
    observed = gen.observed_training_power(sites=sites)
    control = pl.read_parquet(gen.ENS_TRAIN_PATH)
    out_dir = CHILDREN_DIR.with_name("children_smoke") if args.smoke else CHILDREN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds[:1] if args.smoke else args.seeds:
        oof_path, val_path = (
            out_dir / f"oof_seed{seed}.parquet",
            out_dir / f"val_seed{seed}.parquet",
        )
        if oof_path.exists() and val_path.exists():
            continue
        oof_pieces, val_pieces = [], []
        for row in primaries.iter_rows(named=True):
            started = time.monotonic()
            site = arrays[row["site"]]
            oof_pieces.append(primary_oof(row=row, arrays=site, train=train, seed=seed))
            val_pieces.append(
                primary_val(row=row, arrays=site, train=train, seed=seed, runs=val_runs)
            )
            print(
                f"primary seed {seed} {row['site']}: {time.monotonic() - started:.0f} s", flush=True
            )
        for row in generators.iter_rows(named=True):
            started = time.monotonic()
            era5 = generator_training_rows(row=row, analysis=analysis, observed=observed)
            oof, val = generator_oof_and_val(
                row=row, era5=era5, control=control, runs=gen_runs, seed=seed
            )
            oof_pieces.append(oof)
            val_pieces.append(val)
            print(
                f"generator seed {seed} {row['site']}: {time.monotonic() - started:.0f} s",
                flush=True,
            )
        for path, pieces, keys in (
            (oof_path, oof_pieces, common.KEYS[1:]),
            (val_path, val_pieces, MEMBER_KEYS),
        ):
            wide = join_wide(pieces=pieces, keys=keys)
            wide.write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)


if __name__ == "__main__":
    main()
