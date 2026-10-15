"""Fit the BSP and GSP arms of experiment 00036 and write their validation quantiles.

Arms (all N1 per series, experiment 00013's recipe, trained on the ENS control member):

- `b0`: N1.
- `b1`: B0 plus the sum of the Primary child medians and the sum of the generator child medians
  (GSPs hold the generator sum only), each ENS member's row receiving the same member's medians.
- `o` (BSPs): B1 with the observed children where observed, else the child medians.
- `b2` (BSPs): least-squares regression of the BSP on each child's median, plus residual quantiles.
- `b3`: N1 trained on ERA5 at the valid times of the training rows, applied to the ENS members.
- `b4` (GSPs): B0 plus a generic wind power curve on the member's 100 m wind speed and the PV_Live
  solar share forecast for the member's weather.

The 51 members are pooled into nine quantiles as in experiment 00013. Output:
`work/arms/<arm>_seed<s>.parquet`, keyed by series, run, and valid time, in fractions of capacity.

    uv run python ml-experiments/00036-*/e00036_fit.py [--smoke]
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00036_children as children
import e00036_common as common
import e00036_gen_common as gen
import numpy as np
import polars as pl
import xgboost as xgb

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RUN_KEYS: Final[tuple[str, ...]] = children.RUN_KEYS
MEMBER_KEYS: Final[tuple[str, ...]] = children.MEMBER_KEYS
RATED_SPEED: Final[float] = 12.0
CUT_IN_SPEED: Final[float] = 3.0
CUT_OUT_SPEED: Final[float] = 25.0
WEATHER_FROM_ERA5: Final[tuple[str, ...]] = common.WEATHER


def wind_index(*, speed: pl.Expr) -> pl.Expr:
    """Return a generic wind power curve of a 100 m wind speed, from 0 to 1 of rated power."""
    ramp = (speed**3 - CUT_IN_SPEED**3) / (RATED_SPEED**3 - CUT_IN_SPEED**3)
    return (
        pl.when(speed < CUT_IN_SPEED)
        .then(0.0)
        .when(speed < RATED_SPEED)
        .then(ramp)
        .when(speed <= CUT_OUT_SPEED)
        .then(1.0)
        .otherwise(0.0)
    )


def sum_columns(*, columns: list[str]) -> pl.Expr:
    """Return the sum of columns, null when any is null."""
    return pl.when(pl.all_horizontal(pl.col(c).is_not_null() for c in columns)).then(
        pl.sum_horizontal(columns)
    )


def child_columns(*, ids: list[int]) -> list[str]:
    """Return the child column names of some series ids."""
    return [children.child_column(series_id=i) for i in ids]


class Context:
    """The tables an arm reads, loaded once per seed."""

    def __init__(self, *, seed: int, smoke: bool) -> None:
        """Load the series, hierarchy, arrays, training frame, and child and solar tables."""
        self.seed = seed
        self.series = common.series_table()
        self.sites = gen.site_table()
        self.hierarchy = common.load_hierarchy(
            series=self.series, generator_ids=set(self.sites["time_series_id"].to_list())
        )
        self.arrays = common.load_series_arrays(series=self.series)
        folder = (
            children.CHILDREN_DIR.with_name("children_smoke") if smoke else children.CHILDREN_DIR
        )
        self.oof = pl.read_parquet(folder / f"oof_seed{seed}.parquet")
        self.val_children = pl.read_parquet(folder / f"val_seed{seed}.parquet")
        pv_dir = common.WORK_DIR / "pv"
        self.pv_train = pl.read_parquet(pv_dir / f"train_seed{seed}.parquet")
        self.pv_val = pl.read_parquet(pv_dir / f"val_seed{seed}.parquet")
        self.train = pl.read_parquet(common.ENS_TRAIN_PATH).filter(
            pl.col("valid_time") <= common.LAST_TRAINING_VALID
        )
        self.observed = children.observed_mw(series=self.series, sites=self.sites)
        self.val_files = sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
        self.fill_counts: dict[tuple[int, str], tuple[int, int]] = {}

    def child_ids(self, *, row: dict) -> tuple[list[int], list[int]]:
        """Return the Primary and generator children of a BSP or GSP (GSPs: generators only)."""
        bsp = row["time_series_id"]
        if row["substation_type"] == "GSP":
            bsp = self.hierarchy.gsp_to_bsp[bsp]
            return [], self.hierarchy.generators[bsp]
        return self.hierarchy.primaries[bsp], self.hierarchy.generators[bsp]


def base_inputs(*, row: dict, arrays: common.SeriesArrays, frame: pl.DataFrame) -> pl.DataFrame:
    """Return a series' rows with derived columns, `lead_hours`, and the three origin lags."""
    return children.primary_inputs(row=row, arrays=arrays, frame=frame)


def with_extras(
    *, rows: pl.DataFrame, arm: str, row: dict, context: Context, training: bool
) -> tuple[pl.DataFrame, list[str]]:
    """Add an arm's extra columns to rows and return them with the extra column names."""
    primaries, generators = context.child_ids(row=row)
    keys = list(RUN_KEYS if training else MEMBER_KEYS)
    child_table = context.oof if training else context.val_children
    if arm in ("b1", "o"):
        names = child_columns(ids=[*primaries, *generators])
        rows = rows.join(child_table.select(*keys, *names), on=keys, how="left", validate="m:1")
        if arm == "o":
            observed = context.observed.select("valid_time", *names).rename(
                {n: f"{n}_obs" for n in names}
            )
            rows = rows.join(observed, on="valid_time", how="left", validate="m:1")
            filled = sum(rows[f"{n}_obs"].is_null().sum() for n in names)
            context.fill_counts[row["time_series_id"], "train" if training else "val"] = (
                int(filled),
                rows.height * len(names),
            )
            rows = rows.with_columns([pl.coalesce(f"{n}_obs", n).alias(n) for n in names])
        extras = []
        if primaries:
            rows = rows.with_columns(
                primary_children_mw=sum_columns(columns=child_columns(ids=primaries))
            )
            extras.append("primary_children_mw")
        rows = rows.with_columns(
            generator_children_mw=sum_columns(columns=child_columns(ids=generators))
        )
        return rows, [*extras, "generator_children_mw"]
    if arm == "b4":
        pv = context.pv_train if training else context.pv_val
        pv = pv.filter(pl.col("time_series_id") == row["time_series_id"]).select(*keys, "pv_share")
        rows = rows.join(pv, on=keys, how="left", validate="m:1").with_columns(
            wind_power_index=wind_index(speed=pl.col("wind_speed_100m"))
        )
        return rows, ["wind_power_index", "pv_share"]
    return rows, []


def era5_substituted(*, rows: pl.DataFrame, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return training rows whose weather columns are ERA5 at the valid time."""
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    names = common.feature_columns(calendar=True)
    return rows.with_columns(
        *[pl.Series(c, arrays.features[index, names.index(c)]) for c in WEATHER_FROM_ERA5]
    )


def fit_arm(
    *, arm: str, row: dict, context: Context, rows: pl.DataFrame, extras: list[str]
) -> xgb.Booster:
    """Fit one arm's quantile model on training rows with finite power."""
    fit_rows = rows.filter(pl.col("power").is_finite())
    if arm == "b3":
        fit_rows = era5_substituted(rows=fit_rows, arrays=context.arrays[row["site"]])
    columns = [*children.primary_columns(), *extras]
    age_days = (common.LAST_TRAINING_VALID - fit_rows["valid_time"]).dt.total_seconds() / 86_400
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": context.seed},
        xgb.DMatrix(
            fit_rows.select(columns).to_numpy().astype(np.float32),
            label=(fit_rows["power"] / row["effective_capacity_mw"]).to_numpy(),
            weight=0.5 ** (age_days.to_numpy() / common.RECENCY_HALF_LIFE_DAYS),
        ),
        num_boost_round=common.N_ROUNDS,
    )


def pool(*, run: pl.DataFrame, raw: np.ndarray) -> pl.DataFrame:
    """Pool each case's members into the mixture's nine quantiles."""
    members = run["ensemble_member"].n_unique()
    if run.height != run.select(*RUN_KEYS).n_unique() * members or members != common.MEMBERS:
        msg = "every valid time must hold all 51 members"
        raise ValueError(msg)
    member_quantiles = np.sort(raw, axis=1).reshape(-1, members, len(common.LEVELS))
    keys = run.select(*RUN_KEYS).unique(maintain_order=True)
    return keys.hstack(
        pl.DataFrame(
            common.mixture_quantiles(member_quantiles=member_quantiles),
            schema=list(common.LEVEL_COLUMNS),
        )
    )


def validation_run(*, path: Path, row: dict, context: Context) -> pl.DataFrame:
    """Return one run's members for a series with derived columns and lags, sorted by member."""
    run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
    return base_inputs(row=row, arrays=context.arrays[row["site"]], frame=run).sort(*MEMBER_KEYS)


def predict_booster(
    *, arm: str, booster: xgb.Booster, row: dict, context: Context, extras: list[str]
) -> pl.DataFrame:
    """Return an arm's pooled quantiles on every validation run."""
    columns = [*children.primary_columns(), *extras]
    pieces = []
    for path in context.val_files:
        run = validation_run(path=path, row=row, context=context)
        run, _ = with_extras(rows=run, arm=arm, row=row, context=context, training=False)
        raw = booster.predict(xgb.DMatrix(run.select(columns).to_numpy().astype(np.float32)))
        pieces.append(pool(run=run, raw=raw))
    return pl.concat(pieces)


def fit_b2(*, row: dict, context: Context) -> pl.DataFrame:
    """Return B2: a regression of the BSP on its children's medians, plus residual quantiles."""
    primaries, generators = context.child_ids(row=row)
    names = child_columns(ids=[*primaries, *generators])
    capacity = row["effective_capacity_mw"]
    rows = base_inputs(row=row, arrays=context.arrays[row["site"]], frame=context.train)
    rows = rows.join(context.oof.select(*RUN_KEYS, *names), on=list(RUN_KEYS), how="left")
    rows = rows.filter(pl.col("power").is_finite() & pl.all_horizontal(pl.col(names).is_not_null()))
    design = np.column_stack([np.ones(rows.height), rows.select(names).to_numpy()])
    coefficients, *_ = np.linalg.lstsq(
        design, rows["power"].to_numpy().astype(np.float64), rcond=None
    )
    residual = rows["power"].to_numpy() - design @ coefficients
    bands = np.array(
        [
            next(i for i, (_, low, high) in enumerate(common.LEAD_BANDS) if low <= lead < high)
            for lead in rows["lead_hours"].to_numpy()
        ]
    )
    offsets = np.stack(
        [np.quantile(residual[bands == b], common.LEVELS) for b in range(len(common.LEAD_BANDS))]
    )
    pieces = []
    for path in context.val_files:
        run = validation_run(path=path, row=row, context=context)
        run = run.join(
            context.val_children.select(*MEMBER_KEYS, *names), on=list(MEMBER_KEYS), how="left"
        )
        value = coefficients[0] + run.select(names).to_numpy() @ coefficients[1:]
        band = np.array(
            [
                next(i for i, (_, low, high) in enumerate(common.LEAD_BANDS) if low <= lead < high)
                for lead in run["lead_hours"].to_numpy()
            ]
        )
        raw = (value[:, None] + offsets[band]) / capacity
        pieces.append(pool(run=run, raw=raw))
    return pl.concat(pieces)


ARMS_FOR: Final[dict[str, tuple[str, ...]]] = {
    "BSP": ("b0", "b1", "o", "b2", "b3"),
    "GSP": ("b0", "b1", "b3", "b4"),
}


def main() -> None:
    """Fit every arm for every seed and write its tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="one seed, scratch folder")
    parser.add_argument("--only", nargs="+", help="fit only these arms")
    args = parser.parse_args()
    out = ARMS_DIR.with_name("arms_smoke") if args.smoke else ARMS_DIR
    out.mkdir(parents=True, exist_ok=True)
    for seed in common.SEEDS[:1] if args.smoke else common.SEEDS:
        context = Context(seed=seed, smoke=args.smoke)
        targets = context.series.filter(pl.col("substation_type").is_in(["BSP", "GSP"]))
        pieces: dict[str, list[pl.DataFrame]] = {}
        for row in targets.iter_rows(named=True):
            for arm in ARMS_FOR[row["substation_type"]]:
                if args.only and arm not in args.only:
                    continue
                if (out / f"{arm}_seed{seed}.parquet").exists():
                    continue
                started = time.monotonic()
                if arm == "b2":
                    table = fit_b2(row=row, context=context)
                else:
                    training = base_inputs(
                        row=row, arrays=context.arrays[row["site"]], frame=context.train
                    )
                    training, extras = with_extras(
                        rows=training, arm=arm, row=row, context=context, training=True
                    )
                    booster = fit_arm(
                        arm=arm, row=row, context=context, rows=training, extras=extras
                    )
                    table = predict_booster(
                        arm=arm, booster=booster, row=row, context=context, extras=extras
                    )
                pieces.setdefault(arm, []).append(
                    table.with_columns(time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32))
                )
                print(
                    f"{arm} seed {seed} {row['site']}: {time.monotonic() - started:.0f} s",
                    flush=True,
                )
        for arm, tables in pieces.items():
            frame = pl.concat(tables).with_columns(seed=pl.lit(seed).cast(pl.Int8))
            target = out / f"{arm}_seed{seed}.parquet"
            frame.write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
        for (series_id, split), (filled, total) in sorted(context.fill_counts.items()):
            print(
                f"oracle fill share, series {series_id} {split}: {filled / total:.3f}", flush=True
            )


if __name__ == "__main__":
    main()
