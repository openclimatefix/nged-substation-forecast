"""Fit the weight `w` of the solar term for each series, lead band, and season.

For each series, the training window is cross-fitted in 4 time-block folds: the
fold of a row is the valid time's UTC month index modulo 4, so each fold holds every fourth month.
Seed 0 only. Each fold's model is fitted on the other three folds with experiment 00033's N1
parameters and recency weights, and predicts the held-out rows. The out-of-fold residual of the
median is the actual minus the median, in power over effective capacity.

On the daylight rows (a CAMS climatology cell above 0.02) of April to September with a CAMS
anomaly, the residual is regressed through the origin, per lead band and season (April to June,
July to September), on two shifts:

- the oracle shift, `scale * anomaly_cams` (experiment 00051's slope, expected negative);
- the applied shift, `scale * anomaly_mean_calibrated`, the shift `am` uses at forecast
  time. Only this slope feeds `am`.

`scale` is the capacity over the series' effective capacity. A sum of products and a sum of
squares is stored for each shift so that pooled slopes can be formed.

    uv run python ml-experiments/00077-*/e00077_weights.py --job D9 all     # one cross-fit
    uv run python ml-experiments/00077-*/e00077_weights.py --collect        # after all 5 jobs

Each job writes `work/weights_parts/<site>_<variant>.parquet`; `--collect` writes
`work/weights.parquet` (`site`, `variant`, `band`, `season`, `slope_oracle`, `slope_applied`, and
the row count) and prints the slopes.
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00077_common as common
import e00077_model as model
import numpy as np
import polars as pl
import xgboost as xgb

PARTS_DIR: Final[Path] = common.WORK_DIR / "weights_parts"
N_FOLDS: Final[int] = 4
WEIGHT_SEED: Final[int] = 0


def fold_of(*, valid_time: pl.Series) -> np.ndarray:
    """Return each row's fold: the UTC month index of its valid time modulo `N_FOLDS`."""
    index = valid_time.dt.year().cast(pl.Int64) * 12 + valid_time.dt.month().cast(pl.Int64) - 1
    return (index % N_FOLDS).to_numpy()


def out_of_fold_median(
    *, rows: pl.DataFrame, ctx: model.SeriesContext, columns: list[str], n_rounds: int, device: str
) -> np.ndarray:
    """Return the held-out median forecast of every training row from the 4-fold cross-fit.

    Args:
        rows: `model.training_rows()`.
        ctx: The series' context.
        columns: The input columns of the variant.
        n_rounds: Boosting rounds per fit.
        device: `cuda` or `cpu`.

    Returns:
        The raw median (q50) for each row, in power over effective capacity.
    """
    features = rows.select(columns).to_numpy().astype(np.float32)
    label = (rows["power"] / ctx.row["effective_capacity_mw"]).to_numpy()
    age_days = (common.LAST_TRAINING_VALID - rows["valid_time"]).dt.total_seconds() / 86_400
    recency = 0.5 ** (age_days.to_numpy() / common.RECENCY_HALF_LIFE_DAYS)
    fold = fold_of(valid_time=rows["valid_time"])
    median = np.full(rows.height, np.nan)
    for held_out in range(N_FOLDS):
        train = fold != held_out
        booster = model.fit_booster(
            matrix=xgb.DMatrix(features[train], label=label[train], weight=recency[train]),
            seed=WEIGHT_SEED,
            n_rounds=n_rounds,
            device=device,
        )
        median[~train] = booster.predict(xgb.DMatrix(features[~train]))[:, common.MEDIAN]
    return median


def slope_table(
    *, rows: pl.DataFrame, median: np.ndarray, ctx: model.SeriesContext
) -> pl.DataFrame:
    """Return the sums behind the two slopes, per lead band and season.

    Args:
        rows: `model.training_rows()`.
        median: `out_of_fold_median()`.
        ctx: The series' context.

    Returns:
        Columns `band`, `season`, `sum_rs_oracle`, `sum_ss_oracle`, `sum_rs_applied`,
        `sum_ss_applied`, and `n_rows`.
    """
    frame = rows.with_columns(
        residual=rows["power"] / ctx.row["effective_capacity_mw"] - pl.Series(median),
        season=common.season_of_month_expr(month=pl.col("month")),
        oracle=ctx.scale * pl.col("anomaly_cams"),
        applied=ctx.scale * pl.col(model.ANOMALY_COLUMN),
    ).filter(
        pl.col("daylight")
        & pl.col("season").is_not_null()
        & pl.col("oracle").is_not_null()
        & pl.col("residual").is_finite()
    )
    return (
        frame.group_by("band", "season")
        .agg(
            sum_rs_oracle=(pl.col("residual") * pl.col("oracle")).sum(),
            sum_ss_oracle=(pl.col("oracle") ** 2).sum(),
            sum_rs_applied=(pl.col("residual") * pl.col("applied")).sum(),
            sum_ss_applied=(pl.col("applied") ** 2).sum(),
            n_rows=pl.len(),
        )
        .sort("band", "season")
    )


def run_job(
    *, site: str, variant: str, n_rounds: int, device: str, ctx: model.SeriesContext | None = None
) -> pl.DataFrame:
    """Cross-fit one series and variant and return its table of sums.

    Args:
        site: The series label.
        variant: A key of `model.WEIGHT_VARIANTS`.
        n_rounds: Boosting rounds per fit.
        device: `cuda` or `cpu`.
        ctx: The series' context when already loaded.

    Returns:
        `slope_table()` with `site` and `variant` columns.
    """
    ctx = ctx or model.load_context(site=site)
    rows = model.training_rows(ctx=ctx)
    columns = model.d1l_columns(drop=model.WEIGHT_VARIANTS[variant])
    median = out_of_fold_median(
        rows=rows, ctx=ctx, columns=columns, n_rounds=n_rounds, device=device
    )
    return slope_table(rows=rows, median=median, ctx=ctx).with_columns(
        site=pl.lit(site), variant=pl.lit(variant)
    )


def to_slopes(*, sums: pl.DataFrame) -> pl.DataFrame:
    """Turn summed products and squares into slopes through the origin (zero if no variance)."""
    return sums.with_columns(
        slope_oracle=pl.when(pl.col("sum_ss_oracle") > 0)
        .then(pl.col("sum_rs_oracle") / pl.col("sum_ss_oracle"))
        .otherwise(0.0),
        slope_applied=pl.when(pl.col("sum_ss_applied") > 0)
        .then(pl.col("sum_rs_applied") / pl.col("sum_ss_applied"))
        .otherwise(0.0),
    )


def collect() -> None:
    """Join the 5 parts, print the slopes, and write `work/weights.parquet`."""
    parts = sorted(PARTS_DIR.glob("*.parquet"))
    sums = pl.concat([pl.read_parquet(path) for path in parts])
    expected = {(s, v) for s in common.SITES for v in model.WEIGHT_VARIANTS}
    found = set(zip(sums["site"], sums["variant"], strict=True))
    if found != expected:
        msg = f"missing weight jobs: {sorted(expected - found)}"
        raise FileNotFoundError(msg)
    table = to_slopes(sums=sums).select(
        "site", "variant", "band", "season", "slope_oracle", "slope_applied", "n_rows"
    )
    common.write_atomically(frame=table, target=common.WEIGHTS_FILE)
    pooled = to_slopes(
        sums=sums.group_by("variant", "band").agg(
            pl.col("sum_rs_oracle").sum(),
            pl.col("sum_ss_oracle").sum(),
            pl.col("sum_rs_applied").sum(),
            pl.col("sum_ss_applied").sum(),
            n_rows=pl.col("n_rows").sum(),
        )
    ).sort("variant", "band")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        print("Pooled slope of the out-of-fold residual by variant and lead band")
        print(pooled.select("variant", "band", "slope_oracle", "slope_applied", "n_rows"))
        print("Weights of the 'all inputs' variant (fed to sf and am)")
        print(table.filter(pl.col("variant") == "all").drop("variant"))
    print(f"written {common.WEIGHTS_FILE}")


def main() -> None:
    """Run one cross-fit job, or collect the finished jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", nargs=2, metavar=("SITE", "VARIANT"))
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--rounds", type=int, default=common.N_ROUNDS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.collect:
        collect()
        return
    if args.job is None:
        parser.error("give --job SITE VARIANT or --collect")
    site, variant = args.job
    target = PARTS_DIR / f"{site}_{variant}.parquet"
    if target.exists():
        return
    started = time.monotonic()
    table = run_job(site=site, variant=variant, n_rounds=args.rounds, device=args.device)
    common.write_atomically(frame=table, target=target)
    print(f"weights {site} {variant}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
