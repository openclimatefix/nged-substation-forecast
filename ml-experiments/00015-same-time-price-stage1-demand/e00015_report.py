r"""Score experiment 15: absolute errors, the planned contrasts, and the residual breakdowns.

Reads the prediction files `e00015_fit.py` wrote and prints and writes tables under the
experiment's private data folder. Errors are mean absolute error of the median in points of
capacity (1% of the series' effective capacity). The verdict contrasts C1 and C2 use 97.5%
month-resampled intervals (95% familywise across the two). The climatology row is the
month-by-half-hour median of the fold's training window.

    uv run python ml-experiments/00015-same-time-price-stage1-demand/e00015_report.py \
        --fold select_2024
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import e00015_fit as fit
import e00015_inputs as inputs
import polars as pl
from baseline_forecasters.climatology import (
    CLIMATOLOGY_QUANTILE_COLUMNS,
    _local_calendar_cell_keys,
    _pooled_quantile_lookup,
)
from studies.bootstrap import bootstrap_absolute, bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap

RESULTS_DIR: Final[Path] = inputs.EXPERIMENT_DIR / "results"
ARMS: Final[tuple[str, ...]] = ("BASE", "PRICE", "CONTROL", "SHUFFLED", "CLIMATOLOGY")
LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 2)
N_RESAMPLES: Final[int] = 20_000
SERIES_PATH: Final[Path] = inputs.EXPERIMENT_00006_DIR / "series.parquet"
MEDIAN_COLUMN: Final[str] = CLIMATOLOGY_QUANTILE_COLUMNS[25]
"""The climatology's equiprobable quantile at level 25.5 / 51 = 0.5."""
HOUR_BLOCKS: Final[tuple[tuple[int, int], ...]] = (
    (0, 4),
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
)


def read_fits(*, fold: str) -> pl.DataFrame:
    """Return every unit's predictions for a fold, with the seed, series label, and local hour."""
    frames = [
        pl.read_parquet(path).with_columns(seed=pl.lit(int(path.parent.name.removeprefix("seed"))))
        for path in sorted((inputs.EXPERIMENT_DIR / "fits" / fold).glob("seed*/*.parquet"))
    ]
    series = pl.read_parquet(SERIES_PATH).select("time_series_id", "site", "substation_type")
    return (
        pl.concat(frames)
        .join(series, on="time_series_id")
        .with_columns(hour=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour())
    )


def add_climatology(*, fold: str, rows: pl.DataFrame) -> pl.DataFrame:
    """Add a `CLIMATOLOGY` column: the month-by-half-hour median of the fold's training window.

    The column repeats once per seed, because the climatology has none. A row whose cell holds no
    training sample gets null and drops out of the climatology's score.
    """
    window = fit.FOLDS[fold]
    features = pl.read_parquet(
        inputs.EXPERIMENT_00006_DIR / "features.parquet",
        columns=["time_series_id", "valid_time", "power_mw", "effective_capacity_mw"],
    )
    training = features.filter(
        pl.col("power_mw").is_not_null(),
        pl.col("valid_time") >= fit.TRAIN_START,
        pl.col("valid_time") < window.train_end,
    ).select("time_series_id", "valid_time", power=pl.col("power_mw"))
    lookup = _pooled_quantile_lookup(training).select(
        "time_series_id",
        "local_month",
        "local_half_hour_of_day",
        "local_is_weekend",
        median_mw=pl.col(MEDIAN_COLUMN),
    )
    capacity = features.group_by("time_series_id").agg(c=pl.col("effective_capacity_mw").first())
    return (
        rows.with_columns(
            **_local_calendar_cell_keys(pl.col("valid_time"), time_zone="Europe/London")
        )
        .join(
            lookup,
            on=["time_series_id", "local_month", "local_half_hour_of_day", "local_is_weekend"],
            how="left",
        )
        .join(capacity, on="time_series_id")
        .with_columns(CLIMATOLOGY=pl.col("median_mw") / pl.col("c"))
        .drop("median_mw", "c", "local_month", "local_half_hour_of_day", "local_is_weekend")
    )


def loss_frame(*, rows: pl.DataFrame, arms: tuple[str, ...]) -> pl.DataFrame:
    """Return the bootstrap loss frame with `abs_error` in points of capacity."""
    keyed = rows.with_columns(power_fcst_init_time=pl.col("valid_time").dt.truncate("1mo"))
    return losses_for_bootstrap(
        arm_losses={
            arm: keyed.select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "seed",
                abs_error=(pl.col(arm) - pl.col("target")).abs() * 100.0,
            ).drop_nulls("abs_error")
            for arm in arms
        }
    )


def contrast(*, rows: pl.DataFrame, treatment: str, reference: str, label: str) -> dict:
    """Return one paired difference (treatment minus reference) with its 97.5% interval."""
    frame = loss_frame(rows=rows, arms=(treatment, reference))
    low, high = bootstrap_difference_at_level(
        losses=frame,
        treatment=treatment,
        reference=reference,
        metric="abs_error",
        level=LEVEL,
        n_resamples=N_RESAMPLES,
    )
    means = frame.group_by("arm").agg(pl.col("abs_error").mean())
    value = dict(zip(means["arm"], means["abs_error"], strict=True))
    return {
        "scope": label,
        "contrast": f"{treatment} - {reference}",
        "difference": value[treatment] - value[reference],
        "lower": low,
        "upper": high,
    }


def residual_breakdown(*, rows: pl.DataFrame, by: str) -> pl.DataFrame:
    """Return BASE and PRICE error, the PRICE gain, and the squared residual PRICE removes."""
    return (
        rows.with_columns(
            base_error=(pl.col("BASE") - pl.col("target")) * 100.0,
            price_error=(pl.col("PRICE") - pl.col("target")) * 100.0,
        )
        .group_by(by)
        .agg(
            base_mae=pl.col("base_error").abs().mean(),
            price_mae=pl.col("price_error").abs().mean(),
            base_sse=(pl.col("base_error") ** 2).sum(),
            price_sse=(pl.col("price_error") ** 2).sum(),
        )
        .with_columns(
            gain=pl.col("price_mae") - pl.col("base_mae"),
            squared_residual_removed=1.0 - pl.col("price_sse") / pl.col("base_sse"),
        )
        .drop("base_sse", "price_sse")
        .sort(by)
    )


def main() -> None:
    """Print and write the levels, the contrasts, and the breakdowns for a fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True)
    fold = parser.parse_args().fold
    rows = add_climatology(fold=fold, rows=read_fits(fold=fold))
    print(f"fitted series: {rows['site'].n_unique()}; seeds: {sorted(rows['seed'].unique())}")
    scopes: dict[str, pl.DataFrame] = {
        "pooled": rows,
        "B2": rows.filter(pl.col("site") == "B2"),
        "Primary": rows.filter(pl.col("substation_type") == "Primary"),
    }
    levels, contrasts = [], []
    for label, scope in scopes.items():
        if scope.is_empty():
            print(f"{label}: no fitted series in {fold}, skipped")
            continue
        frame = loss_frame(rows=scope, arms=ARMS)
        for arm in ARMS:
            result = bootstrap_absolute(losses=frame, arm=arm, metric="abs_error")
            levels.append(
                {
                    "scope": label,
                    "arm": arm,
                    "value": result["value"],
                    "lower": result["lower_95"],
                    "upper": result["upper_95"],
                }
            )
        for treatment, reference in (
            ("PRICE", "BASE"),
            ("PRICE", "SHUFFLED"),
            ("PRICE", "CONTROL"),
        ):
            contrasts.append(
                contrast(rows=scope, treatment=treatment, reference=reference, label=label)
            )
    for start, end in HOUR_BLOCKS:
        block = rows.filter(pl.col("hour") >= start, pl.col("hour") < end)
        contrasts.append(
            contrast(
                rows=block, treatment="PRICE", reference="BASE", label=f"C4 hours {start}-{end}"
            )
        )
    monthly = (
        loss_frame(rows=rows, arms=("BASE", "PRICE", "SHUFFLED"))
        .with_columns(month=pl.col("valid_time").dt.truncate("1mo"))
        .group_by("month", "arm")
        .agg(pl.col("abs_error").mean())
        .pivot(on="arm", index="month", values="abs_error")
        .sort("month")
        .select(
            "month",
            c1=pl.col("PRICE") - pl.col("BASE"),
            c2=pl.col("PRICE") - pl.col("SHUFFLED"),
        )
    )
    print(f"months with C1 < 0: {(monthly['c1'] < 0).sum()} of {monthly.height}")
    print(f"months with C2 < 0: {(monthly['c2'] < 0).sum()} of {monthly.height}")
    noise_floor = residual_breakdown(rows=rows.with_columns(BASE=pl.col("SHUFFLED")), by="hour")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with pl.Config(tbl_rows=100, tbl_width_chars=200, float_precision=3):
        for name, table in (
            ("levels", pl.DataFrame(levels)),
            ("contrasts", pl.DataFrame(contrasts)),
            ("monthly", monthly),
            ("by_hour_noise_floor", noise_floor),
            (
                "by_series_noise_floor",
                residual_breakdown(rows=rows.with_columns(BASE=pl.col("SHUFFLED")), by="site"),
            ),
            ("by_hour", residual_breakdown(rows=rows, by="hour")),
            ("by_series", residual_breakdown(rows=rows, by="site")),
            ("b2_by_hour", residual_breakdown(rows=scopes["B2"], by="hour")),
        ):
            table.write_parquet(RESULTS_DIR / f"{fold}_{name}.parquet")
            print(f"\n## {fold} {name}\n{table}", file=sys.stdout)


if __name__ == "__main__":
    main()
