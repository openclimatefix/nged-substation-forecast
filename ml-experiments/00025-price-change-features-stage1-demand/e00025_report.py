r"""Score experiment 25: absolute errors and the planned contrasts, by substation type and series.

Reads the prediction files `e00025_fit.py` wrote and prints and writes tables under the
experiment's private data folder. Errors are mean absolute error of the median in points of
capacity (1% of the series' effective capacity). The verdict contrasts C1 and C2 use 97.5%
month-resampled intervals (95% familywise across the two). The climatology row is the
month-by-half-hour median of the fold's training window.

    uv run python ml-experiments/00025-price-change-features-stage1-demand/e00025_report.py \
        --fold select_2024
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import e00025_fit as fit
import e00025_inputs as inputs
import polars as pl
from baseline_forecasters.climatology import (
    CLIMATOLOGY_QUANTILE_COLUMNS,
    _local_calendar_cell_keys,
    _pooled_quantile_lookup,
)
from studies.bootstrap import bootstrap_absolute, bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap

RESULTS_DIR: Final[Path] = inputs.EXPERIMENT_DIR / "results"
ARMS: Final[tuple[str, ...]] = (
    "BASE",
    "PRICE_LEVEL",
    "PRICE_CHANGE",
    "PRICE_SHAPE",
    "SHUFFLED_CHANGE",
    "SHUFFLED_SHAPE",
    "NATIONAL",
    "SYSTEM",
    "CLIMATOLOGY",
)
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("PRICE_CHANGE", "PRICE_LEVEL"),
    ("PRICE_CHANGE", "SHUFFLED_CHANGE"),
    ("PRICE_CHANGE", "NATIONAL"),
    ("PRICE_SHAPE", "PRICE_CHANGE"),
    ("PRICE_SHAPE", "SHUFFLED_SHAPE"),
    ("PRICE_LEVEL", "BASE"),
    ("SYSTEM", "PRICE_LEVEL"),
)
"""The first two are the verdict contrasts C1 and C2, scored on Primary substations only."""
TOP_PRICE_CHANGE_SHARE: Final[float] = 0.05
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


def price_change_days(*, rows: pl.DataFrame) -> pl.Series:
    """Return the local dates whose largest absolute half-hourly price change is in the top 5%."""
    series = pl.read_parquet(inputs.SERIES_PATH, columns=["start", "price_d1"])
    daily = (
        series.with_columns(
            date=(pl.col("start") + inputs.HALF_HOUR)
            .dt.convert_time_zone("Europe/London")
            .dt.date()
        )
        .group_by("date")
        .agg(biggest=pl.col("price_d1").abs().max())
        .drop_nulls()
    )
    cutoff = daily["biggest"].quantile(1.0 - TOP_PRICE_CHANGE_SHARE)
    return daily.filter(pl.col("biggest") >= cutoff)["date"]


def missing_shares(*, fold: str) -> pl.DataFrame:
    """Return the share of validation rows with each added column missing, real and shuffled."""
    window = fit.FOLDS[fold]
    frame = (
        pl.read_parquet(inputs.FEATURES_PATH)
        .filter(pl.col("time_series_id") == 3)
        .filter(pl.col("valid_time") >= window.valid_start, pl.col("valid_time") < window.valid_end)
    )
    frame = fit.add_shuffled_columns(frame=frame, seed=0, train_end=window.train_end)
    columns = [c for c in frame.columns if c in inputs.DERIVED_COLUMNS or c.startswith("shuf_")]
    return pl.DataFrame(
        {
            "column": columns,
            "missing_share": [frame[c].null_count() / frame.height for c in columns],
        }
    )


def main() -> None:
    """Print and write the levels and the contrasts for a fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True)
    fold = parser.parse_args().fold
    rows = add_climatology(fold=fold, rows=read_fits(fold=fold))
    rows = rows.with_columns(
        date=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    )
    print(f"fitted series: {rows['site'].n_unique()}; seeds: {sorted(rows['seed'].unique())}")
    scopes: dict[str, pl.DataFrame] = {
        "Primary": rows.filter(pl.col("substation_type") == "Primary"),
        **{
            site: rows.filter(pl.col("site") == site)
            for site in sorted(rows.filter(pl.col("substation_type") != "Primary")["site"].unique())
        },
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
        for treatment, reference in CONTRASTS:
            contrasts.append(
                contrast(rows=scope, treatment=treatment, reference=reference, label=label)
            )
    primary = scopes["Primary"]
    subsets = {
        f"C7 hours {start}-{end}": primary.filter(pl.col("hour") >= start, pl.col("hour") < end)
        for start, end in HOUR_BLOCKS
    }
    subsets["C7 top 5% price-change days"] = primary.filter(
        pl.col("date").is_in(price_change_days(rows=rows).implode())
    )
    for label, subset in subsets.items():
        for treatment, reference in CONTRASTS[:2]:
            contrasts.append(
                contrast(rows=subset, treatment=treatment, reference=reference, label=label)
            )
    monthly = (
        loss_frame(rows=primary, arms=("PRICE_LEVEL", "PRICE_CHANGE", "SHUFFLED_CHANGE"))
        .with_columns(month=pl.col("valid_time").dt.truncate("1mo"))
        .group_by("month", "arm")
        .agg(pl.col("abs_error").mean())
        .pivot(on="arm", index="month", values="abs_error")
        .sort("month")
        .select(
            "month",
            c1=pl.col("PRICE_CHANGE") - pl.col("PRICE_LEVEL"),
            c2=pl.col("PRICE_CHANGE") - pl.col("SHUFFLED_CHANGE"),
        )
    )
    print(f"months with C1 < 0: {(monthly['c1'] < 0).sum()} of {monthly.height}")
    print(f"months with C2 < 0: {(monthly['c2'] < 0).sum()} of {monthly.height}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with pl.Config(tbl_rows=200, tbl_width_chars=200, float_precision=3):
        for name, table in (
            ("levels", pl.DataFrame(levels)),
            ("contrasts", pl.DataFrame(contrasts)),
            ("monthly", monthly),
            ("missing_shares", missing_shares(fold=fold)),
        ):
            table.write_parquet(RESULTS_DIR / f"{fold}_{name}.parquet")
            print(f"\n## {fold} {name}\n{table}", file=sys.stdout)


if __name__ == "__main__":
    main()
