"""Print the numbers of experiment 00064's write-up that `e00064_stats.py` does not print.

Reads the saved tables under `work/` and experiment 00045's capacity table (read only), and prints:

- the weights table (the slope of the out-of-fold residual on each shift) by variant and lead
  band, as the median over the 5 series and 2 seasons, and the ranges the write-up quotes;
- the spread of the applied slope in the 10+ day band, which is why the weight there is zero;
- the correlation of the ensemble-mean calibrated anomaly with the CAMS anomaly by lead band on
  the training rows, and of the placebo twin's anomaly with the real one on the validation rows;
- the share of arm `b`'s training and validation rows by the source of each lag column's shape,
  and the share of validation rows with no shape (a NaN gross lag at inference);
- the capacities used, their sources, experiment 00045's recovery ratio and experiment 00040's
  constant, and the 2025/26 constant of experiment 00045 against the capacity used.

    uv run python ml-experiments/00064-*/e00064_report.py
"""

from datetime import date
from typing import Final

import e00064_capacity as capacity_script
import e00064_common as common
import polars as pl

FIRST_FOUR_BANDS: Final[int] = 4
"""Bands 0 to 3 (under 10 days), where the fitted weight is used."""
LAG_NAMES: Final[tuple[str, ...]] = (
    "lag_origin",
    "lag_same_time_latest_day",
    "lag_same_time_latest_week",
)
VARIANT_ORDER: Final[tuple[str, ...]] = ("all", "no_lags", "no_cloud_proxies", "neither")
LATER_SEASON_YEAR: Final[int] = 2025
"""The season year (September to August) that starts in 2025."""


def _config() -> pl.Config:
    return pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200)


def print_slope_diagnostic() -> None:
    """Print the slope diagnostic and the ranges of the fitted weights."""
    weights = pl.read_parquet(common.WEIGHTS_FILE)
    bands = [name for name, _low, _high in common.LEAD_BANDS]
    by_variant = (
        weights.group_by("variant", "band")
        .agg(
            median_slope_oracle=pl.col("slope_oracle").median(),
            median_slope_applied=pl.col("slope_applied").median(),
            cells=pl.len(),
        )
        .with_columns(
            variant=pl.col("variant").cast(pl.Enum(list(VARIANT_ORDER))),
            lead_band=pl.col("band").replace_strict(dict(enumerate(bands))),
        )
        .sort("variant", "band")
    )
    print("Slope diagnostic: median over 5 series and 2 seasons, by variant and lead band")
    print(by_variant)
    print("\nRange of the median oracle slope over bands 0 to 3, by variant")
    print(
        by_variant.filter(pl.col("band") < FIRST_FOUR_BANDS)
        .group_by("variant", maintain_order=True)
        .agg(
            lowest=pl.col("median_slope_oracle").min(),
            highest=pl.col("median_slope_oracle").max(),
        )
    )
    mean_over_series = (
        weights.filter(pl.col("variant") == "all")
        .group_by("band", "season")
        .agg(
            mean_slope_oracle=pl.col("slope_oracle").mean(),
            mean_slope_applied=pl.col("slope_applied").mean(),
        )
        .sort("band", "season")
    )
    first_four = mean_over_series.filter(pl.col("band") < FIRST_FOUR_BANDS)
    print("\nMean over series, variant 'all', bands 0 to 3: range over the 8 cells")
    print(
        first_four.select(
            oracle_lowest=pl.col("mean_slope_oracle").min(),
            oracle_highest=pl.col("mean_slope_oracle").max(),
            applied_lowest=pl.col("mean_slope_applied").min(),
            applied_highest=pl.col("mean_slope_applied").max(),
        )
    )
    last = weights.filter((pl.col("variant") == "all") & (pl.col("band") == FIRST_FOUR_BANDS))
    print(
        "\nApplied slope in the 10+ day band, variant 'all', over 5 series and 2 seasons: "
        f"{last['slope_applied'].min():.2f} to {last['slope_applied'].max():.2f}"
    )


def print_anomaly_correlations() -> None:
    """Print the ENS anomaly's correlation with CAMS's, and the placebo twin's with the ENS's."""
    train = pl.read_parquet(common.INPUTS_DIR / "train_shape.parquet").drop_nulls(
        ["anomaly_mean_calibrated", "anomaly_cams"]
    )
    bands = [name for name, _low, _high in common.LEAD_BANDS]
    print("\nTraining rows: correlation of the ensemble-mean calibrated anomaly with CAMS's")
    print(
        train.group_by("band")
        .agg(
            correlation=pl.corr("anomaly_mean_calibrated", "anomaly_cams"),
            rows=pl.len(),
        )
        .with_columns(lead_band=pl.col("band").replace_strict(dict(enumerate(bands))))
        .sort("band")
    )
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    validation = (
        pl.read_parquet(common.INPUTS_DIR / "val_keys.parquet")
        .select(*keys, "anomaly_mean_calibrated")
        .join(
            pl.read_parquet(common.INPUTS_DIR / "placebo_val.parquet"),
            on=keys,
            how="inner",
        )
        .drop_nulls()
    )
    correlation = validation.select(
        pl.corr("anomaly_mean_calibrated", "anomaly_mean_calibrated_twin")
    ).item()
    print(
        "Validation rows: correlation of the placebo twin's anomaly with the real anomaly: "
        f"{correlation:.3f} on {validation.height:,} rows"
    )


def print_lag_sources() -> None:
    """Print the share of rows by the source of each lag column's shape, for arm `b`."""
    for split in ("train", "val"):
        table = pl.read_parquet(common.INPUTS_DIR / f"lag_shapes_{split}.parquet")
        rows = table.height
        shares = pl.concat(
            [
                table.group_by(source=pl.col(f"source_{name}"))
                .agg(share=pl.len() / rows)
                .with_columns(lag=pl.lit(name))
                for name in LAG_NAMES
            ]
        ).pivot(on="source", index="lag", values="share")
        print(f"\nShare of {split} rows ({rows:,}) by the source of each lag column's shape")
        print(shares.fill_null(0.0))
    validation = pl.read_parquet(common.INPUTS_DIR / "lag_shapes_val.parquet")
    missing = {
        name: validation[f"shape_{name}"].null_count() / validation.height for name in LAG_NAMES
    }
    print("\nValidation rows with no lag shape (gross lag NaN at inference):")
    for name, share in missing.items():
        print(f"  {name}: {100 * share:.1f}%")


def print_capacities() -> None:
    """Print the capacities used, their sources, and experiment 00045's and 00040's values."""
    used = pl.read_parquet(common.CAPACITY_FILE)
    final = pl.read_parquet(common.E45_DIR / "capacity_weekly.parquet").filter(
        pl.col("variant") == capacity_script.FIT_NAME
    )
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    e40 = pl.read_parquet(common.E40_CAPACITY_PATH).filter(
        pl.col("year") == common.E40_CAPACITY_SEASON_YEAR
    )
    e40_c = dict(zip(e40["site"], e40["C"], strict=True))
    season_start = date(LATER_SEASON_YEAR, 9, 1)
    season_end = date(LATER_SEASON_YEAR + 1, 9, 1)
    rows = []
    for site in common.SITES:
        own = final.filter(pl.col("time_series_id") == ids[site])
        later = own.filter(
            (pl.col("week_start") >= season_start) & (pl.col("week_start") < season_end)
        )["capacity_mw"]
        capacity = used.filter(pl.col("site") == site)["capacity_mw"].item()
        later_mean = float(later.to_numpy().mean()) if later.len() else None
        rows.append(
            {
                "site": site,
                "capacity_used_mw": capacity,
                "source": used.filter(pl.col("site") == site)["source"].item(),
                "recovery_ratio": float(own["recovery_ratio"][0]),
                "e40_C_mw": float(e40_c[site]),
                "e45_2025_26_mw": later_mean,
                "e45_2025_26_over_used": None if later_mean is None else later_mean / capacity,
            }
        )
    print("\nCapacities (MW) used, with experiment 00045's recovery ratio and 2025/26 constant")
    print(pl.DataFrame(rows))


def main() -> None:
    """Print every extra number."""
    with _config():
        print_slope_diagnostic()
        print_anomaly_correlations()
        print_lag_sources()
        print_capacities()


if __name__ == "__main__":
    main()
