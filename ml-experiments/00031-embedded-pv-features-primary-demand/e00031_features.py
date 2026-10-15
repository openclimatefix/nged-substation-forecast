"""Build experiment 00031's capacity features for every series and forecast origin.

For each series, the origins are every distinct `power_fcst_init_time` of the ENS training frame
and of the validation runs. The table holds F1 to F3, the register capacity `e`, and the capacity
estimates `c` (data only, and with the register as a prior for a Primary or a floor for a BSP or
GSP). Fitted constants (the prior variance) use origins before the validation start only. The
script then runs the known-answer check of the plan: it adds a synthetic PV signal to two
directional Primary series and reports how well F1, F2, and F4 recover it.

Output: `work/features/origin_features.parquet`, `work/features/known_answer.parquet`, and the
register's per-row decisions in `work/ecr_rows.csv`.

    uv run python ml-experiments/00031-embedded-pv-features-primary-demand/e00031_features.py
"""

import time
from typing import Final

import e00031_common as common
import e00031_ecr as ecr
import e00031_pv as pv
import numpy as np
import polars as pl

FEATURES_DIR: Final = common.WORK_DIR / "features"
KNOWN_ANSWER_SITES: Final[tuple[str, ...]] = ("D5", "D10")
"""Two directional Primary series."""
KNOWN_ANSWER_FINAL_SHARE: Final[float] = 0.20
MIN_PRIOR_VARIANCE: Final[float] = 1e-4
"""Floor on the prior variance, in squared shares of effective capacity."""


def origin_times() -> pl.Series:
    """Return the distinct forecast origins of the training frame and the validation runs."""
    train = pl.read_parquet(common.ENS_TRAIN_PATH, columns=["power_fcst_init_time"]).unique()
    validation = pl.concat(
        [
            pl.read_parquet(path, columns=["power_fcst_init_time"]).unique()
            for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
        ]
    ).unique()
    return (
        pl.concat([train, validation]).unique().sort("power_fcst_init_time")["power_fcst_init_time"]
    )


def capacity_estimates(*, features: pl.DataFrame, register: pl.DataFrame) -> pl.DataFrame:
    """Add `register_share`, `cap_data_only`, and `cap_with_register` to the origin features.

    A Primary's estimate pulls the data estimate towards the register capacity by inverse-variance
    weights, with one prior variance for all Primaries fitted on training origins. A BSP's or GSP's
    estimate is the larger of the data estimate and the register capacity. Both are clipped at zero.

    Args:
        features: Per series and origin, from `origin_features`, with `site` and `role`.
        register: `ecr.ecr_table` output.

    Returns:
        The features with the three added columns, and `unregistered_cap` (`c - e`).
    """
    joined = features.join(
        register.select("site", register_share="ecr_share"), on="site", validate="m:1"
    )
    training = joined.filter(
        (pl.col("role") == "prior")
        & (pl.col("power_fcst_init_time") < common.VALIDATION_START)
        & pl.col("cap_slope_28d").is_not_null()
        & pl.col("cap_slope_28d").is_not_nan()
    )
    prior_variance = max(
        float(
            (
                (training["cap_slope_28d"] - training["register_share"]) ** 2
                - training["cap_slope_se"] ** 2
            ).mean()
        ),
        MIN_PRIOR_VARIANCE,
    )
    print(f"prior variance (shares of effective capacity squared): {prior_variance:.5f}")
    weight = prior_variance / (prior_variance + pl.col("cap_slope_se") ** 2)
    estimate = pl.col("cap_slope_28d")
    register_share = pl.col("register_share")
    return joined.with_columns(
        cap_data_only=pl.when(estimate.is_nan())
        .then(None)
        .otherwise(estimate.clip(lower_bound=0.0)),
        cap_with_register=pl.when(pl.col("role") == "prior")
        .then(
            pl.when(estimate.is_nan())
            .then(register_share)
            .otherwise(
                (register_share + weight * (estimate - register_share)).clip(lower_bound=0.0)
            )
        )
        .otherwise(
            pl.when(estimate.is_nan())
            .then(register_share)
            .otherwise(pl.max_horizontal(estimate, register_share))
        ),
    ).with_columns(unregistered_cap=pl.col("cap_with_register") - register_share)


def known_answer(
    *, series: pl.DataFrame, grids: dict[str, pv.SiteGrid], origins: pl.Series
) -> pl.DataFrame:
    """Add `-c(t) * p_unit(t)` to two Primary series and return the recovered and true `c`.

    Args:
        series: `common.series_table()`.
        grids: Each site's grid.
        origins: Forecast origins.

    Returns:
        Per site and origin: the true mean `c` over the window, F1, F1's standard error, and F2.
    """
    arrays = common.load_power(series=series)
    cell, day = pv.local_cells()
    rows = []
    grid_size = common.grid_times().len()
    ramp = np.linspace(0.0, KNOWN_ANSWER_FINAL_SHARE, grid_size)
    for site in KNOWN_ANSWER_SITES:
        injected = arrays[site] - ramp * np.nan_to_num(grids[site].cams_unit_power)
        for origin, index in zip(
            origins, np.asarray(common.grid_index(times=origins)), strict=True
        ):
            end = int(index) - pv.FEATURE_DELAY_STEPS
            fit = pv.window_fit(power=injected, site=grids[site], cell=cell, day=day, end_index=end)
            start = end - pv.WINDOW_DAYS * common.STEPS_PER_DAY + 1
            rows.append(
                {
                    "site": site,
                    "power_fcst_init_time": origin,
                    "true_c": float(ramp[max(start, 0) : end + 1].mean()),
                    "f1": fit.capacity,
                    "f1_se": fit.standard_error,
                    "f2": fit.sunny_minus_cloudy,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    """Build and write the features and the known-answer check."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    register = ecr.ecr_table(series=series, decisions_path=common.WORK_DIR / "ecr_rows.csv")
    print(register)
    origins = origin_times()
    arrays = common.load_power(series=series)
    cell, day = pv.local_cells()
    grids: dict[str, pv.SiteGrid] = {}
    pieces = []
    for row in series.iter_rows(named=True):
        started = time.monotonic()
        grid = pv.site_grid(site=row["site"], latitude=row["latitude"], longitude=row["longitude"])
        grids[row["site"]] = grid
        piece = pv.origin_features(
            power=arrays[row["site"]], site=grid, cell=cell, day=day, origins=origins
        )
        pieces.append(
            piece.with_columns(
                time_series_id=pl.lit(row["time_series_id"]),
                site=pl.lit(row["site"]),
                role=pl.lit("prior" if row["substation_type"] == "Primary" else "floor"),
            )
        )
        print(f"{row['site']}: {time.monotonic() - started:.1f} s", flush=True)
    features = capacity_estimates(features=pl.concat(pieces), register=register)
    features.write_parquet(FEATURES_DIR / "origin_features.parquet")
    answer = known_answer(series=series, grids=grids, origins=origins)
    answer.write_parquet(FEATURES_DIR / "known_answer.parquet")
    summary = (
        answer.join(
            features.select(
                "site",
                "power_fcst_init_time",
                base="cap_slope_28d",
                base_f2="sunny_minus_cloudy_28d",
            ),
            on=["site", "power_fcst_init_time"],
            validate="1:1",
        )
        .filter(pl.col("f1").is_not_nan() & pl.col("base").is_not_nan())
        .with_columns(
            error=pl.col("f1") - pl.col("base") - pl.col("true_c"),
            f2_error=pl.col("f2") - pl.col("base_f2") - pl.col("true_c"),
        )
        .group_by("site")
        .agg(
            origins=pl.len(),
            mean_error=pl.col("error").mean(),
            f2_rmse=(pl.col("f2_error") ** 2).mean().sqrt(),
            rmse=(pl.col("error") ** 2).mean().sqrt(),
            mean_se=pl.col("f1_se").mean(),
            within_one_se=((pl.col("error").abs() <= pl.col("f1_se")).mean()),
        )
    )
    print(summary)


if __name__ == "__main__":
    main()
