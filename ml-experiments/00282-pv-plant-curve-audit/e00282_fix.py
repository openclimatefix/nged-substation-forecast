"""Check 7 of experiment 282: replace the engine's curve with the pooled leave-one-farm-out shape.

For each held-out farm the shape is a table of output over capacity against global irradiance and
sun elevation (`studies.pv_plant_fit`), fitted to the tuning rows of the other five farms and
penalised towards their fleet-shape predictions (experiment 269's pooled empirical shape). One
scale per held-out farm is fitted on its own tuning rows, as for the fleet shape. Run under
`systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue` with `uv run --no-sync python`.
"""

import os
from typing import Final

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import e00282_common as common
import e00282_tables as tables
import numpy as np
import polars as pl
from studies.pv_plant_fit import ResponseGrid, block_folds, fit_response_table

RESPONSE_GRID: Final[ResponseGrid] = ResponseGrid(
    irradiance_w_m2=np.arange(0.0, 1101.0, 50.0), elevation_deg=np.arange(5.0, 66.0, 5.0)
)
PENALTIES: Final[tuple[float, ...]] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
CV_FOLDS: Final[int] = 5
LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)
ARMS: Final[tuple[str, ...]] = ("plant_fleet", "plant_per_farm", "stack_term", "pooled_shape")
GAIN_THRESHOLD: Final[float] = 0.02


def pooled_prediction(*, frame: pl.DataFrame, farm: str) -> np.ndarray:
    """Return the pooled leave-one-farm-out shape's output for every row of `farm`, unscaled."""
    others = frame.filter((pl.col("site") != farm) & ~pl.col("scored"))
    ghi = others["ghi_w_m2"].to_numpy()
    elevation = others["elevation_deg"].to_numpy()
    output = others["output"].to_numpy()
    prior = others["plant_fleet"].to_numpy()
    folds = block_folds(block_of_row=others["month_index"].to_numpy(), n_folds=CV_FOLDS)
    errors = {}
    for penalty in PENALTIES:
        error = 0.0
        for held in folds:
            table = fit_response_table(
                grid=RESPONSE_GRID,
                irradiance_w_m2=ghi[~held],
                elevation_deg=elevation[~held],
                output=output[~held],
                prior_output=prior[~held],
                penalty=penalty,
            )
            predicted = table.predict(irradiance_w_m2=ghi[held], elevation_deg=elevation[held])
            error += float(np.sum((output[held] - predicted) ** 2))
        errors[penalty] = error
    best = min(errors, key=errors.__getitem__)
    table = fit_response_table(
        grid=RESPONSE_GRID,
        irradiance_w_m2=ghi,
        elevation_deg=elevation,
        output=output,
        prior_output=prior,
        penalty=best,
    )
    own = frame.filter(pl.col("site") == farm).sort("time")
    print(f"farm {farm}: penalty {best}", flush=True)
    return table.predict(
        irradiance_w_m2=own["ghi_w_m2"].to_numpy(), elevation_deg=own["elevation_deg"].to_numpy()
    )


def pinball_points(
    *, output: np.ndarray, predicted: np.ndarray, residual_quantiles: np.ndarray
) -> np.ndarray:
    """Return twice the mean pinball loss over `LEVELS` per row, in points of capacity."""
    total = np.zeros_like(output)
    for level, offset in zip(LEVELS, residual_quantiles, strict=True):
        forecast = np.maximum(predicted + offset, 0.0)
        error = output - forecast
        total += np.maximum(level * error, (level - 1.0) * error)
    return 2.0 * 100.0 * total / len(LEVELS)


def build_scored() -> pl.DataFrame:
    """Return the scored rows with one prediction per arm and the pinball loss per arm."""
    frame = pl.read_parquet(common.FRAME_PATH).sort("site", "time")
    parts = []
    for farm in common.FARMS:
        own = frame.filter(pl.col("site") == farm)
        tuning = (~own["scored"]).to_numpy()
        output = own["output"].to_numpy()
        pooled = pooled_prediction(frame=frame, farm=farm)
        scale = float(np.sum(pooled[tuning] * output[tuning]) / np.sum(pooled[tuning] ** 2))
        own = own.with_columns(pooled_shape=pl.Series(scale * pooled))
        for arm in ARMS:
            predicted = own[arm].to_numpy()
            offsets = np.quantile((output - predicted)[tuning], LEVELS)
            own = own.with_columns(
                **{
                    f"pinball_{arm}": pl.Series(
                        pinball_points(
                            output=output, predicted=predicted, residual_quantiles=offsets
                        )
                    ),
                    f"squared_{arm}": pl.Series((output - predicted) ** 2),
                }
            )
        parts.append(own.filter(pl.col("scored")))
    return pl.concat(parts).with_columns(squared_output=pl.col("output") ** 2)


def fraction_fitted_draws(*, scored: pl.DataFrame, arm: str) -> np.ndarray:
    """Return the month-block bootstrap draws of the median-over-farms fraction of output fitted."""
    months = tables.scored_months()
    draws = tables.month_block_draws()
    per_farm = []
    for farm in common.FARMS:
        own = scored.filter(pl.col("site") == farm)
        monthly = own.group_by("month_index").agg(
            error=pl.col(f"squared_{arm}").sum(), power=pl.col("squared_output").sum()
        )
        error = np.zeros(len(months))
        power = np.zeros(len(months))
        for row in monthly.iter_rows(named=True):
            error[months.index(row["month_index"])] = row["error"]
            power[months.index(row["month_index"])] = row["power"]
        per_farm.append(1.0 - error[draws].sum(axis=1) / power[draws].sum(axis=1))
    return np.median(np.array(per_farm), axis=0)


def main() -> None:
    """Write the arm scores per farm and the contrasts, and print them."""
    scored = build_scored()
    rows = [
        {
            "site": farm,
            "arm": arm,
            "fraction_fitted": 1.0
            - float(scored.filter(pl.col("site") == farm)[f"squared_{arm}"].sum())
            / float(scored.filter(pl.col("site") == farm)["squared_output"].sum()),
            "pinball_points": scored.filter(pl.col("site") == farm)[f"pinball_{arm}"].mean(),
            "mae_points": 100.0
            * float(np.sqrt(scored.filter(pl.col("site") == farm)[f"squared_{arm}"]).mean()),
        }
        for farm in common.FARMS
        for arm in ARMS
    ]
    per_farm = pl.DataFrame(rows)
    per_farm.write_csv(common.DATA_DIR / "check7_per_farm.csv")
    contrasts = []
    reference = fraction_fitted_draws(scored=scored, arm="plant_fleet")
    for arm in ARMS:
        draws = fraction_fitted_draws(scored=scored, arm=arm)
        difference = draws - reference
        medians = per_farm.filter(pl.col("arm") == arm)["fraction_fitted"].median()
        contrasts.append(
            {
                "arm": arm,
                "median_fraction_fitted": medians,
                "gain_vs_fleet": float(np.median(difference)),
                "low": float(np.quantile(difference, 0.025)),
                "high": float(np.quantile(difference, 0.975)),
            }
        )
    for arm in ("plant_per_farm",):
        draws = fraction_fitted_draws(scored=scored, arm="pooled_shape") - fraction_fitted_draws(
            scored=scored, arm=arm
        )
        contrasts.append(
            {
                "arm": f"pooled_shape_minus_{arm}",
                "median_fraction_fitted": None,
                "gain_vs_fleet": float(np.median(draws)),
                "low": float(np.quantile(draws, 0.025)),
                "high": float(np.quantile(draws, 0.975)),
            }
        )
    out = pl.DataFrame(contrasts)
    out.write_csv(common.DATA_DIR / "check7_contrasts.csv")
    pl.Config.set_tbl_rows(40)
    print(per_farm.pivot(on="arm", index="site", values="fraction_fitted"))
    print(per_farm.pivot(on="arm", index="site", values="pinball_points"))
    print(out)


if __name__ == "__main__":
    main()
