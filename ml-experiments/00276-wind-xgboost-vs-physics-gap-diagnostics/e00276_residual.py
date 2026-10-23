"""D5: what does the physics core's residual depend on, with ERA5 as the weather?

Per farm, fits the core (P) on the training years, takes its median forecast's residual
(observed minus the median, in points of capacity), fits a small XGBoost regression of that
residual on the D4 inputs plus wind direction in degrees, and reports on the scored cases of the
held-out year (a) the share of the residual's variance the regression explains, (b) permutation
importance of each input (rise in mean squared error when the input is shuffled), and (c)
partial-dependence curves.

Output: `work/d5.json`.

    cd ml-experiments/00276-wind-xgboost-vs-physics-gap-diagnostics
    uv run --no-sync python e00276_residual.py
"""

import json
import os

os.environ.setdefault("E00276_TORCH_THREADS", "4")

import e00276_common as common
import e00276_core as core_module
import e00276_models as models
import numpy as np
import polars as pl
import xgboost as xgb

FEATURES = (
    "wind_speed_100m",
    "wind_speed_10m",
    "direction_deg",
    "temperature_c",
    "hour",
    "doy",
    "year_frac",
)
PARAMETERS = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 50,
    "subsample": 0.8,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 4,
    "seed": 0,
}
ROUNDS = 300
REPEATS = 5
GRID_POINTS = 25
PDP_ROWS = 4_000


def with_direction(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the 100 m wind direction in degrees."""
    return frame.with_columns(
        direction_deg=(
            pl.arctan2(pl.col("wind_dir_sin_100m"), pl.col("wind_dir_cos_100m")).degrees() % 360
        ).cast(pl.Float32)
    )


def main() -> None:
    """Fit the residual model per farm and write the report."""
    sites = common.site_table()
    report: dict = {}
    for row in sites.iter_rows(named=True):
        farm = row["site"]
        one = sites.filter(pl.col("site") == farm)
        era5 = common.era5_rows(sites=one)
        train = with_direction(
            frame=models.add_derived_columns(frame=common.training_table(sites=one, era5=era5))
        )
        cases = common.cases_table(sites=one)
        held = with_direction(
            frame=models.add_derived_columns(
                frame=cases.join(era5, on=["time_series_id", "valid_time"], how="left")
            )
        )
        weather = models.weather_arrays(frame=train, options=core_module.DEFAULT_OPTIONS)
        target = train["target"].to_numpy()
        core = core_module.fit_core(
            weather=weather, target=target, with_idle_layer=farm != "W2", seed=0
        )
        expected = core_module.predict_expected_power(core=core, weather=weather)
        residuals = core_module.ResidualQuantiles.fit(expected=expected, observed=target)
        scale = common.POINTS_PER_FRACTION
        median_train = residuals.quantiles(expected=expected)[:, common.MEDIAN_INDEX]
        held_weather = models.weather_arrays(frame=held, options=core_module.DEFAULT_OPTIONS)
        median_held = residuals.quantiles(
            expected=core_module.predict_expected_power(core=core, weather=held_weather)
        )[:, common.MEDIAN_INDEX]
        y_train = (target - median_train) * scale
        y_held = (held["actual_norm"].to_numpy() - median_held) * scale
        x_train = train.select(FEATURES).to_numpy()
        x_held = held.select(FEATURES).to_numpy()
        booster = xgb.train(PARAMETERS, xgb.DMatrix(x_train, label=y_train), num_boost_round=ROUNDS)
        base = ((booster.predict(xgb.DMatrix(x_held)) - y_held) ** 2).mean()
        total = ((y_held - y_held.mean()) ** 2).mean()
        rng = np.random.default_rng(0)
        importance = {}
        for index, name in enumerate(FEATURES):
            rises = []
            for _ in range(REPEATS):
                shuffled = x_held.copy()
                shuffled[:, index] = rng.permutation(shuffled[:, index])
                rises.append(((booster.predict(xgb.DMatrix(shuffled)) - y_held) ** 2).mean() - base)
            importance[name] = float(np.mean(rises))
        sample = x_held[
            rng.choice(x_held.shape[0], size=min(PDP_ROWS, x_held.shape[0]), replace=False)
        ]
        pdp = {}
        for index, name in enumerate(FEATURES):
            grid = np.quantile(x_held[:, index], np.linspace(0.02, 0.98, GRID_POINTS))
            values = []
            for point in grid:
                varied = sample.copy()
                varied[:, index] = point
                values.append(float(booster.predict(xgb.DMatrix(varied)).mean()))
            pdp[name] = {"grid": [float(g) for g in grid], "effect_points": values}
        report[farm] = {
            "residual_variance_points_squared": float(total),
            "residual_model_mse_points_squared": float(base),
            "share_explained": float(1 - base / total),
            "mean_residual_points": float(y_held.mean()),
            "permutation_importance_mse_rise": importance,
            "partial_dependence": pdp,
        }
        print(farm, round(report[farm]["share_explained"], 3), importance, flush=True)
    (common.WORK_DIR / "d5.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
