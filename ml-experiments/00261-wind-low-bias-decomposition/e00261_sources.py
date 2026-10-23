"""Which weather input carries the shift, and does training on ensemble weather remove it?

Two analyses on experiment 00259's hindcast and scored cases, both using the physics core (R1)
refitted on ERA5 before 2024-04-01 exactly as in `e00261_decompose.py`.

1. **Input swap.** Start from the ERA5 oracle and replace one input at a time by its ensemble-mean
   value: the 100 m wind speed, the 10 m wind speed, the 2 m temperature. The median shift of each
   swap, and of all three together, is in points of capacity on the kept rows.
2. **Training source.** A physics core and an XGBoost quantile model are fitted on the ensemble-mean
   weather of the first 9 months of hindcast cases (init dates 2024-04-01 to 2024-12-31) against
   the observed power, and scored on the ensemble-mean weather of the held-out hindcast months
   (2025-01-01 to 2025-06-16) and of the scored year. The ERA5-trained versions of the same two
   models are scored on the same rows. A single ensemble-mean value per case is the input to every
   model here, so the comparison isolates the training source from the member mixing.

Output, in `work/`: `sources.json`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00261-*/e00261_sources.py
"""

import json
import os
from datetime import UTC, datetime
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00261_common as common  # noqa: E402
import e00261_core as core_module  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402
from e00261_decompose import PRIOR_DIR, weather  # noqa: E402

BASE_END: Final = datetime(2024, 4, 1, tzinfo=UTC)
ENS_TRAIN_END: Final = datetime(2025, 1, 1, tzinfo=UTC)
"""Cases with a run before this date train the ensemble-trained models: 9 months."""
XGBOOST_ROUNDS: Final[int] = 500
SPEED_BIN_WIDTH_M_S: Final[float] = 0.5
XGBOOST_PARAMETERS: Final[dict] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": common.LEVELS,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "nthread": 4,
    "seed": 0,
}
FEATURES: Final[tuple[str, ...]] = ("speed_10m", "speed_100m", "dir_sin", "dir_cos")


def curve(*, speed: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a non-decreasing power curve of the fraction on 100 m wind speed, by 0.5 m/s bin."""
    bins = np.floor(speed / SPEED_BIN_WIDTH_M_S).astype(np.int64)
    ordered = (
        pl.DataFrame({"bin": bins, "power": target})
        .group_by("bin")
        .agg(pl.col("power").mean())
        .sort("bin")
    )
    return (ordered["bin"].to_numpy() + 0.5) * SPEED_BIN_WIDTH_M_S, np.maximum.accumulate(
        ordered["power"].to_numpy()
    )


class QuantileModel:
    """An XGBoost quantile model on four wind columns with a power-curve base margin."""

    def __init__(self, *, features: np.ndarray, speed_100m: np.ndarray, target: np.ndarray) -> None:
        """Fit the model on `features` (rows, 4) against `target` fractions of capacity."""
        self.centres, self.values = curve(speed=speed_100m, target=target)
        self.booster = xgb.train(
            XGBOOST_PARAMETERS,
            xgb.DMatrix(features, label=target, base_margin=self._margin(speed_100m)),
            num_boost_round=XGBOOST_ROUNDS,
        )

    def _margin(self, speed_100m: np.ndarray) -> np.ndarray:
        return np.tile(np.interp(speed_100m, self.centres, self.values)[:, None], (1, 11))

    def median(self, *, features: np.ndarray, speed_100m: np.ndarray) -> np.ndarray:
        """Return the sorted, clipped median forecast of each row."""
        predicted = self.booster.predict(
            xgb.DMatrix(features, base_margin=self._margin(speed_100m))
        )
        return np.sort(np.clip(predicted, 0.0, 1.0), axis=1)[:, common.MEDIAN_INDEX]


def share(*, actual: np.ndarray, median: np.ndarray) -> float:
    """Return the share of rows with the observation strictly below the median."""
    return float((actual < median).mean())


def load_cases() -> pl.DataFrame:
    """Return hindcast and scored cases with ensemble-mean and ERA5 inputs, kept flag, and set."""
    sites = common.site_table()
    era5 = common.era5_rows(sites=sites).select(
        "time_series_id",
        "valid_time",
        era5_temp=pl.col("temperature_c"),
        era5_dir_sin=pl.col("wind_dir_sin_100m"),
        era5_dir_cos=pl.col("wind_dir_cos_100m"),
    )
    pieces = []
    for name in ("hindcast", "scored"):
        cases = pl.read_parquet(common.WORK_DIR / f"cases_{name}.parquet")
        summary = pl.read_parquet(PRIOR_DIR / "base" / f"summaries_{name}.parquet").select(
            *common.KEY_COLUMNS, "temp_mean", "dir_sin", "dir_cos"
        )
        pieces.append(
            cases.join(summary, on=list(common.KEY_COLUMNS), validate="1:1")
            .join(era5, on=["time_series_id", "valid_time"], how="left", validate="m:1")
            .with_columns(set=pl.lit(name))
        )
    return pl.concat(pieces, how="vertical_relaxed")


def era5_training(*, series: int) -> pl.DataFrame:
    """Return one farm's ERA5 training rows before `BASE_END`."""
    sites = common.site_table().filter(pl.col("time_series_id") == series)
    return common.training_table(sites=sites, era5=common.era5_rows(sites=sites)).filter(
        pl.col("valid_time") < BASE_END
    )


def main() -> None:
    """Write `sources.json`."""
    cases = load_cases()
    sites = common.site_table()
    out: dict = {"swap": {}, "training_source": {}}
    arms: dict[str, list[pl.DataFrame]] = {}
    for row in sites.iter_rows(named=True):
        series, farm = int(row["time_series_id"]), row["site"]
        era5_rows = era5_training(series=series)
        era5_target = era5_rows["target"].to_numpy()
        era5_core = core_module.fit_core(
            weather=weather(frame=era5_rows),
            target=era5_target,
            with_idle_layer=farm != "W2",
            seed=0,
        )
        era5_residuals = core_module.ResidualQuantiles.fit(
            expected=core_module.predict_expected_power(
                core=era5_core, weather=weather(frame=era5_rows)
            ),
            observed=era5_target,
        )
        farm_cases = cases.filter(pl.col("time_series_id") == series).drop_nulls(
            ["era5_speed_100m", "era5_temp", "ens_speed_100m", "temp_mean"]
        )

        def q50_from_inputs(
            *,
            speed_10m: str,
            speed_100m: str,
            temperature: str,
            frame: pl.DataFrame = farm_cases,
            core: core_module.WindCore = era5_core,
            residuals: core_module.ResidualQuantiles = era5_residuals,
        ) -> np.ndarray:
            arrays = core_module.WeatherArrays(
                speed_10m=frame[speed_10m].to_numpy().astype(np.float32),
                speed_100m=frame[speed_100m].to_numpy().astype(np.float32),
                temperature_c=frame[temperature].to_numpy().astype(np.float32),
            )
            expected = core_module.predict_expected_power(core=core, weather=arrays)
            return residuals.quantiles(expected=expected)[:, common.MEDIAN_INDEX]

        swaps = {
            "oracle": ("era5_speed_10m", "era5_speed_100m", "era5_temp"),
            "ens 100 m speed only": ("era5_speed_10m", "ens_speed_100m", "era5_temp"),
            "ens 10 m speed only": ("ens_speed_10m", "era5_speed_100m", "era5_temp"),
            "ens temperature only": ("era5_speed_10m", "era5_speed_100m", "temp_mean"),
            "ens 10 m and 100 m speed": ("ens_speed_10m", "ens_speed_100m", "era5_temp"),
            "all three ens": ("ens_speed_10m", "ens_speed_100m", "temp_mean"),
        }
        frame = farm_cases
        for label, (s10, s100, temperature) in swaps.items():
            frame = frame.with_columns(
                pl.Series(
                    f"q50 {label}",
                    q50_from_inputs(speed_10m=s10, speed_100m=s100, temperature=temperature),
                )
            )
        # Training source: ensemble-mean weather of the first 9 months of hindcast cases.
        is_train = (frame["set"] == "hindcast") & (frame["power_fcst_init_time"] < ENS_TRAIN_END)
        train = frame.filter(is_train)
        ens_arrays = core_module.WeatherArrays(
            speed_10m=train["ens_speed_10m"].to_numpy().astype(np.float32),
            speed_100m=train["ens_speed_100m"].to_numpy().astype(np.float32),
            temperature_c=train["temp_mean"].to_numpy().astype(np.float32),
        )
        ens_target = train["actual_norm"].to_numpy().astype(np.float64)
        ens_core = core_module.fit_core(
            weather=ens_arrays, target=ens_target, with_idle_layer=farm != "W2", seed=0
        )
        ens_residuals = core_module.ResidualQuantiles.fit(
            expected=core_module.predict_expected_power(core=ens_core, weather=ens_arrays),
            observed=ens_target,
        )
        frame = frame.with_columns(
            pl.Series(
                "q50 ens-trained core",
                q50_from_inputs(
                    speed_10m="ens_speed_10m",
                    speed_100m="ens_speed_100m",
                    temperature="temp_mean",
                    frame=frame,
                    core=ens_core,
                    residuals=ens_residuals,
                ),
            ),
            pl.Series(
                "q50 era5-trained core",
                q50_from_inputs(
                    speed_10m="ens_speed_10m",
                    speed_100m="ens_speed_100m",
                    temperature="temp_mean",
                    frame=frame,
                ),
            ),
        )
        ens_features = frame.select(
            "ens_speed_10m", "ens_speed_100m", "dir_sin", "dir_cos"
        ).to_numpy()
        train_features = ens_features[is_train.to_numpy()]
        ens_model = QuantileModel(
            features=train_features,
            speed_100m=train["ens_speed_100m"].to_numpy(),
            target=ens_target,
        )
        era5_model = QuantileModel(
            features=era5_rows.select(
                "wind_speed_10m", "wind_speed_100m", "wind_dir_sin_100m", "wind_dir_cos_100m"
            ).to_numpy(),
            speed_100m=era5_rows["wind_speed_100m"].to_numpy(),
            target=era5_target,
        )
        frame = frame.with_columns(
            pl.Series(
                "q50 ens-trained xgboost",
                ens_model.median(
                    features=ens_features, speed_100m=frame["ens_speed_100m"].to_numpy()
                ),
            ),
            pl.Series(
                "q50 era5-trained xgboost",
                era5_model.median(
                    features=ens_features, speed_100m=frame["ens_speed_100m"].to_numpy()
                ),
            ),
        )
        for name, piece in frame.partition_by("set", as_dict=True).items():
            arms.setdefault(str(name[0]), []).append(piece)
        print(f"{farm}: done", flush=True)
    columns = [c for c in arms["scored"][0].columns if c.startswith("q50 ")]
    for name, pieces in arms.items():
        data = pl.concat(pieces)
        if name == "hindcast":
            subsets = {
                "hindcast, first 9 months (in sample for the ensemble-trained models)": data.filter(
                    pl.col("power_fcst_init_time") < ENS_TRAIN_END
                ),
                "hindcast, held-out months 2025-01 to 2025-06": data.filter(
                    pl.col("power_fcst_init_time") >= ENS_TRAIN_END
                ),
            }
        else:
            subsets = {"scored year": data}
        for subset_name, subset in subsets.items():
            kept = subset.filter(~pl.col("masked"))
            actual = kept["actual_norm"].to_numpy()
            out["training_source"][subset_name] = {
                "rows_kept": kept.height,
                "by_farm": {
                    farm: {
                        c: share(
                            actual=kept.filter(pl.col("site") == farm)["actual_norm"].to_numpy(),
                            median=kept.filter(pl.col("site") == farm)[c].to_numpy(),
                        )
                        for c in columns
                    }
                    for farm in sorted(kept["site"].unique())
                },
                "share": {c: share(actual=actual, median=kept[c].to_numpy()) for c in columns},
                "rows_all": subset.height,
                "share_all": {
                    c: share(actual=subset["actual_norm"].to_numpy(), median=subset[c].to_numpy())
                    for c in columns
                },
                "mean_q50_points": {
                    c: float(np.mean(kept[c].to_numpy())) * common.POINTS_PER_FRACTION
                    for c in columns
                },
            }
    (common.WORK_DIR / "sources.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["training_source"], indent=2))


if __name__ == "__main__":
    main()
