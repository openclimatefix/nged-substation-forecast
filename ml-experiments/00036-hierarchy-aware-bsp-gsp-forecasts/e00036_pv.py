"""Fit experiment 00028's PV_Live solar-share models and write their forecasts for B4.

For each GSP and seed, one XGBoost model forecasts PV_Live's share of its own capacity from weather
at the valid time and the sun's position. The final model trains on valid times from 2019-09 to
before 2025-06-17 and scores the validation runs member by member. Five forward-chain models, one
per quarter from 2024-04, train on valid times that end 15 days before the quarter starts and score
the control-member training rows of their quarter, so no training row meets a model that saw its
period. The copy of experiment 00028's loader and fit is trimmed to what B4 needs.

Output: `pv/train_seed<s>.parquet` (GSP, run, valid time, share) and `pv/val_seed<s>.parquet` (the
same with the member). Attribution: Sheffield Solar PV_Live, CC BY 4.0.

    uv run python ml-experiments/00036-*/e00036_pv.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00036_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PV_DIR: Final[Path] = common.WORK_DIR / "pv"
PV_LIVE_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "studies" / "downloads" / "market" / "pv_live_gsp"
) / "pv_live_gsp.parquet"
GSP_PV_LIVE_NAMES: Final[dict[str, str]] = {"D1": "BICF_1", "D6": "WALP_B"}
"""Bicker Fen and Walpole, and the PV_Live name of each."""
FINAL_TRAIN_CUTOFF: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
CHAIN_STARTS: Final[tuple[datetime, ...]] = tuple(start for start, _ in common.BLOCKS)
"""Quarter starts of the forward chain; each quarter's model ends `PURGE` before it starts."""
PV_FEATURES: Final[tuple[str, ...]] = (
    *common.WEATHER,
    "solar_zenith",
    "solar_azimuth",
)
PV_PARAMETERS: Final[dict[str, float | int | str]] = {
    **common.XGBOOST_PARAMETERS,
    "nthread": 8,
}


def sun_columns(*, valid_time: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the sun's zenith and azimuth at the midpoint of each period-ending half-hour."""
    midpoint = valid_time.dt.offset_by("-15m")
    return pl.DataFrame(
        {
            "solar_zenith": zenith(stamps=midpoint, latitude=latitude, longitude=longitude),
            "solar_azimuth": azimuth(stamps=midpoint, latitude=latitude, longitude=longitude),
        }
    ).cast(pl.Float32)


def pv_live_share(*, series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each GSP's PV_Live share of its own capacity on the grid, NaN where missing.

    PV_Live labels each half-hour by its start and the grid by its end, so grid stamp `t` takes the
    estimate that starts at `t - 30 min`.
    """
    grid = pl.DataFrame({"valid_time": common.grid_times()}).with_columns(
        start=pl.col("valid_time") - common.STEP
    )
    pv = pl.read_parquet(PV_LIVE_PATH)
    out = {}
    for row in series.iter_rows(named=True):
        joined = grid.join(
            pv.filter(pl.col("gsp_name") == GSP_PV_LIVE_NAMES[row["site"]]).select(
                start="time", percent="generation_percent_of_capacity"
            ),
            on="start",
            how="left",
            validate="1:1",
        ).sort("valid_time")
        out[row["site"]] = (joined["percent"] / 100.0).to_numpy().astype(np.float32)
    return out


def era5_matrix(*, series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each GSP's solar-model inputs at ERA5 weather on the dense grid."""
    weather = list(common.WEATHER)
    grid = pl.DataFrame({"valid_time": common.grid_times()})
    era5 = pl.read_parquet(common.ERA5_PATH)
    out = {}
    for row in series.iter_rows(named=True):
        joined = grid.join(
            era5.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                "valid_time", *weather
            ),
            on="valid_time",
            how="left",
            validate="1:1",
        ).sort("valid_time")
        sun = sun_columns(
            valid_time=joined["valid_time"], latitude=row["latitude"], longitude=row["longitude"]
        )
        out[row["site"]] = (
            joined.select(weather).with_columns(sun).select(PV_FEATURES).to_numpy().astype("f4")
        )
    return out


def fit_models(*, features: np.ndarray, share: np.ndarray, seed: int) -> dict[str, xgb.Booster]:
    """Fit the final model and the five forward-chain models of one GSP and seed."""
    start = int(common.grid_index(times=common.GRID_START))
    stops = {"final": int(common.grid_index(times=FINAL_TRAIN_CUTOFF))}
    stops |= {
        f"q{tag}": int(common.grid_index(times=quarter - common.PURGE))
        for tag, quarter in enumerate(CHAIN_STARTS)
    }
    models = {}
    for tag, stop in stops.items():
        rows = np.arange(start, stop)
        rows = rows[np.isfinite(share[rows]) & np.isfinite(features[rows]).all(axis=1)]
        models[tag] = xgb.train(
            {**PV_PARAMETERS, "seed": seed},
            xgb.DMatrix(features[rows], label=share[rows]),
            num_boost_round=common.N_ROUNDS,
        )
    return models


def predict_share(
    *, frame: pl.DataFrame, row: dict, models: dict[str, xgb.Booster], chained: bool
) -> np.ndarray:
    """Return the solar share of each row, from the quarter's chain model or the final model."""
    sun = sun_columns(
        valid_time=frame["valid_time"], latitude=row["latitude"], longitude=row["longitude"]
    )
    matrix = (
        frame.select(list(common.WEATHER))
        .with_columns(sun)
        .select(PV_FEATURES)
        .to_numpy()
        .astype(np.float32)
    )
    if not chained:
        return models["final"].predict(xgb.DMatrix(matrix))
    quarter = np.clip(
        np.searchsorted(
            [int(common.grid_index(times=q)) for q in CHAIN_STARTS],
            np.asarray(common.grid_index(times=frame["valid_time"])),
            side="right",
        )
        - 1,
        0,
        len(CHAIN_STARTS) - 1,
    )
    out = np.empty(frame.height, dtype=np.float32)
    for tag in range(len(CHAIN_STARTS)):
        mask = quarter == tag
        if mask.any():
            out[mask] = models[f"q{tag}"].predict(xgb.DMatrix(matrix[mask]))
    return out


def main() -> None:
    """Fit the models and write the training-row and validation-row forecasts for every seed."""
    series = common.series_table().filter(pl.col("site").is_in(list(GSP_PV_LIVE_NAMES)))
    share = pv_live_share(series=series)
    features = era5_matrix(series=series)
    train = pl.read_parquet(common.ENS_TRAIN_PATH).filter(
        pl.col("valid_time") <= common.LAST_TRAINING_VALID
    )
    PV_DIR.mkdir(parents=True, exist_ok=True)
    for seed in common.SEEDS:
        train_path, val_path = (
            PV_DIR / f"train_seed{seed}.parquet",
            PV_DIR / f"val_seed{seed}.parquet",
        )
        if train_path.exists() and val_path.exists():
            continue
        train_pieces, val_pieces = [], []
        for row in series.iter_rows(named=True):
            models = fit_models(features=features[row["site"]], share=share[row["site"]], seed=seed)
            control = train.filter(pl.col("time_series_id") == row["time_series_id"])
            keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
            train_pieces.append(
                control.select(keys).with_columns(
                    pv_share=pl.Series(
                        predict_share(frame=control, row=row, models=models, chained=True)
                    )
                )
            )
            for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
                run = pl.read_parquet(path).filter(
                    pl.col("time_series_id") == row["time_series_id"]
                )
                val_pieces.append(
                    run.select([*keys, "ensemble_member"]).with_columns(
                        pv_share=pl.Series(
                            predict_share(frame=run, row=row, models=models, chained=False)
                        )
                    )
                )
            print(f"pv seed {seed} {row['site']} done", flush=True)
        for path, pieces in ((train_path, train_pieces), (val_path, val_pieces)):
            pl.concat(pieces).write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)


if __name__ == "__main__":
    main()
