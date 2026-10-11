"""Refit the production forecast on the leaderboard fold and predict the sampled ECMWF runs.

This is the first step of the quick-wins study. The refit uses the production forecaster class, its
feature engineer, and `conf/model/xgboost.yaml` unchanged, trained on the fold's training window
from the control member alone, as `trained_cv_model` does. It then predicts every ensemble member
of each run in the sample, as `cv_power_forecasts` does for every run. `report_production_arm.py`
scores the result against the stored `xgboost_baseline` rows.

Run from the repository root:

    uv run python studies/quick_wins/fit_production_arm.py [--colsample-bytree 1.0] [--device cuda]

A finished output is never overwritten: the script skips what exists, so a crash costs one run.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import inputs
import patito as pt
import polars as pl
from contracts.power_schemas import TimeSeriesMetadata
from contracts.settings import Settings
from ml_core.features._parsed_features import ParsedFeatures
from xgboost_forecaster.forecaster import XGBoostConfig, XGBoostForecaster


def _output_dir(*, arm: str) -> Path:
    return inputs.OUTPUT_ROOT / "production_arm" / arm


def _predict_run(
    *,
    forecaster: XGBoostForecaster,
    settings: Settings,
    metadata: pt.DataFrame[TimeSeriesMetadata],
    window: inputs.FoldWindow,
    init_time: datetime,
) -> pl.DataFrame:
    """Predict every ensemble member of one ECMWF run over its valid times inside the window."""
    config = forecaster.model_params
    power, nwp = inputs.engineering_inputs(
        settings=settings,
        time_series_ids=forecaster.trained_time_series_ids,
        metadata=metadata,
        window_start=window.val_start,
        window_end=min(window.val_end, init_time + inputs.MAX_NWP_LEAD),
        init_time_start=init_time,
        init_time_end=init_time,
        power_lookback=ParsedFeatures.from_strings(config.selected_features).max_power_lag(),
    )
    features = forecaster.feature_engineer.engineer(
        selected_features=config.selected_features,
        power_time_series=power,
        time_series_metadata=metadata,
        nwp=nwp,
    )
    forecasts = forecaster.predict(features, fold_id=window.fold_id)
    if forecasts.height == 0:
        msg = f"the run initialised at {init_time} produced no forecast rows"
        raise ValueError(msg)
    return forecasts.select(
        "time_series_id",
        "power_fcst_init_time",
        "nwp_init_time",
        "valid_time",
        "ensemble_member",
        "power_fcst",
    )


def main() -> None:
    """Train the production arm, then predict each sampled run, skipping finished outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    arm = f"colsample{args.colsample_bytree}_seed{args.seed}"
    if args.device != "cpu":
        arm += f"_{args.device}"
    out_dir = _output_dir(arm=arm)
    model_dir = out_dir / "model"
    forecast_dir = out_dir / "forecasts"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    window = inputs.fold_window()
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=window.fold_id)
    metadata = inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    config = XGBoostConfig(
        **{
            **inputs.production_config().model_dump(),
            "colsample_bytree": args.colsample_bytree,
            "random_seed": args.seed,
            "device": args.device,
        }
    )
    power_lookback = ParsedFeatures.from_strings(config.selected_features).max_power_lag()

    if (model_dir / "meta.json").exists():
        forecaster = XGBoostForecaster.load(model_dir)
        if forecaster.model_params != config:
            msg = f"{model_dir} was trained with a different configuration than this run asks for"
            raise ValueError(msg)
        print(f"loaded {model_dir}")
    else:
        started = time.monotonic()
        power, nwp = inputs.engineering_inputs(
            settings=settings,
            time_series_ids=series_ids,
            metadata=metadata,
            window_start=window.train_start,
            window_end=window.train_end,
            ensemble_members=[0],
            power_lookback=power_lookback,
        )
        forecaster = XGBoostForecaster(model_params=config)
        features = forecaster.feature_engineer.engineer(
            selected_features=config.selected_features,
            power_time_series=power,
            time_series_metadata=metadata,
            nwp=nwp,
        )
        forecaster.train(features, series_ids)
        untrained = sorted(set(series_ids) - set(forecaster.trained_time_series_ids))
        if untrained:
            # The booster trainer drops a series with no usable training rows, as production does.
            # The report checks that the stored experiment dropped the same series.
            print(f"WARNING: {len(untrained)} eligible series were not trained: {untrained}")
        for stale in forecast_dir.glob("*.parquet"):
            stale.unlink()
        forecaster.save(model_dir)
        print(
            f"trained {len(forecaster.trained_time_series_ids)} boosters "
            f"in {time.monotonic() - started:.0f} s"
        )

    for init_time in inputs.sample_init_times(window=window):
        path = forecast_dir / f"init={init_time:%Y-%m-%d}.parquet"
        if path.exists():
            continue
        started = time.monotonic()
        forecasts = _predict_run(
            forecaster=forecaster,
            settings=settings,
            metadata=metadata,
            window=window,
            init_time=init_time,
        )
        forecasts.write_parquet(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
        print(
            f"{init_time:%Y-%m-%d}: {forecasts.height:,} rows in {time.monotonic() - started:.0f} s"
        )
    n_files = len(list(forecast_dir.glob("*.parquet")))
    if n_files != len(inputs.sample_init_times(window=window)):
        msg = f"{n_files} forecast files for {len(inputs.sample_init_times(window=window))} runs"
        raise ValueError(msg)
    print(pl.scan_parquet(forecast_dir / "*.parquet").select(pl.len()).collect().item(), "rows")


if __name__ == "__main__":
    main()
