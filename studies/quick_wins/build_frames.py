"""Build the feature frames every quick-wins arm shares, once, and cache them as parquet.

The production feature engineer is run over a superset of feature names: the 24 production
features, the forecast lead time, long-wave radiation, precipitation, and 24 h and 72 h rolling
means of the air temperature. `extra_features` then adds the columns the engineer lacks. An arm
selects its columns from the cached frames, so the slow engineering step runs once per fold.

The training frame holds the control member only, as production trains. Each sampled ECMWF run has
its own validation frame holding every ensemble member.

    uv run python studies/quick_wins/build_frames.py [--fold mid_2025_to_mid_2026]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Final

import extra_features
import patito as pt
import polars as pl
import quick_wins_inputs as inputs
from contracts.power_schemas import TimeSeriesMetadata
from contracts.settings import Settings
from ml_core.features._parsed_features import ParsedFeatures
from ml_core.features.tabular_feature_engineer import TabularFeatureEngineer

ENGINEERED_FEATURES: Final[frozenset[str]] = frozenset(
    {
        *inputs.production_config().selected_features,
        "nwp_lead_time_hours",
        "downward_long_wave_radiation_flux_surface",
        "precipitation_surface",
        "temperature_2m_rolling_mean_24h",
        "temperature_2m_rolling_mean_72h",
    }
)
"""Every feature name the cached frames carry from the production feature engineer."""

KEY_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "valid_time",
    "nwp_init_time",
    "power_fcst_init_time",
    "ensemble_member",
    "power",
)
"""The non-feature columns every cached frame keeps."""


def frames_dir(*, fold_id: str) -> Path:
    """Return the directory that holds a fold's cached frames."""
    return inputs.OUTPUT_ROOT / "frames" / fold_id


def _engineer(
    *,
    settings: Settings,
    series_ids: list[int],
    metadata: pt.DataFrame[TimeSeriesMetadata],
    window_start: datetime,
    window_end: datetime,
    ensemble_members: list[int] | None,
    init_time_start: datetime | None = None,
    init_time_end: datetime | None = None,
) -> pl.DataFrame:
    """Run the production feature engineer and add the extra columns, as Float32 where numeric."""
    power, nwp = inputs.engineering_inputs(
        settings=settings,
        time_series_ids=series_ids,
        metadata=metadata,
        window_start=window_start,
        window_end=window_end,
        ensemble_members=ensemble_members,
        init_time_start=init_time_start,
        init_time_end=init_time_end,
        power_lookback=ParsedFeatures.from_strings(set(ENGINEERED_FEATURES)).max_power_lag(),
    )
    engineered = (
        TabularFeatureEngineer()
        .engineer(
            selected_features=set(ENGINEERED_FEATURES),
            power_time_series=power,
            time_series_metadata=metadata,
            nwp=nwp,
        )
        .collect(engine="streaming")
    )
    keys = [*KEY_COLUMNS]
    features = [name for name in sorted(ENGINEERED_FEATURES) if name not in keys]
    frame = pl.DataFrame(engineered).select(*keys, *features)
    with_extra = extra_features.add_extra_features(frame=frame, metadata=metadata)
    float_columns = [c for c, t in with_extra.schema.items() if t.is_float() and c != "power"]
    return with_extra.cast(dict.fromkeys(float_columns, pl.Float32))


def _check_rows(*, frame: pl.DataFrame, init_time: datetime, fold_id: str) -> None:
    """Raise if a validation frame has duplicate keys or a different height than the refit."""
    if frame.height == 0:
        msg = f"the run initialised at {init_time} produced no rows"
        raise ValueError(msg)
    keys = ["time_series_id", "nwp_init_time", "valid_time", "ensemble_member"]
    if frame.select(keys).is_duplicated().any():
        msg = f"the run initialised at {init_time} has duplicate keys"
        raise ValueError(msg)
    refit = (
        inputs.OUTPUT_ROOT
        / "production_arm"
        / "colsample0.8_seed0"
        / "forecasts"
        / f"init={init_time:%Y-%m-%d}.parquet"
    )
    if fold_id == inputs.FOLD_A and refit.exists():
        expected = pl.scan_parquet(refit).select(pl.len()).collect().item()
        if frame.height != expected:
            msg = f"{init_time}: frame has {frame.height:,} rows, the refit predicted {expected:,}"
            raise ValueError(msg)


def main() -> None:
    """Build the training frame and one validation frame per sampled run, skipping finished ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default=inputs.FOLD_A)
    args = parser.parse_args()

    settings = Settings()
    window = inputs.window_for(fold_id=args.fold)
    # The same 31 series on both folds, on purpose: only the leaderboard fold has an eligibility
    # table, and fold B is a robustness check of the same series, not a second population.
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    metadata = inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    out_dir = frames_dir(fold_id=window.fold_id)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.parquet"
    if not train_path.exists():
        started = time.monotonic()
        train = _engineer(
            settings=settings,
            series_ids=series_ids,
            metadata=metadata,
            window_start=window.train_start,
            window_end=window.train_end,
            ensemble_members=[0],
        )
        train.write_parquet(train_path.with_suffix(".tmp"))
        train_path.with_suffix(".tmp").rename(train_path)
        print(
            f"train: {train.height:,} rows, {train.width} cols, {time.monotonic() - started:.0f} s"
        )

    for init_time in inputs.sample_init_times(window=window):
        path = out_dir / "val" / f"init={init_time:%Y-%m-%d}.parquet"
        if path.exists():
            continue
        started = time.monotonic()
        frame = _engineer(
            settings=settings,
            series_ids=series_ids,
            metadata=metadata,
            window_start=window.val_start,
            window_end=min(window.val_end, init_time + inputs.MAX_NWP_LEAD),
            ensemble_members=None,
            init_time_start=init_time,
            init_time_end=init_time,
        )
        _check_rows(frame=frame, init_time=init_time, fold_id=window.fold_id)
        frame.write_parquet(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
        print(f"{init_time:%Y-%m-%d}: {frame.height:,} rows in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
