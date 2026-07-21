"""Export a CV experiment's forecasts to three parquet files for offline analysis.

Written for issue #179: hand the weather/calendar-only baseline forecasts (the switching-event
"shared baseline") to a colleague for switching-event detection work. The forecasts are read from
the internal ``power_forecasts`` Delta table and written as three self-contained parquet files,
all in physical MW/MVA units (``power_fcst`` is already stored in MW/MVA on disk — see
``PowerForecast.power_fcst``; the per-series unit lives in ``TimeSeriesMetadata.units``).

For each ``(time_series_id, valid_time)`` we keep only the **freshest** forecast run — the one
whose ``power_fcst_init_time`` is largest (i.e. the most recent run that still precedes the target
time). ``PowerForecast`` guarantees ``valid_time > power_fcst_init_time``, so this is an
analysis-proxy view: the shortest-lead hindcast of expected power, which is the natural baseline
for an observed-minus-expected residual.

Three files are written:

- ``*_full_ensemble.parquet`` — every ECMWF ensemble member (~51) of the freshest run.
- ``*_ensemble_mean.parquet`` — the ensemble mean of ``power_fcst`` per timestep.
- ``*_quantiles.parquet``     — the p10 / p50 / p90 of ``power_fcst`` across members per timestep.

Every file also carries ``observed_power`` (the metered value at that ``valid_time``, same MW/MVA
units), left-joined from the ``power_time_series`` table so residuals can be computed directly.

Run from a checkout where ``.env`` resolves (in a worktree, ``.env`` must be symlinked):

    uv run python scripts/export_forecasts_for_alex.py \
        --experiment-name xgboost_no_power_lags \
        --fold-id mid_2025_to_mid_2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from contracts.settings import PROJECT_ROOT, Settings
from contracts.typing_utils import typeddict_to_dict

# Columns carried through to the full-ensemble file (internal-only partition columns
# experiment_name / fold_id / ml_flow_experiment_id are intentionally dropped).
_FORECAST_COLUMNS: tuple[str, ...] = (
    "time_series_id",
    "valid_time",
    "power_fcst_init_time",
    "nwp_init_time",
    "ensemble_member",
    "power_fcst",
)


def _freshest_forecasts(
    power_forecasts_path: str,
    experiment_name: str,
    fold_id: str,
    storage_options: dict[str, str] | None,
) -> pl.LazyFrame:
    """Scan ``power_forecasts`` for one experiment/fold, keeping only the freshest run per timestep.

    ``experiment_name`` / ``fold_id`` are ``String`` partition columns, so both filters push down
    into the Delta scan (partition pruning). For each ``(time_series_id, valid_time)`` the run with
    the largest ``power_fcst_init_time`` is selected via a semi-join on the per-timestep max init
    time — this keeps *all* ensemble members of that run (the whole ensemble shares one init time).

    Args:
        power_forecasts_path: URI of the ``power_forecasts`` Delta table.
        experiment_name: Experiment whose forecasts to export.
        fold_id: CV fold whose forecasts to export.
        storage_options: Object-store options for ``pl.scan_delta``.

    Returns:
        A lazy frame of the freshest-run forecast rows, columns ``_FORECAST_COLUMNS``.
    """
    scan = pl.scan_delta(power_forecasts_path, storage_options=storage_options).filter(
        pl.col("experiment_name") == experiment_name,
        pl.col("fold_id") == fold_id,
    )
    freshest_init = scan.group_by("time_series_id", "valid_time").agg(
        power_fcst_init_time=pl.col("power_fcst_init_time").max()
    )
    return scan.join(
        freshest_init,
        on=["time_series_id", "valid_time", "power_fcst_init_time"],
        how="semi",
    ).select(_FORECAST_COLUMNS)


def _observed_power(
    power_time_series_path: str, storage_options: dict[str, str] | None
) -> pl.LazyFrame:
    """Scan observed metered power, deduplicated on the join key.

    The dedupe on ``(time_series_id, time)`` mirrors the metrics pipeline: a duplicated observation
    would double every ensemble member through the left-join. ``power`` is renamed to
    ``observed_power`` and is in the same MW/MVA units as the forecast.

    Args:
        power_time_series_path: URI of the ``power_time_series`` Delta table.
        storage_options: Object-store options for ``pl.scan_delta``.

    Returns:
        A lazy frame with columns ``time_series_id``, ``time``, ``observed_power``.
    """
    return (
        pl.scan_delta(power_time_series_path, storage_options=storage_options)
        .select("time_series_id", "time", "power")
        .unique(subset=["time_series_id", "time"], keep="any")
        .rename({"power": "observed_power"})
    )


def _with_observed(forecasts: pl.LazyFrame, observed: pl.LazyFrame) -> pl.LazyFrame:
    """Left-join ``observed_power`` onto forecasts on ``valid_time == time``.

    Both operands are plain (non-Patito) lazy frames, so Polars' cross-model join check does not
    fire. The observed ``time`` column is dropped after the join (it duplicates ``valid_time``).

    Args:
        forecasts: Forecast rows with a ``valid_time`` column.
        observed: Deduplicated observed power with ``time`` / ``observed_power`` columns.

    Returns:
        ``forecasts`` with an ``observed_power`` column added.
    """
    return forecasts.join(
        observed,
        left_on=["time_series_id", "valid_time"],
        right_on=["time_series_id", "time"],
        how="left",
    )


def export_forecasts(experiment_name: str, fold_id: str, output_dir: Path) -> dict[str, Path]:
    """Write the three parquet files for one experiment/fold and return their paths.

    Args:
        experiment_name: Experiment whose forecasts to export.
        fold_id: CV fold whose forecasts to export.
        output_dir: Directory to write the parquet files into (created if absent).

    Returns:
        A mapping of ``{"full_ensemble", "ensemble_mean", "quantiles"} -> written path``.
    """
    settings = Settings()
    storage_options = typeddict_to_dict(settings.storage_options)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{experiment_name}__{fold_id}"

    freshest = _freshest_forecasts(
        settings.power_forecasts_data_path, experiment_name, fold_id, storage_options
    )
    observed = _observed_power(settings.power_time_series_data_path, storage_options)
    full = _with_observed(freshest, observed)

    per_timestep = ["time_series_id", "valid_time"]
    ensemble_mean = full.group_by(per_timestep).agg(
        power_fcst_init_time=pl.col("power_fcst_init_time").first(),
        power_fcst_mean=pl.col("power_fcst").mean(),
        observed_power=pl.col("observed_power").first(),
    )
    quantiles = full.group_by(per_timestep).agg(
        power_fcst_init_time=pl.col("power_fcst_init_time").first(),
        power_fcst_p10=pl.col("power_fcst").quantile(0.1),
        power_fcst_p50=pl.col("power_fcst").quantile(0.5),
        power_fcst_p90=pl.col("power_fcst").quantile(0.9),
        observed_power=pl.col("observed_power").first(),
    )

    paths = {
        "full_ensemble": output_dir / f"{stem}_full_ensemble.parquet",
        "ensemble_mean": output_dir / f"{stem}_ensemble_mean.parquet",
        "quantiles": output_dir / f"{stem}_quantiles.parquet",
    }
    # Stream the (large) full-ensemble frame straight to disk; the aggregates are small.
    full.sink_parquet(paths["full_ensemble"])
    ensemble_mean.sort(per_timestep).collect(engine="streaming").write_parquet(
        paths["ensemble_mean"]
    )
    quantiles.sort(per_timestep).collect(engine="streaming").write_parquet(paths["quantiles"])
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="xgboost_no_power_lags")
    parser.add_argument("--fold-id", default="mid_2025_to_mid_2026")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "exports",
        help="Directory for the parquet files (default: <PROJECT_ROOT>/data/exports, gitignored).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = export_forecasts(args.experiment_name, args.fold_id, args.output_dir)
    for kind, path in paths.items():
        print(f"{kind:>14}: {path}")


if __name__ == "__main__":
    main()
