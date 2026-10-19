"""Fit X-mva: lagless XGBoost on `|S|` at each MVA substation, with and without the filter.

X-mva is Part 4's `X-best` recipe (experiment 87's Primary XGBoost hyperparameters without the three
power lags, 11 quantile levels) applied to the apparent-power reading of an MVA substation, with no
solar offset because an MVA substation has no capacity vintage. The training window is the one
Part 4 chose for `X-best` on its tuning year. X-mva is the XGBoost comparator of the meter arms
(exploratory contrast L0 minus X-mva).

- **Unfiltered:** XGBoost's own 11 quantiles.
- **Filtered:** every quantile shifted by the Kalman filter's correction on the median's residuals
  of the same scored year, with the filter's hyperparameters fitted by marginal likelihood on the
  2023-24 residuals of the MVA substations under a shared prior, as for the other arms.

    uv run python ml-experiments/00110-*/e00110_xgb.py [--workers 4]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00110_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    correction_at,
    day_ahead_initialisation,
    fit_filter_hyperparameters,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS, TrainingWindow, training_rows

N_ROUNDS: Final[int] = 500
XGB_PARAMETERS: Final[dict[str, object]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "objective": "reg:quantileerror",
    "seed": 0,
    "nthread": 1,
}
"""Experiment 87's N1 parameters; the quantile levels are added per fit."""
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "temperature_c",
    "temperature_c_k6",
    "temperature_c_k24",
    "temperature_c_k72",
    "temperature_c_k168",
    "ghi_w_m2",
    "elevation_deg",
    "half_hour",
    "day_type",
    "holiday_period",
    "tent",
    "signed_tent",
    "day_of_year",
)
WINDOW_MONTHS: Final[dict[str, int | None]] = {"12m": 12, "24m": 24, "all": None}


def feature_matrix(*, cache: FeatureCache, site: str, index: np.ndarray) -> np.ndarray:
    """Return the lagless inputs of the rows `index`, float32, one column per feature."""
    features = cache.sites[site]
    calendar = cache.calendar
    columns = {
        "temperature_c": features.temperature_c[index],
        "temperature_c_k6": features.smoothed_temperature_c[index, 0],
        "temperature_c_k24": features.smoothed_temperature_c[index, 1],
        "temperature_c_k72": features.smoothed_temperature_c[index, 2],
        "temperature_c_k168": features.smoothed_temperature_c[index, 3],
        "ghi_w_m2": features.ghi_w_m2[index],
        "elevation_deg": features.elevation_deg[index],
        "half_hour": calendar.half_hour[index],
        "day_type": calendar.day_type[index],
        "holiday_period": calendar.holiday_period[index],
        "tent": calendar.tent[index],
        "signed_tent": calendar.signed_tent[index],
        "day_of_year": calendar.day_of_year[index],
    }
    return np.column_stack([columns[name] for name in FEATURE_COLUMNS]).astype(np.float32)


def quantiles_at_cut(
    *, cache: FeatureCache, site: str, label: str, window: TrainingWindow
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the booster at a cut and return the scored rows and their 11 quantiles.

    Args:
        cache: The feature cache.
        site: The substation's label.
        label: The cut label.
        window: The training window.

    Returns:
        The scored grid rows and the sorted quantiles, shape (rows, 11).
    """
    features = cache.sites[site]
    cut = common.CUTS[label]
    train = training_rows(features=features, cache=cache, cut=cut, window=window)
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    parameters: dict[str, object] = {**XGB_PARAMETERS, "quantile_alpha": list(QUANTILE_LEVELS)}
    booster = xgb.train(
        parameters,
        xgb.DMatrix(feature_matrix(cache=cache, site=site, index=train), label=features.y[train]),
        num_boost_round=N_ROUNDS,
    )
    raw = booster.predict(xgb.DMatrix(feature_matrix(cache=cache, site=site, index=index)))
    return index, np.sort(raw.reshape(len(index), len(QUANTILE_LEVELS)), axis=1)


def _site_job(job: tuple[str, str]) -> dict[str, Any]:
    """Fit all three cuts at one site and return the unfiltered frames and residual streams."""
    site, window_name = job
    import torch

    torch.set_num_threads(1)
    cache = common.load_cache()
    window = TrainingWindow(months=WINDOW_MONTHS[window_name])
    out: dict[str, Any] = {"site": site, "frames": {}, "streams": {}}
    for label in common.CUTS:
        try:
            index, quantiles = quantiles_at_cut(cache=cache, site=site, label=label, window=window)
        except ValueError:
            continue
        stream = np.full(len(cache.end_times), np.nan)
        median = quantiles[:, QUANTILE_LEVELS.index(0.5)]
        stream[index] = cache.sites[site].y[index] - median
        out["streams"][label] = stream
        if label != "tuning":
            frame = common.prediction_frame(
                cache=cache,
                site=site,
                index=index,
                quantiles=quantiles,
                mean=median,
                arm="x_mva",
                filtered=False,
                extra={"dark": cache.sites[site].dark[index]},
            )
            out["frames"][label] = frame
    return out


def filtered_frame(
    *, cache: FeatureCache, base: pl.DataFrame, stream: np.ndarray, hyper: FilterHyperparameters
) -> pl.DataFrame:
    """Return the base frame with every quantile shifted by the Kalman correction."""
    index = np.searchsorted(
        cache.end_times,
        base["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]"),
    )
    trace = run_kalman(residual=stream, band=cache.calendar.band, hyper=hyper)
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=day_ahead_initialisation(end_times=cache.end_times[index], days_ahead=1),
        delay_hours=DEFAULT_DELAY_HOURS,
    )
    correction = correction_at(
        trace=trace, cutoff=cutoff, target=index, band=cache.calendar.band[index]
    )
    shift = pl.Series(correction.total.astype(np.float32))
    return base.with_columns(
        *[(pl.col(column) + shift).alias(column) for column in QUANTILE_COLUMNS],
        mean=pl.col("mean") + shift,
        filtered=pl.lit(True),
        level=pl.Series(correction.level.astype(np.float32)),
        anomaly=pl.Series(correction.anomaly.astype(np.float32)),
    )


def main() -> None:
    """Fit X-mva at every MVA substation and write the sets `<cut>_x_mva`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    chosen = json.loads((common.PART_4_DIR / "work" / "tuned_xgb.json").read_text())["window"]
    cache = common.load_cache()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(_site_job, [(site, chosen) for site in common.MVA_SITES]))
    tuning_window = (
        common.to_64(moment=common.TUNING_CUT),
        common.to_64(moment=common.year_end(cut=common.TUNING_CUT)),
    )
    fitted = fit_filter_hyperparameters(
        residuals={r["site"]: r["streams"]["tuning"] for r in results if "tuning" in r["streams"]},
        band=cache.calendar.band,
        end_times=cache.end_times,
        window=tuning_window,
    )
    mean_hyper = np.mean([h.as_array() for h in fitted.values()], axis=0)
    for label in ("replicate", "scored"):
        frames = []
        for result in results:
            if label not in result["frames"]:
                continue
            hyper = fitted.get(result["site"], FilterHyperparameters.from_array(values=mean_hyper))
            base = result["frames"][label].with_columns(
                level=pl.lit(0.0, dtype=pl.Float32), anomaly=pl.lit(0.0, dtype=pl.Float32)
            )
            frames += [
                base,
                filtered_frame(
                    cache=cache,
                    base=base,
                    stream=result["streams"][label],
                    hyper=hyper,
                ),
            ]
        folder = common.store_arm(
            name=f"{label}_x_mva", frame=pl.concat(frames), row_set_id=common.MVA_ROW_SET_ID
        )
        print(f"wrote {folder}")


if __name__ == "__main__":
    main()
