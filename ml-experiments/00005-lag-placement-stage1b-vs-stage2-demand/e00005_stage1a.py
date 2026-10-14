"""Fit the ERA5-trained stage 1a of experiment 00005 and write its out-of-fold ERA5 predictions.

For each configuration (training window), seed, and series, this fits:

- the final model, on ERA5 valid times from the window's start to before 2025-06-17;
- six out-of-fold models over the window up to 2025-07-01, each leaving out one time block and a
  15-day purge on each side, whose predictions of the held-out block are the "stage 1a at the
  observed weather" that stage 1b and the calibrators take residuals against;
- for configurations with calibrators, five forward-chain models, one per quarter from 2024-04,
  each trained on valid times that end 15 days before the quarter starts.

Output under `data/ml-experiments/00005-.../work/`:
`models/<config>/seed<s>/stage1a_<site>_<tag>.ubj`
and `era5_pred/<config>_seed<s>.npy` (one array per site on the dense half-hourly grid: the
out-of-fold prediction before 2025-07-01 and the final model's from then on). Existing files are
kept.

    uv run python ml-experiments/00005-lag-placement-stage1b-vs-stage2-demand/e00005_stage1a.py
"""

import itertools
import time
from pathlib import Path

import e00005_common as common
import numpy as np
import xgboost as xgb


def _save(*, model: xgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def _train_rows(
    *, power: np.ndarray, first: int, stop: int, excluded: tuple[int, int] | None = None
) -> np.ndarray:
    """Return grid indices in `[first, stop)` with observed power, minus an excluded span."""
    index = np.arange(first, stop)
    keep = np.isfinite(power[first:stop])
    if excluded is not None:
        keep &= ~((index >= excluded[0]) & (index < excluded[1]))
    return index[keep]


def fit_series(*, config: str, seed: int, site: str, arrays: common.SeriesArrays) -> np.ndarray:
    """Fit one series' stage 1a models for a configuration and seed; return its ERA5 predictions.

    Args:
        config: A key of `common.CONFIGS`.
        seed: The fitting seed.
        site: The series' label.
        arrays: `features` and `power` on the dense grid, from `load_series_arrays`.

    Returns:
        A float32 array on the grid: out-of-fold before 2025-07-01, the final model's after, and
        NaN before the window's start.
    """
    settings = common.CONFIGS[config]
    features, power = arrays.features, arrays.power
    start = int(common.grid_index(times=settings.start))
    cutoff = int(common.grid_index(times=common.FINAL_TRAIN_CUTOFF))
    validation = int(common.grid_index(times=common.VALIDATION_START))
    purge = int(common.PURGE / common.STEP)

    final_rows = _train_rows(power=power, first=start, stop=cutoff)
    final = common.fit_booster(features=features[final_rows], target=power[final_rows], seed=seed)
    _save(
        model=final,
        path=common.model_path(config=config, seed=seed, kind="stage1a", site=site, tag="final"),
    )
    prediction = np.full(power.shape, np.nan, dtype=np.float32)
    prediction[validation:] = common.predict(model=final, features=features[validation:])
    if not settings.stage1b:
        return prediction

    edges = np.linspace(start, validation, common.BLOCKS + 1).astype(int)
    for first, last in itertools.pairwise(edges):
        rows = _train_rows(
            power=power, first=start, stop=cutoff, excluded=(first - purge, last + purge)
        )
        model = common.fit_booster(features=features[rows], target=power[rows], seed=seed)
        prediction[first:last] = common.predict(model=model, features=features[first:last])
    if settings.calibrators:
        for tag, quarter_start in enumerate(common.CHAIN_STARTS):
            stop = int(common.grid_index(times=quarter_start - common.PURGE))
            rows = _train_rows(power=power, first=start, stop=stop)
            chain = common.fit_booster(features=features[rows], target=power[rows], seed=seed)
            _save(
                model=chain,
                path=common.model_path(
                    config=config, seed=seed, kind="stage1a", site=site, tag=f"q{tag}"
                ),
            )
    return prediction


def main() -> None:
    """Fit every configuration, seed, and series, skipping outputs that exist."""
    series = common.series_table()
    for config in common.CONFIGS:
        targets = {seed: common.era5_pred_path(config=config, seed=seed) for seed in common.SEEDS}
        if all(path.exists() for path in targets.values()):
            continue
        arrays = common.load_series_arrays(series=series, config=config)
        for seed, target in targets.items():
            if target.exists():
                continue
            started = time.monotonic()
            predictions = {
                site: fit_series(config=config, seed=seed, site=site, arrays=data)
                for site, data in arrays.items()
            }
            common.write_era5_pred(config=config, seed=seed, predictions=predictions)
            print(f"{config} seed {seed}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
