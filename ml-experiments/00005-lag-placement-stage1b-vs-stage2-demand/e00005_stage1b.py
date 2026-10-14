"""Fit stage 1b of experiment 00005: residual persistence from simulated forecast origins.

Stage 1b is one XGBoost model per series, trained on the ERA5 history. For each target half-hour it
draws four leads uniformly from 0.5 to 351 hours, which gives the training rows the full lead
range, and places the forecast origin that lead before the target. The target is the stage 1a
residual at the target time (observed power over effective capacity minus the out-of-fold stage 1a
at the observed weather). The inputs are the lead and `e00005_common.RECENT_COLUMNS`, which use
only observations before the origin. No calendar column enters.

The final model trains on targets up to 2025-06-30 23:30. For configurations with calibrators, a
forward chain of five models trains on targets that end 15 days before each quarter from 2024-04.
Models land beside stage 1a's in `models/<config>/seed<s>/stage1b_<site>_<tag>.ubj`.

    uv run python ml-experiments/00005-lag-placement-stage1b-vs-stage2-demand/e00005_stage1b.py
"""

import time
from datetime import datetime

import e00005_common as common
import numpy as np

LEAD_COLUMN: str = "lead_hours"


def residuals(*, power: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return observed minus predicted power over effective capacity, NaN where either is missing.

    Args:
        power: Observed power over effective capacity on the grid.
        prediction: Stage 1a's prediction on the grid.

    Returns:
        The residual on the grid.
    """
    return (power - prediction).astype(np.float32)


def stage1b_matrix(
    *, power: np.ndarray, residual: np.ndarray, target: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Return stage 1b's inputs: the lead in hours, then the recent-origin inputs.

    Args:
        power: Observed power over effective capacity on the grid.
        residual: Stage 1a residuals on the grid.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, 1 + len(RECENT_COLUMNS)).
    """
    lead_hours = ((target - origin) * 0.5).astype(np.float32)[:, None]
    recent = common.recent_origin_inputs(
        power=power, residual=residual, target=target, origin=origin
    )
    return np.concatenate([lead_hours, recent], axis=1)


def training_cases(
    *, residual: np.ndarray, first: int, stop: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the simulated (target, origin) pairs of targets in `[first, stop)` with a residual.

    Args:
        residual: Stage 1a residuals on the grid.
        first: First target's grid index.
        stop: One past the last target's grid index.
        seed: The draw's seed.

    Returns:
        The target and origin grid indices, `LEADS_PER_TARGET` cases per target.
    """
    targets = np.arange(first, stop)
    targets = targets[np.isfinite(residual[first:stop])]
    rng = np.random.default_rng(seed)
    repeated = np.repeat(targets, common.LEADS_PER_TARGET)
    leads = rng.integers(1, common.MAX_LEAD_STEPS + 1, size=repeated.size)
    return repeated, repeated - leads


def fit_series(
    *, config: str, seed: int, site: str, power: np.ndarray, prediction: np.ndarray
) -> None:
    """Fit and save one series' final and forward-chain stage 1b models.

    Args:
        config: A key of `common.CONFIGS`.
        seed: The fitting seed.
        site: The series' label.
        power: Observed power over effective capacity on the grid.
        prediction: Stage 1a's ERA5 predictions on the grid (out of fold before 2025-07-01).
    """
    settings = common.CONFIGS[config]
    residual = residuals(power=power, prediction=prediction)
    first = int(common.grid_index(times=settings.start)) + max(common.LAG_STEPS)
    stops = {"final": int(common.grid_index(times=common.LAST_TRAINING_VALID)) + 1}
    if settings.calibrators:
        for tag, quarter_start in enumerate(common.CHAIN_STARTS):
            stops[f"q{tag}"] = int(common.grid_index(times=quarter_start - common.PURGE))
    for tag, stop in stops.items():
        target, origin = training_cases(residual=residual, first=first, stop=stop, seed=seed)
        matrix = stage1b_matrix(power=power, residual=residual, target=target, origin=origin)
        model = common.fit_booster(features=matrix, target=residual[target], seed=seed)
        path = common.model_path(config=config, seed=seed, kind="stage1b", site=site, tag=tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(path)


def main() -> None:
    """Fit stage 1b for every configuration that has it, skipping a seed whose marker exists."""
    series = common.series_table()
    for config, settings in common.CONFIGS.items():
        if not settings.stage1b:
            continue
        arrays = common.load_series_arrays(series=series, config=config)
        for seed in common.SEEDS:
            marker = common.WORK_DIR / "models" / config / f"seed{seed}" / "stage1b.done"
            if marker.exists():
                continue
            started = time.monotonic()
            predictions = common.read_era5_pred(config=config, seed=seed, sites=list(arrays))
            for site, data in arrays.items():
                fit_series(
                    config=config,
                    seed=seed,
                    site=site,
                    power=data.power,
                    prediction=predictions[site],
                )
            marker.write_text(datetime.now().astimezone().isoformat())
            print(
                f"{config} seed {seed}: stage 1b in {time.monotonic() - started:.0f} s", flush=True
            )


if __name__ == "__main__":
    main()
