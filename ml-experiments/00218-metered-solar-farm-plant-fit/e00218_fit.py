"""Fit the arms of experiment 218 on the tuning months and predict every row of each farm.

Stage 1 searches the grid of tilt, azimuth, and heat coefficient for every (fold, farm), fitting a
scale (and, with the clip, an AC limit) at each grid point. Stage 2 picks the arms from the grid
table, fits the learned response (arm 3), and writes predictions. Run it under
`systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue` with `uv run --no-sync python`.
"""

import itertools
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00218_common as common
import numpy as np
import polars as pl
from studies.pv_physics import SunAndSky, plane_of_array_w_m2
from studies.pv_plant_fit import (
    ResponseGrid,
    block_folds,
    fit_response_table,
    fit_scale_and_clip,
    fit_scale_only,
    plant_output,
    unclipped_unit_power,
)

N_PROCESSES: Final[int] = 4
TILTS_DEG: Final[tuple[float, ...]] = (10.0, 20.0, 30.0, 35.0, 40.0, 50.0, 60.0)
AZIMUTHS_DEG: Final[tuple[float, ...]] = (120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0)
HEAT_PER_KELVIN: Final[tuple[float, ...]] = (0.0, 0.002, 0.004, 0.006)
FLEET_TILT_DEG: Final[float] = 35.0
FLEET_AZIMUTH_DEG: Final[float] = 180.0
FLEET_HEAT_PER_KELVIN: Final[float] = 0.004
RESPONSE_GRID: Final[ResponseGrid] = ResponseGrid(
    irradiance_w_m2=np.arange(0.0, 1101.0, 50.0), elevation_deg=np.arange(5.0, 66.0, 5.0)
)
PENALTIES: Final[tuple[float, ...]] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CV_FOLDS: Final[int] = 5
GRID_PATH: Final[Path] = common.DATA_DIR / "grid_table.parquet"
PREDICTIONS_PATH: Final[Path] = common.DATA_DIR / "predictions.parquet"
PARAMETERS_PATH: Final[Path] = common.DATA_DIR / "arm_parameters.parquet"


@dataclass(frozen=True)
class FarmData:
    """One farm's rows split into the fold's tuning and scored parts."""

    frame: pl.DataFrame
    tuning: np.ndarray
    scored: np.ndarray

    def sky(self) -> SunAndSky:
        """Return the sun and sky of every row."""
        return SunAndSky(
            half_hour_end_time=self.frame["time"].to_numpy(),
            ghi_w_m2=self.frame["ghi_w_m2"].to_numpy(),
            dni_w_m2=self.frame["dni_w_m2"].to_numpy(),
            dhi_w_m2=self.frame["dhi_w_m2"].to_numpy(),
            zenith_deg=self.frame["zenith_deg"].to_numpy(),
            sun_azimuth_deg=self.frame["sun_azimuth_deg"].to_numpy(),
        )


def farm_data(*, rows: pl.DataFrame, site: str, fold: int) -> FarmData | None:
    """Return a farm's rows for a fold, or None when it has too few tuning months."""
    frame = rows.filter(pl.col("site") == site).sort("time")
    start = common.FOLD_SCORED_STARTS[fold]
    end = common.scored_end(fold=fold)
    before = (frame["time"] < start).to_numpy()
    scored = ((frame["time"] >= start) & (frame["time"] < end)).to_numpy()
    n_months = frame.filter(pl.col("time") < start)["month_index"].n_unique()
    if n_months < common.MIN_TUNING_MONTHS or not scored.any():
        return None
    return FarmData(frame=frame, tuning=before, scored=scored)


def _grid_rows(*, fold: int, site: str) -> list[dict[str, float | int | str]]:
    """Search the whole grid for one farm and fold."""
    rows = pl.read_parquet(common.ROWS_PATH)
    data = farm_data(rows=rows, site=site, fold=fold)
    if data is None:
        return []
    sky = data.sky()
    output = data.frame["output"].to_numpy().astype(np.float64)
    temperature = data.frame["temperature_c"].to_numpy()
    tune = data.tuning
    out = []
    for tilt, azimuth in itertools.product(TILTS_DEG, AZIMUTHS_DEG):
        poa = plane_of_array_w_m2(sky=sky, tilt_deg=tilt, azimuth_deg=azimuth, tracker=False)
        for heat in HEAT_PER_KELVIN:
            unclipped = unclipped_unit_power(
                poa_w_m2=poa, temperature_c=temperature, heat_per_kelvin=heat
            )
            clip = fit_scale_and_clip(unclipped=unclipped[tune], output=output[tune])
            plain = fit_scale_only(unclipped=unclipped[tune], output=output[tune])
            out.append(
                {
                    "fold": fold,
                    "site": site,
                    "tilt_deg": tilt,
                    "azimuth_deg": azimuth,
                    "heat_per_kelvin": heat,
                    "scale_clip": clip.scale,
                    "ac_limit": clip.ac_limit,
                    "sse_clip": clip.squared_error,
                    "scale_plain": plain.scale,
                    "sse_plain": plain.squared_error,
                    "tuning_sum_squares": float(np.sum(output[tune] ** 2)),
                    "n_tuning": int(tune.sum()),
                }
            )
    return out


def _grid_task(task: tuple[int, str]) -> list[dict[str, float | int | str]]:
    return _grid_rows(fold=task[0], site=task[1])


def build_grid_table() -> pl.DataFrame:
    """Run stage 1 over every (fold, farm) in parallel and write the grid table."""
    tasks = list(itertools.product(sorted(common.FOLD_SCORED_STARTS), common.FARMS))
    with multiprocessing.Pool(N_PROCESSES) as pool:
        results = pool.map(_grid_task, tasks)
    table = pl.DataFrame([row for part in results for row in part])
    table.write_parquet(GRID_PATH)
    return table


def _pick(*, table: pl.DataFrame, column: str) -> dict[str, float]:
    """Return the grid point with the smallest `column`."""
    return table.sort(column).row(0, named=True)


def _predict_clip(*, data: FarmData, point: dict[str, float]) -> np.ndarray:
    poa = plane_of_array_w_m2(
        sky=data.sky(), tilt_deg=point["tilt_deg"], azimuth_deg=point["azimuth_deg"], tracker=False
    )
    unclipped = unclipped_unit_power(
        poa_w_m2=poa,
        temperature_c=data.frame["temperature_c"].to_numpy(),
        heat_per_kelvin=point["heat_per_kelvin"],
    )
    return plant_output(unclipped=unclipped, scale=point["scale_clip"], ac_limit=point["ac_limit"])


def _predict_plain(*, data: FarmData, point: dict[str, float]) -> np.ndarray:
    poa = plane_of_array_w_m2(
        sky=data.sky(), tilt_deg=point["tilt_deg"], azimuth_deg=point["azimuth_deg"], tracker=False
    )
    unclipped = unclipped_unit_power(
        poa_w_m2=poa,
        temperature_c=data.frame["temperature_c"].to_numpy(),
        heat_per_kelvin=point["heat_per_kelvin"],
    )
    return point["scale_plain"] * unclipped


def _learned_response(*, data: FarmData, prior: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit arm 3 on the tuning rows and return its prediction for every row and the penalty."""
    frame = data.frame
    ghi = frame["ghi_w_m2"].to_numpy()
    elevation = frame["elevation_deg"].to_numpy()
    output = frame["output"].to_numpy().astype(np.float64)
    tune = data.tuning
    blocks = frame["month_index"].to_numpy()[tune]
    folds = block_folds(block_of_row=blocks, n_folds=CV_FOLDS)
    scores = {}
    for penalty in PENALTIES:
        error = 0.0
        for held in folds:
            train = ~held
            table = fit_response_table(
                grid=RESPONSE_GRID,
                irradiance_w_m2=ghi[tune][train],
                elevation_deg=elevation[tune][train],
                output=output[tune][train],
                prior_output=prior[tune][train],
                penalty=penalty,
            )
            predicted = table.predict(
                irradiance_w_m2=ghi[tune][held], elevation_deg=elevation[tune][held]
            )
            error += float(np.sum((output[tune][held] - predicted) ** 2))
        scores[penalty] = error
    best = min(scores, key=scores.__getitem__)
    table = fit_response_table(
        grid=RESPONSE_GRID,
        irradiance_w_m2=ghi[tune],
        elevation_deg=elevation[tune],
        output=output[tune],
        prior_output=prior[tune],
        penalty=best,
    )
    return table.predict(irradiance_w_m2=ghi, elevation_deg=elevation), best


def _lofo_point(*, grid: pl.DataFrame, site: str) -> dict[str, float]:
    """Return the farm's own fit at the geometry and heat pooled from the other farms."""
    keys = ["tilt_deg", "azimuth_deg", "heat_per_kelvin"]
    others = (
        grid.filter(pl.col("site") != site)
        .with_columns(share=pl.col("sse_clip") / pl.col("tuning_sum_squares"))
        .group_by(keys)
        .agg(pl.col("share").sum())
        .sort("share")
        .row(0, named=True)
    )
    own = grid.filter(pl.col("site") == site)
    for key in keys:
        own = own.filter(pl.col(key) == others[key])
    return own.row(0, named=True)


def fit_arms(*, grid_table: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Pick every arm per (fold, farm) and predict all of the farm's rows.

    Args:
        grid_table: The stage 1 grid table.

    Returns:
        The predictions (one row per fold, farm, and half-hour with one column per arm) and the
        chosen parameters (one row per fold, farm, and arm).
    """
    rows = pl.read_parquet(common.ROWS_PATH)
    predictions = []
    parameters = []
    for fold in sorted(common.FOLD_SCORED_STARTS):
        fold_grid = grid_table.filter(pl.col("fold") == fold)
        for site in common.FARMS:
            data = farm_data(rows=rows, site=site, fold=fold)
            if data is None:
                continue
            grid = fold_grid.filter(pl.col("site") == site)
            fleet = grid.filter(
                (pl.col("tilt_deg") == FLEET_TILT_DEG)
                & (pl.col("azimuth_deg") == FLEET_AZIMUTH_DEG)
                & (pl.col("heat_per_kelvin") == FLEET_HEAT_PER_KELVIN)
            ).row(0, named=True)
            chosen = {
                "arm1_fleet_shape": (fleet, _predict_plain),
                "arm2_per_farm": (_pick(table=grid, column="sse_clip"), _predict_clip),
                "arm4_heat_off": (
                    _pick(table=grid.filter(pl.col("heat_per_kelvin") == 0.0), column="sse_clip"),
                    _predict_clip,
                ),
                "ablation_no_clip": (_pick(table=grid, column="sse_plain"), _predict_plain),
                "ablation_fleet_geometry": (
                    _pick(
                        table=grid.filter(
                            (pl.col("tilt_deg") == FLEET_TILT_DEG)
                            & (pl.col("azimuth_deg") == FLEET_AZIMUTH_DEG)
                        ),
                        column="sse_clip",
                    ),
                    _predict_clip,
                ),
                "arm2_pooled_geometry": (
                    _lofo_point(grid=fold_grid, site=site),
                    _predict_clip,
                ),
            }
            columns = {}
            for arm, (point, predictor) in chosen.items():
                columns[arm] = predictor(data=data, point=point)
                parameters.append(
                    {
                        "fold": fold,
                        "site": site,
                        "arm": arm,
                        "tilt_deg": point["tilt_deg"],
                        "azimuth_deg": point["azimuth_deg"],
                        "heat_per_kelvin": point["heat_per_kelvin"],
                        "scale": point[
                            "scale_plain"
                            if arm in {"arm1_fleet_shape", "ablation_no_clip"}
                            else "scale_clip"
                        ],
                        "ac_limit": None
                        if arm in {"arm1_fleet_shape", "ablation_no_clip"}
                        else point["ac_limit"],
                    }
                )
            columns["arm3_learned_response"], penalty = _learned_response(
                data=data, prior=columns["arm2_per_farm"]
            )
            parameters.append(
                {"fold": fold, "site": site, "arm": "arm3_learned_response", "penalty": penalty}
            )
            predictions.append(
                data.frame.select(
                    "site", "time", "month", "month_index", "output", "clear_sky", "temperature_c"
                ).with_columns(
                    fold=pl.lit(fold),
                    scored=pl.Series(data.scored),
                    **{arm: pl.Series(values) for arm, values in columns.items()},
                )
            )
            print(f"fold {fold} farm {site} done", flush=True)
    return pl.concat(predictions), pl.DataFrame(parameters, infer_schema_length=None)


def main() -> None:
    """Run both stages, reusing a grid table already on disk."""
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid_table = pl.read_parquet(GRID_PATH) if GRID_PATH.exists() else build_grid_table()
    predictions, parameters = fit_arms(grid_table=grid_table)
    predictions.write_parquet(PREDICTIONS_PATH)
    parameters.write_parquet(PARAMETERS_PATH)
    print(parameters.filter(pl.col("fold") == 1).sort("site", "arm"))


if __name__ == "__main__":
    main()
