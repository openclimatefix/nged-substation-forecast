"""Stage 1: infer the physical parameters of the single-site solar BMUs, with capacity hidden.

For each of the 11 single-site BMUs that follow the sun, the script fits the forward model of
`studies.pv_physics` to the BMU's settled half-hourly output with CAMS irradiance at the BMU's own
point, in four arms: free tilt and azimuth, tilt and azimuth fixed at 30 and 180 degrees, a
single-axis tracker, and free orientation with clipping switched off (to test whether the AC
capacity is identified). Each arm is fitted on all the data and on each of four contiguous
three-month blocks held out in turn. The script also refits with the output shifted by up to two
half-hours each way, to test the timestamp alignment.

No fit sees a registered capacity. Results go to `stage1_*.parquet` in the study's data folder.
Run: `uv run python studies/solar_disaggregation/stage1_single_sites.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from envelope import envelope_ac_capacity_mw
from inputs import (
    OUTPUT_DIR,
    VALIDATION_BMUS,
    align_output,
    census_table,
    clear_sky_peak_w_m2,
    hourly_cams,
    output_by_bmu,
    settled_mask,
    site_cams_point,
    site_coordinates,
    sky_at,
)
from studies.pv_fit import OrientationType, fit_plant
from studies.pv_physics import SunAndSky, daylight, plant_power_mw

FOLD_OF_MONTH: Final[dict[int, int]] = {
    9: 0,
    10: 0,
    11: 0,
    12: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 3,
}
"""Four contiguous three-month blocks: autumn, winter, spring, summer."""
ARMS: Final[tuple[tuple[str, OrientationType, tuple[float, float]], ...]] = (
    ("free", "free", (1.0, 2.2)),
    ("fixed", "fixed", (1.0, 2.2)),
    ("tracker", "tracker", (1.0, 2.2)),
    ("free_no_clip", "free", (1.0, 1.001)),
)
"""Arm name, orientation, and DC:AC ratio bounds."""
OFFSETS: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2)
"""Half-hours to shift the output by in the timestamp-alignment test."""


def _fold_of_rows(*, sky: SunAndSky) -> np.ndarray:
    months = (
        pl.Series(sky.half_hour_end_time.astype("datetime64[us]")).dt.offset_by("-1m").dt.month()
    )
    return np.array([FOLD_OF_MONTH[month] for month in months.to_list()])


def _shifted(*, output_mw: np.ndarray, offset: int) -> np.ndarray:
    if offset == 0:
        return output_mw
    shifted = np.full_like(output_mw, np.nan)
    if offset > 0:
        shifted[offset:] = output_mw[:-offset]
    else:
        shifted[:offset] = output_mw[-offset:]
    return shifted


def _fit_row(*, bmu_id: str, arm: str, fold: int, offset: int, fit) -> dict:  # noqa: ANN001
    if fit is None:
        return {"bmu": bmu_id, "arm": arm, "fold": fold, "offset": offset, "loss": None}
    p = fit.parameters
    return {
        "bmu": bmu_id,
        "arm": arm,
        "fold": fold,
        "offset": offset,
        "loss": fit.loss,
        "tilt_deg": p.tilt_deg,
        "azimuth_deg": p.azimuth_deg,
        "dc_capacity_mw": p.dc_capacity_mw,
        "ac_capacity_mw": p.ac_capacity_mw,
        "dc_ac_ratio": p.dc_ac_ratio,
        "tracker": p.tracker,
        "half_hours_fitted": fit.half_hours_fitted,
    }


def run_bmu(*, bmu_id: str) -> tuple[list[dict], list[pl.DataFrame], dict]:
    """Fit every arm of one BMU, and return its fits, held-out predictions, and baselines.

    Args:
        bmu_id: The validation BMU's identifier.

    Returns:
        The fit rows (one per arm, fold, and timestamp offset), the held-out prediction frames, and
        the BMU's envelope baselines.
    """
    latitude, longitude = site_coordinates()[bmu_id]
    point = site_cams_point(bmu_id=bmu_id)
    sky = sky_at(hourly=hourly_cams(point_ids=[point]), latitude=latitude, longitude=longitude)
    output = align_output(sky=sky, output=output_by_bmu(bmu_id=bmu_id))
    settled = settled_mask(sky=sky, output_mw=output)
    folds = _fold_of_rows(sky=sky)
    day = daylight(sky=sky)
    fits: list[dict] = []
    predictions: list[pl.DataFrame] = []
    for arm, orientation, ratio_bounds in ARMS:
        full = fit_plant(
            sky=sky,
            output_mw=output,
            orientation=orientation,
            usable=settled,
            ratio_bounds=ratio_bounds,
        )
        fits.append(_fit_row(bmu_id=bmu_id, arm=arm, fold=-1, offset=0, fit=full))
        if arm == "free_no_clip":
            continue
        for fold in range(4):
            train = settled & (folds != fold)
            fit = fit_plant(
                sky=sky,
                output_mw=output,
                orientation=orientation,
                usable=train,
                ratio_bounds=ratio_bounds,
            )
            fits.append(_fit_row(bmu_id=bmu_id, arm=arm, fold=fold, offset=0, fit=fit))
            if fit is None:
                continue
            held_out = settled & (folds == fold) & day
            predicted = plant_power_mw(sky=sky, parameters=fit.parameters)
            predictions.append(
                pl.DataFrame(
                    {
                        "bmu": bmu_id,
                        "arm": arm,
                        "fold": fold,
                        "half_hour_end_time": sky.half_hour_end_time[held_out],
                        "output_mw": output[held_out],
                        "predicted_mw": predicted[held_out],
                    }
                )
            )
    for offset in OFFSETS:
        fit = fit_plant(
            sky=sky,
            output_mw=_shifted(output_mw=output, offset=offset),
            orientation="fixed",
            usable=settled,
        )
        fits.append(_fit_row(bmu_id=bmu_id, arm="offset_scan", fold=-1, offset=offset, fit=fit))
    judged = settled & np.isfinite(output)
    cosine = np.clip(np.cos(np.radians(sky.zenith_deg)), 0.0, None)
    cosine_shape = cosine / cosine[judged].max()
    cams_shape = sky.ghi_w_m2 / clear_sky_peak_w_m2(point_id=point)
    y = output[judged]
    baselines = {
        "bmu": bmu_id,
        "p99_output_mw": float(np.quantile(y, 0.99)),
        "max_output_mw": float(y.max()),
        "envelope_cosine_mw": envelope_ac_capacity_mw(shape=cosine_shape[judged], output_mw=y),
        "envelope_cams_mw": envelope_ac_capacity_mw(shape=cams_shape[judged], output_mw=y),
        "judged_half_hours": int(judged.sum()),
    }
    return fits, predictions, baselines


def main() -> None:
    """Fit every validation BMU and write the three result tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=11) as pool:
        results = list(pool.map(lambda_free_run, VALIDATION_BMUS))
    fits = pl.DataFrame([row for r in results for row in r[0]], infer_schema_length=None)
    predictions = pl.concat([frame for r in results for frame in r[1]])
    baselines = pl.DataFrame([r[2] for r in results])
    capacities = census_table().select(
        bmu="elexon_bmu_id",
        generation_capacity_mw="generation_capacity_mw",
        technology="technology",
        repd_installed_capacity_mw="repd_installed_capacity_mw",
        lccc_contract_capacity_mw="lccc_contract_capacity_mw",
    )
    baselines = baselines.join(capacities, on="bmu")
    fits.write_parquet(OUTPUT_DIR / "stage1_fits.parquet")
    predictions.write_parquet(OUTPUT_DIR / "stage1_predictions.parquet")
    baselines.write_parquet(OUTPUT_DIR / "stage1_baselines.parquet")
    print(fits.filter((pl.col("fold") == -1) & (pl.col("arm") == "free")))


def lambda_free_run(bmu_id: str) -> tuple[list[dict], list[pl.DataFrame], dict]:
    """Run one BMU (a module-level function, so that a process pool can pickle it).

    Args:
        bmu_id: The validation BMU's identifier.

    Returns:
        What `run_bmu` returns for the BMU.
    """
    return run_bmu(bmu_id=bmu_id)


if __name__ == "__main__":
    main()
