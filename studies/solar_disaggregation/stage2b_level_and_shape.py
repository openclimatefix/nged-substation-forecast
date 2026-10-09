"""Stage 2b: does the separation recover the solar level, the solar shape, or both?

An exploratory follow-up to `stage2_synthetic_separation.py`, run because that stage's recovered
solar energy was well below the true energy. For the planned solar halves and the six non-solar
halves, and for each of six strengths of the ridge penalty on the calendar baseline, the script
records the energy ratio (recovered over true), the correlation of the recovered series with the
true series over daylight half-hours, the error before and after rescaling the recovered series to
the true daylight energy, and the false solar a no-solar aggregate gets.

Run: `uv run python studies/solar_disaggregation/stage2b_level_and_shape.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from inputs import (
    OUTPUT_DIR,
    VALIDATION_BMUS,
    nearest_grid_points,
    on_grid,
    output_on_grid,
    window_half_hours,
)
from stage2_synthetic_separation import (
    AGGREGATE_P99_MW,
    NEAREST_POINTS,
    NON_SOLAR_BMUS,
    SOLAR_SETS,
    centroid_of,
    regressor_for,
    stage1_ratio,
)
from studies.pv_separation import baseline_design, separate

RIDGES: Final[tuple[float, ...]] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
"""The ridge penalty on the baseline, relative to the mean squared norm of the design's columns.
The value 1e-3 is the one every other stage uses."""
SHARES: Final[tuple[float, ...]] = (0.0, 0.1, 0.25, 0.5)


def run_set(set_name: str) -> list[dict]:
    """Score every non-solar half, share, and ridge for one solar set.

    Args:
        set_name: A key of `SOLAR_SETS`.

    Returns:
        One row per non-solar half, share, and ridge penalty that the separation fits.
    """
    grid = window_half_hours()
    members = SOLAR_SETS[set_name]
    solar_p99 = {b: float(np.nanquantile(output_on_grid(bmu_id=b), 0.99)) for b in VALIDATION_BMUS}
    ratio = stage1_ratio(excluded=members)
    latitude, longitude = centroid_of(bmus=members, weights=solar_p99)
    points = nearest_grid_points(latitude=latitude, longitude=longitude, count=NEAREST_POINTS)
    regressor = regressor_for(point_ids=points, latitude=latitude, longitude=longitude, ratio=ratio)
    daylight = on_grid(
        half_hour_end_time=regressor.sky.half_hour_end_time,
        values=((90.0 - regressor.sky.zenith_deg) > 5.0).astype(float),
    )
    design = baseline_design(half_hour_end_time=grid, flexibility="seasonal")
    solar_raw = np.nansum([output_on_grid(bmu_id=b) for b in members], axis=0)
    solar_raw[np.any([~np.isfinite(output_on_grid(bmu_id=b)) for b in members], axis=0)] = np.nan
    rows: list[dict] = []
    for other_name, other_bmu in NON_SOLAR_BMUS.items():
        other_raw = output_on_grid(bmu_id=other_bmu)
        for share in SHARES:
            solar = solar_raw * (share * AGGREGATE_P99_MW / np.nanquantile(solar_raw, 0.99))
            other = other_raw * (
                (1.0 - share) * AGGREGATE_P99_MW / np.nanquantile(np.abs(other_raw), 0.99)
            )
            y = solar + other
            valid = np.isfinite(y) & np.isfinite(solar) & np.isfinite(regressor.basis).all(axis=1)
            day = valid & (np.nan_to_num(daylight) > 0)
            for ridge in RIDGES:
                fit = separate(output_mw=y, basis=regressor.basis, design=design, ridge=ridge)
                if fit is None:
                    continue
                estimate = fit.solar_mw
                row: dict[str, object] = {
                    "solar_set": set_name,
                    "non_solar": other_name,
                    "share": share,
                    "ridge": ridge,
                    "estimate_p99_mw": float(np.quantile(estimate[valid], 0.99)),
                }
                if share > 0:
                    truth_energy = float(solar[day].sum())
                    estimate_energy = float(estimate[day].sum())
                    truth_p99 = float(np.quantile(solar[valid], 0.99))
                    row |= {
                        "energy_ratio": float(estimate[valid].sum() / solar[valid].sum()),
                        "correlation": float(np.corrcoef(estimate[day], solar[day])[0, 1])
                        if estimate[day].std() > 0
                        else None,
                        "nmae": float(np.abs(estimate - solar)[day].mean() / truth_p99),
                        "nmae_rescaled": float(
                            np.abs(estimate * (truth_energy / estimate_energy) - solar)[day].mean()
                            / truth_p99
                        )
                        if estimate_energy > 0
                        else None,
                    }
                rows.append(row)
    return rows


def main() -> None:
    """Run the four planned solar sets in parallel and write `stage2b_level_and_shape.parquet`."""
    with ProcessPoolExecutor(max_workers=len(SOLAR_SETS)) as pool:
        results = list(pool.map(run_set, list(SOLAR_SETS)))
    frame = pl.DataFrame([row for rows in results for row in rows], infer_schema_length=None)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUTPUT_DIR / "stage2b_level_and_shape.parquet")
    print(
        frame.filter(pl.col("share") > 0)
        .group_by("ridge")
        .agg(
            pl.col("energy_ratio").mean(),
            pl.col("correlation").mean(),
            pl.col("nmae").mean(),
            pl.col("nmae_rescaled").mean(),
        )
        .sort("ridge")
    )
    print(
        frame.filter(pl.col("share") == 0)
        .group_by("ridge")
        .agg(
            pl.col("estimate_p99_mw").mean().alias("mean_estimate_p99_mw"),
            pl.col("estimate_p99_mw").max().alias("max_estimate_p99_mw"),
        )
        .sort("ridge")
    )


if __name__ == "__main__":
    main()
