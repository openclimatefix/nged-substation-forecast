"""Stage 1b: can the physical fit recover known parameters? A positive control.

Plants with known tilt, azimuth, DC:AC ratio, and AC capacity are simulated at the real sites'
positions with a different irradiance chain from the one the fit assumes: the DISC decomposition
(the fit uses Erbs), the Hay-Davies transposition (the fit is isotropic), and a cell-temperature
derate (the fit has none). The noise is the site's real relative residual from its stage 1 fit,
shifted by a random number of whole days, so it has the real weather-correlated structure. The
simulated output is refitted with the free-orientation arm, and each parameter's error is recorded.

Run: `uv run python studies/solar_disaggregation/stage1b_synthetic_recovery.py`.
"""

from typing import Final

import numpy as np
import polars as pl
import pvlib
from inputs import (
    OUTPUT_DIR,
    VALIDATION_BMUS,
    align_output,
    hourly_cams,
    output_by_bmu,
    settled_mask,
    site_cams_point,
    site_coordinates,
    sky_at,
)
from studies.pv_fit import fit_plant
from studies.pv_physics import PlantParameters, SunAndSky, daylight, plant_power_mw

PLANTS_PER_SITE: Final[int] = 4
SEED: Final[int] = 20261008
TEMPERATURE_COEFFICIENT_PER_K: Final[float] = -0.004
CELL_RISE_K_PER_W_M2: Final[float] = 0.03


def _alternative_power_mw(*, sky: SunAndSky, parameters: PlantParameters) -> np.ndarray:
    """Return power from the alternative chain: DISC, Hay-Davies, and a temperature derate."""
    time = pl.Series(sky.half_hour_end_time.astype("datetime64[us]")).dt.offset_by("-15m")
    doy = time.dt.ordinal_day().to_numpy()
    hour = (time.dt.hour() + time.dt.minute() / 60).to_numpy()
    cos_zenith = np.clip(np.cos(np.radians(sky.zenith_deg)), 0.0, None)
    dni = np.nan_to_num(
        np.asarray(
            pvlib.irradiance.disc(
                ghi=sky.ghi_w_m2, solar_zenith=sky.zenith_deg, datetime_or_doy=doy
            )["dni"]
        )
    )
    dhi = np.clip(sky.ghi_w_m2 - dni * cos_zenith, 0.0, None)
    extra = pvlib.irradiance.get_extra_radiation(doy)
    cos_incidence = pvlib.irradiance.aoi_projection(
        parameters.tilt_deg, parameters.azimuth_deg, sky.zenith_deg, sky.sun_azimuth_deg
    )
    sky_diffuse = pvlib.irradiance.haydavies(
        surface_tilt=parameters.tilt_deg,
        surface_azimuth=parameters.azimuth_deg,
        dhi=dhi,
        dni=dni,
        dni_extra=extra,
        solar_zenith=sky.zenith_deg,
        solar_azimuth=sky.sun_azimuth_deg,
    )
    ground = sky.ghi_w_m2 * 0.2 * (1.0 - np.cos(np.radians(parameters.tilt_deg))) / 2.0
    beam = dni * np.clip(cos_incidence, 0.0, None)
    poa_total = np.clip(beam + np.nan_to_num(np.asarray(sky_diffuse)) + ground, 0.0, None)
    air = (
        10.0
        - 8.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)
        + 3.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    )
    cell = air + CELL_RISE_K_PER_W_M2 * poa_total
    derate = 1.0 + TEMPERATURE_COEFFICIENT_PER_K * (cell - 25.0)
    dc = parameters.dc_capacity_mw * poa_total / 1000.0 * derate
    return np.minimum(dc, parameters.ac_capacity_mw)


def main() -> None:
    """Simulate, refit, and write `stage1b_recovery.parquet`."""
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for bmu_id in VALIDATION_BMUS:
        latitude, longitude = site_coordinates()[bmu_id]
        sky = sky_at(
            hourly=hourly_cams(point_ids=[site_cams_point(bmu_id=bmu_id)]),
            latitude=latitude,
            longitude=longitude,
        )
        output = align_output(sky=sky, output=output_by_bmu(bmu_id=bmu_id))
        settled = settled_mask(sky=sky, output_mw=output)
        real = fit_plant(sky=sky, output_mw=output, orientation="free", usable=settled)
        if real is None:
            continue
        reference = plant_power_mw(sky=sky, parameters=real.parameters)
        day = daylight(sky=sky) & settled & (reference > 0.1 * real.parameters.ac_capacity_mw)
        relative = np.zeros_like(reference)
        relative[day] = (output[day] - reference[day]) / reference[day]
        relative = np.clip(relative, -1.0, 1.0)
        for _ in range(PLANTS_PER_SITE):
            truth = PlantParameters(
                tilt_deg=float(rng.uniform(5, 40)),
                azimuth_deg=float(rng.uniform(130, 230)),
                dc_capacity_mw=0.0,
                ac_capacity_mw=50.0,
            )
            ratio = float(rng.uniform(1.15, 1.7))
            truth = PlantParameters(
                tilt_deg=truth.tilt_deg,
                azimuth_deg=truth.azimuth_deg,
                dc_capacity_mw=50.0 * ratio,
                ac_capacity_mw=50.0,
            )
            clean = _alternative_power_mw(sky=sky, parameters=truth)
            shift = int(rng.integers(30, 300)) * 48
            noisy = np.clip(clean * (1.0 + np.roll(relative, shift)), 0.0, None)
            fit = fit_plant(sky=sky, output_mw=noisy, orientation="free")
            if fit is None:
                continue
            fitted = fit.parameters
            rows.append(
                {
                    "site": bmu_id,
                    "true_tilt_deg": truth.tilt_deg,
                    "true_azimuth_deg": truth.azimuth_deg,
                    "true_dc_ac_ratio": truth.dc_ac_ratio,
                    "true_ac_capacity_mw": truth.ac_capacity_mw,
                    "fitted_tilt_deg": fitted.tilt_deg,
                    "fitted_azimuth_deg": fitted.azimuth_deg,
                    "fitted_dc_ac_ratio": fitted.dc_ac_ratio,
                    "fitted_ac_capacity_mw": fitted.ac_capacity_mw,
                }
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows)
    frame.write_parquet(OUTPUT_DIR / "stage1b_recovery.parquet")
    print(
        frame.select(
            (pl.col("fitted_tilt_deg") - pl.col("true_tilt_deg")).abs().mean().alias("tilt_mae"),
            (pl.col("fitted_azimuth_deg") - pl.col("true_azimuth_deg"))
            .abs()
            .mean()
            .alias("az_mae"),
            ((pl.col("fitted_ac_capacity_mw") / pl.col("true_ac_capacity_mw")) - 1)
            .abs()
            .mean()
            .alias("ac_rel"),
            (pl.col("fitted_dc_ac_ratio") - pl.col("true_dc_ac_ratio"))
            .abs()
            .mean()
            .alias("ratio_mae"),
            pl.corr("fitted_tilt_deg", "true_tilt_deg").alias("tilt_corr"),
            pl.corr("fitted_azimuth_deg", "true_azimuth_deg").alias("az_corr"),
            pl.corr("fitted_dc_ac_ratio", "true_dc_ac_ratio").alias("ratio_corr"),
        )
    )


if __name__ == "__main__":
    main()
