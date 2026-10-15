"""Step 2 of experiment 00029: recover known PV parameters from synthetic plants.

Thirty-six synthetic plants (6 farm coordinates times 6 parameter draws) are driven by the real
CAMS irradiance and ERA5 temperature of a random two-year window before the training cutoff, with
autocorrelated noise. The `same` version generates the power with the fitted model's own chain; the
`different` version uses the Hay-Davies sky in place of the isotropic sky. The script writes one
row per plant and version to `WORK_DIR/synthetic.parquet` and prints the summary the write-up
quotes.

    uv run python ml-experiments/00029-*/e00029_synthetic.py
"""

import numpy as np
import polars as pl
import pvlib
import torch  # ty: ignore[unresolved-import]
from e00029_common import ANALYSIS_PATH, ERA5_START, MAX_ZENITH_DEGREES, TRAIN_END, WORK_DIR
from e00029_physics import (
    CELL_HEATING_K_PER_W_M2,
    GROUND_ALBEDO,
    TEMPERATURE_COEFFICIENT,
    SkyTensors,
    erbs_split,
    fit_plants,
    plant_power,
    sky_tensors,
    soft_clip,
)
from studies.pv_dataset import pv_sites

PLANTS_PER_LOCATION = 8
ON_PRIOR_DRAWS = 6
"""The first draws of a location are on-prior; the last two have off-prior orientations."""
WINDOW_ROWS = 2 * 365 * 48
NOISE_SD = 0.04
NOISE_CORRELATION = 0.95
SEED = 29
PARAMETER_NAMES = ("tilt", "azimuth", "dc", "ac", "ratio")


def hay_davies_power(*, sky_np: dict[str, np.ndarray], truth: dict[str, float]) -> np.ndarray:
    """Return a plant's power from the Hay-Davies transposition, with the same derate and clip."""
    extra = pvlib.irradiance.get_extra_radiation(sky_np["doy"])
    diffuse = pvlib.irradiance.haydavies(
        surface_tilt=truth["tilt"],
        surface_azimuth=truth["azimuth"],
        dhi=sky_np["dhi"],
        dni=sky_np["dni"],
        dni_extra=extra,
        solar_zenith=sky_np["zenith"],
        solar_azimuth=sky_np["sun_azimuth"],
    )
    cos_tilt = np.cos(np.radians(truth["tilt"]))
    cos_inc = np.cos(np.radians(sky_np["zenith"])) * cos_tilt + np.sin(
        np.radians(sky_np["zenith"])
    ) * np.sin(np.radians(truth["tilt"])) * np.cos(
        np.radians(sky_np["sun_azimuth"] - truth["azimuth"])
    )
    beam = sky_np["dni"] * np.clip(cos_inc, 0, None)
    ground = sky_np["ghi"] * GROUND_ALBEDO * (1 - cos_tilt) / 2
    poa = np.clip(beam + np.nan_to_num(np.asarray(diffuse)) + ground, 0, None)
    derate = np.clip(
        1 - TEMPERATURE_COEFFICIENT * (sky_np["temperature"] + CELL_HEATING_K_PER_W_M2 * poa - 25),
        0,
        None,
    )
    dc_power = truth["dc"] * poa / 1000.0 * derate
    return soft_clip(
        power=torch.as_tensor(dc_power, dtype=torch.float32),
        limit=torch.as_tensor(truth["ac"], dtype=torch.float32),
    ).numpy()


def main() -> None:  # noqa: PLR0915
    """Generate the plants, fit them, and write and print the recovery table."""
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda")
    analysis = pl.read_parquet(ANALYSIS_PATH).filter(
        (pl.col("valid_time") >= ERA5_START),
        (pl.col("valid_time") < TRAIN_END),
    )
    series = pv_sites().sort("site")["time_series_id"].to_list()
    rows = []
    sky_batch: list[SkyTensors] = []
    targets, masks = [], []
    for location, series_id in enumerate(series):
        frame = analysis.filter(pl.col("time_series_id") == series_id).sort("valid_time")
        for draw in range(PLANTS_PER_LOCATION):
            start = int(rng.integers(0, frame.height - WINDOW_ROWS))
            window = frame.slice(start, WINDOW_ROWS)
            ghi = window["shortwave_cams"].to_numpy()
            zenith = window["solar_zenith"].to_numpy()
            sun_az = window["solar_azimuth"].to_numpy()
            doy = window["valid_time"].dt.ordinal_day().to_numpy()
            temp = window["temperature_2m"].to_numpy()
            usable = np.isfinite(ghi) & (zenith < MAX_ZENITH_DEGREES)
            dni, dhi = erbs_split(ghi=ghi, zenith_deg=zenith, day_of_year=doy)
            ghi0 = np.clip(np.nan_to_num(ghi), 0, None)
            truth = {
                "tilt": float(rng.uniform(10, 45)),
                "azimuth": float(rng.uniform(150, 210)),
                "dc": float(rng.uniform(0.9, 1.3)),
            }
            if draw >= ON_PRIOR_DRAWS:
                truth["tilt"] = float(rng.choice([5.0, 60.0]))
                truth["azimuth"] = float(rng.choice([110.0, 250.0]))
            truth["ac"] = truth["dc"] * float(rng.uniform(0.7, 1.0))
            truth["ratio"] = truth["dc"] / truth["ac"]
            sky_np = {
                "ghi": ghi0, "dni": dni, "dhi": dhi, "zenith": zenith, "sun_azimuth": sun_az,
                "temperature": np.nan_to_num(temp, nan=10.0), "doy": doy,
            }  # fmt: skip
            noise = np.zeros(WINDOW_ROWS)
            shocks = rng.normal(0, NOISE_SD * np.sqrt(1 - NOISE_CORRELATION**2), WINDOW_ROWS)
            for i in range(1, WINDOW_ROWS):
                noise[i] = NOISE_CORRELATION * noise[i - 1] + shocks[i]
            for version in ("same", "different"):
                sky = sky_tensors(
                    ghi=ghi, zenith_deg=zenith, sun_azimuth_deg=sun_az, day_of_year=doy,
                    temperature_c=temp, device=torch.device("cpu"),
                )  # fmt: skip
                if version == "same":
                    clean = plant_power(
                        sky=sky,
                        params={
                            k: torch.tensor([truth[k]]) for k in ("tilt", "azimuth", "dc", "ac")
                        },
                    )[0].numpy()
                else:
                    clean = hay_davies_power(sky_np=sky_np, truth=truth)
                observed = np.clip(clean + noise, 0, None) * usable
                rows.append(
                    {
                        "version": version, "location": location, "plant": len(rows) // 2,
                        "off_prior": draw >= ON_PRIOR_DRAWS,
                        **{f"truth_{k}": v for k, v in truth.items()},
                        "clip_share": float((clean[usable] > 0.98 * truth["ac"]).mean()),
                    }
                )  # fmt: skip
                sky_batch.append(sky)
                targets.append(observed)
                masks.append(usable)
    results = []
    for version in ("same", "different"):
        index = [i for i, r in enumerate(rows) if r["version"] == version]
        sky = SkyTensors(
            *(
                torch.cat([getattr(sky_batch[i], f) for i in index]).to(device)
                for f in ("ghi", "dni", "dhi", "zenith", "sun_azimuth", "temperature")
            )
        )
        target = torch.as_tensor(
            np.stack([targets[i] for i in index]), dtype=torch.float32, device=device
        )
        mask = torch.as_tensor(np.stack([masks[i] for i in index]), device=device)
        summary = fit_plants(sky=sky, target=target, mask=mask, seed=SEED)
        fitted = plant_power(
            sky=sky,
            params={
                k: torch.as_tensor(summary.point[k], device=device)
                for k in ("tilt", "azimuth", "dc", "ac")
            },
        )
        mae = ((fitted - target).abs() * mask).sum(dim=1) / mask.sum(dim=1)
        for j, i in enumerate(index):
            row = dict(rows[i], fit_mae=float(mae[j]), noise=float(summary.noise[j]))
            for name in PARAMETER_NAMES:
                low, high = summary.interval(name=name)
                row[f"est_{name}"], row[f"lo_{name}"], row[f"hi_{name}"] = (
                    float(summary.point[name][j]), float(low[j]), float(high[j]),
                )  # fmt: skip
            results.append(row)
    frame = pl.DataFrame(results)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(WORK_DIR / "synthetic.parquet")
    for version, off_prior in [(v, o) for v in ("same", "different") for o in (False, True)]:
        part = frame.filter((pl.col("version") == version) & (pl.col("off_prior") == off_prior))
        version = f"{version}{' (off-prior)' if off_prior else ''}"  # noqa: PLW2901
        fit_mae = 100 * float(np.mean(part["fit_mae"].to_numpy()))
        print(f"\n{version}: {part.height} plants, fit MAE {fit_mae:.2f} points")
        for name in PARAMETER_NAMES:
            error = part[f"est_{name}"] - part[f"truth_{name}"]
            cover = (
                (part[f"lo_{name}"] <= part[f"truth_{name}"])
                & (part[f"truth_{name}"] <= part[f"hi_{name}"])
            ).to_numpy()
            print(
                f"  {name:8s} bias {error.mean():+.3f}  MAE {error.abs().mean():.3f}"
                f"  90% coverage {100 * float(np.mean(cover)):.0f}%"
            )


if __name__ == "__main__":
    main()
