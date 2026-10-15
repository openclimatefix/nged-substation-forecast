"""Fit the physics stage 1 and the XGBoost stage 2 layer, and write the pooled mixtures.

Steps (`docs/ml-experiments/00029-physics-stage1-solar-pytorch/result.md`):

1. Fit each farm's physics parameters twice to metered power at rows before `TRAIN_END`, using CAMS
   irradiance and ERA5 temperature: with SciPy (`studies.pv_fit.fit_plant`, free orientation) and
   with the PyTorch PV model of `e00029_physics.py`. The fitted parameters are private.
2. Drive both fitted physics models with ERA5 shortwave and temperature on the training rows, and
   with each ENS member's shortwave and temperature on the validation rows.
3. Train one multi-quantile XGBoost model per farm, seed, and arm on the training rows, and apply
   it to each member of each validation run. Pool the 51 members into one mixture per case.

The arms are `scipy` and `torch` (the layer reads the physics power only) and `torch_hybrid` (the
physics power plus the nine weather columns of `base`). Mixtures go to
`WORK_DIR/mixture/<arm>_seed<seed>.parquet`.

    uv run --no-sync python ml-experiments/00029-*/e00029_fit.py [--smoke]
"""

import argparse
import json
import time
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
import xgboost as xgb
from e00029_common import (
    ANALYSIS_PATH,
    ENS_VALIDATION_DIR,
    ERA5_START,
    KEY_COLUMNS,
    LEVEL_COLUMNS,
    LEVELS,
    MAX_ZENITH_DEGREES,
    MEMBERS,
    N_ROUNDS,
    SEEDS,
    SOLAR_FEATURES,
    TRAIN_END,
    WORK_DIR,
    XGBOOST_PARAMETERS,
    mixture_quantiles,
    observed_training_power,
    read_levels,
    site_table,
)
from e00029_physics import SkyTensors, erbs_split, fit_plants, plant_power, sky_tensors
from studies.pv_fit import fit_plant
from studies.pv_physics import SunAndSky, plant_power_mw

ARMS: Final[tuple[str, ...]] = ("scipy", "torch", "torch_hybrid")
MIXTURE_DIR: Final = WORK_DIR / "mixture"
POINT_DIR: Final = WORK_DIR / "point"
PARAMS_PATH: Final = WORK_DIR / "fitted_parameters.json"
DEVICE: Final = torch.device("cuda")


def sky_arrays(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the physics inputs of a frame's rows: shortwave, geometry, day, and temperature."""
    return {
        "ghi": frame["shortwave"].to_numpy().astype(np.float64),
        "zenith": frame["solar_zenith"].to_numpy().astype(np.float64),
        "sun_azimuth": frame["solar_azimuth"].to_numpy().astype(np.float64),
        "doy": frame["valid_time"].dt.ordinal_day().to_numpy(),
        "temperature": frame["temperature_2m"].to_numpy().astype(np.float64),
    }


def scipy_sky(*, arrays: dict[str, np.ndarray], ghi: np.ndarray) -> SunAndSky:
    """Return the SciPy model's sky: the Erbs split of `ghi` and the sun's position."""
    dni, dhi = erbs_split(ghi=ghi, zenith_deg=arrays["zenith"], day_of_year=arrays["doy"])
    return SunAndSky(
        half_hour_end_time=np.arange(ghi.size),
        ghi_w_m2=np.clip(np.nan_to_num(ghi), 0.0, None),
        dni_w_m2=dni,
        dhi_w_m2=dhi,
        zenith_deg=arrays["zenith"],
        sun_azimuth_deg=arrays["sun_azimuth"],
    )


def fit_physics(*, sites: pl.DataFrame, training: dict[int, pl.DataFrame]) -> dict[int, dict]:
    """Fit both physics models to every farm's CAMS-driven training rows.

    Returns:
        For each series id, `scipy` (tilt, azimuth, DC and AC capacity as fractions of capacity)
        and `torch` (the same four at the posterior mean, plus 90% intervals).
    """
    out: dict[int, dict] = {}
    ids = sites["time_series_id"].to_list()
    batch = []
    for series in ids:
        frame = training[series]
        cams = frame.with_columns(shortwave=pl.col("shortwave_cams"))
        usable = (
            cams["shortwave"].is_not_null().to_numpy()
            & cams["target"].is_not_null().to_numpy()
            & (cams["solar_zenith"].to_numpy() < MAX_ZENITH_DEGREES)
        )
        arrays = sky_arrays(frame=cams)
        sky = scipy_sky(arrays=arrays, ghi=arrays["ghi"])
        target = np.nan_to_num(cams["target"].to_numpy().astype(np.float64))
        result = fit_plant(sky=sky, output_mw=target, orientation="free", usable=usable)
        if result is None:
            msg = f"the SciPy fit of series {series} found too few usable rows"
            raise ValueError(msg)
        p = result.parameters
        out[series] = {
            "scipy": {
                "tilt": p.tilt_deg,
                "azimuth": p.azimuth_deg,
                "dc": p.dc_capacity_mw,
                "ac": p.ac_capacity_mw,
            }
        }
        batch.append((arrays, target, usable))
    sky_t = SkyTensors(
        *(
            torch.cat(
                [
                    sky_tensors(
                        ghi=a["ghi"],
                        zenith_deg=a["zenith"],
                        sun_azimuth_deg=a["sun_azimuth"],
                        day_of_year=a["doy"],
                        temperature_c=a["temperature"],
                        device=DEVICE,
                    ).__getattribute__(field)
                    for a, _, _ in batch
                ]
            )
            for field in ("ghi", "dni", "dhi", "zenith", "sun_azimuth", "temperature")
        )
    )
    target = torch.as_tensor(np.stack([t for _, t, _ in batch]), dtype=torch.float32, device=DEVICE)
    mask = torch.as_tensor(np.stack([u for _, _, u in batch]), device=DEVICE)
    summary = fit_plants(sky=sky_t, target=target, mask=mask, seed=SEEDS[0])
    for j, series in enumerate(ids):
        entry = {name: float(summary.point[name][j]) for name in ("tilt", "azimuth", "dc", "ac")}
        entry["ratio"] = float(summary.point["ratio"][j])
        for name in ("tilt", "azimuth", "dc", "ac", "ratio"):
            low, high = summary.interval(name=name)
            entry[f"{name}_lo"], entry[f"{name}_hi"] = float(low[j]), float(high[j])
        entry["noise"] = float(summary.noise[j])
        out[series]["torch"] = entry
    return out


def physics_power(*, frame: pl.DataFrame, fitted: dict) -> dict[str, np.ndarray]:
    """Return both fitted physics models' power (fraction of capacity) for a frame's rows.

    The frame's `shortwave` and `temperature_2m` drive the models: ERA5 on training rows, an ENS
    member on validation rows. The SciPy model has no temperature derate.
    """
    arrays = sky_arrays(frame=frame)
    sky = scipy_sky(arrays=arrays, ghi=arrays["ghi"])
    from studies.pv_physics import PlantParameters

    s = fitted["scipy"]
    scipy_power = plant_power_mw(
        sky=sky,
        parameters=PlantParameters(
            tilt_deg=s["tilt"],
            azimuth_deg=s["azimuth"],
            dc_capacity_mw=s["dc"],
            ac_capacity_mw=s["ac"],
        ),
    )
    tensors = sky_tensors(
        ghi=arrays["ghi"],
        zenith_deg=arrays["zenith"],
        sun_azimuth_deg=arrays["sun_azimuth"],
        day_of_year=arrays["doy"],
        temperature_c=arrays["temperature"],
        device=DEVICE,
    )
    t = fitted["torch"]
    torch_power = plant_power(
        sky=tensors,
        params={k: torch.tensor([t[k]], device=DEVICE) for k in ("tilt", "azimuth", "dc", "ac")},
    )[0]
    return {"scipy": scipy_power, "torch": torch_power.cpu().numpy().astype(np.float64)}


def layer_matrix(*, arm: str, frame: pl.DataFrame, power: dict[str, np.ndarray]) -> np.ndarray:
    """Return the stage 2 layer's feature matrix for an arm."""
    physics = power["scipy" if arm == "scipy" else "torch"].astype(np.float32)[:, None]
    if arm != "torch_hybrid":
        return physics
    weather = frame.select([pl.col(c).cast(pl.Float32) for c in SOLAR_FEATURES]).to_numpy()
    return np.hstack([physics, weather])


def pooled_mixture(*, part: pl.DataFrame, member_quantiles: np.ndarray) -> pl.DataFrame:
    """Return the keys and 9 mixture quantiles of every case in `part`.

    Args:
        part: One run's rows for one series, with `ensemble_member` and the key columns.
        member_quantiles: Shape (rows, 9), the member quantiles of each row of `part`.

    Returns:
        One row per case.

    Raises:
        ValueError: If a case lacks a member.
    """
    ordered = (
        part.select(*KEY_COLUMNS, "ensemble_member")
        .with_columns(row=pl.int_range(pl.len()))
        .sort(*KEY_COLUMNS, "ensemble_member")
    )
    if ordered.height % MEMBERS:
        msg = f"a case lacks some of the {MEMBERS} members"
        raise ValueError(msg)
    matrix = member_quantiles[ordered["row"].to_numpy()].reshape(-1, MEMBERS, len(LEVEL_COLUMNS))
    keys = ordered.gather_every(MEMBERS).select(KEY_COLUMNS)
    if (ordered["ensemble_member"].to_numpy().reshape(-1, MEMBERS) != np.arange(MEMBERS)).any():
        msg = "the members of a case are not 0 to 50"
        raise ValueError(msg)
    mixture = read_levels(fine=mixture_quantiles(member_quantiles=matrix.astype(np.float32)))
    return pl.concat([keys, pl.DataFrame(mixture, schema=list(LEVEL_COLUMNS))], how="horizontal")


def point_forecast(*, part: pl.DataFrame, power: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return the member-average physics power (fraction of capacity) of every case."""
    return (
        part.select(*KEY_COLUMNS)
        .with_columns(scipy=pl.Series(power["scipy"]), torch=pl.Series(power["torch"]))
        .group_by(KEY_COLUMNS)
        .agg(pl.col("scipy").mean(), pl.col("torch").mean())
    )


def train_layers(
    *, seed: int, training: dict[int, pl.DataFrame], fitted: dict[int, dict], rounds: int
) -> dict[tuple[int, str], xgb.Booster]:
    """Train one multi-quantile XGBoost layer per farm and arm."""
    models = {}
    for series, frame in training.items():
        rows = frame.filter(pl.col("target").is_not_null())
        power = physics_power(frame=rows, fitted=fitted[series])
        for arm in ARMS:
            matrix = xgb.DMatrix(
                layer_matrix(arm=arm, frame=rows, power=power), label=rows["target"].to_numpy()
            )
            models[series, arm] = xgb.train(
                {
                    **XGBOOST_PARAMETERS,
                    "objective": "reg:quantileerror",
                    "quantile_alpha": LEVELS,
                    "seed": seed,
                },
                matrix,
                num_boost_round=rounds,
            )
    return models


def main() -> None:
    """Fit the physics models, then every seed's layers, and write mixtures and point forecasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 rounds, 2 runs, no output")
    smoke = parser.parse_args().smoke
    sites = site_table()
    analysis = pl.read_parquet(ANALYSIS_PATH)
    observed = observed_training_power(sites=sites)
    training = {
        int(series): analysis.filter(
            (pl.col("time_series_id") == series)
            & (pl.col("valid_time") >= ERA5_START)
            & (pl.col("valid_time") < TRAIN_END)
        )
        .sort("valid_time")
        .join(
            observed.filter(pl.col("time_series_id") == series),
            on=["time_series_id", "valid_time"],
            how="left",
            validate="1:1",
        )
        for series in sites["time_series_id"]
    }
    started = time.monotonic()
    fitted = fit_physics(sites=sites, training=training)
    print(f"physics fits in {time.monotonic() - started:.0f} s", flush=True)
    if not smoke:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        label_of = dict(zip(sites["time_series_id"], sites["site"], strict=True))
        PARAMS_PATH.write_text(
            json.dumps({label_of[k]: v for k, v in fitted.items()}, indent=1, sort_keys=True)
        )
    files = sorted(ENS_VALIDATION_DIR.glob("init=*.parquet"))[: 2 if smoke else None]
    for seed in SEEDS[:1] if smoke else SEEDS:
        targets = {arm: MIXTURE_DIR / f"{arm}_seed{seed}.parquet" for arm in ARMS}
        if not smoke and all(p.exists() for p in targets.values()):
            continue
        models = train_layers(
            seed=seed, training=training, fitted=fitted, rounds=20 if smoke else N_ROUNDS
        )
        print(f"seed {seed}: layers in {time.monotonic() - started:.0f} s", flush=True)
        pieces: dict[str, list[pl.DataFrame]] = {arm: [] for arm in ARMS}
        points: list[pl.DataFrame] = []
        for path in files:
            run = pl.read_parquet(path).filter(pl.col("time_series_id").is_in(list(fitted)))
            for (series_key,), part in run.partition_by("time_series_id", as_dict=True).items():
                series = int(series_key)
                power = physics_power(frame=part, fitted=fitted[series])
                for arm in ARMS:
                    matrix = xgb.DMatrix(layer_matrix(arm=arm, frame=part, power=power))
                    quantiles = np.sort(models[series, arm].predict(matrix), axis=1)
                    pieces[arm].append(pooled_mixture(part=part, member_quantiles=quantiles))
                if seed == SEEDS[0]:
                    points.append(point_forecast(part=part, power=power))
        if smoke:
            print("smoke run: no output written", flush=True)
            return
        MIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        for arm, frames in pieces.items():
            pl.concat(frames).sort(*KEY_COLUMNS).write_parquet(targets[arm].with_suffix(".tmp"))
            targets[arm].with_suffix(".tmp").rename(targets[arm])
        if points:
            POINT_DIR.mkdir(parents=True, exist_ok=True)
            pl.concat(points).sort(*KEY_COLUMNS).write_parquet(POINT_DIR / "physics_point.parquet")
        print(f"seed {seed} done in {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
