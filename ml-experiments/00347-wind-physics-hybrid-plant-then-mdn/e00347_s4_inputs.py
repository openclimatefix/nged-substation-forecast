"""Stage 4, step 1 of experiment 00347: pass the 51 ensemble members through the frozen plants.

For each farm, each frozen plant (y1: the Stage 2 curve, y2: the Stage 3 plant with the direction
term, y4: the pass-2 plant of Stage 3b), and each case set (the daily hindcast 2024-04-01 to
2025-06-16 and the 53 scored Tuesday runs), this stores:

- `summary_<plant>_<set>_<farm>.npy`, shape (cases, 6) float32: the median, mean, standard
  deviation, 10th and 90th percentile of the 51 member powers, and the share of members whose
  power is at least 95% of the plant's flat-top output (the saturation share);
- `mix_<plant>_<set>_<farm>.npy`, shape (cases, 11) float32: the equal-weight mixture of the 51
  members' residual-quantile forecasts (experiment 00259's recipe, before its spread factor and
  offset table, which the scoring script applies);
- for the scored set only, `era5_<plant>_scored_<farm>.npy`, shape (cases, 11): the plant driven
  by ERA5 weather at the scored valid times (the Step 0 ceiling) with the same residual quantiles.

Cases follow experiment 00339's order: the hindcast cases with an observed power (farms W1 to W3 in
turn, each sorted by case key), and `e00303_score.ordered_cases()` for the scored cases. The
plants are frozen before 2024-04-01, so the hindcast is out of sample for them.

    PYTHONPATH=<347>:<230>:<303> systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00347-*/e00347_s4_inputs.py [W1 W2 W3]
"""

import os
import sys
import time
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("E00347_TORCH_THREADS", "2")

import e00230_core as core230  # noqa: E402  # ty: ignore[unresolved-import]
import e00303_common as c303  # noqa: E402  # ty: ignore[unresolved-import]
import e00303_ens as ens303  # noqa: E402  # ty: ignore[unresolved-import]
import e00303_ens_final as ens_final  # noqa: E402  # ty: ignore[unresolved-import]
import e00303_score as score303  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_common as common  # noqa: E402
import e00347_plant as plant  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

S4_DIR: Final[Path] = common.DATA_DIR / "s4"
PLANT_FILES: Final[dict[str, str]] = {
    "y1": "plant_stage2.json",
    "y2": "plant_stage3.json",
    "y4": "plant_stage3b_pass2.json",
}
"""Arm -> the file of frozen plant parameters. Arm y3 uses y2's plant."""
MEMBERS: Final[int] = c303.MEMBERS
CHUNK_CASES: Final[int] = 40_000
SATURATION_SHARE: Final[float] = 0.95


def member_weather(*, rows: pl.DataFrame) -> plant.Weather:
    """Return the plant's weather inputs for member rows (one row per case and member)."""
    direction = np.arctan2(
        rows["wind_dir_sin_100m"].to_numpy().astype(np.float64),
        rows["wind_dir_cos_100m"].to_numpy().astype(np.float64),
    )
    return plant.Weather(
        speed_100m=rows["wind_speed_100m"].to_numpy().astype(np.float64),
        speed_10m=rows["wind_speed_10m"].to_numpy().astype(np.float64),
        temperature_c=rows["temperature_c"].to_numpy().astype(np.float64),
        direction_rad=direction,
    )


def expected_in_chunks(
    *, parameters: plant.PlantParameters, weather: plant.Weather, prior_scale: float
) -> np.ndarray:
    """Return the plant's expected power for every row, computed in chunks."""
    size = weather.speed_100m.size
    out = np.empty(size, dtype=np.float32)
    step = 2_000_000
    for start in range(0, size, step):
        index = np.arange(start, min(start + step, size))
        out[index] = plant.expected_power(
            parameters=parameters, weather=weather.take(index), prior_scale=prior_scale
        )
    return out


def summarise(*, power: np.ndarray, flat_top: float) -> np.ndarray:
    """Return (cases, 6): median, mean, sd, p10, p90, and the saturation share of the members."""
    p10, median, p90 = np.quantile(power, [0.1, 0.5, 0.9], axis=1)
    return np.stack(
        [
            median,
            power.mean(axis=1),
            power.std(axis=1),
            p10,
            p90,
            (power >= SATURATION_SHARE * flat_top).mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def member_mixture(*, power: np.ndarray, residuals: core230.ResidualQuantiles) -> np.ndarray:
    """Return the equal-weight mixture of the members' residual-quantile forecasts (cases, 11)."""
    pieces = []
    for start in range(0, power.shape[0], CHUNK_CASES):
        block = power[start : start + CHUNK_CASES]
        member_q = residuals.quantiles(expected=block.reshape(-1).astype(np.float64)).reshape(
            block.shape[0], MEMBERS, 11
        )
        pieces.append(c303.mixture_quantiles(member_quantiles=member_q))
    return np.concatenate(pieces).astype(np.float32)


def hindcast_members(*, series: int, sites: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return one farm's hindcast member rows and the matching cases (key columns, observed)."""
    hind = ens303.hindcast_rows(series=series)
    cases = hind.gather_every(MEMBERS).select(c303.KEY_COLUMNS)
    ens303.check_alignment(members=hind, cases=cases, farm=str(series))
    expected = ens_final.hindcast_cases(sites=sites.filter(pl.col("time_series_id") == series))
    if not expected.select(c303.KEY_COLUMNS).equals(cases):
        msg = f"series {series}: hindcast members do not match the hindcast cases"
        raise ValueError(msg)
    return hind, expected


def scored_members(*, series: int, sites: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return one farm's scored member rows and the matching ordered scored cases."""
    cases_all = score303.ordered_cases()
    site = sites.filter(pl.col("time_series_id") == series)["site"][0]
    cases = cases_all.filter(pl.col("site") == site)
    members = (
        c303.ens_member_rows(sites=sites.filter(pl.col("time_series_id") == series))
        .join(
            cases.select(*c303.KEY_COLUMNS),
            on=list(c303.KEY_COLUMNS),
            how="semi",
            maintain_order="left",
        )
        .sort(*c303.KEY_COLUMNS, "ensemble_member")
    )
    ens303.check_alignment(members=members, cases=cases, farm=site)
    return members, cases


def run_farm(*, farm: str) -> None:
    """Pass one farm's hindcast and scored members through the three frozen plants."""
    started = time.time()
    S4_DIR.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    series = int(sites.filter(pl.col("site") == farm)["time_series_id"][0])
    farm_data = common.load_farms()[farm]
    priors = common.prior_scales()
    parameters = {
        arm: plant.PlantParameters.from_dict(common.read_json(name=file)[farm])
        for arm, file in PLANT_FILES.items()
    }
    train = farm_data.train
    residuals = {}
    for arm, p in parameters.items():
        expected_train = plant.expected_power(
            parameters=p, weather=train.weather, prior_scale=priors[farm]
        )
        residuals[arm] = core230.ResidualQuantiles.fit(
            expected=expected_train.astype(np.float64), observed=train.target
        )
    print(f"{farm}: plants and residual quantiles ready {time.time() - started:.0f} s", flush=True)

    sets = {
        "hindcast": hindcast_members(series=series, sites=sites),
        "scored": scored_members(series=series, sites=sites),
    }
    for set_name, (members, cases) in sets.items():
        usable = (
            cases["actual_norm"].is_finite().fill_null(value=False).to_numpy()
            if set_name == "hindcast"
            else np.ones(cases.height, dtype=bool)
        )
        weather = member_weather(rows=members)
        for arm, p in parameters.items():
            power = expected_in_chunks(parameters=p, weather=weather, prior_scale=priors[farm])
            power = power.reshape(-1, MEMBERS)[usable]
            np.save(
                S4_DIR / f"summary_{arm}_{set_name}_{farm}.npy",
                summarise(power=power, flat_top=p.scale),
            )
            np.save(
                S4_DIR / f"mix_{arm}_{set_name}_{farm}.npy",
                member_mixture(power=power, residuals=residuals[arm]),
            )
            print(f"{farm} {set_name} {arm} done {time.time() - started:.0f} s", flush=True)
        del weather

    # The Step 0 ceiling: each plant driven by ERA5 weather at the scored valid times.
    era5 = c303.era5_rows(sites=sites.filter(pl.col("time_series_id") == series))
    _, scored_cases = sets["scored"]
    joined = scored_cases.select(*c303.KEY_COLUMNS).join(
        era5, on=["time_series_id", "valid_time"], how="left", maintain_order="left"
    )
    if joined["wind_speed_100m"].null_count():
        msg = f"{farm}: scored cases without ERA5 weather"
        raise ValueError(msg)
    weather = plant.Weather(
        speed_100m=joined["wind_speed_100m"].to_numpy().astype(np.float64),
        speed_10m=joined["wind_speed_10m"].to_numpy().astype(np.float64),
        temperature_c=joined["temperature_c"].to_numpy().astype(np.float64),
        direction_rad=np.arctan2(
            joined["wind_dir_sin_100m"].to_numpy().astype(np.float64),
            joined["wind_dir_cos_100m"].to_numpy().astype(np.float64),
        ),
    )
    for arm, p in parameters.items():
        expected = plant.expected_power(parameters=p, weather=weather, prior_scale=priors[farm])
        np.save(
            S4_DIR / f"era5_{arm}_scored_{farm}.npy",
            residuals[arm].quantiles(expected=expected.astype(np.float64)),
        )
    print(f"{farm}: finished {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Run the farms named on the command line (all three by default), one after another."""
    farms = sys.argv[1:] or list(common.FARMS)
    for farm in farms:
        run_farm(farm=farm)


if __name__ == "__main__":
    main()
