"""Swap the weather input stage by stage and record where the wind forecast's median falls.

Experiment 00259's physics core (R1) is refitted on ERA5 before 2024-04-01 with the same code, and
then run on four weather inputs at every case of experiment 00259's hindcast and scored sets:

* ERA5 at the valid time (the oracle: right wind, no ensemble),
* the ensemble-mean 10 m wind, 100 m wind, and 2 m temperature (one value per case),
* each of the 51 members (the expected power of each member, then their mean and median).

The stored R1 and R0 forecasts of experiment 00259 (the 51-member mixture after the spread factor)
are the last stage. The spread factor scales the quantiles about the median, so it cannot move the
median; the script checks that the stored R1 median equals the recomputed mixture median.

Output, in `work/`: `cases_<set>.parquet` (one row per case) and `check.json`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00261-*/e00261_decompose.py
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00261_common as common  # noqa: E402
import e00261_core as core_module  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

PRIOR_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00259-wind-plant-plus-quantile-offset-head" / "work"
)
"""Experiment 00259's hindcast member rows and stored base forecasts (read only)."""
BASE_END: Final = datetime(2024, 4, 1, tzinfo=UTC)
SETS: Final[tuple[str, ...]] = ("hindcast", "scored")
SLOPE_STEP_M_S: Final[float] = 0.25
CHECK_CASES: Final[int] = 600
MEDIAN_TOLERANCE: Final[float] = 2e-4


def weather(
    *, frame: pl.DataFrame, speed_100m: np.ndarray | None = None
) -> core_module.WeatherArrays:
    """Return the core's three weather columns, optionally with the 100 m speed replaced."""
    return core_module.WeatherArrays(
        speed_10m=frame["wind_speed_10m"].to_numpy().astype(np.float32),
        speed_100m=(
            frame["wind_speed_100m"].to_numpy().astype(np.float32)
            if speed_100m is None
            else speed_100m.astype(np.float32)
        ),
        temperature_c=frame["temperature_c"].to_numpy().astype(np.float32),
    )


def case_columns(*, members: pl.DataFrame) -> pl.DataFrame:
    """Return per-case means of the members' three core inputs and their direction vector."""
    shaped = {
        name: members[name].to_numpy().reshape(-1, common.MEMBERS).astype(np.float64)
        for name in ("wind_speed_10m", "wind_speed_100m", "temperature_c")
    }
    return pl.DataFrame(
        {
            "wind_speed_10m": shaped["wind_speed_10m"].mean(axis=1),
            "wind_speed_100m": shaped["wind_speed_100m"].mean(axis=1),
            "temperature_c": shaped["temperature_c"].mean(axis=1),
        }
    ).cast(pl.Float32)


def member_powers(*, core: core_module.WindCore, members: pl.DataFrame) -> np.ndarray:
    """Return shape (cases, 51) expected power of each member."""
    expected = core_module.predict_expected_power(core=core, weather=weather(frame=members))
    return expected.reshape(-1, common.MEMBERS)


def farm_stage_table(
    *,
    core: core_module.WindCore,
    residuals: core_module.ResidualQuantiles,
    members: pl.DataFrame,
    era5_at_cases: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return one farm's per-case stage table, and its member expected powers for checks."""
    keys = members.gather_every(common.MEMBERS).select(common.KEY_COLUMNS)
    mean_wind = case_columns(members=members)
    powers = member_powers(core=core, members=members)
    era5 = keys.join(era5_at_cases, on=["time_series_id", "valid_time"], how="left", validate="m:1")
    oracle_weather = weather(frame=era5)
    oracle = core_module.predict_expected_power(core=core, weather=oracle_weather)
    slope = (
        core_module.predict_expected_power(
            core=core,
            weather=weather(
                frame=era5, speed_100m=era5["wind_speed_100m"].to_numpy() + SLOPE_STEP_M_S
            ),
        )
        - core_module.predict_expected_power(
            core=core,
            weather=weather(
                frame=era5,
                speed_100m=np.maximum(era5["wind_speed_100m"].to_numpy() - SLOPE_STEP_M_S, 0.0),
            ),
        )
    ) / (2 * SLOPE_STEP_M_S)
    at_mean_wind = core_module.predict_expected_power(core=core, weather=weather(frame=mean_wind))
    members_100 = members["wind_speed_100m"].to_numpy().reshape(-1, common.MEMBERS)
    table = keys.with_columns(
        era5_speed_10m=era5["wind_speed_10m"],
        era5_speed_100m=era5["wind_speed_100m"],
        era5_dir_sin=era5["wind_dir_sin_100m"],
        era5_dir_cos=era5["wind_dir_cos_100m"],
        ens_speed_10m=mean_wind["wind_speed_10m"],
        ens_speed_100m=mean_wind["wind_speed_100m"],
        ens_speed_100m_p10=pl.Series(np.quantile(members_100, 0.1, axis=1).astype(np.float32)),
        ens_speed_100m_p90=pl.Series(np.quantile(members_100, 0.9, axis=1).astype(np.float32)),
        e_oracle=pl.Series(oracle),
        slope_per_m_s=pl.Series(slope.astype(np.float32)),
        e_mean_wind=pl.Series(at_mean_wind),
        e_members_mean=pl.Series(powers.mean(axis=1).astype(np.float32)),
        e_members_median=pl.Series(np.median(powers, axis=1).astype(np.float32)),
        q50_oracle=pl.Series(
            residuals.quantiles(expected=oracle)[:, common.MEDIAN_INDEX].astype(np.float32)
        ),
        q50_mean_wind=pl.Series(
            residuals.quantiles(expected=at_mean_wind)[:, common.MEDIAN_INDEX].astype(np.float32)
        ),
    )
    return table, powers


def median_check(
    *,
    residuals: core_module.ResidualQuantiles,
    powers: np.ndarray,
    stored_q50: np.ndarray,
    seed: int,
) -> float:
    """Return the largest gap between the recomputed mixture median and the stored R1 median.

    Only cases that experiment 00259 stored (those with a finite observation) are compared.
    """
    rng = np.random.default_rng(seed)
    stored = np.flatnonzero(np.isfinite(stored_q50))
    pick = rng.choice(stored, size=min(CHECK_CASES, stored.size), replace=False)
    member_quantiles = np.stack(
        [residuals.quantiles(expected=powers[pick, m]) for m in range(common.MEMBERS)], axis=1
    )
    mixture = common.mixture_quantiles(member_quantiles=member_quantiles)
    return float(np.abs(mixture[:, common.MEDIAN_INDEX] - stored_q50[pick]).max())


def main() -> None:
    """Write the per-case stage tables for the hindcast and scored sets."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table()
    era5_all = common.era5_rows(sites=sites)
    train = common.training_table(sites=sites, era5=era5_all).filter(
        pl.col("valid_time") < BASE_END
    )
    checks: dict[str, float] = {}
    fitted = {}
    for row in sites.iter_rows(named=True):
        farm_train = train.filter(pl.col("time_series_id") == row["time_series_id"])
        target = farm_train["target"].to_numpy()
        farm_weather = weather(frame=farm_train)
        core = core_module.fit_core(
            weather=farm_weather, target=target, with_idle_layer=row["site"] != "W2", seed=0
        )
        residuals = core_module.ResidualQuantiles.fit(
            expected=core_module.predict_expected_power(core=core, weather=farm_weather),
            observed=target,
        )
        fitted[row["site"]] = (int(row["time_series_id"]), core, residuals)
        print(f"{row['site']}: core fitted on {farm_train.height} ERA5 rows", flush=True)
    scored_members = common.ens_member_rows(sites=sites)
    for name in SETS:
        summary = pl.read_parquet(PRIOR_DIR / "base" / f"summaries_{name}.parquet")
        stored_r1 = pl.read_parquet(PRIOR_DIR / "base" / f"R1_{name}_seed0.parquet").select(
            *common.KEY_COLUMNS, q50_stored_r1="q50"
        )
        stored_r0 = pl.read_parquet(PRIOR_DIR / "base" / f"R0_{name}_seed0.parquet").select(
            *common.KEY_COLUMNS, q50_stored_r0="q50"
        )
        pieces = []
        for farm, (series, core, residuals) in fitted.items():
            if name == "hindcast":
                paths = sorted((PRIOR_DIR / "hindcast").glob("init=*.parquet"))
                members = (
                    pl.concat(
                        [
                            pl.scan_parquet(p).filter(pl.col("time_series_id") == series)
                            for p in paths
                        ]
                    )
                    .sort(*common.KEY_COLUMNS, "ensemble_member")
                    .collect()
                )
            else:
                members = scored_members.filter(pl.col("time_series_id") == series)
            era5_farm = era5_all.filter(pl.col("time_series_id") == series)
            table, powers = farm_stage_table(
                core=core, residuals=residuals, members=members, era5_at_cases=era5_farm
            )
            farm_stored = table.join(stored_r1, on=list(common.KEY_COLUMNS), how="left")
            checks[f"{name}_{farm}_max_median_gap"] = median_check(
                residuals=residuals,
                powers=powers,
                stored_q50=farm_stored["q50_stored_r1"].fill_null(float("nan")).to_numpy(),
                seed=series,
            )
            pieces.append(table)
            print(f"{name} {farm}: {table.height} cases", flush=True)
        stages = pl.concat(pieces)
        out = (
            summary.select(
                *common.KEY_COLUMNS, "site", "lead_hours", "lead_day", "actual_norm", "masked"
            )
            .join(stages, on=list(common.KEY_COLUMNS), how="left", validate="1:1")
            .join(stored_r1, on=list(common.KEY_COLUMNS), how="left", validate="1:1")
            .join(stored_r0, on=list(common.KEY_COLUMNS), how="left", validate="1:1")
        )
        out.write_parquet(common.WORK_DIR / f"cases_{name}.parquet")
        print(f"{name}: wrote {out.height} cases", flush=True)
    (common.WORK_DIR / "check.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
