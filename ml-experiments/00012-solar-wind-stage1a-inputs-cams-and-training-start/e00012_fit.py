"""Fit experiment 00012's stage 1a arms and apply them to the 51 ENS members of fold A.

For each seed and series this fits one XGBoost model per arm on the half-hourly valid times before
2025-06-17 (see `e00012_common.py` for the arms), then predicts the power fraction of every ENS
member of the 53 validation runs and averages the members. For the `cams_stage0` arm a stage 0
model first maps each member's shortwave to CAMS irradiance. Outputs under
`data/ml-experiments/00012-.../work/` (each skipped if it exists, so a crash costs one seed):

- `member_average/seed<s>.parquet`: keys, `arm`, and `point_norm` (the member average, a fraction
  of capacity) for every validation row of the series each arm covers.
- `ceiling/seed<s>.parquet`: arm `w` applied to ERA5 weather at the validation valid times.
- `stage0/sanity.md`: stage 0's error against CAMS on its last 6 weeks, from a fit that stops
  before them (seed 0).

    uv run python ml-experiments/00012-*/e00012_fit.py
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00012_common as common
import numpy as np
import polars as pl
import xgboost as xgb

MEMBER_AVERAGE_DIR: Final[Path] = common.WORK_DIR / "member_average"
CEILING_DIR: Final[Path] = common.WORK_DIR / "ceiling"
STAGE0_SANITY_PATH: Final[Path] = common.WORK_DIR / "stage0" / "sanity.md"
STAGE0_SANITY_START: Final[datetime] = datetime(2025, 5, 6, tzinfo=UTC)
"""The last 6 weeks before `common.TRAIN_END`, held out of the seed-0 sanity fit."""


@dataclass
class FittedArm:
    """One fitted arm for one series and seed.

    Attributes:
        arm: The arm's name.
        booster: The fitted model.
        columns: The input columns, in order.
        clear_sky_constant: For `csr`, power over clear-sky irradiance; else `None`.
        curve: For `base_margin`, the power curve's speeds and power fractions; else `None`.
    """

    arm: common.ArmType
    booster: xgb.Booster
    columns: list[str]
    clear_sky_constant: float | None = None
    curve: tuple[np.ndarray, np.ndarray] | None = field(default=None)


def arm_columns(*, arm: common.ArmType, technology: common.TechnologyType) -> list[str]:
    """Return the input columns of an arm.

    Args:
        arm: The arm's name.
        technology: The series' technology.

    Returns:
        The column names, in model order.
    """
    if arm == "small_block":
        return list(common.SMALL_SOLAR_FEATURES)
    if arm == "speed_block":
        return list(common.SPEED_WIND_FEATURES)
    return list(common.SOLAR_FEATURES if technology == "solar" else common.WIND_FEATURES)


def _matrix(*, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


def fit_stage0(
    *, ens_train: pl.DataFrame, cams: pl.DataFrame, seed: int, before: datetime
) -> xgb.Booster:
    """Fit the stage 0 model: ENS shortwave and companions to CAMS irradiance.

    Args:
        ens_train: The ENS control member's solar rows, with `lead_hours`.
        cams: `time_series_id`, `valid_time`, and `shortwave_cams` (CAMS irradiance, W/m2).
        seed: The XGBoost seed.
        before: Exclusive end of the training valid times.

    Returns:
        The fitted model, with `common.STAGE0_FEATURES` as inputs.
    """
    rows = (
        ens_train.filter(
            (pl.col("valid_time") < before) & (pl.col("solar_zenith") < common.MAX_ZENITH_DEGREES)
        )
        .join(cams, on=["time_series_id", "valid_time"], how="inner", validate="m:1")
        .drop_nulls("shortwave_cams")
    )
    return xgb.train(
        {**common.STAGE0_PARAMETERS, "seed": seed},
        xgb.DMatrix(
            _matrix(frame=rows, columns=list(common.STAGE0_FEATURES)),
            label=rows["shortwave_cams"].to_numpy(),
        ),
        num_boost_round=common.STAGE0_ROUNDS,
    )


def apply_stage0(*, booster: xgb.Booster, frame: pl.DataFrame) -> np.ndarray:
    """Return stage 0's irradiance for each row: zero at night, never negative."""
    predicted = booster.predict(
        xgb.DMatrix(_matrix(frame=frame, columns=list(common.STAGE0_FEATURES)))
    )
    daylight = frame["solar_zenith"].to_numpy() < common.MAX_ZENITH_DEGREES
    return np.where(daylight, np.clip(predicted, 0.0, None), 0.0)


def training_rows(
    *, series: int, analysis: pl.DataFrame, observed: pl.DataFrame, arm: common.ArmType
) -> pl.DataFrame:
    """Return one series' ERA5 (or CAMS-irradiance) rows with their target for an arm."""
    start = common.STARTS.get(arm, common.ERA5_START)
    rows = (
        analysis.filter(pl.col("time_series_id") == series)
        .join(
            observed.filter(pl.col("time_series_id") == series),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="1:1",
        )
        .filter((pl.col("valid_time") >= start) & (pl.col("valid_time") < common.TRAIN_END))
    )
    if arm in ("cams", "cams_stage0"):
        rows = rows.with_columns(shortwave=pl.col("shortwave_cams")).drop_nulls("shortwave")
    return rows


def fit_arm(
    *, rows: pl.DataFrame, arm: common.ArmType, technology: common.TechnologyType, seed: int
) -> FittedArm:
    """Fit one arm on its training rows.

    Args:
        rows: `training_rows(...)` for the series and arm.
        arm: The arm to fit.
        technology: The series' technology.
        seed: The XGBoost seed.

    Returns:
        The fitted arm.

    Raises:
        ValueError: If the arm has no training rows.
    """
    if rows.height == 0:
        msg = f"arm {arm} has no training rows"
        raise ValueError(msg)
    columns = arm_columns(arm=arm, technology=technology)
    parameters = {**common.XGBOOST_PARAMETERS, "seed": seed}
    features = _matrix(frame=rows, columns=columns)
    target = rows["target"].to_numpy()
    if arm == "csr":
        clear_sky = common.clear_sky_ghi(solar_zenith=rows["solar_zenith"].to_numpy())
        constant = common.clear_sky_constant(power_fraction=target, clear_sky=clear_sky)
        clear_sky_power = constant * clear_sky
        usable = clear_sky_power >= common.MIN_CLEAR_SKY_FRACTION
        ratio = np.clip(target[usable] / clear_sky_power[usable], 0.0, common.MAX_RATIO)
        matrix = xgb.DMatrix(features[usable], label=ratio, weight=clear_sky_power[usable] ** 2)
        booster = xgb.train(parameters, matrix, num_boost_round=common.N_ROUNDS)
        return FittedArm(arm=arm, booster=booster, columns=columns, clear_sky_constant=constant)
    if arm == "base_margin":
        speeds, curve = common.power_curve(
            speed=rows["wind_speed_100m"].to_numpy(), power_fraction=target
        )
        margin = np.interp(rows["wind_speed_100m"].to_numpy(), speeds, curve)
        matrix = xgb.DMatrix(features, label=target, base_margin=margin)
        booster = xgb.train(parameters, matrix, num_boost_round=common.N_ROUNDS)
        with_margin = booster.predict(matrix)
        without_margin = booster.predict(xgb.DMatrix(features))
        if np.allclose(with_margin, without_margin):
            msg = "the base margin did not change the predictions"
            raise ValueError(msg)
        return FittedArm(arm=arm, booster=booster, columns=columns, curve=(speeds, curve))
    booster = xgb.train(
        parameters, xgb.DMatrix(features, label=target), num_boost_round=common.N_ROUNDS
    )
    return FittedArm(arm=arm, booster=booster, columns=columns)


def predict_arm(*, fitted: FittedArm, frame: pl.DataFrame) -> np.ndarray:
    """Return the arm's power fraction for every row of `frame` (the weather already in place)."""
    features = _matrix(frame=frame, columns=fitted.columns)
    if fitted.arm == "base_margin" and fitted.curve is not None:
        speeds, curve = fitted.curve
        margin = np.interp(frame["wind_speed_100m"].to_numpy(), speeds, curve)
        return fitted.booster.predict(xgb.DMatrix(features, base_margin=margin))
    predicted = fitted.booster.predict(xgb.DMatrix(features))
    if fitted.arm == "csr" and fitted.clear_sky_constant is not None:
        clear_sky = common.clear_sky_ghi(solar_zenith=frame["solar_zenith"].to_numpy())
        ratio = np.clip(predicted, 0.0, common.MAX_RATIO)
        return ratio * fitted.clear_sky_constant * clear_sky
    return predicted


def weather_for_arm(
    *, arm: common.ArmType, frame: pl.DataFrame, stage0: xgb.Booster | None
) -> pl.DataFrame:
    """Return the frame with the shortwave an arm reads: stage 0's for `cams_stage0`."""
    if arm != "cams_stage0":
        return frame
    if stage0 is None:
        msg = "arm cams_stage0 needs a stage 0 model"
        raise ValueError(msg)
    return frame.with_columns(shortwave=pl.Series(apply_stage0(booster=stage0, frame=frame)))


def member_average(*, frame: pl.DataFrame, point: np.ndarray, arm: str) -> pl.DataFrame:
    """Average an arm's per-member predictions over the 51 members of each case.

    Args:
        frame: The ENS rows the predictions were made for, with the key columns and
            `ensemble_member`.
        point: The arm's power fraction for each row of `frame`.
        arm: The arm's name.

    Returns:
        One row per case with the keys, `point_norm` (the member average), `control_norm` (member
        0's value), and `arm`.

    Raises:
        ValueError: If a case does not hold every member.
    """
    averaged = (
        frame.select(*common.KEY_COLUMNS)
        .with_columns(point_norm=pl.Series(point), ensemble_member=frame["ensemble_member"])
        .group_by(common.KEY_COLUMNS)
        .agg(
            pl.col("point_norm").mean(),
            control_norm=pl.col("point_norm").filter(pl.col("ensemble_member") == 0).first(),
            members=pl.len(),
        )
    )
    if (averaged["members"] != common.MEMBERS).any():
        msg = f"arm {arm}: a case does not have all {common.MEMBERS} members"
        raise ValueError(msg)
    return averaged.drop("members").with_columns(arm=pl.lit(arm)).sort(common.KEY_COLUMNS)


def stage0_sanity(*, ens_train: pl.DataFrame, cams: pl.DataFrame) -> None:
    """Write stage 0's error against CAMS on the last 6 weeks, from a fit that stops before them."""
    booster = fit_stage0(ens_train=ens_train, cams=cams, seed=0, before=STAGE0_SANITY_START)
    held_out = (
        ens_train.filter(
            (pl.col("valid_time") >= STAGE0_SANITY_START)
            & (pl.col("valid_time") < common.TRAIN_END)
            & (pl.col("solar_zenith") < common.MAX_ZENITH_DEGREES)
        )
        .join(cams, on=["time_series_id", "valid_time"], how="inner", validate="m:1")
        .drop_nulls("shortwave_cams")
    )
    stage0 = apply_stage0(booster=booster, frame=held_out)
    truth = held_out["shortwave_cams"].to_numpy()
    raw = held_out["shortwave"].to_numpy()
    lines = [
        "# Stage 0 sanity (seed 0, fit stops before 2025-05-06, scored on the next 6 weeks)",
        "",
        f"- Held-out daylight rows: {held_out.height:,}",
        (
            f"- Raw ENS control-member shortwave: mean absolute error "
            f"{np.abs(raw - truth).mean():.1f} W/m2, bias {(raw - truth).mean():+.1f}"
        ),
        (
            f"- Stage 0: mean absolute error {np.abs(stage0 - truth).mean():.1f} W/m2, bias "
            f"{(stage0 - truth).mean():+.1f}"
        ),
    ]
    STAGE0_SANITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE0_SANITY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def write_ceiling(
    *,
    seed: int,
    sites: pl.DataFrame,
    analysis: pl.DataFrame,
    fitted: dict[tuple[int, common.ArmType], FittedArm],
) -> None:
    """Apply arm `w` to ERA5 weather at the validation valid times and write the result."""
    ceiling_target = CEILING_DIR / f"seed{seed}.parquet"
    window = analysis.filter(
        (pl.col("valid_time") >= common.VALIDATION_START) & (pl.col("valid_time") < common.HOLDOUT)
    )
    ceiling = [
        window.filter(pl.col("time_series_id") == series)
        .select("time_series_id", "valid_time")
        .with_columns(
            point_norm=pl.Series(
                predict_arm(
                    fitted=fitted[series, "w"],
                    frame=window.filter(pl.col("time_series_id") == series),
                )
            )
        )
        for series in sites["time_series_id"].to_list()
    ]
    CEILING_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(ceiling).write_parquet(ceiling_target.with_suffix(".tmp"))
    ceiling_target.with_suffix(".tmp").rename(ceiling_target)


def fit_seed(
    *,
    seed: int,
    sites: pl.DataFrame,
    analysis: pl.DataFrame,
    observed: pl.DataFrame,
    ens_train: pl.DataFrame,
    cams: pl.DataFrame,
) -> None:
    """Fit every arm for one seed and write the member averages and the ceiling."""
    target = MEMBER_AVERAGE_DIR / f"seed{seed}.parquet"
    ceiling_target = CEILING_DIR / f"seed{seed}.parquet"
    if target.exists() and ceiling_target.exists():
        return
    started = time.monotonic()
    stage0 = fit_stage0(ens_train=ens_train, cams=cams, seed=seed, before=common.TRAIN_END)
    fitted: dict[tuple[int, common.ArmType], FittedArm] = {}
    for row in sites.iter_rows(named=True):
        series, technology = int(row["time_series_id"]), row["technology"]
        arms = common.SOLAR_ARMS if technology == "solar" else common.WIND_ARMS
        for arm in arms:
            if arm == "cams_stage0":
                fitted[series, arm] = fitted[series, "cams"]
                continue
            rows = training_rows(series=series, analysis=analysis, observed=observed, arm=arm)
            fitted[series, arm] = fit_arm(rows=rows, arm=arm, technology=technology, seed=seed)
    print(f"seed {seed}: fits in {time.monotonic() - started:.0f} s", flush=True)

    pieces: list[pl.DataFrame] = []
    for path in sorted(common.ENS_VALIDATION_DIR.glob("init=*.parquet")):
        frame = pl.read_parquet(path).with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        for (series_key,), part in frame.partition_by("time_series_id", as_dict=True).items():
            series = int(series_key)
            if (series, "w") not in fitted:
                continue
            for (fitted_series, arm_name), model in fitted.items():
                if fitted_series != series:
                    continue
                inputs = weather_for_arm(arm=arm_name, frame=part, stage0=stage0)
                point = predict_arm(fitted=model, frame=inputs)
                pieces.append(member_average(frame=part, point=point, arm=arm_name))
    MEMBER_AVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)

    write_ceiling(seed=seed, sites=sites, analysis=analysis, fitted=fitted)
    print(f"seed {seed} done in {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Fit every seed, skipping seeds already written."""
    sites = common.site_table()
    analysis = pl.read_parquet(common.ANALYSIS_PATH)
    observed = common.observed_training_power(sites=sites)
    solar_ids = sites.filter(pl.col("technology") == "solar")["time_series_id"].to_list()
    ens_train = (
        pl.read_parquet(common.ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id").is_in(solar_ids))
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
    )
    cams = analysis.filter(pl.col("time_series_id").is_in(solar_ids)).select(
        "time_series_id", "valid_time", "shortwave_cams"
    )
    if not STAGE0_SANITY_PATH.exists():
        stage0_sanity(ens_train=ens_train, cams=cams)
    for seed in common.SEEDS:
        fit_seed(
            seed=seed,
            sites=sites,
            analysis=analysis,
            observed=observed,
            ens_train=ens_train,
            cams=cams,
        )


if __name__ == "__main__":
    main()
