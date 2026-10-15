"""Fit the generator arms of experiment 00033 that no earlier experiment stored.

Per seed and series this fits (a) N1 for generators: a 9-level quantile XGBoost model trained on the
ENS control member of every daily run with valid times from 2024-04-01 to 2025-06-30, recency
weights with a half-life of 2 years anchored at 2025-06-30, the technology's common columns,
`lead_hours`, and three power lags from the forecast origin, and (b) the ERA5-trained models of the
best guess (solar `base` and `cams`, wind `margin`, as in experiment 00020) for the perfect-weather
arms. The perfect-weather arms feed ERA5 (or CAMS `shortwave`) at the valid time: either to every
column (one input per case, so the forecast is the model's 9 quantiles) or to one column only while
the 51 ENS members supply the rest (a pooled mixture, as in 00020).

Output: `work/generator_arms/<arm>_<technology>_seed<s>.parquet`, keyed by series, run, and valid
time, with the nine quantile columns as a fraction of capacity. Existing files are kept.

    uv run python ml-experiments/00033-*/e00033_fit_generators.py [--smoke]
"""

import argparse
import time
from typing import Final

import e00033_common as demand
import numpy as np
import polars as pl
import xgboost as xgb
from e00033_gen_common import (
    ANALYSIS_PATH,
    ENS_TRAIN_PATH,
    ENS_VALIDATION_DIR,
    ERA5_START,
    KEY_COLUMNS,
    LEVEL_COLUMNS,
    MEMBERS,
    N_ROUNDS,
    SEEDS,
    SOLAR_FEATURES,
    TRAIN_END,
    WIND_FEATURES,
    mixture_quantiles,
    observed_training_power,
    power_curve,
    read_levels,
    site_table,
    validation_cases,
)
from e00033_gen_common import XGBOOST_PARAMETERS as PARAMETERS
from studies.power import scan_power


def _matrix(*, frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


class FittedQuantiles:
    """One series' fitted quantile model, with the wind power curve when a base margin is used."""

    def __init__(
        self,
        *,
        rows: pl.DataFrame,
        columns: tuple[str, ...],
        seed: int,
        rounds: int,
        use_margin: bool,
    ) -> None:
        """Fit the model; `use_margin` makes the power curve the base margin of all 9 levels."""
        if rows.height == 0:
            msg = "no training rows"
            raise ValueError(msg)
        self.columns = columns
        self.curve: tuple[np.ndarray, np.ndarray] | None = None
        features = _matrix(frame=rows, columns=columns)
        target = rows["target"].to_numpy()
        margin = None
        if use_margin:
            self.curve = power_curve(
                speed=rows["wind_speed_100m"].to_numpy(), power_fraction=target
            )
            margin = self._margin(speed=rows["wind_speed_100m"].to_numpy())
        matrix = xgb.DMatrix(features, label=target, base_margin=margin)
        self.booster = xgb.train({**PARAMETERS, "seed": seed}, matrix, num_boost_round=rounds)
        if margin is not None:
            plain = self.booster.predict(xgb.DMatrix(features[:2000]))
            with_margin = self.booster.predict(
                xgb.DMatrix(features[:2000], base_margin=margin[:2000])
            )
            if np.allclose(plain, with_margin):
                msg = "the base margin did not change the predictions"
                raise ValueError(msg)

    def _margin(self, *, speed: np.ndarray) -> np.ndarray:
        if self.curve is None:
            msg = "no power curve"
            raise ValueError(msg)
        centres, curve = self.curve
        return np.tile(np.interp(speed, centres, curve)[:, None], (1, len(LEVEL_COLUMNS)))

    def predict(self, *, frame: pl.DataFrame) -> np.ndarray:
        """Return the 9 sorted quantile columns for every row (a fraction of capacity)."""
        margin = (
            None if self.curve is None else self._margin(speed=frame["wind_speed_100m"].to_numpy())
        )
        matrix = xgb.DMatrix(_matrix(frame=frame, columns=self.columns), base_margin=margin)
        return np.sort(self.booster.predict(matrix), axis=1)


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


OUT_DIR: Final = demand.GENERATOR_ARMS_DIR
N1_START: Final = demand.datetime(2024, 4, 1, tzinfo=demand.UTC)
CORE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "solar": SOLAR_FEATURES,
    "wind": WIND_FEATURES,
}
N1_EXTRA: Final[tuple[str, ...]] = ("lead_hours", *demand.LAG_COLUMNS)
HALF_LIFE_DAYS: Final[float] = demand.RECENCY_HALF_LIFE_DAYS
ERA5_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "solar": SOLAR_FEATURES,
    "wind": WIND_FEATURES,
}
ARM_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "solar": ("n1", "pw_era5", "pw_cams", "pw_sw_era5", "pw_sw_cams"),
    "wind": ("n1", "pw_era5", "pw_ws100"),
}
"""The arms written per technology; `pw_` arms are the ERA5-weather ceiling."""


def full_power(*, sites: pl.DataFrame) -> dict[int, np.ndarray]:
    """Return each series' observed power over capacity on the dense half-hourly grid."""
    grid = pl.DataFrame({"valid_time": demand.grid_times()})
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= ERA5_START,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
    )
    out = {}
    for row in sites.iter_rows(named=True):
        joined = grid.join(
            power.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                "valid_time", "power"
            ),
            on="valid_time",
            how="left",
            validate="1:1",
        )
        out[int(row["time_series_id"])] = (
            joined["power"].to_numpy().astype(np.float64) / row["effective_capacity_mw"]
        ).astype(np.float32)
    return out


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add `lead_hours` and the three origin lags to rows with an origin and a valid time."""
    lags = demand.origin_lags(
        power=power,
        target=np.asarray(demand.grid_index(times=frame["valid_time"])),
        origin=np.asarray(demand.grid_index(times=frame["power_fcst_init_time"])),
    )
    return frame.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    ).with_columns(pl.DataFrame(lags, schema=list(demand.LAG_COLUMNS), orient="row"))


def era5_rows(
    *, series: int, analysis: pl.DataFrame, observed: pl.DataFrame, cams: bool
) -> pl.DataFrame:
    """Return one series' ERA5 rows with their target; CAMS irradiance replaces the shortwave."""
    rows = (
        analysis.filter(pl.col("time_series_id") == series)
        .join(
            observed.filter(pl.col("time_series_id") == series),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="1:1",
        )
        .filter((pl.col("valid_time") >= ERA5_START) & (pl.col("valid_time") < TRAIN_END))
    )
    if cams:
        rows = rows.with_columns(shortwave=pl.col("shortwave_cams")).drop_nulls("shortwave")
    return rows


def fit_n1(
    *, row: dict, ens_train: pl.DataFrame, power: np.ndarray, seed: int, rounds: int
) -> xgb.Booster:
    """Fit N1 for one generator on the ENS control member's rows."""
    if ens_train["ensemble_member"].unique().to_list() != [0]:
        msg = "the training frame must hold the control member (0) only"
        raise ValueError(msg)
    own = ens_train.filter(
        (pl.col("time_series_id") == row["time_series_id"])
        & pl.col("power").is_finite()
        & (pl.col("valid_time") >= N1_START)
        & (pl.col("valid_time") <= demand.LAST_TRAINING_VALID)
        & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    own = add_lags(frame=own, power=power)
    columns = [*CORE_COLUMNS[row["technology"]], *N1_EXTRA]
    age_days = (demand.LAST_TRAINING_VALID - own["valid_time"]).dt.total_seconds() / 86_400
    return xgb.train(
        {**PARAMETERS, "seed": seed},
        xgb.DMatrix(
            own.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy(),
            label=(own["power"] / row["effective_capacity_mw"]).to_numpy(),
            weight=0.5 ** (age_days.to_numpy() / HALF_LIFE_DAYS),
        ),
        num_boost_round=rounds,
    )


def _quantile_frame(*, keys: pl.DataFrame, quantiles: np.ndarray) -> pl.DataFrame:
    return keys.hstack(pl.DataFrame(quantiles, schema=list(LEVEL_COLUMNS)))


def predict_series(
    *,
    part: pl.DataFrame,
    technology: str,
    models: dict[str, FittedQuantiles | xgb.Booster],
    era5: pl.DataFrame,
    power: np.ndarray,
    cases: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Return every arm's table for one series and one run.

    Raises:
        ValueError: If ERA5 lacks a valid time of the run.
    """
    at_valid = part.select("valid_time").join(era5, on="valid_time", how="left")
    if at_valid[ERA5_COLUMNS[technology][0]].null_count():
        msg = "ERA5 is missing for some valid time of a run"
        raise ValueError(msg)
    n1_part = add_lags(frame=part, power=power)
    n1_columns = [*CORE_COLUMNS[technology], *N1_EXTRA]
    n1_model = models["n1"]
    raw = n1_model.predict(
        xgb.DMatrix(n1_part.select([pl.col(c).cast(pl.Float32) for c in n1_columns]).to_numpy())
    )
    out = {"n1": pooled_mixture(part=part, member_quantiles=np.sort(raw, axis=1))}
    keys = (
        part.select(KEY_COLUMNS)
        .unique(maintain_order=True)
        .join(cases, on=list(KEY_COLUMNS), how="semi")
    )
    if keys.is_empty():
        return out
    one_input = keys.join(era5, on="valid_time", how="left")
    if technology == "solar":
        base, cams = models["base"], models["cams"]
        out["pw_era5"] = _quantile_frame(keys=keys, quantiles=base.predict(frame=one_input))
        out["pw_cams"] = _quantile_frame(
            keys=keys,
            quantiles=cams.predict(
                frame=one_input.with_columns(shortwave=pl.col("shortwave_cams"))
            ),
        )
        for arm, model, source in (
            ("pw_sw_era5", base, "shortwave"),
            ("pw_sw_cams", cams, "shortwave_cams"),
        ):
            swapped = part.drop("shortwave").join(
                era5.select("valid_time", shortwave=source), on="valid_time", how="left"
            )
            out[arm] = pooled_mixture(part=swapped, member_quantiles=model.predict(frame=swapped))
    else:
        margin = models["margin"]
        out["pw_era5"] = _quantile_frame(keys=keys, quantiles=margin.predict(frame=one_input))
        swapped = part.drop("wind_speed_100m").join(
            era5.select("valid_time", "wind_speed_100m"), on="valid_time", how="left"
        )
        out["pw_ws100"] = pooled_mixture(
            part=swapped, member_quantiles=margin.predict(frame=swapped)
        )
    return out


def fit_seed(*, seed: int, rounds: int, limit_runs: int | None) -> None:
    """Fit every generator arm for one seed and write its tables."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = site_table()
    targets = {
        (technology, arm): OUT_DIR / f"{arm}_{technology}_seed{seed}.parquet"
        for technology, arms in ARM_NAMES.items()
        for arm in arms
    }
    if limit_runs is None and all(p.exists() for p in targets.values()):
        return
    started = time.monotonic()
    analysis = pl.read_parquet(ANALYSIS_PATH)
    observed = observed_training_power(sites=sites)
    power = full_power(sites=sites)
    ens_train = pl.read_parquet(ENS_TRAIN_PATH)
    cases = validation_cases().select(KEY_COLUMNS)
    models: dict[tuple[int, str], FittedQuantiles | xgb.Booster] = {}
    for row in sites.iter_rows(named=True):
        series, technology = int(row["time_series_id"]), row["technology"]
        columns = ERA5_COLUMNS[technology]
        fit_names = ("base", "cams") if technology == "solar" else ("margin",)
        for fit_name in fit_names:
            models[series, fit_name] = FittedQuantiles(
                rows=era5_rows(
                    series=series, analysis=analysis, observed=observed, cams=fit_name == "cams"
                ),
                columns=columns,
                seed=seed,
                rounds=rounds,
                use_margin=fit_name == "margin",
            )
        models[series, "n1"] = fit_n1(
            row=row,
            ens_train=ens_train,
            power=power[series],
            seed=seed,
            rounds=rounds,
        )
    print(f"seed {seed}: fits in {time.monotonic() - started:.0f} s", flush=True)
    pieces: dict[tuple[str, str], list[pl.DataFrame]] = {key: [] for key in targets}
    technology_of = dict(zip(sites["time_series_id"], sites["technology"], strict=True))
    for path in sorted(ENS_VALIDATION_DIR.glob("init=*.parquet"))[:limit_runs]:
        frame = pl.read_parquet(path)
        for (series_key,), run_part in frame.partition_by("time_series_id", as_dict=True).items():
            series, technology = int(series_key), technology_of[series_key]
            arms = predict_series(
                part=run_part.sort(*KEY_COLUMNS, "ensemble_member"),
                technology=technology,
                models={
                    arm: models[series, arm]
                    for arm in ("n1", "base", "cams", "margin")
                    if (series, arm) in models
                },
                era5=analysis.filter(pl.col("time_series_id") == series),
                power=power[series],
                cases=cases,
            )
            for arm, table in arms.items():
                pieces[technology, arm].append(table)
    if limit_runs is not None:
        print("smoke run: no output written", flush=True)
        return
    for (technology, arm), frames in pieces.items():
        target = targets[technology, arm]
        pl.concat(frames).sort(*KEY_COLUMNS).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
    print(f"seed {seed} done in {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Fit every seed, skipping seeds already written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 rounds, 2 runs, no output")
    smoke = parser.parse_args().smoke
    for seed in SEEDS[:1] if smoke else SEEDS:
        fit_seed(seed=seed, rounds=20 if smoke else N_ROUNDS, limit_runs=2 if smoke else None)


if __name__ == "__main__":
    main()
