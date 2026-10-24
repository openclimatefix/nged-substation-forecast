"""The model variants of experiment 00276 and the code that fits and applies them.

A `Spec` names one variant: a physics core (with options), a binned power curve, or an XGBoost
quantile model on a feature list, plus how its training rows are chosen (mask, window) and whether
the ERA5 speeds are rescaled. `fit_predict` fits one spec on one farm's training rows and returns
11 quantiles for each evaluation frame it is given.
"""

import os
from dataclasses import dataclass, field
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00276_common as common  # noqa: E402
import e00276_core as core_module  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

N_ROUNDS: Final[int] = 500
XGBOOST_PARAMETERS: Final[dict] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": common.LEVELS,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 2,
}
"""Experiment 230's XGBoost settings, with column subsampling stated as off."""
SPEED_BIN_WIDTH_M_S: Final[float] = 0.5
FIRST_BLOCK_YEAR: Final[int] = 2019
LAST_TRAIN_BLOCK: Final[int] = 5
"""Index of the July-to-June year ending at `common.TRAIN_END`, whose factor later years reuse."""

BASE = ("wind_speed_100m",)
DIRECTION_100 = ("wind_dir_sin_100m", "wind_dir_cos_100m")
LEVEL_10 = ("wind_speed_10m", "wind_dir_sin_10m", "wind_dir_cos_10m")
DATE_HOUR_DOY = ("hour", "doy")
FEATURE_SETS: Final[dict[str, tuple[str, ...]]] = {
    "X": common.XGBOOST_FEATURES,
    "X1": BASE,
    "X2": (*BASE, *DIRECTION_100),
    "X3": (*BASE, *DIRECTION_100, "temperature_c"),
    "X4": (*BASE, *DIRECTION_100, "temperature_c", *LEVEL_10),
    "X5": (*BASE, *DIRECTION_100, "temperature_c", *LEVEL_10, *DATE_HOUR_DOY),
    "X6": (*BASE, *DIRECTION_100, "temperature_c", *LEVEL_10, *DATE_HOUR_DOY, "year_frac"),
}
"""D4's cumulative feature sets, and experiment 230's six features (`X`)."""


@dataclass(frozen=True)
class Spec:
    """One model variant.

    Attributes:
        name: Short name used in file names and the report.
        kind: `core`, `binned`, or `xgb`.
        features: XGBoost feature columns (`xgb` only).
        options: The core's extra terms (`core` only).
        loss: `l1` (core plus residual quantiles) or `pinball` (joint fit).
        bins_on: What the residual quantiles are binned on: `expected` or `speed`.
        train_mask: Whether experiment 210's idle-band rows are dropped from training.
        window_months: Training length in months before `common.TRAIN_END`, or None for all.
        scale_fit: Whether ERA5 speeds are multiplied by a fitted farm-specific factor.
        ens: Whether the variant is also applied to the 51 ensemble members.
        seeds: The seeds to fit (deterministic variants use one).
    """

    name: str
    kind: str
    features: tuple[str, ...] = ()
    options: core_module.CoreOptions = core_module.DEFAULT_OPTIONS
    loss: str = "l1"
    bins_on: str = "expected"
    train_mask: bool = False
    window_months: int | None = None
    scale_fit: bool = False
    ens: bool = False
    seeds: tuple[int, ...] = field(default=(0,))


def _xgb(
    name: str,
    features: str,
    *,
    ens: bool = False,
    train_mask: bool = False,
    window_months: int | None = None,
    scale_fit: bool = False,
) -> Spec:
    """Return an XGBoost spec on the named feature set, fitted for the three seeds."""
    return Spec(
        name=name,
        kind="xgb",
        features=FEATURE_SETS[features],
        ens=ens,
        seeds=common.SEEDS,
        train_mask=train_mask,
        window_months=window_months,
        scale_fit=scale_fit,
    )


def build_specs() -> dict[str, Spec]:
    """Return every variant of the experiment, by name."""
    specs: list[Spec] = [
        Spec("P", "core", ens=True),
        Spec("S", "binned", ens=True),
        _xgb("X", "X", ens=True),
        Spec("P_nm", "core", train_mask=True, ens=True),
        _xgb("X_nm", "X", train_mask=True, ens=True),
        _xgb("X1", "X1"),
        _xgb("X2", "X2"),
        _xgb("X3", "X3"),
        _xgb("X4", "X4", ens=True),
        _xgb("X5", "X5", ens=True),
        _xgb("X6", "X6", ens=True),
        Spec("P_noshear", "core", options=core_module.CoreOptions(shear=False)),
        Spec("P_dir", "core", options=core_module.CoreOptions(direction_terms=True)),
        Spec("P_dens", "core", options=core_module.CoreOptions(free_density_exponent=True)),
        Spec("P_net", "core", options=core_module.CoreOptions(monotone_net=True)),
        Spec("P_bin_speed", "core", bins_on="speed", ens=True),
        Spec("P_pinball", "core", loss="pinball", ens=True),
        Spec(
            "P_year",
            "core",
            options=core_module.CoreOptions(group_count=LAST_TRAIN_BLOCK + 1),
            ens=True,
        ),
        Spec("P_month", "core", options=core_module.CoreOptions(group_count=12), ens=True),
        Spec("P_scaled", "core", scale_fit=True),
        _xgb("X_scaled", "X", scale_fit=True),
    ]
    for months in (3, 6, 12, 24, 48):
        specs.append(Spec(f"P_w{months}", "core", window_months=months))
        specs.append(_xgb(f"X_w{months}", "X", window_months=months))
    return {spec.name: spec for spec in specs}


def add_derived_columns(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `hour`, `doy`, `year_frac`, `block` (July-to-June year index), and `month_index`."""
    valid = pl.col("valid_time")
    return frame.with_columns(
        hour=(valid.dt.hour() + valid.dt.minute() / 60).cast(pl.Float32),
        doy=valid.dt.ordinal_day().cast(pl.Float32),
        year_frac=(valid.dt.year() + (valid.dt.ordinal_day() - 1) / 366).cast(pl.Float32),
        block=(valid.dt.offset_by("-6mo").dt.year() - FIRST_BLOCK_YEAR)
        .clip(0, LAST_TRAIN_BLOCK)
        .cast(pl.Int64),
        month_index=(valid.dt.month() - 1).cast(pl.Int64),
    )


def weather_arrays(
    *, frame: pl.DataFrame, options: core_module.CoreOptions
) -> core_module.WeatherArrays:
    """Return the arrays the core reads, including the direction and group columns it needs."""
    group = None
    if options.group_count == 12:
        group = frame["month_index"].to_numpy()
    elif options.group_count:
        group = frame["block"].to_numpy()
    return core_module.WeatherArrays(
        speed_10m=frame["wind_speed_10m"].to_numpy(),
        speed_100m=frame["wind_speed_100m"].to_numpy(),
        temperature_c=frame["temperature_c"].to_numpy(),
        direction_sin=frame["wind_dir_sin_100m"].to_numpy() if options.direction_terms else None,
        direction_cos=frame["wind_dir_cos_100m"].to_numpy() if options.direction_terms else None,
        group=group,
    )


def binned_curve(*, speed: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a non-decreasing power curve of the fraction on 100 m wind speed, by 0.5 m/s bin."""
    bins = np.floor(speed / SPEED_BIN_WIDTH_M_S).astype(np.int64)
    ordered = (
        pl.DataFrame({"bin": bins, "power": target})
        .group_by("bin")
        .agg(pl.col("power").mean())
        .sort("bin")
    )
    centres = (ordered["bin"].to_numpy() + 0.5) * SPEED_BIN_WIDTH_M_S
    return centres, np.maximum.accumulate(ordered["power"].to_numpy())


def select_training_rows(*, train: pl.DataFrame, spec: Spec, mask: pl.DataFrame) -> pl.DataFrame:
    """Return the farm's training rows after the spec's mask and window."""
    rows = train
    if spec.train_mask:
        rows = rows.join(mask, on=["time_series_id", "valid_time"], how="anti")
    if spec.window_months is not None:
        start = common.TRAIN_END.replace(tzinfo=None)
        rows = rows.filter(
            pl.col("valid_time")
            >= pl.lit(start).dt.replace_time_zone("UTC").dt.offset_by(f"-{spec.window_months}mo")
        )
    return rows


def _scale_speeds(*, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Return `frame` with both ERA5 or ensemble wind speeds multiplied by `factor`."""
    return frame.with_columns(
        pl.col("wind_speed_10m", "wind_speed_100m") * pl.lit(factor, dtype=pl.Float32)
    )


def fit_predict(
    *,
    spec: Spec,
    train: pl.DataFrame,
    evals: dict[str, pl.DataFrame],
    seed: int,
    with_idle_layer: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit `spec` on `train` and return 11 quantiles for each frame in `evals`, plus fit facts.

    Args:
        spec: The variant.
        train: One farm's selected training rows (weather, derived columns, `target`).
        evals: Frames to predict, by name; each has the weather and derived columns.
        seed: The XGBoost or network seed.
        with_idle_layer: Whether the core has the apparent-power idle layer (false for W2).

    Returns:
        The quantiles by frame name, shape (rows, 11), and a dictionary of fitted values.
    """
    facts: dict = {}
    target = train["target"].to_numpy()
    options = spec.options
    if spec.scale_fit:
        factor = core_module.fit_speed_scale(
            weather=weather_arrays(frame=train, options=core_module.DEFAULT_OPTIONS), target=target
        )
        facts["speed_scale"] = factor
        train = _scale_speeds(frame=train, factor=factor)
        evals = {name: _scale_speeds(frame=frame, factor=factor) for name, frame in evals.items()}
    if spec.kind == "xgb":
        return _xgb_predict(spec=spec, train=train, evals=evals, seed=seed), facts
    speed = train["wind_speed_100m"].to_numpy()
    if spec.kind == "binned":
        centres, curve = binned_curve(speed=speed, target=target)
        residuals = core_module.ResidualQuantiles.fit(
            expected=np.interp(speed, centres, curve), observed=target
        )
        return {
            name: residuals.quantiles(
                expected=np.interp(f["wind_speed_100m"].to_numpy(), centres, curve)
            )
            for name, f in evals.items()
        }, facts
    weather = weather_arrays(frame=train, options=options)
    if spec.loss == "pinball":
        fitted = core_module.fit_core_pinball(
            weather=weather, target=target, with_idle_layer=with_idle_layer, seed=seed
        )
        facts["cut_in_m_s"] = float(fitted.core.cut_in_m_s.detach())
        facts["rated_m_s"] = float(fitted.core.rated_m_s.detach())
        return {
            name: fitted.quantiles(weather=weather_arrays(frame=f, options=options))
            for name, f in evals.items()
        }, facts
    core = core_module.fit_core(
        weather=weather,
        target=target,
        with_idle_layer=with_idle_layer,
        seed=seed,
        options=options,
    )
    facts["cut_in_m_s"] = float(core.cut_in_m_s.detach())
    facts["rated_m_s"] = float(core.rated_m_s.detach())
    facts["hub_height_m"] = float(core.hub_height_m.detach())
    expected_train = core_module.predict_expected_power(core=core, weather=weather)
    bin_train = (
        core_module.predict_hub_speed(core=core, weather=weather)
        if spec.bins_on == "speed"
        else None
    )
    residuals = core_module.ResidualQuantiles.fit(
        expected=expected_train, observed=target, bin_variable=bin_train
    )
    out = {}
    for name, f in evals.items():
        w = weather_arrays(frame=f, options=options)
        expected = core_module.predict_expected_power(core=core, weather=w)
        bins = (
            core_module.predict_hub_speed(core=core, weather=w) if spec.bins_on == "speed" else None
        )
        out[name] = residuals.quantiles(expected=expected, bin_variable=bins)
    return out, facts


def _xgb_predict(
    *, spec: Spec, train: pl.DataFrame, evals: dict[str, pl.DataFrame], seed: int
) -> dict[str, np.ndarray]:
    """Return XGBoost quantiles with experiment 230's binned-curve base margin."""
    target = train["target"].to_numpy()
    speed = train["wind_speed_100m"].to_numpy()
    centres, curve = binned_curve(speed=speed, target=target)
    columns = list(spec.features)
    booster = xgb.train(
        {**XGBOOST_PARAMETERS, "seed": seed},
        xgb.DMatrix(
            train.select(columns).to_numpy(),
            label=target,
            base_margin=np.tile(np.interp(speed, centres, curve)[:, None], (1, 11)),
        ),
        num_boost_round=N_ROUNDS,
    )
    out = {}
    for name, f in evals.items():
        margin = np.tile(
            np.interp(f["wind_speed_100m"].to_numpy(), centres, curve)[:, None], (1, 11)
        )
        predicted = booster.predict(xgb.DMatrix(f.select(columns).to_numpy(), base_margin=margin))
        out[name] = np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32)
    return out
