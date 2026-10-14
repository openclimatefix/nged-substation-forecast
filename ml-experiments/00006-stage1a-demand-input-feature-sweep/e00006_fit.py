"""Fit the input-feature sweep's XGBoost models and write each arm's validation predictions.

An arm is a name such as `W+clock+t24|recency`: the weather block `W`, the column groups after the
`+` signs, and optional treatments after the `|` (`block` replaces the nearest-cell weather by the
3 by 3 block mean, `recency` weights older rows less, `nocovid` drops the lockdown rows, and
`from2023-09` starts training later). Every arm in a fit uses identical XGBoost settings.

One unit of work is a fold and a series (or, for the pooled configuration, a fold and a substation
type). A unit writes one parquet file holding the observed target and one prediction column per
arm, and is skipped if the file exists, so a crash costs one unit.

    uv run python ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_fit.py --help
"""

import argparse
import json
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import e00006_inputs as inputs
import numpy as np
import polars as pl
import xgboost as xgb

FOLDS_DIR: Final[Path] = inputs.EXPERIMENT_DIR / "fits"
N_ROUNDS: Final[int] = 500
PERMUTATION_REPEATS: Final[int] = 3
PURGE: Final[timedelta] = timedelta(days=7)
MIN_TRAIN_ROWS: Final[int] = 17_520
"""A series needs a year of observed half-hours before a fold's purge to be fitted in that fold."""
RECENCY_HALF_LIFE_YEARS: Final[float] = 2.0
QUANTILE_LEVELS: Final[tuple[float, ...]] = (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95)
PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
}
"""Identical in every arm. Column subsampling is off so a wider set gets no free win."""

FoldNameType = Literal["select_2024", "select_2023", "confirm_2025"]


@dataclass(frozen=True)
class Fold:
    """A validation year and the training window that ends a purge before it."""

    valid_start: datetime
    valid_end: datetime

    @property
    def train_end(self) -> datetime:
        """The exclusive end of training: one week before validation starts."""
        return self.valid_start - PURGE


FOLDS: Final[dict[str, Fold]] = {
    "select_2024": Fold(datetime(2024, 7, 1, tzinfo=UTC), datetime(2025, 7, 1, tzinfo=UTC)),
    "select_2023": Fold(datetime(2023, 7, 1, tzinfo=UTC), datetime(2024, 7, 1, tzinfo=UTC)),
    "confirm_2025": Fold(datetime(2025, 7, 1, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)),
}
"""The selection folds decide every choice. The confirmation fold is scored once, after freezing."""

GROUP_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "clock": ("hour_local", "day_of_week_local"),
    "doy1": ("doy_sin_1", "doy_cos_1"),
    "doy2": ("doy_sin_1", "doy_cos_1", "doy_sin_2", "doy_cos_2"),
    "hol_name": ("holiday_name_code",),
    "hol_flags": ("is_holiday", "holiday_day_before", "holiday_day_after"),
    "hol_nearest": ("days_to_nearest_holiday",),
    "xmas": ("days_to_christmas",),
    "bridge": ("bridge_day",),
    "t24": ("temperature_mean_24h",),
    "t72": ("temperature_mean_72h",),
    "t7d": ("temperature_mean_7d",),
    "ewm_t": ("ewm_t_12h", "ewm_t_72h", "ewm_t_7d"),
    "hdh": ("heating_dh_24h", "heating_dh_72h"),
    "sun_day": ("day_length_h", "hours_since_sunrise"),
    "azimuth_sincos": ("azimuth_sin", "azimuth_cos"),
    "utc_offset": ("utc_offset_h",),
    "windchill": ("windchill_c",),
    "wind_heat_loss": ("wind_heat_loss",),
    "covid_flag": ("covid_indicator",),
    "covid_graded": ("covid_graded",),
    "month_cat": ("month_local",),
}
"""The column groups an arm can add to `W`, each computable at forecast time."""

TREATMENTS: Final[tuple[str, ...]] = ("block", "recency", "nocovid")
DROP_START: Final[datetime] = inputs.LOCKDOWN_START
DROP_END: Final[datetime] = inputs.LOCKDOWN_END + timedelta(days=1)


@dataclass(frozen=True)
class Arm:
    """One feature set with its treatments, parsed from and rendered as a name."""

    groups: tuple[str, ...]
    dropped: tuple[str, ...] = ()
    treatments: tuple[str, ...] = ()
    train_start: str | None = None
    series_code: Literal["none", "int", "cat"] = "none"

    @property
    def name(self) -> str:
        """The arm's name, such as `W+clock+t24|recency|from2023-09`."""
        text = "+".join(["W" + "".join(f"-{c}" for c in self.dropped), *self.groups])
        extras = [*self.treatments]
        if self.train_start:
            extras.append(f"from{self.train_start}")
        if self.series_code != "none":
            extras.append(f"id{self.series_code}")
        return "|".join([text, *extras])

    @property
    def columns(self) -> tuple[str, ...]:
        """The input columns in a fixed order, with the pooled series code last."""
        base = tuple(f"{c}_block" if "block" in self.treatments else c for c in inputs.W[:7])
        columns = [c for c in (*base, *inputs.W[7:]) if c not in self.dropped]
        for group in self.groups:
            columns += [c for c in GROUP_COLUMNS[group] if c not in columns]
        if self.series_code != "none":
            columns.append("series_code")
        return tuple(columns)

    @property
    def categorical(self) -> frozenset[str]:
        """The columns XGBoost treats as unordered categories."""
        return frozenset(
            c
            for c in self.columns
            if c in ("holiday_name_code", "month_local") or (c == "series_code" and self.series_code == "cat")
        )

    @classmethod
    def from_name(cls, name: str) -> "Arm":
        """Parse a name written by `name`."""
        head, *extras = name.split("|")
        groups = tuple(head.split("+")[1:])
        dropped = tuple(head.split("+")[0].split("-")[1:])
        unknown = [g for g in groups if g not in GROUP_COLUMNS]
        unknown += [c for c in dropped if c not in inputs.W]
        if head.split("+")[0].split("-")[0] != "W" or unknown:
            msg = f"unknown arm {name!r}"
            raise ValueError(msg)
        treatments = tuple(e for e in extras if e in TREATMENTS)
        start = next((e.removeprefix("from") for e in extras if e.startswith("from")), None)
        code = next((e.removeprefix("id") for e in extras if e in ("idint", "idcat")), "none")
        return cls(groups=groups, dropped=dropped, treatments=treatments, train_start=start, series_code=code)  # ty: ignore[invalid-argument-type]


def arm_names_with(*, base: Sequence[str], extra: Sequence[str]) -> list[str]:
    """Return arm names `W+base+extra_i` for each extra group, skipping groups already in `base`."""
    return [Arm(groups=(*base, g)).name for g in extra if g not in base]


def _train_start(*, arm: Arm, fold: Fold) -> datetime:
    if arm.train_start is None:
        return inputs.TRAIN_START
    year, month = arm.train_start.split("-")
    return datetime(int(year), int(month), 1, tzinfo=UTC)


def _matrix(*, frame: pl.DataFrame, arm: Arm) -> xgb.DMatrix:
    values = frame.select([pl.col(c).cast(pl.Float32) for c in arm.columns]).to_numpy()
    types = ["c" if c in arm.categorical else "q" for c in arm.columns]
    return xgb.DMatrix(values, feature_types=types, enable_categorical=True)


def _train_matrix(*, rows: pl.DataFrame, arm: Arm, weights: np.ndarray | None) -> xgb.DMatrix:
    matrix = _matrix(frame=rows, arm=arm)
    matrix.set_label(rows["target"].to_numpy())
    if weights is not None:
        matrix.set_weight(weights)
    return matrix


def _train_rows(*, arm: Arm, fold: Fold, frame: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray | None]:
    rows = frame.filter(
        pl.col("target").is_not_null(),
        pl.col("valid_time") >= _train_start(arm=arm, fold=fold),
        pl.col("valid_time") < fold.train_end,
    )
    if "nocovid" in arm.treatments:
        rows = rows.filter((pl.col("valid_time") < DROP_START) | (pl.col("valid_time") >= DROP_END))
    weights = None
    if "recency" in arm.treatments:
        age_years = (fold.train_end - rows["valid_time"]).dt.total_days() / 365.25
        weights = 0.5 ** (age_years.to_numpy() / RECENCY_HALF_LIFE_YEARS)
    return rows, weights


def fit_booster(
    *, arm: Arm, fold: Fold, frame: pl.DataFrame, seed: int, nthread: int, quantiles: bool
) -> tuple[xgb.Booster, pl.DataFrame]:
    """Fit one arm on a frame's training rows.

    Args:
        arm: The feature set and treatments.
        fold: The fold, which sets the training and validation windows.
        frame: Rows with every column in `arm.columns`, `target`, and `valid_time`.
        seed: The XGBoost seed.
        nthread: Threads for this fit.
        quantiles: Fit the 9 quantile levels instead of the median.

    Returns:
        The booster and the frame's validation rows.
    """
    rows, weights = _train_rows(arm=arm, fold=fold, frame=frame)
    valid = frame.filter(
        pl.col("target").is_not_null(),
        pl.col("valid_time") >= fold.valid_start,
        pl.col("valid_time") < fold.valid_end,
    )
    objective: dict[str, object] = (
        {"objective": "reg:quantileerror", "quantile_alpha": np.array(QUANTILE_LEVELS)}
        if quantiles
        else {"objective": "reg:absoluteerror"}
    )
    booster = xgb.train(
        {**PARAMETERS, **objective, "seed": seed, "nthread": nthread},
        _train_matrix(rows=rows, arm=arm, weights=weights),
        num_boost_round=N_ROUNDS,
    )
    return booster, valid


def importance_rows(
    *, arm: Arm, booster: xgb.Booster, valid: pl.DataFrame, unit: str, seed: int
) -> list[dict[str, object]]:
    """Return gain and permutation importance of each input column of a median model.

    Gain is the mean loss reduction of the column's splits, as a share of the total. Permutation
    importance is the rise in the validation mean absolute error, in points of capacity, when the
    column is shuffled across the validation rows. Gain favours columns with many distinct values,
    so the two are reported side by side.
    """
    gain = booster.get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    names = {f"f{i}": c for i, c in enumerate(arm.columns)}
    baseline_matrix = _matrix(frame=valid, arm=arm)
    target = valid["target"].to_numpy()
    baseline = float(np.abs(booster.predict(baseline_matrix) - target).mean()) * 100.0
    rng = np.random.default_rng(seed)
    rows = []
    for column in arm.columns:
        increases = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = valid.with_columns(pl.Series(column, rng.permutation(valid[column].to_numpy())))
            error = np.abs(booster.predict(_matrix(frame=shuffled, arm=arm)) - target).mean()
            increases.append(float(error) * 100.0 - baseline)
        rows.append(
            {
                "arm": arm.name,
                "unit": unit,
                "seed": seed,
                "column": column,
                "gain_share": gain.get(next(k for k, v in names.items() if v == column), 0.0) / total,
                "permutation_points": float(np.mean(increases)),
            }
        )
    return rows


def unit_path(*, stage: str, fold_name: str, config: str, unit: str, seed: int) -> Path:
    """Return where a unit's predictions are written."""
    return FOLDS_DIR / stage / fold_name / config / f"seed{seed}" / f"{unit}.parquet"


def _load(*, unit: str, config: str, fold: Fold) -> pl.DataFrame | None:
    """Load a unit's rows (one series, or every series of one type), or None if it cannot be fitted."""
    series = pl.read_parquet(inputs.INPUTS_DIR / "series.parquet")
    wanted = (
        series.filter(pl.col("time_series_id") == int(unit))
        if config == "series"
        else series.filter(pl.col("substation_type") == unit)
    )
    frame = (
        pl.scan_parquet(inputs.INPUTS_DIR / "features.parquet")
        .filter(pl.col("time_series_id").is_in(wanted["time_series_id"].to_list()))
        .collect()
    )
    enough = (
        frame.filter(pl.col("target").is_not_null(), pl.col("valid_time") < fold.train_end)
        .group_by("time_series_id")
        .len()
        .filter(pl.col("len") >= MIN_TRAIN_ROWS)["time_series_id"]
    )
    frame = frame.filter(pl.col("time_series_id").is_in(enough.to_list()))
    if frame.is_empty():
        return None
    codes = {sid: i for i, sid in enumerate(sorted(frame["time_series_id"].unique().to_list()))}
    return frame.with_columns(
        series_code=pl.col("time_series_id").replace_strict(codes, return_dtype=pl.Int32),
        month_local=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.month(),
    )


def run_unit(
    *,
    stage: str,
    fold_name: str,
    config: str,
    unit: str,
    arm_names: Sequence[str],
    seed: int,
    nthread: int,
    quantiles: bool,
    importance: bool = False,
) -> str:
    """Fit every arm for one unit and write the predictions, skipping a file that exists."""
    path = unit_path(stage=stage, fold_name=fold_name, config=config, unit=unit, seed=seed)
    if path.exists():
        return f"{path.name}: exists"
    fold = FOLDS[fold_name]
    frame = _load(unit=unit, config=config, fold=fold)
    if frame is None:
        return f"{unit}: no series has a year of training data in {fold_name}"
    valid = frame.filter(
        pl.col("target").is_not_null(),
        pl.col("valid_time") >= fold.valid_start,
        pl.col("valid_time") < fold.valid_end,
    ).select("time_series_id", "valid_time", "target")
    columns: dict[str, np.ndarray] = {}
    importances: list[dict[str, object]] = []
    for name in arm_names:
        arm = Arm.from_name(name)
        booster, valid_rows = fit_booster(
            arm=arm, fold=fold, frame=frame, seed=seed, nthread=nthread, quantiles=quantiles
        )
        predicted = booster.predict(_matrix(frame=valid_rows, arm=arm))
        if importance:
            importances += importance_rows(
                arm=arm, booster=booster, valid=valid_rows, unit=unit, seed=seed
            )
        if quantiles:
            for level, values in zip(QUANTILE_LEVELS, predicted.T, strict=True):
                columns[f"{name}@q{level}"] = values.astype(np.float32)
        else:
            columns[name] = predicted.astype(np.float32)
    out = valid.with_columns(*[pl.Series(k, v) for k, v in columns.items()])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    if importance:
        pl.DataFrame(importances).write_parquet(path.with_name(f"{unit}_importance.parquet"))
    return f"{path}: {out.height:,} rows, {len(arm_names)} arms"


def run_stage(
    *,
    stage: str,
    fold_name: str,
    config: str,
    arm_names: Sequence[str],
    seeds: Sequence[int],
    quantiles: bool = False,
    importance: bool = False,
    workers: int = 12,
) -> None:
    """Run every unit of a stage in parallel processes, one thread per fit."""
    series = pl.read_parquet(inputs.INPUTS_DIR / "series.parquet")
    units = (
        [str(i) for i in series["time_series_id"].to_list()]
        if config == "series"
        else sorted(series["substation_type"].unique().to_list())
    )
    nthread = 1 if config == "series" else max(1, workers // len(units))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(units) * len(seeds)), mp_context=context) as pool:
        futures = [
            pool.submit(
                run_unit,
                stage=stage,
                fold_name=fold_name,
                config=config,
                unit=unit,
                arm_names=list(arm_names),
                seed=seed,
                nthread=nthread,
                quantiles=quantiles,
                importance=importance,
            )
            for seed in seeds
            for unit in units
        ]
        for future in futures:
            print(future.result(), flush=True)


def main() -> None:
    """Run a stage described by a JSON file of arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON: stage, fold, config, arms, seeds, quantiles")
    spec = json.loads(parser.parse_args().spec.read_text())
    run_stage(
        stage=spec["stage"],
        fold_name=spec["fold"],
        config=spec["config"],
        arm_names=spec["arms"],
        seeds=spec.get("seeds", [0]),
        quantiles=spec.get("quantiles", False),
        importance=spec.get("importance", False),
        workers=spec.get("workers", 12),
    )


if __name__ == "__main__":
    main()
