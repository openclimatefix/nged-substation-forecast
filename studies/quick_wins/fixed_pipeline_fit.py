"""Fit explicit fixed-pipeline recipes against immutable, bounded feature exports.

This runner reads only its --frames directory. Forecasts, models and row-level diagnostics are
private artifacts; a separate reporting step anonymises them before publication. A completed
output is reused only when both its input files and its full execution specification match.
"""

import argparse
import hashlib
import inspect
import json
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

import arms
import cleaning_rules
import fit_arms
import ml_core.metrics as metrics_module
import numpy as np
import patito as pt
import per_run_scores as score_module
import polars as pl
import quick_wins_inputs as inputs
import studies.cross_validation as validation_module
import xgboost as xgb
from contracts.ml_schemas import AllFeatures
from per_run_scores import normalised_by_capacity, per_run_scores
from studies.cross_validation import SENSITIVITY_HYPER_PARAMETERS
from xgboost_forecaster.forecaster import XGBoostConfig, XGBoostForecaster

SettingType = Literal["main", "sensitivity"]
PaddingType = Literal["equal", "none"]
BoosterMap = dict[int, tuple[xgb.Booster, list[str], float]]

KEYS: Final[tuple[str, ...]] = (
    "time_series_id",
    "power_fcst_init_time",
    "nwp_init_time",
    "valid_time",
)
SAMPLE_KEYS: Final[tuple[str, ...]] = (*KEYS, "ensemble_member")
SCORE_KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
STUDY_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
PINBALL_LEVELS: Final[np.ndarray] = (np.arange(19, dtype=float) + 0.5) / 19
SOLAR_GEOMETRY: Final[tuple[str, str]] = ("solar_zenith", "solar_azimuth")
LONGWAVE: Final[str] = "downward_long_wave_radiation_flux_surface"
LONGWAVE_RESIDUAL: Final[str] = "longwave_minus_sigma_t4"
EXACT_RECIPES: Final[frozenset[str]] = frozenset(
    {"production", "production_cleaned", "r2_e1_c1_a3w_d3", "r2_c4_c2_c1_a3w_d3"}
)


@dataclass(frozen=True)
class Recipe:
    """A training arm and the named study-only windchill transformation."""

    arm: arms.Arm
    guarded_windchill: bool = False


def _changed_columns(*, series_type: str, columns: tuple[str, ...]) -> arms.ColumnPolicy:
    def policy(kind: str) -> tuple[str, ...]:
        return columns if kind == series_type else arms.PRODUCTION_COLUMNS

    return policy


def recipe_registry() -> dict[str, Recipe]:
    """Return the exact references, lean contrasts, cumulative solar ladder and controls."""
    registry = {
        "production": Recipe(arms.ALL_ARMS["production_colsample0.8"]),
        **{name: Recipe(arms.ALL_ARMS[name]) for name in EXACT_RECIPES - {"production"}},
    }
    base = arms.ALL_ARMS["production_cleaned"]
    for kind, name in (("PV", "lean_pv"), ("Wind", "lean_wind")):
        registry[name] = Recipe(
            replace(
                base,
                name=name,
                columns=_changed_columns(
                    series_type=kind, columns=tuple(sorted(arms.lean_generators(kind)))
                ),
            )
        )
    solar = set(arms.lean_generators("PV"))
    additions = (
        ("solar_add_wind10", ("wind_speed_10m",)),
        ("solar_add_pressure", ("pressure_surface", "pressure_reduced_to_mean_sea_level")),
        ("solar_add_height", ("geopotential_height_500hpa",)),
        ("solar_add_longwave", (LONGWAVE, LONGWAVE_RESIDUAL)),
        ("solar_replace_clock", SOLAR_GEOMETRY),
    )
    for name, added in additions:
        solar.update(added)
        if name == "solar_replace_clock":
            solar -= {"local_time_of_day_sin", "local_time_of_day_cos", "local_utc_offset_minutes"}
        registry[name] = Recipe(
            replace(
                base,
                name=name,
                columns=_changed_columns(series_type="PV", columns=tuple(sorted(solar))),
            )
        )
    for column in sorted(arms.WIND_IRRELEVANT_COLUMNS | arms.WEEKDAY_COLUMNS):
        name = f"wind_add_{column}"
        registry[name] = Recipe(
            replace(
                base,
                name=name,
                columns=_changed_columns(
                    series_type="Wind",
                    columns=tuple(sorted({*arms.lean_generators("Wind"), column})),
                ),
            )
        )
    for column in ("local_utc_offset_minutes", "local_day_of_week"):
        name = f"without_{column}"
        registry[name] = Recipe(
            replace(
                base,
                name=name,
                columns=lambda _kind, omitted=column: tuple(
                    c for c in arms.PRODUCTION_COLUMNS if c != omitted
                ),
            )
        )
    for source in ("production_cleaned", "lean_pv", "lean_wind"):
        name = f"{source}_guarded_windchill"
        registry[name] = Recipe(replace(registry[source].arm, name=name), guarded_windchill=True)
    return registry


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_context(*, root: Path) -> tuple[dict[str, Any], inputs.FoldWindow]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError(
            "fitting requires complete feature exports; pilot frames are diagnostic only"
        )
    required = {"train.parquet", "metadata.parquet", "capacity.parquet", "bounded_power.parquet"}
    required.update(str(path.relative_to(root)) for path in (root / "val").glob("init=*.parquet"))
    if not any(name.startswith("val/") for name in required):
        raise ValueError("no validation exports")
    files = manifest["files"]
    listed_validation = {name for name in files if name.startswith("val/")}
    actual_validation = {str(path.relative_to(root)) for path in (root / "val").glob("*.parquet")}
    if listed_validation != actual_validation:
        raise ValueError("validation exports differ from the manifest inventory")
    if not required <= files.keys():
        raise ValueError("manifest does not fingerprint every required export")
    for relative in files:
        if _sha256(path=root / relative) != files[relative]:
            raise ValueError("feature export fingerprint mismatch")
    dates = manifest["window"]
    if any(
        datetime.fromisoformat(dates[name]).tzinfo is None
        for name in ("train_start", "train_end", "val_start", "val_end")
    ):
        raise ValueError("fold windows must have explicit UTC offsets")
    window = inputs.FoldWindow(
        fold_id=dates["fold_id"],
        **{
            name: datetime.fromisoformat(dates[name]).astimezone(UTC)
            for name in ("train_start", "train_end", "val_start", "val_end")
        },
    )
    if max(window.train_end, window.val_end) >= STUDY_END:
        raise ValueError("fold reaches the held-out study boundary")
    if window.train_start > window.train_end or window.val_start > window.val_end:
        raise ValueError("invalid fold window")
    if not (window.train_end < window.val_start or window.val_end < window.train_start):
        raise ValueError("training and validation windows overlap")
    expected_validation = {
        f"val/init={stamp:%Y-%m-%d}.parquet" for stamp in inputs.sample_init_times(window=window)
    }
    if listed_validation != expected_validation:
        raise ValueError("validation exports differ from the complete scheduled fold")
    return manifest, window


def _check_frame(*, frame: pl.DataFrame, start: datetime, end: datetime, training: bool) -> None:
    if frame.is_empty() or frame.select(*SAMPLE_KEYS).is_duplicated().any():
        raise ValueError("empty frame or duplicate forecast input keys")
    if frame.filter(~pl.col("valid_time").is_between(start, end)).height:
        raise ValueError("feature frame exceeds its declared fold window")
    members = set(frame["ensemble_member"].unique().to_list())
    if members != ({0} if training else set(range(51))):
        raise ValueError("incorrect source ensemble labels")
    if not training:
        counts = frame.group_by(*KEYS).agg(n=pl.col("ensemble_member").n_unique())
        if counts.filter(pl.col("n") != 51).height:
            raise ValueError("validation keys do not each hold all 51 source members")
    for column in arms.LAG_COLUMNS:
        hours = int(column.removeprefix("power_lag_").removesuffix("h"))
        unavailable = pl.col("valid_time") - pl.duration(hours=hours) >= pl.col(
            "power_fcst_init_time"
        )
        if frame.filter(unavailable & pl.col(column).is_not_null()).height:
            raise ValueError("power lag includes an observation unavailable at forecast issuance")
    if frame.filter(pl.col("power_fcst_init_time") > pl.col("valid_time")).height:
        raise ValueError("forecast issued after its valid time")


def _prepare(
    *, frame: pl.DataFrame, recipe: Recipe, padding: PaddingType, registry: dict[str, Recipe]
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    if recipe.guarded_windchill:
        frame = frame.with_columns(
            windchill=pl.when(
                (pl.col("temperature_2m") <= 10) & (pl.col("wind_speed_10m") * 3.6 >= 4.8)
            )
            .then(pl.col("windchill"))
            .otherwise(pl.col("temperature_2m"))
        )
    columns = {}
    for kind in frame["time_series_type"].unique().to_list():
        real = list(recipe.arm.columns(kind))
        # Exact reference feature order is the historical arm's order. Ablations are sorted.
        if recipe.arm.name not in {arms.ALL_ARMS[n].name for n in EXACT_RECIPES - {"production"}}:
            real = sorted(real)
        shown = list(real)
        if (
            padding == "equal"
            and recipe.arm.name != "production_colsample0.8"
            and (recipe.arm.name not in EXACT_RECIPES)
        ):
            width = max(
                len(r.arm.columns(kind))
                for name, r in registry.items()
                if name not in EXACT_RECIPES
            )
            for index in range(width - len(real)):
                name = f"padding_{kind.replace(' ', '_')}_{index:02d}"
                frame = frame.with_columns(pl.col(sorted(real)[index % len(real)]).alias(name))
                shown.append(name)
        columns[kind] = shown
    return frame, columns


def _fit_one(
    *,
    group: pl.DataFrame,
    columns: list[str],
    arm: arms.Arm,
    window: inputs.FoldWindow,
    device: str,
    seed: int,
    setting: SettingType,
) -> xgb.Booster:
    params = fit_arms._parameters(arm=arm, device=device, seed=seed)
    rounds = fit_arms.N_TREES
    if setting == "sensitivity":
        sensitivity = dict(SENSITIVITY_HYPER_PARAMETERS)
        rounds = int(cast(int, sensitivity.pop("num_boost_round")))
        params.update(sensitivity)
        params["eta"] = params.pop("learning_rate")
    params["nthread"] = 8
    features = fit_arms.feature_matrix(frame=group, columns=columns)
    labels = group["power"].cast(pl.Float32)
    if arm.early_stopping:
        fit_mask = group["valid_time"] <= window.train_end - fit_arms.EARLY_STOPPING_HOLDOUT
        if fit_mask.all() or not fit_mask.any():
            raise ValueError("early-stopping probe needs both training and holdout rows")
        probe = xgb.train(
            params=params,
            dtrain=xgb.DMatrix(features.filter(fit_mask), label=labels.filter(fit_mask)),
            num_boost_round=rounds,
            evals=[
                (xgb.DMatrix(features.filter(~fit_mask), label=labels.filter(~fit_mask)), "hold")
            ],
            early_stopping_rounds=fit_arms.EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        rounds = max(fit_arms.MIN_TREES, probe.best_iteration + 1)
    return xgb.train(
        params=params, dtrain=xgb.DMatrix(features, label=labels), num_boost_round=rounds
    )


def _fit_historical(
    *,
    training: pl.DataFrame,
    recipe: Recipe,
    columns: dict[str, list[str]],
    window: inputs.FoldWindow,
    device: str,
    seed: int,
    setting: SettingType,
) -> BoosterMap:
    arm = recipe.arm
    models: BoosterMap = {}
    if not arm.pooled_by_type:
        for (series_id,), group in training.group_by("time_series_id"):
            shown = columns[group["time_series_type"][0]]
            models[int(series_id)] = (
                _fit_one(
                    group=group,
                    columns=shown,
                    arm=arm,
                    window=window,
                    device=device,
                    seed=seed,
                    setting=setting,
                ),
                shown,
                1.0,
            )
        return models
    capacities = cleaning_rules.training_capacity(power=training.select("time_series_id", "power"))
    if capacities.filter(~pl.col("capacity").is_finite() | (pl.col("capacity") <= 0)).height:
        raise ValueError("pooled training capacity must be finite and positive")
    scaled = training.join(capacities, on="time_series_id", maintain_order="left")
    scaled = pl.concat(
        fit_arms.pooled_inputs(frame=part, scale=float(part["capacity"][0]), arm=arm).with_columns(
            power=pl.col("power") / pl.col("capacity")
        )
        for part in scaled.partition_by("time_series_id")
    )
    for (kind,), group in scaled.group_by("time_series_type"):
        shown = [*columns[kind], *(["site_id"] if arm.pooled_site_id else [])]
        model = _fit_one(
            group=group,
            columns=shown,
            arm=arm,
            window=window,
            device=device,
            seed=seed,
            setting=setting,
        )
        for part in group.partition_by("time_series_id"):
            models[int(part["time_series_id"][0])] = (model, shown, float(part["capacity"][0]))
    return models


class _SensitivityConfig(XGBoostConfig):
    """Add the study's regularisation setting while keeping public production train/predict."""

    reg_lambda: float = 5.0

    def to_xgb_params(self) -> dict[str, Any]:
        return {**super().to_xgb_params(), "reg_lambda": self.reg_lambda, "nthread": 8}


def _production(
    *, training: pl.DataFrame, series_ids: list[int], seed: int, device: str, setting: SettingType
) -> XGBoostForecaster:
    raw: dict[str, Any] = {
        **inputs.production_config().model_dump(),
        "random_seed": seed,
        "device": device,
    }
    config_class = XGBoostConfig
    if setting == "sensitivity":
        sensitivity = dict(SENSITIVITY_HYPER_PARAMETERS)
        raw.update(sensitivity)
        raw["n_estimators"] = raw.pop("num_boost_round")
        raw["min_child_weight"] = int(raw["min_child_weight"])
        config_class = _SensitivityConfig
    model = XGBoostForecaster(model_params=config_class(**raw))
    model.train(
        data=pt.LazyFrame.from_existing(training.lazy()).set_model(AllFeatures),
        time_series_ids=series_ids,
    )
    if set(model.trained_time_series_ids) != set(series_ids):
        raise ValueError("production training population differs from frame population")
    return model


def _predict(*, frame: pl.DataFrame, models: BoosterMap, arm: arms.Arm) -> pl.DataFrame:
    pieces = []
    for (series_id,), rows in frame.group_by("time_series_id"):
        booster, columns, scale = models[int(series_id)]
        shown = (
            fit_arms.pooled_inputs(frame=rows, scale=scale, arm=arm) if arm.pooled_by_type else rows
        )
        predicted = booster.predict(
            xgb.DMatrix(fit_arms.feature_matrix(frame=shown, columns=columns))
        )
        predicted = predicted * scale
        if not np.isfinite(predicted).all():
            raise ValueError("non-finite forecast output")
        keys = rows.select(*KEYS)
        pieces.append(
            fit_arms._quantile_sample(keys=keys, rows=rows, predicted=predicted)
            if arm.quantile_levels
            else keys.with_columns(
                ensemble_member=rows["ensemble_member"],
                power_fcst=pl.Series(predicted, dtype=pl.Float32),
            )
        )
    return pl.concat(pieces)


def _score(
    *, forecasts: pl.DataFrame, actuals: pl.DataFrame, capacity: pl.DataFrame
) -> pl.DataFrame:
    if forecasts.select(*SAMPLE_KEYS).is_duplicated().any():
        raise ValueError("duplicate predicted sample keys")
    if forecasts.filter(~pl.col("power_fcst").is_finite()).height:
        raise ValueError("non-finite forecast output")
    scores = normalised_by_capacity(
        scores=per_run_scores(
            forecasts=forecasts, actuals=actuals.filter(pl.col("power").is_not_null())
        ),
        capacity=capacity,
    )
    samples = forecasts.group_by(*SCORE_KEYS).agg(values=pl.col("power_fcst").sort())
    joined = scores.join(samples, on=list(SCORE_KEYS), validate="1:1", maintain_order="left")
    values = np.array(joined["values"].to_list(), dtype=float)
    observed = joined["actual"].to_numpy()
    ranks = np.ceil(values.shape[1] * PINBALL_LEVELS).astype(int) - 1
    residual = observed[:, None] - values[:, ranks]
    losses = np.maximum(PINBALL_LEVELS * residual, (PINBALL_LEVELS - 1) * residual)
    scales = joined["effective_capacity_mw"].to_numpy()
    extra = [
        pl.Series(f"pinball_{index:02d}_norm", losses[:, index] / scales) for index in range(19)
    ]
    for mass in (0.8, 0.9):
        lower, upper = (1 - mass) / 2, (1 + mass) / 2
        low = values[:, int(np.ceil(values.shape[1] * lower)) - 1]
        high = values[:, int(np.ceil(values.shape[1] * upper)) - 1]
        extra.append(
            pl.Series(f"coverage{int(mass * 100)}", (observed >= low) & (observed <= high))
        )
    return joined.drop("values").with_columns(
        *extra, pl.Series("pinball19_norm", 2 * losses.mean(axis=1) / scales)
    )


def _save_models(*, models: BoosterMap, output: Path) -> None:
    rows = []
    for index, (series_id, (booster, columns, scale)) in enumerate(sorted(models.items())):
        booster.save_model(output / "models" / f"model_{index:03d}.ubj")
        gains = booster.get_score(importance_type="total_gain")
        rows.append(
            {
                "time_series_id": series_id,
                "model_file": f"model_{index:03d}.ubj",
                "features": columns,
                "scale": scale,
                "total_gain": gains,
                "rounds": booster.num_boosted_rounds(),
                "effective_xgboost_config": json.loads(booster.save_config()),
            }
        )
    (output / "models.json").write_text(json.dumps(rows, indent=2))


def _load_training(
    *,
    args: argparse.Namespace,
    window: inputs.FoldWindow,
    recipe: Recipe,
    registry: dict[str, Recipe],
    output: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[int], dict[str, list[str]]]:
    training = pl.read_parquet(args.frames / "train.parquet")
    _check_frame(frame=training, start=window.train_start, end=window.train_end, training=True)
    metadata = pl.read_parquet(args.frames / "metadata.parquet").select(
        "time_series_id", pl.col("time_series_type").cast(pl.String)
    )
    training = training.join(metadata, on="time_series_id", validate="m:1", maintain_order="left")
    series_ids = sorted(metadata["time_series_id"].to_list())
    power = pl.read_parquet(args.frames / "bounded_power.parquet")
    if power.filter(pl.col("time") >= STUDY_END).height:
        raise ValueError("bounded power reaches held-out study boundary")
    if power.filter(pl.col("power").is_not_null() & ~pl.col("power").is_finite()).height:
        raise ValueError("bounded observations contain non-finite values")
    if power.select("time_series_id", "time").is_duplicated().any():
        raise ValueError("bounded power has duplicate observation keys")
    guard = ~((pl.col("temperature_2m") <= 10) & (pl.col("wind_speed_10m") * 3.6 >= 4.8))
    training.group_by("time_series_type").agg(
        windchill_outside_domain=guard.fill_null(True).sum(),
        rows=pl.len(),
        **{f"null_{name}": pl.col(name).null_count() for name in arms.PRODUCTION_COLUMNS},
    ).write_parquet(output / "training_input_diagnostics.parquet")
    before = training.group_by("time_series_id").agg(rows_before=pl.len())
    training = training.filter(pl.col("power").is_not_null())
    if training.filter(~pl.col("power").is_finite()).height:
        raise ValueError("non-finite training targets")
    targets = power.rename({"time": "valid_time", "power": "bounded_target"}).select(
        "time_series_id", "valid_time", "bounded_target"
    )
    checked = training.join(
        targets,
        on=["time_series_id", "valid_time"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if checked.filter(
        pl.col("bounded_target").is_null() | (pl.col("power") != pl.col("bounded_target"))
    ).height:
        raise ValueError("training targets differ from frozen bounded observations")
    for rule in (recipe.arm.cleaning, *recipe.arm.extra_cleaning):
        if rule != "none":
            drops = cleaning_rules.rows_to_drop(
                rule=rule,
                power=power.filter(pl.col("time") <= window.train_end),
                types=metadata,
                train_start=window.train_start,
            ).rename({"time": "valid_time"})
            training = training.join(
                drops, on=["time_series_id", "valid_time"], how="anti", maintain_order="left"
            )
    before.join(
        training.group_by("time_series_id").agg(rows_kept=pl.len()), on="time_series_id", how="left"
    ).write_parquet(output / "training_counts.parquet")
    training, columns = _prepare(
        frame=training, recipe=recipe, padding=args.padding, registry=registry
    )
    if set(training["time_series_id"].unique().to_list()) != set(series_ids):
        raise ValueError("cleaning removed a whole series or metadata population differs")
    actual_columns = {
        kind: [*names, *(["site_id"] if recipe.arm.pooled_site_id else [])]
        for kind, names in columns.items()
    }
    (output / "features.json").write_text(json.dumps(actual_columns, indent=2))
    return training, metadata, power, series_ids, columns


def _predict_all(
    *,
    args: argparse.Namespace,
    window: inputs.FoldWindow,
    recipe: Recipe,
    registry: dict[str, Recipe],
    output: Path,
    metadata: pl.DataFrame,
    power: pl.DataFrame,
    series_ids: list[int],
    models: BoosterMap,
    production: XGBoostForecaster | None,
) -> None:
    capacity = pl.read_parquet(args.frames / "capacity.parquet")
    if capacity.filter(
        ~pl.col("effective_capacity_mw").is_finite() | (pl.col("effective_capacity_mw") <= 0)
    ).height:
        raise ValueError("score capacities must be finite and positive")
    score_parts = []
    for path in sorted((args.frames / "val").glob("init=*.parquet")):
        frame = pl.read_parquet(path)
        _check_frame(frame=frame, start=window.val_start, end=window.val_end, training=False)
        if set(frame["time_series_id"].unique().to_list()) != set(series_ids):
            raise ValueError("validation population differs from training population")
        frame = frame.join(metadata, on="time_series_id", validate="m:1", maintain_order="left")
        frame, _ = _prepare(frame=frame, recipe=recipe, padding=args.padding, registry=registry)
        forecasts = (
            pl.DataFrame(
                production.predict(
                    data=pt.LazyFrame.from_existing(frame.lazy()).set_model(AllFeatures),
                    fold_id=window.fold_id,
                )
            ).select(*SAMPLE_KEYS, "power_fcst")
            if production is not None
            else _predict(frame=frame, models=models, arm=recipe.arm)
        )
        width = len(recipe.arm.quantile_levels or (1,))
        if forecasts.height != frame.height * width:
            raise ValueError("forecast sample row count mismatch")
        forecasts.write_parquet(output / "forecasts" / path.name)
        scores = _score(forecasts=forecasts, actuals=power, capacity=capacity)
        expected = (
            frame.select(*SCORE_KEYS)
            .unique()
            .join(
                power.filter(pl.col("power").is_not_null())
                .rename({"time": "valid_time"})
                .select("time_series_id", "valid_time"),
                on=["time_series_id", "valid_time"],
                how="semi",
            )
        )
        if not scores.select(*SCORE_KEYS).sort(*SCORE_KEYS).equals(expected.sort(*SCORE_KEYS)):
            raise ValueError("scored rows differ from common target-supported rows")
        score_parts.append(
            scores.join(metadata, on="time_series_id", validate="m:1", maintain_order="left")
        )
    pl.concat(score_parts).write_parquet(output / "scores.parquet")


def _arguments(*, registry: dict[str, Recipe]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recipe", required=True, choices=sorted(registry))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True, choices=("cuda", "cpu"))
    parser.add_argument("--setting", required=True, choices=("main", "sensitivity"))
    parser.add_argument("--padding", choices=("equal", "none"), default="equal")
    return parser.parse_args()


def main() -> None:
    """Fit one frozen recipe and save private forecasts, normalised diagnostics and provenance."""
    registry = recipe_registry()
    args = _arguments(registry=registry)
    manifest, window = _load_context(root=args.frames)
    recipe = registry[args.recipe]
    specification = {
        "input_fingerprint": manifest["input_fingerprint"],
        "input_manifest_sha256": _sha256(path=args.frames / "manifest.json"),
        "script_sha256": _sha256(path=Path(__file__)),
        "recipe": args.recipe,
        "seed": args.seed,
        "device": args.device,
        "setting": args.setting,
        "padding": args.padding,
        "input_manifest": manifest,
        "xgboost_version": xgb.__version__,
        "polars_version": pl.__version__,
        "sensitivity_hyperparameters": dict(SENSITIVITY_HYPER_PARAMETERS),
        "dependency_sha256": {
            str(path): _sha256(path=path)
            for path in (
                Path(arms.__file__),
                Path(fit_arms.__file__),
                Path(cleaning_rules.__file__),
                Path(inputs.__file__),
                Path(score_module.__file__),
                Path(metrics_module.__file__),
                Path(validation_module.__file__),
                Path(inspect.getfile(XGBoostForecaster)),
                Path(inspect.getfile(AllFeatures)),
                inputs.PRODUCTION_MODEL_CONFIG,
            )
        },
    }
    output = (
        args.output / args.recipe / args.setting / f"{args.device}_seed{args.seed}_{args.padding}"
    )
    checkpoint = output / "manifest.json"
    if checkpoint.exists():
        if json.loads(checkpoint.read_text()) != specification:
            raise ValueError("output specification mismatch; use a new output directory")
        if (output / "complete.json").exists():
            completion = json.loads((output / "complete.json").read_text())
            for relative, digest in completion["files"].items():
                if _sha256(path=output / relative) != digest:
                    raise ValueError("completed output fingerprint mismatch")
            print("Matching completed fit exists; skipping.")
            return
        raise ValueError("incomplete output exists; inspect and preserve it before rerunning")
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "forecasts").mkdir()
    checkpoint.write_text(json.dumps(specification, indent=2))
    training, metadata, power, series_ids, columns = _load_training(
        args=args, window=window, recipe=recipe, registry=registry, output=output
    )
    started = time.monotonic()
    production = None
    models: BoosterMap = {}
    if args.recipe == "production":
        production = _production(
            training=training,
            series_ids=series_ids,
            seed=args.seed,
            device=args.device,
            setting=args.setting,
        )
        production.save(output / "models")
        for series_id in series_ids:
            booster = xgb.Booster()
            booster.load_model(output / "models" / f"{series_id}.ubj")
            models[series_id] = (booster, list(arms.PRODUCTION_COLUMNS), 1.0)
    else:
        models = _fit_historical(
            training=training,
            recipe=recipe,
            columns=columns,
            window=window,
            device=args.device,
            seed=args.seed,
            setting=args.setting,
        )
    _save_models(models=models, output=output)
    fitted = time.monotonic()
    _predict_all(
        args=args,
        window=window,
        recipe=recipe,
        registry=registry,
        output=output,
        metadata=metadata,
        power=power,
        series_ids=series_ids,
        models=models,
        production=production,
    )
    timing: dict[str, Any] = {
        "fit_seconds": fitted - started,
        "predict_score_seconds": time.monotonic() - fitted,
        "source_member_count": 51,
        "sample_count": 51 * len(recipe.arm.quantile_levels or (1,)),
    }
    final_manifest, _ = _load_context(root=args.frames)
    if final_manifest != manifest:
        raise ValueError("input manifest changed during fitting")
    timing["files"] = {
        str(path.relative_to(output)): _sha256(path=path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != checkpoint
    }
    (output / "complete.json").write_text(json.dumps(timing, indent=2))
    print(
        f"Completed {args.recipe}: fit {timing['fit_seconds']:.0f} s; "
        f"prediction/scoring {timing['predict_score_seconds']:.0f} s."
    )


if __name__ == "__main__":
    main()
