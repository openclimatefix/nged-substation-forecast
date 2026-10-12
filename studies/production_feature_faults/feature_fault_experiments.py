"""Study-only feature corrections and paired GPU refits; production files are read only.

Prepared correction files contain row positions and weather values, never source identifiers or
power. Saved losses are capacity-normalised and use a separate anonymous 31-series permutation.
"""

import argparse
import hashlib
import inspect
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
import xgboost as xgb
import yaml
from contracts.settings import PROJECT_ROOT, Settings
from deltalake import DeltaTable
from ml_core.metrics import _fair_crps_expr
from studies.cross_validation import SENSITIVITY_HYPER_PARAMETERS
from studies.power import CLEANED_POWER_DELTA_URI, scan_power
from studies.resample import clear_sky_index_resample
from studies.solar import zenith

RAD: Final[str] = "downward_short_wave_radiation_flux_surface"
DIRECTIONS: Final[tuple[str, ...]] = ("wind_direction_10m", "wind_direction_100m")
CORRECTIONS: Final[tuple[str, ...]] = (RAD, *DIRECTIONS, "windchill")
GROUP: Final[list[str]] = ["time_series_id", "nwp_init_time", "ensemble_member"]
KEY: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
ARMS: Final[tuple[str, ...]] = (
    "baseline",
    "all_fixed",
    "without_radiation",
    "without_wind",
    "without_windchill",
    "mask_precipitation",
    "mask_weekday",
)


def digest(path: Path) -> str:
    """Hash an input without loading the whole file."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def correction_controls() -> None:
    """Require the radiation helper to recover a constant clear-sky fraction and preserve nulls."""
    values = np.array([[100.0, 200.0, 100.0]])
    sky = values * 2
    result = clear_sky_index_resample(
        values=values,
        step_clear_sky=sky,
        step_midpoints=np.array([1.0, 4.0, 7.0]),
        morning=np.array([True, True, False]),
        target_clear_sky=np.array([[20.0, 80.0, 0.0]]),
        target_midpoints=np.array([2.0, 3.0, 5.0]),
        daylight_floor_w_m2=5,
    )
    np.testing.assert_allclose(result, [[10.0, 40.0, 0.0]])
    hours = np.arange(0, 6.5, 0.5)
    knots = np.array([0.0, 3.0, 6.0])
    frame = pl.DataFrame(
        {
            "row": np.arange(len(hours), dtype=np.uint32),
            "valid_time": [
                datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=float(h)) for h in hours
            ],
            "nwp_init_time": [datetime(2025, 1, 1, tzinfo=UTC)] * len(hours),
            RAD: np.interp(hours, knots, [50.0, 100.0, 50.0]),
            "wind_direction_10m": np.interp(hours, knots, [350.0, 10.0, 30.0]),
            "wind_direction_100m": np.interp(hours, knots, [350.0, 10.0, 30.0]),
            "cs_half": np.full(len(hours), 100.0),
            "cs_3": np.interp(hours, knots, [100.0, 200.0, 100.0]),
            "cs_6": np.interp(hours, knots, [100.0, 200.0, 100.0]),
            "temperature_2m": np.full(len(hours), 20.0),
            "wind_speed_10m": np.full(len(hours), 1.0),
            "windchill": np.full(len(hours), 5.0),
        }
    )
    corrected, _ = correct_group(frame)
    np.testing.assert_allclose(corrected[RAD + "__fixed"], 50.0)
    np.testing.assert_allclose(corrected["windchill__fixed"], 20.0)
    np.testing.assert_allclose(corrected["wind_direction_10m__fixed"][3], 0.0)
    missing, _ = correct_group(frame.with_columns(pl.lit(float("nan")).alias(RAD)))
    assert missing[RAD + "__fixed"].is_nan().all()
    trimmed, _ = correct_group(frame.filter(pl.col("row").is_between(1, 11)))
    np.testing.assert_allclose(trimmed[RAD + "__fixed"][0], frame[RAD][1])


def configuration() -> dict[str, Any]:
    """Read the production recipe without changing it."""
    return yaml.safe_load((PROJECT_ROOT / "conf/model/xgboost.yaml").read_text())["model_params"]


def solar_table(frame: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
    """Compute backward clear-sky means on a complete half-hour grid, including window padding."""
    pieces = []
    for (series,), times in (
        frame.select("time_series_id", "valid_time")
        .unique()
        .partition_by("time_series_id", as_dict=True)
        .items()
    ):
        first, last = times["valid_time"].min(), times["valid_time"].max()
        if not isinstance(first, datetime) or not isinstance(last, datetime):
            raise TypeError("solar times must be datetimes")
        stamps = pl.datetime_range(
            first - timedelta(hours=6),
            last,
            interval="30m",
            eager=True,
        ).alias("valid_time")
        lat, lon = (
            metadata.filter(pl.col("time_series_id") == series)
            .select("latitude", "longitude")
            .row(0)
        )
        samples = []
        for seconds in (450, 1350):
            angles = zenith(stamps=stamps.dt.offset_by(f"-{seconds}s"), latitude=lat, longitude=lon)
            cosine = np.maximum(np.cos(np.deg2rad(angles)), 0)
            samples.append(
                np.where(cosine > 0, 1098 * cosine * np.exp(-0.059 / np.maximum(cosine, 1e-8)), 0)
            )
        pieces.append(
            pl.DataFrame({"valid_time": stamps, "cs_half": np.mean(samples, axis=0)})
            .with_columns(
                time_series_id=pl.lit(series).cast(pl.Int32),
            )
            .with_columns(
                cs_3=pl.col("cs_half").rolling_mean(6),
                cs_6=pl.col("cs_half").rolling_mean(12),
            )
        )
    return pl.concat(pieces)


def correct_group(part: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int | float]]:
    """Reconstruct cached native-grid features, checking linear round-trip on supported rows."""
    part = part.sort("valid_time")
    hours = (part["valid_time"] - part["nwp_init_time"]).dt.total_seconds().to_numpy() / 3600
    native = np.where(hours <= 144, hours % 3 == 0, hours % 6 == 0)
    counts = {
        "rows": part.height,
        "radiation_no_daylight_rows": 0,
        "wind_fallback_rows": 0,
        "radiation_fallback_rows": 0,
    }
    corrected = {}
    for column in (RAD, *DIRECTIONS):
        original = part[column].cast(pl.Float64).to_numpy()
        anchors = native & np.isfinite(original)
        x, values = hours[anchors], original[anchors]
        result = original.copy()
        if len(x) < 2:
            counts["radiation_no_daylight_rows" if column == RAD else "wind_fallback_rows"] += (
                part.height
            )
            corrected[column + "__fixed"] = result
            continue
        supported = (hours >= x[0]) & (hours <= x[-1]) & np.isfinite(original)
        reconstruction = np.interp(hours, x, values)
        # Float32 rounding accumulates slightly when recovering a historical interpolated cache.
        if np.max(np.abs(reconstruction[supported] - original[supported]), initial=0) > 0.003:
            raise ValueError(
                "native-grid linear reconstruction differs from cached production values"
            )
        if column == RAD:
            widths = np.where(x <= 144, 3.0, 6.0)
            sky = np.where(
                x <= 144, part["cs_3"].to_numpy()[anchors], part["cs_6"].to_numpy()[anchors]
            )
            day = np.isfinite(sky) & (sky >= 5)
            if day.sum() < 2:
                counts["radiation_no_daylight_rows"] += part.height
            else:
                seconds = (
                    part["valid_time"].dt.hour().to_numpy()[anchors]
                    + part["valid_time"].dt.minute().to_numpy()[anchors] / 60
                )
                result = clear_sky_index_resample(
                    values=values[None, :],
                    step_clear_sky=sky[None, :],
                    step_midpoints=x - widths / 2,
                    morning=(seconds - widths / 2) % 24 < 12,
                    target_clear_sky=part["cs_half"].to_numpy()[None, :],
                    target_midpoints=hours - 0.25,
                    daylight_floor_w_m2=5,
                )[0]
                # No formerly missing production weather is silently filled by the candidate.
                result[~supported] = original[~supported]
                counts["radiation_fallback_rows"] += int((~supported).sum())
                starts = np.searchsorted(hours, x - widths, side="right")
                ends = np.searchsorted(hours, x, side="right")
                sums = np.r_[0, np.cumsum(np.where(np.isfinite(result), result, 0))]
                valid = np.r_[0, np.cumsum(np.isfinite(result) & supported)]
                complete = (ends - starts == widths * 2) & (
                    valid[ends] - valid[starts] == widths * 2
                )
                energy_delta = (sums[ends] - sums[starts]) / (widths * 2) - values
                counts["radiation_energy_intervals"] = int(complete.sum())
                counts["radiation_sum_abs_mean_delta_wm2"] = float(
                    np.abs(energy_delta[complete]).sum()
                )
        else:
            left = np.clip(np.searchsorted(x, hours, side="right") - 1, 0, len(x) - 2)
            delta = (values[left + 1] - values[left] + 180) % 360 - 180
            eligible = supported & (np.abs(delta) < 179.999)
            fraction = (hours - x[left]) / (x[left + 1] - x[left])
            result[eligible] = (values[left] + fraction * delta)[eligible] % 360
            counts["wind_fallback_rows"] += int((~eligible & ~native).sum())
        corrected[column + "__fixed"] = result
    temp, speed = part["temperature_2m"].to_numpy(), part["wind_speed_10m"].to_numpy()
    chill = part["windchill"].to_numpy().copy()
    outside = (
        np.isfinite(temp)
        & np.isfinite(speed)
        & ((temp > 10) | (speed * 3.6 < 4.8))
        & np.isfinite(chill)
    )
    chill[outside] = temp[outside]
    corrected["windchill__fixed"] = chill
    return part.select("row").with_columns(
        pl.Series(k, v, dtype=pl.Float32) for k, v in corrected.items()
    ), counts


def prepare(path: Path, destination: Path, metadata: pl.DataFrame, columns: list[str]) -> None:
    """Build a correction cache whose provenance binds it to its exact source and script."""
    stamp = {
        "source_sha256": digest(path),
        "script_sha256": digest(Path(__file__)),
        "metadata_sha256": digest(Path(Settings().metadata_path)),
        "resample_sha256": digest(Path(inspect.getfile(clear_sky_index_resample))),
        "solar_sha256": digest(Path(inspect.getfile(zenith))),
    }
    stamp_path = destination.with_suffix(".json")
    if destination.exists():
        if json.loads(stamp_path.read_text()) != stamp:
            raise ValueError("correction cache provenance differs; choose a new output directory")
        return
    frame = pl.read_parquet(
        path, columns=[*GROUP, "valid_time", *CORRECTIONS, "temperature_2m", "wind_speed_10m"]
    ).with_row_index("row")
    geometry = solar_table(frame, metadata)
    frame = frame.join(geometry, on=["time_series_id", "valid_time"], how="left")
    pieces, totals = [], {}
    for part in frame.partition_by(GROUP, maintain_order=True):
        values, counts = correct_group(part)
        pieces.append(values)
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
    result = pl.concat(pieces).sort("row")
    if result.height != frame.height or result["row"].n_unique() != frame.height:
        raise ValueError("correction cache changed row membership")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(destination)
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
    destination.with_suffix(".counts.json").write_text(json.dumps(totals, indent=2) + "\n")
    print(f"prepared {path.name}: {result.height:,} rows, {totals}", flush=True)


def matrix(frame: pl.DataFrame, columns: list[str], arm: str) -> pl.DataFrame:
    """Return exactly the production column count, substituting only an arm's declared values."""
    changed = set(CORRECTIONS) if arm != "baseline" else set()
    if arm == "without_radiation":
        changed.remove(RAD)
    if arm == "without_wind":
        changed.difference_update(DIRECTIONS)
    if arm == "without_windchill":
        changed.remove("windchill")
    expressions = []
    for column in columns:
        source = column + "__fixed" if column in changed else column
        value = pl.col(source)
        if isinstance(frame.schema[source], pl.Enum):
            value = value.to_physical()
        if (arm == "mask_precipitation" and column == "categorical_precipitation_type_surface") or (
            arm == "mask_weekday" and column == "local_day_of_week"
        ):
            value = pl.lit(None)
        expressions.append(value.cast(pl.Float32).alias(column))
    return frame.select(expressions)


def parameters(setting: str, seed: int) -> tuple[dict[str, Any], int]:
    """Use production parameters, with explicit controlled and exact-production settings."""
    config = configuration()
    params = {
        key: config[key]
        for key in ("learning_rate", "max_depth", "min_child_weight", "subsample", "objective")
    }
    params.update(
        device="cuda",
        tree_method="hist",
        nthread=8,
        seed=seed,
        colsample_bytree=0.8 if setting == "exact_production" else 1.0,
    )
    rounds = config["n_estimators"]
    if setting == "sensitivity":
        params.update(SENSITIVITY_HYPER_PARAMETERS)
        rounds = params.pop("num_boost_round")
    return params, int(rounds)


def load_frame(path: Path, prepared: Path, columns: list[str], *, training: bool) -> pl.DataFrame:
    """Read the exact input order and append its position-matched correction values."""
    fields = [*KEY, "nwp_init_time", "ensemble_member", *columns, *(["power"] if training else [])]
    frame = pl.read_parquet(path, columns=fields).with_row_index("row")
    fixes = pl.read_parquet(prepared)
    if not frame["row"].equals(fixes["row"]):
        raise ValueError("correction rows differ from the input")
    return frame.hstack(fixes.drop("row"))


def fit_models(
    training: pl.DataFrame, columns: list[str], arm: str, setting: str, seed: int
) -> tuple[dict[int, xgb.Booster], list[dict[str, Any]]]:
    """Fit one CUDA booster per series and retain every column's normalised total gain."""
    params, rounds = parameters(setting, seed)
    models, importance = {}, []
    for (series,), part in training.partition_by("time_series_id", as_dict=True).items():
        usable = part.filter(pl.col("power").is_finite()).sort(*KEY, "ensemble_member")
        if not usable.height:
            raise ValueError("a series has no finite training target")
        values = matrix(usable, columns, arm)
        booster = xgb.train(
            params,
            xgb.QuantileDMatrix(values, label=usable["power"], nthread=8),
            num_boost_round=rounds,
        )
        if json.loads(booster.save_config())["learner"]["generic_param"]["device"] != "cuda:0":
            raise ValueError("XGBoost did not train on the requested GPU")
        models[int(series)] = booster
        gains = booster.get_score(importance_type="total_gain")
        if any(not isinstance(value, float) for value in gains.values()):
            raise TypeError("point-model gains must be scalar")
        gains = {key: float(value) for key, value in gains.items() if isinstance(value, float)}
        total = sum(gains.values()) or 1
        importance.extend(
            {
                "feature": c,
                "share": gains.get(c, 0) / total,
                "arm": arm,
                "setting": setting,
                "seed": seed,
            }
            for c in columns
        )
    return models, importance


def score_run(
    frame: pl.DataFrame,
    models: dict[int, xgb.Booster],
    columns: list[str],
    arm: str,
    observations: pl.DataFrame,
    capacities: pl.DataFrame,
    labels: dict[int, str],
    metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Collapse all members using the production fair-CRPS expression, then normalise per series."""
    pieces = []
    for (series,), part in frame.partition_by("time_series_id", as_dict=True).items():
        forecast = models[int(series)].predict(xgb.DMatrix(matrix(part, columns, arm), nthread=8))
        if not np.isfinite(forecast).all():
            raise ValueError("non-finite predictions")
        pieces.append(
            part.select(*KEY, "ensemble_member").with_columns(pl.Series("power_fcst", forecast))
        )
    forecasts = pl.concat(pieces)
    forecast_keys = forecasts.select(*KEY, "ensemble_member").sort(*KEY, "ensemble_member")
    if forecast_keys.is_duplicated().any() or not forecast_keys.equals(
        frame.select(*KEY, "ensemble_member").sort(*KEY, "ensemble_member")
    ):
        raise ValueError("prediction keys differ from the common input cases")
    member_sets = forecasts.group_by(KEY).agg(pl.col("ensemble_member").sort())
    if any(members != list(range(51)) for members in member_sets["ensemble_member"].to_list()):
        raise ValueError("member labels differ from 0 through 50")
    joined = forecasts.join(observations, on=["time_series_id", "valid_time"], how="inner").filter(
        pl.col("power_actual").is_finite()
    )
    scores = joined.group_by(KEY).agg(
        crps=_fair_crps_expr(),
        mae=(pl.col("power_fcst").mean() - pl.col("power_actual").first()).abs(),
        members=pl.len(),
        actual=pl.col("power_actual").first(),
    )
    if scores["members"].min() != 51 or scores["members"].max() != 51:
        raise ValueError("scoring requires exactly 51 members per forecast case")
    scores = scores.join(capacities, on="time_series_id", how="left").join(
        metadata.select("time_series_id", "time_series_type"), on="time_series_id"
    )
    if (
        scores["effective_capacity_mw"].null_count()
        or scores.filter(
            ~pl.col("effective_capacity_mw").is_finite() | (pl.col("effective_capacity_mw") <= 0)
        ).height
    ):
        raise ValueError("every scored series needs a positive capacity")
    return scores.select(
        pl.col("time_series_id").replace_strict(labels).alias("site"),
        "power_fcst_init_time",
        "valid_time",
        "time_series_type",
        "members",
        actual=pl.col("actual") / pl.col("effective_capacity_mw"),
        crps=100 * pl.col("crps") / pl.col("effective_capacity_mw"),
        mae=100 * pl.col("mae") / pl.col("effective_capacity_mw"),
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600,
    ).sort("site", "power_fcst_init_time", "valid_time")


def run_job(
    args: argparse.Namespace,
    metadata: pl.DataFrame,
    columns: list[str],
    observations: pl.DataFrame,
    capacities: pl.DataFrame,
    labels: dict[int, str],
    arm: str,
    setting: str,
    seed: int,
) -> None:
    """Checkpoint a complete job's anonymous losses; all model fitting runs serially on CUDA."""
    destination = args.output / "scores" / f"{setting}_{arm}_seed{seed}.parquet"
    importance_path = destination.with_name(destination.stem + "_importance.parquet")
    if destination.exists() and importance_path.exists():
        return
    start = time.monotonic()
    training = load_frame(
        args.frames / "train.parquet",
        args.output / "prepared/train.parquet",
        columns,
        training=True,
    )
    models, importance = fit_models(training, columns, arm, setting, seed)
    del training
    print(
        f"fitted {setting}/{arm}/seed{seed}: {len(models)} models "
        f"in {time.monotonic() - start:.0f}s",
        flush=True,
    )
    scores = []
    for path in sorted((args.frames / "val").glob("*.parquet")):
        frame = load_frame(path, args.output / "prepared" / path.name, columns, training=False)
        scores.append(
            score_run(frame, models, columns, arm, observations, capacities, labels, metadata)
        )
    result = pl.concat(scores).with_columns(
        arm=pl.lit(arm), setting=pl.lit(setting), seed=pl.lit(seed)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    result.write_parquet(temporary)
    pl.DataFrame(importance).write_parquet(importance_path)
    temporary.rename(destination)
    print(
        f"scored {setting}/{arm}/seed{seed}: {result.height:,} cases "
        f"in {time.monotonic() - start:.0f}s",
        flush=True,
    )


def validate_inputs(frames: Path, paths: list[Path], columns: list[str]) -> None:
    """Refuse caches outside either preregistered fold and enforce the common population."""
    fold = frames.name
    expected = 54 if fold == "mid_2025_to_mid_2026" else 67
    if len(paths) != expected or len(columns) != 24:
        raise ValueError("unexpected weekly run count or production feature count")
    dates = {
        "mid_2025_to_mid_2026": ("2024-04-01", "2025-07-01", "2026-07-01"),
        "backwards_2025_07_to_2026_06": ("2025-07-01", "2026-07-01", "2025-07-01"),
    }
    if fold not in dates:
        raise ValueError("unsupported fold")
    lower, boundary, upper = dates[fold]
    source_train = pl.scan_parquet(paths[0])
    bounds = source_train.select(
        pl.col("valid_time").min().alias("first"),
        pl.col("valid_time").max().alias("last"),
        pl.col("ensemble_member").unique().alias("members"),
    ).collect()
    if (
        str(bounds["first"][0])[:10] < lower
        or str(bounds["last"][0])[:10] >= boundary
        or bounds["members"].to_list() != [0]
    ):
        raise ValueError("training cache violates fold or control-member protocol")
    for path in paths[1:]:
        times = (
            pl.scan_parquet(path)
            .select(
                pl.col("valid_time").min().alias("first"), pl.col("valid_time").max().alias("last")
            )
            .collect()
        )
        validation_lower = "2025-07-01" if fold == "mid_2025_to_mid_2026" else "2024-04-01"
        if str(times["first"][0])[:10] < validation_lower or str(times["last"][0])[:10] >= upper:
            raise ValueError("validation cache lies outside the planned year")
    ids = source_train.select("time_series_id").unique().collect()["time_series_id"]
    if len(ids) != 31:
        raise ValueError("expected 31 training series")
    for path in paths[1:]:
        current = (
            pl.scan_parquet(path).select("time_series_id").unique().collect()["time_series_id"]
        )
        if set(current) != set(ids):
            raise ValueError("validation series differ from training")


def main() -> None:
    """Prepare weather corrections, or fit and score the predeclared experiment jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=["controlled", "sensitivity", "exact_production"],
        default=["controlled", "sensitivity"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    correction_controls()
    settings = Settings()
    metadata = pl.read_parquet(settings.metadata_path)
    columns = sorted(configuration()["selected_features"])
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.frames / "train.parquet", *sorted((args.frames / "val").glob("*.parquet"))]
    validate_inputs(args.frames, paths, columns)
    for path in paths:
        prepare(path, args.output / "prepared" / path.name, metadata, columns)
    if args.prepare_only:
        return
    version = DeltaTable(CLEANED_POWER_DELTA_URI).version()
    observations = pl.DataFrame(
        scan_power()
        .select(
            "time_series_id",
            pl.col("time").alias("valid_time"),
            pl.col("power").alias("power_actual"),
        )
        .collect()
    )
    if (
        DeltaTable(CLEANED_POWER_DELTA_URI).version() != version
        or observations.select("time_series_id", "valid_time").is_duplicated().any()
    ):
        raise ValueError("unstable or duplicate observation snapshot")
    capacity_version = DeltaTable(settings.effective_capacity_data_path).version()
    capacities = pl.read_delta(
        settings.effective_capacity_data_path, version=capacity_version
    ).select("time_series_id", "effective_capacity_mw")
    if capacities["time_series_id"].n_unique() != capacities.height:
        raise ValueError("duplicate capacity keys")
    ids = (
        pl.scan_parquet(paths[0])
        .select("time_series_id")
        .unique()
        .collect()["time_series_id"]
        .sort()
        .to_list()
    )
    permutation = np.random.default_rng(2026101201).permutation(len(ids))
    labels = {
        series: f"S{position + 1:02d}" for series, position in zip(ids, permutation, strict=True)
    }
    manifest = {
        "script_sha256": digest(Path(__file__)),
        "features": columns,
        "parameters": {
            name: parameters(name, 0) for name in ("controlled", "sensitivity", "exact_production")
        },
        "power_version": version,
        "capacity_version": capacity_version,
        "capacity_source": str(Path(settings.effective_capacity_data_path).resolve()),
        "prepared": {
            path.name: json.loads(
                (args.output / "prepared" / path.name).with_suffix(".json").read_text()
            )
            for path in paths
        },
        "xgboost_version": xgb.__version__,
        "scope": (
            "Study scores on cached weekly runs, not leaderboard scores; "
            "31-series anonymous permutation 2026101201."
        ),
    }
    stamp = args.output / "fit_manifest.json"
    if stamp.exists() and json.loads(stamp.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("fit provenance differs; choose a new output directory")
    stamp.write_text(json.dumps(manifest, indent=2) + "\n")
    for setting in args.settings:
        for arm in args.arms:
            for seed in args.seeds:
                run_job(
                    args, metadata, columns, observations, capacities, labels, arm, setting, seed
                )


if __name__ == "__main__":
    main()
