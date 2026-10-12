"""Diagnose marginal calibration from completed corrected-pipeline forecasts without fitting.

The output contains only type aggregates and hashes. Each forecast parquet is read separately;
only per-series monthly sufficient statistics survive to the next run. Native control-member
quantile knots retain their nominal levels and never become nine exchangeable samples.
"""

import argparse
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, cast

# Polars initialises its worker pool on import, so the cap precedes the import.
os.environ["POLARS_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import polars as pl
from scipy import stats

KEYS: Final[tuple[str, ...]] = (
    "time_series_id",
    "power_fcst_init_time",
    "nwp_init_time",
    "valid_time",
)
LEVELS19: Final[np.ndarray] = (np.arange(19) + 0.5) / 19
LEVELS9: Final[np.ndarray] = np.arange(1, 10) / 10
STUDY_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
NOVEMBER_BOUNDARY: Final[datetime] = datetime(2024, 11, 13, tzinfo=UTC)
MAY_BOUNDARY: Final[datetime] = datetime(2026, 5, 12, tzinfo=UTC)
BOOTSTRAP_SEED: Final[int] = 20261012
RESAMPLES: Final[int] = 2000
QUANTILE_RECIPE: Final[str] = "r2_e1_c1_a3w_d3"
REVIEWED_FITTER_SHA256: Final[frozenset[str]] = frozenset(
    {
        "6494889ab56a674b50d86553ab35ebaa04b37ab0d3943889c34f39f69fc20701",
        "b073be43f8809290746d952bf1acf988ea1f01cf61a7e2ed11e6f51f2de57ca3",
    }
)
POINT_RECIPE: Final[str] = "r2_c4_c2_c1_a3w_d3"
RECIPES: Final[frozenset[str]] = frozenset({"production", POINT_RECIPE, QUANTILE_RECIPE})
REVIEWED_ENCODER_HASHES: Final[dict[str, str]] = {
    "arms.py": "9e0ee05a39d054932f014b5cb9e99a111b3168b584152859304ca5bff9f03150",
    "fit_arms.py": "8a33ecfc55a7dcb3f59b13a7004d25629a9098a3a574fdedac46346d50b1a0af",
}
DELIVERED_METRICS: Final[tuple[str, ...]] = (
    "pinball19",
    "mae",
    "bias",
    "coverage80",
    "coverage90",
    "width80",
    "width90",
    "spread",
    *(f"pinball19_level_{index:02d}" for index in range(19)),
    *(f"cdf_bin_{index:02d}" for index in range(10)),
)
NINE_METRICS: Final[tuple[str, ...]] = (
    "pinball9",
    "coverage80",
    "width80",
    "crossing_rate",
    "invalid_interval_rate",
    "crossing_extent",
    *(f"pinball9_level_{index:02d}" for index in range(9)),
)


@dataclass(frozen=True)
class FitInput:
    """One completed fit, its verified forecast inventory and safe reporting identity."""

    root: Path
    recipe: str
    seed: int
    manifest: dict[str, Any]
    completion: dict[str, Any]
    forecasts: tuple[Path, ...]
    control_encoding_verified: bool


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_hash(*, frame: pl.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.write_ipc(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _read_frames(*, root: Path) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("complete") is not True or manifest["context"]["pipeline_tag"] != "combined":
        raise ValueError("diagnostics require complete combined-pipeline exports")
    context_hash = hashlib.sha256(
        json.dumps(manifest["context"], sort_keys=True).encode()
    ).hexdigest()
    if context_hash != manifest["input_fingerprint"]:
        raise ValueError("frame context fingerprint mismatch")
    for name, digest in manifest["files"].items():
        if _sha256(path=root / name) != digest:
            raise ValueError("frozen frame file fingerprint mismatch")
    window = manifest["window"]
    for field in ("train_start", "train_end", "val_start", "val_end"):
        instant = datetime.fromisoformat(window[field])
        if instant.tzinfo is None or instant >= STUDY_END:
            raise ValueError("frame window lacks timezone or reaches the held-out boundary")
    power = pl.read_parquet(root / "bounded_power.parquet").select(
        "time_series_id", "time", "power"
    )
    capacity = pl.read_parquet(root / "capacity.parquet").select(
        "time_series_id", "effective_capacity_mw"
    )
    metadata = pl.read_parquet(root / "metadata.parquet").select(
        "time_series_id", pl.col("time_series_type").cast(pl.String)
    )
    if power.select("time_series_id", "time").is_duplicated().any():
        raise ValueError("duplicate observed-power keys")
    if power.filter(pl.col("time").is_null() | (pl.col("time") >= STUDY_END)).height:
        raise ValueError("observations exceed the study boundary")
    if power.filter(pl.col("power").is_not_null() & ~pl.col("power").is_finite()).height:
        raise ValueError("non-finite observed power")
    if (
        capacity["time_series_id"].n_unique() != capacity.height
        or capacity.filter(
            pl.col("effective_capacity_mw").is_null()
            | ~pl.col("effective_capacity_mw").is_finite()
            | (pl.col("effective_capacity_mw") <= 0)
        ).height
    ):
        raise ValueError("score capacities must be unique, finite and positive")
    if (
        metadata["time_series_id"].n_unique() != metadata.height
        or metadata.null_count().sum_horizontal().item()
    ):
        raise ValueError("metadata is duplicated or incomplete")
    allowed_types = {"PV", "Wind", "Disaggregated Demand", "Raw Flow", "BESS", "Biofuel"}
    if not set(metadata["time_series_type"].to_list()) <= allowed_types:
        raise ValueError("metadata contains an unapproved public type label")
    return manifest, power, capacity, metadata


def _read_fit(*, root: Path, frames: dict[str, Any], frame_root: Path) -> FitInput:
    manifest = json.loads((root / "manifest.json").read_text())
    completion = json.loads((root / "complete.json").read_text())
    if manifest["recipe"] not in RECIPES or manifest["input_manifest"] != frames:
        raise ValueError("fit recipe or frozen frame provenance mismatch")
    if manifest["input_manifest_sha256"] != _sha256(path=frame_root / "manifest.json"):
        raise ValueError("fit refers to a different frame manifest")
    if manifest["input_fingerprint"] != frames["input_fingerprint"]:
        raise ValueError("fit input context mismatch")
    for name, digest in completion["files"].items():
        if _sha256(path=root / name) != digest:
            raise ValueError("completed fit artifact fingerprint mismatch")
    expected = {name for name in frames["files"] if name.startswith("val/")}
    paths = tuple(sorted((root / "forecasts").glob("init=*.parquet")))
    if {f"val/{path.name}" for path in paths} != expected:
        raise ValueError("forecast run inventory differs from complete frame inventory")
    if any(f"forecasts/{path.name}" not in completion["files"] for path in paths):
        raise ValueError("unfingerprinted forecast file")
    if int(manifest["seed"]) not in {0, 1, 2}:
        raise ValueError("only preregistered fitting seeds zero, one and two are supported")
    quantile = manifest["recipe"] == QUANTILE_RECIPE
    if completion["source_member_count"] != 51 or completion["sample_count"] != (
        459 if quantile else 51
    ):
        raise ValueError("completion manifest has unexpected sample semantics")
    dependencies = {
        Path(name).name: digest for name, digest in manifest["dependency_sha256"].items()
    }
    verified = (
        quantile
        and manifest["script_sha256"] in REVIEWED_FITTER_SHA256
        and all(
            dependencies.get(name) == digest for name, digest in REVIEWED_ENCODER_HASHES.items()
        )
    )
    return FitInput(
        root=root,
        recipe=manifest["recipe"],
        seed=int(manifest["seed"]),
        manifest=manifest,
        completion=completion,
        forecasts=paths,
        control_encoding_verified=verified,
    )


def _pinball(*, residual: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.maximum(levels * residual, (levels - 1) * residual)


def _empirical(*, values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.sort(values, axis=1)[:, np.ceil(values.shape[1] * levels).astype(int) - 1]


def _cdf(*, values: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return (
        (values < actual[:, None]).sum(axis=1) + 0.5 * (values == actual[:, None]).sum(axis=1)
    ) / values.shape[1]


def _delivered(
    *, values: np.ndarray, actual: np.ndarray, scale: np.ndarray
) -> dict[str, np.ndarray]:
    quantiles = _empirical(values=values, levels=LEVELS19)
    losses = _pinball(residual=actual[:, None] - quantiles, levels=LEVELS19) / scale[:, None]
    average = values.mean(axis=1)
    result = {
        "pinball19": 2 * losses.mean(axis=1),
        "mae": abs(average - actual) / scale,
        "bias": (average - actual) / scale,
        "spread": values.std(axis=1) / scale,
    }
    for index in range(19):
        result[f"pinball19_level_{index:02d}"] = losses[:, index]
    for mass in (0.8, 0.9):
        lower, upper = _empirical(
            values=values, levels=np.array([(1 - mass) / 2, (1 + mass) / 2])
        ).T
        result[f"coverage{int(mass * 100)}"] = ((actual >= lower) & (actual <= upper)).astype(float)
        result[f"width{int(mass * 100)}"] = (upper - lower) / scale
    cdf = _cdf(values=values, actual=actual)
    bins = np.minimum((cdf * 10).astype(int), 9)
    for index in range(10):
        result[f"cdf_bin_{index:02d}"] = (bins == index).astype(float)
    return result


def _nine(*, knots: np.ndarray, actual: np.ndarray, scale: np.ndarray) -> dict[str, np.ndarray]:
    losses = _pinball(residual=actual[:, None] - knots, levels=LEVELS9) / scale[:, None]
    drops = np.maximum(knots[:, :-1] - knots[:, 1:], 0)
    result = {
        "pinball9": 2 * losses.mean(axis=1),
        "coverage80": ((actual >= knots[:, 0]) & (actual <= knots[:, -1])).astype(float),
        "width80": (knots[:, -1] - knots[:, 0]) / scale,
        "crossing_rate": (drops > 0).any(axis=1).astype(float),
        "invalid_interval_rate": (knots[:, 0] > knots[:, -1]).astype(float),
        "crossing_extent": drops.sum(axis=1) / scale,
    }
    for index in range(9):
        result[f"pinball9_level_{index:02d}"] = losses[:, index]
    return result


def _monthly(
    *, rows: pl.DataFrame, metrics: dict[str, np.ndarray], fit: FitInput, variant: str
) -> pl.DataFrame:
    data = rows.select(
        "time_series_id", "time_series_type", "valid_time", "nwp_init_time", "lead_hours"
    )
    data = data.with_columns(
        *[pl.Series(name, values) for name, values in metrics.items()],
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
        lead_band=pl.when(pl.col("lead_hours") < 72)
        .then(pl.lit("days 0 to 2"))
        .when(pl.col("lead_hours") < 264)
        .then(pl.lit("days 3 to 10"))
        .otherwise(pl.lit("days 11 onward")),
        source_era=pl.when(pl.col("nwp_init_time") < NOVEMBER_BOUNDARY)
        .then(pl.lit("before November 2024 boundary"))
        .otherwise(pl.lit("after November 2024 boundary")),
    )
    return (
        data.group_by("time_series_id", "time_series_type", "month", "lead_band", "source_era")
        .agg(*[pl.col(name).sum() for name in metrics], n=pl.len())
        .with_columns(recipe=pl.lit(fit.recipe), seed=pl.lit(fit.seed), variant=pl.lit(variant))
    )


def _read_run(
    *,
    path: Path,
    fit: FitInput,
    frame_root: Path,
    power: pl.DataFrame,
    capacity: pl.DataFrame,
    metadata: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray, str, int]:
    predictions = pl.read_parquet(path).select(*KEYS, "ensemble_member", "power_fcst")
    if predictions.is_empty() or predictions.select(*KEYS, "ensemble_member").is_duplicated().any():
        raise ValueError("empty predictions or duplicate sample keys")
    if (
        predictions.null_count().sum_horizontal().item()
        or predictions.filter(~pl.col("power_fcst").is_finite()).height
    ):
        raise ValueError("missing keys or non-finite predictions")
    sample_count = int(fit.completion["sample_count"])
    grouped = (
        predictions.group_by(*KEYS)
        .agg(
            labels=pl.col("ensemble_member").sort(),
            values=pl.col("power_fcst").sort_by("ensemble_member"),
        )
        .sort(*KEYS)
    )
    if grouped.filter(pl.col("labels") != pl.lit(list(range(sample_count)))).height:
        raise ValueError("sample labels are not the complete expected encoding")
    expected = (
        pl.read_parquet(frame_root / "val" / path.name, columns=list(KEYS)).unique().sort(*KEYS)
    )
    if not grouped.select(*KEYS).equals(expected):
        raise ValueError("predicted keys differ from frozen validation keys")
    rows = (
        grouped.drop("labels", "values")
        .join(
            power.rename({"time": "valid_time", "power": "actual"}),
            on=["time_series_id", "valid_time"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .join(capacity, on="time_series_id", how="left", validate="m:1", maintain_order="left")
        .join(metadata, on="time_series_id", how="left", validate="m:1", maintain_order="left")
    )
    if rows["effective_capacity_mw"].null_count() or rows["time_series_type"].null_count():
        raise ValueError("forecast series lacks frozen capacity or type")
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
    )
    if rows.filter((pl.col("lead_hours") < 0) | (pl.col("valid_time") >= STUDY_END)).height:
        raise ValueError("invalid lead or held-out observation")
    signature = _frame_hash(frame=rows)
    supported = rows["actual"].is_not_null()
    values = np.array(grouped.filter(supported)["values"].to_list(), dtype=float)
    if not supported.any():
        raise ValueError("forecast run has no target-supported rows")
    return rows.filter(supported), values, signature, rows.height - int(cast(int, supported.sum()))


def _summarise(
    *,
    fits: list[FitInput],
    frame_root: Path,
    power: pl.DataFrame,
    capacity: pl.DataFrame,
    metadata: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    pieces = []
    signatures: dict[str, str] = {}
    exclusions: dict[str, int] = {}
    for fit in fits:
        for path in fit.forecasts:
            rows, values, signature, excluded = _read_run(
                path=path,
                fit=fit,
                frame_root=frame_root,
                power=power,
                capacity=capacity,
                metadata=metadata,
            )
            if signatures.setdefault(path.name, signature) != signature:
                raise ValueError(
                    "recipes or seeds have different keys, targets, capacities or types"
                )
            exclusions[path.name] = excluded
            actual, scale = rows["actual"].to_numpy(), rows["effective_capacity_mw"].to_numpy()
            pieces.append(
                _monthly(
                    rows=rows,
                    metrics=_delivered(values=values, actual=actual, scale=scale),
                    fit=fit,
                    variant="delivered",
                )
            )
            if fit.control_encoding_verified:
                variants = {
                    "control_native": values[:, :9],
                    "control_rearranged": np.sort(values[:, :9], axis=1),
                    "pooled_nine": _empirical(values=values, levels=LEVELS9),
                }
                for variant, knots in variants.items():
                    pieces.append(
                        _monthly(
                            rows=rows,
                            metrics=_nine(knots=knots, actual=actual, scale=scale),
                            fit=fit,
                            variant=variant,
                        )
                    )
    return pl.concat(pieces, how="diagonal_relaxed"), exclusions


def _scopes(*, monthly: pl.DataFrame) -> list[tuple[str, str, str]]:
    return [
        (kind, band, era)
        for kind in ("all", *sorted(monthly["time_series_type"].unique().to_list()))
        for band in ("all", "days 0 to 2", "days 3 to 10", "days 11 onward")
        for era in ("all", *sorted(monthly["source_era"].unique().to_list()))
    ]


def _filter_scope(*, frame: pl.DataFrame, scope: tuple[str, str, str]) -> pl.DataFrame:
    for column, value in zip(("time_series_type", "lead_band", "source_era"), scope, strict=True):
        if value != "all":
            frame = frame.filter(pl.col(column) == value)
    return frame


def _absolute(*, monthly: pl.DataFrame) -> pl.DataFrame:
    output = []
    for (recipe, variant), part in monthly.group_by("recipe", "variant"):
        metrics = DELIVERED_METRICS if variant == "delivered" else NINE_METRICS
        for scope in _scopes(monthly=part):
            subset = _filter_scope(frame=part, scope=scope)
            if subset.is_empty():
                continue
            means = (
                subset.group_by("seed", "time_series_id")
                .agg(*[(pl.col(name).sum() / pl.col("n").sum()).alias(name) for name in metrics])
                .group_by("seed")
                .agg(*[pl.col(name).mean() for name in metrics])
            )
            output.extend(
                {
                    "recipe": recipe,
                    "variant": variant,
                    "type": scope[0],
                    "lead_band": scope[1],
                    "source_era": scope[2],
                    "metric": metric,
                    "value_percent": 100 * float(cast(float, means[metric].mean())),
                    "seed_spread_points": 100
                    * (
                        float(cast(float, means[metric].max()))
                        - float(cast(float, means[metric].min()))
                    ),
                    "seeds": means.height,
                    "series": subset["time_series_id"].n_unique(),
                    "months": subset["month"].n_unique(),
                    "rows_per_seed": subset["n"].sum() // means.height,
                }
                for metric in metrics
            )
    return pl.DataFrame(output)


def _contrast_arrays(
    *, left: pl.DataFrame, right: pl.DataFrame, metric: str
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    keys = ["seed", "time_series_id", "month"]
    left = left.group_by(keys).agg(total=pl.col(metric).sum(), n=pl.col("n").sum()).sort(keys)
    right = right.group_by(keys).agg(total=pl.col(metric).sum(), n=pl.col("n").sum()).sort(keys)
    if not left.select(*keys, "n").equals(right.select(*keys, "n")):
        raise ValueError("contrast monthly sufficient statistics are not paired")
    seeds = sorted(left["seed"].unique().to_list())
    series = sorted(left["time_series_id"].unique().to_list())
    months = sorted(left["month"].unique().to_list())
    indices = [
        {value: index for index, value in enumerate(axis)} for axis in (seeds, series, months)
    ]
    totals = np.zeros((len(seeds), len(series), len(months)))
    counts = np.zeros_like(totals)
    differences = left.with_columns(total=left["total"] - right["total"])
    for row in differences.iter_rows(named=True):
        index = tuple(mapping[row[name]] for mapping, name in zip(indices, keys, strict=True))
        totals[index], counts[index] = row["total"], row["n"]
    return totals, counts, seeds, months


def _equal_series(*, totals: np.ndarray, counts: np.ndarray) -> np.ndarray:
    summed_counts = counts.sum(axis=-1)
    means = np.divide(
        totals.sum(axis=-1),
        summed_counts,
        out=np.full_like(summed_counts, np.nan),
        where=summed_counts > 0,
    )
    return np.nanmean(means, axis=-1)


def _interval(*, totals: np.ndarray, counts: np.ndarray) -> dict[str, float | int | bool]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(RESAMPLES)
    for index in range(RESAMPLES):
        seed = int(generator.integers(totals.shape[0]))
        months = generator.integers(totals.shape[2], size=totals.shape[2])
        draws[index] = _equal_series(totals=totals[seed][:, months], counts=counts[seed][:, months])
    seed_means = _equal_series(totals=totals, counts=counts)
    monthly = np.array(
        [
            _equal_series(
                totals=totals[:, :, index : index + 1], counts=counts[:, :, index : index + 1]
            ).mean()
            for index in range(totals.shape[2])
        ]
    )
    equal_month = monthly.mean()
    half_width = (
        stats.t.ppf(0.975, df=len(monthly) - 1) * stats.sem(monthly) if len(monthly) > 1 else np.nan
    )
    return {
        "difference_points": 100 * seed_means.mean(),
        "lower95_points": 100 * np.percentile(draws, 2.5),
        "upper95_points": 100 * np.percentile(draws, 97.5),
        "seed_spread_points": 100 * np.ptp(seed_means),
        "equal_month_difference_points": 100 * equal_month,
        "t_lower95_points": 100 * (equal_month - half_width),
        "t_upper95_points": 100 * (equal_month + half_width),
        "months_lower_loss": int((monthly < 0).sum()),
        "months": len(monthly),
        "seeds": len(seed_means),
        "series": totals.shape[1],
        "rows_per_seed": int(counts.sum() / counts.shape[0]),
        "provisional": totals.shape[0] < 3 or totals.shape[2] < 6,
    }


def _contrasts(*, monthly: pl.DataFrame) -> pl.DataFrame:
    output = []
    specifications = [
        (QUANTILE_RECIPE, "delivered", recipe, "delivered", "pinball19")
        for recipe in ("production", POINT_RECIPE)
    ]
    specifications += [
        (QUANTILE_RECIPE, "pooled_nine", QUANTILE_RECIPE, "control_native", "pinball9"),
        (QUANTILE_RECIPE, "pooled_nine", QUANTILE_RECIPE, "control_rearranged", "pinball9"),
    ]
    for treatment, variant, reference, reference_variant, metric in specifications:
        left = monthly.filter((pl.col("recipe") == treatment) & (pl.col("variant") == variant))
        right = monthly.filter(
            (pl.col("recipe") == reference) & (pl.col("variant") == reference_variant)
        )
        if left.is_empty() or right.is_empty():
            continue
        if set(left["seed"].unique().to_list()) != set(right["seed"].unique().to_list()):
            continue
        for scope in _scopes(monthly=left):
            scoped_left, scoped_right = (
                _filter_scope(frame=left, scope=scope),
                _filter_scope(frame=right, scope=scope),
            )
            if scoped_left.is_empty() or scoped_right.is_empty():
                continue
            totals, counts, _, _ = _contrast_arrays(
                left=scoped_left, right=scoped_right, metric=metric
            )
            output.append(
                {
                    "treatment": treatment,
                    "variant": variant,
                    "reference": reference,
                    "reference_variant": reference_variant,
                    "metric": metric,
                    "type": scope[0],
                    "lead_band": scope[1],
                    "source_era": scope[2],
                    **_interval(totals=totals, counts=counts),
                }
            )
    return (
        pl.DataFrame(output)
        if output
        else pl.DataFrame(schema={"treatment": pl.String, "metric": pl.String})
    )


def _synthetic() -> pl.DataFrame:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    values = generator.normal(size=(4096, 51))
    actual = generator.normal(size=4096)
    cases = {
        "finite iid calibrated": values,
        "underdispersed": values * 0.3,
        "biased": values + 1,
        "tied": np.zeros_like(values),
    }
    rows = []
    for name, sample in cases.items():
        metrics = _delivered(values=sample, actual=actual, scale=np.ones(4096))
        for metric, result in metrics.items():
            rows.append({"case": name, "metric": metric, "value": float(result.mean())})
    if not np.array_equal(
        _cdf(
            values=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), actual=np.array([-1.0, 0.0, 1.0])
        ),
        np.array([0.0, 0.5, 1.0]),
    ):
        raise ValueError("synthetic deterministic tie check failed")
    calibrated_coverage = np.mean(
        _delivered(values=values, actual=actual, scale=np.ones(4096))["coverage80"]
    )
    narrow_coverage = np.mean(
        _delivered(values=values * 0.3, actual=actual, scale=np.ones(4096))["coverage80"]
    )
    if not 0.7 < calibrated_coverage < 0.9 or narrow_coverage >= calibrated_coverage - 0.2:
        raise ValueError("synthetic calibrated/underdispersed diagnostic failed")
    reference = np.minimum((np.arange(52) / 51 * 10).astype(int), 9)
    rows.extend(
        {
            "case": "finite iid no-tie reference",
            "metric": f"cdf_bin_{index:02d}",
            "value": float(np.mean(reference == index)),
        }
        for index in range(10)
    )
    crossed = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]])
    native = _nine(knots=crossed, actual=np.array([0.5]), scale=np.ones(1))
    rearranged = _nine(knots=np.sort(crossed, axis=1), actual=np.array([0.5]), scale=np.ones(1))
    if (
        native["coverage80"][0] != 0
        or native["width80"][0] >= 0
        or rearranged["coverage80"][0] != 1
    ):
        raise ValueError("native versus rearranged crossing oracle failed")
    return pl.DataFrame(rows)


def _source_inventory(
    *, frame_root: Path, manifest: dict[str, Any], metadata: pl.DataFrame, output: Path
) -> None:
    features = [
        name
        for name in manifest["context"]["engineered_features"]
        if name not in {*KEYS, "ensemble_member", "power"}
    ]
    parts = []
    for relative in sorted(name for name in manifest["files"] if name.startswith("val/")):
        frame = (
            pl.read_parquet(frame_root / relative, columns=[*KEYS, *features])
            .join(metadata, on="time_series_id", validate="m:1", maintain_order="left")
            .with_columns(
                source_era=pl.when(pl.col("nwp_init_time") < NOVEMBER_BOUNDARY)
                .then(pl.lit("before November 2024 boundary"))
                .otherwise(pl.lit("after November 2024 boundary")),
                may_cycle=pl.when(pl.col("nwp_init_time") < MAY_BOUNDARY)
                .then(pl.lit("before May 2026 boundary"))
                .otherwise(pl.lit("after May 2026 boundary")),
            )
        )
        parts.append(
            frame.group_by("time_series_type", "source_era", "may_cycle").agg(
                source_rows=pl.len(),
                source_runs=pl.col("nwp_init_time").n_unique(),
                **{f"null_{name}": pl.col(name).null_count() for name in features},
            )
        )
    inventory = (
        pl.concat(parts)
        .group_by("time_series_type", "source_era", "may_cycle")
        .agg(pl.exclude("time_series_type", "source_era", "may_cycle").sum())
    )
    inventory.write_parquet(output / "source_inventory.parquet")


def _table(*, frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "Pending: no matched completed fits."
    header = "| " + " | ".join(frame.columns) + " |\n"
    header += "|" + "---|" * frame.width + "\n"
    return header + "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |" for row in frame.iter_rows()
    )


def _write_report(
    *,
    output: Path,
    absolute: pl.DataFrame,
    contrasts: pl.DataFrame,
    fits: list[FitInput],
    exclusions: dict[str, int],
    elapsed: float,
) -> None:
    absolute.write_parquet(output / "absolute.parquet")
    contrasts.write_parquet(output / "contrasts.parquet")
    synthetic = _synthetic()
    synthetic.write_parquet(output / "synthetic.parquet")
    summaries = absolute.filter(
        (pl.col("type") == "all")
        & (pl.col("lead_band") == "all")
        & (pl.col("source_era") == "all")
        & ~pl.col("metric").str.contains("level|cdf_bin")
    )
    if not contrasts.is_empty():
        headline_contrasts = contrasts.filter(
            (pl.col("type") == "all")
            & (pl.col("lead_band") == "all")
            & (pl.col("source_era") == "all")
        )
    else:
        headline_contrasts = contrasts
    missing = sorted(RECIPES - {fit.recipe for fit in fits})
    available = "; ".join(
        f"{recipe}: seeds {sorted(fit.seed for fit in fits if fit.recipe == recipe)}"
        for recipe in sorted(RECIPES)
    )
    verified_controls = sorted(fit.seed for fit in fits if fit.control_encoding_verified)
    text = f"""# Marginal calibration diagnostics on the corrected pipeline

Study scores, not leaderboard scores. Equal weight per series, then per fitting seed.
Coverage and CDF bins are percentages of cases. Losses, bias, spread and widths are percentages
of frozen capacity. Negative bias means underprediction. Width alone is not a ranking.

{_table(frame=summaries)}

{_table(frame=headline_contrasts)}

Paired differences are treatment minus reference. Intervals resample whole valid-time months
and one paired fitting seed 2,000 times. Equal-month t-intervals target a separate estimand.
Provisional means fewer than three seeds or six months. Fold B and sensitivity confirmation
remain required before any recommendation.

The delivered empirical distribution uses inverse-CDF rank ceil(n*q)-1. The primary score is
twice mean pinball at 19 midpoint levels. Native control knots use nominal .1 through .9 in the
separate nine-level diagnostic. Native crossed endpoints fail coverage; their width is signed.
The rearranged control variant is exploratory. Control 90% coverage and CDF are pending.

Delivered-sample CDF diagnostic: (#values<actual + .5*#values==actual)/n, no randomisation.
Bins include their lower endpoint; the last includes one. Finite continuous iid draws have
equal reference mass at k/n, k=0,...,n, not continuous uniform density. This reference does not
apply to fitted quantile pools. No forecast recipe is asserted to be iid; CDF shape cannot
identify the cause of miscalibration.

Native crossings count any descending adjacent knots; crossing extent sums positive adjacent
drops. Invalid intervals have q.1>q.9. All level-specific losses, types, lead bands and source-era
summaries are in absolute.parquet; paired contrasts are in contrasts.parquet. Source-run era
inventory and engineered-column null counts are in source_inventory.parquet.

November eras use source nwp_init_time before/after 13 November 2024. Availability-boundary
provenance remains pending independent ingestion verification; seasons and exposure differ.
The 12 May 2026 source-cycle comparison remains pending. Training coverage, calibration fitting,
joint coherence, G5 and G9 are outside this report.

Pending recipes: {", ".join(missing) if missing else "none"}.
Available fits: {available}.
Positively verified control-member extraction seeds: {verified_controls}.
Unverified control encoding remains pending. Missing seeds, folds and settings remain pending.
Target-unsupported forecast keys excluded per fit: {sum(exclusions.values())}.
Scoring time: {elapsed:.1f} seconds. Synthetic controls: synthetic.parquet.
"""
    (output / "report.md").write_text(text)


def main() -> None:
    """Validate saved fits and write a fresh immutable aggregate diagnostic report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--fit", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    frame_manifest_hash = _sha256(path=args.frames / "manifest.json")
    frame_manifest, power, capacity, metadata = _read_frames(root=args.frames)
    fits = [
        _read_fit(root=path, frames=frame_manifest, frame_root=args.frames) for path in args.fit
    ]
    if len({(fit.recipe, fit.seed) for fit in fits}) != len(fits):
        raise ValueError("duplicate recipe and fitting seed")
    if len({(fit.manifest["device"], fit.manifest["setting"]) for fit in fits}) != 1:
        raise ValueError("one report requires one device and hyperparameter setting")
    args.output.mkdir(parents=True, exist_ok=False)
    _source_inventory(
        frame_root=args.frames, manifest=frame_manifest, metadata=metadata, output=args.output
    )
    monthly, exclusions = _summarise(
        fits=fits, frame_root=args.frames, power=power, capacity=capacity, metadata=metadata
    )
    absolute, contrasts = _absolute(monthly=monthly), _contrasts(monthly=monthly)
    if _sha256(path=args.frames / "manifest.json") != frame_manifest_hash:
        raise ValueError("frame manifest changed during diagnostics")
    for name in ("bounded_power.parquet", "capacity.parquet", "metadata.parquet"):
        if _sha256(path=args.frames / name) != frame_manifest["files"][name]:
            raise ValueError("frozen observations or metadata changed during diagnostics")
    for fit in fits:
        if json.loads((fit.root / "manifest.json").read_text()) != fit.manifest or (
            json.loads((fit.root / "complete.json").read_text()) != fit.completion
        ):
            raise ValueError("completed fit manifest changed during diagnostics")
    _write_report(
        output=args.output,
        absolute=absolute,
        contrasts=contrasts,
        fits=fits,
        exclusions=exclusions,
        elapsed=time.monotonic() - started,
    )
    safe_manifest = {
        "script_sha256": _sha256(path=Path(__file__)),
        "frame_manifest_sha256": _sha256(path=args.frames / "manifest.json"),
        "input_fingerprint": frame_manifest["input_fingerprint"],
        "fold": frame_manifest["window"]["fold_id"],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_probabilities": LEVELS19.tolist(),
        "control_probabilities": LEVELS9.tolist(),
        "empirical_inverse_cdf": "ceil(n*q)-1",
        "sample_counts": {fit.recipe: fit.completion["sample_count"] for fit in fits},
        "source_member_count": 51,
        "reviewed_control_encoder_hashes": REVIEWED_ENCODER_HASHES,
        "reviewed_fitter_sha256": sorted(REVIEWED_FITTER_SHA256),
        "dependencies": {name: version(name) for name in ("numpy", "polars", "scipy")},
        "resamples": RESAMPLES,
        "setting": fits[0].manifest["setting"],
        "device": fits[0].manifest["device"],
        "fits": [
            {
                "recipe": fit.recipe,
                "seed": fit.seed,
                "control_encoding_verified": fit.control_encoding_verified,
                "manifest_sha256": _sha256(path=fit.root / "manifest.json"),
                "completion_sha256": _sha256(path=fit.root / "complete.json"),
            }
            for fit in fits
        ],
        "files": {
            path.name: _sha256(path=path) for path in args.output.iterdir() if path.is_file()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(safe_manifest, indent=2))
    print("Completed immutable aggregate calibration diagnostics; no calibrators fitted.")


if __name__ == "__main__":
    main()
