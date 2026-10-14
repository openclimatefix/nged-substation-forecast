"""Audit saved training gains for five requested feature sets, keeping settings separate.

This post hoc description never fits a model. Padding credit returns to its
original real input before within-booster normalization. Ranges describe six
type-averaged fold/seed refits, not confidence or held-out feature benefit.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Final, cast

import fixed_pipeline_fit as fitter
import numpy as np
import polars as pl
import xgboost as xgb

RECIPES: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "solar_minimal_requested": (
        "PV",
        ("temperature_2m", "downward_short_wave_radiation_flux_surface"),
    ),
    "solar_expanded_requested": (
        "PV",
        (
            "temperature_2m",
            "dew_point_temperature_2m",
            "wind_speed_10m",
            "downward_long_wave_radiation_flux_surface",
            "downward_short_wave_radiation_flux_surface",
            "precipitation_surface",
            "categorical_precipitation_type_surface",
            "sun_east",
            "sun_north",
            "sun_up",
            "clear_sky_irradiance",
        ),
    ),
    "wind_minimal_requested": (
        "Wind",
        (
            "temperature_2m",
            "wind_direction_100m",
            "wind_direction_10m",
            "wind_speed_100m",
            "wind_speed_10m",
        ),
    ),
    "wind_minimal_uv_requested": (
        "Wind",
        (
            "temperature_2m",
            "wind_direction_100m",
            "wind_direction_10m",
            "wind_speed_100m",
            "wind_speed_10m",
            "wind_u_10m",
            "wind_v_10m",
            "wind_u_100m",
            "wind_v_100m",
        ),
    ),
    "wind_minimal_circular_requested": (
        "Wind",
        (
            "temperature_2m",
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_direction_10m_sin",
            "wind_direction_10m_cos",
            "wind_direction_100m_sin",
            "wind_direction_100m_cos",
        ),
    ),
}
FOLDS: Final[frozenset[str]] = frozenset({"mid_2025_to_mid_2026", "backwards_2025_07_to_2026_06"})
AUGMENTED_RECIPES: Final[frozenset[str]] = frozenset(
    {"solar_expanded_requested", "wind_minimal_uv_requested", "wind_minimal_circular_requested"}
)
SETTINGS: Final[frozenset[str]] = frozenset({"main", "sensitivity"})
PARAMETERS: Final[dict[str, dict[str, float | int]]] = {
    "main": {
        "rounds": 500,
        "max_depth": 6.0,
        "eta": 0.05,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
    },
    "sensitivity": {
        "rounds": 1200,
        "max_depth": 4.0,
        "eta": 0.03,
        "min_child_weight": 50.0,
        "lambda": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
    },
}
PUBLIC_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "recipe",
        "setting",
        "series_type",
        "feature",
        "mean_share",
        "min_share",
        "max_share",
        "refits",
        "min_series_per_refit",
        "max_series_per_refit",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    """Reject invalid aggregate/source invariants without exposing private values."""
    if not condition:
        raise ValueError(message)


def _plain(path: str | Path) -> Path:
    """Require canonical explicit unlinked paths."""
    path = Path(path)
    _require(
        condition=path.is_absolute() and path.resolve() == path,
        message="Noncanonical or linked source",
    )
    return path


def _sha(path: Path) -> str:
    """Hash a canonical source file."""
    _plain(path=path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _pin(path: Path, expected: str, fingerprints: dict[Path, str]) -> None:
    """Record an immutable fingerprint for a final independent source recheck."""
    if path in fingerprints:
        _require(
            condition=fingerprints[path] == expected, message="Reviewed source fingerprint changed"
        )
        return
    _require(condition=_sha(path=path) == expected, message="Reviewed source fingerprint changed")
    fingerprints[path] = expected


def _json(path: Path, fingerprints: dict[Path, str]) -> dict[str, Any] | list[Any]:
    """Read a private JSON artifact with an unchanged content fingerprint."""
    content = _plain(path=path).read_bytes()
    _pin(path=path, expected=hashlib.sha256(content).hexdigest(), fingerprints=fingerprints)
    return json.loads(content)


def _keys(value: dict[str, Any], names: set[str]) -> None:
    """Refuse missing or unknown configuration fields."""
    _require(
        condition=isinstance(value, dict) and set(value) == names,
        message="Unknown or missing configuration fields",
    )


def _complete_inventory(
    root: Path, listed: dict[str, str], extras: set[str], fingerprints: dict[Path, str]
) -> None:
    """Verify every completed file and refuse unknown or linked artifact entries."""
    for name, expected in listed.items():
        relative = Path(name)
        _require(
            condition=not relative.is_absolute() and ".." not in relative.parts,
            message="Artifact traversal refused",
        )
        _pin(path=root / relative, expected=expected, fingerprints=fingerprints)
    actual = set()
    for path in root.rglob("*"):
        _plain(path=path)
        if path.is_file():
            actual.add(str(path.relative_to(root)))
        else:
            _require(condition=path.is_dir(), message="Unknown artifact file type")
    _require(
        condition=actual == set(listed) | extras, message="Completed artifact inventory differs"
    )


def _gain_vector(raw: dict[str, float], shown: list[str]) -> np.ndarray:
    """Map named or positional gains to the actual ordered trained columns."""
    gains = np.zeros(len(shown), dtype=np.float64)
    assigned = set()
    for name, gain in raw.items():
        if name in shown:
            index = shown.index(name)
        else:
            _require(
                condition=name.startswith("f") and name[1:].isdigit(),
                message="Unmapped booster feature",
            )
            index = int(name[1:])
        _require(
            condition=index < len(shown)
            and index not in assigned
            and not isinstance(gain, bool)
            and math.isfinite(float(gain))
            and float(gain) >= 0,
            message="Invalid booster total gain",
        )
        gains[index] = float(gain)
        assigned.add(index)
    return gains


def _frame_inventory(root: Path, manifest: dict[str, Any], fingerprints: dict[Path, str]) -> None:
    """Verify context bytes and both historical and augmented inventory conventions."""
    context = cast(dict[str, Any], _json(path=root / "context.json", fingerprints=fingerprints))
    _require(condition=context == manifest["context"], message="Frame context differs")
    _complete_inventory(
        root=root,
        listed=manifest["files"],
        extras={"manifest.json", "context.json"},
        fingerprints=fingerprints,
    )


def _frame_lineage(
    *,
    recipe: str,
    fold: str,
    frames: Path,
    manifest: dict[str, Any],
    spec: dict[str, Any],
    fingerprints: dict[Path, str],
) -> None:
    """Bind minimal inputs to originals and derived inputs to their exact frozen parent."""
    binding = spec["original_frames"][fold]
    _keys(value=binding, names={"path", "manifest_sha256", "context_sha256"})
    original_root = _plain(path=binding["path"])
    _pin(
        path=original_root / "manifest.json",
        expected=binding["manifest_sha256"],
        fingerprints=fingerprints,
    )
    _pin(
        path=original_root / "context.json",
        expected=binding["context_sha256"],
        fingerprints=fingerprints,
    )
    parent = cast(
        dict[str, Any], _json(path=original_root / "manifest.json", fingerprints=fingerprints)
    )
    _require(
        condition=parent["complete"] is True
        and parent["context"]["pipeline_tag"] == "combined"
        and parent["window"]["fold_id"] == fold
        and "augmentation_code_sha256" not in parent["context"]
        and "parent_provenance" not in parent,
        message="Original frame binding is not a frozen combined export",
    )
    _frame_inventory(root=original_root, manifest=parent, fingerprints=fingerprints)
    if recipe not in AUGMENTED_RECIPES:
        _require(
            condition=frames == original_root and manifest == parent,
            message="Minimal recipe must consume the exact original frames",
        )
        return
    context = manifest["context"]
    provenance = manifest["parent_provenance"]
    _require(
        condition=frames != original_root
        and context["augmentation_code_sha256"]["builder"] == spec["augmentation_builder"]["sha256"]
        and context["parent_manifest_sha256"] == binding["manifest_sha256"]
        and context["parent_context_sha256"] == binding["context_sha256"]
        and context["parent_input_fingerprint"] == parent["input_fingerprint"]
        and provenance["manifest_sha256"] == binding["manifest_sha256"]
        and provenance["historical_parent_manifest"] == parent
        and provenance["original_frames"] == parent["frames"]
        and manifest["window"] == parent["window"]
        and all(context[name] == value for name, value in parent["context"].items()),
        message="Augmented frames differ from the explicitly bound frozen parent",
    )


def _model_shares(
    model: dict[str, Any],
    saved: xgb.Booster,
    real: list[str],
    shown: list[str],
    setting: str,
    kind: str,
    seed: int,
    device: str,
) -> np.ndarray:
    """Verify a saved booster and return normalized real-input gain shares."""
    _keys(
        value=model,
        names={
            "time_series_id",
            "model_file",
            "features",
            "scale",
            "total_gain",
            "rounds",
            "booster_runtime_config",
        },
    )
    config = model["booster_runtime_config"]
    learner = config["learner"]
    generic = learner["generic_param"]
    actual_device = generic["device"]
    cuda_device = actual_device == "cuda" or (
        isinstance(actual_device, str)
        and actual_device.startswith("cuda:")
        and actual_device.removeprefix("cuda:").isdigit()
    )
    _require(
        condition=int(generic["seed"]) == seed
        and ((device == "cpu" and actual_device == "cpu") or (device == "cuda" and cuda_device))
        and learner["gradient_booster"]["name"] == "gbtree",
        message="Recorded actual training seed, device or booster type differs",
    )
    expected = PARAMETERS[setting]
    _require(
        condition=learner["objective"]["name"] == "reg:squarederror"
        and model["scale"] == 1.0
        and model["rounds"] == expected["rounds"]
        and model["features"] == shown,
        message="Saved model is not the requested per-meter squared-error fit",
    )
    actual = learner["gradient_booster"]["tree_train_param"]
    _require(
        condition=all(
            math.isclose(float(actual[name]), value, rel_tol=1e-6, abs_tol=1e-8)
            for name, value in expected.items()
            if name != "rounds"
        ),
        message="Training settings differ",
    )
    _require(
        condition=saved.num_boosted_rounds() == expected["rounds"]
        and saved.num_features() == len(shown)
        and json.loads(saved.save_config())["learner"]["objective"]["name"] == "reg:squarederror",
        message="Saved booster structure/objective differs",
    )
    recorded = _gain_vector(raw=model["total_gain"], shown=shown)
    regenerated = _gain_vector(
        raw=cast(dict[str, float], saved.get_score(importance_type="total_gain")), shown=shown
    )
    _require(
        condition=np.array_equal(recorded, regenerated),
        message="Recorded gains differ from fingerprinted saved booster",
    )
    expected_shown = [
        *real,
        *[f"padding_{kind.replace(' ', '_')}_{index:02d}" for index in range(24 - len(real))],
    ]
    _require(
        condition=shown == expected_shown,
        message="Actual trained feature order or duplicate policy differs",
    )
    mapped = [*real, *[real[index % len(real)] for index in range(24 - len(real))]]
    combined = dict.fromkeys(real, 0.0)
    for name, gain in zip(mapped, recorded, strict=True):
        combined[name] += float(gain)
    total = sum(combined.values())
    _require(
        condition=math.isfinite(total) and total > 0,
        message="Zero-total-gain booster has no gain-share distribution",
    )
    shares = np.array([combined[name] / total for name in real])
    _require(
        condition=np.isfinite(shares).all()
        and math.isclose(float(shares.sum()), 1.0, abs_tol=1e-12),
        message="Invalid normalized total-gain shares",
    )
    return shares


def _verify_registry(*, registry: dict[str, Any], recipe: str, kind: str, real: list[str]) -> None:
    """Verify the exact point recipe and its real-source cyclic duplicate policy."""
    policy = registry[recipe]
    arm = policy.arm
    base = registry["production_cleaned"].arm
    _require(
        condition=sorted(arm.columns(kind)) == real
        and not policy.guarded_windchill
        and arm.objective == "reg:squarederror"
        and not arm.pooled_by_type
        and not arm.pooled_site_id
        and arm.quantile_levels is None
        and not arm.early_stopping
        and arm.colsample_bytree == 1.0,
        message="New registry recipe differs from the exact requested point model",
    )
    _require(
        condition=arm.cleaning == base.cleaning
        and arm.extra_cleaning == base.extra_cleaning
        and arm.recency_half_life_days is None
        and arm.train_last_months is None,
        message="Requested recipe changes historical cleaning or training weights/window",
    )
    width = max(
        len(item.arm.columns(kind))
        for name, item in registry.items()
        if name not in fitter.EXACT_RECIPES
    )
    _require(condition=width == 24, message="Registry maximum padded width changed")
    fixture = pl.DataFrame(
        {
            "time_series_type": [kind],
            **{
                name: pl.Series([float(index + 1)], dtype=pl.Float32)
                for index, name in enumerate(real)
            },
        }
    )
    prepared, fixture_columns = fitter._prepare(
        frame=fixture, recipe=policy, padding="equal", registry=registry
    )
    expected_shown = [
        *real,
        *[f"padding_{kind.replace(' ', '_')}_{index:02d}" for index in range(24 - len(real))],
    ]
    _require(
        condition=fixture_columns[kind] == expected_shown,
        message="Imported registry preparation order differs",
    )
    for index in range(24 - len(real)):
        _require(
            condition=prepared[f"padding_{kind.replace(' ', '_')}_{index:02d}"][0]
            == float(index % len(real) + 1),
            message="Imported fitter duplicate columns do not cycle over their own retained inputs",
        )


def _read_fit(
    run: dict[str, Any],
    spec: dict[str, Any],
    registry: dict[str, Any],
    fingerprints: dict[Path, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit one explicit completed fit and aggregate only its targeted meters."""
    _keys(
        value=run,
        names={
            "recipe",
            "setting",
            "fold",
            "seed",
            "path",
            "frames",
            "manifest_sha256",
            "completion_sha256",
            "frame_manifest_sha256",
        },
    )
    recipe, setting, fold, seed = (run[name] for name in ("recipe", "setting", "fold", "seed"))
    _require(
        condition=recipe in RECIPES
        and setting in SETTINGS
        and fold in FOLDS
        and type(seed) is int
        and seed in {0, 1, 2},
        message="Unknown importance fit",
    )
    root, frames = _plain(path=run["path"]), _plain(path=run["frames"])
    _pin(path=root / "manifest.json", expected=run["manifest_sha256"], fingerprints=fingerprints)
    _pin(path=root / "complete.json", expected=run["completion_sha256"], fingerprints=fingerprints)
    _pin(
        path=frames / "manifest.json",
        expected=run["frame_manifest_sha256"],
        fingerprints=fingerprints,
    )
    manifest = cast(dict[str, Any], _json(path=root / "manifest.json", fingerprints=fingerprints))
    completion = cast(dict[str, Any], _json(path=root / "complete.json", fingerprints=fingerprints))
    frame_manifest = cast(
        dict[str, Any], _json(path=frames / "manifest.json", fingerprints=fingerprints)
    )
    _require(
        condition=all(manifest[name] == run[name] for name in ("recipe", "setting", "seed"))
        and manifest["padding"] == "equal"
        and manifest["device"] == spec["device"]
        and manifest["script_sha256"] == spec["fitter"]["sha256"]
        and manifest["dependency_sha256"] == spec["fitter_dependency_sha256"]
        and manifest["input_manifest_sha256"] == run["frame_manifest_sha256"]
        and manifest["input_manifest"] == frame_manifest
        and frame_manifest["complete"] is True
        and frame_manifest["context"]["pipeline_tag"] == "combined"
        and frame_manifest["window"]["fold_id"] == fold
        and manifest["xgboost_version"] == xgb.__version__,
        message="Fit/frame source provenance differs",
    )
    _require(
        condition=completion["source_member_count"] == completion["sample_count"] == 51,
        message="Only per-meter 51-member point forecasts are accepted",
    )
    _complete_inventory(
        root=root,
        listed=completion["files"],
        extras={"manifest.json", "complete.json"},
        fingerprints=fingerprints,
    )
    _frame_inventory(root=frames, manifest=frame_manifest, fingerprints=fingerprints)
    _frame_lineage(
        recipe=recipe,
        fold=fold,
        frames=frames,
        manifest=frame_manifest,
        spec=spec,
        fingerprints=fingerprints,
    )
    metadata = pl.read_parquet(frames / "metadata.parquet").select(
        "time_series_id", pl.col("time_series_type").cast(pl.String)
    )
    _require(
        condition=metadata.height == 31
        and metadata.null_count().sum_horizontal().item() == 0
        and not metadata["time_series_id"].is_duplicated().any()
        and set(metadata["time_series_type"])
        <= {"PV", "Wind", "Disaggregated Demand", "Raw Flow", "BESS", "Biofuel"},
        message="Frozen population differs",
    )
    kinds = dict(metadata.iter_rows())
    kind, requested = RECIPES[recipe]
    real = sorted(requested)
    _verify_registry(registry=registry, recipe=recipe, kind=kind, real=real)
    features = cast(dict[str, Any], _json(path=root / "features.json", fingerprints=fingerprints))
    models = cast(list[dict[str, Any]], _json(path=root / "models.json", fingerprints=fingerprints))
    model_ids = [model["time_series_id"] for model in models]
    _require(
        condition=len(model_ids) == 31
        and len(set(model_ids)) == 31
        and set(model_ids) == set(kinds),
        message="Saved boosters do not cover the frozen per-meter population",
    )
    selected = [model for model in models if kinds[model["time_series_id"]] == kind]
    count = 6 if kind == "PV" else 3
    _require(condition=len(selected) == count, message="Target type population differs")
    source = []
    model_paths = []
    for model in selected:
        model_file = model["model_file"]
        _require(
            condition=Path(model_file).name == model_file and model_file.endswith(".ubj"),
            message="Unexpected saved booster path",
        )
        model_path = root / "models" / model_file
        _require(
            condition=str(model_path.relative_to(root)) in completion["files"],
            message="Saved booster lacks completion fingerprint",
        )
        saved = xgb.Booster()
        saved.load_model(model_path)
        shares = _model_shares(
            model=model,
            saved=saved,
            real=real,
            shown=features[kind],
            setting=setting,
            kind=kind,
            seed=seed,
            device=spec["device"],
        )
        source.append(shares)
        model_paths.append(completion["files"][str(model_path.relative_to(root))])
    average = np.mean(np.stack(source), axis=0)
    rows = [
        {
            "recipe": recipe,
            "setting": setting,
            "series_type": kind,
            "fold": fold,
            "seed": seed,
            "feature": feature,
            "mean_share": float(share),
            "series_count": count,
        }
        for feature, share in zip(real, average, strict=True)
    ]
    population = metadata.sort("time_series_id").write_json().encode()
    provenance = {
        "recipe": recipe,
        "setting": setting,
        "fold": fold,
        "seed": seed,
        "series_type": kind,
        "series_count": count,
        "manifest_sha256": run["manifest_sha256"],
        "completion_sha256": run["completion_sha256"],
        "frame_manifest_sha256": run["frame_manifest_sha256"],
        "population_sha256": hashlib.sha256(population).hexdigest(),
        "saved_booster_sha256": sorted(model_paths),
        "verified_training_parameters": PARAMETERS[setting],
        "verified_training_seed": seed,
        "verified_actual_devices": sorted(
            {
                model["booster_runtime_config"]["learner"]["generic_param"]["device"]
                for model in selected
            }
        ),
        "real_features": real,
        "trained_features": features[kind],
        "padded_width": 24,
    }
    return rows, provenance


def _markdown(table: pl.DataFrame) -> str:
    """Generate public numerical Markdown directly from aggregate report rows."""
    lines = [
        "# Requested feature-set training gain",
        "",
        (
            "Post hoc descriptive total training gain, with duplicate-column credit "
            "returned to its original input."
        ),
        (
            "Shares normalize within each per-meter booster, then weight meters "
            "equally within each fold/seed refit."
        ),
        (
            "Means/minima/maxima use six refits separately for each training "
            "setting; ranges are not confidence intervals."
        ),
        (
            "Every retained real input appears, including zero gain. Gain does not "
            "establish held-out feature benefit."
        ),
        "",
        "| " + " | ".join(table.columns) + " |",
        "|" + "---|" * table.width,
    ]
    lines.extend(
        "| "
        + " | ".join(f"{value:.10f}" if isinstance(value, float) else str(value) for value in row)
        + " |"
        for row in table.iter_rows()
    )
    return "\n".join(lines) + "\n"


def _recheck_inventories(spec: dict[str, Any], fingerprints: dict[Path, str]) -> None:
    """Recheck exact source inventories without rereading shared large files repeatedly."""
    seen = set()
    for run in spec["runs"]:
        root, frames = _plain(path=run["path"]), _plain(path=run["frames"])
        for directory, manifest_name, extras in (
            (root, "complete.json", {"manifest.json", "complete.json"}),
            (frames, "manifest.json", {"manifest.json", "context.json"}),
        ):
            if directory in seen:
                continue
            manifest = cast(
                dict[str, Any], _json(path=directory / manifest_name, fingerprints=fingerprints)
            )
            _complete_inventory(
                root=directory, listed=manifest["files"], extras=extras, fingerprints=fingerprints
            )
            seen.add(directory)
    for binding in spec["original_frames"].values():
        root = _plain(path=binding["path"])
        if root not in seen:
            manifest = cast(
                dict[str, Any], _json(path=root / "manifest.json", fingerprints=fingerprints)
            )
            _frame_inventory(root=root, manifest=manifest, fingerprints=fingerprints)
            seen.add(root)


def main() -> None:
    """Write a complete immutable aggregate report from sixty explicitly pinned saved fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fingerprints: dict[Path, str] = {}
    script = Path(__file__).resolve()
    _pin(path=script, expected=_sha(path=script), fingerprints=fingerprints)
    spec_path, output = _plain(path=args.spec), _plain(path=args.output)
    spec = cast(dict[str, Any], _json(path=spec_path, fingerprints=fingerprints))
    _keys(
        value=spec,
        names={
            "preparation_only",
            "fitter",
            "augmentation_builder",
            "original_frames",
            "fitter_dependency_sha256",
            "device",
            "runs",
        },
    )
    _require(
        condition=spec["preparation_only"] is False
        and spec["device"] in {"cuda", "cpu"}
        and len(spec["runs"]) == 60
        and not output.exists()
        and output.parent.is_dir(),
        message="Report is not ready for immutable execution",
    )
    _keys(value=spec["fitter"], names={"path", "sha256"})
    _keys(value=spec["augmentation_builder"], names={"path", "sha256"})
    _keys(value=spec["original_frames"], names=set(FOLDS))
    _pin(
        path=_plain(path=spec["augmentation_builder"]["path"]),
        expected=spec["augmentation_builder"]["sha256"],
        fingerprints=fingerprints,
    )
    _require(
        condition=_plain(path=spec["fitter"]["path"]) == Path(inspect.getfile(fitter)).resolve(),
        message="Imported fitter differs from reviewed pin",
    )
    _pin(
        path=_plain(path=spec["fitter"]["path"]),
        expected=spec["fitter"]["sha256"],
        fingerprints=fingerprints,
    )
    for path, expected in spec["fitter_dependency_sha256"].items():
        _pin(path=_plain(path=path), expected=expected, fingerprints=fingerprints)
    expected_runs = {
        (recipe, setting, fold, seed)
        for recipe in RECIPES
        for setting in SETTINGS
        for fold in FOLDS
        for seed in range(3)
    }
    actual_runs = [
        (run["recipe"], run["setting"], run["fold"], run["seed"]) for run in spec["runs"]
    ]
    _require(
        condition=len(set(actual_runs)) == 60 and set(actual_runs) == expected_runs,
        message="Exact five-recipe/fold/seed/setting matrix differs",
    )
    for run in spec["runs"]:
        for name in ("path", "frames"):
            input_root = _plain(path=run[name])
            _require(
                condition=not output.is_relative_to(input_root)
                and not input_root.is_relative_to(output),
                message="Report output overlaps a source artifact",
            )
    registry = fitter.recipe_registry()
    rows, audits = [], []
    for run in spec["runs"]:
        piece, audit = _read_fit(run=run, spec=spec, registry=registry, fingerprints=fingerprints)
        rows.extend(piece)
        audits.append(audit)
    _require(
        condition=len({audit["population_sha256"] for audit in audits}) == 1,
        message="Source populations differ between fits",
    )
    refits = pl.DataFrame(rows)
    table = (
        refits.group_by("recipe", "setting", "series_type", "feature")
        .agg(
            mean_share=pl.col("mean_share").mean(),
            min_share=pl.col("mean_share").min(),
            max_share=pl.col("mean_share").max(),
            refits=pl.len(),
            min_series_per_refit=pl.col("series_count").min(),
            max_series_per_refit=pl.col("series_count").max(),
        )
        .sort("recipe", "setting", "mean_share", "feature", descending=[False, False, True, False])
    )
    _require(
        condition=set(table.columns) == PUBLIC_COLUMNS
        and table.filter(pl.col("refits") != 6).is_empty(),
        message="Aggregate schema or refit support differs",
    )
    _recheck_inventories(spec=spec, fingerprints=fingerprints)
    for path, expected in fingerprints.items():
        _require(condition=_sha(path=path) == expected, message="Source changed during gain audit")
    output.mkdir(mode=0o700, exist_ok=False)
    for filename, frame in (("importance.parquet", table), ("refit_importance.parquet", refits)):
        path = output / filename
        with path.open("xb") as stream:
            frame.write_parquet(stream)
        path.chmod(mode=0o600)
    for filename, content in (
        ("report.md", _markdown(table=table).encode()),
        ("spec.json", spec_path.read_bytes()),
    ):
        fd = os.open(output / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
    manifest = {
        "complete": True,
        "spec_sha256": fingerprints[spec_path],
        "script_sha256": fingerprints[script],
        "fitter_sha256": spec["fitter"]["sha256"],
        "xgboost_version": xgb.__version__,
        "polars_version": pl.__version__,
        "refits": audits,
        "source_file_hashes_by_path_hash": {
            hashlib.sha256(str(path).encode()).hexdigest(): value
            for path, value in fingerprints.items()
        },
        "statistics": (
            "Padding gains summed with real source before per-booster "
            "normalization; equal target meters, then six fold/seed refits "
            "separately for each setting; min/max are refit variability."
        ),
        "outputs": {
            name: _sha(path=output / name)
            for name in ("importance.parquet", "refit_importance.parquet", "report.md", "spec.json")
        },
    }
    for path, expected in fingerprints.items():
        _require(
            condition=_sha(path=path) == expected,
            message="Source changed before gain report completion",
        )
    _recheck_inventories(spec=spec, fingerprints=fingerprints)
    fd = os.open(output / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(
        "Completed requested feature-set gain report; five recipes; "
        "settings separate; no models fitted"
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        OSError,
        TypeError,
        KeyError,
        RuntimeError,
        AssertionError,
        pl.exceptions.PolarsError,
    ):
        raise SystemExit(
            "Requested gain audit failed; partial private outputs preserved; no automatic retry"
        ) from None
