"""Proposed private reuse and unaffected-prediction audit; never fit or predict a model.

Place the reviewed file in studies/quick_wins before use. Sibling imports intentionally rely
on that normal script directory; this proposal does not modify import paths or load old code.
"""

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, cast

THREADS: Final[tuple[str, ...]] = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
if any(os.environ.get(name) != "8" for name in THREADS):
    raise SystemExit("Private audit requires all seven thread caps to equal eight before import.")

# The caller's caps are validated before numerical imports can initialize thread pools.
import fixed_pipeline_fit as fit  # noqa: E402
import fixed_pipeline_report as report  # noqa: E402
import fixed_pipeline_unaffected as unaffected  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pvlib  # noqa: E402
from contracts.power_schemas import PowerTimeSeries  # noqa: E402
from contracts.weather_schemas import Nwp  # noqa: E402
from pvlib import spa as pvlib_spa  # noqa: E402
from studies.baselines import haurwitz_w_m2  # noqa: E402
from studies.solar import zenith  # noqa: E402

type JsonObject = dict[str, Any]
type RecipeRegistry = dict[str, fit.Recipe]
type FeatureColumns = dict[str, list[str]]
type FrameContext = tuple[Path, JsonObject, set[str], pl.DataFrame, fit.inputs.FoldWindow]
type FitAudit = tuple[Path, JsonObject, JsonObject, FeatureColumns]
type PreparedTraining = tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[int], FeatureColumns]

BASELINE_SHA: Final[str] = "b073be43f8809290746d952bf1acf988ea1f01cf61a7e2ed11e6f51f2de57ca3"
NEW: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "solar_minimal_requested": (
        "PV",
        (
            "temperature_2m",
            "downward_short_wave_radiation_flux_surface",
        ),
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
ORDERED_ADDITIONS: Final[tuple[str, ...]] = (
    "sun_east",
    "sun_north",
    "sun_up",
    "clear_sky_irradiance",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
    "wind_direction_10m_sin",
    "wind_direction_10m_cos",
    "wind_direction_100m_sin",
    "wind_direction_100m_cos",
)
ADDITIONS: Final[frozenset[str]] = frozenset(ORDERED_ADDITIONS)
CONTROLS: Final[frozenset[str]] = frozenset({"production_cleaned", "lean_pv", "lean_wind"})
GUARDS: Final[frozenset[str]] = frozenset(
    {"production_cleaned_guarded_windchill", "lean_wind_guarded_windchill"}
)
AUGMENTED: Final[frozenset[str]] = frozenset(
    {"solar_expanded_requested", "wind_minimal_uv_requested", "wind_minimal_circular_requested"}
)
FIT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "label",
        "root",
        "recipe",
        "seed",
        "frame",
        "manifest_sha256",
        "complete_sha256",
        "script_sha256",
    }
)
SPEC_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "phase",
        "scope",
        "baseline",
        "sources",
        "frames",
        "augmentation_proof",
        "fits",
        "pairs",
    }
)


def _require(ok: bool) -> None:
    """Reject a failed invariant without disclosing private values."""
    if not ok:
        raise ValueError("Private audit check failed")


def _fields(value: Any, names: set[str] | frozenset[str]) -> None:
    """Require exactly the closed field inventory."""
    _require(isinstance(value, dict) and set(value) == names)


def _digest(value: Any) -> None:
    """Require a lowercase SHA256 binding."""
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _plain(path: Path) -> None:
    """Reject relative, traversing or linked paths."""
    _require(
        path.is_absolute()
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents))
    )


class Pins:
    """Retain private paths in memory only; bind all inputs before and after each phase."""

    def __init__(self) -> None:
        """Create an empty immutable input binding collection."""
        self.values: dict[Path, str] = {}

    def pin(self, path: Path, expected: str) -> None:
        """Bind a canonical file and reject changes against any existing binding."""
        _plain(path)
        _digest(expected)
        _require(path.is_file() and fit._sha256(path=path) == expected)
        _require(path not in self.values or self.values[path] == expected)
        self.values[path] = expected

    def record(self, value: Any) -> Path:
        """Validate and pin a closed path/hash record."""
        _fields(value=value, names={"path", "sha256"})
        path = Path(value["path"])
        self.pin(path=path, expected=value["sha256"])
        return path

    def finish(self) -> None:
        """Recheck every file bound during the phase."""
        for path, expected in self.values.items():
            self.pin(path=path, expected=expected)


def _validate(spec: Any) -> None:
    """Enforce the fixed recipes, seeds, scopes and pairing inventory."""
    _fields(value=spec, names=SPEC_FIELDS)
    _require(spec["schema_version"] == 1 and type(spec["schema_version"]) is int)
    _require(spec["phase"] in {"reuse_preflight", "prediction_audit"})
    _fields(value=spec["scope"], names={"fold", "setting", "padding"})
    scope = spec["scope"]
    _require(scope["fold"] in unaffected.FOLDS)
    _require(scope["setting"] in {"main", "sensitivity"})
    _require(scope["padding"] in {"equal", "none"})
    _require(not (scope["setting"] == "sensitivity" and scope["padding"] == "none"))
    _fields(
        value=spec["sources"],
        names={"audit", "fitter", "reporter", "unaffected_audit", "augmentation_builder"},
    )
    _fields(value=spec["frames"], names={"original", "augmented"})
    for record in spec["frames"].values():
        _fields(value=record, names={"root", "manifest_sha256"})
        _plain(Path(record["root"]))
        _digest(record["manifest_sha256"])
    _require(isinstance(spec["fits"], list) and bool(spec["fits"]))
    labels = set()
    for entry in spec["fits"]:
        _fields(value=entry, names=FIT_FIELDS)
        _require(
            isinstance(entry["label"], str)
            and re.fullmatch(r"[a-z0-9_]+", entry["label"]) is not None
        )
        _require(entry["label"] not in labels)
        labels.add(entry["label"])
        _require(entry["recipe"] in CONTROLS | GUARDS | set(NEW))
        _require(type(entry["seed"]) is int and entry["seed"] in (0, 1, 2))
        _require(entry["frame"] in spec["frames"])
        _require(entry["frame"] == ("augmented" if entry["recipe"] in AUGMENTED else "original"))
        _plain(Path(entry["root"]))
        for name in ("manifest_sha256", "complete_sha256", "script_sha256"):
            _digest(entry[name])
    _require(len({(e["recipe"], e["seed"]) for e in spec["fits"]}) == len(spec["fits"]))
    required_recipes = CONTROLS | (GUARDS if scope["padding"] == "equal" else set())
    if spec["phase"] == "reuse_preflight":
        _require(
            {(e["recipe"], e["seed"]) for e in spec["fits"]}
            == {(r, seed) for r in required_recipes for seed in (0, 1, 2)}
        )
        _require(spec["pairs"] == [])
    else:
        _require(
            {(e["recipe"], e["seed"]) for e in spec["fits"]}
            == {(r, seed) for r in set(NEW) | {"production_cleaned"} for seed in (0, 1, 2)}
        )
        _require(isinstance(spec["pairs"], list) and len(spec["pairs"]) == 15)
        by_label = {e["label"]: e for e in spec["fits"]}
        candidates = set()
        for pair in spec["pairs"]:
            _fields(value=pair, names={"candidate", "reference", "target_type"})
            _require(pair["candidate"] in by_label and pair["reference"] in by_label)
            candidate, reference = by_label[pair["candidate"]], by_label[pair["reference"]]
            _require(candidate["recipe"] in NEW and reference["recipe"] == "production_cleaned")
            _require(candidate["seed"] == reference["seed"])
            _require(pair["target_type"] == NEW[candidate["recipe"]][0])
            _require(pair["candidate"] not in candidates)
            candidates.add(pair["candidate"])
        _require(len(candidates) == 15)


def _verify_sources(spec: JsonObject, pins: Pins) -> RecipeRegistry:
    """Permit only the exact five registry additions to the pinned historical fitter AST."""
    paths = {role: pins.record(record) for role, record in spec["sources"].items()}
    _require(paths["audit"].resolve() == Path(__file__).resolve())
    _require(paths["fitter"].resolve() == Path(fit.__file__).resolve())
    _require(paths["reporter"].resolve() == Path(report.__file__).resolve())
    _require(paths["unaffected_audit"].resolve() == Path(unaffected.__file__).resolve())
    baseline = pins.record(spec["baseline"])
    _require(spec["baseline"]["sha256"] == BASELINE_SHA)
    old, current = [ast.parse(p.read_text()) for p in (baseline, paths["fitter"])]
    registry_node = next(
        n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == "recipe_registry"
    )
    additions = []
    for statement in registry_node.body:
        if not isinstance(statement, ast.For) or not isinstance(statement.iter, ast.Tuple):
            continue
        try:
            rows = ast.literal_eval(statement.iter)
        except ValueError, TypeError:
            continue
        if (
            isinstance(rows, tuple)
            and len(rows) == 5
            and all(isinstance(row, tuple) and len(row) == 3 and row[1] in NEW for row in rows)
        ):
            additions.append((statement, rows))
    _require(len(additions) == 1)
    addition, rows = additions[0]
    _require({name: (kind, tuple(columns)) for kind, name, columns in rows} == NEW)
    _require(len(rows) == len({row[1] for row in rows}))
    expected_loop = ast.parse(
        "for kind, name, columns in " + repr(rows) + ":\n"
        "    registry[name] = Recipe(replace(base, name=name, "
        'description=f"Post hoc requested {name} inputs; other series keep full inputs.", '
        "columns=_changed_columns(series_type=kind, columns=tuple(sorted(columns)))))\n"
    ).body[0]
    _require(
        ast.dump(addition, include_attributes=False)
        == ast.dump(expected_loop, include_attributes=False)
    )
    # Strip only the five fixed additions; all original module statements/functions must match.
    registry_node.body.remove(addition)
    _require(ast.dump(old, include_attributes=False) == ast.dump(current, include_attributes=False))
    registry = fit.recipe_registry()
    base = registry["production_cleaned"].arm
    for name, (target, columns) in NEW.items():
        recipe = registry[name]
        _require(not recipe.guarded_windchill)
        _require(not recipe.arm.early_stopping and not recipe.arm.pooled_by_type)
        _require(not recipe.arm.pooled_site_id and not recipe.arm.quantile_levels)
        for attr in (
            "cleaning",
            "extra_cleaning",
            "objective",
            "colsample_bytree",
            "min_child_weight",
            "recency_half_life_days",
            "train_last_months",
        ):
            _require(getattr(recipe.arm, attr) == getattr(base, attr))
        for kind in unaffected.TYPES:
            expected = tuple(sorted(columns)) if kind == target else fit.arms.PRODUCTION_COLUMNS
            _require(recipe.arm.columns(kind) == expected)
    for kind in unaffected.TYPES:
        _require(
            max(len(r.arm.columns(kind)) for n, r in registry.items() if n not in fit.EXACT_RECIPES)
            == 24
        )
    return registry


def _bit_equal(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Compare ordered schemas, null masks and nonnull floating bit patterns."""
    _require(left.schema == right.schema and left.shape == right.shape)
    for name, dtype in left.schema.items():
        a, b = left[name], right[name]
        _require(a.is_null().equals(b.is_null()))
        if dtype in (pl.Float32, pl.Float64):
            valid = a.is_not_null()
            av, bv = a.filter(valid).to_numpy(), b.filter(valid).to_numpy()
            _require(av.dtype == bv.dtype and av.tobytes() == bv.tobytes())
        else:
            _require(a.equals(b))


def _frame_context(record: JsonObject, scope: JsonObject, pins: Pins) -> FrameContext:
    """Pin frozen exports while preserving historical engineering evidence as archived data."""
    root = Path(record["root"])
    pins.pin(path=root / "manifest.json", expected=record["manifest_sha256"])
    manifest, sha, files, metadata = unaffected._frame_context(root=root, fold=scope["fold"])
    _require(sha == record["manifest_sha256"])
    _require("manifest.json" not in manifest["files"])
    context = manifest["context"]
    _require(
        (
            context["archive_version"],
            context["power_version"],
            context["capacity_version"],
            context["population_version"],
        )
        == (924, 70, 1, 1)
    )
    _require(context["study_end_exclusive"] == fit.STUDY_END.isoformat())
    _require(manifest["window"] == context["window"])
    _require(
        manifest["input_fingerprint"]
        == hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    )
    _require(isinstance(context["python"], str) and bool(context["python"]))
    _require(isinstance(context["dependencies"], dict) and bool(context["dependencies"]))
    _require(isinstance(context["code_sha256"], dict) and bool(context["code_sha256"]))
    # Historical source locations are evidence, not dependencies consumed by this audit.
    # Current fitter dependencies and augmentation geometry receive live checks separately.
    for path, source_sha in context["code_sha256"].items():
        _require(Path(path).is_absolute())
        _digest(source_sha)
    context_path = root / "context.json"
    pins.pin(path=context_path, expected=fit._sha256(path=context_path))
    _require(json.loads(context_path.read_text()) == context)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    _require(actual == set(manifest["files"]) | {"manifest.json", "context.json"})
    for name, sha in manifest["files"].items():
        pins.pin(path=root / name, expected=sha)
    _, window = fit._load_context(root=root)
    _require(window.fold_id == unaffected.FOLDS[scope["fold"]])
    return root, manifest, files, metadata, window


def _verify_augmentation(spec: JsonObject, contexts: dict[str, FrameContext], pins: Pins) -> None:
    """Independently bind additive provenance, semantic evidence and ordered original values."""
    a, b = contexts["original"], contexts["augmented"]
    _require(a[2] == b[2] and a[1]["window"] == b[1]["window"])
    context = b[1]["context"]
    _require(all(context.get(name) == value for name, value in a[1]["context"].items()))
    lineage = b[1]["parent_provenance"]
    _require(lineage["historical_parent_manifest"] == a[1])
    _require(lineage["manifest_sha256"] == spec["frames"]["original"]["manifest_sha256"])
    _require(context["parent_manifest_sha256"] == lineage["manifest_sha256"])
    _require(context["parent_input_fingerprint"] == a[1]["input_fingerprint"])
    _require(context["parent_context_sha256"] == pins.values[a[0] / "context.json"])
    _require(context["new_features"] == list(ORDERED_ADDITIONS))
    _verify_augmentation_runtime(context=context, root=b[0], spec=spec, pins=pins)
    semantic = json.loads((b[0] / "augmentation-audit.json").read_text())
    _require(semantic["original_columns_rows_keys_targets_preserved"] is True)
    _require(semantic["support_files_byte_identical"] is True)
    support_flags = {
        "support_schemas",
        "support_population",
        "positive_finite_capacity",
        "bounded_power_unique_sorted_finite_nonnull_before_study_end",
        "early_history_preserved",
        "capacity_future_history_is_scoring_only",
    }
    _fields(value=semantic["support_semantic_proof"], names=support_flags)
    _require(all(value is True for value in semantic["support_semantic_proof"].values()))
    for role in ("source_frame_semantic_proof", "augmented_frame_semantic_proof"):
        evidence = semantic[role]
        _fields(value=evidence, names={"train.parquet", *a[2]})
        for name, record in evidence.items():
            _fields(
                value=record,
                names={
                    "rows",
                    "schema_keys_members_sorted_issuance_lags_targets_passed",
                    "nullable_targets_preserved",
                },
            )
            _require(record["rows"] == a[1]["frames"][name]["rows"])
            _require(record["schema_keys_members_sorted_issuance_lags_targets_passed"] is True)
            _require(record["nullable_targets_preserved"] is True)
    proof_path = pins.record(spec["augmentation_proof"])
    _require(not any(proof_path.is_relative_to(side[0]) for side in (a, b)))
    proof = json.loads(proof_path.read_text())
    _fields(
        value=proof,
        names={
            "schema_version",
            "status",
            "original_manifest_sha256",
            "augmented_manifest_sha256",
            "original_columns_bit_exact",
            "copied_artifacts_byte_exact",
            "builder_sha256",
        },
    )
    _require(
        type(proof["schema_version"]) is int
        and proof["schema_version"] == 1
        and proof["status"] == "complete"
    )
    _require(proof["original_manifest_sha256"] == spec["frames"]["original"]["manifest_sha256"])
    _require(proof["augmented_manifest_sha256"] == spec["frames"]["augmented"]["manifest_sha256"])
    _require(
        proof["original_columns_bit_exact"] is True and proof["copied_artifacts_byte_exact"] is True
    )
    _require(proof["builder_sha256"] == spec["sources"]["augmentation_builder"]["sha256"])
    for name in ("metadata.parquet", "capacity.parquet", "bounded_power.parquet"):
        _require(a[1]["files"][name] == b[1]["files"][name])
    for name in ["train.parquet", *sorted(a[2])]:
        original, augmented = pl.read_parquet(a[0] / name), pl.read_parquet(b[0] / name)
        _require(augmented.columns == original.columns + list(ORDERED_ADDITIONS))
        _require(not set(original.columns).intersection(ADDITIONS))
        _bit_equal(left=original, right=augmented.select(original.columns))
        _require(all(augmented.schema[c] == pl.Float32 for c in ADDITIONS))
        training = name == "train.parquet"
        fit._check_frame(
            frame=original,
            start=a[4].train_start if training else a[4].val_start,
            end=a[4].train_end if training else a[4].val_end,
            training=training,
        )
        fit._check_frame(
            frame=augmented,
            start=b[4].train_start if training else b[4].val_start,
            end=b[4].train_end if training else b[4].val_end,
            training=training,
        )


def _verify_augmentation_runtime(
    *, context: JsonObject, root: Path, spec: JsonObject, pins: Pins
) -> None:
    """Check approved and actual consumed geometry sources without importing builder code."""
    runtime_path = root / "runtime-spec.json"
    _require("runtime-spec.json" in json.loads((root / "manifest.json").read_text())["files"])
    pins.pin(path=runtime_path, expected=context["approved_runtime_spec_sha256"])
    approved = json.loads(runtime_path.read_text())
    _fields(value=approved, names={"versions", "sources", "geometry_defaults"})
    _fields(value=context["approved_and_actual_runtime"], names={"approved", "actual"})
    _require(context["approved_and_actual_runtime"] == {"approved": approved, "actual": approved})
    version_names = {"numpy", "polars", "pyarrow", "pvlib", "pandas", "patito"}
    _fields(value=approved["versions"], names=version_names)
    _require(
        all(
            importlib.metadata.version(name) == value
            for name, value in approved["versions"].items()
        )
    )
    _require(context["augmentation_python"] == sys.version)
    dependency_names = version_names - {"patito"}
    _fields(value=context["augmentation_dependencies"], names=dependency_names)
    _require(
        context["augmentation_dependencies"]
        == {name: approved["versions"][name] for name in dependency_names}
    )
    _require(context["thread_caps"] == dict.fromkeys(THREADS, "8"))
    actual_paths = {
        "solar_wrapper": Path(inspect.getfile(zenith)).resolve(),
        "baselines_haurwitz": Path(inspect.getfile(haurwitz_w_m2)).resolve(),
        "pvlib_solarposition": Path(inspect.getfile(pvlib.solarposition)).resolve(),
        "pvlib_spa": Path(inspect.getfile(pvlib_spa)).resolve(),
        "weather_contract": Path(inspect.getfile(Nwp)).resolve(),
        "power_contract": Path(inspect.getfile(PowerTimeSeries)).resolve(),
    }
    _fields(value=approved["sources"], names=set(actual_paths))
    _fields(value=context["augmentation_code_sha256"], names={"builder", *actual_paths})
    _require(
        context["augmentation_code_sha256"]["builder"]
        == spec["sources"]["augmentation_builder"]["sha256"]
    )
    for role, path in actual_paths.items():
        binding = approved["sources"][role]
        _require(pins.record(binding) == path)
        _require(binding["sha256"] == context["augmentation_code_sha256"][role])
    _require(approved["versions"]["pvlib"] == context["dependencies"]["pvlib"])
    _require(
        context["code_sha256"].get(str(actual_paths["solar_wrapper"]))
        == approved["sources"]["solar_wrapper"]["sha256"]
    )
    _require(
        approved["geometry_defaults"]
        == {
            "method": "nrel_numpy",
            "altitude_m": 0.0,
            "pressure_pa": 101325.0,
            "temperature_c": 12.0,
        }
    )
    parameters = inspect.signature(pvlib.solarposition.get_solarposition).parameters
    _require(
        parameters["method"].default == "nrel_numpy"
        and parameters["temperature"].default == 12.0
        and parameters["altitude"].default is None
        and parameters["pressure"].default is None
    )


def _current_dependencies() -> dict[str, str]:
    """Hash only the implementation dependencies consumed by current fitting preparation."""
    paths = (
        Path(fit.arms.__file__),
        Path(fit.fit_arms.__file__),
        Path(fit.cleaning_rules.__file__),
        Path(fit.inputs.__file__),
        Path(fit.score_module.__file__),
        Path(fit.metrics_module.__file__),
        Path(fit.validation_module.__file__),
        Path(inspect.getfile(fit.XGBoostForecaster)),
        Path(inspect.getfile(fit.AllFeatures)),
        fit.inputs.PRODUCTION_MODEL_CONFIG,
    )
    return {str(p): fit._sha256(path=p) for p in paths}


def _verify_model_parameters(
    root: Path, manifest: JsonObject, metadata: pl.DataFrame, features: FeatureColumns
) -> None:
    """Check every immutable per-series fitted configuration at serialized precision."""
    # Read the report's parameter evidence, then verify every stored per-series configuration.
    report._training_parameters(root=root, manifest=manifest)
    rows = json.loads((root / "models.json").read_text())
    kinds = dict(metadata.select("time_series_id", "time_series_type").iter_rows())
    _require(len(rows) == len(kinds) and {r["time_series_id"] for r in rows} == set(kinds))
    recipe = fit.recipe_registry()[manifest["recipe"]]
    expected = fit.fit_arms._parameters(arm=recipe.arm, device="cuda", seed=manifest["seed"])
    rounds = fit.fit_arms.N_TREES
    if manifest["setting"] == "sensitivity":
        extra = dict(fit.SENSITIVITY_HYPER_PARAMETERS)
        rounds = int(cast(int, extra.pop("num_boost_round")))
        expected.update(extra)
        expected["eta"] = expected.pop("learning_rate")
    aliases = {"reg_lambda": "lambda"}
    for row in rows:
        _require(row["scale"] == 1 and row["rounds"] == rounds)
        _require(row["features"] == features[kinds[row["time_series_id"]]])
        config = row.get("booster_runtime_config") or row.get("effective_xgboost_config")
        learner = config["learner"]
        tree, generic = learner["gradient_booster"]["tree_train_param"], learner["generic_param"]
        _require(learner["learner_train_param"]["objective"] == expected["objective"])
        _require(int(generic["seed"]) == manifest["seed"] and int(generic["nthread"]) == 8)
        _require(generic["device"] in {"cuda", "cuda:0"})
        _require(int(learner["learner_model_param"]["num_feature"]) == len(row["features"]))
        for name, value in expected.items():
            if name not in {"device", "objective", "seed"}:
                _require(
                    np.float32(tree[aliases.get(name, name)]).tobytes()
                    == np.float32(cast(int | float, value)).tobytes()
                )


def _audit_fit(
    entry: JsonObject, spec: JsonObject, contexts: dict[str, FrameContext], pins: Pins
) -> FitAudit:
    """Bind each completed fit to its exact frame, configuration and consumed dependencies."""
    root = Path(entry["root"])
    context = contexts[entry["frame"]]
    pins.pin(path=root / "manifest.json", expected=entry["manifest_sha256"])
    pins.pin(path=root / "complete.json", expected=entry["complete_sha256"])
    manifest = json.loads((root / "manifest.json").read_text())
    completion = json.loads((root / "complete.json").read_text())
    scope = spec["scope"]
    unaffected._audit_fit(
        root=root,
        recipe=entry["recipe"],
        seed=entry["seed"],
        setting=scope["setting"],
        padding=scope["padding"],
        frame_manifest=context[1],
        frame_sha=spec["frames"][entry["frame"]]["manifest_sha256"],
        expected=context[2],
    )
    _require(manifest["script_sha256"] == entry["script_sha256"])
    if entry["recipe"] in NEW:
        _require(entry["script_sha256"] == spec["sources"]["fitter"]["sha256"])
    elif spec["phase"] == "reuse_preflight":
        _require(entry["script_sha256"] == BASELINE_SHA)
    else:
        _require(entry["script_sha256"] in {BASELINE_SHA, spec["sources"]["fitter"]["sha256"]})
    _require(manifest["dependency_sha256"] == _current_dependencies())
    _require(
        manifest["xgboost_version"] == fit.xgb.__version__
        and manifest["polars_version"] == pl.__version__
    )
    _require(manifest["sensitivity_hyperparameters"] == dict(fit.SENSITIVITY_HYPER_PARAMETERS))
    for path, sha in manifest["dependency_sha256"].items():
        pins.pin(path=Path(path), expected=sha)
    for name, sha in completion["files"].items():
        pins.pin(path=root / name, expected=sha)
    features = json.loads((root / "features.json").read_text())
    _verify_model_parameters(root=root, manifest=manifest, metadata=context[3], features=features)
    return root, manifest, completion, features


def _prepared(
    root: Path,
    context: FrameContext,
    recipe: Any,
    registry: RecipeRegistry,
    padding: str,
    scratch: Path,
) -> PreparedTraining:
    """Prepare deterministic historical cleaning and matrices without fitting or predicting."""
    scratch.mkdir(mode=0o700)
    args = argparse.Namespace(frames=root, padding=padding)
    with (
        (scratch / "private-helper.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        return fit._load_training(
            args=args, window=context[4], recipe=recipe, registry=registry, output=scratch
        )


def _compare_selected(
    a: pl.DataFrame, b: pl.DataFrame, columns: FeatureColumns, target: str | None = None
) -> int:
    """Compare selected ordered keys, targets and actual Float32 model matrices."""
    kinds = [target] if target else sorted(columns)
    count = 0
    for kind in kinds:
        left = a.filter(pl.col("time_series_type") == kind)
        right = b.filter(pl.col("time_series_type") == kind)
        _require(not left.is_empty())
        _bit_equal(
            left=left.select(*fit.SAMPLE_KEYS, "power")
            if "power" in left.columns
            else left.select(*fit.SAMPLE_KEYS),
            right=right.select(*fit.SAMPLE_KEYS, "power")
            if "power" in right.columns
            else right.select(*fit.SAMPLE_KEYS),
        )
        _bit_equal(
            left=fit.fit_arms.feature_matrix(frame=left, columns=columns[kind]),
            right=fit.fit_arms.feature_matrix(frame=right, columns=columns[kind]),
        )
        count += left.height
    return count


def _validation_prepared(
    context: FrameContext,
    name: str,
    recipe: Any,
    registry: RecipeRegistry,
    padding: fit.PaddingType,
) -> tuple[pl.DataFrame, FeatureColumns]:
    """Prepare one immutable validation run under the selected recipe and padding."""
    frame = pl.read_parquet(context[0] / name).join(
        context[3], on="time_series_id", validate="m:1", maintain_order="left"
    )
    return fit._prepare(frame=frame, recipe=recipe, registry=registry, padding=padding)


def _reuse_preflight(
    spec: JsonObject,
    contexts: dict[str, FrameContext],
    audited: dict[str, FitAudit],
    registry: RecipeRegistry,
    output: Path,
) -> list[JsonObject]:
    """Compare complete historical control preparation on original and augmented inputs."""
    summaries = []
    for index, entry in enumerate(spec["fits"]):
        recipe = registry[entry["recipe"]]
        sides = [
            _prepared(
                root=c[0],
                context=c,
                recipe=recipe,
                registry=registry,
                padding=spec["scope"]["padding"],
                scratch=output / "scratch" / f"control_{index}_{side}",
            )
            for side, c in contexts.items()
        ]
        original, augmented = sides
        _require(original[4] == augmented[4] == audited[entry["label"]][3])
        _compare_selected(a=original[0], b=augmented[0], columns=original[4])
        root = audited[entry["label"]][0]
        counts_path = output / "scratch" / f"control_{index}_original" / "training_counts.parquet"
        current_counts = pl.read_parquet(counts_path)
        _bit_equal(
            left=current_counts.sort("time_series_id"),
            right=pl.read_parquet(root / "training_counts.parquet").sort("time_series_id"),
        )
        for name in sorted(contexts["original"][2]):
            left, cols = _validation_prepared(
                context=contexts["original"],
                name=name,
                recipe=recipe,
                registry=registry,
                padding=spec["scope"]["padding"],
            )
            right, other = _validation_prepared(
                context=contexts["augmented"],
                name=name,
                recipe=recipe,
                registry=registry,
                padding=spec["scope"]["padding"],
            )
            _require(cols == other == original[4])
            _compare_selected(a=left, b=right, columns=cols)
        summaries.append(
            {
                "recipe": entry["recipe"],
                "seed": entry["seed"],
                "original_vs_augmented_preparation_bit_exact": True,
                "saved_feature_order_and_training_counts_match": True,
                "historical_reuse_basis": (
                    "pinned original rows and unchanged cleaning/preparation code"
                ),
            }
        )
    return summaries


def _guard_inputs(
    spec: JsonObject, contexts: dict[str, FrameContext], registry: RecipeRegistry, output: Path
) -> list[JsonObject]:
    """Check target-input invariance without claiming separately generated guarded forecasts."""
    summaries = []
    padding = spec["scope"]["padding"]
    for index, (name, (target, _)) in enumerate(NEW.items()):
        context = contexts["augmented" if name in AUGMENTED else "original"]
        recipe = registry[name]
        guarded = replace(recipe, guarded_windchill=True)
        left = _prepared(
            root=context[0],
            context=context,
            recipe=recipe,
            registry=registry,
            padding=padding,
            scratch=output / "scratch" / f"new_{index}_ordinary",
        )
        right = _prepared(
            root=context[0],
            context=context,
            recipe=guarded,
            registry=registry,
            padding=padding,
            scratch=output / "scratch" / f"new_{index}_guarded",
        )
        _require(left[4] == right[4] and "windchill" not in left[4][target])
        _compare_selected(a=left[0], b=right[0], columns=left[4], target=target)
        for filename in sorted(context[2]):
            a, cols = _validation_prepared(
                context=context, name=filename, recipe=recipe, registry=registry, padding=padding
            )
            b, other = _validation_prepared(
                context=context, name=filename, recipe=guarded, registry=registry, padding=padding
            )
            _require(cols == other == left[4])
            _compare_selected(a=a, b=b, columns=cols, target=target)
        summaries.append(
            {
                "recipe": name,
                "target_type": target,
                "ordinary_vs_guarded_target_prepared_inputs_bit_exact": True,
                "separate_guarded_forecast_generated": False,
            }
        )
    return summaries


def _prediction_audit(
    spec: JsonObject,
    contexts: dict[str, FrameContext],
    audited: dict[str, FitAudit],
    registry: RecipeRegistry,
    output: Path,
) -> list[JsonObject]:
    """Compare every untargeted completed forecast against its explicit full reference."""
    entries = {e["label"]: e for e in spec["fits"]}
    summaries = []
    for index, pair in enumerate(spec["pairs"]):
        candidate, reference = [entries[pair[n]] for n in ("candidate", "reference")]
        context = contexts[candidate["frame"]]
        reference_context = contexts[reference["frame"]]
        for entry, side in ((candidate, "candidate"), (reference, "reference")):
            c = contexts[entry["frame"]]
            loaded = _prepared(
                root=c[0],
                context=c,
                recipe=registry[entry["recipe"]],
                registry=registry,
                padding=spec["scope"]["padding"],
                scratch=output / "scratch" / f"pair_{index}_{side}",
            )
            _require(loaded[4] == audited[entry["label"]][3])
            current_path = output / "scratch" / f"pair_{index}_{side}" / "training_counts.parquet"
            saved_path = audited[entry["label"]][0] / "training_counts.parquet"
            _bit_equal(
                left=pl.read_parquet(current_path).sort("time_series_id"),
                right=pl.read_parquet(saved_path).sort("time_series_id"),
            )
        count = 0
        ids = (
            context[3]
            .filter(pl.col("time_series_type") != pair["target_type"])
            .select("time_series_id")
        )
        for name in sorted(context[2]):
            raw = pl.read_parquet(context[0] / name, columns=unaffected.KEYS)
            other_raw = pl.read_parquet(reference_context[0] / name, columns=unaffected.KEYS)
            _bit_equal(left=raw, right=other_raw)
            expected = raw.join(ids, on="time_series_id", how="semi").sort(unaffected.KEYS)
            selected = [
                unaffected._selected_predictions(
                    path=audited[entry["label"]][0] / "forecasts" / Path(name).name,
                    ids=ids,
                    expected=expected,
                    all_keys=raw.sort(unaffected.KEYS),
                )
                for entry in (candidate, reference)
            ]
            _bit_equal(left=selected[0], right=selected[1])
            count += selected[0].height
        summaries.append(
            {
                "recipe": candidate["recipe"],
                "reference": reference["recipe"],
                "seed": candidate["seed"],
                "excluded_target_type": pair["target_type"],
                "forecast_files": len(context[2]),
                "compared_unaffected_rows": count,
                "unaffected_predictions_bit_exact": True,
                "guard_evidence": (
                    "prepared-input invariance plus immutable model/prediction-source "
                    "bindings; no second forecast artifact"
                ),
            }
        )
    return summaries


def _write_json(path: Path, value: Any) -> None:
    """Write a private JSON artifact exclusively without replacing existing bytes."""
    descriptor = os.open(path=path, flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o600)
    with os.fdopen(fd=descriptor, mode="w") as stream:
        json.dump(obj=value, fp=stream, indent=2)
        stream.write("\n")


def _run(spec_path: Path, output: Path) -> None:
    """Run one complete private audit phase and finalize only after input rechecks."""
    _require(all(os.environ.get(name) == "8" for name in THREADS))
    os.umask(0o077)
    _plain(spec_path)
    pins = Pins()
    pins.pin(path=spec_path, expected=fit._sha256(path=spec_path))
    spec = json.loads(spec_path.read_text())
    _validate(spec)
    _plain(output)
    for value in [*spec["frames"].values(), *spec["fits"]]:
        root = Path(value["root"])
        _require(not output.is_relative_to(root) and not root.is_relative_to(output))
    _require(not output.exists())
    output.mkdir(mode=0o700)
    try:
        (output / "scratch").mkdir(mode=0o700)
        registry = _verify_sources(spec=spec, pins=pins)
        contexts = {
            name: _frame_context(record=spec["frames"][name], scope=spec["scope"], pins=pins)
            for name in ("original", "augmented")
        }
        _verify_augmentation(spec=spec, contexts=contexts, pins=pins)
        audited = {
            e["label"]: _audit_fit(entry=e, spec=spec, contexts=contexts, pins=pins)
            for e in spec["fits"]
        }
        guards = _guard_inputs(spec=spec, contexts=contexts, registry=registry, output=output)
        checks = (
            _reuse_preflight(
                spec=spec, contexts=contexts, audited=audited, registry=registry, output=output
            )
            if spec["phase"] == "reuse_preflight"
            else _prediction_audit(
                spec=spec, contexts=contexts, audited=audited, registry=registry, output=output
            )
        )
        # Reject additions/removals as well as changed bytes during the audit.
        for record in spec["frames"].values():
            _frame_context(record=record, scope=spec["scope"], pins=pins)
        for entry in spec["fits"]:
            _audit_fit(entry=entry, spec=spec, contexts=contexts, pins=pins)
        pins.finish()
        _require(all(os.environ.get(name) == "8" for name in THREADS))
        result = {
            "schema_version": 1,
            "status": "complete",
            "phase": spec["phase"],
            "scope": spec["scope"],
            "spec_sha256": pins.values[spec_path],
            "audit_sha256": spec["sources"]["audit"]["sha256"],
            "baseline_sha256": BASELINE_SHA,
            "thread_caps": {name: os.environ[name] for name in THREADS},
            "historical_provenance_policy": (
                "archived parent context/source hashes retained; "
                "unused live historical locations are not reuse gates"
            ),
            "checks": checks,
            "guard_inputs": guards,
            "guard_forecast_statement": (
                "input/model/function binding argument only; "
                "not equality of separately generated guarded artifacts"
            ),
        }
        _write_json(path=output / "proof.json", value=result)
        descriptor = os.open(
            path=output / "report.md", flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o600
        )
        with os.fdopen(fd=descriptor, mode="w") as stream:
            stream.write(
                "# Requested-feature audit\n\nCompleted the specified private audit phase.\n\n"
            )
            stream.write(f"Phase: {spec['phase']}. Aggregate comparisons: {len(checks)}.\n\n")
            stream.write(
                "Guard evidence is prepared-input invariance with immutable bindings, "
                "not a second forecast artifact.\n"
            )
        pins.finish()
        _write_json(
            path=output / "complete.json",
            value={
                "status": "complete",
                "files": {
                    name: fit._sha256(path=output / name) for name in ("proof.json", "report.md")
                },
            },
        )
    except Exception:
        _write_json(
            path=output / "failure.json",
            value={"status": "failed", "phase": spec["phase"], "private_scratch_preserved": True},
        )
        raise


def main() -> None:
    """Parse an explicit private scope and suppress private details on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        _run(spec_path=args.spec.absolute(), output=args.output.absolute())
    except Exception:  # noqa: BLE001 — library exceptions may contain private keys or values.
        raise SystemExit("Private audit failed; no row details emitted.") from None


if __name__ == "__main__":
    main()
