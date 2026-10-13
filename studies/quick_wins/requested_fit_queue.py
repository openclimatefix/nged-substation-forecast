"""Run only the reviewed fixed 90-CUDA-plus-one-CPU requested-feature inventory.

Private orchestration proposal. It reads completed preflight evidence and immutable inputs,
then runs the existing fitter in the foreground. It does not import fitting or audit modules.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

JsonObject = dict[str, Any]
ProcessEnvironment = dict[str, str]
FoldType = Literal["fold_a", "fold_b"]
SettingType = Literal["main", "sensitivity"]
PaddingType = Literal["equal", "none"]
DeviceType = Literal["cuda", "cpu"]


class Job(TypedDict):
    """One fixed fitter invocation, including its explicit frame and output bindings."""

    label: str
    fold: FoldType
    recipe: str
    setting: SettingType
    padding: PaddingType
    seed: int
    device: DeviceType
    frame: str
    output: str
    role: Literal["requested_feature_fit", "numerical_diagnostic"]


RECIPES: Final[tuple[str, ...]] = (
    "solar_minimal_requested",
    "solar_expanded_requested",
    "wind_minimal_requested",
    "wind_minimal_uv_requested",
    "wind_minimal_circular_requested",
)
AUGMENTED: Final[frozenset[str]] = frozenset({RECIPES[1], RECIPES[3], RECIPES[4]})
THREADS: Final[tuple[str, ...]] = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
FITTER_SHA: Final[str] = "96249cf95590d045764157fc49dea2fb2c1993ff83faa0a9262ebb57037b96ac"
BUILDER_SHA: Final[str] = "d371d635662b6e36e44869b656e2805c391b659b652691787b8be77b3d334c87"
AUDIT_SHA: Final[str] = "3134fb3549cf6602af15e4210a22d4526fb07e6e6e16fe5b32bb2c868dc3048d"
TOP_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "state",
        "repository_root",
        "data_root",
        "extension_root",
        "log_root",
        "python",
        "queue",
        "fitter",
        "augmentation_builder",
        "audit",
        "source_files",
        "runtime",
        "environment",
        "frames",
        "augmentation_proofs",
        "reuse_preflights",
        "original_gpu_completion",
        "failure_markers",
        "jobs",
    }
)


def _require(*, condition: bool) -> None:
    if not condition:
        raise ValueError("Private queue binding or state check failed")


def _fields(*, value: Any, names: set[str] | frozenset[str]) -> None:
    _require(condition=isinstance(value, dict) and set(value) == names)


def _path(*, value: str) -> Path:
    path = Path(value)
    _require(condition=path.is_absolute() and ".." not in path.parts)
    _require(condition=not any(p.is_symlink() for p in (path, *path.parents)))
    return path


def _sha(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(mode="rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(*, record: JsonObject) -> Path:
    _fields(value=record, names={"path", "sha256"})
    _require(
        condition=isinstance(record["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
    )
    path = _path(value=record["path"])
    _require(condition=path.is_file() and _sha(path=path) == record["sha256"])
    return path


def _json(*, path: Path) -> JsonObject:
    value = json.loads(path.read_text())
    _require(condition=isinstance(value, dict))
    return value


def _absent(*, path: Path) -> None:
    _path(value=str(path))
    _require(condition=not path.exists())


def _write(*, path: Path, value: Any) -> None:
    descriptor = os.open(path=path, flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o600)
    with os.fdopen(fd=descriptor, mode="w") as stream:
        json.dump(obj=value, fp=stream, indent=2)
        stream.write("\n")


def _inventory(*, root: Path, files: JsonObject, extra: set[str]) -> None:
    _require(condition=isinstance(files, dict) and bool(files))
    _require(condition=not set(files).intersection(extra - {"context.json"}))
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    _require(condition=actual == set(files) | extra)
    for relative, expected in files.items():
        name = Path(relative)
        _require(condition=not name.is_absolute() and ".." not in name.parts)
        _pin(record={"path": str(root / name), "sha256": expected})


def _expected_job_keys() -> set[tuple[str, str, str, str, int, str]]:
    keys = {
        (fold, recipe, setting, padding, seed, "cuda")
        for fold in ("fold_a", "fold_b")
        for recipe in RECIPES
        for seed in (0, 1, 2)
        for setting, padding in (("main", "equal"), ("sensitivity", "equal"), ("main", "none"))
    }
    keys.add(("fold_a", "solar_minimal_requested", "main", "equal", 0, "cpu"))
    return keys


def _validate(*, spec: JsonObject) -> list[Job]:
    _fields(value=spec, names=TOP_FIELDS)
    _require(condition=type(spec["schema_version"]) is int and spec["schema_version"] == 1)
    _require(condition=spec["state"] == "reviewed_ready")
    repository = _path(value=spec["repository_root"])
    data = _path(value=spec["data_root"])
    extension = _path(value=spec["extension_root"])
    _require(condition=extension == data / "requested_features")
    _require(
        condition=_path(value=spec["log_root"]).is_relative_to(repository / ".claude/worktrees")
    )
    _require(
        condition=Path(spec["fitter"]["path"])
        == repository / "studies/quick_wins/fixed_pipeline_fit.py"
    )
    _require(condition=spec["fitter"]["sha256"] == FITTER_SHA)
    _require(condition=spec["augmentation_builder"]["sha256"] == BUILDER_SHA)
    _require(condition=spec["audit"]["sha256"] == AUDIT_SHA)
    _fields(
        value=spec["frames"],
        names={f"{f}_{kind}" for f in ("fold_a", "fold_b") for kind in ("original", "augmented")},
    )
    for record in spec["frames"].values():
        _fields(value=record, names={"root", "manifest_sha256", "context_sha256"})
        _path(value=record["root"])
    _fields(value=spec["augmentation_proofs"], names={"fold_a", "fold_b"})
    _require(condition=isinstance(spec["source_files"], list) and bool(spec["source_files"]))
    _require(condition=isinstance(spec["failure_markers"], list) and bool(spec["failure_markers"]))
    _require(condition=isinstance(spec["jobs"], list) and len(spec["jobs"]) == 91)
    expected = _expected_job_keys()
    actual = set()
    labels = set()
    for value in spec["jobs"]:
        _fields(value=value, names=set(Job.__annotations__))
        _require(
            condition=type(value["seed"]) is int
            and isinstance(value["label"], str)
            and re.fullmatch(r"[a-z0-9_]+", value["label"]) is not None
        )
        key = tuple(value[n] for n in ("fold", "recipe", "setting", "padding", "seed", "device"))
        _require(condition=key in expected and key not in actual and value["label"] not in labels)
        actual.add(key)
        labels.add(value["label"])
        frame_kind = "augmented" if value["recipe"] in AUGMENTED else "original"
        _require(condition=value["frame"] == f"{value['fold']}_{frame_kind}")
        output = (
            extension
            / "fits/combined"
            / value["fold"]
            / value["recipe"]
            / value["setting"]
            / f"{value['device']}_seed{value['seed']}_{value['padding']}"
        )
        _require(condition=_path(value=value["output"]) == output)
        _require(
            condition=value["role"]
            == ("numerical_diagnostic" if value["device"] == "cpu" else "requested_feature_fit")
        )
    _require(condition=actual == expected)
    _require(
        condition=spec["jobs"][0]["recipe"] == "solar_minimal_requested"
        and spec["jobs"][0]["fold"] == "fold_a"
        and spec["jobs"][0]["setting"] == "main"
        and spec["jobs"][0]["padding"] == "equal"
        and spec["jobs"][0]["seed"] == 0
        and spec["jobs"][0]["device"] == "cuda"
    )
    _require(condition=spec["jobs"][-1]["device"] == "cpu")
    return cast(list[Job], spec["jobs"])


def _runtime(*, spec: JsonObject) -> None:
    python = spec["python"]
    _fields(value=python, names={"path", "resolved_path", "sha256", "version", "venv_config"})
    # A virtualenv interpreter may be a declared symlink; pin its explicit resolved binary.
    executable = Path(python["path"])
    _require(condition=executable.is_absolute() and ".." not in executable.parts)
    _path(value=str(executable.parent))
    _require(
        condition=executable.resolve()
        == _pin(record={"path": python["resolved_path"], "sha256": python["sha256"]})
    )
    _require(condition=Path(sys.executable).absolute() == executable)
    _require(condition=sys.version.split()[0] == python["version"])
    config = _pin(record=python["venv_config"])
    _require(condition=config == executable.parent.parent / "pyvenv.cfg")
    _require(condition=Path(sys.prefix).absolute() == executable.parent.parent)
    runtime = spec["runtime"]
    _fields(value=runtime, names={"root", "inventory", "pythonpath"})
    root = _path(value=runtime["root"])
    expected_pythonpath = [str(root / "packages/contracts/src"), str(root / "packages/ml_core/src")]
    _require(condition=runtime["pythonpath"] == expected_pythonpath)
    inventory = _json(path=_pin(record=runtime["inventory"]))["combined"]
    _require(condition=isinstance(inventory, dict) and bool(inventory))
    _require(condition=all(_path(value=tree).is_dir() for tree in expected_pythonpath))
    for relative, sha in inventory.items():
        _pin(record={"path": str(root / relative), "sha256": sha})
    # These two explicit source trees are bounded code inventories, not data/root discovery.
    actual = {
        str(p.relative_to(root))
        for tree in expected_pythonpath
        for p in _path(value=tree).rglob("*.py")
    }
    _require(condition=actual == set(inventory))
    environment = _json(path=_pin(record=spec["environment"]))
    _fields(value=environment, names={"python", "versions", "data_paths"})
    _require(condition=environment["python"] == sys.version)
    _require(condition=isinstance(environment["versions"], dict) and bool(environment["versions"]))
    for name, version in environment["versions"].items():
        _require(condition=importlib.metadata.version(name) == version)
    _fields(
        value=environment["data_paths"],
        names={"DATA_PATH_INTERNAL", "DATA_PATH_DELIVERY", "REPO_DATA_DIR"},
    )
    for path in environment["data_paths"].values():
        _require(condition=_path(value=path).is_dir())


def _frame(*, record: JsonObject, fold: str) -> JsonObject:
    root = _path(value=record["root"])
    manifest = _json(
        path=_pin(record={"path": str(root / "manifest.json"), "sha256": record["manifest_sha256"]})
    )
    context = _json(
        path=_pin(record={"path": str(root / "context.json"), "sha256": record["context_sha256"]})
    )
    _require(condition=manifest["complete"] is True and manifest["context"] == context)
    expected_fold = "mid_2025_to_mid_2026" if fold == "fold_a" else "backwards_2025_07_to_2026_06"
    _require(
        condition=manifest["window"] == context["window"]
        and manifest["window"]["fold_id"] == expected_fold
    )
    _require(
        condition=(
            context["archive_version"],
            context["power_version"],
            context["capacity_version"],
            context["population_version"],
        )
        == (924, 70, 1, 1)
    )
    _require(
        condition=context["pipeline_tag"] == "combined"
        and context["study_end_exclusive"] == "2026-07-01T00:00:00+00:00"
    )
    _inventory(root=root, files=manifest["files"], extra={"manifest.json", "context.json"})
    return manifest


def _old_completion(*, record: JsonObject) -> None:
    receipt = _json(path=_pin(record=record))
    _fields(
        value=receipt,
        names={
            "schema_version",
            "execution_session_id",
            "exit_code",
            "expected_fits",
            "completed_fits",
            "footer_expected",
            "unrelated_gpu_jobs_absent",
            "completion_inventory",
        },
    )
    _require(
        condition=receipt["schema_version"] == 1
        and receipt["execution_session_id"] == 16755
        and receipt["exit_code"] == 0
        and receipt["expected_fits"] == 102
        and receipt["completed_fits"] == 102
        and receipt["footer_expected"] is False
        and receipt["unrelated_gpu_jobs_absent"] is True
    )
    inventory = _json(path=_pin(record=receipt["completion_inventory"]))
    _fields(value=inventory, names={"fits"})
    _require(condition=isinstance(inventory["fits"], list) and len(inventory["fits"]) == 102)
    roots = set()
    for entry in inventory["fits"]:
        _fields(value=entry, names={"root", "manifest_sha256", "complete_sha256"})
        root = _path(value=entry["root"])
        _require(condition=root not in roots)
        roots.add(root)
        for name, field in (
            ("manifest.json", "manifest_sha256"),
            ("complete.json", "complete_sha256"),
        ):
            _pin(record={"path": str(root / name), "sha256": entry[field]})


def _preflights(*, spec: JsonObject) -> None:
    _require(
        condition=isinstance(spec["reuse_preflights"], list) and len(spec["reuse_preflights"]) == 6
    )
    scopes = set()
    for entry in spec["reuse_preflights"]:
        _fields(value=entry, names={"scope", "root", "complete_sha256", "spec"})
        _fields(value=entry["scope"], names={"fold", "setting", "padding"})
        scope = entry["scope"]
        key = (scope["fold"], scope["setting"], scope["padding"])
        _require(condition=key not in scopes)
        scopes.add(key)
        root = _path(value=entry["root"])
        _absent(path=root / "failure.json")
        completion = _json(
            path=_pin(
                record={"path": str(root / "complete.json"), "sha256": entry["complete_sha256"]}
            )
        )
        _fields(value=completion, names={"status", "files"})
        _require(
            condition=completion["status"] == "complete"
            and set(completion["files"]) == {"proof.json", "report.md"}
        )
        for name, sha in completion["files"].items():
            _pin(record={"path": str(root / name), "sha256": sha})
        proof = _json(path=root / "proof.json")
        audit_spec = _json(path=_pin(record=entry["spec"]))
        _require(
            condition=proof["status"] == "complete"
            and proof["phase"] == "reuse_preflight"
            and audit_spec["phase"] == "reuse_preflight"
            and proof["scope"] == scope == audit_spec["scope"]
            and proof["spec_sha256"] == entry["spec"]["sha256"]
            and proof["audit_sha256"] == AUDIT_SHA
            and proof["thread_caps"] == dict.fromkeys(THREADS, "8")
        )
        for role, binding in audit_spec["sources"].items():
            _pin(record=binding)
            expected = {
                "fitter": spec["fitter"],
                "audit": spec["audit"],
                "augmentation_builder": spec["augmentation_builder"],
            }.get(role)
            if expected is not None:
                _require(condition=binding == expected)
        _pin(record=audit_spec["baseline"])
        for kind in ("original", "augmented"):
            binding = spec["frames"][f"{scope['fold']}_{kind}"]
            _require(
                condition=audit_spec["frames"][kind]
                == {"root": binding["root"], "manifest_sha256": binding["manifest_sha256"]}
            )
        _require(
            condition=audit_spec["augmentation_proof"] == spec["augmentation_proofs"][scope["fold"]]
        )
        checks = {(r["recipe"], r["seed"]) for r in proof["checks"]}
        expected_controls = {"production_cleaned", "lean_pv", "lean_wind"}
        if scope["padding"] == "equal":
            expected_controls |= {
                "production_cleaned_guarded_windchill",
                "lean_wind_guarded_windchill",
            }
        _require(condition=checks == {(r, seed) for r in expected_controls for seed in (0, 1, 2)})
        _require(condition=len(proof["checks"]) == len(checks))
        for row in proof["checks"]:
            _require(
                condition=row["original_vs_augmented_preparation_bit_exact"] is True
                and row["saved_feature_order_and_training_counts_match"] is True
            )
        _require(
            condition=len(proof["guard_inputs"]) == 5
            and {r["recipe"] for r in proof["guard_inputs"]} == set(RECIPES)
        )
        for row in proof["guard_inputs"]:
            _require(
                condition=row["ordinary_vs_guarded_target_prepared_inputs_bit_exact"] is True
                and row["separate_guarded_forecast_generated"] is False
            )
        for fit_record in audit_spec["fits"]:
            fit_root = _path(value=fit_record["root"])
            manifest = _json(
                path=_pin(
                    record={
                        "path": str(fit_root / "manifest.json"),
                        "sha256": fit_record["manifest_sha256"],
                    }
                )
            )
            complete = _json(
                path=_pin(
                    record={
                        "path": str(fit_root / "complete.json"),
                        "sha256": fit_record["complete_sha256"],
                    }
                )
            )
            _inventory(
                root=fit_root, files=complete["files"], extra={"manifest.json", "complete.json"}
            )
            for source, sha in manifest["dependency_sha256"].items():
                _pin(record={"path": source, "sha256": sha})
    _require(
        condition=scopes
        == {
            (fold, setting, padding)
            for fold in ("fold_a", "fold_b")
            for setting, padding in (("main", "equal"), ("sensitivity", "equal"), ("main", "none"))
        }
    )


def _bindings(*, spec: JsonObject, spec_path: Path, spec_sha: str) -> None:
    _pin(record={"path": str(spec_path), "sha256": spec_sha})
    _require(condition=_pin(record=spec["queue"]).resolve() == Path(__file__).resolve())
    for role in ("fitter", "augmentation_builder", "audit"):
        _pin(record=spec[role])
    for binding in spec["source_files"]:
        _pin(record=binding)
    _runtime(spec=spec)
    for name, record in spec["frames"].items():
        _frame(record=record, fold=name.removesuffix("_original").removesuffix("_augmented"))
    for fold, binding in spec["augmentation_proofs"].items():
        proof = _json(path=_pin(record=binding))
        _require(
            condition=proof["status"] == "complete"
            and proof["builder_sha256"] == BUILDER_SHA
            and proof["original_manifest_sha256"]
            == spec["frames"][f"{fold}_original"]["manifest_sha256"]
            and proof["augmented_manifest_sha256"]
            == spec["frames"][f"{fold}_augmented"]["manifest_sha256"]
            and proof["original_columns_bit_exact"] is True
            and proof["copied_artifacts_byte_exact"] is True
        )
    _old_completion(record=spec["original_gpu_completion"])
    _preflights(spec=spec)
    for name in [*spec["failure_markers"], str(Path(spec["extension_root"]) / "queue-failed.json")]:
        _absent(path=Path(name))


def _argv(*, spec: JsonObject, job: Job) -> list[str]:
    return [
        spec["python"]["path"],
        spec["fitter"]["path"],
        "--frames",
        spec["frames"][job["frame"]]["root"],
        "--output",
        str(Path(job["output"]).parents[2]),
        "--recipe",
        job["recipe"],
        "--seed",
        str(job["seed"]),
        "--device",
        job["device"],
        "--setting",
        job["setting"],
        "--padding",
        job["padding"],
    ]


def _finished_fit(*, spec: JsonObject, job: Job) -> JsonObject:
    root = _path(value=job["output"])
    manifest, complete = [_json(path=root / name) for name in ("manifest.json", "complete.json")]
    _require(
        condition=all(
            manifest[name] == job[name]
            for name in ("recipe", "seed", "device", "setting", "padding")
        )
    )
    _require(
        condition=manifest["script_sha256"] == FITTER_SHA
        and manifest["input_manifest_sha256"] == spec["frames"][job["frame"]]["manifest_sha256"]
        and complete["source_member_count"] == complete["sample_count"] == 51
    )
    frame_manifest = _json(path=Path(spec["frames"][job["frame"]]["root"]) / "manifest.json")
    _require(
        condition=manifest["input_manifest"] == frame_manifest
        and manifest["input_fingerprint"] == frame_manifest["input_fingerprint"]
    )
    _inventory(root=root, files=complete["files"], extra={"manifest.json", "complete.json"})
    return {
        "label": job["label"],
        "role": job["role"],
        "manifest_sha256": _sha(path=root / "manifest.json"),
        "complete_sha256": _sha(path=root / "complete.json"),
    }


def _environment(*, spec: JsonObject) -> ProcessEnvironment:
    """Construct the bound child environment before launching any fitter."""
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP"):
        _require(condition=name not in environment)
    environment.update(dict.fromkeys(THREADS, "8"))
    recorded_environment = _json(path=_pin(record=spec["environment"]))
    environment.update(recorded_environment["data_paths"])
    environment.update(
        PYTHONPATH=":".join(spec["runtime"]["pythonpath"]),
        MLFLOW_DISABLE_AGENT_HINT="1",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return environment


def _run(*, spec_path: Path, expected_sha: str) -> None:
    os.umask(0o077)
    spec = _json(path=_pin(record={"path": str(spec_path), "sha256": expected_sha}))
    jobs = _validate(spec=spec)
    repository = _path(value=spec["repository_root"])
    lock_path = repository / ".claude/worktrees/part1b-prepared/part1b-gpu.lock"
    _path(value=str(lock_path))
    descriptor = os.open(path=lock_path, flags=os.O_RDWR | os.O_CREAT, mode=0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _bindings(spec=spec, spec_path=spec_path, spec_sha=expected_sha)
        logs = _path(value=spec["log_root"])
        _absent(path=logs)
        for job in jobs:
            _absent(path=Path(job["output"]))
        for value in spec["frames"].values():
            root = Path(value["root"])
            _require(condition=not logs.is_relative_to(root) and not root.is_relative_to(logs))
        Path(spec["extension_root"]).mkdir(mode=0o700, exist_ok=True)
        logs.mkdir(mode=0o700, exist_ok=False)
        try:
            _write(path=logs / "execution-spec.json", value=spec)
            environment = _environment(spec=spec)
            results = []
            for job in jobs:
                _bindings(spec=spec, spec_path=spec_path, spec_sha=expected_sha)
                _absent(path=Path(job["output"]))
                Path(job["output"]).parents[2].mkdir(parents=True, exist_ok=True, mode=0o700)
                argv = _argv(spec=spec, job=job)
                _write(path=logs / f"{job['label']}.argv.json", value={"argv": argv, "job": job})
                log_fd = os.open(
                    path=logs / f"{job['label']}.log",
                    flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    mode=0o600,
                )
                started = time.monotonic()
                with os.fdopen(fd=log_fd, mode="w") as stream:
                    subprocess.run(
                        args=argv,
                        cwd=repository,
                        env=environment,
                        check=True,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        pass_fds=(descriptor,),
                    )
                elapsed = time.monotonic() - started
                _bindings(spec=spec, spec_path=spec_path, spec_sha=expected_sha)
                result = {**_finished_fit(spec=spec, job=job), "elapsed_seconds": elapsed}
                _write(path=logs / f"{job['label']}.completed.json", value=result)
                results.append(result)
                print(
                    f"Completed {job['label']}; measured elapsed {elapsed:.1f} seconds.", flush=True
                )
            for job, result in zip(jobs, results, strict=True):
                final = _finished_fit(spec=spec, job=job)
                _require(condition=all(result[name] == value for name, value in final.items()))
            _bindings(spec=spec, spec_path=spec_path, spec_sha=expected_sha)
            _write(
                path=logs / "complete.json",
                value={"status": "complete", "gpu_jobs": 90, "cpu_jobs": 1, "jobs": results},
            )
            print(
                "Requested feature queue complete: 90 GPU fits and one CPU diagnostic.", flush=True
            )
        except Exception, KeyboardInterrupt:
            failure = {"status": "failed", "no_automatic_retry": True}
            _write(path=logs / "failure.json", value=failure)
            _write(path=Path(spec["extension_root"]) / "queue-failed.json", value=failure)
            raise
    finally:
        os.close(descriptor)


def main() -> None:
    """Run an explicitly hash-bound reviewed queue; unbound templates always fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    try:
        _run(spec_path=args.spec.absolute(), expected_sha=args.spec_sha256)
    except Exception, KeyboardInterrupt:  # noqa: BLE001 — suppress private artifact paths.
        raise SystemExit(
            "Private queue stopped; preserve outputs and logs; no automatic retry."
        ) from None


if __name__ == "__main__":
    main()
