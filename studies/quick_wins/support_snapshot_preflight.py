"""Check current selected population, metadata and capacity against frozen support.

Private proposal for normal studies/quick_wins placement. No frame building or estimator calls.
Use separate before/after receipts around each full build; power91 equality has its own receipt.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Final

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
    raise SystemExit("Support gate requires seven thread caps8 before imports")

import patito as pt  # noqa: E402
import polars as pl  # noqa: E402
import power91_bounded_preflight as gate  # noqa: E402
from contracts.power_schemas import EffectiveCapacity, TimeSeriesMetadata  # noqa: E402

type JsonObject = dict[str, Any]


def _pins(*, spec: JsonObject) -> dict[str, str]:
    """Check finite imported roles and the approved staged runtime/environment files."""
    actual = {
        "builder": inspect.getfile(gate.builder),
        "input_reader": inspect.getfile(gate.builder.inputs),
        "power_gate": inspect.getfile(gate),
        "support_gate": str(Path(__file__).resolve()),
    }
    gate._require(condition=set(spec["sources"]) == set(actual))
    for role, record in spec["sources"].items():
        gate._require(condition=str(gate._pin(record=record)) == actual[role])
    gate._require(
        condition=spec["sources"]["builder"]["sha256"]
        == "715d6cf3806528b308ca159e59957b8f95cae76fe5013412f6e53014b8c23a79"
    )
    gate._require(
        condition=spec["sources"]["power_gate"]["sha256"]
        == "ae8393191599cc17ead6c85be821ff73c8dbaf969eb4be07e3a174e5b51ebb50"
    )
    environment = json.loads(gate._pin(record=spec["environment"]).read_bytes())
    gate._require(condition=sys.version == environment["python"])
    for key, value in environment["environment"].items():
        gate._require(condition=os.environ.get(key) == value)
    for name, expected in environment["versions"].items():
        gate._require(condition=importlib.metadata.version(name) == expected)
    dependencies = json.loads(gate._pin(record=spec["dependencies"]).read_bytes())
    for path, digest in dependencies["preexisting_consumed_source_sha256"].items():
        gate._pin(record={"path": path, "sha256": digest})
    roles = {
        "frame_builder": inspect.getfile(gate.builder),
        "feature_inventory": inspect.getfile(sys.modules["build_frames"]),
        "input_reader": inspect.getfile(gate.builder.inputs),
        "solar_wrapper": inspect.getfile(gate.builder.zenith),
        "power_reader": inspect.getfile(gate.power),
        "weather_contract": inspect.getfile(sys.modules["contracts.weather_schemas"]),
        "radiation": inspect.getfile(sys.modules["ml_core.features._radiation"]),
        "tabular": inspect.getfile(sys.modules["ml_core.features.tabular_feature_engineer"]),
        "wind": inspect.getfile(sys.modules["weather_utils.wind"]),
    }
    gate._require(condition=roles == dependencies["source_role_paths"])
    receipt_path = gate._pin(record=spec["staging_receipt"])
    staged = json.loads(receipt_path.read_bytes())
    gate._require(
        condition=staged["complete"] is True
        and len(staged["files"]) == 27
        and staged["commit"] == "01ddf5332063e369827787c5f23ec75d58a607a6"
    )
    for record in staged["files"]:
        gate._pin(
            record={"path": str(receipt_path.parent / record["path"]), "sha256": record["sha256"]}
        )
    for contract in (TimeSeriesMetadata, EffectiveCapacity):
        gate._require(
            condition=Path(inspect.getfile(contract)).resolve()
            == receipt_path.parent / "packages/contracts/src/contracts/power_schemas.py"
        )
    return roles


def _state(*, spec: JsonObject, settings: gate.Settings) -> JsonObject:
    """Guard current heads and physical metadata file before and after materialization."""
    paths = {
        "population": str(settings.eligible_time_series_data_path),
        "capacity": str(settings.effective_capacity_data_path),
        "metadata": str(settings.metadata_path),
        "power": gate.power.CLEANED_POWER_DELTA_URI,
    }
    gate._require(condition=paths == spec["paths"])
    heads = {
        name: gate.builder._version(uri=paths[name], settings=settings)
        for name in ("population", "capacity", "power")
    }
    gate._require(condition=heads == {"population": 1, "capacity": 1, "power": 91})
    archive = gate.DeltaTable(table_uri=gate.CANONICAL_ARCHIVE, version=924)
    gate._require(condition=archive.version() == 924)
    archive_id = hashlib.sha256(str(archive.metadata().id).encode()).hexdigest()
    gate._require(condition=archive_id == gate.TABLE_ID_SHA)
    archive_commit = gate._sha(
        path=Path(gate.CANONICAL_ARCHIVE) / "_delta_log" / f"{924:020d}.json"
    )
    gate._require(condition=archive_commit == gate.COMMIT_PINS[924])
    return {
        "heads": heads,
        "metadata_file_sha256": gate._sha(path=gate._plain(value=paths["metadata"])),
        "archive_commit924_sha256": archive_commit,
        "archive_table_id_sha256": archive_id,
    }


def _current(*, settings: gate.Settings) -> dict[str, pl.DataFrame]:
    """Mirror unchanged builder main's selected support construction exactly."""
    options = gate.builder.typeddict_to_dict(settings.storage_options)
    population = (
        pl.read_delta(settings.eligible_time_series_data_path, version=1, storage_options=options)
        .filter(pl.col("fold_id") == gate.builder.inputs.FOLD_A)
        .select("time_series_id")
        .sort("time_series_id")
    )
    gate._require(
        condition=population.height == 31 and population["time_series_id"].n_unique() == 31
    )
    ids = population["time_series_id"].to_list()
    metadata = gate.builder.inputs.series_metadata(settings=settings, time_series_ids=ids)
    metadata = pt.DataFrame(metadata.sort("time_series_id")).set_model(TimeSeriesMetadata)
    gate._require(condition=metadata.height == 31)
    capacity = (
        pl.read_delta(settings.effective_capacity_data_path, version=1, storage_options=options)
        .filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")
    )
    EffectiveCapacity.validate(capacity)
    gate._require(condition=capacity.height == 31 and capacity["time_series_id"].n_unique() == 31)
    return {"population": population, "metadata": pl.DataFrame(metadata), "capacity": capacity}


def _frozen(*, binding: JsonObject, current: dict[str, pl.DataFrame]) -> JsonObject:
    root = gate._plain(value=binding["root"])
    for name in ("manifest", "context"):
        gate._pin(record={"path": str(root / f"{name}.json"), "sha256": binding[f"{name}_sha256"]})
    manifest = json.loads((root / "manifest.json").read_bytes())
    context = json.loads((root / "context.json").read_bytes())
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    gate._require(
        condition=manifest["complete"] is True
        and manifest["context"] == context
        and manifest["input_fingerprint"] == fingerprint == binding["input_fingerprint"]
        and context["population_version"] == 1
        and context["capacity_version"] == 1
    )
    old = {}
    for name in ("metadata", "capacity"):
        path = root / f"{name}.parquet"
        gate._pin(record={"path": str(path), "sha256": manifest["files"][f"{name}.parquet"]})
        old[name] = pl.read_parquet(path)
    old["population"] = old["metadata"].select("time_series_id").sort("time_series_id")
    for name in ("population", "metadata", "capacity"):
        gate._exact(left=old[name], right=current[name])
        expected = context["input_hashes"][name]
        # Metadata's historical IPC precedes a representation-changing Parquet write.
        # File pin + exact semantic equality + CURRENT historical IPC retain its binding.
        if name != "metadata":
            gate._require(condition=gate.builder._digest_frame(frame=old[name]) == expected)
        gate._require(condition=gate.builder._digest_frame(frame=current[name]) == expected)
    return {
        "root": str(root),
        "manifest_sha256": binding["manifest_sha256"],
        "context_sha256": binding["context_sha256"],
        "input_fingerprint": fingerprint,
        "support_equal": True,
        "historical_ipc_fingerprints": {
            name: context["input_hashes"][name] for name in ("population", "metadata", "capacity")
        },
    }


def main() -> None:
    """Write a private before/after support receipt, without launching any build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("before", "after"))
    args = parser.parse_args()
    path = gate._plain(value=str(args.spec))
    spec_hash = gate._sha(path=path)
    spec = json.loads(path.read_bytes())
    gate._require(
        condition=set(spec)
        == {
            "sources",
            "environment",
            "dependencies",
            "staging_receipt",
            "paths",
            "frozen",
            "power_archive_proof",
        }
    )
    imports = _pins(spec=spec)
    gate._require(condition=set(spec["frozen"]) == {"fold_a", "fold_b"})
    power_proof = json.loads(gate._pin(record=spec["power_archive_proof"]).read_bytes())
    gate._require(
        condition=power_proof["complete"] is True
        and power_proof["spec_sha256"]
        == "27e394fa8680b1f85c27cd991e0d4d47574b480f9036eaf14689f67bcef933c4"
        and power_proof["source_sha256"] == spec["sources"]["power_gate"]["sha256"]
        and power_proof["current_power_version"] == 91
        and set(power_proof["frozen"]) == {"fold_a", "fold_b"}
        and all(row["bounded_exact_equal"] is True for row in power_proof["frozen"].values())
        and power_proof["archive"]["metadata_gate_passed"] is True
    )
    output = gate._plain(value=str(args.output))
    gate._require(condition=not output.exists() and output.parent.is_dir())
    for binding in spec["frozen"].values():
        root = gate._plain(value=binding["root"])
        gate._require(condition=not output.is_relative_to(root) and not root.is_relative_to(output))
    output.mkdir(mode=0o700)
    settings = gate.Settings()
    with (
        (output / "private.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        gate._pin(record=power_proof["current_metadata"])
        before = _state(spec=spec, settings=settings)
        gate._require(
            condition=before["metadata_file_sha256"] == power_proof["current_metadata"]["sha256"]
            and spec["paths"]["metadata"] == power_proof["current_metadata"]["path"]
        )
        current = _current(settings=settings)
        frozen = {
            fold: _frozen(binding=binding, current=current)
            for fold, binding in spec["frozen"].items()
        }
        for fold, row in frozen.items():
            gate._require(
                condition=all(
                    row[key] == power_proof["frozen"][fold][key]
                    for key in ("root", "manifest_sha256", "context_sha256", "input_fingerprint")
                )
            )
        gate._require(condition=_state(spec=spec, settings=settings) == before)
        for binding in spec["frozen"].values():
            _frozen(binding=binding, current=current)
        gate._require(condition=_pins(spec=spec) == imports)
        gate._pin(record=spec["power_archive_proof"])
        gate._pin(record=power_proof["current_metadata"])
        gate._require(condition=gate._sha(path=path) == spec_hash)
    proof = {
        "complete": True,
        "phase": args.phase,
        "spec_sha256": spec_hash,
        "source_sha256": gate._sha(path=Path(__file__).resolve()),
        "frozen": frozen,
        "state_before_after": before,
        "actual_import_roles": imports,
        "power_archive_proof": spec["power_archive_proof"],
        "build_started": False,
        "full_build_clearance": False,
    }
    (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        RuntimeError,
        gate.DeltaError,
        gate.DataFrameValidationError,
        pl.exceptions.PolarsError,
    ):
        raise SystemExit("Support gate failed; private output retained; no fallback") from None
