"""Compare current guarded power91 with frozen power70 exports; never build frames.

Private source proposal. Review and place in studies/quick_wins before execution.
Archive evidence checks metadata and recorded cutover lineage, not NWP row values.
"""

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import sys
from datetime import datetime
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
    raise SystemExit("Preflight requires seven thread caps8 before imports")

import fixed_pipeline_frames as builder  # noqa: E402
import patito as pt  # noqa: E402
import polars as pl  # noqa: E402
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata  # noqa: E402
from contracts.settings import Settings  # noqa: E402
from deltalake import DeltaTable  # noqa: E402
from deltalake.exceptions import DeltaError  # noqa: E402
from patito.exceptions import DataFrameValidationError  # noqa: E402

from studies import power  # noqa: E402

type JsonObject = dict[str, Any]
CANONICAL_ARCHIVE: Final[str] = "/mnt/data/nged-substation-forecast/data/NWP"
OLD_ARCHIVE: Final[str] = (
    "/home/jack/.codex/worktrees/f0c9/nged-substation-forecast/data/NWP-components-issue-525"
)
TABLE_ID_SHA: Final[str] = "d81e5458bb9bfa79c96cb2f18789fa85a8088364f74d4e5350b2eecae1cbd35b"
COMMIT_PINS: Final[dict[int, str]] = {
    0: "68d0dce1b8caf1c7cd09ba5b5884ef29bc9ea2bf093bc727ec50e2ce51629b56",
    924: "c66476fbfd0e0369e811b466b95916779cfc3d68d4dea0c52551b00c7235ece7",
}


def _require(*, condition: bool) -> None:
    if not condition:
        raise ValueError("Preflight failed; private partial output retained; no fallback")


def _plain(*, value: str) -> Path:
    path = Path(value)
    _require(condition=path.is_absolute() and path.resolve() == path)
    return path


def _sha(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(*, record: JsonObject) -> Path:
    _require(condition=set(record) == {"path", "sha256"})
    path = _plain(value=record["path"])
    _require(condition=_sha(path=path) == record["sha256"])
    return path


def _archive(*, spec: JsonObject) -> JsonObject:
    """Check fixed snapshot metadata and explicit recorded old/new location roles."""
    _require(
        condition=spec["old_path"] == OLD_ARCHIVE
        and spec["new_path"] == CANONICAL_ARCHIVE
        and spec["version"] == 924
    )
    cutover = _pin(record=spec["cutover_evidence"])
    readiness = json.loads(_pin(record=spec["independent_readiness"]).read_bytes())
    validation = readiness["validation"]
    _require(
        condition=readiness["status"] == "ready"
        and readiness["table_path"] == OLD_ARCHIVE
        and readiness["destination_snapshot_version"] == 924
        and all(
            validation[key] is True
            for key in (
                "full_inventory_matches",
                "all_partitions_verified",
                "source_file_lists_match",
                "partition_row_counts_match",
                "non_wind_columns_and_keys_unchanged",
                "wind_round_trip_within_precision",
            )
        )
    )
    root = _plain(value=CANONICAL_ARCHIVE)
    log_hashes = {}
    schema = None
    for version in range(925):
        path = _plain(value=str(root / "_delta_log" / f"{version:020d}.json"))
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        log_hashes[str(version)] = digest
        if version in COMMIT_PINS:
            _require(condition=digest == COMMIT_PINS[version])
        for line in content.splitlines():
            action = json.loads(line)
            if "metaData" in action:
                metadata = action["metaData"]
                _require(
                    condition=hashlib.sha256(metadata["id"].encode()).hexdigest() == TABLE_ID_SHA
                )
                current_schema = json.loads(metadata["schemaString"])
                _require(condition=schema is None or schema == current_schema)
                schema = current_schema
    _require(condition=schema is not None)
    schema = cast(JsonObject, schema)
    fields = {field["name"] for field in schema["fields"]}
    _require(condition={"wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"} <= fields)
    _require(
        condition=not fields.intersection(
            {
                "wind_speed_10m",
                "wind_speed_100m",
                "wind_direction_10m",
                "wind_direction_100m",
            }
        )
    )
    table = DeltaTable(table_uri=CANONICAL_ARCHIVE, version=924)
    _require(condition=table.version() == 924)
    _require(
        condition=hashlib.sha256(cast(str, table.metadata().id).encode()).hexdigest()
        == TABLE_ID_SHA
    )
    _require(condition=json.loads(table.schema().to_json()) == schema)
    files = table.file_uris()
    _require(condition=len(files) == 925 and len(set(files)) == 925)
    for value in files:
        path = _plain(value=value)
        _require(condition=path.is_relative_to(root) and path.is_file())
    return {
        "old_path": OLD_ARCHIVE,
        "new_path": CANONICAL_ARCHIVE,
        "version": 924,
        "table_id_sha256": TABLE_ID_SHA,
        "commit_sha256": log_hashes,
        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        "cutover_evidence_sha256": _sha(path=cutover),
        "independent_readiness_sha256": spec["independent_readiness"]["sha256"],
        "metadata_gate_passed": True,
        "new_full_file_checksum_verification": False,
        "snapshot_files_present": True,
        "identity_basis": "Recorded verified cutover plus exact snapshot metadata/logchain",
    }


def _exact(*, left: pl.DataFrame, right: pl.DataFrame) -> None:
    _require(condition=left.schema == right.schema and left.shape == right.shape)
    for column, dtype in left.schema.items():
        a, b = left[column], right[column]
        _require(condition=a.is_null().equals(b.is_null()))
        if dtype in (pl.Float32, pl.Float64):
            valid = a.is_not_null()
            _require(
                condition=a.filter(valid).to_numpy().tobytes()
                == b.filter(valid).to_numpy().tobytes()
            )
        else:
            _require(condition=a.equals(b))


def _power(*, binding: JsonObject, settings: Settings) -> JsonObject:
    """Use the unchanged guarded current reader; never read Delta version70."""
    root = _plain(value=binding["root"])
    for name in ("manifest", "context"):
        _pin(record={"path": str(root / f"{name}.json"), "sha256": binding[f"{name}_sha256"]})
    manifest = json.loads((root / "manifest.json").read_bytes())
    context = json.loads((root / "context.json").read_bytes())
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    _require(
        condition=manifest["complete"] is True
        and manifest["context"] == context
        and manifest["input_fingerprint"] == fingerprint
        and fingerprint == binding["input_fingerprint"]
        and context["power_version"] == 70
        and context["archive"] == OLD_ARCHIVE
        and context["archive_version"] == 924
    )
    for filename in ("metadata.parquet", "bounded_power.parquet"):
        _pin(record={"path": str(root / filename), "sha256": manifest["files"][filename]})
    metadata = pl.read_parquet(root / "metadata.parquet")
    ids = sorted(metadata["time_series_id"].to_list())
    _require(condition=len(ids) == 31 and len(set(ids)) == 31)
    selected = builder.inputs.series_metadata(settings=settings, time_series_ids=ids)
    selected = pl.DataFrame(
        pt.DataFrame(selected.sort("time_series_id")).set_model(TimeSeriesMetadata)
    )
    _exact(left=metadata, right=selected)
    # Historical IPC precedes Parquet; pinned readback bytes can differ in representation.
    _require(condition=builder._digest_frame(frame=selected) == context["input_hashes"]["metadata"])
    frozen = pl.read_parquet(root / "bounded_power.parquet")
    current = builder._power(settings=settings, version=91, series_ids=ids)
    for frame in (frozen, current):
        PowerTimeSeries.validate(frame)
        _require(
            condition=frame.columns == ["time_series_id", "time", "power"]
            and frame.equals(frame.sort("time_series_id", "time"))
            and cast(datetime, frame["time"].max()) < builder.STUDY_END
            and not frame.select("time_series_id", "time").is_duplicated().any()
        )
    _exact(left=frozen, right=current)
    expected = context["input_hashes"]["bounded_power"]
    # File pin + exact semantic equality + current historical IPC binds the old values.
    # Requiring readback IPC as well would incorrectly constrain Parquet representation.
    _require(condition=builder._digest_frame(frame=current) == expected)
    return {
        "root": str(root),
        "manifest_sha256": binding["manifest_sha256"],
        "context_sha256": binding["context_sha256"],
        "input_fingerprint": fingerprint,
        "old_power_version": 70,
        "new_power_version": 91,
        "bounded_exact_equal": True,
        "bounded_power_sha256": manifest["files"]["bounded_power.parquet"],
        "metadata_sha256": manifest["files"]["metadata.parquet"],
        "bounded_power_context_fingerprint": expected,
        "rows_schema_order_nulls_numeric_bits_equal": True,
    }


def main() -> None:
    """Write a private conditional data-binding receipt only after both folds pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec_path = _plain(value=str(args.spec))
    spec_hash = _sha(path=spec_path)
    spec = json.loads(spec_path.read_bytes())
    _require(
        condition=set(spec)
        == {"sources", "frozen", "archive", "current_metadata", "metadata_diagnostic"}
    )
    _require(
        condition=set(spec["sources"])
        == {"builder", "power", "storage", "preflight", "input_reader"}
    )
    for record in spec["sources"].values():
        _pin(record=record)
    _require(
        condition=spec["sources"]["builder"]["path"] == inspect.getfile(builder)
        and spec["sources"]["power"]["path"] == inspect.getfile(power)
        and spec["sources"]["preflight"]["path"] == str(Path(__file__).resolve())
        and spec["sources"]["storage"]["path"] == inspect.getfile(power.scan_cleaned_power)
        and spec["sources"]["input_reader"]["path"] == inspect.getfile(builder.inputs)
        and builder.scan_power is power.scan_power
    )
    _require(
        condition=spec["sources"]["builder"]["sha256"]
        == "715d6cf3806528b308ca159e59957b8f95cae76fe5013412f6e53014b8c23a79"
    )
    _require(
        condition=spec["sources"]["power"]["sha256"]
        == "bef0e4a6c4c95057f5b674ff9d350dfd2fefe502e8725f0bd929d2792c6560a4"
    )
    _require(condition=set(spec["frozen"]) == {"fold_a", "fold_b"})
    diagnostic = spec["metadata_diagnostic"]
    for record in diagnostic.values():
        _pin(record=record)
    diagnosis = json.loads(_pin(record=diagnostic["result"]).read_bytes())
    diagnostic_spec = json.loads(_pin(record=diagnostic["spec"]).read_bytes())
    _require(
        condition=diagnostic_spec["frozen"] == spec["frozen"]
        and diagnostic_spec["current_metadata"] == spec["current_metadata"]
        and diagnosis["diagnostic_completed"] is True
        and diagnosis["authorises_reuse_or_build"] is False
        and set(diagnosis["folds"]) == {"fold_a", "fold_b"}
    )
    for result in diagnosis["folds"].values():
        _require(
            condition=result["context_binding_passed"] is True
            and result["frozen_file_pin_passed"] is True
            and result["current_ipc_matches_historical"] is True
            and result["current_roundtrip_ipc_matches_historical"] is False
            and result["frozen_ipc_matches_historical"] is False
            and all(value is True for value in result["current_vs_frozen"].values())
            and all(value is True for value in result["current_parquet_roundtrip"].values())
        )
    metadata_path = _pin(record=spec["current_metadata"])
    output = _plain(value=str(args.output))
    _require(condition=not output.exists() and output.parent.is_dir())
    for binding in spec["frozen"].values():
        root = _plain(value=binding["root"])
        _require(condition=not output.is_relative_to(root) and not root.is_relative_to(output))
    output.mkdir(mode=0o700)
    settings = Settings()
    _require(condition=str(settings.metadata_path) == str(metadata_path))
    with (
        (output / "private.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        archive = _archive(spec=spec["archive"])
        frozen = {
            fold: _power(binding=binding, settings=settings)
            for fold, binding in spec["frozen"].items()
        }
        _require(
            condition=builder._version(uri=power.CLEANED_POWER_DELTA_URI, settings=settings) == 91
        )
        _require(condition=_archive(spec=spec["archive"]) == archive)
        for binding in spec["frozen"].values():
            root = Path(binding["root"])
            for name in ("manifest", "context"):
                _pin(
                    record={"path": str(root / f"{name}.json"), "sha256": binding[f"{name}_sha256"]}
                )
            manifest = json.loads((root / "manifest.json").read_bytes())
            for name in ("bounded_power.parquet", "metadata.parquet"):
                _pin(record={"path": str(root / name), "sha256": manifest["files"][name]})
        for record in spec["sources"].values():
            _pin(record=record)
        _pin(record=spec["current_metadata"])
        for record in diagnostic.values():
            _pin(record=record)
        _require(condition=_sha(path=spec_path) == spec_hash)
    proof = {
        "complete": True,
        "metadata_diagnostic": diagnostic,
        "current_metadata": spec["current_metadata"],
        "spec_sha256": spec_hash,
        "source_sha256": _sha(path=Path(__file__).resolve()),
        "frozen": frozen,
        "archive": archive,
        "current_power_path": power.CLEANED_POWER_DELTA_URI,
        "current_power_version": 91,
        "old_delta70_read_or_cloned": False,
        "frame_build_clearance": False,
        "numerical_audit_or_loss_claim": False,
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
        DeltaError,
        DataFrameValidationError,
        pl.exceptions.PolarsError,
    ) as error:
        trace = error.__traceback__
        stages = []
        while trace is not None:
            stages.append((trace.tb_frame.f_code.co_name, trace.tb_lineno))
            trace = trace.tb_next
        print(json.dumps({"error_type": type(error).__name__, "stages": stages}), file=sys.stderr)
        raise SystemExit(
            "Power/archive preflight failed; private output retained; no fallback"
        ) from None
