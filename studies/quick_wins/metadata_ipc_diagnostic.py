"""Private metadata-only diagnostic; never authorise power reuse or a build."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

if any(
    os.environ.get(name) != "8"
    for name in (
        "POLARS_MAX_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
):
    raise SystemExit("Metadata diagnostic requires seven thread caps8 before imports")

import patito as pt
import polars as pl
import power91_bounded_preflight as gate
from contracts.power_schemas import TimeSeriesMetadata

type JsonObject = dict[str, Any]


def _comparison(*, left: pl.DataFrame, right: pl.DataFrame) -> dict[str, bool]:
    schema = left.schema == right.schema
    shape = left.shape == right.shape
    nulls = (
        schema
        and shape
        and all(left[name].is_null().equals(right[name].is_null()) for name in left.columns)
    )
    exact = False
    if nulls:
        try:
            gate._exact(left=left, right=right)
            exact = True
        except ValueError:
            pass
    return {
        "schema_equal": schema,
        "shape_equal": shape,
        "null_masks_equal": nulls,
        "values_and_float_bits_equal": exact,
    }


def _roundtrip(*, frame: pl.DataFrame) -> pl.DataFrame:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    buffer.seek(0)
    return pl.read_parquet(buffer)


def _pins(*, spec: JsonObject) -> None:
    roles = {
        "builder": inspect.getfile(gate.builder),
        "input_reader": inspect.getfile(gate.builder.inputs),
        "power_gate": inspect.getfile(gate),
        "diagnostic": str(Path(__file__).resolve()),
    }
    gate._require(condition=set(spec["sources"]) == set(roles))
    for name, record in spec["sources"].items():
        gate._require(condition=str(gate._pin(record=record)) == roles[name])
    environment = json.loads(gate._pin(record=spec["environment"]).read_bytes())
    gate._require(condition=sys.version == environment["python"])
    for name, value in environment["environment"].items():
        gate._require(condition=os.environ.get(name) == value)
    for name, value in environment["versions"].items():
        gate._require(condition=importlib.metadata.version(name) == value)
    dependencies = json.loads(gate._pin(record=spec["dependencies"]).read_bytes())
    for path, digest in dependencies["preexisting_consumed_source_sha256"].items():
        gate._pin(record={"path": path, "sha256": digest})
    receipt_path = gate._pin(record=spec["staging_receipt"])
    receipt = json.loads(receipt_path.read_bytes())
    gate._require(condition=receipt["complete"] is True and len(receipt["files"]) == 27)
    for record in receipt["files"]:
        gate._pin(
            record={"path": str(receipt_path.parent / record["path"]), "sha256": record["sha256"]}
        )
    gate._require(
        condition=Path(inspect.getfile(TimeSeriesMetadata)).resolve()
        == receipt_path.parent / "packages/contracts/src/contracts/power_schemas.py"
    )


def _fold(*, binding: JsonObject, settings: gate.Settings) -> dict[str, Any]:
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
    )
    gate._pin(
        record={
            "path": str(root / "metadata.parquet"),
            "sha256": manifest["files"]["metadata.parquet"],
        }
    )
    frozen = pl.read_parquet(root / "metadata.parquet")
    ids = frozen["time_series_id"].to_list()
    gate._require(
        condition=len(ids) == 31
        and len(set(ids)) == 31
        and frozen.equals(frozen.sort("time_series_id"))
    )
    current = gate.builder.inputs.series_metadata(settings=settings, time_series_ids=ids)
    current = pl.DataFrame(
        pt.DataFrame(current.sort("time_series_id")).set_model(TimeSeriesMetadata)
    )
    gate._require(condition=current.height == 31)
    expected = context["input_hashes"]["metadata"]
    current_rt = _roundtrip(frame=current)
    frozen_rt = _roundtrip(frame=frozen)
    return {
        "frozen_file_pin_passed": True,
        "context_binding_passed": True,
        "current_vs_frozen": _comparison(left=current, right=frozen),
        "current_parquet_roundtrip": _comparison(left=current, right=current_rt),
        "frozen_parquet_roundtrip": _comparison(left=frozen, right=frozen_rt),
        "current_chunk_layout_matches_frozen": all(
            current[name].n_chunks() == frozen[name].n_chunks() for name in current.columns
        )
        if current.columns == frozen.columns
        else False,
        "current_ipc_matches_historical": gate.builder._digest_frame(frame=current) == expected,
        "frozen_ipc_matches_historical": gate.builder._digest_frame(frame=frozen) == expected,
        "current_roundtrip_ipc_matches_historical": gate.builder._digest_frame(frame=current_rt)
        == expected,
        "frozen_roundtrip_ipc_matches_historical": gate.builder._digest_frame(frame=frozen_rt)
        == expected,
        "current_vs_roundtrip_ipc_equal": gate.builder._digest_frame(frame=current)
        == gate.builder._digest_frame(frame=current_rt),
    }


def main() -> None:
    """Emit only private diagnostic booleans after before/after pins pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    gate._require(condition=not args.output.exists())
    _pins(spec=spec)
    metadata = gate._pin(record=spec["current_metadata"])
    settings = gate.Settings()
    gate._require(condition=str(settings.metadata_path) == str(metadata))
    results = {
        name: _fold(binding=binding, settings=settings) for name, binding in spec["frozen"].items()
    }
    gate._require(condition=set(results) == {"fold_a", "fold_b"})
    gate._pin(record=spec["current_metadata"])
    _pins(spec=spec)
    for binding in spec["frozen"].values():
        root = gate._plain(value=binding["root"])
        manifest = json.loads((root / "manifest.json").read_bytes())
        for name in ("manifest", "context"):
            gate._pin(
                record={"path": str(root / f"{name}.json"), "sha256": binding[f"{name}_sha256"]}
            )
        gate._pin(
            record={
                "path": str(root / "metadata.parquet"),
                "sha256": manifest["files"]["metadata.parquet"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {"diagnostic_completed": True, "authorises_reuse_or_build": False, "folds": results},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


if __name__ == "__main__":
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main()
    except Exception:  # noqa: BLE001 — prevent metadata values in third-party exceptions
        raise SystemExit("Metadata diagnostic failed; no successful diagnostic receipt") from None
