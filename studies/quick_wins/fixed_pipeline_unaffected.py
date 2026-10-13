"""Verify exact predictions for types unaffected by the lean and first solar recipes.

Read complete private fits and frame exports under an explicit study data root. Compare all
forecast sample keys and unaffected predictions exactly, then write a new private aggregate
summary. Input artifacts are read only; no prediction values or private identifiers are printed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final, Literal

# Polars initializes its worker pool on import, so the cap precedes the import.
os.environ.setdefault("POLARS_MAX_THREADS", "8")

import polars as pl

FoldType = Literal["fold_a", "fold_b"]
SettingType = Literal["main", "sensitivity"]
PaddingType = Literal["equal", "none"]
ArtifactHashes = dict[str, str]
Manifest = dict[str, Any]

KEYS: Final[list[str]] = [
    "time_series_id",
    "power_fcst_init_time",
    "nwp_init_time",
    "valid_time",
    "ensemble_member",
]
TYPES: Final[frozenset[str]] = frozenset(
    {"PV", "Wind", "Disaggregated Demand", "Raw Flow", "BESS", "Biofuel"}
)
PAIRS: Final[tuple[tuple[str, str, str], ...]] = (
    ("lean_pv", "production_cleaned", "PV"),
    ("lean_wind", "production_cleaned", "Wind"),
    ("solar_add_wind10", "lean_pv", "PV"),
)
FOLDS: Final[dict[FoldType, str]] = {
    "fold_a": "mid_2025_to_mid_2026",
    "fold_b": "backwards_2025_07_to_2026_06",
}


def _require(*, condition: bool) -> None:
    if not condition:
        raise ValueError("Private verification check failed")


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(*, path: Path) -> None:
    _require(condition=not any(p.is_symlink() for p in (path, *path.parents)))


def _checked_inventory(*, root: Path, files: ArtifactHashes) -> None:
    """Check every listed artifact without exposing private model filenames."""
    _require(condition=isinstance(files, dict) and bool(files))
    for name, expected in files.items():
        relative = Path(name)
        _require(condition=not relative.is_absolute() and ".." not in relative.parts)
        path = root / relative
        _plain(path=path)
        _require(condition=path.is_file() and _sha256(path=path) == expected)


def _frame_context(*, root: Path, fold: FoldType) -> tuple[Manifest, str, set[str], pl.DataFrame]:
    """Bind complete frame exports and private metadata to their manifest."""
    _plain(path=root)
    manifest_path = root / "manifest.json"
    _plain(path=manifest_path)
    pinned = _sha256(path=manifest_path)
    manifest = json.loads(manifest_path.read_text())
    _require(condition=manifest.get("complete") is True)
    _require(condition=manifest["window"]["fold_id"] == FOLDS[fold])
    _require(condition=manifest["context"]["pipeline_tag"] == "combined")
    _checked_inventory(root=root, files=manifest["files"])
    expected = {name for name in manifest["files"] if name.startswith("val/")}
    actual = {
        str(path.relative_to(root))
        for path in (root / "val").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _require(condition=bool(expected) and expected == actual)
    _require(
        condition=all(
            Path(name).parent == Path("val")
            and Path(name).name.startswith("init=")
            and Path(name).suffix == ".parquet"
            for name in expected
        )
    )
    _require(condition="metadata.parquet" in manifest["files"])
    metadata = pl.read_parquet(root / "metadata.parquet").select(
        "time_series_id", pl.col("time_series_type").cast(pl.String)
    )
    _require(
        condition=not metadata.is_empty() and not metadata.null_count().sum_horizontal().item()
    )
    _require(condition=not metadata["time_series_id"].is_duplicated().any())
    _require(condition=set(metadata["time_series_type"].to_list()) <= TYPES)
    return manifest, pinned, expected, metadata


def _audit_fit(
    *,
    root: Path,
    recipe: str,
    seed: int,
    setting: SettingType,
    padding: PaddingType,
    frame_manifest: Manifest,
    frame_sha: str,
    expected: set[str],
) -> ArtifactHashes:
    """Verify the full completed fit inventory and its shared frame provenance."""
    _plain(path=root)
    for name in ("manifest.json", "complete.json"):
        _plain(path=root / name)
    pinned = {name: _sha256(path=root / name) for name in ("manifest.json", "complete.json")}
    manifest = json.loads((root / "manifest.json").read_text())
    completion = json.loads((root / "complete.json").read_text())
    _require(
        condition=(
            manifest["recipe"],
            manifest["seed"],
            manifest["setting"],
            manifest["padding"],
            manifest["device"],
        )
        == (recipe, seed, setting, padding, "cuda")
    )
    _require(condition=manifest["input_manifest"] == frame_manifest)
    _require(condition=manifest["input_manifest_sha256"] == frame_sha)
    _require(condition=manifest["input_fingerprint"] == frame_manifest["input_fingerprint"])
    _require(condition=completion["source_member_count"] == 51 and completion["sample_count"] == 51)
    files = completion["files"]
    _require(
        condition="manifest.json" in files and files["manifest.json"] == pinned["manifest.json"]
    )
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _require(condition=actual == set(files) | {"complete.json"})
    forecasts = {name for name in files if name.startswith("forecasts/")}
    _require(condition=forecasts == {"forecasts/" + Path(name).name for name in expected})
    _checked_inventory(root=root, files=files)
    return pinned


def _selected_predictions(
    *, path: Path, ids: pl.DataFrame, expected: pl.DataFrame, all_keys: pl.DataFrame
) -> pl.DataFrame:
    """Select unaffected predictions after checking the full file's forecast keys."""
    frame = pl.read_parquet(path)
    _require(condition=set(frame.columns) == set(KEYS) | {"power_fcst"})
    _require(condition=not frame.is_empty() and not frame.null_count().sum_horizontal().item())
    _require(condition=frame["power_fcst"].is_finite().all())
    _require(condition=not frame.select(KEYS).is_duplicated().any())
    _require(condition=frame.select(KEYS).sort(KEYS).equals(all_keys))
    selected = (
        frame.join(ids, on="time_series_id", how="semi").select(*KEYS, "power_fcst").sort(KEYS)
    )
    _require(condition=not selected.is_empty() and selected.select(KEYS).equals(expected))
    return selected


def _compare_forecasts(
    *,
    frames: Path,
    expected_files: set[str],
    metadata: pl.DataFrame,
    ids: pl.DataFrame,
    roots: list[Path],
) -> int:
    """Compare every forecast file exactly without rounding or numerical tolerances."""
    count = 0
    source_runs = set()
    for name in sorted(expected_files):
        validation = pl.read_parquet(frames / name, columns=KEYS)
        _require(
            condition=not validation.is_empty()
            and not validation.null_count().sum_horizontal().item()
        )
        _require(
            condition=set(validation["time_series_id"].unique().to_list())
            == set(metadata["time_series_id"].to_list())
        )
        _require(condition=not validation.is_duplicated().any())
        run_times = set(validation["nwp_init_time"].unique().to_list())
        _require(condition=len(run_times) == 1 and not source_runs.intersection(run_times))
        source_runs.update(run_times)
        expected = validation.join(ids, on="time_series_id", how="semi").sort(KEYS)
        all_keys = validation.sort(KEYS)
        left, right = [
            _selected_predictions(
                path=root / "forecasts" / Path(name).name,
                ids=ids,
                expected=expected,
                all_keys=all_keys,
            )
            for root in roots
        ]
        _require(condition=left.schema == right.schema and left.equals(right))
        count += left.height
    return count


def _verify(
    *, data_root: Path, output: Path, fold: FoldType, setting: SettingType, padding: PaddingType
) -> None:
    """Write a new private summary only after exact comparison and final hash checks."""
    _plain(path=output)
    _require(condition=not output.exists())
    _require(condition=not (setting == "sensitivity" and padding == "none"))
    frames = data_root / "frames/combined" / fold
    manifest, frame_sha, expected_files, metadata = _frame_context(root=frames, fold=fold)
    summaries: list[dict[str, Any]] = []
    audited: dict[Path, ArtifactHashes] = {}
    pairs = PAIRS if padding == "equal" else PAIRS[:2]
    for left, right, changed_type in pairs:
        unaffected = metadata.filter(pl.col("time_series_type") != changed_type)
        ids = unaffected.select("time_series_id")
        _require(condition=not ids.is_empty())
        for seed in (0, 1, 2):
            roots = [
                data_root / "fits/combined" / fold / recipe / setting / f"cuda_seed{seed}_{padding}"
                for recipe in (left, right)
            ]
            for recipe, root in zip((left, right), roots, strict=True):
                if root not in audited:
                    audited[root] = _audit_fit(
                        root=root,
                        recipe=recipe,
                        seed=seed,
                        setting=setting,
                        padding=padding,
                        frame_manifest=manifest,
                        frame_sha=frame_sha,
                        expected=expected_files,
                    )
            count = _compare_forecasts(
                frames=frames,
                expected_files=expected_files,
                metadata=metadata,
                ids=ids,
                roots=roots,
            )
            summaries.append(
                {
                    "left": left,
                    "right": right,
                    "seed": seed,
                    "unaffected_types": sorted(unaffected["time_series_type"].unique().to_list()),
                    "forecast_files": len(expected_files),
                    "compared_rows": count,
                    "exact_match": True,
                }
            )
    # Recheck manifests, inventories, and artifact bytes after comparison.
    final_manifest, final_sha, final_files, _ = _frame_context(root=frames, fold=fold)
    _require(
        condition=final_sha == frame_sha
        and final_manifest == manifest
        and final_files == expected_files
    )
    for root, pinned in audited.items():
        recipe = root.parents[1].name
        seed = int(root.name.split("_")[1].removeprefix("seed"))
        _require(
            condition=_audit_fit(
                root=root,
                recipe=recipe,
                seed=seed,
                setting=setting,
                padding=padding,
                frame_manifest=manifest,
                frame_sha=frame_sha,
                expected=expected_files,
            )
            == pinned
        )
    result = {
        "status": "exact unaffected-type predictions verified",
        "fold": fold,
        "setting": setting,
        "padding": padding,
        "script_sha256": _sha256(path=Path(__file__)),
        "frame_manifest_sha256": frame_sha,
        "fit_manifests": [
            {"recipe": root.parents[1].name, "seed_directory": root.name, **pinned}
            for root, pinned in sorted(audited.items())
        ],
        "comparisons": summaries,
    }
    # Never serialize keys, predictions, or private identifiers.
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd=descriptor, mode="w") as stream:
        json.dump(obj=result, fp=stream, indent=2)
        stream.write("\n")
    print("Verified exact unaffected-type predictions; aggregate summary saved privately.")


def main() -> None:
    """Verify a complete fold's unaffected predictions and save a private aggregate summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--setting", choices=("main", "sensitivity"), required=True)
    parser.add_argument("--padding", choices=("equal", "none"), default="equal")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        _verify(
            data_root=args.data_root,
            output=args.output,
            fold=args.fold,
            setting=args.setting,
            padding=args.padding,
        )
    except Exception:  # noqa: BLE001 — unexpected library errors can disclose private keys.
        # Library exceptions can contain private row keys, file paths or values.
        raise SystemExit("Verification failed; no private row details emitted.") from None


if __name__ == "__main__":
    main()
