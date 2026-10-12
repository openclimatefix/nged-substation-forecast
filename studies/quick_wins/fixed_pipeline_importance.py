"""Describe point-control training gain across both folds and three fitting seeds.

Read only completed production_cleaned fits on the combined pipeline. Normalize total gain
within each per-series booster, then average equally over series within a type and over six
refits. The reported min/max is variability across refits, never a confidence interval.
"""

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import arms
import polars as pl
import quick_wins_inputs as inputs

FOLDS: Final[frozenset[str]] = frozenset({inputs.FOLD_A, inputs.FOLD_B})
TYPES: Final[frozenset[str]] = frozenset(
    {"PV", "Wind", "Disaggregated Demand", "Raw Flow", "BESS", "Biofuel"}
)
RECIPE: Final[str] = "production_cleaned"
TITLE: Final[str] = "Training gain describes splits; feature tests decide usefulness"
SUBTITLE: Final[tuple[str, ...]] = (
    "Combined pipeline; full-feature point control; solar and wind panels.",
    "Dot: mean share of total training gain. Line: min/max across six fold/seed refits.",
    "Each refit averages series equally; each booster's feature shares sum to one.",
    "Ranges describe variability across refits, not confidence intervals.",
    "Gain divides credit between correlated features and does not establish benefit.",
)


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(*, root: Path, name: str, digest: str) -> None:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or _sha256(path=path) != digest:
        raise ValueError("artifact path or fingerprint mismatch")


def _gain_shares(*, model: dict[str, Any], features: list[str]) -> list[float]:
    config = model.get("booster_runtime_config") or model.get("effective_xgboost_config")
    if (
        config is None
        or config["learner"]["objective"]["name"] != "reg:squarederror"
        or model["scale"] != 1.0
        or model["rounds"] != 500
    ):
        raise ValueError("importance requires a 500-tree per-series squared-error point booster")
    if model["features"] != features or not features or len(set(features)) != len(features):
        raise ValueError("booster feature order differs from the verified recipe features")
    gains = dict.fromkeys(features, 0.0)
    assigned = set()
    for raw_name, raw_gain in model["total_gain"].items():
        if raw_name in gains:
            name = raw_name
        elif raw_name.startswith("f") and raw_name[1:].isdigit():
            index = int(raw_name[1:])
            if index >= len(features):
                raise ValueError("positional booster feature lies outside its feature list")
            name = features[index]
        else:
            raise ValueError("unmapped booster gain feature")
        if name in assigned or isinstance(raw_gain, bool):
            raise ValueError("duplicate feature gain or invalid gain type")
        gain = float(raw_gain)
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("feature gain must be finite and nonnegative")
        gains[name] = gain
        assigned.add(name)
    total = sum(gains.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("zero-total-gain booster has no defined gain-share distribution")
    return [gains[name] / total for name in features]


def _read_fit(*, run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root, frames = Path(run["path"]).resolve(), Path(run["frames"]).resolve()
    manifest_path, completion_path = root / "manifest.json", root / "complete.json"
    manifest_sha, completion_sha = _sha256(path=manifest_path), _sha256(path=completion_path)
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads(completion_path.read_text())
    if (
        manifest["recipe"] != RECIPE
        or manifest["setting"] != "main"
        or manifest["padding"] != "equal"
        or type(run["seed"]) is not int
        or run["seed"] not in {0, 1, 2}
        or manifest["seed"] != run["seed"]
        or manifest["device"] not in {"cuda", "cpu"}
    ):
        raise ValueError("importance requires the requested main equal-padding point-control fit")
    frame_manifest_path = frames / "manifest.json"
    frame_manifest = json.loads(frame_manifest_path.read_text())
    fold = frame_manifest["window"]["fold_id"]
    if (
        _sha256(path=frame_manifest_path) != manifest["input_manifest_sha256"]
        or frame_manifest != manifest["input_manifest"]
        or frame_manifest.get("complete") is not True
        or frame_manifest["context"]["pipeline_tag"] != "combined"
        or fold not in FOLDS
        or ("fold" in run and run["fold"] != fold)
    ):
        raise ValueError(
            "frame manifest does not bind the requested complete combined-pipeline fold"
        )
    if completion["source_member_count"] != 51 or completion["sample_count"] != 51:
        raise ValueError("gain report accepts point models only")
    required = {"models.json", "features.json", "training_counts.parquet"}
    if not required <= completion["files"].keys():
        raise ValueError("completed fit does not fingerprint all importance inputs")
    for name, digest in completion["files"].items():
        _verify_file(root=root, name=name, digest=digest)
    for name in ("metadata.parquet", "train.parquet"):
        _verify_file(root=frames, name=name, digest=frame_manifest["files"][name])
    metadata = pl.read_parquet(frames / "metadata.parquet").select(
        "time_series_id", pl.col("time_series_type").cast(pl.String)
    )
    if (
        metadata.null_count().sum_horizontal().item()
        or metadata["time_series_id"].is_duplicated().any()
        or not set(metadata["time_series_type"].to_list()) <= TYPES
    ):
        raise ValueError("metadata must have one approved non-null type per unique series")
    kinds = dict(metadata.iter_rows())
    features = json.loads((root / "features.json").read_text())
    expected = list(arms.PRODUCTION_COLUMNS)
    if set(features) != set(kinds.values()) or any(
        names != expected for names in features.values()
    ):
        raise ValueError(
            "point-control columns differ from the full sorted production feature list"
        )
    models = json.loads((root / "models.json").read_text())
    model_ids = [model["time_series_id"] for model in models]
    if len(model_ids) != len(set(model_ids)) or set(model_ids) != set(kinds):
        raise ValueError("booster population differs from the frozen metadata population")
    rows = []
    for model in models:
        kind = kinds[model["time_series_id"]]
        shares = _gain_shares(model=model, features=features[kind])
        rows.extend(
            {"series_type": kind, "feature": name, "share": share}
            for name, share in zip(features[kind], shares, strict=True)
        )
    # Private identifiers have served their lookup purpose and never enter aggregate rows.
    per_refit = (
        pl.DataFrame(rows)
        .group_by("series_type", "feature")
        .agg(mean_share=pl.col("share").mean(), series_count=pl.len())
        .with_columns(fold=pl.lit(fold), seed=pl.lit(run["seed"]))
    )
    if (
        _sha256(path=manifest_path) != manifest_sha
        or _sha256(path=completion_path) != completion_sha
    ):
        raise ValueError("fit manifests changed during the importance audit")
    for name in required:
        _verify_file(root=root, name=name, digest=completion["files"][name])
    _verify_file(
        root=frames, name="metadata.parquet", digest=frame_manifest["files"]["metadata.parquet"]
    )
    provenance = {
        "fold": fold,
        "seed": run["seed"],
        "device": manifest["device"],
        "manifest_sha256": manifest_sha,
        "fitter_script_sha256": manifest["script_sha256"],
        "fitter_dependency_sha256": manifest["dependency_sha256"],
        "completion_sha256": completion_sha,
        "frame_manifest_sha256": manifest["input_manifest_sha256"],
        "consumed_hashes": {name: completion["files"][name] for name in sorted(required)},
        "metadata_sha256": frame_manifest["files"]["metadata.parquet"],
        "training_frame_sha256": frame_manifest["files"]["train.parquet"],
        "population_sha256": hashlib.sha256(
            metadata.sort("time_series_id").write_json().encode()
        ).hexdigest(),
        "xgboost_version": manifest["xgboost_version"],
    }
    return per_refit.to_dicts(), provenance


def _report(*, table: pl.DataFrame, support: pl.DataFrame) -> str:
    lines = [
        "# Point-control training gain",
        "",
        "Total gain is normalized within each point booster; unsplit features have zero.",
        "Each type/refit averages series equally; the final mean averages all six refits equally.",
        "Min/max describe variability across refits, not confidence intervals or feature benefit.",
        "Zero-total-gain boosters are refused. Quantile models are not included.",
        "",
    ]
    for heading, frame in (("Refit support", support), ("Feature gain shares", table)):
        lines.extend(["## " + heading, "", "| " + " | ".join(frame.columns) + " |"])
        lines.append("|" + "---|" * frame.width)
        for row in frame.iter_rows():
            values = [f"{value:.8f}" if isinstance(value, float) else str(value) for value in row]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Write immutable public total-gain aggregates and text for a later reviewed chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("importance output exists; use a new immutable directory")
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("runs"), list)
        or len(spec["runs"]) != 6
    ):
        raise ValueError("importance requires exactly six fold-by-seed point-control runs")
    pieces, provenance = [], []
    for run in spec["runs"]:
        rows, audited = _read_fit(run=run)
        pieces.extend(rows)
        provenance.append(audited)
    expected = {(fold, seed) for fold in FOLDS for seed in range(3)}
    if {(run["fold"], run["seed"]) for run in provenance} != expected:
        raise ValueError("importance requires both folds and every seed exactly once")
    if len({run["device"] for run in provenance}) != 1:
        raise ValueError("importance refits must use the same device")
    if len({run["population_sha256"] for run in provenance}) != 1:
        raise ValueError("importance refits must use the same series population and type mapping")
    refits = pl.DataFrame(pieces)
    support = (
        refits.group_by("fold", "seed", "series_type")
        .agg(feature_count=pl.len(), series_count=pl.col("series_count").first())
        .sort("series_type", "fold", "seed")
    )
    if support.filter(pl.col("feature_count") != len(arms.PRODUCTION_COLUMNS)).height:
        raise ValueError("importance feature counts differ from the full point control")
    table = (
        refits.group_by("series_type", "feature")
        .agg(
            mean_share=pl.col("mean_share").mean(),
            min_share=pl.col("mean_share").min(),
            max_share=pl.col("mean_share").max(),
            refits=pl.len(),
            min_series_per_refit=pl.col("series_count").min(),
            max_series_per_refit=pl.col("series_count").max(),
        )
        .sort("series_type", "mean_share", descending=[False, True])
    )
    if table.filter(pl.col("refits") != 6).height or not {"PV", "Wind"} <= set(
        table["series_type"]
    ):
        raise ValueError("every reported type/feature requires six refits, including PV and Wind")
    args.output.mkdir(parents=True, exist_ok=False)
    table.write_parquet(args.output / "importance.parquet")
    refits.write_parquet(args.output / "refit_importance.parquet")
    support.write_parquet(args.output / "support.parquet")
    (args.output / "report.md").write_text(_report(table=table, support=support))
    (args.output / "figure-text.md").write_text(
        "# Point-control importance figure text\n\n"
        + TITLE
        + "\n\n"
        + "\n".join(SUBTITLE)
        + "\n\nPanels: Solar farms; Wind farms.\n"
        + "Axis: Share of total training gain (%); larger is more gain, not more benefit.\n"
        + "Feature labels: "
        + ", ".join(arms.PRODUCTION_COLUMNS)
        + ".\n"
    )
    manifest = {
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "script_sha256": _sha256(path=Path(__file__)),
        "arms_sha256": _sha256(path=Path(arms.__file__)),
        "production_config_sha256": _sha256(path=inputs.PRODUCTION_MODEL_CONFIG),
        "polars_version": version("polars"),
        "recipe": RECIPE,
        "refits": provenance,
        "statistics": "within-booster share; equal series and refits; min/max refit variability",
        "outputs": {
            path.name: _sha256(path=path)
            for path in sorted(args.output.iterdir())
            if path.is_file()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
