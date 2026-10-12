"""Report paired fixed-pipeline contrasts without exposing private series identifiers.

The specification lists fitted outputs by label and seed, plus contrasts chosen before fitting.
Each contrast checks exact forecast keys and normalisation, then resamples whole valid-time
months and one fitting seed jointly. Published tables contain only series types and aggregate
scores. Saved private losses remain read only.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from scipy import stats

KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
METRICS: Final[tuple[str, ...]] = (
    "crps_norm",
    "abs_error_norm",
    "pinball19_norm",
    "coverage80",
    "coverage90",
    *(f"pinball_{index:02d}_norm" for index in range(19)),
)
RESAMPLES: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 20261012
GROUPS: Final[tuple[str, ...]] = ("all", "PV", "Wind")
BANDS: Final[tuple[str, ...]] = ("all", "days 0 to 2", "days 3 to 10", "days 11 onward")


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MonthlyLosses:
    """Loss totals and counts, indexed by fitting seed, series, and valid-time month."""

    totals: np.ndarray
    counts: np.ndarray
    months: tuple[str, ...]
    seeds: tuple[int, ...]


def _filter(*, frame: pl.DataFrame, group: str, band: str) -> pl.DataFrame:
    if group != "all":
        frame = frame.filter(pl.col("time_series_type") == group)
    limits = {
        "all": pl.lit(True),
        "days 0 to 2": pl.col("lead_hours") < 72,
        "days 3 to 10": pl.col("lead_hours").is_between(72, 264, closed="left"),
        "days 11 onward": pl.col("lead_hours") >= 264,
    }
    return frame.filter(limits[band])


def _read_runs(*, runs: list[dict[str, Any]]) -> dict[str, dict[int, pl.DataFrame]]:
    found: dict[str, dict[int, pl.DataFrame]] = {}
    for run in runs:
        root = Path(run["path"])
        manifest = json.loads((root / "manifest.json").read_text())
        completion = json.loads((root / "complete.json").read_text())
        for name in ("scores.parquet", "features.json"):
            if _sha256(path=root / name) != completion["files"][name]:
                raise ValueError("completed fit input hash mismatch")
        seed = int(run["seed"])
        if int(manifest["seed"]) != seed:
            raise ValueError("run specification and fit manifest disagree about seed")
        frame = pl.read_parquet(root / "scores.parquet").sort(*KEYS)
        if frame.is_empty() or any(frame[column].null_count() for column in frame.columns):
            raise ValueError("score inputs are empty or contain nulls")
        numeric = [*METRICS, "actual", "effective_capacity_mw", "lead_hours"]
        if (
            frame.select(
                pl.any_horizontal(~pl.col(*numeric).cast(pl.Float64).is_finite()).any()
            ).item()
            or frame.filter(pl.col("effective_capacity_mw") <= 0).height
        ):
            raise ValueError("score inputs contain nonfinite metrics")
        expected_lead = (
            pl.col("valid_time") - pl.col("power_fcst_init_time")
        ).dt.total_seconds() / 3600
        if frame.filter((pl.col("lead_hours") - expected_lead).abs() > 1e-6).height:
            raise ValueError("score lead does not match forecast timestamps")
        if frame.select(*KEYS).is_duplicated().any():
            raise ValueError("scores contain duplicate forecast keys")
        if frame.filter(pl.col("valid_time") >= pl.datetime(2026, 7, 1, time_zone="UTC")).height:
            raise ValueError("scores exceed the study window")
        if seed in found.setdefault(run["label"], {}):
            raise ValueError("duplicate fitting seed for a label")
        found[run["label"]][seed] = frame
    return found


def _paired(*, left: dict[int, pl.DataFrame], right: dict[int, pl.DataFrame]) -> None:
    if left.keys() != right.keys():
        raise ValueError("a contrast requires the same fitting seeds")
    reference = right[min(right)]
    checked = [*left.values(), *right.values()]
    columns = [*KEYS, "effective_capacity_mw", "actual", "time_series_type", "lead_hours"]
    for frame in checked:
        if not frame.select(columns).equals(reference.select(columns)):
            raise ValueError("paired contrast has different rows, targets, types, or capacities")


def _training_parameters(*, root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    completion = json.loads((root / "complete.json").read_text())
    relative = "models/meta.json" if manifest["recipe"] == "production" else "models.json"
    if _sha256(path=root / relative) != completion["files"][relative]:
        raise ValueError("training-parameter source fingerprint mismatch")
    stored = json.loads((root / relative).read_text())
    if manifest["recipe"] == "production":
        return stored["model_params"]
    model_config = stored[0].get("booster_runtime_config") or stored[0]["effective_xgboost_config"]
    learner = model_config["learner"]
    safe_names = (
        "eta",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "lambda",
        "alpha",
    )
    tree = learner["gradient_booster"]["tree_train_param"]
    return {
        **{name: tree[name] for name in safe_names if name in tree},
        "objective": learner["learner_train_param"]["objective"],
        **learner["objective"].get("quantile_loss_param", {}),
        "seed": manifest["seed"],
        "device": manifest["device"],
        "rounds_min": min(model["rounds"] for model in stored),
        "rounds_max": max(model["rounds"] for model in stored),
    }


def _monthly(*, frames: dict[int, pl.DataFrame], metric: str) -> MonthlyLosses:
    seeds = tuple(sorted(frames))
    first = frames[seeds[0]]
    series = tuple(sorted(first["time_series_id"].unique().to_list()))
    months = tuple(sorted(first["valid_time"].dt.strftime("%Y-%m").unique().to_list()))
    totals = np.zeros((len(seeds), len(series), len(months)))
    counts = np.zeros_like(totals)
    site_index = {value: index for index, value in enumerate(series)}
    month_index = {value: index for index, value in enumerate(months)}
    for seed_index, seed in enumerate(seeds):
        grouped = (
            frames[seed]
            .with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
            .group_by("time_series_id", "month")
            .agg(total=pl.col(metric).sum(), count=pl.len())
        )
        for site, month, total, count in grouped.iter_rows():
            totals[seed_index, site_index[site], month_index[month]] = total
            counts[seed_index, site_index[site], month_index[month]] = count
    return MonthlyLosses(totals=totals, counts=counts, months=months, seeds=seeds)


def _means(*, totals: np.ndarray, counts: np.ndarray, axis: int) -> np.ndarray:
    values = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    return np.nanmean(values, axis=axis)


def _draws(*, monthly: MonthlyLosses) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed=BOOTSTRAP_SEED)
    months = generator.integers(len(monthly.months), size=(RESAMPLES, len(monthly.months)))
    seeds = generator.integers(len(monthly.seeds), size=RESAMPLES)
    weights = np.stack([np.bincount(row, minlength=len(monthly.months)) for row in months])
    return seeds, weights


def _estimate(*, monthly: MonthlyLosses) -> float:
    per_seed = _means(totals=monthly.totals.sum(axis=2), counts=monthly.counts.sum(axis=2), axis=1)
    return float(per_seed.mean())


def _resampled(*, monthly: MonthlyLosses, seeds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # einsum avoids materialising a resample-by-series-by-month copy of the loss arrays.
    totals = np.einsum("rsm,rm->rs", monthly.totals[seeds], weights)
    counts = np.einsum("rsm,rm->rs", monthly.counts[seeds], weights)
    return _means(totals=totals, counts=counts, axis=1)


def _support_diagnostics(
    *, monthly: MonthlyLosses, seeds: np.ndarray, weights: np.ndarray
) -> dict[str, int]:
    supported_months = (monthly.counts > 0).sum(axis=2)
    resampled_counts = np.einsum("rsm,rm->rs", monthly.counts[seeds], weights)
    dropped = (resampled_counts == 0).sum(axis=1)
    return {
        "min_supported_months": int(supported_months.min()),
        "max_supported_months": int(supported_months.max()),
        "series_with_missing_months": int(
            (supported_months < len(monthly.months)).sum(axis=1).max()
        ),
        "bootstrap_draws_dropping_series": int((dropped > 0).sum()),
        "max_dropped_series_per_draw": int(dropped.max()),
    }


def _contrast(*, left: MonthlyLosses, right: MonthlyLosses) -> dict[str, Any]:
    if (
        left.months != right.months
        or left.seeds != right.seeds
        or not np.array_equal(left.counts, right.counts)
    ):
        raise ValueError("monthly pairing mismatch")
    seeds, weights = _draws(monthly=left)
    draws = _resampled(monthly=left, seeds=seeds, weights=weights) - _resampled(
        monthly=right, seeds=seeds, weights=weights
    )
    monthly_difference = (
        _means(totals=left.totals, counts=left.counts, axis=1)
        - _means(totals=right.totals, counts=right.counts, axis=1)
    ).mean(axis=0)
    n_months = len(monthly_difference)
    if n_months < 2:
        raise ValueError("inference requires at least two validation months")
    half_width = (
        float(stats.t.ppf(0.975, df=n_months - 1))
        * float(monthly_difference.std(ddof=1))
        / np.sqrt(n_months)
    )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "difference": 100 * (_estimate(monthly=left) - _estimate(monthly=right)),
        "lower95": 100 * float(lower),
        "upper95": 100 * float(upper),
        "equal_month_difference": 100 * float(monthly_difference.mean()),
        "t_lower95": 100 * (float(monthly_difference.mean()) - half_width),
        "t_upper95": 100 * (float(monthly_difference.mean()) + half_width),
        "months_better": int((monthly_difference < 0).sum()),
        "months": n_months,
        "seeds": len(left.seeds),
        **_support_diagnostics(monthly=left, seeds=seeds, weights=weights),
    }


def _table(*, frame: pl.DataFrame) -> str:
    columns = frame.columns
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.iter_rows():
        formatted = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _absolute_scores(*, runs: dict[str, dict[int, pl.DataFrame]]) -> list[dict[str, Any]]:
    levels = []
    for label, frames in runs.items():
        _paired(left=frames, right=frames)
        for group in GROUPS:
            for band in BANDS:
                filtered = {
                    seed: _filter(frame=frame, group=group, band=band)
                    for seed, frame in frames.items()
                }
                if any(frame.is_empty() for frame in filtered.values()):
                    continue
                for metric in METRICS:
                    monthly = _monthly(frames=filtered, metric=metric)
                    seeds, weights = _draws(monthly=monthly)
                    lower, upper = np.quantile(
                        _resampled(monthly=monthly, seeds=seeds, weights=weights), [0.025, 0.975]
                    )
                    levels.append(
                        {
                            "label": label,
                            "group": group,
                            "lead_band": band,
                            "metric": metric,
                            "value": 100 * _estimate(monthly=monthly),
                            "lower95": 100 * float(lower),
                            "upper95": 100 * float(upper),
                            "seeds": len(frames),
                            "months": len(monthly.months),
                            "series": monthly.totals.shape[1],
                            "rows_per_seed": filtered[min(filtered)].height,
                            "provisional": len(frames) != 3 or len(monthly.months) < 6,
                            "quantile_probability": (
                                (int(metric.split("_")[1]) + 0.5) / 19
                                if metric.startswith("pinball_")
                                else None
                            ),
                            **_support_diagnostics(monthly=monthly, seeds=seeds, weights=weights),
                        }
                    )
    return levels


def _check_training_pair(*, a_root: Path, b_root: Path, match_counts: bool) -> None:
    a_manifest = json.loads((a_root / "manifest.json").read_text())
    b_manifest = json.loads((b_root / "manifest.json").read_text())
    a_training = a_manifest["input_manifest"]["frames"]["train.parquet"]
    b_training = b_manifest["input_manifest"]["frames"]["train.parquet"]
    if a_training["key_sha256"] != b_training["key_sha256"]:
        raise ValueError("contrast has different raw training forecast keys")
    if a_manifest["recipe"] == b_manifest["recipe"] or match_counts:
        for root in (a_root, b_root):
            completion = json.loads((root / "complete.json").read_text())
            if (
                _sha256(path=root / "training_counts.parquet")
                != completion["files"]["training_counts.parquet"]
            ):
                raise ValueError("cleaned training counts fingerprint mismatch")
        a_counts = pl.read_parquet(a_root / "training_counts.parquet").sort("time_series_id")
        b_counts = pl.read_parquet(b_root / "training_counts.parquet").sort("time_series_id")
        if not a_counts.equals(b_counts):
            raise ValueError("contrast requires matching cleaned training counts")


def _interaction_inputs(
    *,
    specification: dict[str, Any],
    runs: dict[str, dict[int, pl.DataFrame]],
    fit_roots: dict[tuple[str, int], Path],
) -> dict[str, dict[int, pl.DataFrame]]:
    roles = ("combined", "radiation", "wind", "old")
    if len({specification[role] for role in roles}) != 4:
        raise ValueError("an interaction requires four distinct treatment labels")
    arms = {role: runs[specification[role]] for role in roles}
    identities = set()
    for role, frames in arms.items():
        _paired(left=arms["combined"], right=frames)
        for seed in frames:
            root = fit_roots[(specification[role], seed)]
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest["input_manifest"]["context"]["pipeline_tag"] != role:
                raise ValueError("interaction role differs from the input pipeline treatment")
            identities.add(
                tuple(manifest[name] for name in ("recipe", "setting", "device", "padding"))
            )
            _check_training_pair(
                a_root=fit_roots[(specification["combined"], seed)],
                b_root=root,
                match_counts=True,
            )
    if len(identities) != 1:
        raise ValueError("interaction requires the same recipe, setting, device and padding")
    return arms


def _interaction_monthly(*, arms: dict[str, MonthlyLosses]) -> MonthlyLosses:
    reference = arms["combined"]
    if any(
        arm.months != reference.months
        or arm.seeds != reference.seeds
        or not np.array_equal(arm.counts, reference.counts)
        for arm in arms.values()
    ):
        raise ValueError("four-treatment monthly pairing mismatch")
    return MonthlyLosses(
        totals=reference.totals
        - arms["radiation"].totals
        - arms["wind"].totals
        + arms["old"].totals,
        counts=reference.counts,
        months=reference.months,
        seeds=reference.seeds,
    )


def _interaction_rows(
    *,
    specification: dict[str, Any],
    runs: dict[str, dict[int, pl.DataFrame]],
    fit_roots: dict[tuple[str, int], Path],
) -> list[dict[str, Any]]:
    arms = _interaction_inputs(specification=specification, runs=runs, fit_roots=fit_roots)
    output = []
    for group in specification.get("groups", GROUPS):
        for band in specification.get("bands", BANDS):
            selected = {
                role: {
                    seed: _filter(frame=frame, group=group, band=band)
                    for seed, frame in frames.items()
                }
                for role, frames in arms.items()
            }
            if any(frame.is_empty() for frames in selected.values() for frame in frames.values()):
                continue
            for metric in specification.get("metrics", METRICS):
                effect = _interaction_monthly(
                    arms={
                        role: _monthly(frames=frames, metric=metric)
                        for role, frames in selected.items()
                    }
                )
                zero = MonthlyLosses(
                    totals=np.zeros_like(effect.totals),
                    counts=effect.counts,
                    months=effect.months,
                    seeds=effect.seeds,
                )
                output.append(
                    {
                        "contrast": specification["label"],
                        "kind": specification["kind"],
                        "left": f"{specification['combined']} + {specification['old']}",
                        "right": f"{specification['radiation']} + {specification['wind']}",
                        "group": group,
                        "lead_band": band,
                        "metric": metric,
                        "provisional": len(effect.seeds) != 3,
                        **_contrast(left=effect, right=zero),
                    }
                )
    return output


def _support_note(*, interactions: bool) -> str:
    text = (
        "A bootstrap draw omits a series when none of its supported months were drawn; "
        "remaining series retain equal weight. Support columns report month coverage, "
        "draws omitting any series and the largest number omitted in a draw. "
        "series_with_missing_months is the largest such count across fitting seeds.\n"
    )
    if interactions:
        text += (
            "Four-treatment interactions are combined - radiation - wind + old. "
            "Their intervals use that paired monthly loss combination against zero, "
            "with original case counts and shared month/seed draws.\n"
        )
    return text


def main() -> None:
    """Write every absolute score and prespecified paired contrast into a reproducible report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("report output already exists; use a new immutable directory")
    spec = json.loads(args.spec.read_text())
    runs = _read_runs(runs=spec["runs"])
    contrasts = []
    features = []
    public_features = []
    parameters = []
    fit_roots = {(run["label"], int(run["seed"])): Path(run["path"]) for run in spec["runs"]}
    for run in spec["runs"]:
        manifest = json.loads((Path(run["path"]) / "manifest.json").read_text())
        features.append({"label": run["label"], "seed": run["seed"], "manifest": manifest})
        parameters.append(
            {
                "label": run["label"],
                "seed": run["seed"],
                "training_parameters": json.dumps(
                    _training_parameters(root=Path(run["path"]), manifest=manifest), sort_keys=True
                ),
            }
        )
        columns = json.loads((Path(run["path"]) / "features.json").read_text())
        public_features.extend(
            {
                "label": run["label"],
                "seed": run["seed"],
                "series_type": kind,
                "count": len(names),
                "features": ", ".join(names),
            }
            for kind, names in columns.items()
        )
    levels = _absolute_scores(runs=runs)
    for comparison in spec["contrasts"]:
        left, right = runs[comparison["left"]], runs[comparison["right"]]
        _paired(left=left, right=right)
        for seed in left:
            _check_training_pair(
                a_root=fit_roots[(comparison["left"], seed)],
                b_root=fit_roots[(comparison["right"], seed)],
                match_counts=bool(comparison.get("match_training_counts", False)),
            )
        for group in comparison.get("groups", GROUPS):
            for band in comparison.get("bands", BANDS):
                a = {
                    seed: _filter(frame=frame, group=group, band=band)
                    for seed, frame in left.items()
                }
                b = {
                    seed: _filter(frame=frame, group=group, band=band)
                    for seed, frame in right.items()
                }
                if any(frame.is_empty() for frame in [*a.values(), *b.values()]):
                    continue
                contrasts.extend(
                    {
                        "contrast": comparison["label"],
                        "kind": comparison["kind"],
                        "left": comparison["left"],
                        "right": comparison["right"],
                        "group": group,
                        "lead_band": band,
                        "metric": metric,
                        "provisional": len(left) != 3,
                        **_contrast(
                            left=_monthly(frames=a, metric=metric),
                            right=_monthly(frames=b, metric=metric),
                        ),
                    }
                    for metric in comparison.get("metrics", METRICS)
                )
    for interaction in spec.get("interactions", []):
        contrasts.extend(
            _interaction_rows(specification=interaction, runs=runs, fit_roots=fit_roots)
        )
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "specification": spec,
        "script_sha256": _sha256(path=Path(__file__)),
        "specification_sha256": _sha256(path=args.spec),
        "inputs": {
            str(Path(run["path"]) / name): _sha256(path=Path(run["path"]) / name)
            for run in spec["runs"]
            for name in ("manifest.json", "complete.json", "scores.parquet", "features.json")
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resamples": RESAMPLES,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    level_frame, contrast_frame = pl.DataFrame(levels), pl.DataFrame(contrasts)
    level_frame.write_parquet(args.output / "levels.parquet")
    contrast_frame.write_parquet(args.output / "contrasts.parquet")
    # Manifests may contain private model keys; this audit stays in the private output directory.
    (args.output / "fit-manifests.json").write_text(json.dumps(features, indent=2) + "\n")
    text = "# Fixed-pipeline forecast report\n\n"
    text += (
        "Losses are percentages of capacity; coverage is percent of cases. "
        "Differences are percentage points.\n"
        "For coverage, months_better counts negative differences, not improved calibration.\n"
        "CRPS is diagnostic fair-CRPS; cross-recipe comparisons use 19-midpoint pinball.\n"
        "Quantiles use empirical inverse-CDF ranks ceil(n*q)-1, without extrapolation.\n"
    )
    text += (
        "Negative loss differences favour the first recipe. Coverage differences are descriptive.\n"
    )
    text += "Intervals resample whole calendar months and one paired fitting seed, 2,000 times.\n"
    text += _support_note(interactions=bool(spec.get("interactions")))
    text += (
        "The t-interval uses equal-month means, a different estimand "
        "from the equal-series score.\n\n"
    )
    text += "## Absolute scores\n\n" + _table(frame=level_frame) + "\n\n"
    text += "## Paired contrasts\n\n" + _table(frame=contrast_frame) + "\n"
    text += "\n## Features\n\n" + _table(frame=pl.DataFrame(public_features)) + "\n"
    text += "\n## Training parameters\n\n" + _table(frame=pl.DataFrame(parameters)) + "\n"
    text += (
        "\nProduction training parameters come from its verified models/meta.json. "
        "The earlier models.json effective_xgboost_config records defaults after Booster reload, "
        "not production training settings; it does not invalidate the fitted predictions.\n"
    )
    (args.output / "report.md").write_text(text)
    print(f"Wrote {len(levels)} aggregate scores and {len(contrasts)} paired contrasts")


if __name__ == "__main__":
    main()
