r"""Experiment 358: the known-answer test of the MVA fold at five folded signed-MW Primaries.

Each job folds one Primary's signed power to apparent power at one injection level and one reactive
setting, fits the signed series (arm S), the folded reading as if signed (arm N), and the folded
reading through the meter (arm L0, experiment 110's `fit_gam_through_meter`) at one cut, and scores
the following year on past weather. Each job writes its own JSON file atomically, so a crashed
worker is rerun alone. Only ratios and accuracies leave the private folder, because the injected
farm is a metered generator.

    PYTHONPATH=ml-experiments/00110-differentiable-physics-5-embedded-generation-and-capacity \\
        uv run --no-sync python ml-experiments/00358-*/e00358_known_answer.py --workers 5
"""

import argparse
import json
import multiprocessing
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import e00110_arms as arms  # ty: ignore[unresolved-import]
import e00110_common as common  # ty: ignore[unresolved-import]
import e00110_inject as inject  # ty: ignore[unresolved-import]
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import fit_gam, pinned_solar_points, predict_gam
from studies.dp_meter import fit_gam_through_meter, predict_meter
from studies.dp_operator import tan_phi_from_power_factor
from studies.dp_pinned_capacity import pinned_capacity, read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 358
DATA_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00358-mva-fold-known-answer-at-folded-mw-primaries"
)
OUT_DIR: Final[Path] = DATA_DIR / "work" / "known_answer"
SITES: Final[tuple[str, ...]] = common.DECISION_SITES
CUTS: Final[dict[str, datetime]] = {
    "replicate": common.REPLICATE_CUT,
    "scored": common.SCORED_CUT,
}
LEVELS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0)
POWER_FACTOR: Final[float] = 0.97
REACTIVE: Final[tuple[str, ...]] = ("Q0", "Q1")
MIN_TRUE_POINTS: Final[float] = 5.0
SUNNY_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
SUNNY_GHI_W_M2: Final[float] = 500.0
FARM: Final[str] = "A"


def _sunny(*, cache: FeatureCache, site: str, index: np.ndarray) -> np.ndarray:
    """Return True where a row is a sunny daylight half-hour: April to September, 10:00 to 13:30."""
    start = cache.end_times[index] - np.timedelta64(30, "m")
    minutes = (start.astype("datetime64[m]").astype(np.int64)) % 1440
    months = start.astype("datetime64[M]").astype(np.int64) % 12 + 1
    ghi = cache.sites[site].ghi_w_m2[index]
    return (
        np.isin(months, SUNNY_MONTHS)
        & (minutes >= 600)
        & (minutes <= 810)
        & (np.nan_to_num(ghi, nan=0.0) >= SUNNY_GHI_W_M2)
    )


def _balanced_accuracy(*, truth_positive: np.ndarray, predicted_positive: np.ndarray) -> float:
    """Return the mean of the true-positive and true-negative rates, NaN if a class is empty."""
    if truth_positive.all() or not truth_positive.any():
        return float("nan")
    return 0.5 * (
        float(predicted_positive[truth_positive].mean())
        + float((~predicted_positive[~truth_positive]).mean())
    )


def run_job(job: tuple[str, str, float, str]) -> Path:  # noqa: PLR0915
    """Run one (site, cut, level, reactive) job and return its result file."""
    site, cut_name, level, reactive = job
    path = OUT_DIR / f"{site}_{cut_name}_L{level}_{reactive}.json"
    if path.exists():
        return path
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(2)
    cut = CUTS[cut_name]
    cache = common.load_cache()
    vintages = read_vintages()
    part_4 = arms.read_part_4_tuned()
    spec_b = arms.base_specs(part_4=part_4)["b"]
    spec = replace(spec_b, capacity="fitted")
    window = arms.arm_definitions(part_4=part_4, tuned=arms.read_tuned())["lq"].window
    features = cache.sites[site]
    pinned = pinned_capacity(vintages=vintages, table=features.table, cut=cut)
    fit_b = fit_gam(
        cache=cache,
        site=site,
        specs=[spec_b],
        cut=cut,
        window=window,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    usable = np.flatnonzero(np.isfinite(features.y) & np.isfinite(features.pv_unit_shape))
    usable = usable[cache.end_times[usable] <= common.to_64(moment=cut)]
    demand = np.full(len(cache.end_times), np.nan)
    rows = np.flatnonzero(np.isfinite(features.y) & np.isfinite(features.pv_unit_shape))
    demand[rows] = predict_gam(cache=cache, fit=fit_b, index=rows) + pinned_solar_points(
        cache=cache, features=features, pinned=pinned, index=rows
    )
    scale = inject.midday_median_absolute(cache=cache, site=site, before=cut)
    signed_cache = (
        cache
        if level == 0.0
        else inject.inject_farm(cache=cache, site=site, farm=FARM, capacity_points=level * scale)
    )
    signed = signed_cache.sites[site].y
    tan_phi = 0.0 if reactive == "Q0" else tan_phi_from_power_factor(power_factor=POWER_FACTOR)
    folded_cache = inject.inject_meter(
        cache=signed_cache, site=site, signed=signed, demand=demand, tan_phi=tan_phi
    )
    folded = folded_cache.sites[site].y

    def capacity(fit: Any) -> float:
        return float(fit.capacity_points()[fit.last_season_year])

    fit_s = fit_gam(
        cache=signed_cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=window,
        season_years=common.SEASON_YEARS,
    )
    fit_n = fit_gam(
        cache=folded_cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=window,
        season_years=common.SEASON_YEARS,
    )
    meter = fit_gam_through_meter(
        cache=folded_cache,
        site=site,
        spec=spec,
        cut=cut,
        window=window,
        season_years=common.SEASON_YEARS,
        qmode="zero",
    )
    index = common.scoring_index(
        cache=folded_cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    keep = np.isfinite(signed[index]) & np.isfinite(demand[index]) & ~features.switching_mask[index]
    index = index[keep]
    truth = signed[index]
    reading = folded[index]
    pred_s = predict_gam(cache=signed_cache, fit=fit_s, index=index)
    pred_n = predict_gam(cache=folded_cache, fit=fit_n, index=index)
    pred_l0 = predict_meter(cache=folded_cache, meter=meter, index=index)
    sunny = _sunny(cache=cache, site=site, index=index)
    clear = np.abs(truth) >= MIN_TRUE_POINTS
    result: dict[str, Any] = {
        "site": site,
        "cut": cut_name,
        "level": level,
        "reactive": reactive,
        "n_rows": len(index),
        "n_sunny_rows": int(sunny.sum()),
        "reverse_flow_share_all": float((truth < 0).mean()),
        "reverse_flow_share_sunny": float((truth[sunny] < 0).mean()) if sunny.any() else None,
        "injected_points": float(level * scale),
        "capacity_s": capacity(fit_s),
        "capacity_n": capacity(fit_n),
        "capacity_l0": capacity(meter.fit),
        "reverse_flow_share_train_l0": float(meter.reverse_flow_share),
        "converged": bool(meter.converged),
        "starts_capacity_spread": float(meter.capacity_spread),
    }
    for label, mask in (
        ("sunny", sunny & clear),
        ("all", clear),
        ("sunny_reverse", sunny & clear & (truth < 0)),
    ):
        if mask.sum() == 0:
            continue
        positive = truth[mask] > 0
        result[f"bal_acc_l0_{label}"] = _balanced_accuracy(
            truth_positive=positive, predicted_positive=pred_l0.real_power[mask] > 0
        )
        result[f"bal_acc_n_{label}"] = _balanced_accuracy(
            truth_positive=positive, predicted_positive=pred_n[mask] > 0
        )
        result[f"bal_acc_s_{label}"] = _balanced_accuracy(
            truth_positive=positive, predicted_positive=pred_s[mask] > 0
        )
        for arm, predicted in (("l0", pred_l0.real_power), ("n", pred_n), ("s", pred_s)):
            result[f"acc_{arm}_{label}"] = float(((predicted[mask] > 0) == positive).mean())
        result[f"n_{label}"] = int(mask.sum())
    for label, mask in (("all", np.ones(len(index), bool)), ("sunny", sunny)):
        if mask.sum() == 0:
            continue
        result[f"mae_reading_l0_{label}"] = float(
            np.abs(pred_l0.apparent_power[mask] - reading[mask]).mean()
        )
        result[f"mae_reading_n_{label}"] = float(np.abs(pred_n[mask] - reading[mask]).mean())
        result[f"mae_reading_s_abs_{label}"] = float(
            np.abs(np.abs(pred_s[mask]) - reading[mask]).mean()
        )
    negative = truth < 0
    if negative.any():
        result["mae_signed_l0_negative"] = float(
            np.abs(pred_l0.real_power[negative] - truth[negative]).mean()
        )
        result["mae_signed_s_negative"] = float(np.abs(pred_s[negative] - truth[negative]).mean())
        result["mae_signed_n_negative"] = float(np.abs(pred_n[negative] - truth[negative]).mean())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=float) + "\n")
    temporary.replace(path)
    return path


def main() -> None:
    """Run every job not yet done, in a spawned pool."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--sites", nargs="*", default=list(SITES))
    parser.add_argument("--cuts", nargs="*", default=list(CUTS))
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "2"
    common.load_cache()
    jobs = [
        (site, cut, level, reactive)
        for cut in arguments.cuts
        for level in LEVELS
        for reactive in REACTIVE
        for site in arguments.sites
    ]
    context = multiprocessing.get_context("spawn")
    with context.Pool(arguments.workers) as pool:
        for finished in pool.imap_unordered(run_job, jobs):
            print("done", finished.name, flush=True)


if __name__ == "__main__":
    main()
