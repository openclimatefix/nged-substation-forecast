# ruff: noqa: E501, PLR0915
"""Realism checks on the injector, run before any network is trained.

Check 1: a classifier trained to tell injected events from held-out real events, on features measured
from the residual alone, with area under the curve pooled over the three folds. Check 2: the recall
of a rule-based stand-in (a 24-hour step above RO-NC's threshold) on injected events and on real
events. The injector is fitted without the fold being scored.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00189-*/e00195_realism.py
"""

from __future__ import annotations

import json
import sys
from typing import Final

import numpy as np
import polars as pl
from e00195_common import (
    GRID_LENGTH,
    TUNING_MONTHS,
    WORK_DIR,
    load_panel,
    noise_units,
    pair_folds,
    real_transfers,
    stamp_months,
)
from e00195_injector import fit_injector
from e00195_worlds import Calendar, disjoint_world, inject, step_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_INJECTED: Final[int] = 300
SEED: Final[int] = 189
RONC_THRESHOLD: Final[float] = 4.35
"""RO-NC's alarm threshold on its 24-hour step statistic, in noise units (experiment 130)."""
STEP_WINDOW: Final[int] = 48
ONSET_TOLERANCE: Final[int] = 4


def level_shift(*, values: np.ndarray) -> np.ndarray:
    """Mean of the next 24 hours minus the mean of the previous 24 hours, NaN under 75% present."""
    total = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
    count = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    t = np.arange(GRID_LENGTH)
    lo = np.clip(t - STEP_WINDOW, 0, GRID_LENGTH)
    hi = np.clip(t + STEP_WINDOW, 0, GRID_LENGTH)
    n_before, n_after = count[t] - count[lo], count[hi] - count[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = (total[hi] - total[t]) / n_after - (total[t] - total[lo]) / n_before
    shift[(n_before < 0.75 * STEP_WINDOW) | (n_after < 0.75 * STEP_WINDOW)] = np.nan
    return shift


def standin_peaks(
    *, residual: dict[int, np.ndarray], scales: dict[int, float]
) -> dict[int, np.ndarray]:
    """Stamps per series where the 24-hour step exceeds RO-NC's threshold and is a local peak."""
    out = {}
    for s, values in residual.items():
        if scales[s] != scales[s]:
            out[s] = np.array([], dtype=int)
            continue
        z = np.nan_to_num(np.abs(level_shift(values=values) / scales[s]))
        hot = np.flatnonzero(z > RONC_THRESHOLD)
        keep = [t for t in hot if z[t] == z[max(t - STEP_WINDOW, 0) : t + STEP_WINDOW + 1].max()]
        out[s] = np.array(keep, dtype=int)
    return out


def standin_scales(*, residual: dict[int, np.ndarray]) -> dict[int, float]:
    """The odd-month robust scale of each series' 24-hour step statistic."""
    months = stamp_months()
    odd = np.isin(months, list(TUNING_MONTHS))
    out = {}
    for s, values in residual.items():
        shift = level_shift(values=values)
        chosen = shift[odd & np.isfinite(shift)]
        out[s] = (
            float(1.4826 * np.median(np.abs(chosen - np.median(chosen))))
            if chosen.size > 500
            else np.nan
        )
    return out


def recall(*, peaks: dict[int, np.ndarray], series: tuple[int, ...], onset: int) -> bool:
    """True when the stand-in has a peak within 2 hours of `onset` on one of the event's series."""
    return any(
        peaks.get(s, np.array([])).size and (np.abs(peaks[s] - onset) <= ONSET_TOLERANCE).any()
        for s in series
    )


def main() -> None:
    """Run both checks and write the feature table and a JSON summary."""
    panel = load_panel()
    residual, power = panel["residual"], panel["power"]
    sigma = noise_units(residual=residual)
    calendar = Calendar()
    folds = pair_folds()
    real = real_transfers().with_columns(fold=pl.col("pair").replace_strict(folds))
    scales = standin_scales(residual=residual)
    real_peaks = standin_peaks(residual=residual, scales=scales)
    rng = np.random.default_rng(SEED)
    rows = []
    for row in real.iter_rows(named=True):
        feats = step_features(
            residual=residual,
            sigma=sigma,
            series=tuple(row["series"]),
            onset=row["onset"],
            end=row["end"],
            calendar=calendar,
        )
        if feats is None:
            continue
        rows.append(
            {
                **feats,
                "real": 1,
                "fold": row["fold"],
                "recalled": recall(
                    peaks=real_peaks, series=tuple(row["series"]), onset=row["onset"]
                ),
                "ronc_onset_found": row["onset_found_ronc"],
                "long": row["hours"] > 24,
            }
        )
    n_real = len(rows)
    for fold in range(3):
        held_out = frozenset(p for p, f in folds.items() if f == fold)
        fit = fit_injector(held_out_pairs=held_out, sigma=sigma)
        events = disjoint_world(
            fit=fit,
            n_events=N_INJECTED,
            residual=residual,
            power=power,
            sigma=sigma,
            calendar=calendar,
            rng=rng,
            allow_months=None,
        )
        injected = inject(residual=residual, events=events)
        peaks = standin_peaks(residual=injected, scales=scales)
        for placed in events:
            feats = step_features(
                residual=injected,
                sigma=sigma,
                series=placed.series,
                onset=placed.onset,
                end=placed.end,
                calendar=calendar,
            )
            if feats is None:
                continue
            rows.append(
                {
                    **feats,
                    "real": 0,
                    "fold": fold,
                    "recalled": recall(peaks=peaks, series=placed.series, onset=placed.onset),
                    "ronc_onset_found": False,
                    "long": placed.end - placed.onset > 48,
                }
            )
    table = pl.DataFrame(rows)
    table.write_parquet(WORK_DIR / "realism_features.parquet")
    names = [
        c
        for c in table.columns
        if c not in {"real", "fold", "recalled", "ronc_onset_found", "long"}
    ]
    results = {}
    for kind in ("logistic", "boosted"):
        scores, labels = [], []
        for fold in range(3):
            train, test = table.filter(pl.col("fold") != fold), table.filter(pl.col("fold") == fold)
            weights = np.where(
                train["real"].to_numpy() == 1,
                (train["real"] == 0).sum() / max((train["real"] == 1).sum(), 1),  # ty: ignore[unsupported-operator]
                1.0,
            )
            if kind == "logistic":
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
                model.fit(
                    train.select(names).to_numpy(),
                    train["real"].to_numpy(),
                    logisticregression__sample_weight=weights,
                )
            else:
                model = HistGradientBoostingClassifier(
                    max_depth=2,
                    max_iter=60,
                    learning_rate=0.1,
                    min_samples_leaf=10,
                    random_state=SEED,
                )
                model.fit(
                    train.select(names).to_numpy(), train["real"].to_numpy(), sample_weight=weights
                )
            scores += model.predict_proba(test.select(names).to_numpy())[:, 1].tolist()
            labels += test["real"].to_list()
        scores_a, labels_a = np.array(scores), np.array(labels)
        boots = []
        for _ in range(500):
            pick = rng.integers(0, len(labels_a), len(labels_a))
            if labels_a[pick].min() != labels_a[pick].max():
                boots.append(roc_auc_score(labels_a[pick], scores_a[pick]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results[kind] = (float(roc_auc_score(labels_a, scores_a)), float(lo), float(hi))
    auc, lo, hi = max(results.values())
    long_real = table.filter(pl.col("real") == 1, pl.col("long"))
    all_real = table.filter(pl.col("real") == 1)
    injected_t = table.filter(pl.col("real") == 0)
    summary = {
        "n_real_with_features": n_real,
        "n_injected": injected_t.height,
        "auc": float(auc),
        "auc_95": [float(lo), float(hi)],
        "auc_by_classifier": results,
        "standin_recall_real_all": float(all_real["recalled"].mean()),  # ty: ignore[invalid-argument-type]
        "standin_recall_real_long": float(long_real["recalled"].mean()),  # ty: ignore[invalid-argument-type]
        "ronc_onset_found_real_all": float(all_real["ronc_onset_found"].mean()),  # ty: ignore[invalid-argument-type]
        "ronc_onset_found_real_long": float(long_real["ronc_onset_found"].mean()),  # ty: ignore[invalid-argument-type]
        "standin_recall_injected_all": float(injected_t["recalled"].mean()),  # ty: ignore[invalid-argument-type]
        "standin_recall_injected_long": float(injected_t.filter(pl.col("long"))["recalled"].mean()),  # ty: ignore[invalid-argument-type]
    }
    for name in names:
        summary[f"median_{name}"] = {
            "real": float(all_real[name].median()),  # ty: ignore[invalid-argument-type]
            "injected": float(injected_t[name].median()),  # ty: ignore[invalid-argument-type]
        }
    tag = sys.argv[1] if len(sys.argv) > 1 else "default"
    (WORK_DIR / f"realism_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
