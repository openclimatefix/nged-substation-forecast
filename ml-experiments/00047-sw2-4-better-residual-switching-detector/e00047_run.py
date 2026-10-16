r"""Match thresholds on odd months and score the detectors on injected and real transfers.

Stages, in order. Each writes per-arm files to the output folder and skips nothing, so rerun a
stage to refresh it.

- `harness-check`: the stored `H0.5` panel through this code at experiment 43's thresholds must
  reproduce experiment 43's alarm counts and every injected-event outcome. Stop if it does not.
- `tune`: for each arm-seed, the noise scales, the matched thresholds, the threshold curves, the
  real alarms at the matched and at experiment 43's thresholds, and the real labelled transfers.
- `score`: for each arm-seed, the outcome of every injected transfer at the matched thresholds.

    uv run python ml-experiments/00047-*/e00047_run.py harness-check
    uv run python ml-experiments/00047-*/e00047_run.py tune
    uv run python ml-experiments/00047-*/e00047_run.py score
    uv run python ml-experiments/00047-*/e00047_run.py tune score --names HOL_s0 --out SCRATCH \\
        --fits SCRATCH/fits --fallback                   # smoke test
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00047_common import EXPERIMENT_43_DIR, WORK_DIR, near_log_masks
from e00047_detect import (
    EXPLAIN_PAD,
    OLD_THRESHOLDS,
    SAMPLE_NAMES,
    SETTINGS,
    TARGET_COUNTS,
    ArmInputs,
    ArmPanel,
    alarm_rows,
    arm_name,
    detect_all,
    drawn_events,
    init_worker,
    load_arm_panel,
    odd_unexplained,
    real_events,
    score_batch,
    score_real,
    stored_panel,
    tune_setting,
)
from e00047_fit import FITS_DIR, FITTED_JOBS
from e00047_inject import batches, events_frame

ALL_NAMES: Final[tuple[str, ...]] = (
    "H0.5_s0",
    *(arm_name(arm=arm, seed=seed) for arm, seed in FITTED_JOBS),
)
N_WORKERS: Final[int] = 4
HARNESS_ALARM_COUNTS: Final[dict[str, int]] = {"D0_8": 432, "D1_8": 444, "D0_48": 297, "D1_48": 110}
"""Experiment 43's alarm counts at its thresholds, over all months."""
HARNESS_FOUND_D0_48: Final[int] = 47
"""Experiment 43's count of headline-band transfers found by D0 at 24 hours (of 160)."""


def _logged() -> dict[int, np.ndarray]:
    return near_log_masks(pad_stamps=EXPLAIN_PAD)


def _score_panel(
    *, panel: ArmPanel, thresholds: dict[str, float], scales: dict[int, dict[int, float]]
) -> pl.DataFrame:
    """Score all 600 injected transfers on a panel, one batch per task, in forked workers."""
    events = drawn_events(power=panel.power)
    groups = batches(events=events)
    state = {"panel": panel, "scales": scales, "thresholds": thresholds}
    with ProcessPoolExecutor(
        N_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(state,),
    ) as pool:
        rows = [r for part in pool.map(score_batch, list(enumerate(groups))) for r in part]
    return pl.DataFrame(rows)


def harness_check(*, out: Path) -> None:
    """Reproduce experiment 43's alarm counts and event outcomes from the stored panel."""
    panel = stored_panel()
    arm_inputs = ArmInputs.of(residual=panel.residual)
    results = detect_all(
        residual=panel.residual,
        expected=panel.expected,
        scales=arm_inputs.scales(),
        thresholds=OLD_THRESHOLDS,
    )
    for setting, expected in HARNESS_ALARM_COUNTS.items():
        count = results[(setting, "plain")][0].height
        print(f"{setting}: {count} alarms (experiment 43: {expected})")
        if count != expected:
            raise SystemExit(f"STOP: {setting} alarm count differs from experiment 43")
    logged = near_log_masks(pad_stamps=48)
    for setting, target in TARGET_COUNTS.items():
        count = odd_unexplained(table=results[(setting, "plain")][0], logged=logged)
        print(f"{setting}: {count} odd-month unexplained alarms (target {target})")
        if count != target:
            raise SystemExit(f"STOP: {setting} odd-month unexplained count differs from the target")
    scored = _score_panel(panel=panel, thresholds=OLD_THRESHOLDS, scales=arm_inputs.scales())
    stored = pl.read_parquet(EXPERIMENT_43_DIR / "event_results.parquet").with_columns(
        setting=pl.col("setting").replace_strict({v: k for k, v in SAMPLE_NAMES.items()})
    )
    keys = ["event_id", "setting", "variant"]
    joined = scored.join(stored, on=keys, suffix="_43", validate="1:1")
    different = joined.filter(
        (pl.col("found") != pl.col("found_43"))
        | (pl.col("onset_found") != pl.col("onset_found_43"))
    )
    events = events_frame(events=drawn_events(power=panel.power))
    headline = events.filter(
        pl.col("size_band") == "25% or more", pl.col("duration_band") == "over 1 d"
    )["event_id"]
    found = joined.filter(
        pl.col("setting") == "D0_48",
        pl.col("variant") == "plain",
        pl.col("event_id").is_in(headline.implode()),
    )["found"].sum()
    print(f"{joined.height} event rows, {different.height} differ; D0 24 h found {found} of 160")
    if different.height or found != HARNESS_FOUND_D0_48:
        raise SystemExit("STOP: the harness does not reproduce experiment 43")
    out.mkdir(parents=True, exist_ok=True)
    events.write_parquet(out / "events.parquet")
    print("harness reproduced")


def tune_one(args: tuple[str, Path, bool, Path]) -> str:
    """Tune thresholds for one arm-seed and write its real-alarm and real-event tables."""
    name, fits_dir, fallback, out = args
    panel = load_arm_panel(name=name, fits_dir=fits_dir, fallback=fallback)
    logged = _logged()
    arm_inputs = ArmInputs.of(residual=panel.residual)
    thresholds: dict[str, float] = {}
    curves: list[dict] = []
    for setting, target in TARGET_COUNTS.items():
        thresholds[setting], curve = tune_setting(
            arm_inputs=arm_inputs, setting=setting, target=target, logged=logged
        )
        curves += [
            {"setting": setting, "threshold": t, "odd_unexplained": c, "target": target}
            for t, c in curve
        ]
    scales = arm_inputs.scales()
    pl.DataFrame(curves).write_parquet(out / f"curve_{name}.parquet")
    pl.DataFrame(
        [
            {"window": w, "series": s, "scale": v}
            for w, per_series in scales.items()
            for s, v in per_series.items()
        ]
    ).write_parquet(out / f"scales_{name}.parquet")
    for label, used in (("matched", thresholds), ("old", OLD_THRESHOLDS)):
        results = detect_all(
            residual=panel.residual, expected=panel.expected, scales=scales, thresholds=used
        )
        alarm_rows(results=results, logged=logged).write_parquet(
            out / f"alarms_{label}_{name}.parquet"
        )
        if label == "matched":
            real = score_real(results=results, real=real_events(power=panel.power))
            real.write_parquet(out / f"real_results_{name}.parquet")
        odd = {
            setting: odd_unexplained(table=results[(setting, "plain")][0], logged=logged)
            for setting in SETTINGS
        }
        (out / f"odd_counts_{label}_{name}.json").write_text(json.dumps(odd))
    (out / f"thresholds_{name}.json").write_text(json.dumps(thresholds))
    return f"{name}: thresholds {thresholds}"


def score_one(*, name: str, fits_dir: Path, fallback: bool, out: Path) -> str:
    """Score the injected transfers on one arm-seed at its matched thresholds."""
    panel = load_arm_panel(name=name, fits_dir=fits_dir, fallback=fallback)
    thresholds = json.loads((out / f"thresholds_{name}.json").read_text())
    scales = ArmInputs.of(residual=panel.residual).scales()
    scored = _score_panel(panel=panel, thresholds=thresholds, scales=scales)
    scored.write_parquet(out / f"event_results_{name}.parquet")
    return f"{name}: {scored.height} event rows"


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=("harness-check", "tune", "score"))
    parser.add_argument("--names", nargs="*", default=list(ALL_NAMES))
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    parser.add_argument("--fits", type=Path, default=FITS_DIR)
    parser.add_argument("--fallback", action="store_true", help="smoke test: fill missing months")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for stage in args.stages:
        if stage == "harness-check":
            harness_check(out=args.out)
        elif stage == "tune":
            jobs = [(n, args.fits, args.fallback, args.out) for n in args.names]
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(N_WORKERS, mp_context=context) as pool:
                for line in pool.map(tune_one, jobs):
                    print(line, flush=True)
        else:
            for name in args.names:
                print(
                    score_one(name=name, fits_dir=args.fits, fallback=args.fallback, out=args.out),
                    flush=True,
                )


if __name__ == "__main__":
    main()
