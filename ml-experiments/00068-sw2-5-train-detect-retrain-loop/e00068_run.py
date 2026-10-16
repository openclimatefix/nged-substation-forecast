r"""Run experiment 68's loop: refit, detect, score, mask, refit again.

Stages (each skips work whose output exists, so a crash costs one stage):

- `world`: write world B's injected MW and event table.
- `check`: round-0 reproduction checks against experiment 17's stored `H0.5` predictions and
  experiment 61's PE recall on the stored panel.
- `loop`: rounds 0 to 3 in worlds A and B, with the random-mask control after round 0.

    uv run python ml-experiments/00068-*/e00068_run.py world check loop
"""

from __future__ import annotations

import argparse
import json

import polars as pl
from e00068_common import SERIES_IDS, WORK_DIR, load_panel
from e00068_detectors import prepare
from e00068_fit import fit_round, round_dir
from e00068_round import (
    WINDOW,
    build_panel,
    detect_round,
    error_table,
    random_mask,
    score_real_transfers,
    score_world_a,
    score_world_b,
)
from e00068_world import build, draw, spread_out, world_a_batches

N_ROUNDS_LOOP = 3
STORED_FITS = WORK_DIR.parent / "00017-training-window-and-retraining-rule-drift-demand" / "fits"


def scale0() -> dict:
    """The noise scales of the stored panel, which fix the units of the donor-step tables."""
    return prepare(residuals=load_panel()["residual"], window=WINDOW).scale


def check() -> None:
    """Round 0 must match experiment 17's stored H0.5 predictions; PE must match experiment 61."""
    fit_round(name="A_r0")
    worst = 0.0
    for s in SERIES_IDS:
        mine = pl.read_parquet(round_dir(name="A_r0") / f"{s}.parquet")
        stored = pl.read_parquet(STORED_FITS / f"{s}.parquet").select("valid_time", "H0.5")
        joined = mine.join(stored, on="valid_time", validate="1:1")
        gap = joined.select((pl.col("pred") - pl.col("H0.5")).abs().max()).item()
        worst = max(worst, float(gap))
    print(f"largest difference from experiment 17's H0.5 predictions: {worst:.2e}")
    panel = load_panel()
    batches = world_a_batches(power=panel["power"])
    scored = score_world_a(
        name="stored", batches=batches, scale0=scale0(), panel=panel, threshold=4.0
    )
    scored.write_parquet(WORK_DIR / "check_stored_panel.parquet")
    for shape in ("S2", "S3", "S4"):
        part = scored.filter(pl.col("shape") == shape)
        w = part["moved_mw"].to_numpy()
        recall = (w * part["found"].to_numpy()).sum() / w.sum()
        print(f"stored panel, PE, {shape}: MW-weighted recall {recall:.3f} (n={part.height})")


def loop() -> None:
    """Rounds 0 to 3 in both worlds, with scoring after each detection."""
    power = load_panel()["power"]
    batches = world_a_batches(power=power)
    events_b = spread_out(events=draw(power=power))
    zero = scale0()
    results: list[pl.DataFrame] = []
    errors: list[pl.DataFrame] = []
    reals: list[pl.DataFrame] = []
    stats: dict[str, dict] = {}
    reference = None
    for k in range(N_ROUNDS_LOOP + 1):
        for world in "AB":
            name = f"{world}_r{k}"
            round_dir(name=name).mkdir(parents=True, exist_ok=True)
            fit_round(name=name)
            build_panel(name=name)
            following = f"{world}_r{k + 1}" if k < N_ROUNDS_LOOP else None
            stats[name] = detect_round(
                name=name, next_name=following, reference_threshold=reference
            )
            if name == "A_r0":
                reference = stats[name]["threshold"]
            if world == "A":
                results.append(score_world_a(name=name, batches=batches, scale0=zero))
                errors.append(error_table(name=name))
                reals.append(score_real_transfers(name=name))
            else:
                results.append(score_world_b(name=name, events=events_b, scale0=zero))
        if k == 0:
            random_mask(source_name="A_r1", name="A_rc1")
            fit_round(name="A_rc1")
            build_panel(name="A_rc1")
            stats["A_rc1"] = detect_round(
                name="A_rc1", next_name=None, reference_threshold=reference
            )
            results.append(score_world_a(name="A_rc1", batches=batches, scale0=zero))
            errors.append(error_table(name="A_rc1"))
            reals.append(score_real_transfers(name="A_rc1"))
    pl.concat(results).write_parquet(WORK_DIR / "event_results.parquet")
    pl.concat(errors).write_parquet(WORK_DIR / "errors.parquet")
    pl.concat(reals).write_parquet(WORK_DIR / "real_results.parquet")
    (WORK_DIR / "round_stats.json").write_text(json.dumps(stats))
    print("loop done")


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=("world", "check", "loop"))
    args = parser.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    stages = {"world": build, "check": check, "loop": loop}
    for stage in args.stages:
        stages[stage]()


if __name__ == "__main__":
    main()
