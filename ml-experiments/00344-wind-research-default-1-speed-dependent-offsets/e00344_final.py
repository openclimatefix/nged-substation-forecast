"""Experiment 00344: build the speed-dependent offset arms from experiment 00336's stored mixtures.

For the default (read from experiment 00336's `work/mix/`), repeat experiment 00336's spread factor
and offset table, then fit three corrections on the usable hindcast runs only, keyed by the
ensemble-mean 100 m wind speed at the valid time:

- `x1`: an offset per farm, lead day, quantile level, and speed bin.
- `x2`: a monotone smooth curve in speed per farm and quantile level.
- `x3`: a curve per quantile level shared by the three farms, times one scale per farm.
- `x2drift` and `x2oracle` (post hoc, descriptive, never adopted): the x2 curves fitted on the
  2025-01 to 2025-06 hindcast runs scored with the table fitted before 2025 (the drift, as in
  experiment 00340's Sdrift), and fitted on the scored year itself.
- `x1shuf`, `x2shuf`, `x3shuf`: the placebos, the same fitted offsets applied with the speed of
  another case of the same farm and lead day.

The replicate versions refit the table and the offsets on hindcast valid times before 2025-01-01.
The three seeds run in parallel in a pool started with the "spawn" method, one process per seed,
and each seed writes its own files atomically. Outputs, in this experiment's `work/final/`:
`<arm>.npy` (3 seeds, scored cases, 11), `<arm>_rep.npy`, `default_notable.npy` and
`replicate_cases.parquet` (copied), `fits.json`, and `reproduction.json`.

    uv run --no-sync python ml-experiments/00344-*/e00344_final.py   (PYTHONPATH: see e00344_common)
"""

import json
import multiprocessing
import os
import shutil
import sys
import traceback
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_final as final336  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00344_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
c336 = common.c336
SEED_DIR = common.FINAL_DIR / "seeds"
ARMS = ("default", "x1", "x2", "x3", "x1shuf", "x2shuf", "x3shuf", "x2drift", "x2oracle")


def _save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, array)
    temporary.rename(path)


def run_seed(seed: int) -> str:
    """Fit and write every arm for one seed; return a one-line summary."""
    try:
        return _run_seed(seed)
    except Exception:  # noqa: BLE001
        return "FAILED seed " + str(seed) + "\n" + traceback.format_exc()


def _run_seed(seed: int) -> str:  # noqa: PLR0915
    out_dir = SEED_DIR / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    hind = ens_final.hindcast_cases(sites=sites)
    scored = ens_final.scored_cases(sites=sites)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hind_usable = hind.filter(pl.Series(usable))
    hind_actual = hind_usable["actual_norm"].to_numpy().astype(np.float64)
    mask = c336.idle_mask_frame()
    hind_masked = (
        hind_usable.join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")[
            "masked"
        ]
        .fill_null(value=False)
        .to_numpy()
    )
    in_replicate = (hind_usable["power_fcst_init_time"] >= common.REPLICATE_START).to_numpy()
    fit_rows = (hind_usable["valid_time"] < common.REPLICATE_START).to_numpy()
    replicate_cases = pl.read_parquet(c336.FINAL_DIR / "replicate_cases.parquet")
    all_cases = pl.concat(
        [
            hind_usable.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
            scored.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
        ],
        how="vertical_relaxed",
    )
    era5_hind = final336.stats(kind="era5", set_name="hindcast")[usable]
    speed_hind = final336.stats(kind="speed", set_name="hindcast")[usable][:, 0]
    speed_scored = final336.stats(kind="speed", set_name="scored")[:, 0]
    farm_hind = hind_usable["site"].to_numpy()
    farm_scored = scored["site"].to_numpy()
    farm_rep = replicate_cases["site"].to_numpy()
    day_hind = hind_usable["lead_day"].clip(0, 10).to_numpy()
    day_scored = scored["lead_day"].clip(0, 10).to_numpy()
    day_rep = replicate_cases["lead_day"].clip(0, 10).to_numpy()

    raw_h = final336.load(arm="default", seed=seed, set_name="hindcast")[usable]
    raw_s = final336.load(arm="default", seed=seed, set_name="scored")
    adjusted = c303.apply_spread(
        cases=all_cases, mixture=np.concatenate([raw_h, raw_s]).astype(np.float64)
    )
    adj_h, adj_s = adjusted[: hind_usable.height], adjusted[hind_usable.height :]
    drop = final336.table_drop(
        arm="default", cases=hind_usable, era5_speed=era5_hind, masked=hind_masked, seed=seed
    )
    table, keys = ens_final.offsets_table(
        hindcast=hind_usable.filter(pl.Series(~drop)), quantiles=adj_h[~drop]
    )
    scored_q = ens_final.apply_offsets(cases=scored, quantiles=adj_s, table=table, keys=keys)
    hind_q = ens_final.apply_offsets(cases=hind_usable, quantiles=adj_h, table=table, keys=keys)
    keep_fit = fit_rows & ~drop
    table_r, keys_r = ens_final.offsets_table(
        hindcast=hind_usable.filter(pl.Series(keep_fit)), quantiles=adj_h[keep_fit]
    )
    rep_q = ens_final.apply_offsets(
        cases=replicate_cases, quantiles=adj_h[in_replicate], table=table_r, keys=keys_r
    )
    hind_q_rep = ens_final.apply_offsets(
        cases=hind_usable.filter(pl.Series(fit_rows)),
        quantiles=adj_h[fit_rows],
        table=table_r,
        keys=keys_r,
    )
    # Inputs of the fits: (post-table quantiles, observed, farm, lead day, speed) and the targets.
    speed_rep = speed_hind[in_replicate]
    fits_input = {
        "main": (hind_q, hind_actual, farm_hind, day_hind, speed_hind),
        "rep": (
            hind_q_rep,
            hind_actual[fit_rows],
            farm_hind[fit_rows],
            day_hind[fit_rows],
            speed_hind[fit_rows],
        ),
    }
    targets = {
        "main": (scored_q, farm_scored, day_scored, speed_scored),
        "rep": (rep_q, farm_rep, day_rep, speed_rep),
    }
    record: dict = {}
    produced: dict[str, dict[str, np.ndarray]] = {a: {} for a in ARMS}
    for period in ("main", "rep"):
        quantiles, actual, farms, days, speed = fits_input[period]
        base, t_farms, t_days, t_speed = targets[period]
        cells = common.cell_ids(sites=farms, lead_day=days, speed=speed)
        x1 = common.fit_x1(quantiles=quantiles, actual=actual, cells=cells)
        x2 = common.fit_x2(quantiles=quantiles, actual=actual, sites=farms, speed=speed)
        x3 = common.fit_x3(quantiles=quantiles, actual=actual, sites=farms, speed=speed)
        record[period] = {
            "x1_cells": len(x1),
            "x1_table": {str(k): v.tolist() for k, v in x1.items()},
            "x2": x2,
            "x3": x3,
        }
        shuffled = common.shuffled_speed(
            speed=t_speed, sites=t_farms, lead_day=t_days, seed=1000 + seed
        )
        for suffix, key in (("", t_speed), ("shuf", shuffled)):
            t_cells = common.cell_ids(sites=t_farms, lead_day=t_days, speed=key)
            produced[f"x1{suffix}"][period] = common.apply_x1(
                quantiles=base, cells=t_cells, table=x1
            )
            produced[f"x2{suffix}"][period] = common.apply_x2(
                quantiles=base, sites=t_farms, speed=key, curves=x2
            )
            produced[f"x3{suffix}"][period] = common.apply_x3(
                quantiles=base, sites=t_farms, speed=key, fit=x3
            )
        produced["default"][period] = base
        if period == "main":
            drift = common.fit_x2(
                quantiles=rep_q, actual=hind_actual[in_replicate], sites=farm_rep, speed=speed_rep
            )
            oracle = common.fit_x2(
                quantiles=scored_q,
                actual=scored["actual_norm"].to_numpy().astype(np.float64),
                sites=farm_scored,
                speed=speed_scored,
            )
            produced["x2drift"][period] = common.apply_x2(
                quantiles=base, sites=t_farms, speed=t_speed, curves=drift
            )
            produced["x2oracle"][period] = common.apply_x2(
                quantiles=base, sites=t_farms, speed=t_speed, curves=oracle
            )
            record[period]["x2drift"] = drift
            record[period]["x2oracle"] = oracle
        else:
            produced["x2drift"][period] = produced["x2"][period]
            produced["x2oracle"][period] = produced["x2"][period]
    for arm, parts in produced.items():
        _save(out_dir / f"{arm}.npy", parts["main"].astype(np.float32))
        _save(out_dir / f"{arm}_rep.npy", parts["rep"].astype(np.float32))
    (out_dir / "fits.json").write_text(json.dumps(record))
    return f"seed {seed} done: x1 cells {record['main']['x1_cells']}"


def main() -> None:
    """Run the three seeds in a spawn pool, then stack the arms and write the gate."""
    common.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    with context.Pool(len(common.SEEDS)) as pool:
        for message in pool.imap_unordered(run_seed, common.SEEDS):
            print(message, flush=True)
            if message.startswith("FAILED"):
                pool.terminate()
                sys.exit(1)
    shutil.copy(c336.FINAL_DIR / "replicate_cases.parquet", common.FINAL_DIR)
    shutil.copy(c336.FINAL_DIR / "default_notable.npy", common.FINAL_DIR)
    fits = {}
    for arm in ARMS:
        for suffix in ("", "_rep"):
            stack = np.stack(
                [np.load(SEED_DIR / f"seed{s}" / f"{arm}{suffix}.npy") for s in common.SEEDS]
            )
            np.save(common.FINAL_DIR / f"{arm}{suffix}.npy", stack)
    for seed in common.SEEDS:
        fits[f"seed{seed}"] = json.loads((SEED_DIR / f"seed{seed}" / "fits.json").read_text())
    (common.FINAL_DIR / "fits.json").write_text(json.dumps(fits))
    sites = c303.site_table()
    hind = ens_final.hindcast_cases(sites=sites)
    scored = ens_final.scored_cases(sites=sites)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hind_usable = hind.filter(pl.Series(usable))
    scored_actual = scored["actual_norm"].to_numpy().astype(np.float64)
    keep = (~pl.col("masked")).alias("kept")
    keep_rows = scored.with_columns(keep)["kept"].to_numpy()
    mine = np.load(common.FINAL_DIR / "default.npy")
    report = {
        "known_answer_test": common.known_answer_test(),
        "leakage_check_fit_rows": {
            "hindcast_usable_rows": int(hind_usable.height),
            "scored_rows": int(scored.height),
            "latest_hindcast_valid_time": str(hind_usable["valid_time"].max()),
            "earliest_scored_valid_time": str(scored["valid_time"].min()),
            "key": "ensemble-mean 100 m speed at the valid time (stats kind 'speed', column 0)",
        },
        "default_max_abs_difference_to_experiment_336": float(
            np.abs(mine - np.load(c336.FINAL_DIR / "default.npy")).max()
        ),
        "default_kept_score": final336.score(quantiles=mine, actual=scored_actual, keep=keep_rows),
    }
    stored_p1 = c336.c326.STORED_303_FINAL_DIR / "P1.npy"
    report["kept_score_stored_p1"] = final336.score(
        quantiles=np.load(stored_p1), actual=scored_actual, keep=keep_rows
    )
    report["gate_difference"] = abs(report["default_kept_score"] - report["kept_score_stored_p1"])
    report["gate_passed"] = bool(report["gate_difference"] <= final336.GATE_TOLERANCE)
    (common.FINAL_DIR / "reproduction.json").write_text(json.dumps(report, indent=1))
    print("reproduction", report, flush=True)


if __name__ == "__main__":
    main()
