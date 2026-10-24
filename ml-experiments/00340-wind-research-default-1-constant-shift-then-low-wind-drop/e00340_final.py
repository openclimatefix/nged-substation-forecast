"""Experiment 00340: build the shift arms from experiment 00336's stored mixtures.

For the default and for x2 (read from experiment 00336's `work/mix/`), repeat experiment 00336's
spread factor and offset table, then fit the shifts on the usable hindcast runs only:

- `S`: one constant added to every final quantile (fitted on the default's hindcast quantiles).
- `Sq`: one constant per quantile level (the same hindcast).
- `SX2`: x2's final quantiles plus one constant fitted on x2's hindcast quantiles.
- `Srev`, `Sqrev`, `SX2rev`: the placebos, S, S-q, and S+X2 with the sign of the shift reversed.
- `Soracle`: S fitted on the scored year itself (descriptive, never adopted).
- `Sdrift` and `SX2drift` (post hoc, descriptive, never adopted): the constant fitted on the
  replicate-period hindcast runs (2025-01 to 2025-06) scored with the table fitted before 2025-01,
  so it measures the drift of the bias out of sample, then applied to the scored year.

The replicate versions refit the table and the shifts on hindcast valid times before 2025-01-01.
Outputs, in this experiment's `work/final/`: `<arm>.npy` (3 seeds, scored cases, 11),
`<arm>_rep.npy`, `default_notable.npy` (copied), `replicate_cases.parquet` (copied), `shifts.json`,
and `reproduction.json`.

    uv run --no-sync python ml-experiments/00340-*/e00340_final.py   (PYTHONPATH: see e00340_common)
"""

import json
import os
import shutil

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_final as final336  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00340_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
c336 = common.c336
POINTS = c303.POINTS_PER_FRACTION


def main() -> None:  # noqa: PLR0915
    """Write the quantiles of every arm and the fitted shifts."""
    common.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    hind = ens_final.hindcast_cases(sites=sites)
    scored = ens_final.scored_cases(sites=sites)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hind_usable = hind.filter(pl.Series(usable))
    hind_actual = hind_usable["actual_norm"].to_numpy().astype(np.float64)
    scored_actual = scored["actual_norm"].to_numpy().astype(np.float64)
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
    shutil.copy(c336.FINAL_DIR / "replicate_cases.parquet", common.FINAL_DIR)
    shutil.copy(c336.FINAL_DIR / "default_notable.npy", common.FINAL_DIR)
    all_cases = pl.concat(
        [
            hind_usable.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
            scored.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
        ],
        how="vertical_relaxed",
    )
    era5_hind = final336.stats(kind="era5", set_name="hindcast")[usable]
    sites_hind = hind_usable["site"].to_numpy()
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    shifts: dict[str, dict] = {"default": {}, "x2": {}}

    def put(*, arm: str, main: np.ndarray, rep: np.ndarray) -> None:
        out.setdefault(arm, {"main": [], "rep": []})
        out[arm]["main"].append(main.astype(np.float32))
        out[arm]["rep"].append(rep.astype(np.float32))

    for seed in common.SEEDS:
        base: dict[str, dict[str, np.ndarray]] = {}
        for arm in ("default", "x2"):
            raw_h = final336.load(arm=arm, seed=seed, set_name="hindcast")[usable]
            raw_s = final336.load(arm=arm, seed=seed, set_name="scored")
            adjusted = c303.apply_spread(
                cases=all_cases, mixture=np.concatenate([raw_h, raw_s]).astype(np.float64)
            )
            adj_h, adj_s = adjusted[: hind_usable.height], adjusted[hind_usable.height :]
            drop = final336.table_drop(
                arm=arm, cases=hind_usable, era5_speed=era5_hind, masked=hind_masked, seed=seed
            )
            table, keys = ens_final.offsets_table(
                hindcast=hind_usable.filter(pl.Series(~drop)), quantiles=adj_h[~drop]
            )
            scored_q = ens_final.apply_offsets(
                cases=scored, quantiles=adj_s, table=table, keys=keys
            )
            hind_q = ens_final.apply_offsets(
                cases=hind_usable, quantiles=adj_h, table=table, keys=keys
            )
            keep_fit = fit_rows & ~drop
            table_r, keys_r = ens_final.offsets_table(
                hindcast=hind_usable.filter(pl.Series(keep_fit)), quantiles=adj_h[keep_fit]
            )
            rep_q = ens_final.apply_offsets(
                cases=replicate_cases,
                quantiles=adj_h[in_replicate],
                table=table_r,
                keys=keys_r,
            )
            hind_q_rep = ens_final.apply_offsets(
                cases=hind_usable.filter(pl.Series(fit_rows)),
                quantiles=adj_h[fit_rows],
                table=table_r,
                keys=keys_r,
            )
            base[arm] = {
                "scored": scored_q,
                "rep": rep_q,
                "hind": hind_q,
                "hind_rep": hind_q_rep,
            }
            fit_main = common.fit_constant_shift(quantiles=hind_q, actual=hind_actual)
            fit_rep = common.fit_constant_shift(quantiles=hind_q_rep, actual=hind_actual[fit_rows])
            shifts[arm][f"seed{seed}"] = {
                "constant": fit_main,
                "constant_replicate": fit_rep,
                "hindcast_rows": int(hind_q.shape[0]),
                "hindcast_rows_replicate_fit": int(hind_q_rep.shape[0]),
                "per_farm": {
                    farm: common.fit_constant_shift(
                        quantiles=hind_q[sites_hind == farm],
                        actual=hind_actual[sites_hind == farm],
                    )
                    for farm in common.FARMS
                },
            }
            shifts[arm][f"seed{seed}"]["drift_out_of_sample"] = common.fit_constant_shift(
                quantiles=rep_q, actual=hind_actual[in_replicate]
            )
            if arm == "default":
                per_level = common.fit_level_shifts(quantiles=hind_q, actual=hind_actual)
                per_level_rep = common.fit_level_shifts(
                    quantiles=hind_q_rep, actual=hind_actual[fit_rows]
                )
                shifts[arm][f"seed{seed}"]["per_level"] = per_level.tolist()
                shifts[arm][f"seed{seed}"]["per_level_replicate"] = per_level_rep.tolist()
                oracle = common.fit_constant_shift(quantiles=scored_q, actual=scored_actual)
                shifts[arm][f"seed{seed}"]["oracle_scored_year"] = oracle
        d, x = base["default"], base["x2"]
        s, s_rep = (
            shifts["default"][f"seed{seed}"]["constant"],
            shifts["default"][f"seed{seed}"]["constant_replicate"],
        )
        sx, sx_rep = (
            shifts["x2"][f"seed{seed}"]["constant"],
            shifts["x2"][f"seed{seed}"]["constant_replicate"],
        )
        q, q_rep = (
            np.array(shifts["default"][f"seed{seed}"]["per_level"]),
            np.array(shifts["default"][f"seed{seed}"]["per_level_replicate"]),
        )
        oracle = shifts["default"][f"seed{seed}"]["oracle_scored_year"]
        drift = shifts["default"][f"seed{seed}"]["drift_out_of_sample"]
        drift_x = shifts["x2"][f"seed{seed}"]["drift_out_of_sample"]
        put(
            arm="Sdrift",
            main=common.shifted(quantiles=d["scored"], shift=drift),
            rep=common.shifted(quantiles=d["rep"], shift=drift),
        )
        put(
            arm="SX2drift",
            main=common.shifted(quantiles=x["scored"], shift=drift_x),
            rep=common.shifted(quantiles=x["rep"], shift=drift_x),
        )
        put(arm="default", main=d["scored"], rep=d["rep"])
        put(arm="X2", main=x["scored"], rep=x["rep"])
        put(
            arm="S",
            main=common.shifted(quantiles=d["scored"], shift=s),
            rep=common.shifted(quantiles=d["rep"], shift=s_rep),
        )
        put(
            arm="Sq",
            main=common.shifted(quantiles=d["scored"], shift=q),
            rep=common.shifted(quantiles=d["rep"], shift=q_rep),
        )
        put(
            arm="SX2",
            main=common.shifted(quantiles=x["scored"], shift=sx),
            rep=common.shifted(quantiles=x["rep"], shift=sx_rep),
        )
        put(
            arm="Srev",
            main=common.shifted(quantiles=d["scored"], shift=-s),
            rep=common.shifted(quantiles=d["rep"], shift=-s_rep),
        )
        put(
            arm="Sqrev",
            main=common.shifted(quantiles=d["scored"], shift=-q),
            rep=common.shifted(quantiles=d["rep"], shift=-q_rep),
        )
        put(
            arm="SX2rev",
            main=common.shifted(quantiles=x["scored"], shift=-sx),
            rep=common.shifted(quantiles=x["rep"], shift=-sx_rep),
        )
        put(
            arm="Soracle",
            main=common.shifted(quantiles=d["scored"], shift=oracle),
            rep=common.shifted(quantiles=d["rep"], shift=oracle),
        )
        print(
            f"seed {seed}: S {s * POINTS:+.3f} pts, S+X2 {sx * POINTS:+.3f} pts, "
            f"per level {np.round(q * POINTS, 2).tolist()}, oracle {oracle * POINTS:+.3f} pts, "
            f"drift {drift * POINTS:+.3f} pts, x2 drift {drift_x * POINTS:+.3f} pts",
            flush=True,
        )
    for arm, parts in out.items():
        np.save(common.FINAL_DIR / f"{arm}.npy", np.stack(parts["main"]))
        np.save(common.FINAL_DIR / f"{arm}_rep.npy", np.stack(parts["rep"]))
    stored = {
        "default": np.load(c336.FINAL_DIR / "default.npy"),
        "X2": np.load(c336.FINAL_DIR / "x2.npy"),
    }
    cases = scored.with_columns(kept=~pl.col("masked"))
    keep = cases["kept"].to_numpy()
    report = {
        "known_answer_test": common.known_answer_test(),
        "leakage_check_fit_rows": {
            "hindcast_usable_rows": int(hind_usable.height),
            "scored_rows": int(scored.height),
            "latest_hindcast_valid_time": str(hind_usable["valid_time"].max()),
            "earliest_scored_valid_time": str(scored["valid_time"].min()),
        },
    }
    for arm, array in stored.items():
        mine = np.stack(out[arm]["main"])
        report[f"{arm}_max_abs_difference_to_experiment_336"] = float(np.abs(mine - array).max())
        report[f"{arm}_kept_score"] = final336.score(
            quantiles=mine, actual=scored_actual, keep=keep
        )
    stored_p1 = c336.c326.STORED_303_FINAL_DIR / "P1.npy"
    report["kept_score_stored_p1"] = final336.score(
        quantiles=np.load(stored_p1), actual=scored_actual, keep=keep
    )
    report["gate_difference"] = abs(report["default_kept_score"] - report["kept_score_stored_p1"])
    report["gate_passed"] = bool(report["gate_difference"] <= final336.GATE_TOLERANCE)
    (common.FINAL_DIR / "reproduction.json").write_text(json.dumps(report, indent=1))
    (common.FINAL_DIR / "shifts.json").write_text(json.dumps(shifts, indent=1))
    print("reproduction", report, flush=True)


if __name__ == "__main__":
    main()
