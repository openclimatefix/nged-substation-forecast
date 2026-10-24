"""Experiment 00336, second step: spread factor, offset table, and scored quantiles for every arm.

Follows experiment 00326's `e00326_final.py` for every fitted arm (experiment 33's spread factor,
then experiment 259's offset table), with one change: the offset table of the training arms `x2*`
is fitted on the hindcast cases that those arms keep (the ERA5 speed at the case's valid time is
not below the cut-in, and for `x2e` the experiment 210 mask is not set). The post-hoc arms `x1*`
zero all 11 quantiles of the default's final quantiles wherever their case mask is set.

Outputs, in `work/final/`: `<arm>.npy` (the scored year's quantiles with the table, shape (3 seeds,
scored cases, 11)), `<arm>_rep.npy` (the replicate's, table fitted on hindcast cases with valid
times before 2025-01-01), `default_notable.npy`, `replicate_cases.parquet`, and
`reproduction.json`.

    uv run --no-sync python ml-experiments/00336-*/e00336_final.py   (PYTHONPATH as for the fit)
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
GATE_TOLERANCE = 0.01


def load(*, arm: str, seed: int, set_name: str) -> np.ndarray:
    """Return one arm's mixtures for the three farms in order, shape (cases, 11)."""
    return np.concatenate(
        [np.load(common.MIX_DIR / f"{arm}_s{seed}_{set_name}_{farm}.npy") for farm in common.FARMS]
    )


def stats(*, kind: str, set_name: str) -> np.ndarray:
    """Return the per-case statistics of the three farms in order."""
    return np.concatenate(
        [np.load(common.stats_path(kind=kind, set_name=set_name, farm=f)) for f in common.FARMS]
    )


def cut_in_per_case(*, sites: np.ndarray, suffix: str) -> np.ndarray:
    """Return the cut-in speed (m/s) of each case's farm for a cut-in variant."""
    by_farm = {farm: common.cut_in_for(farm=farm, suffix=suffix) for farm in common.FARMS}
    return np.array([by_farm[s] for s in sites])


def table_drop(
    *, arm: str, cases: pl.DataFrame, era5_speed: np.ndarray, masked: np.ndarray, seed: int
) -> np.ndarray:
    """Return the hindcast cases that the offset table of `arm` leaves out (True = leave out)."""
    if not arm.startswith("x2"):
        return np.zeros(cases.height, dtype=bool)
    rest = arm[2:]
    suffix = rest if rest in common.CUT_IN_VARIANTS else ""
    sites = cases["site"].to_numpy()
    drop = era5_speed < cut_in_per_case(sites=sites, suffix=suffix)
    if rest == "e":
        drop = drop | masked
    if rest == "pl":
        group = np.char.add(
            sites.astype(str),
            common.month_labels(
                valid_time=cases["valid_time"].dt.replace_time_zone(None).to_numpy()
            ),
        )
        drop = common.random_like(flag=drop, group=group, seed=common.PLACEBO_SEED + seed)
    return drop


def score(*, quantiles: np.ndarray, actual: np.ndarray, keep: np.ndarray) -> float:
    """Return the mean quantile score (points of capacity) of seed-averaged scores over `keep`."""
    per_seed = [
        c303.quantile_score(quantiles=q[keep].astype(np.float64), actual=actual[keep]).mean()
        for q in quantiles
    ]
    return float(np.mean(per_seed) * c303.POINTS_PER_FRACTION)


def post_hoc_masks(
    *, sites: np.ndarray, month: np.ndarray, speed: np.ndarray
) -> dict[str, np.ndarray]:
    """Return each post-hoc arm's case mask from the member speed statistics (mean, maximum)."""
    out = {}
    for arm, suffix in (("x1", ""), ("x1lo", "lo"), ("x1hi", "hi"), ("x1g3", "g3")):
        out[arm] = speed[:, 0] < cut_in_per_case(sites=sites, suffix=suffix)
    out["x1a"] = speed[:, 1] < cut_in_per_case(sites=sites, suffix="")
    out["x1pl"] = common.random_like(
        flag=out["x1"], group=np.char.add(sites.astype(str), month), seed=common.PLACEBO_SEED
    )
    return out


def zeroed(*, quantiles: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return `quantiles` (seeds, cases, 11) with every quantile of the masked cases set to zero."""
    out = quantiles.copy()
    out[:, mask, :] = 0.0
    return out


def main() -> None:  # noqa: PLR0915
    """Write the scored and replicate quantiles of every arm that is stored."""
    common.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    hind = ens_final.hindcast_cases(sites=sites)
    scored = ens_final.scored_cases(sites=sites)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hind_usable = hind.filter(pl.Series(usable))
    mask = common.idle_mask_frame()
    hind_masked = (
        hind_usable.join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")[
            "masked"
        ]
        .fill_null(value=False)
        .to_numpy()
    )
    in_replicate = (hind_usable["power_fcst_init_time"] >= common.REPLICATE_START).to_numpy()
    fit_rows = (hind_usable["valid_time"] < common.REPLICATE_START).to_numpy()
    replicate_cases = (
        hind_usable.filter(pl.Series(in_replicate))
        .join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(
            masked=pl.col("masked").fill_null(value=False),
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        )
    )
    replicate_cases.write_parquet(common.FINAL_DIR / "replicate_cases.parquet")
    all_cases = pl.concat(
        [
            hind_usable.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
            scored.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
        ],
        how="vertical_relaxed",
    )
    era5_hind = stats(kind="era5", set_name="hindcast")[usable]
    for arm in common.FIT_ARMS:
        if (common.FINAL_DIR / f"{arm}.npy").exists() or not all(
            (common.MIX_DIR / f"{arm}_s{s}_{n}_{f}.npy").exists()
            for s in common.SEEDS
            for n in ("hindcast", "scored")
            for f in common.FARMS
        ):
            print("skipping", arm, flush=True)
            continue
        main_seeds, none_seeds, rep_seeds = [], [], []
        for seed in common.SEEDS:
            raw_h = load(arm=arm, seed=seed, set_name="hindcast")[usable]
            raw_s = load(arm=arm, seed=seed, set_name="scored")
            adjusted = c303.apply_spread(
                cases=all_cases, mixture=np.concatenate([raw_h, raw_s]).astype(np.float64)
            )
            adj_h, adj_s = adjusted[: hind_usable.height], adjusted[hind_usable.height :]
            drop = table_drop(
                arm=arm, cases=hind_usable, era5_speed=era5_hind, masked=hind_masked, seed=seed
            )
            table, keys = ens_final.offsets_table(
                hindcast=hind_usable.filter(pl.Series(~drop)), quantiles=adj_h[~drop]
            )
            main_seeds.append(
                ens_final.apply_offsets(cases=scored, quantiles=adj_s, table=table, keys=keys)
            )
            none_seeds.append(adj_s.astype(np.float32))
            keep_fit = fit_rows & ~drop
            table, keys = ens_final.offsets_table(
                hindcast=hind_usable.filter(pl.Series(keep_fit)), quantiles=adj_h[keep_fit]
            )
            rep_seeds.append(
                ens_final.apply_offsets(
                    cases=replicate_cases, quantiles=adj_h[in_replicate], table=table, keys=keys
                )
            )
            print(arm, "seed", seed, f"table drops {int(drop.sum())} of {drop.size}", flush=True)
        if arm == "default":
            np.save(common.FINAL_DIR / "default_notable.npy", np.stack(none_seeds))
        np.save(common.FINAL_DIR / f"{arm}_rep.npy", np.stack(rep_seeds))
        np.save(common.FINAL_DIR / f"{arm}.npy", np.stack(main_seeds))
    default_path = common.FINAL_DIR / "default.npy"
    if default_path.exists():
        default, default_rep = np.load(default_path), np.load(common.FINAL_DIR / "default_rep.npy")
        speed_scored, speed_hind = (
            stats(kind="speed", set_name="scored"),
            stats(kind="speed", set_name="hindcast"),
        )
        for label, base, cases, speed in (
            ("", default, scored, speed_scored),
            ("_rep", default_rep, replicate_cases, speed_hind[usable][in_replicate]),
        ):
            masks = post_hoc_masks(
                sites=cases["site"].to_numpy(),
                month=cases["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(),
                speed=speed,
            )
            for arm, case_mask in masks.items():
                np.save(
                    common.FINAL_DIR / f"{arm}{label}.npy", zeroed(quantiles=base, mask=case_mask)
                )
                print(arm, label or "scored", f"mask share {case_mask.mean():.3f}", flush=True)
        stored_p1 = common.c326.STORED_303_FINAL_DIR / "P1.npy"
        cases = scored.with_columns(kept=~pl.col("masked"))
        actual = cases["actual_norm"].to_numpy().astype(np.float64)
        keep = cases["kept"].to_numpy()
        refit, stored = default, np.load(stored_p1)
        report = {
            "max_abs_difference": float(np.abs(refit - stored).max()),
            "kept_score_refit": score(quantiles=refit, actual=actual, keep=keep),
            "kept_score_stored_p1": score(quantiles=stored, actual=actual, keep=keep),
        }
        report["gate_difference"] = abs(report["kept_score_refit"] - report["kept_score_stored_p1"])
        report["gate_passed"] = bool(report["gate_difference"] <= GATE_TOLERANCE)
        (common.FINAL_DIR / "reproduction.json").write_text(json.dumps(report, indent=1))
        print("reproduction gate", report, flush=True)


if __name__ == "__main__":
    main()
