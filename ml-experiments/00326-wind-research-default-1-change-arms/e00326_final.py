"""Experiment 00326, second step: spread factor, offset table, and scored quantiles for every arm.

Follows experiment 00303's `e00303_final.py` for every arm, so that the default, the fitted arms,
the placebos, and the stored arms all pass through the same code: experiment 33's spread factor
(fitted month by month on earlier cases, over the usable hindcast cases and the scored cases in time
order), then experiment 259's offset table (the pinball-optimal offset per farm, lead day, and
level, fitted on the usable hindcast cases).

Outputs, in `work/final/`:

- `<arm>.npy`: the scored year's quantiles with the table, shape (3 seeds, scored cases, 11), in the
  order of `e00303_score.ordered_cases`. `<arm>_notable.npy` is the same without the table.
- `<arm>_rep.npy`: the replicate's quantiles with the table, shape (3, replicate cases, 11), in the
  order of `replicate_cases.parquet`, with the table fitted on hindcast cases whose valid times
  precede 2025-01-01 and scored on the hindcast runs initialised from 2025-01-01.
- `replicate_cases.parquet`: the replicate cases, their month, mask flag, and the actual.
- `reproduction.json`: the maximum absolute difference of the refitted default from experiment
  303's stored P1, and both kept-row scores.

    uv run --no-sync python ml-experiments/00326-*/e00326_final.py   (PYTHONPATH as for the fit)
"""

import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
STORED_FILES = {"stored_default": ("285", "xgb_B0"), "c": ("303", "P8")}
ARMS = ("stored_default", "default", "a", "b", "c", "d", "a_pl", "b_pl", "c_pl")


def mixture_path(*, arm: str, seed: int, set_name: str, farm: str) -> Path:
    """Return the path of one stored mixture."""
    if arm in STORED_FILES:
        source, prefix = STORED_FILES[arm]
        folder = common.STORED_285_MIX_DIR if source == "285" else common.STORED_303_MIX_DIR
        return folder / f"{prefix}_s{seed}_{set_name}_{farm}.npy"
    prefix = "d_rep" if (arm == "d" and set_name == "replicate") else arm
    return (
        common.MIX_DIR
        / f"{prefix}_s{seed}_{'hindcast' if set_name == 'replicate' else set_name}_{farm}.npy"
    )


def load(*, arm: str, seed: int, set_name: str) -> np.ndarray:
    """Return one arm's mixtures for the three farms in order, shape (cases, 11)."""
    return np.concatenate(
        [np.load(mixture_path(arm=arm, seed=seed, set_name=set_name, farm=f)) for f in common.FARMS]
    )


def available(*, arm: str) -> bool:
    """Return whether every file of `arm` is stored."""
    sets = ("hindcast", "scored", "replicate") if arm == "d" else ("hindcast", "scored")
    return all(
        mixture_path(arm=arm, seed=s, set_name=n, farm=f).exists()
        for s in common.SEEDS
        for n in sets
        for f in common.FARMS
    )


def score(*, quantiles: np.ndarray, actual: np.ndarray, keep: np.ndarray) -> float:
    """Return the mean quantile score (points of capacity) of seed-averaged scores over `keep`."""
    per_seed = [
        c303.quantile_score(quantiles=q[keep].astype(np.float64), actual=actual[keep]).mean()
        for q in quantiles
    ]
    return float(np.mean(per_seed) * c303.POINTS_PER_FRACTION)


def main() -> None:
    """Write the scored and replicate quantiles of every arm that is stored."""
    common.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    hind = ens_final.hindcast_cases(sites=sites)
    scored = ens_final.scored_cases(sites=sites)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hind_usable = hind.filter(pl.Series(usable))
    mask = c303.idle_mask().with_columns(masked=pl.lit(value=True))
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
    table_cases = hind_usable.filter(pl.Series(fit_rows))
    for arm in ARMS:
        if (common.FINAL_DIR / f"{arm}.npy").exists() or not available(arm=arm):
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
            table, keys = ens_final.offsets_table(hindcast=hind_usable, quantiles=adj_h)
            main_seeds.append(
                ens_final.apply_offsets(cases=scored, quantiles=adj_s, table=table, keys=keys)
            )
            none_seeds.append(adj_s.astype(np.float32))
            if arm == "d":
                raw_h = load(arm=arm, seed=seed, set_name="replicate")[usable]
                adjusted = c303.apply_spread(
                    cases=all_cases, mixture=np.concatenate([raw_h, raw_s]).astype(np.float64)
                )
                adj_h = adjusted[: hind_usable.height]
            table, keys = ens_final.offsets_table(hindcast=table_cases, quantiles=adj_h[fit_rows])
            rep_seeds.append(
                ens_final.apply_offsets(
                    cases=replicate_cases, quantiles=adj_h[in_replicate], table=table, keys=keys
                )
            )
            print(arm, "seed", seed, flush=True)
        np.save(common.FINAL_DIR / f"{arm}_notable.npy", np.stack(none_seeds))
        np.save(common.FINAL_DIR / f"{arm}_rep.npy", np.stack(rep_seeds))
        np.save(common.FINAL_DIR / f"{arm}.npy", np.stack(main_seeds))
    stored_p1 = common.STORED_303_FINAL_DIR / "P1.npy"
    mine = common.FINAL_DIR / "default.npy"
    if stored_p1.exists() and mine.exists():
        cases = scored.with_columns(kept=~pl.col("masked"))
        actual = cases["actual_norm"].to_numpy().astype(np.float64)
        keep = cases["kept"].to_numpy()
        refit, stored = np.load(mine), np.load(stored_p1)
        report = {
            "max_abs_difference": float(np.abs(refit - stored).max()),
            "kept_score_refit": score(quantiles=refit, actual=actual, keep=keep),
            "kept_score_stored_p1": score(quantiles=stored, actual=actual, keep=keep),
        }
        report["gate_difference"] = abs(report["kept_score_refit"] - report["kept_score_stored_p1"])
        (common.FINAL_DIR / "reproduction.json").write_text(json.dumps(report, indent=1))
        print("reproduction gate", report, flush=True)


if __name__ == "__main__":
    main()
