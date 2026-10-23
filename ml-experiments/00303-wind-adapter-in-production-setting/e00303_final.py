"""Experiment 00303, second step: spread factor, offset table, head, and scored quantiles.

Reads the member mixtures of `e00303_fit.py` (and experiment 00285's xgb_B0 for P0), applies
experiment 33's spread factor (fitted month by month on earlier cases over the hindcast and the
scored cases in time order), builds the offset table on the hindcast (experiment 259's T5), and
stores each arm's scored quantiles, shape (3 seeds, scored cases, 11), in the order of
`e00303_score.ordered_cases`, in `work/final/<arm>.npy`.

Arms: P0 (XGBoost on members), P1 (P0 plus table), P2, P3 (P2 plus table), P4, P5, P7, P8 (P8 plus
table, as the plan says), and the post hoc extras P5T (P5 plus table) and P4T. P6 is added by
`e00303_head.py`. Hindcast quantile scores of each arm are stored for the choice of P6's base.

    uv run --no-sync python e00303_final.py
"""

import json
import os
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as common  # noqa: E402
import e00303_ens_final as ens_final  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

OUT_DIR: Final[Path] = common.WORK_DIR / "final"
HINDCAST_DIR: Final[Path] = common.WORK_DIR / "final_hindcast"
FARMS: Final[tuple[str, ...]] = ens_final.FARMS
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MIXTURE_ARMS: Final[tuple[str, ...]] = ("P0", "P2", "P4", "P5", "P7", "P8")
TABLE_ARMS: Final[dict[str, str]] = {
    "P1": "P0",
    "P3": "P2",
    "P4T": "P4",
    "P5T": "P5",
    "P7T": "P7",
    "P8": "P8",
}
"""Arms that apply the offset table after the spread factor: name -> the mixture arm it adds to."""
PLAIN_ARMS: Final[tuple[str, ...]] = ("P0", "P2", "P4", "P5", "P7")
"""Arms stored without the table. P8 is stored with the table only (the plan's definition)."""


def load_mixture(*, arm: str, seed: int, set_name: str) -> np.ndarray:
    """Return one arm's mixtures for the three farms in order, shape (cases, 11)."""
    parts = []
    for farm in FARMS:
        if arm == "P0":
            path = common.ENS_MIX_285_DIR / f"xgb_B0_s{seed}_{set_name}_{farm}.npy"
        else:
            path = common.MIX_DIR / f"{arm}_s{seed}_{set_name}_{farm}.npy"
        parts.append(np.load(path))
    return np.concatenate(parts)


def available(*, arm: str) -> bool:
    """Return whether every farm, seed, and set of `arm` is stored."""
    for farm in FARMS:
        for seed in SEEDS:
            for name in ("hindcast", "scored"):
                base = common.ENS_MIX_285_DIR if arm == "P0" else common.MIX_DIR
                prefix = f"xgb_B0" if arm == "P0" else arm
                if not (base / f"{prefix}_s{seed}_{name}_{farm}.npy").exists():
                    return False
    return True


def main() -> None:
    """Write the scored (and hindcast) quantiles of every arm that is stored."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HINDCAST_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table()
    hind = ens_final.hindcast_cases(sites=sites)
    scored = ens_final.scored_cases(sites=sites)
    for farm in FARMS:
        stored = np.load(common.MIX_DIR / f"P2_s0_hindcast_{farm}.npy", mmap_mode="r")
        if stored.shape[0] != hind.filter(pl.col("site") == farm).height:
            msg = f"{farm}: stored hindcast mixtures and hindcast cases differ in count"
            raise ValueError(msg)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hind_usable = hind.filter(pl.Series(usable))
    hind_actual = hind_usable["actual_norm"].to_numpy().astype(np.float64)
    all_cases = pl.concat(
        [
            hind_usable.select(*common.KEY_COLUMNS, "lead_hours", "actual_norm"),
            scored.select(*common.KEY_COLUMNS, "lead_hours", "actual_norm"),
        ],
        how="vertical_relaxed",
    )
    results: dict[str, list[np.ndarray]] = {}
    hindcast_results: dict[str, list[np.ndarray]] = {}
    hindcast_scores: dict[str, float] = {}
    for arm in MIXTURE_ARMS:
        if not available(arm=arm):
            print("skipping", arm, "(not all stored)", flush=True)
            continue
        for seed in SEEDS:
            raw_h = load_mixture(arm=arm, seed=seed, set_name="hindcast")
            raw_s = load_mixture(arm=arm, seed=seed, set_name="scored")
            adjusted = common.apply_spread(
                cases=all_cases,
                mixture=np.concatenate([raw_h[usable], raw_s]).astype(np.float64),
            )
            adj_h, adj_s = adjusted[: hind_usable.height], adjusted[hind_usable.height :]
            if arm in PLAIN_ARMS:
                results.setdefault(arm, []).append(adj_s.astype(np.float32))
                hindcast_results.setdefault(arm, []).append(adj_h.astype(np.float32))
            if seed == 0 and arm in PLAIN_ARMS:
                hindcast_scores[arm] = float(
                    common.quantile_score(quantiles=adj_h, actual=hind_actual).mean()
                    * common.POINTS_PER_FRACTION
                )
            for name, source in TABLE_ARMS.items():
                if source != arm:
                    continue
                table, keys = ens_final.offsets_table(hindcast=hind_usable, quantiles=adj_h)
                stacked = ens_final.apply_offsets(
                    cases=scored, quantiles=adj_s, table=table, keys=keys
                )
                results.setdefault(name, []).append(stacked)
                # the table is fitted on these hindcast cases, so its hindcast score is in sample
                hindcast_results.setdefault(name, []).append(
                    ens_final.apply_offsets(
                        cases=hind_usable, quantiles=adj_h, table=table, keys=keys
                    ).astype(np.float32)
                )
            print(arm, "seed", seed, flush=True)
    for name, seeds in results.items():
        np.save(OUT_DIR / f"{name}.npy", np.stack(seeds))
    for name, seeds in hindcast_results.items():
        np.save(HINDCAST_DIR / f"{name}.npy", np.stack(seeds))
    (OUT_DIR / "hindcast_scores.json").write_text(json.dumps(hindcast_scores, indent=1))
    if (common.FINAL_285_DIR / "xgb_B0.npy").exists() and "P0" in results:
        for mine, theirs in (("P0", "xgb_B0"), ("P1", "xgb_B4")):
            difference = np.abs(np.stack(results[mine]) - np.load(common.FINAL_285_DIR / f"{theirs}.npy"))
            print(f"reproduction check {mine} vs 285 {theirs}: max abs difference", difference.max())


if __name__ == "__main__":
    main()
