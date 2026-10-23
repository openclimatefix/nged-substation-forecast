"""Part B of experiment 00285, second step: spread factor, offset table, and scored quantiles.

Reads the member mixtures that `e00285_ens.py` stored, applies experiment 33's spread factor
(fitted month by month on earlier cases, over the hindcast and scored cases together in time order,
as in experiment 00259), builds the offset table (B4 on the raw members, B5 on each corrected
version), and stores the scored quantiles of every arm.

Output: `work/ens_final/<model>_<arm>.npy`, shape (3 seeds, scored cases, 11), in the order of
`e00285_score.ordered_cases`; and `work/ens_final/hindcast_scores.json`, each arm's quantile score
on the hindcast cases (used to pick the best of B1 to B3 without looking at the scored year).

    uv run --no-sync python e00285_ens_final.py
"""

import json
import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00285_common as common  # noqa: E402
import e00285_ens as ens  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

OUT_DIR: Final = common.WORK_DIR / "ens_final"
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
MODELS: Final[tuple[str, ...]] = ("core", "xgb")
CORRECTIONS: Final[tuple[str, ...]] = ("B0", "B1", "B2", "B3")


def hindcast_cases(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the hindcast cases (key columns, farm, observed fraction, lead), farm by farm."""
    paths = sorted(common.HINDCAST_DIR.glob("init=*.parquet"))
    actual = common.observed_training_power(sites=sites).rename({"target": "actual_norm"})
    pieces = []
    for row in sites.iter_rows(named=True):
        keys = (
            pl.concat(
                [
                    pl.scan_parquet(p)
                    .filter(
                        (pl.col("ensemble_member") == 0)
                        & (pl.col("time_series_id") == row["time_series_id"])
                    )
                    .select(common.KEY_COLUMNS)
                    for p in paths
                ]
            )
            .collect()
            .sort(common.KEY_COLUMNS)
        )
        pieces.append(
            keys.join(actual, on=["time_series_id", "valid_time"], how="left", validate="m:1")
            .with_columns(
                site=pl.lit(row["site"]),
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0,
            )
            .with_columns(
                lead_day=common.lead_day(
                    power_fcst_init_time=pl.col("power_fcst_init_time"),
                    valid_time=pl.col("valid_time"),
                ).clip(0, ens.MAX_LEAD_DAY)
            )
        )
    return pl.concat(pieces)


def scored_cases(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the scored cases, farm by farm, with the lead day clipped as the hindcast's is."""
    cases = common.cases_table(sites=sites)
    return pl.concat(
        [cases.filter(pl.col("site") == row["site"]) for row in sites.iter_rows(named=True)]
    ).with_columns(lead_day=pl.col("lead_day").clip(0, ens.MAX_LEAD_DAY))


def load_mixture(*, model: str, arm: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the (hindcast, scored) mixtures of one arm, farms stacked in order."""
    core_seed = seed if arm == "B3" else 0
    use = core_seed if model == "core" else seed
    parts = {
        name: np.concatenate(
            [np.load(ens.OUT_DIR / f"{model}_{arm}_s{use}_{name}_{farm}.npy") for farm in FARMS]
        )
        for name in ("hindcast", "scored")
    }
    return parts["hindcast"], parts["scored"]


def offsets_table(
    *, hindcast: pl.DataFrame, quantiles: np.ndarray
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Return the pinball-optimal offsets per (farm, lead day): shape (groups, 11), and the keys."""
    frame = hindcast.select("site", "lead_day").hstack(
        pl.DataFrame(
            (hindcast["actual_norm"].to_numpy()[:, None] - quantiles).astype(np.float64),
            schema=list(common.LEVEL_COLUMNS),
        )
    )
    grouped = frame.group_by("site", "lead_day").agg(
        [
            pl.col(q).quantile(level, interpolation="linear")
            for q, level in zip(common.LEVEL_COLUMNS, common.LEVELS, strict=True)
        ]
    )
    keys = [(row[0], row[1]) for row in grouped.select("site", "lead_day").iter_rows()]
    return grouped.select(common.LEVEL_COLUMNS).to_numpy(), keys


def apply_offsets(
    *,
    cases: pl.DataFrame,
    quantiles: np.ndarray,
    table: np.ndarray,
    keys: list[tuple[str, int]],
) -> np.ndarray:
    """Return `quantiles` plus the offsets of each case's (farm, lead day), clipped and sorted."""
    index = {key: i for i, key in enumerate(keys)}
    rows = np.array(
        [
            index.get((site, day), -1)
            for site, day in zip(cases["site"].to_list(), cases["lead_day"].to_list(), strict=True)
        ]
    )
    correction = np.where((rows >= 0)[:, None], table[np.maximum(rows, 0)], 0.0)
    return np.sort(np.clip(quantiles + correction, 0.0, 1.0), axis=1).astype(np.float32)


def main() -> None:
    """Write the scored quantiles of every Part B arm."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table()
    hind = hindcast_cases(sites=sites)
    scored = scored_cases(sites=sites)
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
    results: dict[tuple[str, str], list[np.ndarray]] = {}
    hindcast_scores: dict[str, float] = {}
    for model in MODELS:
        for arm in CORRECTIONS:
            for seed in (0, 1, 2):
                raw_h, raw_s = load_mixture(model=model, arm=arm, seed=seed)
                adjusted = common.apply_spread(
                    cases=all_cases,
                    mixture=np.concatenate([raw_h[usable], raw_s]).astype(np.float64),
                )
                adj_h, adj_s = adjusted[: hind_usable.height], adjusted[hind_usable.height :]
                results.setdefault((model, arm), []).append(adj_s.astype(np.float32))
                table, keys = offsets_table(hindcast=hind_usable, quantiles=adj_h)
                stacked = apply_offsets(cases=scored, quantiles=adj_s, table=table, keys=keys)
                offset_arm = "B4" if arm == "B0" else f"B5_{arm}"
                results.setdefault((model, offset_arm), []).append(stacked)
                if seed == 0:
                    hindcast_scores[f"{model}_{arm}"] = float(
                        common.quantile_score(
                            quantiles=adj_h.astype(np.float64), actual=hind_actual
                        ).mean()
                        * common.POINTS_PER_FRACTION
                    )
                print(f"{model} {arm} seed {seed}", flush=True)
    for (model, arm), seeds in results.items():
        np.save(OUT_DIR / f"{model}_{arm}.npy", np.stack(seeds))
    (OUT_DIR / "hindcast_scores.json").write_text(json.dumps(hindcast_scores, indent=1))


if __name__ == "__main__":
    main()
