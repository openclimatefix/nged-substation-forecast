"""Build the pooled mixture of experiment 00002's ERA5-trained stage 1a and fit the corrections.

For each seed the 51 members' 9 quantiles are pooled into a mixture quantile function per scored
case (cached under `work/mixture/`). Each of the four correction arms is then fitted month by
month on the cases whose valid times precede the first run of the scored month (the cases of that
month are the runs whose initialisation time falls in it), so a fit sees no outcome later than the
earliest forecast it will be applied to. A fit uses only non-trivial cases, those whose uncorrected
95% quantile exceeds `TRIVIAL_LEVEL`; trivial cases (night-time solar zeros) are left uncorrected.
With fewer than `MIN_FIT_CASES` such cases an arm returns the uncorrected mixture.

Output: one prediction set per arm under `data/ml-experiments/00016-.../predictions/` (columns:
keys, `seed`, `lead_hours`, `actual_norm`, `technology`, `band`, `trivial`, the 9 quantile
columns), and `work/parameters.parquet` with every fitted parameter.

    uv run python ml-experiments/00016-*/e00016_fit.py
"""

import numpy as np
import polars as pl
from e00016_common import (
    ARMS,
    CASE_ORDER,
    ENS_VALIDATION_DIR,
    EXPERIMENT_ID,
    KEY_COLUMNS,
    LEAD_BANDS,
    LEVEL_COLUMNS,
    MAX_FIT_CASES,
    MEMBER_QUANTILES_DIR,
    MEMBERS,
    MIN_FIT_CASES,
    ROW_SET_ID,
    SEEDS,
    SUBSAMPLE_SEED,
    TECHNOLOGIES,
    TRIVIAL_LEVEL,
    UPPER_INDEX,
    WORK_DIR,
    ArmType,
    add_level_offsets,
    crps_estimate,
    fit_level_offsets,
    fit_location_and_scale,
    fit_spread_factor,
    mixture_quantiles,
    read_levels,
    scale_about_median,
    validation_cases,
)

from studies import experiment_store


def mixture_for_seed(*, seed: int, cases: pl.DataFrame) -> np.ndarray:
    """Return the 9 pooled-mixture quantiles of every case, in `cases` order (cached on disk).

    Raises:
        ValueError: If a case lacks a member, or the case count differs from `cases`.
    """
    cache = WORK_DIR / "mixture" / f"seed{seed}.npy"
    if cache.exists():
        return np.load(cache)
    keys = cases.select(KEY_COLUMNS)
    pieces = []
    key_pieces = []
    for path in sorted(ENS_VALIDATION_DIR.glob("init=*.parquet")):
        members = (
            pl.read_parquet(MEMBER_QUANTILES_DIR / f"seed{seed}" / path.name)
            .join(keys, on=list(KEY_COLUMNS), how="inner")
            .sort(*CASE_ORDER, "ensemble_member")
        )
        if members.height % MEMBERS:
            msg = f"{path.name}: a case lacks some of the {MEMBERS} members"
            raise ValueError(msg)
        key_pieces.append(
            members.filter(pl.col("ensemble_member") == members["ensemble_member"].min()).select(
                KEY_COLUMNS
            )
        )
        matrix = members.select(LEVEL_COLUMNS).to_numpy().reshape(-1, MEMBERS, len(LEVEL_COLUMNS))
        pieces.append(
            read_levels(fine=mixture_quantiles(member_quantiles=matrix.astype(np.float32)))
        )
    mixture = np.concatenate(pieces)
    if mixture.shape[0] != cases.height or not pl.concat(key_pieces).equals(keys):
        msg = "the mixtures are not in the order of the scored cases"
        raise ValueError(msg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, mixture)
    return mixture


def _subsample(*, index: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Return at most `MAX_FIT_CASES` of the indices, drawn without replacement."""
    if index.size <= MAX_FIT_CASES:
        return index
    return generator.choice(index, MAX_FIT_CASES, replace=False)


def correct(
    *, cases: pl.DataFrame, mixture: np.ndarray, seed: int
) -> tuple[dict[ArmType, np.ndarray], pl.DataFrame]:
    """Fit and apply every arm month by month; return each arm's quantiles and the parameters."""
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    valid = cases["valid_time"].to_numpy()
    init = cases["power_fcst_init_time"].to_numpy()
    technology = cases["technology"].to_numpy()
    band = cases["band"].to_numpy()
    month = cases["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    scorable = mixture[:, UPPER_INDEX] > TRIVIAL_LEVEL
    out: dict[ArmType, np.ndarray] = {arm: mixture.copy() for arm in ARMS}
    generator = np.random.default_rng(SUBSAMPLE_SEED + seed)
    parameters = []
    for month_label in sorted(set(month)):
        in_month = month == month_label
        cutoff = init[in_month].min()
        for tech in TECHNOLOGIES:
            for group_name, group_bands in (
                ("all", [None]),
                ("band", [name for name, _, _ in LEAD_BANDS]),
            ):
                for name in group_bands:
                    target = in_month & (technology == tech) & scorable
                    fit_scope = (valid < cutoff) & (technology == tech) & scorable
                    if name is not None:
                        target &= band == name
                        fit_scope &= band == name
                    if not target.any():
                        continue
                    fit_index = np.flatnonzero(fit_scope)
                    base = {
                        "seed": seed,
                        "month": month_label,
                        "technology": tech,
                        "band": name or "all",
                        "fit_cases": int(fit_index.size),
                    }
                    if fit_index.size < MIN_FIT_CASES:
                        parameters.append({**base, "arm": group_name, "fitted": False})
                        continue
                    fit_q, fit_a = mixture[fit_index], actual[fit_index]
                    sampled = _subsample(index=fit_index, generator=generator)
                    if group_name == "all":
                        factor = fit_spread_factor(quantiles=fit_q, actual=fit_a)
                        out["spread"][target] = scale_about_median(
                            quantiles=mixture[target], shift=0.0, factor=factor
                        )
                        parameters.append(
                            {**base, "arm": "spread", "fitted": True, "factor": factor}
                        )
                        continue
                    factor = fit_spread_factor(quantiles=fit_q, actual=fit_a)
                    out["spread_band"][target] = scale_about_median(
                        quantiles=mixture[target], shift=0.0, factor=factor
                    )
                    parameters.append(
                        {**base, "arm": "spread_band", "fitted": True, "factor": factor}
                    )
                    shift, scale = fit_location_and_scale(
                        quantiles=mixture[sampled], actual=actual[sampled]
                    )
                    out["locscale"][target] = scale_about_median(
                        quantiles=mixture[target], shift=shift, factor=scale
                    )
                    parameters.append(
                        {**base, "arm": "locscale", "fitted": True, "shift": shift, "factor": scale}
                    )
                    offsets = fit_level_offsets(quantiles=fit_q, actual=fit_a)
                    out["free_quantile"][target] = add_level_offsets(
                        quantiles=mixture[target], offsets=offsets
                    )
                    parameters.append(
                        {
                            **base,
                            "arm": "free_quantile",
                            "fitted": True,
                            "median_offset": float(offsets[4]),
                        }
                    )
    return out, pl.DataFrame(parameters, infer_schema_length=None)


def _check_reproduction(*, none: pl.DataFrame) -> None:
    """Raise unless `none` reproduces experiment 00002's mean CRPS per technology.

    Raises:
        ValueError: If a technology's mean CRPS differs by more than 0.05 points of capacity from
            11.11 (wind) or 5.05 (solar).
    """
    quantiles = none.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    crps = 100.0 * crps_estimate(quantiles=quantiles, actual=none["actual_norm"].to_numpy())
    for tech, published in (("wind", 11.11), ("solar", 5.05)):
        here = float(crps[(none["technology"] == tech).to_numpy()].mean())
        print(f"{tech}: none scores {here:.3f}, experiment 00002 {published}", flush=True)
        if abs(here - published) > 0.05:
            msg = f"{tech}: none scores {here:.3f}, experiment 00002 scored {published}"
            raise ValueError(msg)


def main() -> None:
    """Fit every arm for every seed and save the prediction sets and parameters."""
    cases = validation_cases()
    frames: dict[ArmType, list[pl.DataFrame]] = {arm: [] for arm in ARMS}
    all_parameters = []
    for seed in SEEDS:
        mixture = mixture_for_seed(seed=seed, cases=cases)
        arms, parameters = correct(cases=cases, mixture=mixture, seed=seed)
        all_parameters.append(parameters)
        trivial = pl.Series("trivial", mixture[:, UPPER_INDEX] <= TRIVIAL_LEVEL)
        for arm, quantiles in arms.items():
            frames[arm].append(
                cases.with_columns(seed=pl.lit(seed), trivial=trivial).hstack(
                    pl.DataFrame(quantiles.astype(np.float32), schema=list(LEVEL_COLUMNS))
                )
            )
        print(f"seed {seed} done", flush=True)
    _check_reproduction(none=pl.concat(frames["none"]))
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(all_parameters, how="diagonal").write_parquet(WORK_DIR / "parameters.parquet")
    for arm, pieces in frames.items():
        experiment_store.write_predictions(
            experiment_id=EXPERIMENT_ID,
            name=arm.replace("_", "-"),
            frame=pl.concat(pieces),
            kind="quantiles",
            row_set_id=ROW_SET_ID,
        )


if __name__ == "__main__":
    main()
