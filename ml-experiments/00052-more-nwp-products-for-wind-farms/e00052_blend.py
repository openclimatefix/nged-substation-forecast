"""Blend the best guess with each product's quantiles and write the arms.

Reads experiment 33's stored `bg_full` wind table (read only) and `e00052_fit.py`'s product
quantiles. For runs initialised in month m, the weights use only cases whose run was initialised
in an earlier month and whose valid time is before the first day of m, and only cases that hold
every product of the arm. A case that lacks a product renormalises the weights over the members it
has. A spread factor is then fitted on the blend, per lead group, on earlier cases that hold a
product, and applied to cases that hold a product.

Output: `work/arms/<arm>.parquet` (all seeds), `work/weights_<arm>.parquet`, and
`work/spread_<arm>.parquet`.

    uv run python ml-experiments/00052-more-nwp-products-for-wind-farms/e00052_blend.py
"""

import itertools
from datetime import UTC, datetime
from typing import Final

import e00052_common as common
import numpy as np
import polars as pl
from e00052_common import KEY_COLUMNS, LEVEL_COLUMNS

ARMS_DIR = common.WORK_DIR / "arms"
GRID_STEP: Final[int] = 4
"""Weights are multiples of 1/4."""
SHORT_BANDS: Final[frozenset[int]] = frozenset({0, 1})
"""The spread factor is fitted on days 0 to 2 (bands 0 and 1) and on the rest separately."""
ARMS: Final[dict[str, tuple[str, ...]]] = {
    "ens_icon_ifs": ("icon", "ifs"),
    "ens_icon": ("icon",),
    "ens_ifs": ("ifs",),
    "ens_gfs": ("gfs",),
}
PRIMARY: Final[str] = "ens_icon_ifs"
PLACEBO_SEED: Final[int] = 52


def weight_grid(*, members: int) -> np.ndarray:
    """Return every weight vector on the grid that sums to 1, shape (vectors, members)."""
    rows = [
        v for v in itertools.product(range(GRID_STEP + 1), repeat=members) if sum(v) == GRID_STEP
    ]
    return np.array(rows, dtype=np.float64) / GRID_STEP


def vincent(*, member_quantiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the weighted average of each quantile level, sorted along the levels.

    Args:
        member_quantiles: Shape (members, cases, 9); a member a case lacks holds NaN.
        weights: Shape (members,) or (cases, members). A missing member's weight is dropped and the
            rest renormalised; a case left with no weight takes member 0 (the best guess).

    Returns:
        The blended quantiles, shape (cases, 9).
    """
    present = ~np.isnan(member_quantiles[:, :, 0])
    per_case = np.broadcast_to(weights, (member_quantiles.shape[1], member_quantiles.shape[0]))
    effective = per_case * present.T
    total = effective.sum(axis=1)
    effective[total <= 0, 0] = 1.0
    effective /= effective.sum(axis=1, keepdims=True)
    values = np.nan_to_num(member_quantiles.astype(np.float64))
    return np.sort(np.einsum("cm,mcl->cl", effective, values), axis=1)


def load_cases() -> pl.DataFrame:
    """Return the best guess's wind rows of all seeds, with month, lead day, and band."""
    frame = (
        pl.read_parquet(common.BEST_GUESS_PATH)
        .filter(pl.col("technology") == "wind")
        .filter(
            pl.all_horizontal(pl.col(*LEVEL_COLUMNS).is_finite())
            & pl.col("actual_norm").is_finite()
            & (pl.col("valid_time") < common.HOLDOUT)
        )
        .with_columns(
            lead_day=common.lead_day(
                power_fcst_init_time=pl.col("power_fcst_init_time"),
                valid_time=pl.col("valid_time"),
            ),
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            seed=pl.col("seed").cast(pl.Int8),
        )
        .sort("seed", *KEY_COLUMNS)
    )
    bands = [name for name, _, _ in common.LEAD_DAY_BANDS]
    names = common.band_of(days=frame["lead_day"].to_numpy())
    return frame.with_columns(band=pl.Series([bands.index(n) for n in names]))


def product_array(*, cases: pl.DataFrame, product: str) -> np.ndarray:
    """Return the product's quantiles aligned to `cases`, NaN where the product is absent."""
    tables = [
        pl.read_parquet(common.WORK_DIR / "product_quantiles" / f"{product}_seed{seed}.parquet")
        .with_columns(seed=pl.lit(seed, dtype=pl.Int8))
        .select("seed", *KEY_COLUMNS, *LEVEL_COLUMNS)
        for seed in common.SEEDS
    ]
    joined = cases.select("seed", *KEY_COLUMNS).join(
        pl.concat(tables), on=["seed", *KEY_COLUMNS], how="left", maintain_order="left"
    )
    return joined.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)


def placebo(*, cases: pl.DataFrame, array: np.ndarray) -> np.ndarray:
    """Shuffle the product's rows among cases of one farm, seed, month, lead day, and hour."""
    rng = np.random.default_rng(PLACEBO_SEED)
    groups = cases.select(
        "seed", "time_series_id", "month", "lead_day", hour=pl.col("valid_time").dt.hour()
    ).with_row_index()
    out = array.copy()
    for part in groups.partition_by("seed", "time_series_id", "month", "lead_day", "hour"):
        index = part["index"].to_numpy()
        present = index[~np.isnan(array[index, 0])]
        out[present] = array[rng.permutation(present)]
    return out


def choose_weights(
    *, cases: pl.DataFrame, members: np.ndarray, actual: np.ndarray, by_band: bool
) -> pl.DataFrame:
    """Return the rolling-origin weight vector for each seed-independent month and lead band.

    Weights pool the three seeds. Ties go to the vector nearest equal weights.
    """
    count = members.shape[0]
    grid = weight_grid(members=count)
    equal = np.full(count, 1.0 / count)
    nearest_equal = np.argsort(np.abs(grid - equal).sum(axis=1), kind="stable")
    complete = ~np.isnan(members[:, :, 0]).any(axis=0)
    scores = np.full((members.shape[1], len(grid)), np.nan)
    for k, w in enumerate(grid):
        scores[complete, k] = common.quantile_score(
            quantiles=vincent(member_quantiles=members[:, complete], weights=w),
            actual=actual[complete],
        )
    month_col = cases["month"].to_numpy()
    valid = cases["valid_time"].to_numpy().astype("datetime64[us]")
    band = cases["band"].to_numpy()
    rows = []
    for month in sorted(set(month_col)):
        start = np.datetime64(
            datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC).replace(tzinfo=None), "us"
        )
        history = (month_col < month) & (valid < start) & complete
        for b in range(len(common.LEAD_DAY_BANDS)) if by_band else [-1]:
            mask = history & ((band == b) if by_band else True)
            n = int(mask.sum())
            if n == 0:
                chosen = equal
            else:
                mean = scores[mask].mean(axis=0)
                chosen = grid[nearest_equal[np.argsort(mean[nearest_equal], kind="stable")][0]]
            rows.append(
                {"month": month, "band": b, "history_cases": n}
                | {f"w{i}": float(w) for i, w in enumerate(chosen)}
            )
    return pl.DataFrame(rows)


def blend_with(
    *, cases: pl.DataFrame, members: np.ndarray, table: pl.DataFrame, by_band: bool
) -> np.ndarray:
    """Return the blend of the members under a weight table, per case."""
    keyed = cases.select("month", band=pl.col("band") if by_band else pl.lit(-1))
    joined = keyed.join(table, on=["month", "band"], how="left", validate="m:1")
    weights = joined.select([f"w{i}" for i in range(members.shape[0])]).to_numpy()
    if np.isnan(weights).any():
        msg = "a case has no weight row"
        raise ValueError(msg)
    return vincent(member_quantiles=members, weights=weights)


def apply_spread(
    *, cases: pl.DataFrame, quantiles: np.ndarray, actual: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, pl.DataFrame]:
    """Return the quantiles after a spread factor fitted month by month on earlier cases.

    Only `eligible` cases (those holding a product) are fitted on and changed. A month's factor is
    fitted on cases whose valid time is at least 3.5 hours before the month's first run.
    """
    valid = cases["valid_time"].to_numpy().astype("datetime64[us]")
    init = cases["power_fcst_init_time"].to_numpy().astype("datetime64[us]")
    month = cases["month"].to_numpy()
    short = np.isin(cases["band"].to_numpy(), list(SHORT_BANDS))
    scorable = quantiles[:, common.UPPER_INDEX] > common.TRIVIAL_LEVEL
    out = quantiles.copy()
    rows = []
    for label in sorted(set(month)):
        in_month = month == label
        cutoff = init[in_month].min()
        for name, group in (("days 0-2", short), ("days 3+", ~short)):
            target = in_month & group & eligible
            fit_index = np.flatnonzero(
                (valid <= cutoff - common.TELEMETRY_DELAY) & group & scorable & eligible
            )
            factor = 1.0
            if target.any() and fit_index.size >= common.MIN_FIT_CASES:
                factor = common.fit_spread_factor(
                    quantiles=quantiles[fit_index], actual=actual[fit_index]
                )
                out[target] = common.scale_about_median(quantiles=quantiles[target], factor=factor)
            rows.append(
                {"month": label, "group": name, "factor": factor, "fit_cases": int(fit_index.size)}
            )
    return out, pl.DataFrame(rows)


def write(*, name: str, cases: pl.DataFrame, quantiles: np.ndarray) -> None:
    """Write one arm's table (all seeds)."""
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    frame = cases.select("seed", *KEY_COLUMNS, "lead_day", "band", "actual_norm").hstack(
        pl.DataFrame(quantiles.astype(np.float32), schema=list(LEVEL_COLUMNS))
    )
    target = ARMS_DIR / f"{name}.parquet"
    frame.write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)


def main() -> None:
    """Write the best-guess reference and every blend arm."""
    cases = load_cases()
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    best_guess = cases.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    write(name="ens", cases=cases, quantiles=best_guess)
    products = {p: product_array(cases=cases, product=p) for p in common.PRODUCTS}
    for p, array in products.items():
        write(name=f"only_{p}", cases=cases, quantiles=np.where(np.isnan(array), best_guess, array))
    runs = {name: tuple(parts) for name, parts in ARMS.items()}
    runs["placebo"] = ARMS[PRIMARY]
    for name, parts in runs.items():
        arrays = [best_guess, *[products[p] for p in parts]]
        if name == "placebo":
            arrays = [best_guess, *[placebo(cases=cases, array=a) for a in arrays[1:]]]
        members = np.stack(arrays)
        eligible = (~np.isnan(members[1:, :, 0])).any(axis=0)
        for by_band, suffix in ((True, ""), (False, "_one")):
            table = choose_weights(cases=cases, members=members, actual=actual, by_band=by_band)
            table.write_parquet(common.WORK_DIR / f"weights_{name}{suffix}.parquet")
            blended = blend_with(cases=cases, members=members, table=table, by_band=by_band)
            blended = np.where(eligible[:, None], blended, best_guess)
            write(name=f"{name}{suffix}_nospread", cases=cases, quantiles=blended)
            spread, factors = apply_spread(
                cases=cases, quantiles=blended, actual=actual, eligible=eligible
            )
            factors.write_parquet(common.WORK_DIR / f"spread_{name}{suffix}.parquet")
            write(name=f"{name}{suffix}", cases=cases, quantiles=spread)
        print(f"arm {name} done", flush=True)
    # Each product alone, spread-corrected, for the stop rule.
    for p, array in products.items():
        present = ~np.isnan(array[:, 0])
        spread, _ = apply_spread(
            cases=cases, quantiles=np.nan_to_num(array), actual=actual, eligible=present
        )
        write(
            name=f"only_{p}_spread",
            cases=cases,
            quantiles=np.where(present[:, None], spread, best_guess),
        )


if __name__ == "__main__":
    main()
