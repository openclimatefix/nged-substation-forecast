"""Fit the capacity with and without an injected farm on all rows and on clear half-hours only.

For each Primary and farm, the script adds the farm's output at 50% of the Primary's pinned capacity
and builds one Gram matrix per calendar month of training rows, for the base series and the injected
series, once over all rows and once over the clear, high-sun half-hours only. Every month-block
bootstrap resample is a weighted sum of those monthly matrices followed by one bounded
least-squares solve. The method follows experiment 231; the plan is in
`docs/ml-experiments/00236-pv-capacity-clear-sky-fit/result.md`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00236-*/e00236_fit.py --sites D10
"""

import argparse
from dataclasses import dataclass, replace
from typing import Final

import e00236_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    DEFAULT_MAX_ITERATIONS,
    GamLayout,
    GamSpec,
    derated_unit_shape,
    design_matrix,
    penalty_matrix,
    prepare_fit,
    row_chunks,
    solve_bounded_quadratic,
)
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

SOLVE_BATCH: Final[int] = 25
"""Bounded solves per call, which keeps the batch of Gram matrices within memory."""


@dataclass(frozen=True)
class MonthlyStats:
    """The sums every fit is built from, one entry per calendar month.

    Attributes:
        gram: `X^T X` per month, shape (M, p, p).
        rhs: `X^T y` per month for the base and the injected series, shape (M, 2, p).
        farm_cross: Per month, the sum over read-out season-year rows of the farm's normalised
            output times the derated unit shape, shape (M,).
        farm_square: The same sum of the squared derated unit shape, shape (M,).
        n_rows: The training rows in each month.
    """

    gram: torch.Tensor
    rhs: torch.Tensor
    farm_cross: np.ndarray
    farm_square: np.ndarray
    n_rows: np.ndarray


def row_months(*, cache: FeatureCache, rows: np.ndarray) -> np.ndarray:
    """Return each row's calendar month, as months since 1970-01, from the half-hour's start."""
    start = cache.end_times[rows] - np.timedelta64(30, "m")
    return start.astype("datetime64[M]").astype(np.int64)


def monthly_stats(
    *,
    cache: FeatureCache,
    site: str,
    layout: GamLayout,
    index: np.ndarray,
    months: np.ndarray,
    output: np.ndarray,
    norm: np.ndarray,
) -> MonthlyStats:
    """Sum the Gram matrices, the right-hand sides, and the farm-only regression by month."""
    features = cache.sites[site]
    slot = row_months(cache=cache, rows=index) - months[0]
    p = layout.n_columns
    gram = torch.zeros((len(months), p, p), dtype=torch.float64)
    rhs = torch.zeros((len(months), 2, p), dtype=torch.float64)
    base = features.y[index]
    injected = base - np.nan_to_num(output[index])
    for chunk in row_chunks(index=index):
        rows = index[chunk]
        x = torch.as_tensor(
            design_matrix(cache=cache, features=features, layout=layout, index=rows),
            dtype=torch.float64,
        )
        targets = torch.as_tensor(np.stack([base[chunk], injected[chunk]]), dtype=torch.float64)
        for month in np.unique(slot[chunk]):
            members = torch.as_tensor(np.flatnonzero(slot[chunk] == month))
            xm = x[members]
            gram[month] += xm.T @ xm
            rhs[month] += targets[:, members] @ xm
    shape = derated_unit_shape(features=features, index=index)
    in_year = cache.calendar.season_year[index] == common.READ_OUT_SEASON_YEAR
    return MonthlyStats(
        gram=gram,
        rhs=rhs,
        farm_cross=np.bincount(
            slot[in_year], weights=(norm[index] * shape)[in_year], minlength=len(months)
        ),
        farm_square=np.bincount(slot[in_year], weights=(shape**2)[in_year], minlength=len(months)),
        n_rows=np.bincount(slot, minlength=len(months)),
    )


def month_block_counts(*, months: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Return month weights of shape (n_resamples + 1, M); row 0 is all ones (the point fit).

    Each resample draws, within each (season-year, quarter of the year) stratum, as many months as
    the stratum has, with replacement, so that no season-year loses its summer.
    """
    quarter = (months % 12) // 3
    stratum = ((months + 6) // 12) * 4 + quarter
    generator = np.random.default_rng(seed)
    counts = np.ones((n_resamples + 1, len(months)))
    for resample in range(1, n_resamples + 1):
        counts[resample] = 0.0
        for value in np.unique(stratum):
            members = np.flatnonzero(stratum == value)
            drawn = generator.choice(members, size=len(members), replace=True)
            counts[resample] += np.bincount(drawn, minlength=len(months))
    return counts


def solve_in_batches(
    *, matrix: torch.Tensor, rhs: torch.Tensor, constrained: np.ndarray
) -> np.ndarray:
    """Solve the bounded problems in batches of `SOLVE_BATCH`; return theta, shape (batch, p)."""
    flag = torch.as_tensor(constrained)
    pieces = []
    for start in range(0, rhs.shape[0], SOLVE_BATCH):
        theta, _ = solve_bounded_quadratic(
            matrix=matrix[start : start + SOLVE_BATCH],
            rhs=rhs[start : start + SOLVE_BATCH],
            constrained=flag,
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )
        pieces.append(theta.numpy())
    return np.concatenate(pieces)


def recovery(
    *,
    stats: MonthlyStats,
    layout: GamLayout,
    weights: np.ndarray,
    spec_cost: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the base and the injected series under each weighting.

    Args:
        stats: The monthly sums.
        layout: The columns.
        weights: Month weights, shape (B, M).
        spec_cost: The penalty matrix and the linear cost vector.

    Returns:
        The capacity at the read-out season-year of the base fit and of the injected fit, shape
        (B,) each, and the farm-only scale on the same months, shape (B,).
    """
    penalty, cost = spec_cost
    w = torch.as_tensor(weights, dtype=torch.float64)
    gram = torch.einsum("bm,mpq->bpq", w, stats.gram)
    rhs = torch.einsum("bm,mkp->bkp", w, stats.rhs)
    matrix = gram + torch.as_tensor(penalty, dtype=torch.float64)[None]
    shifted = rhs - 0.5 * torch.as_tensor(cost, dtype=torch.float64)[None, None, :]
    b = weights.shape[0]
    theta = solve_in_batches(
        matrix=torch.cat([matrix, matrix]),
        rhs=torch.cat([shifted[:, 0], shifted[:, 1]]),
        constrained=layout.constrained,
    )
    capacity = layout.slices["capacity"]
    columns = common.READ_OUT_COLUMNS
    cap_base = theta[:b][:, capacity][:, :columns].sum(axis=1)
    cap_injected = theta[b:][:, capacity][:, :columns].sum(axis=1)
    farm_only = (weights @ stats.farm_cross) / (weights @ stats.farm_square)
    return cap_base, cap_injected, farm_only


def run_farm(
    *, cache: FeatureCache, site: str, farm: str, n_resamples: int, injected_points: float
) -> list[dict]:
    """Return the result rows of one Primary and farm, for all rows and for clear rows."""
    features = cache.sites[site]
    norm = common.farm_norm(cache=cache, farm=farm)
    output = injected_points * norm
    masked = replace(cache, sites={**cache.sites, site: replace(features, y=features.y - output)})
    spec = GamSpec(ridge=common.RIDGE, capacity="fitted")
    layout, index = prepare_fit(
        cache=masked,
        site=site,
        spec=spec,
        cut=common.CUT,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
    )
    ghi, clear_sky = common.clear_sky_index(cache=cache, site=site)
    clear = (
        (clear_sky[index] >= common.CLEAR_SKY_MIN_W_M2)
        & (ghi[index] >= common.CLEAR_INDEX * clear_sky[index])
        & (features.elevation_deg[index] >= common.MIN_ELEVATION_DEG)
    )
    all_months = row_months(cache=cache, rows=index)
    months = np.arange(all_months.min(), all_months.max() + 1)
    counts = month_block_counts(
        months=months, n_resamples=n_resamples, seed=1000 + common.ALL_SITES.index(site)
    )
    spec_cost = penalty_matrix(layout=layout, n_rows=len(index))
    rows: list[dict] = []
    for row_set, rows_used in (("all", index), ("clear", index[clear])):
        stats = monthly_stats(
            cache=cache,
            site=site,
            layout=layout,
            index=rows_used,
            months=months,
            output=output,
            norm=norm,
        )
        cap_base, cap_injected, farm_only = recovery(
            stats=stats, layout=layout, weights=counts, spec_cost=spec_cost
        )
        in_year = cache.calendar.season_year[rows_used] == common.READ_OUT_SEASON_YEAR
        for name, values in (
            ("recovery", (cap_injected - cap_base) / injected_points),
            ("farm_only", farm_only),
        ):
            rows.extend(
                {
                    "site": site,
                    "farm": farm,
                    "row_set": row_set,
                    "stat": name,
                    "b": b,
                    "value": float(v),
                }
                for b, v in enumerate(values)
            )
        rows.extend(
            {"site": site, "farm": farm, "row_set": row_set, "stat": name, "b": 0, "value": value}
            for name, value in (
                ("cap_base", float(cap_base[0])),
                ("cap_injected", float(cap_injected[0])),
                ("injected_points", injected_points),
                ("n_rows", float(len(rows_used))),
                ("n_rows_read_out_year", float(in_year.sum())),
            )
        )
    return rows


def main() -> None:
    """Write `work/clear_<site>.parquet` for each Primary."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    parser.add_argument("--farms", nargs="+", default=list(common.FARMS))
    parser.add_argument("--resamples", type=int, default=common.N_RESAMPLES)
    parser.add_argument("--suffix", default="")
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    for site in arguments.sites:
        pinned = pinned_capacity(
            vintages=read_vintages(), table=cache.sites[site].table, cut=common.CUT
        ).held_points
        injected_points = common.INJECTED_SHARE * pinned
        table: list[dict] = []
        for farm in arguments.farms:
            table += run_farm(
                cache=cache,
                site=site,
                farm=farm,
                n_resamples=arguments.resamples,
                injected_points=injected_points,
            )
            print(f"{site} farm {farm}: done", flush=True)
        pl.DataFrame(table).write_parquet(
            common.WORK_DIR / f"clear_{site}{arguments.suffix}.parquet"
        )


if __name__ == "__main__":
    main()
