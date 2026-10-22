"""Run the ladder of known-answer fits of experiment 231, one Primary at a time.

For each Primary and farm, the script adds the farm's output at 50% of the Primary's pinned capacity
and builds one Gram matrix per calendar month of training rows for the base series and the injected
series. Every rung of the ladder, and every month-block bootstrap resample of it, is then a weighted
sum of those monthly matrices followed by one bounded least-squares solve. The rungs are described
in `docs/ml-experiments/00231-pv-capacity-audit-where-recovery-goes/result.md`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00231-*/e00231_fit.py --sites D10
"""

import argparse
from dataclasses import dataclass, replace
from typing import Final

import e00231_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    DEFAULT_MAX_ITERATIONS,
    GamLayout,
    GamSpec,
    accumulate_gram,
    derated_unit_shape,
    design_matrix,
    fit_gam,
    penalty_matrix,
    prepare_fit,
    row_chunks,
    solve_bounded_quadratic,
)
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

SOLVE_BATCH: Final[int] = 25
"""Bounded solves per call, which keeps the batch of Gram matrices within memory."""
WIRING_TOLERANCE: Final[float] = 1e-8
RECOVERY_TOLERANCE: Final[float] = 0.005


@dataclass(frozen=True)
class MonthlyStats:
    """The sums every rung is built from, one entry per calendar month of training rows.

    Attributes:
        months: Each month as months since 1970-01.
        gram: `X^T X` per month, shape (M, p, p).
        rhs: `X^T y` per month for the base and the injected series, shape (M, 2, p).
        farm_cross: Per month, the sum over read-out season-year rows of the farm's normalised
            output times the derated unit shape: all daylight rows first, clear half-hours second.
            Shape (2, M).
        farm_square: The same sums of the squared derated unit shape, shape (2, M).
        n_rows: The training rows in each month.
    """

    months: np.ndarray
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
    output: np.ndarray,
    norm: np.ndarray,
) -> MonthlyStats:
    """Sum the Gram matrices, the right-hand sides, and the farm-only regressions by month."""
    features = cache.sites[site]
    months_of_rows = row_months(cache=cache, rows=index)
    months = np.arange(months_of_rows.min(), months_of_rows.max() + 1)
    slot = months_of_rows - months[0]
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
    ghi, clear_sky = common.clear_sky_index(cache=cache, site=site)
    in_year = cache.calendar.season_year[index] == common.READ_OUT_SEASON_YEAR
    clear = (
        in_year
        & (clear_sky[index] >= common.CLEAR_SKY_MIN_W_M2)
        & (ghi[index] >= common.CLEAR_INDEX * clear_sky[index])
    )
    cross = np.zeros((2, len(months)))
    square = np.zeros((2, len(months)))
    for k, selected in enumerate((in_year, clear)):
        cross[k] = np.bincount(
            slot[selected], weights=(norm[index] * shape)[selected], minlength=len(months)
        )
        square[k] = np.bincount(slot[selected], weights=(shape**2)[selected], minlength=len(months))
    return MonthlyStats(
        months=months,
        gram=gram,
        rhs=rhs,
        farm_cross=cross,
        farm_square=square,
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


def weighted_sums(*, stats: MonthlyStats, weights: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(gram, rhs)` summed over months with the weights, shapes (B, p, p) and (B, 2, p)."""
    w = torch.as_tensor(weights, dtype=torch.float64)
    return (
        torch.einsum("bm,mpq->bpq", w, stats.gram),
        torch.einsum("bm,mkp->bkp", w, stats.rhs),
    )


def full_fit(
    *, stats: MonthlyStats, layout: GamLayout, weights: np.ndarray, spec_cost: tuple
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Fit the base and the injected series under each weighting.

    Returns:
        `(theta_base, theta_injected, gram, rhs)`: coefficients of shape (B, p), and the weighted
        sums for later use.
    """
    penalty, cost = spec_cost
    gram, rhs = weighted_sums(stats=stats, weights=weights)
    matrix = gram + torch.as_tensor(penalty, dtype=torch.float64)[None]
    shifted = rhs - 0.5 * torch.as_tensor(cost, dtype=torch.float64)[None, None, :]
    b = weights.shape[0]
    theta = solve_in_batches(
        matrix=torch.cat([matrix, matrix]),
        rhs=torch.cat([shifted[:, 0], shifted[:, 1]]),
        constrained=layout.constrained,
    )
    return theta[:b], theta[b:], gram, rhs


def frozen_fit(
    *,
    layout: GamLayout,
    gram: torch.Tensor,
    rhs: torch.Tensor,
    theta_base: np.ndarray,
    spec_cost: tuple,
) -> np.ndarray:
    """Fit the injected series' capacity increments, every other column held at the base fit.

    Returns:
        The increments, shape (B, number of season-years).
    """
    penalty, cost = spec_cost
    capacity = layout.slices["capacity"]
    other = np.r_[0 : capacity.start, capacity.stop : layout.n_columns]
    theta_other = torch.as_tensor(theta_base[:, other], dtype=torch.float64)
    cross = torch.einsum("bqo,bo->bq", gram[:, capacity][:, :, other], theta_other)
    matrix = (
        gram[:, capacity, capacity]
        + torch.as_tensor(penalty[capacity, capacity], dtype=torch.float64)[None]
    )
    target = rhs[:, 1, capacity] - cross - 0.5 * torch.as_tensor(cost[capacity])[None]
    return solve_in_batches(matrix=matrix, rhs=target, constrained=layout.constrained[capacity])


def run_farm(
    *, cache: FeatureCache, site: str, farm: str, n_resamples: int, injected_points: float
) -> tuple[list[dict], list[dict]]:
    """Return the ladder rows and the divisor rows of one Primary and farm."""
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
    stats = monthly_stats(
        cache=cache, site=site, layout=layout, index=index, output=output, norm=norm
    )
    spec_cost = penalty_matrix(layout=layout, n_rows=len(index))
    seed = 1000 + common.ALL_SITES.index(site)
    counts = month_block_counts(months=stats.months, n_resamples=n_resamples, seed=seed)
    calendar_month = stats.months % 12 + 1
    capacity = layout.slices["capacity"]
    rows: list[dict] = []

    def add(rung: str, values: np.ndarray) -> None:
        rows.extend(
            {"site": site, "farm": farm, "rung": rung, "b": b, "value": float(v)}
            for b, v in enumerate(values)
        )

    # Rungs 1 and 2: the farm's output regressed on the derated unit shape, no demand.
    for rung, k in (("r1", 0), ("r2", 1)):
        add(rung, (counts @ stats.farm_cross[k]) / (counts @ stats.farm_square[k]))

    # Rung 4 (and the base of rung 3): the full joint fit on all rows.
    theta_b, theta_i, gram, rhs = full_fit(
        stats=stats, layout=layout, weights=counts, spec_cost=spec_cost
    )
    cap_b = theta_b[:, capacity][:, : common.READ_OUT_COLUMNS].sum(axis=1)
    cap_i = theta_i[:, capacity][:, : common.READ_OUT_COLUMNS].sum(axis=1)
    add("r4", (cap_i - cap_b) / injected_points)
    # Rung 3: demand frozen at the base fit.
    frozen = frozen_fit(layout=layout, gram=gram, rhs=rhs, theta_base=theta_b, spec_cost=spec_cost)
    add("r3", (frozen[:, : common.READ_OUT_COLUMNS].sum(axis=1) - cap_b) / injected_points)
    # Rung 5: summer rows only, winter rows only.
    for rung, season in (("r5_summer", common.SUMMER_MONTHS), ("r5_winter", common.WINTER_MONTHS)):
        mask = np.isin(calendar_month, season).astype(np.float64)
        sb, si, _, _ = full_fit(
            stats=stats, layout=layout, weights=counts * mask, spec_cost=spec_cost
        )
        add(
            rung,
            (
                si[:, capacity][:, : common.READ_OUT_COLUMNS].sum(axis=1)
                - sb[:, capacity][:, : common.READ_OUT_COLUMNS].sum(axis=1)
            )
            / injected_points,
        )
    point = [
        {
            "site": site,
            "farm": farm,
            "rung": name,
            "b": 0,
            "value": float(value),
        }
        for name, value in (
            ("cap_base", cap_b[0]),
            ("cap_injected", cap_i[0]),
            ("injected_points", injected_points),
            ("n_rows", len(index)),
        )
    ]
    rows += point

    # Rung 6: 99th percentile of the recovered output against the farm's, by season.
    in_year = cache.calendar.season_year[index] == common.READ_OUT_SEASON_YEAR
    shape = derated_unit_shape(features=features, index=index)
    daylight = in_year & (shape > common.DAYLIGHT_UNIT_SHAPE)
    recovered = (cap_i[0] - cap_b[0]) * shape
    farm_output = injected_points * norm[index]
    row_calendar_month = row_months(cache=cache, rows=index) % 12 + 1
    divisor = []
    windows = {
        "year": np.ones_like(daylight),
        **{
            season: np.isin(row_calendar_month, months) for season, months in common.SEASONS.items()
        },
    }
    for window, selected in windows.items():
        chosen = daylight & selected
        divisor.append(
            {
                "site": site,
                "farm": farm,
                "window": window,
                "n_rows": int(chosen.sum()),
                "ratio": float(
                    np.percentile(recovered[chosen], 99) / np.percentile(farm_output[chosen], 99)
                ),
            }
        )
    return rows, divisor


def wiring_check(*, cache: FeatureCache, site: str) -> None:
    """Check that the monthly sums reproduce `accumulate_gram` and `fit_gam` at one Primary."""
    farm = common.FARMS[0]
    features = cache.sites[site]
    norm = common.farm_norm(cache=cache, farm=farm)
    output = np.zeros_like(norm)
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
    stats = monthly_stats(
        cache=cache, site=site, layout=layout, index=index, output=output, norm=norm
    )
    gram, rhs = weighted_sums(stats=stats, weights=np.ones((1, len(stats.months))))
    reference_gram, reference_rhs = accumulate_gram(
        cache=masked, features=masked.sites[site], layout=layout, index=index
    )
    relative = float((gram - reference_gram).abs().max() / reference_gram.abs().max())
    relative_rhs = float((rhs[:, 1] - reference_rhs).abs().max() / reference_rhs.abs().max())
    print(f"wiring {site}: gram {relative:.2e} rhs {relative_rhs:.2e}")
    spec_cost = penalty_matrix(layout=layout, n_rows=len(index))
    theta_b, _, _, _ = full_fit(
        stats=stats, layout=layout, weights=np.ones((1, len(stats.months))), spec_cost=spec_cost
    )
    mine = float(theta_b[0, layout.slices["capacity"]][: common.READ_OUT_COLUMNS].sum())
    reference = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=common.CUT,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
        device="cpu",
        dtype=torch.float64,
    )
    print(
        f"wiring {site}: base capacity {mine:.6f} fit_gam "
        f"{reference.capacity_points()[common.READ_OUT_SEASON_YEAR]:.6f}"
    )
    if abs(mine - reference.capacity_points()[common.READ_OUT_SEASON_YEAR]) > 1e-6 * max(
        1.0, abs(mine)
    ):
        msg = f"the base capacity differs from fit_gam at {site}"
        raise ValueError(msg)
    if relative > WIRING_TOLERANCE or relative_rhs > WIRING_TOLERANCE:
        msg = f"the monthly sums differ from accumulate_gram at {site}"
        raise ValueError(msg)


def main() -> None:
    """Write `work/ladder_<site>.parquet` and `work/divisor_<site>.parquet` for each Primary."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    parser.add_argument("--farms", nargs="+", default=list(common.FARMS))
    parser.add_argument("--resamples", type=int, default=common.N_RESAMPLES)
    parser.add_argument("--wiring", action="store_true", help="only run the wiring check")
    parser.add_argument("--suffix", default="")
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    for site in arguments.sites:
        if arguments.wiring:
            wiring_check(cache=cache, site=site)
            continue
        pinned = pinned_capacity(
            vintages=read_vintages(), table=cache.sites[site].table, cut=common.CUT
        ).held_points
        injected_points = common.INJECTED_SHARE * pinned
        ladder: list[dict] = []
        divisor: list[dict] = []
        for farm in arguments.farms:
            farm_rows, farm_divisor = run_farm(
                cache=cache,
                site=site,
                farm=farm,
                n_resamples=arguments.resamples,
                injected_points=injected_points,
            )
            ladder += farm_rows
            divisor += farm_divisor
            print(f"{site} farm {farm}: done", flush=True)
        pl.DataFrame(ladder).write_parquet(
            common.WORK_DIR / f"ladder_{site}{arguments.suffix}.parquet"
        )
        pl.DataFrame(divisor).write_parquet(
            common.WORK_DIR / f"divisor_{site}{arguments.suffix}.parquet"
        )


if __name__ == "__main__":
    main()
