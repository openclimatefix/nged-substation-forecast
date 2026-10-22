"""Run the injection test of experiment 239, one Primary at a time.

For each Primary and injected farm, the script adds the farm's output at 50% of the Primary's
pinned capacity and fits the capacity under two solar shapes: the satellite-based shape, and the
same shape multiplied by a seasonal efficiency curve fitted on the other farms. It builds one Gram
matrix per calendar month of training rows, so every month-block bootstrap resample is a weighted
sum followed by one bounded least-squares solve. The plan is in
`docs/ml-experiments/00239-pv-capacity-seasonal-efficiency-curve/result.md`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00239-*/e00239_fit.py --sites D10
"""

import argparse
from dataclasses import dataclass, replace
from itertools import product
from typing import Final

import e00239_common as common
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
WIRING_TOLERANCE: Final[float] = 1e-8


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
    *,
    stats: MonthlyStats,
    layout: GamLayout,
    weights: np.ndarray,
    spec_cost: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Fit the base and the injected series under each weighting.

    Args:
        stats: The monthly sums.
        layout: The columns.
        weights: Month weights, shape (B, M).
        spec_cost: The penalty matrix and the linear cost vector.

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
    spec_cost: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Fit the injected series' capacity increments, every other column held at the base fit.

    Args:
        layout: The columns.
        gram: The weighted Gram matrices, shape (B, p, p).
        rhs: The weighted right-hand sides, shape (B, 2, p).
        theta_base: The base fit's coefficients, shape (B, p).
        spec_cost: The penalty matrix and the linear cost vector.

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


def explained_share(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    norm: np.ndarray,
    counts: np.ndarray,
    months: np.ndarray,
    stats: MonthlyStats,
) -> np.ndarray:
    """Return, per month-block resample, the share of the farm's squared output one scale explains.

    The share is `(sum x u)^2 / (sum x^2 sum u^2)` over daylight rows of the read-out season-year,
    for the farm's normalised output `x` and the derated unit shape `u`.
    """
    in_year = cache.calendar.season_year[index] == common.READ_OUT_SEASON_YEAR
    slot = row_months(cache=cache, rows=index) - months[0]
    x = norm[index]
    xsq = np.bincount(slot[in_year], weights=(x**2)[in_year], minlength=len(months))
    cross = counts @ stats.farm_cross[0]
    usq = counts @ stats.farm_square[0]
    return cross**2 / ((counts @ xsq) * usq)


def run_arm(
    *,
    cache: FeatureCache,
    site: str,
    arm: str,
    shape: np.ndarray,
    norm: np.ndarray,
    n_resamples: int,
    injected_points: float,
    farm: str,
) -> list[dict]:
    """Return the result rows of one solar shape at one Primary and injected farm."""
    features = cache.sites[site]
    output = injected_points * norm
    shaped = replace(cache, sites={**cache.sites, site: replace(features, pv_unit_shape=shape)})
    masked = replace(
        shaped,
        sites={**shaped.sites, site: replace(shaped.sites[site], y=features.y - output)},
    )
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
        cache=shaped, site=site, layout=layout, index=index, output=output, norm=norm
    )
    spec_cost = penalty_matrix(layout=layout, n_rows=len(index))
    seed = 1000 + common.ALL_SITES.index(site)
    counts = month_block_counts(months=stats.months, n_resamples=n_resamples, seed=seed)
    capacity = layout.slices["capacity"]
    theta_b, theta_i, gram, rhs = full_fit(
        stats=stats, layout=layout, weights=counts, spec_cost=spec_cost
    )
    cap_b = theta_b[:, capacity][:, : common.READ_OUT_COLUMNS].sum(axis=1)
    cap_i = theta_i[:, capacity][:, : common.READ_OUT_COLUMNS].sum(axis=1)
    in_year = cache.calendar.season_year[index] == common.READ_OUT_SEASON_YEAR
    derated = derated_unit_shape(features=shaped.sites[site], index=index)
    daylight = in_year & (derated > common.DAYLIGHT_UNIT_SHAPE) & (norm[index] > 0.0)
    p99_factor = float(
        np.percentile(derated[daylight], 99) / np.percentile(norm[index][daylight], 99)
    )
    frozen = frozen_fit(layout=layout, gram=gram, rhs=rhs, theta_base=theta_b, spec_cost=spec_cost)
    values = {
        "recovery": (cap_i - cap_b) / injected_points * p99_factor,
        "capacity_ratio": (cap_i - cap_b) / injected_points,
        "recovery_frozen": (frozen[:, : common.READ_OUT_COLUMNS].sum(axis=1) - cap_b)
        / injected_points
        * p99_factor,
        "farm_only_scale": (counts @ stats.farm_cross[0]) / (counts @ stats.farm_square[0]),
        "explained": explained_share(
            cache=shaped,
            site=site,
            index=index,
            norm=norm,
            counts=counts,
            months=stats.months,
            stats=stats,
        ),
    }
    rows = [
        {
            "site": site,
            "farm": farm,
            "arm": arm,
            "stat": name,
            "b": b,
            "value": float(v),
        }
        for name, series in values.items()
        for b, v in enumerate(series)
    ]
    rows += [
        {
            "site": site,
            "farm": farm,
            "arm": arm,
            "stat": name,
            "b": 0,
            "value": float(v),
        }
        for name, v in (
            ("n_rows", len(index)),
            ("injected_points", injected_points),
            ("p99_factor", p99_factor),
        )
    ]
    return rows


def own_site_fit(*, cache: FeatureCache, farm: str, curve: np.ndarray) -> list[dict]:
    """Return the share of a farm's squared output that one scale of its own shape explains.

    The share is `(sum x v)^2 / (sum x^2 sum v^2)` over daylight half-hours before the cut, with and
    without the curve, over all years and over the read-out season-year.
    """
    norm = common.farm_norm(cache=cache, farm=farm, max_zenith_deg=common.FULL_SUN_ZENITH_DEG)
    shape = common.farm_own_shape(cache=cache, farm=farm)
    month = common.calendar_months(cache=cache)
    usable = (
        (cache.end_times <= np.datetime64(common.CUT.replace(tzinfo=None)))
        & np.isfinite(norm)
        & np.isfinite(shape)
        & (shape > common.MIN_DAYLIGHT_SHAPE)
    )
    windows = {
        "all_years": usable,
        "read_out_year": usable & (cache.calendar.season_year == common.READ_OUT_SEASON_YEAR),
    }
    rows = []
    for window, selected in windows.items():
        for arm, scaled in (("plain", shape), ("curve", shape * curve[month - 1])):
            x = norm[selected]
            v = scaled[selected]
            share = float((x @ v) ** 2 / ((x @ x) * (v @ v)))
            rows.append(
                {
                    "site": "own",
                    "farm": farm,
                    "arm": arm,
                    "stat": f"own_{window}",
                    "b": 0,
                    "value": share,
                }
            )
    return rows


def run_farm(
    *, cache: FeatureCache, site: str, farm: str, n_resamples: int, injected_points: float
) -> list[dict]:
    """Return the result rows of the two arms at one Primary and injected farm."""
    curve = common.curve_excluding(cache=cache, injected=farm)
    plain = cache.sites[site].pv_unit_shape
    curved = common.apply_curve(cache=cache, site=site, curve=curve)
    features = cache.sites[site]
    check = replace(features, pv_unit_shape=curved)
    wanted = np.flatnonzero(np.isfinite(curved) & (plain > common.MIN_DAYLIGHT_SHAPE))
    expected = (
        common.derate(unit=plain, temperature_c=features.temperature_c)
        * curve[common.calendar_months(cache=cache) - 1]
    )[wanted]
    error = float(np.abs(derated_unit_shape(features=check, index=wanted) - expected).max())
    if error > common.DERATING_TOLERANCE:
        msg = f"the curved shape differs from the satellite shape times the curve by {error}"
        raise ValueError(msg)
    rows: list[dict] = []
    injections = {
        "orig": common.farm_norm(cache=cache, farm=farm),
        "full": common.farm_norm(cache=cache, farm=farm, max_zenith_deg=common.FULL_SUN_ZENITH_DEG),
    }
    for (injection, norm), (shape_name, shape) in product(
        injections.items(), (("plain", plain), ("curve", curved))
    ):
        rows += run_arm(
            cache=cache,
            site=site,
            arm=f"{injection}_{shape_name}",
            shape=shape,
            norm=norm,
            n_resamples=n_resamples,
            injected_points=injected_points,
            farm=farm,
        )
    counts = {r["arm"]: r["value"] for r in rows if r["stat"] == "n_rows"}
    for injection in injections:
        if counts[f"{injection}_plain"] != counts[f"{injection}_curve"]:
            msg = f"the arms train on different rows at {site}, farm {farm}"
            raise ValueError(msg)
    return rows


def curve_rows(*, cache: FeatureCache) -> list[dict]:
    """Return the monthly scales of each farm and the curve for each left-out farm."""
    rows = []
    for farm in common.FARMS:
        for month, value in enumerate(common.monthly_scales(cache=cache, farm=farm), start=1):
            rows.append({"kind": "farm_scale", "farm": farm, "month": month, "value": float(value)})
        curve = common.curve_excluding(cache=cache, injected=farm)
        for month, value in enumerate(curve, start=1):
            rows.append(
                {"kind": "curve_without", "farm": farm, "month": month, "value": float(value)}
            )
    return rows


def main() -> None:
    """Write `work/result_<site>.parquet` for each Primary, and the curves and own-site fits."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    parser.add_argument("--farms", nargs="+", default=list(common.FARMS))
    parser.add_argument("--resamples", type=int, default=common.N_RESAMPLES)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--curves", action="store_true", help="only write the curves and own fits")
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    if arguments.curves:
        pl.DataFrame(curve_rows(cache=cache)).write_parquet(common.WORK_DIR / "curves.parquet")
        own = []
        for farm in common.FARMS:
            own += own_site_fit(
                cache=cache, farm=farm, curve=common.curve_excluding(cache=cache, injected=farm)
            )
        pl.DataFrame(own).write_parquet(common.WORK_DIR / "own_site.parquet")
        return
    for site in arguments.sites:
        pinned = pinned_capacity(
            vintages=read_vintages(), table=cache.sites[site].table, cut=common.CUT
        ).held_points
        injected_points = common.INJECTED_SHARE * pinned
        results: list[dict] = []
        for farm in arguments.farms:
            farm_rows = run_farm(
                cache=cache,
                site=site,
                farm=farm,
                n_resamples=arguments.resamples,
                injected_points=injected_points,
            )
            results += farm_rows
            print(f"{site} farm {farm}: {len(farm_rows)} rows", flush=True)
        pl.DataFrame(results).write_parquet(
            common.WORK_DIR / f"result_{site}{arguments.suffix}.parquet"
        )


if __name__ == "__main__":
    main()
