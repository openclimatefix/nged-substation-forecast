"""Fit the two-stage and joint solar capacities with and without an injected farm, one Primary.

Both arms train on the same rows before `ORIGIN`. Arm 2 is experiment 129's capacity: one
non-negative increment per season-year, with a ridge on the cumulative capacity. The joint arm
gives the capacity a non-negative level for each calendar month, with a penalty on the squared
change of the level from month to month. The year columns are sums of the month columns, so one
Gram matrix of the month design serves both arms (the year Gram is `T^T G T`).

The base and injected fits of a farm share one row set (the rows where the farm has a reading), so
their difference never mixes two row sets. Recovery is the injected fit's capacity minus the base
fit's, divided by the injected capacity.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00157-*/e00157_fit.py --sites D10
"""

import argparse
from dataclasses import dataclass, replace
from typing import Final

import e00157_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    DEFAULT_MAX_ITERATIONS,
    GamLayout,
    GamSpec,
    design_matrix,
    fit_gam,
    penalty_matrix,
    prepare_fit,
    row_chunks,
    solve_bounded_quadratic,
)
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

LEVEL_RIDGE_SHARE: Final[float] = 1e-6
"""The joint path's ridge on the level itself, as a share of the growth penalty, so that a month
with no training row stays well posed."""
WIRING_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True)
class MonthDesign:
    """The monthly capacity columns of one Primary's training rows.

    Attributes:
        first_month: The first month as months since 1970-01.
        n_months: The number of months from the first to the last month with a training row.
        season_year_of_month: The season-year (July to June, named by the July's year) of each.
    """

    first_month: int
    n_months: int
    season_year_of_month: np.ndarray


def row_months(*, cache: FeatureCache, rows: np.ndarray) -> np.ndarray:
    """Return each row's calendar month, as months since 1970-01, from the half-hour's start."""
    start = cache.end_times[rows] - np.timedelta64(30, "m")
    return start.astype("datetime64[M]").astype(np.int64)


def month_design(*, cache: FeatureCache, index: np.ndarray) -> MonthDesign:
    """Return the span of months the training rows cover, including months with no row."""
    months = row_months(cache=cache, rows=index)
    first, last = int(months.min()), int(months.max())
    all_months = np.arange(first, last + 1)
    calendar_year = 1970 + all_months // 12
    calendar_month = all_months % 12 + 1
    season = np.where(calendar_month >= 7, calendar_year, calendar_year - 1)
    return MonthDesign(first_month=first, n_months=last - first + 1, season_year_of_month=season)


def month_columns(
    *, cache: FeatureCache, rows: np.ndarray, design: MonthDesign, shape: np.ndarray
) -> np.ndarray:
    """Return the minus-signed month columns: `-shape` in the column of the row's month."""
    columns = np.zeros((len(rows), design.n_months))
    columns[np.arange(len(rows)), row_months(cache=cache, rows=rows) - design.first_month] = -shape
    return columns


def accumulate_month_gram(
    *,
    cache: FeatureCache,
    site: str,
    layout: GamLayout,
    index: np.ndarray,
    design: MonthDesign,
    targets: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum `X^T X` and `X^T y` for the month design over the rows `index`.

    Args:
        cache: The cache.
        site: The substation's label.
        layout: The season-year layout, whose capacity columns the month columns replace.
        index: The training rows.
        design: The months.
        targets: The targets, shape (n_targets, len(index)).

    Returns:
        `(gram, rhs)` of shapes (p', p') and (n_targets, p') in float64.
    """
    features = cache.sites[site]
    cap = layout.slices["capacity"]
    p_new = layout.n_columns - (cap.stop - cap.start) + design.n_months
    gram = torch.zeros((p_new, p_new), dtype=torch.float64)
    rhs = torch.zeros((targets.shape[0], p_new), dtype=torch.float64)
    target_tensor = torch.as_tensor(targets, dtype=torch.float64)
    for chunk in row_chunks(index=index):
        rows = index[chunk]
        x = design_matrix(cache=cache, features=features, layout=layout, index=rows)
        years = np.array(layout.season_years)
        season = cache.calendar.season_year[rows]
        capacity_columns = x[:, cap]
        # The year columns are -shape times a season indicator, so the shape is the column of the
        # earliest season-year.
        shape = -capacity_columns[:, 0] if season.min() >= years[0] else None
        if shape is None:
            msg = "a training row precedes the first season-year with a capacity increment"
            raise ValueError(msg)
        x_new = np.concatenate(
            [
                x[:, : cap.start],
                month_columns(cache=cache, rows=rows, design=design, shape=shape),
                x[:, cap.stop :],
            ],
            axis=1,
        )
        x_tensor = torch.as_tensor(x_new, dtype=torch.float64)
        gram += x_tensor.T @ x_tensor
        rhs += target_tensor[:, chunk] @ x_tensor
    return gram, rhs


def year_transform(*, layout: GamLayout, design: MonthDesign) -> torch.Tensor:
    """Return `T` with `X_year = X_month @ T`: a year column sums the months from its season on."""
    cap = layout.slices["capacity"]
    n_years = cap.stop - cap.start
    p_new = layout.n_columns - n_years + design.n_months
    transform = torch.zeros((p_new, layout.n_columns), dtype=torch.float64)
    before = np.arange(cap.start)
    transform[before, before] = 1.0
    for j, year in enumerate(layout.season_years):
        transform[
            cap.start + np.flatnonzero(design.season_year_of_month >= year), cap.start + j
        ] = 1.0
    after_new = np.arange(cap.start + design.n_months, p_new)
    after_old = np.arange(cap.stop, layout.n_columns)
    transform[after_new, after_old] = 1.0
    return transform


def solve(
    *,
    gram: torch.Tensor,
    rhs: torch.Tensor,
    penalty: np.ndarray,
    cost: np.ndarray,
    constrained: np.ndarray,
) -> np.ndarray:
    """Solve the penalised, bounded least squares for each right-hand side; return theta."""
    batch = rhs.shape[0]
    matrix = gram[None] + torch.as_tensor(penalty, dtype=torch.float64)[None]
    theta, _ = solve_bounded_quadratic(
        matrix=matrix.expand(batch, -1, -1),
        rhs=rhs - 0.5 * torch.as_tensor(cost, dtype=torch.float64)[None],
        constrained=torch.as_tensor(constrained),
        max_iterations=DEFAULT_MAX_ITERATIONS,
    )
    return theta.numpy()


def joint_penalty(
    *, layout: GamLayout, design: MonthDesign, n_rows: int, gram: torch.Tensor, share: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the joint arm's penalty matrix, cost vector, and bound mask on the month design."""
    cap = layout.slices["capacity"]
    n_years = cap.stop - cap.start
    old_matrix, old_cost = penalty_matrix(layout=layout, n_rows=n_rows)
    p_new = layout.n_columns - n_years + design.n_months
    keep_old = np.r_[0 : cap.start, cap.stop : layout.n_columns]
    keep_new = np.r_[0 : cap.start, cap.start + design.n_months : p_new]
    matrix = np.zeros((p_new, p_new))
    matrix[np.ix_(keep_new, keep_new)] = old_matrix[np.ix_(keep_old, keep_old)]
    cost = np.zeros(p_new)
    cost[keep_new] = old_cost[keep_old]
    months = slice(cap.start, cap.start + design.n_months)
    diagonal = torch.diagonal(gram)[months].numpy()
    growth = share * float(diagonal[diagonal > 0].mean())
    difference = np.diff(np.eye(design.n_months), axis=0)
    matrix[months, months] = (
        growth * difference.T @ difference + LEVEL_RIDGE_SHARE * growth * np.eye(design.n_months)
    )
    constrained = np.zeros(p_new, dtype=bool)
    constrained[keep_new] = layout.constrained[keep_old]
    constrained[months] = True
    return matrix, cost, constrained


def fit_one(
    *, cache: FeatureCache, site: str, output: np.ndarray, farm: str
) -> tuple[list[dict], list[dict]]:
    """Fit both arms with and without `output` (in points) at one Primary; return result rows.

    Args:
        cache: The cache.
        site: The substation's label.
        output: The injected output on the grid in points of effective capacity; `NaN` where the
            farm has no reading. All zeros injects nothing.
        farm: The farm's label, or `none`.

    Returns:
        One row per arm and base or injected fit, and the joint arm's monthly paths.
    """
    features = cache.sites[site]
    masked = cache.sites[site].y - output

    masked_cache = replace(cache, sites={**cache.sites, site: replace(features, y=masked)})
    spec = GamSpec(ridge=common.RIDGE, capacity="fitted")
    layout, index = prepare_fit(
        cache=masked_cache,
        site=site,
        spec=spec,
        cut=common.ORIGIN,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
    )
    base = features.y[index]
    targets = np.stack([base, base - np.nan_to_num(output[index])])
    design = month_design(cache=cache, index=index)
    gram, rhs = accumulate_month_gram(
        cache=cache, site=site, layout=layout, index=index, design=design, targets=targets
    )
    results = []
    # Arm 2: season-year steps, from the same Gram.
    transform = year_transform(layout=layout, design=design)
    matrix, cost = penalty_matrix(layout=layout, n_rows=len(index))
    theta = solve(
        gram=transform.T @ gram @ transform,
        rhs=rhs @ transform,
        penalty=matrix,
        cost=cost,
        constrained=layout.constrained,
    )
    cap = layout.slices["capacity"]
    years = list(layout.season_years)
    for k, kind in enumerate(("base", "injected")):
        by_year = dict(zip(years, np.cumsum(theta[k, cap]).tolist(), strict=True))
        results.append(
            {
                "arm": "two_stage",
                "share": float("nan"),
                "fit": kind,
                "year_mean": by_year[common.LATEST_SEASON_YEAR],
                "last_month": by_year[common.LATEST_SEASON_YEAR],
                "year_weighted_mean": by_year[common.LATEST_SEASON_YEAR],
            }
        )
    in_year = design.season_year_of_month == common.LATEST_SEASON_YEAR
    shape_weight = torch.diagonal(gram)[cap.start : cap.start + design.n_months].numpy()
    paths = []
    for share in common.GROWTH_SHARES:
        matrix, cost, constrained = joint_penalty(
            layout=layout, design=design, n_rows=len(index), gram=gram, share=share
        )
        theta = solve(gram=gram, rhs=rhs, penalty=matrix, cost=cost, constrained=constrained)
        months = slice(cap.start, cap.start + design.n_months)
        for k, kind in enumerate(("base", "injected")):
            path = theta[k, months]
            results.append(
                {
                    "arm": "joint",
                    "share": share,
                    "fit": kind,
                    "year_mean": float(path[in_year].mean()),
                    "last_month": float(path[-1]),
                    "year_weighted_mean": float(
                        np.average(path[in_year], weights=shape_weight[in_year])
                    ),
                }
            )
            paths += [
                {"share": share, "fit": kind, "month": design.first_month + m, "level": float(v)}
                for m, v in enumerate(path)
            ]
    rows = [{**row, "farm": farm, "n_rows": len(index)} for row in results]
    return rows, [{**row, "farm": farm} for row in paths]


def run_site(*, cache: FeatureCache, site: str) -> pl.DataFrame:
    """Return the recovery table of one Primary: every farm, arm, and growth penalty."""
    table = cache.sites[site].table
    pinned_points = pinned_capacity(
        vintages=read_vintages(), table=table, cut=common.ORIGIN
    ).held_points
    injected_points = common.INJECTED_SHARE * pinned_points
    scale = table.effective_capacity_mw / 100.0
    rows = []
    paths = []
    for farm in common.FARMS:
        output = injected_points * common.farm_norm(cache=cache, farm=farm)
        farm_rows, farm_paths = fit_one(cache=cache, site=site, output=output, farm=farm)
        rows += farm_rows
        paths += farm_paths
        print(f"{site} farm {farm}: done", flush=True)
    pl.DataFrame(paths).with_columns(site=pl.lit(site), scale=pl.lit(scale)).write_parquet(
        common.WORK_DIR / f"paths_{site}.parquet"
    )
    frame = pl.DataFrame(rows)
    keys = ["farm", "arm", "share"]
    wide = frame.pivot(
        on="fit", index=[*keys, "n_rows"], values=["year_mean", "last_month", "year_weighted_mean"]
    ).with_columns(site=pl.lit(site))
    return wide.with_columns(
        injected_points=pl.lit(injected_points),
        pinned_points=pl.lit(pinned_points),
        scale_mw_per_point=pl.lit(scale),
        recovery_year=(pl.col("year_mean_injected") - pl.col("year_mean_base")) / injected_points,
        recovery_last_month=(pl.col("last_month_injected") - pl.col("last_month_base"))
        / injected_points,
        recovery_weighted=(
            pl.col("year_weighted_mean_injected") - pl.col("year_weighted_mean_base")
        )
        / injected_points,
    )


def wiring_check(*, cache: FeatureCache, site: str) -> None:
    """Check that arm 2 with no injection reproduces `fit_gam` at the origin."""
    rows, _ = fit_one(
        cache=cache, site=site, output=np.zeros(len(cache.sites[site].y)), farm="none"
    )
    mine = next(r for r in rows if r["arm"] == "two_stage" and r["fit"] == "base")["year_mean"]
    reference = fit_gam(
        cache=cache,
        site=site,
        specs=[GamSpec(ridge=common.RIDGE, capacity="fitted")],
        cut=common.ORIGIN,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
        device="cpu",
        dtype=torch.float64,
    ).capacity_points()[common.LATEST_SEASON_YEAR]
    print(f"wiring {site}: mine {mine:.6f} fit_gam {reference:.6f}")
    if abs(mine - reference) > WIRING_TOLERANCE * max(1.0, abs(reference)):
        msg = f"arm 2 differs from fit_gam at {site}: {mine} against {reference}"
        raise ValueError(msg)


def main() -> None:
    """Write `work/recovery_<site>.parquet` for each requested Primary."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    parser.add_argument("--wiring", action="store_true", help="only run the wiring check")
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    for site in arguments.sites:
        if arguments.wiring:
            wiring_check(cache=cache, site=site)
            continue
        table = run_site(cache=cache, site=site)
        table.write_parquet(common.WORK_DIR / f"recovery_{site}.parquet")


if __name__ == "__main__":
    main()
