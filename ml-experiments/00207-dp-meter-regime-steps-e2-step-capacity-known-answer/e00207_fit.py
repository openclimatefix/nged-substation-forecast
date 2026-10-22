"""Fit three solar-capacity paths with and without injected farm steps, one Primary at a time.

All arms train on the same summer rows (April to August) before `ORIGIN`, for each farm's row set,
and read from one month-design Gram matrix: experiment 129's season-year steps, experiment 157's
smooth monthly path, and a total-variation (step-capable) monthly path. Each Primary and farm has
20 injected targets: two shares of the pinned capacity, an on-step and an off-step, each at a
seeded random date.

    systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00207-*/e00207_fit.py --sites D10
"""

import argparse
import zlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Final

import e00207_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    DEFAULT_MAX_ITERATIONS,
    GamLayout,
    GamSpec,
    design_matrix,
    penalty_matrix,
    prepare_fit,
    row_chunks,
    solve_bounded_quadratic,
)
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

LEVEL_RIDGE_SHARE: Final[float] = 1e-6
"""The path's ridge on the level itself, as a share of the growth penalty, so that a month with no
training row (every month outside the summer) stays well posed."""


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


@dataclass(frozen=True)
class Injection:
    """One injected step.

    Attributes:
        share: The injected capacity as a share of the Primary's pinned capacity.
        direction: `on` (the output is removed from the target from the step date on) or `off`
            (the output is removed before the step date).
        step_date: The first of the month at which the step falls.
        month_index: The step's month, as months since 1970-01.
    """

    share: float
    direction: str
    step_date: datetime
    month_index: int


def row_months(*, cache: FeatureCache, rows: np.ndarray) -> np.ndarray:
    """Return each row's calendar month, as months since 1970-01, from the half-hour's start."""
    start = cache.end_times[rows] - np.timedelta64(30, "m")
    return start.astype("datetime64[M]").astype(np.int64)


def summer_rows(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Keep the rows whose calendar month is in `SUMMER_MONTHS`."""
    months = row_months(cache=cache, rows=index) % 12 + 1
    return index[np.isin(months, common.SUMMER_MONTHS)]


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

    Raises:
        ValueError: If a training row precedes the first season-year with a capacity increment.
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
        shape = -x[:, cap][:, 0] if season.min() >= years[0] else None
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


def step_transform(*, layout: GamLayout, design: MonthDesign) -> torch.Tensor:
    """Return `T` with `X_step = X_month @ T` for the step parametrisation of the monthly path.

    The capacity block of the step design is `[l0, u_1 .. u_(M-1), d_1 .. d_(M-1)]`: the level of
    the first month, then an up-step and a down-step at each month boundary, all non-negative.
    The level of month `m` is `l0 + sum_(k<=m) (u_k - d_k)`.
    """
    cap = layout.slices["capacity"]
    n_months = design.n_months
    p_month = layout.n_columns - (cap.stop - cap.start) + n_months
    n_steps = 1 + 2 * (n_months - 1)
    p_step = p_month - n_months + n_steps
    transform = torch.zeros((p_month, p_step), dtype=torch.float64)
    before = np.arange(cap.start)
    transform[before, before] = 1.0
    block = cap.start
    transform[block : block + n_months, block] = 1.0
    for k in range(1, n_months):
        transform[block + k : block + n_months, block + k] = 1.0
        transform[block + k : block + n_months, block + n_months - 1 + k] = -1.0
    after_month = np.arange(cap.start + n_months, p_month)
    after_step = np.arange(block + n_steps, p_step)
    transform[after_month, after_step] = 1.0
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


def smooth_penalty(
    *, layout: GamLayout, design: MonthDesign, n_rows: int, gram: torch.Tensor, share: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return experiment 157's smooth-path penalty matrix, cost vector, and bound mask."""
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


def step_penalty(
    *,
    layout: GamLayout,
    design: MonthDesign,
    n_rows: int,
    gram: torch.Tensor,
    share: float,
    transform: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the total-variation path's penalty matrix, linear cost, and bound mask.

    The linear cost on every up-step and down-step is `share` times the mean positive diagonal of
    the month Gram matrix, per point of step. The first-month level has the small ridge only.
    """
    cap = layout.slices["capacity"]
    n_years = cap.stop - cap.start
    old_matrix, old_cost = penalty_matrix(layout=layout, n_rows=n_rows)
    p_month = layout.n_columns - n_years + design.n_months
    p_step = transform.shape[1]
    n_steps = 1 + 2 * (design.n_months - 1)
    keep_old = np.r_[0 : cap.start, cap.stop : layout.n_columns]
    keep_new = np.r_[0 : cap.start, cap.start + n_steps : p_step]
    matrix = np.zeros((p_step, p_step))
    matrix[np.ix_(keep_new, keep_new)] = old_matrix[np.ix_(keep_old, keep_old)]
    cost = np.zeros(p_step)
    cost[keep_new] = old_cost[keep_old]
    months = slice(cap.start, cap.start + design.n_months)
    diagonal = torch.diagonal(gram)[months].numpy()
    unit = float(diagonal[diagonal > 0].mean())
    block = slice(cap.start, cap.start + n_steps)
    matrix[block, block] = LEVEL_RIDGE_SHARE * unit * np.eye(n_steps)
    cost[cap.start + 1 : cap.start + n_steps] = share * unit
    constrained = np.zeros(p_step, dtype=bool)
    constrained[keep_new] = layout.constrained[keep_old]
    constrained[block] = True
    del p_month
    return matrix, cost, constrained


def draw_injections(*, site: str, farm: str, pinned_points: float) -> list[Injection]:
    """Return the 4 injections of a Primary and farm: two shares times on and off."""
    injections = []
    for share in common.INJECTED_SHARES:
        for direction in common.DIRECTIONS:
            seed = zlib.crc32(f"{common.SEED}|{site}|{farm}|{share}|{direction}".encode())
            rng = np.random.default_rng(seed)
            year = int(rng.choice(common.STEP_YEARS))
            month = int(rng.choice(common.STEP_MONTHS))
            injections.append(
                Injection(
                    share=share,
                    direction=direction,
                    step_date=datetime(year, month, 1, tzinfo=UTC),
                    month_index=(year - 1970) * 12 + month - 1,
                )
            )
    del pinned_points
    return injections


def injected_output(
    *, cache: FeatureCache, norm: np.ndarray, pinned_points: float, injection: Injection
) -> np.ndarray:
    """Return the output to subtract from the target, in points; `NaN` where the farm has no row."""
    step = np.datetime64(injection.step_date.replace(tzinfo=None), "us")
    after = cache.end_times > step
    keep = after if injection.direction == "on" else ~after
    return injection.share * pinned_points * norm * keep


def levels_from(
    *, arm: str, theta: np.ndarray, layout: GamLayout, design: MonthDesign
) -> np.ndarray:
    """Return the monthly levels (points) of each target, shape (n_targets, n_months)."""
    cap = layout.slices["capacity"]
    n_months = design.n_months
    if arm == "two_stage":
        years = list(layout.season_years)
        by_year = np.cumsum(theta[:, cap], axis=1)
        position = np.array([years.index(int(s)) for s in design.season_year_of_month])
        return by_year[:, position]
    if arm == "smooth":
        return theta[:, cap.start : cap.start + n_months]
    first = theta[:, cap.start]
    ups = theta[:, cap.start + 1 : cap.start + n_months]
    downs = theta[:, cap.start + n_months : cap.start + 2 * n_months - 1]
    return first[:, None] + np.concatenate(
        [np.zeros((theta.shape[0], 1)), np.cumsum(ups - downs, axis=1)], axis=1
    )


def fit_farm(
    *, cache: FeatureCache, site: str, farm: str, pinned_points: float
) -> tuple[list[dict], list[dict]]:
    """Fit every arm for one Primary and farm; return recovery rows and level-path rows."""
    features = cache.sites[site]
    norm = common.farm_norm(cache=cache, farm=farm)
    injections = draw_injections(site=site, farm=farm, pinned_points=pinned_points)
    outputs = [
        injected_output(cache=cache, norm=norm, pinned_points=pinned_points, injection=i)
        for i in injections
    ]
    row_set_output = common.INJECTED_SHARES[0] * pinned_points * norm
    masked = replace(features, y=features.y - row_set_output)
    spec = GamSpec(ridge=common.RIDGE, capacity="fitted")
    layout, index = prepare_fit(
        cache=replace(cache, sites={**cache.sites, site: masked}),
        site=site,
        spec=spec,
        cut=common.ORIGIN,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
    )
    index = summer_rows(cache=cache, index=index)
    base = features.y[index]
    targets = np.stack([base] + [base - np.nan_to_num(o[index]) for o in outputs])
    design = month_design(cache=cache, index=index)
    gram, rhs = accumulate_month_gram(
        cache=cache, site=site, layout=layout, index=index, design=design, targets=targets
    )
    arms: dict[str, np.ndarray] = {}
    year_t = year_transform(layout=layout, design=design)
    matrix, cost = penalty_matrix(layout=layout, n_rows=len(index))
    arms["two_stage"] = levels_from(
        arm="two_stage",
        theta=solve(
            gram=year_t.T @ gram @ year_t,
            rhs=rhs @ year_t,
            penalty=matrix,
            cost=cost,
            constrained=layout.constrained,
        ),
        layout=layout,
        design=design,
    )
    matrix, cost, constrained = smooth_penalty(
        layout=layout, design=design, n_rows=len(index), gram=gram, share=common.SMOOTH_SHARE
    )
    arms["smooth"] = levels_from(
        arm="smooth",
        theta=solve(gram=gram, rhs=rhs, penalty=matrix, cost=cost, constrained=constrained),
        layout=layout,
        design=design,
    )
    step_t = step_transform(layout=layout, design=design)
    for share in common.TV_SHARES:
        matrix, cost, constrained = step_penalty(
            layout=layout,
            design=design,
            n_rows=len(index),
            gram=gram,
            share=share,
            transform=step_t,
        )
        arms[f"step_{share:g}"] = levels_from(
            arm="step",
            theta=solve(
                gram=step_t.T @ gram @ step_t,
                rhs=rhs @ step_t,
                penalty=matrix,
                cost=cost,
                constrained=constrained,
            ),
            layout=layout,
            design=design,
        )
    return read_out(
        site=site,
        farm=farm,
        injections=injections,
        arms=arms,
        design=design,
        pinned_points=pinned_points,
        n_rows=len(index),
    )


def read_out(
    *,
    site: str,
    farm: str,
    injections: list[Injection],
    arms: dict[str, np.ndarray],
    design: MonthDesign,
    pinned_points: float,
    n_rows: int,
) -> tuple[list[dict], list[dict]]:
    """Return the step size ratio and date error of each arm and injection, and the paths.

    The size is the mean level from the step month to the end of that August minus the mean from
    that April to the month before the step, injected fit minus base fit, over the injected step
    (plus for an on-step, minus for an off-step). The date is the month boundary, between April and
    August of the step year, with the largest absolute change of the injected minus base path.
    """
    months = design.first_month + np.arange(design.n_months)
    rows, paths = [], []
    for arm, levels in arms.items():
        base_levels = levels[0]
        for k, injection in enumerate(injections, start=1):
            diff = levels[k] - base_levels
            year = 1970 + injection.month_index // 12
            april = (year - 1970) * 12 + 3
            august = (year - 1970) * 12 + 7
            before = (months >= april) & (months < injection.month_index)
            after = (months >= injection.month_index) & (months <= august)
            sign = 1.0 if injection.direction == "on" else -1.0
            injected = injection.share * pinned_points
            size_ratio = float((diff[after].mean() - diff[before].mean()) / (sign * injected))
            window = (months >= april) & (months <= august)
            boundaries = np.flatnonzero(window[1:] & window[:-1]) + 1
            changes = np.abs(diff[boundaries] - diff[boundaries - 1])
            found = int(months[boundaries[int(np.argmax(changes))]])
            date_error_days = abs(
                (found - injection.month_index) * 30.4
            )  # month boundaries; 30.4 days per month
            rows.append(
                {
                    "site": site,
                    "farm": farm,
                    "arm": arm,
                    "share": injection.share,
                    "direction": injection.direction,
                    "step_date": injection.step_date.date(),
                    "size_ratio": size_ratio,
                    "date_error_days": date_error_days,
                    "n_rows": n_rows,
                    "injected_points": injected,
                }
            )
        paths += [
            {
                "site": site,
                "farm": farm,
                "arm": arm,
                "target": k,
                "month": int(m),
                "level": float(v),
            }
            for k in range(levels.shape[0])
            for m, v in zip(months, levels[k], strict=True)
        ]
    return rows, paths


def run_site(*, cache: FeatureCache, site: str) -> None:
    """Write the recovery and path tables of one Primary."""
    table = cache.sites[site].table
    pinned_points = pinned_capacity(
        vintages=read_vintages(), table=table, cut=common.ORIGIN
    ).held_points
    rows, paths = [], []
    for farm in common.FARMS:
        farm_rows, farm_paths = fit_farm(
            cache=cache, site=site, farm=farm, pinned_points=pinned_points
        )
        rows += farm_rows
        paths += farm_paths
        print(f"{site} farm {farm}: done", flush=True)
    scale = table.effective_capacity_mw / 100.0
    pl.DataFrame(rows).with_columns(
        scale_mw_per_point=pl.lit(scale), pinned_points=pl.lit(pinned_points)
    ).write_parquet(common.WORK_DIR / f"recovery_{site}.parquet")
    pl.DataFrame(paths).with_columns(scale_mw_per_point=pl.lit(scale)).write_parquet(
        common.WORK_DIR / f"paths_{site}.parquet"
    )


def main() -> None:
    """Write `work/recovery_<site>.parquet` and `work/paths_<site>.parquet`."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    for site in arguments.sites:
        run_site(cache=cache, site=site)


if __name__ == "__main__":
    main()
