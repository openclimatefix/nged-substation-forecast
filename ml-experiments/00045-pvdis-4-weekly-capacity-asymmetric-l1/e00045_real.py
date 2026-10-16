"""Real-series weekly capacity fits, recovery-ratio correction, bootstrap, and vintages (00045).

Four variants are fitted to each of the 8 Primaries, on all of the Primary's usable half-hours:

- `frozen`: the frozen placeholder pair (a = 4, r = 30), which equals the strictly non-decreasing
  fit and failed the injection test.
- `alt_post_hoc`: a = 2, r = 10, chosen after the injection test (post hoc): the pair that met the
  false-step limits on the placebo sky and recovered 0.79 of a 26-week fall in tuning.
- `season_year_constant`: one capacity per season-year.
- `free_weekly`: a = 0, one unconstrained (non-negative) capacity per week; the gap diagnostic.

Every path is divided by the Primary's recovery ratio of that variant, the median over the three
tuning farms of the mean fitted increment of a constant injection divided by the injected
capacity. The 90% sampling interval is a 7-day residual block bootstrap that refits the whole path
(200 draws per Primary and variant). The bootstrap does not carry the uncertainty of the recovery
ratio. In an `unobserved` week the interval is widened: each side of the interval grows to at
least the mean width of the nearest observed week on each side of the gap, and then is multiplied
by the square root of 1 plus the number of unobserved weeks in the gap. The two systematic
scenarios are separate columns and are never merged into the interval:

- `rooftop_low_mw` and `rooftop_high_mw`: the capacity minus and plus 10% (the farm-versus-rooftop
  shape difference of experiment 00040's clean-Primary ratio of the daily to the six-arm capacity).
- `scenario_dayL_mw`: the capacity if an undetected day-long load had made the estimate too high
  by experiment 00040's bound B (the estimate over the truth, minus one).
- `scenario_sunload_mw`: the capacity if a load that rises with sunshine had made the estimate too
  low (experiment 00040's median undetected bias at amplitude 0.25).

Vintages (frozen and `alt_post_hoc`) refit at origins every 4 weeks over the last 3 years plus the
latest week, truncated at the origin, with `s_P` and the recovery ratio recomputed from data before
the origin. Vintages carry no bootstrap interval.

    POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_real.py
"""

import argparse
import dataclasses
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final, Literal

import e00045_bootstrap as bootstrap
import e00045_common as common
import e00045_inject as inject
import e00045_model as model
import e00045_score as score
import numpy as np
import polars as pl

VariantName = Literal["frozen", "alt_post_hoc", "season_year_constant", "free_weekly"]
VARIANTS: Final[dict[VariantName, inject.FitSpec]] = {
    "frozen": ("asymmetric", 4.0, 30.0),
    "alt_post_hoc": ("asymmetric", 2.0, 10.0),
    "season_year_constant": ("season_year", 0.0, 0.0),
    "free_weekly": ("free", 0.0, 0.0),
}
"""Each variant's fit name, `a`, and `r`."""
VINTAGE_VARIANTS: Final[tuple[VariantName, ...]] = ("frozen", "alt_post_hoc")
RECOMMENDED: Final[VariantName] = "season_year_constant"
RECOMMENDED_REASON: Final[str] = (
    "season_year_constant has the lowest path error on held-out units (0.027), and neither weekly "
    "variant beat it. alt_post_hoc (a=2, r=10) is offered for a change within a year, but on "
    "held-out units its constant-path false steps (median 1.22, worst 3.32 per Primary-year) and "
    "its coverage failed the plan's limits, and it recovered 0.52 of an 8-week fall. "
    "season_year_constant falls to about 0 MW at Mount Bridge Boston from 2023-09-04 to "
    "2025-08-25 (104 weeks, all unobserved), outside the report period; do not read those weeks."
)
STATUS: Final[dict[str, str]] = {
    "D5": "range-only",
    "D9": "low-power",
    "D10": "clean",
    "D11": "clean",
    "D14": "clean",
    "D15": "not-clean",
    "D16": "clean",
    "D18": "clean",
}
"""Kirton is range-only, Trusthorpe low-power, and Wrangle T1 not-clean."""
N_DRAWS: Final[int] = 200
ROOFTOP_SHARE: Final[float] = 0.10
SUNLOAD_UNDETECTED_BIAS: Final[float] = -0.333581
"""Experiment 00040's median undetected bias of a load that rises with sunshine, amplitude 0.25."""
SCENARIO_BOUNDS_PATH: Final[Path] = inject.EXPERIMENT_00040_SCORES_PATH
REAL_DIR: Final[Path] = common.DATA_DIR / "real"
LOW_PERCENTILE: Final[float] = 5.0
HIGH_PERCENTILE: Final[float] = 95.0
FINAL_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "week_start": pl.Date,
    "time_series_id": pl.Int32,
    "primary_name": pl.String,
    "variant": pl.String,
    "capacity_mw": pl.Float64,
    "low_mw": pl.Float64,
    "high_mw": pl.Float64,
    "status": pl.String,
    "unobserved": pl.Boolean,
    "recovery_ratio": pl.Float64,
    "rooftop_low_mw": pl.Float64,
    "rooftop_high_mw": pl.Float64,
    "scenario_dayL_mw": pl.Float64,
    "scenario_sunload_mw": pl.Float64,
    "recommended": pl.Boolean,
}


def ratio_designs(*, site: str) -> tuple[inject.PrimaryBase, list[tuple[model.Design, np.ndarray]]]:
    """Build the three tuning farms' constant-injection designs of one Primary.

    Args:
        site: Frame site code.

    Returns:
        The Primary's baseline and one (design, observed-week mask) pair per tuning farm.
    """
    inputs = inject.load_inputs()
    base = inject.primary_base(inputs=inputs, site=site)
    designs = [
        score.ratio_design(base=base, inputs=inputs, farm=farm)[:2] for farm in inject.TUNING_FARMS
    ]
    return base, designs


def recovery_ratio_at(
    *,
    designs: list[tuple[model.Design, np.ndarray]],
    spec: inject.FitSpec,
    capacity: float,
    origin: int | None,
) -> float:
    """Return the recovery ratio of a fit from data before `origin` (all data when `None`).

    Args:
        designs: The tuning farms' constant-injection designs and observed-week masks.
        spec: The fit.
        capacity: The injected constant capacity, MW.
        origin: First week not to use, or `None`.

    Returns:
        The median over the farms of the mean fitted increment over the injected capacity.
    """
    grid = np.arange(common.N_WEEKS)
    values = []
    for design, observed in designs:
        path = model.FITS[spec[0]](design=design, a=spec[1], r=spec[2], origin_week=origin)
        use = observed & np.isfinite(path)
        if origin is not None:
            use &= grid < origin
        values.append(float(np.mean((path - capacity)[use]) / capacity))
    return float(np.median(values))


def scenario_bounds() -> pl.DataFrame:
    """Return experiment 00040's bound B of a day-long load by site and season-year."""
    return pl.read_parquet(SCENARIO_BOUNDS_PATH).select("site", "year", "B_dayL")


def bound_by_week(*, site: str, bounds: pl.DataFrame) -> np.ndarray:
    """Return the bound B of every week: that of its season-year, or of the nearest year held.

    Args:
        site: Frame site code.
        bounds: `scenario_bounds()`.

    Returns:
        B of each of the `common.N_WEEKS` weeks.
    """
    mine = bounds.filter(pl.col("site") == site).sort("year")
    years = mine["year"].to_numpy()
    values = mine["B_dayL"].to_numpy()
    week_years = model.season_years_of_weeks()
    return np.array([values[np.argmin(np.abs(years - year))] for year in week_years])


def widen_unobserved(
    *, point: np.ndarray, low: np.ndarray, high: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Widen the interval of every unobserved week inside the fitted span.

    Args:
        point: The corrected path, MW (NaN outside the span).
        low: The bootstrap lower bound, MW.
        high: The bootstrap upper bound, MW.
        observed: Mask of weeks that hold rows.

    Returns:
        The new lower and upper bounds. A gap of `k` unobserved weeks gets, on each side, the larger
        of its own width and the mean width of the observed weeks next to the gap, times
        `sqrt(1 + k)`.
    """
    low, high = low.copy(), high.copy()
    finite = np.isfinite(point)
    below, above = point - low, high - point
    week = 0
    n = len(point)
    while week < n:
        if observed[week] or not finite[week]:
            week += 1
            continue
        end = week
        while end < n and finite[end] and not observed[end]:
            end += 1
        k = end - week
        neighbours = [w for w in (week - 1, end) if 0 <= w < n and observed[w] and finite[w]]
        floor_below = np.mean([below[w] for w in neighbours]) if neighbours else 0.0
        floor_above = np.mean([above[w] for w in neighbours]) if neighbours else 0.0
        factor = math.sqrt(1 + k)
        for w in range(week, end):
            low[w] = point[w] - max(below[w], floor_below) * factor
            high[w] = point[w] + max(above[w], floor_above) * factor
        week = end
    return np.maximum(low, 0.0), high


@dataclasses.dataclass(frozen=True)
class RealFit:
    """One Primary's fitted paths and bootstrap draws for one variant.

    Attributes:
        variant: The variant.
        ratio: Recovery ratio used.
        point: Corrected path, MW.
        low: Lower 90% bound, MW.
        high: Upper 90% bound, MW.
        observed: Mask of weeks that hold rows.
        clipped: Number of weeks where the bootstrap interval missed the point and was extended.
    """

    variant: VariantName
    ratio: float
    point: np.ndarray
    low: np.ndarray
    high: np.ndarray
    observed: np.ndarray
    clipped: int


def fit_variant(
    *,
    design: model.Design,
    table: pl.DataFrame,
    variant: VariantName,
    ratio: float,
    rng: np.random.Generator,
    n_draws: int,
) -> RealFit:
    """Fit one variant on the real series and bootstrap its interval.

    Args:
        design: The Primary's design.
        table: The Primary's `model.usable_rows_table` (the rows of `design`, in order).
        variant: The variant.
        ratio: The Primary's recovery ratio of the variant.
        rng: The random generator.
        n_draws: Number of bootstrap draws.

    Returns:
        The fit.
    """
    name, a, r = VARIANTS[variant]
    fit = model.FITS[name]
    raw = fit(design=design, a=a, r=r)
    point = raw / ratio
    resid = bootstrap.residuals(rows=design.rows, path=raw)
    fitted = design.rows.power - resid
    grid, slot = bootstrap.residual_grid(table=table, residual=resid)
    problem = design.problem()
    observed = np.zeros(design.n_weeks, dtype=bool)
    observed[problem.span[0] : problem.span[1] + 1] = problem.observed
    draws = np.full((n_draws, design.n_weeks), np.nan)
    for draw in range(n_draws):
        source = bootstrap.source_weeks(observed=observed, rng=rng)
        new_power = fitted + grid[source[design.rows.week], slot[:, 0], slot[:, 1]]
        redrawn = model.Design(
            rows=dataclasses.replace(design.rows, power=new_power),
            n_weeks=design.n_weeks,
            week_season_year=design.week_season_year,
        )
        draws[draw] = fit(design=redrawn, a=a, r=r, recovery_ratio=ratio)
    low, high = np.nanpercentile(draws, [LOW_PERCENTILE, HIGH_PERCENTILE], axis=0)
    clipped = int(np.sum(np.isfinite(point) & ((low > point) | (high < point))))
    low, high = np.fmin(low, point), np.fmax(high, point)
    if variant != "free_weekly":
        low, high = widen_unobserved(point=point, low=low, high=high, observed=observed)
    return RealFit(
        variant=variant,
        ratio=ratio,
        point=point,
        low=low,
        high=high,
        observed=observed,
        clipped=clipped,
    )


def weekly_frame(
    *,
    site: str,
    variant: VariantName,
    ratio: float,
    point: np.ndarray,
    low: np.ndarray | None,
    high: np.ndarray | None,
    observed: np.ndarray,
    bound: np.ndarray,
) -> pl.DataFrame:
    """Return one Primary's rows for one variant: a row per week with a path value or a gap.

    Args:
        site: Frame site code.
        variant: The variant.
        ratio: Recovery ratio used.
        point: Corrected path, MW (NaN outside the span; also NaN for the free fit in a gap).
        low: Lower 90% bound, or `None` for no interval.
        high: Upper 90% bound, or `None` for no interval.
        observed: Mask of weeks that hold rows.
        bound: Experiment 00040's B of every week.

    Returns:
        Rows of `FINAL_SCHEMA` for the weeks inside the span.
    """
    span = np.nonzero(observed | np.isfinite(point))[0]
    first, last = int(span.min()), int(span.max())
    weeks = np.arange(first, last + 1)
    cap = point[weeks]
    nan = np.full(len(weeks), np.nan)
    return pl.DataFrame(
        {
            "week_start": [common.week_start(week=int(w)).date() for w in weeks],
            "time_series_id": [common.PRIMARY_SERIES_IDS[site]] * len(weeks),
            "primary_name": [common.PRIMARY_NAMES[site]] * len(weeks),
            "variant": [variant] * len(weeks),
            "capacity_mw": cap,
            "low_mw": nan if low is None else low[weeks],
            "high_mw": nan if high is None else high[weeks],
            "status": [STATUS[site]] * len(weeks),
            "unobserved": ~observed[weeks],
            "recovery_ratio": [ratio] * len(weeks),
            "rooftop_low_mw": cap * (1 - ROOFTOP_SHARE),
            "rooftop_high_mw": cap * (1 + ROOFTOP_SHARE),
            "scenario_dayL_mw": cap / (1 + bound[weeks]),
            "scenario_sunload_mw": cap / (1 + SUNLOAD_UNDETECTED_BIAS),
            "recommended": [variant == RECOMMENDED] * len(weeks),
        },
        schema=FINAL_SCHEMA,
    ).with_columns(
        pl.col(
            "capacity_mw",
            "low_mw",
            "high_mw",
            "rooftop_low_mw",
            "rooftop_high_mw",
            "scenario_dayL_mw",
            "scenario_sunload_mw",
        ).fill_nan(None)
    )


def final_job(*, site: str, n_draws: int) -> Path:
    """Fit the four variants on one Primary, bootstrap them, and write the table atomically.

    Args:
        site: Frame site code.
        n_draws: Bootstrap draws per variant.

    Returns:
        The path of the output file.
    """
    out_file = REAL_DIR / f"final_{site}.parquet"
    if out_file.exists():
        return out_file
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    inputs = inject.load_inputs()
    table = model.usable_rows_table(half_hourly=inputs.half_hourly, weeks=inputs.weeks, site=site)
    design = model.design_from_rows_table(table=table)
    base, ratio_sets = ratio_designs(site=site)
    bound = bound_by_week(site=site, bounds=scenario_bounds())
    parts, notes = [], []
    for index, variant in enumerate(VARIANTS):
        started = time.monotonic()
        ratio = recovery_ratio_at(
            designs=ratio_sets, spec=VARIANTS[variant], capacity=base.capacity, origin=None
        )
        rng = np.random.default_rng([common.PRIMARIES.index(site), index, 45, 7])
        fit = fit_variant(
            design=design, table=table, variant=variant, ratio=ratio, rng=rng, n_draws=n_draws
        )
        parts.append(
            weekly_frame(
                site=site,
                variant=variant,
                ratio=ratio,
                point=fit.point,
                low=fit.low,
                high=fit.high,
                observed=fit.observed,
                bound=bound,
            )
        )
        notes.append(
            f"{site} {variant}: ratio {ratio:.3f}, {fit.clipped} weeks with the point outside "
            f"the interval, {time.monotonic() - started:.0f} s"
        )
    score.atomic_write(frame=pl.concat(parts), path=out_file)
    print("\n".join(notes), flush=True)
    return out_file


def _final_job(args: tuple[str, int]) -> str:
    final_job(site=args[0], n_draws=args[1])
    return f"final {args[0]}"


def vintage_job(*, site: str) -> Path:
    """Refit `frozen` and `alt_post_hoc` at every origin on one Primary, truncated at the origin.

    Args:
        site: Frame site code.

    Returns:
        The path of the output file.
    """
    out_file = REAL_DIR / f"vintage_{site}.parquet"
    if out_file.exists():
        return out_file
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    inputs = inject.load_inputs()
    table = model.usable_rows_table(half_hourly=inputs.half_hourly, weeks=inputs.weeks, site=site)
    design = model.design_from_rows_table(table=table)
    base, ratio_sets = ratio_designs(site=site)
    bound = bound_by_week(site=site, bounds=scenario_bounds())
    origins = (*score.vintage_origins(), common.N_WEEKS)
    parts = []
    for origin in origins:
        problem_rows = design.rows.before(origin_week=origin)
        observed_before = np.bincount(problem_rows.week, minlength=common.N_WEEKS) > 0
        if observed_before.sum() < score.MIN_VINTAGE_WEEKS or any(
            observed[:origin].sum() < score.MIN_VINTAGE_WEEKS for _, observed in ratio_sets
        ):
            continue
        for variant in VINTAGE_VARIANTS:
            spec = VARIANTS[variant]
            ratio = recovery_ratio_at(
                designs=ratio_sets, spec=spec, capacity=base.capacity, origin=origin
            )
            path = model.FITS[spec[0]](
                design=design, a=spec[1], r=spec[2], origin_week=origin, recovery_ratio=ratio
            )
            frame = weekly_frame(
                site=site,
                variant=variant,
                ratio=ratio,
                point=path,
                low=None,
                high=None,
                observed=observed_before,
                bound=bound,
            ).with_columns(origin_monday=pl.lit(common.week_start(week=origin).date()))
            parts.append(frame)
    score.atomic_write(frame=pl.concat(parts), path=out_file)
    return out_file


def _vintage_job(site: str) -> str:
    vintage_job(site=site)
    return f"vintage {site}"


def main() -> None:
    """Run the final fits (with bootstrap) and the vintages, one job per Primary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("final", "vintages", "all"), default="all")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sites = common.PRIMARIES
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage in ("final", "all"):
            for line in pool.map(_final_job, [(s, args.draws) for s in sites]):
                print(line, flush=True)
        if args.stage in ("vintages", "all"):
            for line in pool.map(_vintage_job, sites):
                print(line, flush=True)


if __name__ == "__main__":
    main()
