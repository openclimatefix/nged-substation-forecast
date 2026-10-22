"""The ensemble-driven GAM-K forecast of one site with price-following-asset arms (experiment 191).

The engine is experiment 152's arm N (a GAM-K fit with fitted solar capacity, monthly refits on a
24-month window, experiment 133's shared filter cell, a spread fitted on tuning months, driven by
the calibrated ensemble mean), with four price indicators added to the design matrix.

**Arms.** One monthly cut produces three fits that share their training rows.

- `a0`: the model without a price input.
- `joint`: the model with four price columns, each `-gate * indicator` with a non-negative
  coefficient, so a price-following asset can only lower net load. Arms `h`, `m` and `p` use this
  fit and differ only in the price input at forecast time: the true price (`h`, a hindsight
  ceiling), the training-period expected indicator given the half-hour of the day, the day type and
  the quarter (`m`), or the last settled day's indicator at the same half-hour (`p`).
- `t`: the model without a price input, refitted with the half-hours that the joint fit explains as
  price-driven removed from training.

The filter's residual stream for `h`, `m` and `p` subtracts the joint fit's price term at the true
price, because settled prices are known for the past.
"""

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Final

import e00191_common as common
import e00191_inputs as inputs
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    correction_at,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_ens import LeadHourSpread, assert_cut_not_after, latest_cut_index
from studies.dp_ens_weather import RunWindow, proxy_cache, replace_site
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import (
    QUANTILE_COLUMNS,
    GamFit,
    GamSpec,
    TrainingWindow,
    design_matrix,
    last_signal_season_year,
    penalty_matrix,
    predict_gam,
    prepare_fit,
    row_chunks,
    solve_bounded_quadratic,
)

N_PRICE: Final[int] = len(common.FEATURE_NAMES)
HALF_HOURS_PER_DAY: Final[int] = 48
KINDS: Final[dict[str, tuple[str, str | None]]] = {
    "a0": ("a0", None),
    "h": ("joint", "true"),
    "m": ("joint", "expected"),
    "p": ("joint", "persist"),
    "t": ("t", None),
    "jb": ("joint", None),
}
"""Each kind's fit family and the price input it takes at forecast time. `jb` is a diagnostic: the
joint fit's base model with its price term switched off at forecast time."""
FAMILIES: Final[tuple[str, ...]] = ("a0", "joint", "t")


def filter_hyper() -> FilterHyperparameters:
    """Return experiment 133's shared filter cell (POOL)."""
    return FilterHyperparameters.from_array(
        values=np.array(json.loads(common.POOL_PATH.read_text())["hyper"])
    )


def gam_spec() -> GamSpec:
    """Return the GAM-K penalties: experiment 104's ridge, no drift, a fitted capacity."""
    step_penalty = json.loads(common.PART_4_TUNED.read_text())["step_penalty"]
    return GamSpec(
        ridge=common.RIDGE,
        step_penalty=step_penalty,
        drift="none",
        capacity="fitted",
        final_knot="free",
    )


def masked_cache(*, cache: FeatureCache, site: str) -> FeatureCache:
    """Return the cache with D8's rows before its first valid day removed (D8 only)."""
    if site != "D8":
        return cache
    features = cache.sites[site]
    early = cache.end_times <= naive_utc(moment=common.D8_FIRST_DAY)
    reading = features.y.copy()
    reading[early] = np.nan
    return replace_site(
        cache=cache,
        site=site,
        y=reading,
        present=features.present & ~early,
        default_rows=features.default_rows & ~early,
    )


# --- The price inputs on the grid ------------------------------------------------------------


@dataclass(frozen=True)
class PriceGrid:
    """The price indicators on a site's grid.

    Attributes:
        indicator: The ungated indicators, shape (T, 4); 0 where the price is missing.
        gate: 1 from the asset's connection date, else 0, shape (T,).
        key: The expected-value cell of each row (day type, quarter, local half-hour).
        spread_positive: True where the spark spread is positive, shape (T,), for scoring cuts.
        price_p75: True where the price is above its 75th percentile, for scoring cuts.
        n_missing: The number of grid rows with a gate of 1 and no price.
    """

    indicator: np.ndarray
    gate: np.ndarray
    key: np.ndarray
    spread_positive: np.ndarray
    price_p75: np.ndarray
    n_missing: int


def build_price_grid(
    *, cache: FeatureCache, site: str, ungated: bool, features: str = "halfhourly"
) -> PriceGrid:
    """Return the price indicators and the asset's gate on the cache's grid."""
    prices = pl.read_parquet(common.PRICES_PATH).sort("start")
    start = prices["start"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    grid_start = cache.end_times - np.timedelta64(30, "m")
    position = np.clip(np.searchsorted(start, grid_start), 0, len(start) - 1)
    found = start[position] == grid_start
    indicator = np.zeros((len(grid_start), N_PRICE))
    for column, name in enumerate(common.FEATURE_SETS[features]):
        values = prices[name].fill_null(0.0).to_numpy()
        indicator[:, column] = np.where(found, values[position], 0.0)
    connected = common.CONNECTED.get(site)
    if ungated:
        gate = np.ones(len(grid_start))
    elif connected is None:
        gate = np.zeros(len(grid_start))
    else:
        gate = (cache.end_times > naive_utc(moment=connected)).astype(np.float64)
    calendar = cache.calendar
    key = (
        calendar.day_type.astype(np.int64) * 4 + calendar.quarter.astype(np.int64)
    ) * HALF_HOURS_PER_DAY + calendar.half_hour.astype(np.int64)
    spread = np.where(found, prices["spread_pos"].fill_null(0.0).to_numpy()[position], 0.0) > 0.5
    p75 = np.where(found, prices["price_p75"].fill_null(0.0).to_numpy()[position], 0.0) > 0.5
    return PriceGrid(
        indicator=indicator,
        gate=gate,
        key=key,
        spread_positive=spread,
        price_p75=p75,
        n_missing=int(((gate > 0) & ~found).sum()),
    )


def expected_table(*, grid: PriceGrid, index: np.ndarray) -> np.ndarray:
    """Return the mean ungated indicators per cell over the training rows `index`.

    A cell with no training row takes the mean over all training rows.
    """
    n_cells = int(grid.key.max()) + 1
    cells = grid.key[index]
    counts = np.bincount(cells, minlength=n_cells).astype(np.float64)
    table = np.zeros((n_cells, N_PRICE))
    for column in range(N_PRICE):
        sums = np.bincount(cells, weights=grid.indicator[index, column], minlength=n_cells)
        table[:, column] = np.where(
            counts > 0, sums / np.maximum(counts, 1.0), grid.indicator[index, column].mean()
        )
    return table


def true_indicator(*, grid: PriceGrid, rows: np.ndarray) -> np.ndarray:
    """Return the gated true indicators at grid rows, shape (n, 4)."""
    return grid.gate[rows, None] * grid.indicator[rows]


def persisted_indicator(
    *, cache: FeatureCache, grid: PriceGrid, rows: np.ndarray, origin: datetime
) -> np.ndarray:
    """Return the gated indicators of the last settled day at the same half-hour, shape (n, 4).

    A row whose half-hour ends `d` after the origin takes the row `48 * ceil(d / 24 h)` half-hours
    earlier, the newest day whose half-hour had ended by the origin.
    """
    origin64 = naive_utc(moment=origin)
    ahead_days = np.ceil((cache.end_times[rows] - origin64) / np.timedelta64(24, "h")).astype(
        np.int64
    )
    source = rows - HALF_HOURS_PER_DAY * np.maximum(ahead_days, 1)
    if (source < 0).any() or (cache.end_times[source] > origin64).any():
        msg = "a persisted price row ends after the origin"
        raise ValueError(msg)
    return grid.gate[rows, None] * grid.indicator[source]


# --- Fits ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceFit:
    """A fit with a price term: base coefficients, price coefficients, and the expected table."""

    fit: GamFit
    theta_price: np.ndarray
    expected: np.ndarray


@dataclass(frozen=True)
class CutFits:
    """The three fits of one monthly cut."""

    cut: datetime
    a0: GamFit
    joint: PriceFit
    t: GamFit
    n_masked: int
    n_rows: int


def _cut_fits(*, cache: FeatureCache, site: str, cut: datetime, grid: PriceGrid) -> CutFits:
    """Fit a0, joint and t at one cut from one pass over the training rows."""
    spec = gam_spec()
    window = TrainingWindow(months=common.WINDOW_MONTHS)
    features = cache.sites[site]
    layout, index = prepare_fit(
        cache=cache,
        site=site,
        spec=spec,
        cut=cut,
        window=window,
        season_years=common.SEASON_YEARS,
    )
    p = layout.n_columns
    q = p + N_PRICE
    gram = np.zeros((q, q))
    rhs = np.zeros(q)
    for chunk in row_chunks(index=index):
        rows = index[chunk]
        x = np.hstack(
            [
                design_matrix(cache=cache, features=features, layout=layout, index=rows),
                -true_indicator(grid=grid, rows=rows),
            ]
        )
        gram += x.T @ x
        rhs += x.T @ features.y[rows]
    penalty_base, cost_base = penalty_matrix(layout=layout, n_rows=len(index))
    constrained_base = torch.as_tensor(layout.constrained)

    def solve(*, size: int, gram_part: np.ndarray, rhs_part: np.ndarray) -> np.ndarray:
        penalty = np.zeros((size, size))
        penalty[:p, :p] = penalty_base
        penalty[p:, p:] = common.RIDGE * np.eye(size - p)
        cost = np.zeros(size)
        cost[:p] = cost_base
        constrained = torch.cat([constrained_base, torch.ones(size - p, dtype=torch.bool)])
        theta, _ = solve_bounded_quadratic(
            matrix=torch.as_tensor(gram_part + penalty)[None],
            rhs=torch.as_tensor(rhs_part - 0.5 * cost)[None],
            constrained=constrained,
        )
        return theta[0].numpy()

    def gam_fit(*, theta: np.ndarray) -> GamFit:
        return GamFit(
            layout=layout,
            theta=theta[None, :p].copy(),
            cut=cut,
            n_rows=len(index),
            site=site,
            last_season_year=last_signal_season_year(
                cache=cache, site=site, index=index, cut=cut, pinned=None
            ),
            pinned=None,
        )

    theta_joint = solve(size=q, gram_part=gram, rhs_part=rhs)
    theta_a0 = solve(size=p, gram_part=gram[:p, :p], rhs_part=rhs[:p])
    price_output = true_indicator(grid=grid, rows=index) @ theta_joint[p:]
    largest = float(price_output.max()) if len(price_output) else 0.0
    masked = (
        price_output > common.MASK_SHARE_OF_MAXIMUM * largest
        if largest >= common.MASK_MINIMUM_POINTS
        else np.zeros(len(index), dtype=bool)
    )
    if masked.any():
        gram_m = np.zeros((p, p))
        rhs_m = np.zeros(p)
        for chunk in row_chunks(index=index[masked]):
            rows = index[masked][chunk]
            x = design_matrix(cache=cache, features=features, layout=layout, index=rows)
            gram_m += x.T @ x
            rhs_m += x.T @ features.y[rows]
        penalty_t, cost_t = penalty_matrix(layout=layout, n_rows=len(index) - int(masked.sum()))
        theta_t = solve_masked(
            gram=gram[:p, :p] - gram_m,
            rhs=rhs[:p] - rhs_m,
            penalty=penalty_t,
            cost=cost_t,
            constrained=constrained_base,
        )
    else:
        theta_t = theta_a0
    return CutFits(
        cut=cut,
        a0=gam_fit(theta=theta_a0),
        joint=PriceFit(
            fit=gam_fit(theta=theta_joint[:p]),
            theta_price=theta_joint[p:].copy(),
            expected=expected_table(grid=grid, index=index),
        ),
        t=gam_fit(theta=theta_t),
        n_masked=int(masked.sum()),
        n_rows=len(index),
    )


def solve_masked(
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
    penalty: np.ndarray,
    cost: np.ndarray,
    constrained: torch.Tensor,
) -> np.ndarray:
    """Solve the bounded penalised least squares problem on a Gram matrix."""
    theta, _ = solve_bounded_quadratic(
        matrix=torch.as_tensor(gram + penalty)[None],
        rhs=torch.as_tensor(rhs - 0.5 * cost)[None],
        constrained=constrained,
    )
    return theta[0].numpy()


def first_cut_index(*, cache: FeatureCache, site: str, cuts: list[datetime]) -> int:
    """Return the index of the first cut with `MIN_TRAINING_DAYS` days of usable rows."""
    features = cache.sites[site]
    usable = features.default_rows & np.isfinite(features.y)
    needed = common.MIN_TRAINING_DAYS * 48
    for index, cut in enumerate(cuts):
        if int((usable & (cache.end_times <= naive_utc(moment=cut))).sum()) >= needed:
            return index
    msg = f"{site} never has {common.MIN_TRAINING_DAYS} days of usable rows"
    raise ValueError(msg)


def fit_all_cuts(*, cache: FeatureCache, site: str, grid: PriceGrid, tag: str) -> list[CutFits]:
    """Fit every monthly cut from the site's first, caching the result on disk."""
    path = common.FITS_DIR / tag / f"{site}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    cuts = common.monthly_cuts()
    fits = [
        _cut_fits(cache=cache, site=site, cut=cut, grid=grid)
        for cut in cuts[first_cut_index(cache=cache, site=site, cuts=cuts) :]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(fits))
    temporary.rename(path)
    return fits


# --- Forecasting --------------------------------------------------------------------------------


def family_fit(*, cut_fits: CutFits, family: str) -> GamFit:
    """Return the base GAM of a family at one cut."""
    return {"a0": cut_fits.a0, "joint": cut_fits.joint.fit, "t": cut_fits.t}[family]


def price_output(
    *,
    cut_fits: CutFits,
    mode: str,
    cache: FeatureCache,
    grid: PriceGrid,
    rows: np.ndarray,
    origin: datetime | None,
) -> np.ndarray:
    """Return the power the price term takes off net load at grid rows, in points (>= 0)."""
    theta = cut_fits.joint.theta_price
    if mode == "true":
        indicator = true_indicator(grid=grid, rows=rows)
    elif mode == "expected":
        indicator = grid.gate[rows, None] * cut_fits.joint.expected[grid.key[rows]]
    elif mode == "persist":
        assert origin is not None
        indicator = persisted_indicator(cache=cache, grid=grid, rows=rows, origin=origin)
    else:
        msg = f"unknown price mode {mode!r}"
        raise ValueError(msg)
    return indicator @ theta


def residual_stream(
    *, cache: FeatureCache, fits: list[CutFits], family: str, grid: PriceGrid, site: str
) -> np.ndarray:
    """Return the reading minus the chained fits' prediction on a cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut. The joint family subtracts its
    price term at the true price.
    """
    if not fits or any(b.cut <= a.cut for a, b in pairwise(fits)):
        msg = "the residual stream needs fits with strictly ascending cuts"
        raise ValueError(msg)
    features = cache.sites[site]
    stream = np.full(len(cache.end_times), np.nan)
    cuts = [
        int(np.searchsorted(cache.end_times, naive_utc(moment=f.cut), side="right")) - 1
        for f in fits
    ]
    ends = [*cuts[1:], len(stream) - 1]
    for cut_fits, cut, end in zip(fits, cuts, ends, strict=True):
        rows = np.arange(cut + 1, end + 1)
        usable = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
            & np.isfinite(features.y[rows])
        )
        rows = rows[usable]
        if not len(rows):
            continue
        prediction = predict_gam(
            cache=cache, fit=family_fit(cut_fits=cut_fits, family=family), index=rows
        )
        if family == "joint":
            prediction = prediction - price_output(
                cut_fits=cut_fits, mode="true", cache=cache, grid=grid, rows=rows, origin=None
            )
        stream[rows] = features.y[rows] - prediction
    return stream


@dataclass(frozen=True)
class RunForecast:
    """One run's forecast of every kind before the spread."""

    window: slice
    reading: dict[str, np.ndarray]
    level: dict[str, np.ndarray]
    anomaly: dict[str, np.ndarray]


def forecast_run(
    *,
    cache: FeatureCache,
    setup: inputs.SiteSetup,
    run_index: int,
    fits: list[CutFits],
    traces: dict,
    grid: PriceGrid,
    kinds: tuple[str, ...],
) -> RunForecast:
    """Forecast one run from the ensemble mean with every kind."""
    run = setup.runs[run_index]
    origin = setup.origins[run_index]
    members = run.usable_members()
    mean_run = type(run)(
        origin_position=run.origin_position,
        temperature_c=run.temperature_c[members].mean(axis=0)[None],
        shortwave_w_m2=run.shortwave_w_m2[members].mean(axis=0)[None],
    )
    window = RunWindow.build(cache=cache, context=setup.context, run=run)
    driven = window.driven_caches(context=setup.context, run=mean_run, members=[0])[0]
    n_targets = window.window.stop - window.window.start
    target = np.arange(window.window.start, window.window.stop)
    band = cache.calendar.band[target].astype(np.int64)
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=np.array([naive_utc(moment=origin)]),
        delay_hours=DEFAULT_DELAY_HOURS,
    )
    chosen = latest_cut_index(cuts=[f.cut for f in fits], moment=origin)
    if chosen < 0:
        msg = f"no fit has a cut at or before {origin}"
        raise ValueError(msg)
    cut_fits = fits[chosen]
    base: dict[str, np.ndarray] = {}
    out = RunForecast(window=window.window, reading={}, level={}, anomaly={})
    for kind in kinds:
        family, mode = KINDS[kind]
        fit = family_fit(cut_fits=cut_fits, family=family)
        assert_cut_not_after(fit=fit, origin=origin)
        if family not in base:
            base[family] = predict_gam(cache=driven, fit=fit, index=np.arange(n_targets))
        reading = base[family]
        if mode is not None:
            reading = reading - price_output(
                cut_fits=cut_fits, mode=mode, cache=cache, grid=grid, rows=target, origin=origin
            )
        correction = correction_at(
            trace=traces[family], cutoff=np.full(n_targets, cutoff[0]), target=target, band=band
        )
        out.reading[kind] = reading
        out.level[kind], out.anomaly[kind] = correction.level, correction.anomaly
    return out


def sunny_free_frames(
    *,
    cache: FeatureCache,
    setup: inputs.SiteSetup,
    run_indices: list[int],
    fits: list[CutFits],
    traces: dict,
    grid: PriceGrid,
    kinds: tuple[str, ...],
) -> pl.DataFrame:
    """Forecast the runs and return one row per scored half-hour and kind, before any spread."""
    features = cache.sites[setup.site]
    frames = []
    for index in run_indices:
        run = setup.runs[index]
        if len(cache.end_times) - run.origin_position - 1 < common.MIN_WINDOW_HALF_HOURS:
            continue
        forecast = forecast_run(
            cache=cache,
            setup=setup,
            run_index=index,
            fits=fits,
            traces=traces,
            grid=grid,
            kinds=kinds,
        )
        rows = np.arange(forecast.window.start, forecast.window.stop)
        keep = features.present[rows] & np.isfinite(features.y[rows])
        if not keep.any():
            continue
        origin64 = np.datetime64(setup.origins[index].replace(tzinfo=None), "us")
        valid = cache.end_times[rows][keep]
        n = int(keep.sum())
        for kind in kinds:
            frames.append(  # noqa: PERF401
                pl.DataFrame(
                    {
                        "run": np.full(n, index, dtype=np.int32),
                        "power_fcst_init_time": pl.Series(
                            np.full(n, origin64)
                        ).dt.replace_time_zone("UTC"),
                        "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
                        "lead_hours": ((valid - origin64) / np.timedelta64(1, "h")).astype(
                            np.float64
                        ),
                        "kind": np.full(n, kind),
                        "actual": features.y[rows][keep],
                        "reading": forecast.reading[kind][keep],
                        "level": forecast.level[kind][keep],
                        "anomaly": forecast.anomaly[kind][keep],
                        "hour_band": cache.calendar.band[rows][keep].astype(np.int8),
                        "quarter": cache.calendar.quarter[rows][keep].astype(np.int8),
                        "switching_mask": features.switching_mask[rows][keep],
                        "spread_positive": grid.spread_positive[rows][keep],
                        "price_p75": grid.price_p75[rows][keep],
                        "gate": grid.gate[rows][keep],
                    }
                )
            )
    return pl.concat(frames)


def arm_frames(
    *, base: pl.DataFrame, tuning: pl.DataFrame, site_id: int, spread_kind: dict[str, str]
) -> pl.DataFrame:
    """Turn raw forecasts into stored prediction frames, one block per kind.

    Args:
        base: Raw forecasts to score, all kinds.
        tuning: Raw forecasts of the tuning runs whose residuals fit each spread.
        site_id: The series identifier.
        spread_kind: For each kind in `base`, the kind in `tuning` whose residuals set its spread.

    Returns:
        Columns `time_series_id`, `run`, `power_fcst_init_time`, `valid_time`, `lead_hours`,
        `seed`, `arm`, `q01` to `q99`, `actual`, `mean`, and the scoring flags.
    """
    parts = []
    for arm in base["kind"].unique(maintain_order=True):
        fit_rows = tuning.filter((pl.col("kind") == spread_kind[arm]) & ~pl.col("switching_mask"))
        corrected = fit_rows["reading"] + fit_rows["level"] + fit_rows["anomaly"]
        scored = base.filter(pl.col("kind") == arm)
        mean = (scored["reading"] + scored["level"] + scored["anomaly"]).to_numpy()
        spread = LeadHourSpread.fit(
            residuals=(fit_rows["actual"] - corrected).to_numpy().astype(np.float64),
            band=fit_rows["hour_band"].to_numpy().astype(np.int64),
            quarter=fit_rows["quarter"].to_numpy().astype(np.int64),
            lead_hours=fit_rows["lead_hours"].to_numpy(),
        )
        quantiles = np.sort(
            spread.quantiles(
                mean=mean,
                band=scored["hour_band"].to_numpy().astype(np.int64),
                quarter=scored["quarter"].to_numpy().astype(np.int64),
                lead_hours=scored["lead_hours"].to_numpy(),
            ),
            axis=1,
        )
        part = scored.select(
            pl.lit(site_id, dtype=pl.Int32).alias("time_series_id"),
            "run",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            pl.lit(0, dtype=pl.Int8).alias("seed"),
            pl.lit(arm).alias("arm"),
        ).with_columns(
            *[
                pl.Series(column, quantiles[:, position].astype(np.float32))
                for position, column in enumerate(QUANTILE_COLUMNS)
            ],
            scored["actual"].cast(pl.Float32),
            pl.Series("mean", mean.astype(np.float32)),
            scored["level"].cast(pl.Float32),
            scored["anomaly"].cast(pl.Float32),
            scored["hour_band"],
            scored["quarter"],
            scored["switching_mask"],
            scored["spread_positive"],
            scored["price_p75"],
            scored["gate"].cast(pl.Float32),
        )
        parts.append(part)
    return pl.concat(parts)


@dataclass
class Prepared:
    """One site's fitted state for forecasting."""

    work: FeatureCache
    setup: inputs.SiteSetup
    fits: list[CutFits]
    traces: dict
    grid: PriceGrid
    first_origin: datetime

    def usable(self, index: int) -> bool:
        """Return True when a run's origin is at or after the first fitted cut plus the delay."""
        return self.setup.origins[index] >= self.first_origin

    def raw(self, *, periods: set[str], kinds: tuple[str, ...]) -> pl.DataFrame | None:
        """Return the raw forecasts of the usable runs of some periods, or `None` if none."""
        indices = [i for i in self.setup.indices(periods=periods) if self.usable(i)]
        if not indices:
            return None
        return sunny_free_frames(
            cache=self.work,
            setup=self.setup,
            run_indices=indices,
            fits=self.fits,
            traces=self.traces,
            grid=self.grid,
            kinds=kinds,
        )


def variant_tag(*, ungated: bool, features: str) -> str:
    """Return the folder name of a gate and feature-set variant."""
    return ("ungated" if ungated else "gated") + (
        "" if features == "halfhourly" else f"_{features}"
    )


def prepare_site(
    *, site: str, cache: FeatureCache, ungated: bool, features: str = "halfhourly"
) -> Prepared:
    """Fit the monthly models and run the filters for one site."""
    work = masked_cache(cache=cache, site=site)
    setup = inputs.calibrated_setup(cache=work, site=site)
    grid = build_price_grid(cache=work, site=site, ungated=ungated, features=features)
    tag = variant_tag(ungated=ungated, features=features)
    fits = fit_all_cuts(cache=work, site=site, grid=grid, tag=tag)
    proxy = proxy_cache(cache=work, context=setup.context)
    hyper = filter_hyper()
    traces = {
        family: run_kalman(
            residual=residual_stream(cache=proxy, fits=fits, family=family, grid=grid, site=site),
            band=work.calendar.band,
            hyper=hyper,
        )
        for family in FAMILIES
    }
    first_origin = fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    return Prepared(
        work=work, setup=setup, fits=fits, traces=traces, grid=grid, first_origin=first_origin
    )


def run_site(
    *,
    site: str,
    cache: FeatureCache,
    ungated: bool,
    kinds: tuple[str, ...],
    features: str = "halfhourly",
) -> dict[str, int]:
    """Fit, filter and forecast one site, and write each period's frame."""
    prepared = prepare_site(site=site, cache=cache, ungated=ungated, features=features)
    tuning = prepared.raw(periods={"tuning"}, kinds=kinds)
    spread_kind = {kind: kind for kind in kinds}
    if tuning is None:
        donor_frames = []
        for donor in common.DONORS:
            donor_prepared = prepare_site(site=donor, cache=cache, ungated=False)
            frame = donor_prepared.raw(periods={"tuning"}, kinds=("a0",))
            if frame is not None:
                donor_frames.append(frame)
        tuning = pl.concat(donor_frames)
        spread_kind = dict.fromkeys(kinds, "a0")
    site_id = prepared.work.sites[site].table.time_series_id
    tag = variant_tag(ungated=ungated, features=features)
    written: dict[str, int] = {}
    for period in ("replicate", "scored"):
        base = prepared.raw(periods={period}, kinds=kinds)
        if base is None:
            continue
        frame = arm_frames(base=base, tuning=tuning, site_id=site_id, spread_kind=spread_kind)
        target = common.PREDS_DIR / tag / f"{period}_{site}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        written[period] = frame.height
    return written
