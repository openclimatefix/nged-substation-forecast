# ruff: noqa: E501
"""The ensemble-driven GAM-K forecast of one site with regime terms (experiment 294).

Copied from experiment 191's engine, which is experiment 152's GAM-K (fitted solar capacity, monthly
refits, experiment 133's shared filter cell, a spread fitted on tuning months, driven by the
calibrated ensemble mean). Experiment 191's four price columns are replaced by two regime columns.

**Regime columns.** Both enter the design as `-column`, so a positive coefficient takes power off
net load.

- `break`: 1 from 2023-08-15 (the day the engine's output left D8's reading), else 0. Its
  coefficient is free in sign.
- `engine`: the probability that one or more engine units run at the row. Its coefficient is
  non-negative, so the term can only lower net load. It comes in two kinds: the **filtered**
  probability (readings up to that half-hour only), and the **smoothed** probability (all readings).

**Families.** One monthly cut produces five fits from one pass over the training rows.

- `a0`: no regime column (the single model).
- `ind`: the `break` column only.
- `regF`: `break` and the filtered `engine` column. The deciding arm 3.
- `regS`: `break` and the smoothed `engine` column. The hindsight arm.
- `eng`: the filtered `engine` column only, for the training windows in which `break` is constant.
- `regP`: `break` and the filtered probability lagged 1 to 7 days (post hoc), the state as a
  forecast would have seen it, so the coefficient learns the value of a stale state.
- `regU`: `break` and the filtered expected number of running units (post hoc), so the coefficient is
  the offset per running unit.

**Forecast-time state.** `persist` reads the filtered probability at the same half-hour of the last
day that was settled at the forecast origin. `true` reads the smoothed probability at the target
half-hour itself. The Kalman filter's residual stream subtracts the regime terms at each row's own
training state (filtered for `regF` and `eng`, smoothed for `regS`).
"""

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Final

import e00294_common as common
import e00294_inputs as inputs
import numpy as np
import polars as pl
import torch
from e00294_detcommon import DATA_DIR as STATE_DIR
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

HALF_HOURS_PER_DAY: Final[int] = 48
EXTRA_COLUMNS: Final[tuple[str, ...]] = (
    "break",
    "engine_filtered",
    "engine_smoothed",
    "units_filtered",
    "persist_matched",
)
FAMILY_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "a0": (),
    "ind": ("break",),
    "regF": ("break", "engine_filtered"),
    "regS": ("break", "engine_smoothed"),
    "eng": ("engine_filtered",),
    "regU": ("break", "units_filtered"),
    "regP": ("break", "persist_matched"),
}
FORECAST_SOURCE: Final[dict[str, int]] = {
    "engine_filtered": 1,
    "units_filtered": 3,
    "persist_matched": 1,
}
"""The `EXTRA_COLUMNS` position that the `persist` mode reads at the source row, for each engine column."""
"""The regime columns each family fits."""
KINDS: Final[dict[str, tuple[str, str | None]]] = {
    "a0": ("a0", None),
    "ind": ("ind", None),
    "regF_persist": ("regF", "persist"),
    "regS_true": ("regS", "true"),
    "eng_persist": ("eng", "persist"),
    "regU_persist": ("regU", "persist"),
    "regP_persist": ("regP", "persist"),
}
"""Each kind's family and the engine state it reads at forecast time."""
CONFIGS: Final[dict[str, dict]] = {
    "whole": {
        "masked": False,
        "window": TrainingWindow(),
        "kinds": ("a0", "ind", "regF_persist", "regS_true", "regU_persist", "regP_persist"),
    },
    "post": {
        "masked": True,
        "window": TrainingWindow(months=common.WINDOW_MONTHS),
        "kinds": ("a0", "eng_persist"),
    },
}
"""`whole`: all of the site's rows, no mask. `post`: experiment 191's reference rows (D8 from 2025-04-01, 24-month window)."""
FAMILIES: Final[tuple[str, ...]] = tuple(FAMILY_COLUMNS)


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


# --- The regime columns on the grid ----------------------------------------------------------


@dataclass(frozen=True)
class RegimeGrid:
    """The regime columns on a site's grid, shape (T, 3) in `EXTRA_COLUMNS` order.

    Attributes:
        values: The `break` indicator and the filtered and smoothed engine probabilities. The
            engine probabilities are zero before the break and where the detector has no value.
        n_outside: The number of grid rows after the break that the detector grid does not cover.
    """

    values: np.ndarray
    n_outside: int


def build_regime_grid(*, cache: FeatureCache, site: str) -> RegimeGrid:
    """Return the regime columns on the cache's grid, from the stored detector states."""
    grid_start = cache.end_times - np.timedelta64(30, "m")
    origin = np.datetime64(common.STATE_GRID_START.replace(tzinfo=None), "us")
    position = ((grid_start - origin) / np.timedelta64(30, "m")).astype(np.int64)
    after = (cache.end_times > naive_utc(moment=common.BREAK)).astype(np.float64)
    out = np.zeros((len(grid_start), len(EXTRA_COLUMNS)))
    out[:, 0] = after
    n_outside = 0
    for column, kind in ((1, "filtered"), (2, "smoothed"), (3, "units_filtered")):
        states = np.nan_to_num(np.load(STATE_DIR / f"p_{kind}_{site}.npy").astype(np.float64))
        inside = (position >= 0) & (position < len(states))
        n_outside = int(((after > 0) & ~inside).sum())
        out[:, column] = (
            np.where(inside, states[np.clip(position, 0, len(states) - 1)], 0.0) * after
        )
    out[:, 4] = matched_persistence(cache=cache, filtered=out[:, 1])
    return RegimeGrid(values=out, n_outside=n_outside)


def matched_persistence(*, cache: FeatureCache, filtered: np.ndarray) -> np.ndarray:
    """Return the filtered probability as a forecast would have seen it, lagged 1 to 7 whole days.

    Row `i` takes the probability `48 * k` half-hours earlier, with `k` cycling through 1 to 7 by
    day, so that a fit trained on this column learns how much the lagged state is worth. The lags
    mimic the `persist` mode, whose lag is `ceil(lead / 24 h)` days for leads of 0.5 to 168 hours.
    """
    day = np.floor((cache.end_times - cache.end_times[0]) / np.timedelta64(24, "h")).astype(
        np.int64
    )
    lag = HALF_HOURS_PER_DAY * (1 + day % 7)
    source = np.arange(len(filtered)) - lag
    return np.where(source >= 0, filtered[np.clip(source, 0, None)], 0.0)


def column_indices(*, family: str) -> list[int]:
    """Return the positions in `EXTRA_COLUMNS` that a family fits."""
    return [EXTRA_COLUMNS.index(name) for name in FAMILY_COLUMNS[family]]


# --- Fits ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyFit:
    """A fit of one family: base coefficients and the regime coefficients (in family order)."""

    fit: GamFit
    theta_extra: np.ndarray


@dataclass(frozen=True)
class CutFits:
    """The fits of one monthly cut."""

    cut: datetime
    families: dict[str, FamilyFit]
    n_rows: int


def _solve(
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
    p: int,
    n_extra: int,
    penalty_base: np.ndarray,
    cost_base: np.ndarray,
    constrained_base: torch.Tensor,
    free_extra: list[bool],
) -> np.ndarray:
    """Solve the bounded penalised least squares for the base columns and `n_extra` regime columns."""
    size = p + n_extra
    penalty = np.zeros((size, size))
    penalty[:p, :p] = penalty_base
    penalty[p:, p:] = common.RIDGE * np.eye(n_extra)
    cost = np.zeros(size)
    cost[:p] = cost_base
    extra_mask = torch.as_tensor(np.array([not free for free in free_extra], dtype=bool))
    constrained = torch.cat([constrained_base, extra_mask])
    theta, _ = solve_bounded_quadratic(
        matrix=torch.as_tensor(gram + penalty)[None],
        rhs=torch.as_tensor(rhs - 0.5 * cost)[None],
        constrained=constrained,
    )
    return theta[0].numpy()


def _cut_fits(
    *,
    cache: FeatureCache,
    site: str,
    cut: datetime,
    grid: RegimeGrid,
    window: TrainingWindow,
    families: tuple[str, ...],
) -> CutFits:
    """Fit every family at one cut from one pass over the training rows."""
    spec = gam_spec()
    features = cache.sites[site]
    layout, index = prepare_fit(
        cache=cache, site=site, spec=spec, cut=cut, window=window, season_years=common.SEASON_YEARS
    )
    p = layout.n_columns
    q = p + len(EXTRA_COLUMNS)
    gram = np.zeros((q, q))
    rhs = np.zeros(q)
    for chunk in row_chunks(index=index):
        rows = index[chunk]
        x = np.hstack(
            [
                design_matrix(cache=cache, features=features, layout=layout, index=rows),
                -grid.values[rows],
            ]
        )
        gram += x.T @ x
        rhs += x.T @ features.y[rows]
    penalty_base, cost_base = penalty_matrix(layout=layout, n_rows=len(index))
    constrained_base = torch.as_tensor(layout.constrained)
    fits: dict[str, FamilyFit] = {}
    for family in families:
        extra = column_indices(family=family)
        pick = [*range(p), *[p + e for e in extra]]
        theta = _solve(
            gram=gram[np.ix_(pick, pick)],
            rhs=rhs[pick],
            p=p,
            n_extra=len(extra),
            penalty_base=penalty_base,
            cost_base=cost_base,
            constrained_base=constrained_base,
            free_extra=[EXTRA_COLUMNS[e] == "break" for e in extra],
        )
        fits[family] = FamilyFit(
            fit=GamFit(
                layout=layout,
                theta=theta[None, :p].copy(),
                cut=cut,
                n_rows=len(index),
                site=site,
                last_season_year=last_signal_season_year(
                    cache=cache, site=site, index=index, cut=cut, pinned=None
                ),
                pinned=None,
            ),
            theta_extra=theta[p:].copy(),
        )
    return CutFits(cut=cut, families=fits, n_rows=len(index))


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


def fit_all_cuts(*, cache: FeatureCache, site: str, grid: RegimeGrid, config: str) -> list[CutFits]:
    """Fit every monthly cut from the site's first, caching the result on disk."""
    path = common.FITS_DIR / config / f"{site}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    settings = CONFIGS[config]
    families = tuple(dict.fromkeys(KINDS[kind][0] for kind in settings["kinds"]))
    cuts = common.monthly_cuts()
    fits = [
        _cut_fits(
            cache=cache, site=site, cut=cut, grid=grid, window=settings["window"], families=families
        )
        for cut in cuts[first_cut_index(cache=cache, site=site, cuts=cuts) :]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(fits))
    temporary.rename(path)
    return fits


# --- Forecasting --------------------------------------------------------------------------------


def state_column(*, family: str) -> int | None:
    """Return the position in `EXTRA_COLUMNS` of a family's engine column, or `None`."""
    for name in FAMILY_COLUMNS[family]:
        if name.startswith(("engine", "units", "persist")):
            return EXTRA_COLUMNS.index(name)
    return None


def regime_values(
    *,
    family: str,
    mode: str | None,
    cache: FeatureCache,
    grid: RegimeGrid,
    rows: np.ndarray,
    origin: datetime | None,
) -> np.ndarray:
    """Return the family's regime columns at grid rows, shape (n, n_columns), in family order.

    `mode` is the engine state read at forecast time: `persist` takes the filtered probability at
    the same half-hour of the last day settled at `origin`, `true` takes the column at the row
    itself, and `None` (the training state) also takes the row itself.
    """
    columns = column_indices(family=family)
    values = grid.values[rows][:, columns].copy()
    engine = state_column(family=family)
    if mode == "persist" and engine is not None:
        assert origin is not None
        origin64 = naive_utc(moment=origin)
        ahead_days = np.ceil((cache.end_times[rows] - origin64) / np.timedelta64(24, "h")).astype(
            np.int64
        )
        source = rows - HALF_HOURS_PER_DAY * np.maximum(ahead_days, 1)
        if (source < 0).any() or (cache.end_times[source] > origin64).any():
            msg = "a persisted state row ends after the origin"
            raise ValueError(msg)
        position = columns.index(engine)
        values[:, position] = grid.values[source, FORECAST_SOURCE[EXTRA_COLUMNS[engine]]]
    return values


def regime_output(*, family_fit: FamilyFit, values: np.ndarray) -> np.ndarray:
    """Return the power the regime terms take off net load, in points (can be negative for `break`)."""
    if values.shape[1] == 0:
        return np.zeros(len(values))
    return values @ family_fit.theta_extra


def residual_stream(
    *, cache: FeatureCache, fits: list[CutFits], family: str, grid: RegimeGrid, site: str
) -> np.ndarray:
    """Return the reading minus the chained fits' prediction on a cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut. The regime terms are subtracted
    at each row's own state.
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
        family_fit = cut_fits.families[family]
        prediction = predict_gam(cache=cache, fit=family_fit.fit, index=rows)
        values = regime_values(
            family=family, mode=None, cache=cache, grid=grid, rows=rows, origin=None
        )
        stream[rows] = features.y[rows] - (
            prediction - regime_output(family_fit=family_fit, values=values)
        )
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
    grid: RegimeGrid,
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
        family_fit = cut_fits.families[family]
        assert_cut_not_after(fit=family_fit.fit, origin=origin)
        if family not in base:
            base[family] = predict_gam(cache=driven, fit=family_fit.fit, index=np.arange(n_targets))
        values = regime_values(
            family=family, mode=mode, cache=cache, grid=grid, rows=target, origin=origin
        )
        reading = base[family] - regime_output(family_fit=family_fit, values=values)
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
    grid: RegimeGrid,
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
                        "p_engine_filtered": grid.values[rows, 1][keep],
                        "p_engine_smoothed": grid.values[rows, 2][keep],
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
        `seed`, `arm`, `q01` to `q99`, `actual`, `mean`, and the engine states.
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
            scored["p_engine_filtered"].cast(pl.Float32),
            scored["p_engine_smoothed"].cast(pl.Float32),
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
    grid: RegimeGrid
    first_origin: datetime
    kinds: tuple[str, ...]

    def usable(self, index: int) -> bool:
        """Return True when a run's origin is at or after the first fitted cut plus the delay."""
        return self.setup.origins[index] >= self.first_origin

    def raw(self, *, periods: set[str]) -> pl.DataFrame | None:
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
            kinds=self.kinds,
        )


def prepare_site(*, site: str, cache: FeatureCache, config: str) -> Prepared:
    """Fit the monthly models and run the filters for one site."""
    settings = CONFIGS[config]
    work = masked_cache(cache=cache, site=site) if settings["masked"] else cache
    setup = inputs.calibrated_setup(cache=work, site=site)
    grid = (
        build_regime_grid(cache=work, site=site)
        if (STATE_DIR / f"p_filtered_{site}.npy").exists()
        else RegimeGrid(values=np.zeros((len(work.end_times), len(EXTRA_COLUMNS))), n_outside=0)
    )
    fits = fit_all_cuts(cache=work, site=site, grid=grid, config=config)
    proxy = proxy_cache(cache=work, context=setup.context)
    hyper = filter_hyper()
    kinds = tuple(settings["kinds"])
    families = tuple(dict.fromkeys(KINDS[kind][0] for kind in kinds))
    traces = {
        family: run_kalman(
            residual=residual_stream(cache=proxy, fits=fits, family=family, grid=grid, site=site),
            band=work.calendar.band,
            hyper=hyper,
        )
        for family in families
    }
    first_origin = fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    return Prepared(
        work=work,
        setup=setup,
        fits=fits,
        traces=traces,
        grid=grid,
        first_origin=first_origin,
        kinds=kinds,
    )


def run_site(*, site: str, cache: FeatureCache, config: str) -> dict[str, int]:
    """Fit, filter and forecast one site, and write each period's frame."""
    prepared = prepare_site(site=site, cache=cache, config=config)
    kinds = prepared.kinds
    tuning = prepared.raw(periods={"tuning"})
    spread_kind = {kind: kind for kind in kinds}
    if tuning is None:
        donor_frames = []
        for donor in common.DONORS:
            donor_prepared = prepare_site(site=donor, cache=cache, config="post")
            donor_prepared.kinds = ("a0",)
            frame = donor_prepared.raw(periods={"tuning"})
            if frame is not None:
                donor_frames.append(frame)
        tuning = pl.concat(donor_frames)
        spread_kind = dict.fromkeys(kinds, "a0")
    site_id = prepared.work.sites[site].table.time_series_id
    written: dict[str, int] = {}
    for period in ("replicate", "scored"):
        base = prepared.raw(periods={period})
        if base is None:
            continue
        frame = arm_frames(base=base, tuning=tuning, site_id=site_id, spread_kind=spread_kind)
        target = common.PREDS_DIR / config / f"{period}_{site}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        written[period] = frame.height
    return written
