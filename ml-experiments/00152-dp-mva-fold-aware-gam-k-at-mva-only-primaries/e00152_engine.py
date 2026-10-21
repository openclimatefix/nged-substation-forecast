"""The ensemble-driven GAM-K forecast of one site with the three arms of experiment 152.

- **N** reads the reading `|S|` as signed power: a GAM-K fit (`studies.dp_gam.fit_gam`, capacity
  fitted) with the additive filter and symmetric quantiles.
- **A** fits the convex model through the meter (`studies.dp_meter.fit_gam_through_meter`, `Q = 0`,
  experiment 110's L0), corrects the reading `|S|` with the additive filter, and builds symmetric
  quantiles.
- **F** shares A's fits and corrected reading. It takes the sign of the fitted flow and builds its
  quantiles by folding: a draw of the signed flow plus a signed residual from the tuning months,
  taken to its absolute value. (Stop rule S0 of the plan dropped the latent filter.)
- **S**, a ceiling used only in the known-answer test, is N's fit on the unfolded signed series.

All arms are driven by the calibrated ensemble mean (experiment 136's A2 pattern without T-all),
refit monthly on a 24-month window, filtered by experiment 133's shared cell (POOL) on the
residuals against the analysis proxy, with the spread fitted on the tuning months.
"""

import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Final

import e00152_common as common
import e00152_ens_inputs as inputs
import numpy as np
import polars as pl
from studies.baselines import haurwitz_w_m2
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    correction_at,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_ens import LeadHourSpread, assert_cut_not_after, latest_cut_index
from studies.dp_ens_weather import (
    RunWindow,
    lead_band_of_hours,
    proxy_cache,
    replace_site,
)
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import (
    N_BANDS,
    QUANTILE_COLUMNS,
    QUANTILE_LEVELS,
    GamFit,
    GamSpec,
    TrainingWindow,
    fit_gam,
    predict_gam,
)
from studies.dp_meter import MeterFit, fit_gam_through_meter, predict_meter
from studies.solar import zenith

PART_4_TUNED: Final[Path] = common.CACHE_DIR.parent / "work" / "tuned.json"
MIN_WINDOW_HALF_HOURS: Final[int] = 48
N_FOLD_SAMPLES: Final[int] = 200
FOLD_SEED: Final[int] = 152
LEVELS: Final[np.ndarray] = np.array(QUANTILE_LEVELS)
FITS_DIR: Final[Path] = common.WORK_DIR / "fits"
PREDS_DIR: Final[Path] = common.WORK_DIR / "preds"


def filter_hyper() -> FilterHyperparameters:
    """Return experiment 133's shared filter cell (POOL)."""
    return FilterHyperparameters.from_array(
        values=np.array(json.loads(common.POOL_PATH.read_text())["hyper"])
    )


def gam_spec() -> GamSpec:
    """Return the GAM-K penalties: experiment 104's ridge, no drift, a fitted capacity."""
    step_penalty = json.loads(PART_4_TUNED.read_text())["step_penalty"]
    return GamSpec(
        ridge=common.RIDGE,
        step_penalty=step_penalty,
        drift="none",
        capacity="fitted",
        final_knot="free",
    )


def masked_cache(*, cache: FeatureCache, site: str, y: np.ndarray | None = None) -> FeatureCache:
    """Return the cache with a site's reading replaced and D8's early rows removed.

    Args:
        cache: The feature cache.
        site: The site's label.
        y: A replacement reading in points (for example a folded series); the cache's if `None`.

    Returns:
        A cache whose site arrays `y`, `present`, and `default_rows` hold no row before D8's first
        day (for D8 only).
    """
    features = cache.sites[site]
    reading = features.y.copy() if y is None else y.copy()
    present = features.present & np.isfinite(reading)
    default_rows = features.default_rows & np.isfinite(reading)
    if site == "D8":
        early = cache.end_times <= naive_utc(moment=common.D8_FIRST_DAY)
        reading[early] = np.nan
        present = present & ~early
        default_rows = default_rows & ~early
    return replace_site(
        cache=cache, site=site, y=reading, present=present, default_rows=default_rows
    )


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


def fit_cuts(
    *, cache: FeatureCache, site: str, kind: str, cuts: list[datetime]
) -> list[GamFit | MeterFit]:
    """Fit one kind (`naive` or `meter`) at every monthly cut from the site's first cut."""
    spec = gam_spec()
    window = TrainingWindow(months=common.WINDOW_MONTHS)
    fits: list[GamFit | MeterFit] = []
    for cut in cuts[first_cut_index(cache=cache, site=site, cuts=cuts) :]:
        if kind == "naive":
            fits.append(
                fit_gam(
                    cache=cache,
                    site=site,
                    specs=[spec],
                    cut=cut,
                    window=window,
                    season_years=common.SEASON_YEARS,
                )
            )
        else:
            fits.append(
                fit_gam_through_meter(
                    cache=cache,
                    site=site,
                    spec=spec,
                    cut=cut,
                    window=window,
                    season_years=common.SEASON_YEARS,
                    qmode="zero",
                )
            )
    return fits


def cut_of(*, fit: GamFit | MeterFit) -> datetime:
    """Return a fit's cut."""
    return fit.cut if isinstance(fit, GamFit) else fit.fit.cut


def predict(
    *, cache: FeatureCache, fit: GamFit | MeterFit, index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the predicted reading and the predicted signed flow, in points.

    A naive fit reads the reading as signed power, so both are its prediction.
    """
    if isinstance(fit, GamFit):
        value = predict_gam(cache=cache, fit=fit, index=index)
        return value, value
    result = predict_meter(cache=cache, meter=fit, index=index)
    return result.apparent_power, result.real_power


def residual_stream(*, cache: FeatureCache, fits: list[GamFit | MeterFit], site: str) -> np.ndarray:
    """Return the reading minus the chained fits' prediction on a cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut, so no residual is in-sample.
    """
    if not fits or any(cut_of(fit=b) <= cut_of(fit=a) for a, b in pairwise(fits)):
        msg = "the residual stream needs fits with strictly ascending cuts"
        raise ValueError(msg)
    features = cache.sites[site]
    stream = np.full(len(cache.end_times), np.nan)
    cuts = [
        int(np.searchsorted(cache.end_times, naive_utc(moment=cut_of(fit=fit)), side="right")) - 1
        for fit in fits
    ]
    ends = [*cuts[1:], len(stream) - 1]
    for fit, cut, end in zip(fits, cuts, ends, strict=True):
        rows = np.arange(cut + 1, end + 1)
        usable = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
            & np.isfinite(features.y[rows])
        )
        rows = rows[usable]
        if len(rows):
            stream[rows] = features.y[rows] - predict(cache=cache, fit=fit, index=rows)[0]
    return stream


def sunny_midday_flags(*, cache: FeatureCache, site: str) -> np.ndarray:
    """Return True on the grid's sunny-midday half-hours (CAMS irradiance, not observed power)."""
    features = cache.sites[site]
    ends = cache.end_times
    starts = ends - np.timedelta64(30, "m")
    month = starts.astype("datetime64[M]").astype(int) % 12 + 1
    hour = (starts - starts.astype("datetime64[D]")) / np.timedelta64(1, "h")
    midday = (
        np.isin(month, common.SUNNY_MONTHS)
        & (hour >= common.MIDDAY_FIRST_HALF_HOUR_UTC)
        & (hour <= common.MIDDAY_LAST_HALF_HOUR_UTC)
    )
    middle = pl.Series(ends - np.timedelta64(15, "m")).dt.replace_time_zone("UTC")
    clear = haurwitz_w_m2(
        apparent_zenith_deg=zenith(
            stamps=middle, latitude=features.table.latitude, longitude=features.table.longitude
        )
    )
    index = np.where(clear > 20.0, features.ghi_w_m2 / np.maximum(clear, 1.0), np.nan)
    return midday & (np.nan_to_num(index, nan=0.0) >= common.CLEAR_SKY_INDEX_MINIMUM)


@dataclass(frozen=True)
class RunForecast:
    """One run's forecasts for the arms that share a pair of fits.

    Attributes:
        window: The slice of the full grid the forecast covers.
        reading: The predicted reading before the correction, in points, by kind.
        flow: The predicted signed flow before the correction, by kind.
        level: The filter's persistent correction by kind.
        anomaly: The filter's decaying correction by kind.
    """

    window: slice
    reading: dict[str, np.ndarray]
    flow: dict[str, np.ndarray]
    level: dict[str, np.ndarray]
    anomaly: dict[str, np.ndarray]


def forecast_run(
    *,
    cache: FeatureCache,
    setup: inputs.SiteSetup,
    run_index: int,
    fits: dict[str, list[GamFit | MeterFit]],
    traces: dict,
) -> RunForecast:
    """Forecast one run from the ensemble mean with each kind of fit and its filter."""
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
    out = RunForecast(window=window.window, reading={}, flow={}, level={}, anomaly={})
    for kind, kind_fits in fits.items():
        chosen = latest_cut_index(cuts=[cut_of(fit=f) for f in kind_fits], moment=origin)
        if chosen < 0:
            msg = f"no {kind} fit has a cut at or before {origin}"
            raise ValueError(msg)
        fit = kind_fits[chosen]
        assert_cut_not_after(
            fit=fit if isinstance(fit, GamFit) else fit.fit,
            origin=origin,  # ty: ignore[invalid-argument-type]
        )
        reading, flow = predict(cache=driven, fit=fit, index=np.arange(n_targets))
        correction = correction_at(
            trace=traces[kind], cutoff=np.full(n_targets, cutoff[0]), target=target, band=band
        )
        out.reading[kind], out.flow[kind] = reading, flow
        out.level[kind], out.anomaly[kind] = correction.level, correction.anomaly
    return out


def folded_quantiles(
    *,
    flow: np.ndarray,
    pools: dict[tuple[int, int, int], np.ndarray],
    lead_band: np.ndarray,
    band: np.ndarray,
    quarter: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return the 11 quantiles of `|flow + r|` with `r` drawn from the row's residual pool.

    Args:
        flow: The corrected signed flow of each row, in points.
        pools: Centred signed residuals keyed by (lead band, hour band, quarter).
        lead_band: Each row's lead band.
        band: Each row's hour band.
        quarter: Each row's quarter.
        seed: The generator's seed.

    Returns:
        Shape (rows, 11), ascending along the second axis.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((len(flow), len(LEVELS)))
    for (lead, hour, q), pool in pools.items():
        rows = np.flatnonzero((lead_band == lead) & (band == hour) & (quarter == q))
        if len(rows):
            draws = rng.choice(pool, size=(len(rows), N_FOLD_SAMPLES))
            out[rows] = np.quantile(np.abs(flow[rows, None] + draws), LEVELS, axis=1).T
    return out


def signed_pools(
    *,
    flow: np.ndarray,
    actual: np.ndarray,
    lead_band: np.ndarray,
    band: np.ndarray,
    quarter: np.ndarray,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Return centred signed residuals `sign(flow) * actual - flow` per (lead, band, quarter) cell.

    A cell with under 30 rows borrows its lead band's rows of the same hour band.
    """
    signed = np.where(flow < 0.0, -1.0, 1.0) * actual - flow
    pools: dict[tuple[int, int, int], np.ndarray] = {}
    for lead in range(4):
        for hour in range(N_BANDS):
            for q in range(4):
                cell = (lead_band == lead) & (band == hour) & (quarter == q)
                if cell.sum() < 30:
                    cell = (lead_band == lead) & (band == hour)
                values = signed[cell]
                pools[(lead, hour, q)] = values - values.mean() if len(values) else np.zeros(1)
    return pools


def frames_for_runs(
    *,
    cache: FeatureCache,
    setup: inputs.SiteSetup,
    run_indices: list[int],
    fits: dict[str, list[GamFit | MeterFit]],
    traces: dict,
    sunny: np.ndarray,
) -> pl.DataFrame:
    """Forecast the runs and return one row per scored half-hour and kind, before any spread.

    Columns: `run`, `power_fcst_init_time`, `valid_time`, `lead_hours`, `kind`, `actual`,
    `reading`, `flow`, `level`, `anomaly`, `hour_band`, `quarter`, `switching_mask`,
    `sunny_midday`.
    """
    features = cache.sites[setup.site]
    frames = []
    for index in run_indices:
        run = setup.runs[index]
        if len(cache.end_times) - run.origin_position - 1 < MIN_WINDOW_HALF_HOURS:
            continue
        forecast = forecast_run(cache=cache, setup=setup, run_index=index, fits=fits, traces=traces)
        rows = np.arange(forecast.window.start, forecast.window.stop)
        keep = features.present[rows] & np.isfinite(features.y[rows])
        if not keep.any():
            continue
        origin64 = np.datetime64(setup.origins[index].replace(tzinfo=None), "us")
        valid = cache.end_times[rows][keep]
        for kind in fits:
            n = int(keep.sum())
            frames.append(
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
                        "flow": forecast.flow[kind][keep],
                        "level": forecast.level[kind][keep],
                        "anomaly": forecast.anomaly[kind][keep],
                        "hour_band": cache.calendar.band[rows][keep].astype(np.int8),
                        "quarter": cache.calendar.quarter[rows][keep].astype(np.int8),
                        "switching_mask": features.switching_mask[rows][keep],
                        "sunny_midday": sunny[rows][keep],
                    }
                )
            )
    return pl.concat(frames)


def build_arm_frames(
    *, base: pl.DataFrame, tuning: pl.DataFrame, arms: dict[str, str], site_id: int
) -> pl.DataFrame:
    """Turn raw forecasts into the stored prediction frames of the named arms.

    Args:
        base: Raw forecasts to score (`frames_for_runs`), all kinds.
        tuning: Raw forecasts of the tuning runs, from which the spread and pools are fitted.
        arms: Arm name to kind (`naive` or `meter`); an arm whose name ends in `_fold` folds.
        site_id: The series identifier.

    Returns:
        Columns `time_series_id`, `run`, `power_fcst_init_time`, `valid_time`, `lead_hours`, `seed`,
        `arm`, `q01` to `q99`, `actual`, `mean`, `flow`, `level`, `anomaly`, `hour_band`,
        `quarter`, `switching_mask`, and `sunny_midday`.
    """
    parts = []
    for arm, kind in arms.items():
        fit_rows = tuning.filter((pl.col("kind") == kind) & ~pl.col("switching_mask"))
        corrected = fit_rows["reading"] + fit_rows["level"] + fit_rows["anomaly"]
        scored = base.filter(pl.col("kind") == kind)
        mean = (scored["reading"] + scored["level"] + scored["anomaly"]).to_numpy()
        lead_band = lead_band_of_hours(lead_hours=scored["lead_hours"].to_numpy())
        if arm.endswith("_fold"):
            sign = np.where(scored["flow"].to_numpy() < 0.0, -1.0, 1.0)
            flow = (
                scored["flow"].to_numpy() + sign * (scored["level"] + scored["anomaly"]).to_numpy()
            )
            tuning_sign = np.where(fit_rows["flow"].to_numpy() < 0.0, -1.0, 1.0)
            tuning_flow = (
                fit_rows["flow"].to_numpy()
                + tuning_sign * (fit_rows["level"] + fit_rows["anomaly"]).to_numpy()
            )
            pools = signed_pools(
                flow=tuning_flow,
                actual=fit_rows["actual"].to_numpy(),
                lead_band=lead_band_of_hours(lead_hours=fit_rows["lead_hours"].to_numpy()),
                band=fit_rows["hour_band"].to_numpy().astype(np.int64),
                quarter=fit_rows["quarter"].to_numpy().astype(np.int64),
            )
            quantiles = folded_quantiles(
                flow=flow,
                pools=pools,
                lead_band=lead_band,
                band=scored["hour_band"].to_numpy().astype(np.int64),
                quarter=scored["quarter"].to_numpy().astype(np.int64),
                seed=FOLD_SEED,
            )
            flow_column = flow
        else:
            spread = LeadHourSpread.fit(
                residuals=(fit_rows["actual"] - corrected).to_numpy().astype(np.float64),
                band=fit_rows["hour_band"].to_numpy().astype(np.int64),
                quarter=fit_rows["quarter"].to_numpy().astype(np.int64),
                lead_hours=fit_rows["lead_hours"].to_numpy(),
            )
            quantiles = spread.quantiles(
                mean=mean,
                band=scored["hour_band"].to_numpy().astype(np.int64),
                quarter=scored["quarter"].to_numpy().astype(np.int64),
                lead_hours=scored["lead_hours"].to_numpy(),
            )
            quantiles = np.maximum(quantiles, 0.0) if kind == "meter" else quantiles
            flow_column = scored["flow"].to_numpy()
        n = scored.height
        quantiles = np.sort(quantiles, axis=1)
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
            pl.Series("flow", flow_column.astype(np.float32)),
            scored["level"].cast(pl.Float32),
            scored["anomaly"].cast(pl.Float32),
            scored["hour_band"],
            scored["quarter"],
            scored["switching_mask"],
            scored["sunny_midday"],
        )
        assert part.height == n
        parts.append(part)
    return pl.concat(parts)


@dataclass
class Prepared:
    """One site's fitted state for forecasting."""

    work: FeatureCache
    setup: inputs.SiteSetup
    fits: dict[str, list[GamFit | MeterFit]]
    traces: dict
    sunny: np.ndarray
    first_origin: datetime

    def usable(self, index: int) -> bool:
        """Return True when a run's origin is at or after the first fitted cut plus the delay."""
        return self.setup.origins[index] >= self.first_origin

    def tuning(self) -> pl.DataFrame | None:
        """Return the raw forecasts of the tuning runs, or `None` when no tuning run is usable."""
        indices = [i for i in self.setup.indices(periods={"tuning"}) if self.usable(i)]
        if not indices:
            return None
        return frames_for_runs(
            cache=self.work,
            setup=self.setup,
            run_indices=indices,
            fits=self.fits,
            traces=self.traces,
            sunny=self.sunny,
        )


def prepare_site(
    *, site: str, variant: str, cache: FeatureCache, kinds: tuple[str, ...], y: np.ndarray | None
) -> Prepared:
    """Fit the monthly models and run the filters for one site."""
    work = masked_cache(cache=cache, site=site, y=y)
    setup = inputs.calibrated_setup(cache=work, site=site)
    cuts = common.monthly_cuts()
    fit_dir = FITS_DIR / variant
    fit_dir.mkdir(parents=True, exist_ok=True)
    fits: dict[str, list[GamFit | MeterFit]] = {}
    for kind in kinds:
        path = fit_dir / f"{site}_{kind}.pkl"
        if path.exists():
            fits[kind] = pickle.loads(path.read_bytes())
        else:
            fits[kind] = fit_cuts(cache=work, site=site, kind=kind, cuts=cuts)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(pickle.dumps(fits[kind]))
            temporary.rename(path)
    proxy = proxy_cache(cache=work, context=setup.context)
    hyper = filter_hyper()
    traces = {
        kind: run_kalman(
            residual=residual_stream(cache=proxy, fits=kind_fits, site=site),
            band=work.calendar.band,
            hyper=hyper,
        )
        for kind, kind_fits in fits.items()
    }
    sunny = sunny_midday_flags(cache=work, site=site)
    first_origin = max(cut_of(fit=kind_fits[0]) for kind_fits in fits.values()) + timedelta(
        hours=DEFAULT_DELAY_HOURS
    )

    return Prepared(
        work=work, setup=setup, fits=fits, traces=traces, sunny=sunny, first_origin=first_origin
    )


def run_site(
    *,
    site: str,
    variant: str,
    cache: FeatureCache,
    kinds: tuple[str, ...],
    arms: dict[str, str],
    y: np.ndarray | None = None,
    periods: tuple[str, ...] = ("replicate", "scored"),
    donors: tuple[str, ...] = (),
) -> dict[str, int]:
    """Fit, filter, and forecast one site's variant and write each period's frame.

    Args:
        site: The site's label.
        variant: A name for the stored folder, such as `real` or `fold_l2_q0`.
        cache: The full feature cache.
        kinds: The fit kinds to run, `naive` and/or `meter`.
        arms: Arm name to kind.
        y: A replacement reading in points; the cache's if `None`.
        periods: The periods to forecast.
        donors: Sites whose tuning-month forecasts supply the spread and residual pools when the
            site has no usable tuning run of its own (D8, whose series starts in April 2025).

    Returns:
        The number of rows written per period.
    """
    prepared = prepare_site(site=site, variant=variant, cache=cache, kinds=kinds, y=y)
    tuning = prepared.tuning()
    if tuning is None:
        donor_frames = [
            frame
            for donor in donors
            if (
                frame := prepare_site(
                    site=donor, variant=variant, cache=cache, kinds=kinds, y=None
                ).tuning()
            )
            is not None
        ]
        tuning = pl.concat(donor_frames)
    work, setup, fits = prepared.work, prepared.setup, prepared.fits
    usable, traces, sunny = prepared.usable, prepared.traces, prepared.sunny
    written: dict[str, int] = {}
    site_id = work.sites[site].table.time_series_id
    for period in periods:
        indices = [i for i in setup.indices(periods={period}) if usable(i)]
        if not indices:
            continue
        base = frames_for_runs(
            cache=work, setup=setup, run_indices=indices, fits=fits, traces=traces, sunny=sunny
        )
        frame = build_arm_frames(base=base, tuning=tuning, arms=arms, site_id=site_id)
        target = PREDS_DIR / variant / f"{period}_{site}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        written[period] = frame.height
    return written


Predictor = Callable[..., tuple[np.ndarray, np.ndarray]]
