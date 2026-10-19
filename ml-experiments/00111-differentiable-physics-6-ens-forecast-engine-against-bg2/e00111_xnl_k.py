"""X-nl plus the Kalman filter: the lagless XGBoost arm given the DP's recent-level correction.

So that a filter gain is not credited to the DP, the same Kalman filter that corrects `gamk` is run
on X-nl's residuals. The residual stream holds, for each half-hour in the 24 hours after a run's
origin, the observed power minus the median (q50) of that run's X-nl mixture, so a residual is the
error of a forecast made at a NWP lead of 9.5 to 33 hours, as the analysis proxy's blocks are. The
filter's hyperparameters are fitted on the tuning months' stream by marginal likelihood, with
Part 4's fitted prior. The filter's correction at a row is its state after the newest reading
that ended 3.5 hours before the run's origin, added to all 11 quantiles of X-nl.

The scored year's X-nl rows are SIMP1's stored monthly no-lag refits (a reuse; one refit reproduces
within 0.003 points of quantile score), and the tuning and replicate rows are this experiment's own
fits.

    uv run python ml-experiments/00111-*/e00111_xnl_k.py
"""

from typing import Final

import e00111_arms as arms
import e00111_common as common
import e00111_score as score
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    correction_at,
    fit_site_hyperparameters,
    run_kalman,
    usable_cutoff_positions,
)

BLOCK_HALF_HOURS: Final[int] = 48
ARM: Final[str] = "xnl_k"


def raw_xnl(*, site: str) -> pl.DataFrame:
    """Return every run's X-nl rows at a site (no scorability filters), in points."""
    own = score.load_xgb_raw(arm="xnl", sites=(site,))
    simp1 = score.load_simp1_raw(sites=(site,))
    return pl.concat([own, simp1]).unique(
        subset=["time_series_id", "power_fcst_init_time", "valid_time"], keep="first"
    )


def residual_stream(*, frame: pl.DataFrame, n_rows: int, end_times: np.ndarray) -> np.ndarray:
    """Return observed minus the X-nl median for each run's first 24 hours, on the grid."""
    stream = np.full(n_rows, np.nan)
    block = frame.filter(
        (pl.col("valid_time") > pl.col("power_fcst_init_time"))
        & (pl.col("valid_time") <= pl.col("power_fcst_init_time") + pl.duration(hours=24))
    )
    position = np.searchsorted(end_times, block["valid_time"].dt.replace_time_zone(None).to_numpy())
    stream[position] = (block["actual"] - block["q50"]).to_numpy()
    return stream


def correct_site(*, site: str) -> pl.DataFrame:
    """Return X-nl plus the Kalman correction at a site, for every run, in the common shape."""
    cache = common.load_cache()
    frame = raw_xnl(site=site)
    stream = residual_stream(frame=frame, n_rows=len(cache.end_times), end_times=cache.end_times)
    mask = arms.in_months(cache=cache, months=common.TUNING_MONTHS)
    prior_mean, prior_spread = arms.part_4_prior()
    start = arms.part_4_hyper(site=site)
    hyper = fit_site_hyperparameters(
        residual=np.where(mask, stream, np.nan),
        band=cache.calendar.band,
        prior_mean=prior_mean,
        prior_spread=prior_spread,
        start=prior_mean if start is None else start,
    )
    trace = run_kalman(residual=stream, band=cache.calendar.band, hyper=hyper)
    target = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    origins = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times, initialisation=origins, delay_hours=DEFAULT_DELAY_HOURS
    )
    correction = correction_at(
        trace=trace, cutoff=cutoff, target=target, band=cache.calendar.band[target].astype(np.int64)
    )
    shifted = frame.with_columns(
        [pl.col(q) + pl.Series(correction.total) for q in score.LEVEL_COLUMNS]
    )
    common.write_json_atomic(
        path=common.WORK_DIR / "xnl_k_hyper" / f"{site}.json",
        content={"hyper": hyper.as_array().tolist()},
    )
    return shifted
