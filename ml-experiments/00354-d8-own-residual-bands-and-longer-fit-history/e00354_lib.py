"""Band rules, slice flags, and paired scoring shared by experiment 354's scripts.

Every band rule takes a forecast frame (one site, rows sorted by origin then valid time, outside
the hand-filtered events, with a reading) and returns the same rows with `q01` to `q99` rebuilt
from the frame's own median and the as-of residuals of the same frame. Experiment 264's resolution
and refit-block code does the bookkeeping: a residual is `actual - q50` at the shortest lead of
each valid time, a reading is known six hours after its valid time, a fit uses the trailing 365
days, and fits are refreshed every 7 origins. A block with under 28 days of resolved rows keeps the
frame's own (donated) quantiles.
"""

import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from typing import Final

import e00252_score as score_module  # ty: ignore[unresolved-import]
import e00264_asof as asof  # ty: ignore[unresolved-import]
import e00264_common as c264  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_calendar import HOLIDAY_PERIODS

QUANTILES: Final[tuple[str, ...]] = score_module.QUANTILES
KEY: Final[list[str]] = score_module.KEY
SHRINKAGE_ROWS: Final[float] = 2000.0
"""Experiment 265's weight constant: the own table has weight n / (n + K)."""
HORIZON_HOURS: Final[float] = 168.0
FIRST_THREE_MONTHS: Final[tuple[str, ...]] = ("2025-07", "2025-08", "2025-09")


def clean(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the scored rows outside the hand-filtered events, with a reading, up to 168 hours."""
    return (
        frame.filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") > 0.0)
            & (pl.col("lead_hours") <= HORIZON_HOURS)
        )
        .sort("power_fcst_init_time", "valid_time")
        .with_columns(pl.col("run").cast(pl.Int64))
    )


def _replace_quantiles(*, frame: pl.DataFrame, values: np.ndarray) -> pl.DataFrame:
    return frame.with_columns(
        [pl.Series(name, values[:, i].astype(np.float32)) for i, name in enumerate(QUANTILES)]
    )


def own_residual_band(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Arm 2: the median plus the as-of empirical quantiles of the frame's own residuals."""
    return asof.empirical_residual_quantiles(reference=frame)


def cell_residual_band(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Arm 3: arm 2 by cell of UTC six-hour block and season.

    A cell with under 300 resolved rows uses its hour block alone, and an hour block with under 300
    uses every row (experiment 264's cell rule).
    """
    resolved = asof.resolved_from_reference(reference=frame)
    quantiles = frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
    origin_ns = frame["power_fcst_init_time"].dt.epoch("ns").to_numpy()
    valid_ns = frame["valid_time"].dt.epoch("ns").to_numpy()
    hours, seasons = asof.hour_block(valid_ns=valid_ns), asof.season(valid_ns=valid_ns)
    median = quantiles[:, QUANTILES.index("q50")]
    out = quantiles.copy()
    for fit_ns, origins in asof.blocks(origins_ns=np.unique(origin_ns)):
        rows = asof.fit_rows(resolved=resolved, fit_ns=fit_ns)
        if len(rows) < c264.MIN_FIT_ROWS:
            continue
        table = asof._cell_quantiles(
            ratio=resolved.residual[rows],
            hours=asof.hour_block(valid_ns=resolved.valid_ns[rows]),
            seasons=asof.season(valid_ns=resolved.valid_ns[rows]),
            probabilities=c264.LEVELS,
        )
        applies = np.isin(origin_ns, origins)
        out[applies] = median[applies, None] + table[hours[applies], seasons[applies]]
    return _replace_quantiles(frame=frame, values=out)


def shrunk_band(*, frame: pl.DataFrame, prior: pl.DataFrame) -> pl.DataFrame:
    """Arm 4: arm 2's residual quantiles shrunk towards the prior forecast's as-of quantiles.

    The weight on the frame's own table is `n / (n + 2,000)`, with `n` its resolved rows. The prior
    is BG2's own forecast at the same site, with the same latency, window, and refit blocks.
    """
    own = asof.resolved_from_reference(reference=frame)
    other = asof.resolved_from_reference(reference=prior)
    quantiles = frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
    origin_ns = frame["power_fcst_init_time"].dt.epoch("ns").to_numpy()
    median = quantiles[:, QUANTILES.index("q50")]
    out = quantiles.copy()
    for fit_ns, origins in asof.blocks(origins_ns=np.unique(origin_ns)):
        rows = asof.fit_rows(resolved=own, fit_ns=fit_ns)
        if len(rows) < c264.MIN_FIT_ROWS:
            continue
        prior_rows = asof.fit_rows(resolved=other, fit_ns=fit_ns)
        weight = len(rows) / (len(rows) + SHRINKAGE_ROWS)
        shifts = np.quantile(own.residual[rows], c264.LEVELS)
        if len(prior_rows):
            shifts = weight * shifts + (1 - weight) * np.quantile(
                other.residual[prior_rows], c264.LEVELS
            )
        applies = np.isin(origin_ns, origins)
        out[applies] = median[applies, None] + shifts[None, :]
    return _replace_quantiles(frame=frame, values=out)


def bg2_prior_frame(*, frame: pl.DataFrame, site: str) -> pl.DataFrame:
    """Return BG2's quantiles on the frame's rows, with the frame's actual and lead."""
    reference = score_module.reference_frame(name="bg2", site=site)
    renamed = reference.rename({f"bg2_{q}": q for q in QUANTILES})
    return frame.select(*KEY, "run", "lead_hours", "actual").join(
        renamed, on=KEY, how="inner", validate="1:1"
    )


def with_flags(*, frame: pl.DataFrame, cache, site: str) -> pl.DataFrame:  # noqa: ANN001
    """Add the month, the quarter of the scored year, and the slice flags."""
    features = cache.sites[site]
    index = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    unit = features.pv_unit_shape[index]
    daylight = unit > score_module.SUN_UP
    edge = float(np.quantile(unit[daylight], 2 / 3))
    local_month = cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % 12 + 1
    holiday = cache.calendar.holiday_period[index].astype(np.int64)
    night = features.elevation_deg[index] < -5
    christmas = [HOLIDAY_PERIODS.index(n) for n in ("Christmas Day", "Boxing Day")]
    month = frame["valid_time"].dt.convert_time_zone("Europe/London").dt.strftime("%Y-%m")
    return frame.with_columns(
        month=month,
        all=pl.lit(True),
        sunny=pl.Series(unit >= edge),
        daylight=pl.Series(daylight),
        winter_night=pl.Series(np.isin(local_month, [11, 12, 1, 2, 3]) & night),
        cold_night=pl.Series(night & (features.temperature_c[index] < 2.0)),
        christmas=pl.Series(np.isin(holiday, christmas)),
        first3=month.is_in(list(FIRST_THREE_MONTHS)),
        y2026=month.str.starts_with("2026"),
    )


def row_scores(*, actual: np.ndarray, frame: pl.DataFrame, prefix: str = "") -> np.ndarray:
    """Return the per-row quantile score of a frame's (optionally prefixed) quantile columns."""
    names = [f"{prefix}{q}" for q in QUANTILES]
    return score_module.score(
        actual=actual, quantiles=frame.select(names).to_numpy().astype(np.float64)
    )


def contrast_both(*, wide: pl.DataFrame, column: str, flag: str, site: str) -> dict:
    """Return a paired month-resampled contrast at the 95% and the 99% (Bonferroni) level."""
    out = {}
    for label, interval in (("ci95", (2.5, 97.5)), ("ci99", (0.5, 99.5))):
        score_module.INTERVAL = interval
        point, low, high = score_module.contrast(frames={site: wide}, column=column, flag=flag)[
            "pooled"
        ]
        out[label] = [point, low, high]
    score_module.INTERVAL = (2.5, 97.5)
    return out


def coverage_and_tails(*, frame: pl.DataFrame, prefix: str = "") -> dict:
    """Return the shares of readings inside q05 to q95, below q01, below q05, and above q99."""
    actual = frame["actual"].to_numpy()
    q = {name: frame[f"{prefix}{name}"].to_numpy() for name in QUANTILES}
    return {
        "inside_q05_q95": float(((actual >= q["q05"]) & (actual <= q["q95"])).mean()),
        "below_q01": float((actual < q["q01"]).mean()),
        "below_q05": float((actual < q["q05"]).mean()),
        "above_q95": float((actual > q["q95"]).mean()),
        "above_q99": float((actual > q["q99"]).mean()),
    }
