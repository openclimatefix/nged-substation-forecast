"""Corrections, spreads, and forecast frames for the arms of experiment 145.

**Every arm is GAM-K's cached member forecasts plus a correction, then a rebuilt spread.** An arm's
correction is a flat array with one value per row of `e00145_site.SiteData`. The spread is rebuilt
for every arm by experiment 111's routine: the centred post-correction residual quantiles of the
tuning rows, per hour band, quarter, and lead band, scaled by a factor per lead band. The 51
members are pooled by `studies.ens_members.mixture_quantiles`.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00145_common as common
import e00145_site as site_module
import numpy as np
import polars as pl
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.dp_gam import QUANTILE_COLUMNS
from studies.ens_members import mixture_quantiles

FACTOR_GRID: Final[tuple[float, ...]] = (0.8, 0.85, 0.9, 1.0, 1.15, 1.3)
WORK_DIR: Final[Path] = common.WORK_DIR
FRAMES_DIR: Final[Path] = WORK_DIR / "frames"
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)


@dataclass(frozen=True)
class Correction:
    """An arm's correction at every row of a site.

    Attributes:
        level: The persistent level part.
        anomaly: The decaying transient part.
        shape: The shape part (the profile shift of an analog or recipe R arm).
    """

    level: np.ndarray
    anomaly: np.ndarray
    shape: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """The correction to add to the member means."""
        return self.level + self.anomaly + self.shape


def spread_offsets(
    *, spread: LeadHourSpread, band: np.ndarray, quarter: np.ndarray, lead_hours: np.ndarray
) -> np.ndarray:
    """Return the spread's 11 offsets at each row, shape (rows, 11), factors applied."""
    lead = lead_band_of_hours(lead_hours=lead_hours)
    out = np.empty((len(band), len(LEVELS)))
    for index in range(N_LEAD_BANDS):
        chosen = lead == index
        if chosen.any():
            out[chosen] = (
                spread.factors[index]
                * spread.by_lead_band[index].table[band[chosen], quarter[chosen]]
            )
    return out


def quantiles_for_runs(
    *,
    data: site_module.SiteData,
    correction: np.ndarray,
    spread: LeadHourSpread,
    run_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mixture quantiles of chosen runs, and the rows they cover.

    Args:
        data: The site data.
        correction: The arm's total correction at every row.
        spread: The spread, with factors.
        run_indices: Indices into `data.runs`.

    Returns:
        The quantiles, shape (rows covered, 11), and the covered rows' indices.
    """
    pieces, covered = [], []
    offsets = spread_offsets(
        spread=spread, band=data.hour_band, quarter=data.quarter, lead_hours=data.lead_hours
    )
    for run in run_indices:
        rows = np.arange(data.offsets[run], data.offsets[run + 1])
        corrected = data.member_means[run] + correction[rows][None, :]
        per_member = corrected.T[:, :, None] + offsets[rows][:, None, :]
        pieces.append(mixture_quantiles(member_quantiles=per_member))
        covered.append(rows)
    return np.concatenate(pieces), np.concatenate(covered)


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points."""
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(
        np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error), axis=1
    )


def fit_spread(*, data: site_module.SiteData, correction: np.ndarray) -> LeadHourSpread | None:
    """Fit an arm's centred spread on the tuning rows' corrected-mean residuals.

    Args:
        data: The site data.
        correction: The arm's total correction at every row.

    Returns:
        The spread with every factor 1, or `None` if the site has no tuning rows.
    """
    rows = data.period_rows(period="tuning") & data.scorable
    if not rows.any():
        return None
    residual = data.actual[rows] - (data.member_mean + correction)[rows]
    return LeadHourSpread.fit(
        residuals=residual,
        band=data.hour_band[rows],
        quarter=data.quarter[rows],
        lead_hours=data.lead_hours[rows],
    )


def tuning_band_scores(
    *, data: site_module.SiteData, correction: np.ndarray, spread: LeadHourSpread
) -> np.ndarray:
    """Return the sum and count of tuning-row quantile scores per lead band, shape (4, 2)."""
    rows = data.period_rows(period="tuning") & data.scorable
    runs = np.unique(data.run_of_row[rows])
    quantiles, covered = quantiles_for_runs(
        data=data, correction=correction, spread=spread, run_indices=runs
    )
    mask = np.zeros(data.n_rows, dtype=bool)
    mask[covered] = True
    keep = rows[covered]
    scores = quantile_score(actual=data.actual[covered][keep], quantiles=quantiles[keep])
    lead = data.lead_band[covered][keep]
    out = np.zeros((N_LEAD_BANDS, 2))
    for index in range(N_LEAD_BANDS):
        out[index] = (scores[lead == index].sum(), (lead == index).sum())
    return out


def d1_tuning_score(
    *, data: site_module.SiteData, correction: np.ndarray, spread: LeadHourSpread
) -> tuple[float, int]:
    """Return the sum and count of tuning-row quantile scores at D+1."""
    rows = data.period_rows(period="tuning") & data.scorable & data.d1_rows()
    runs = np.unique(data.run_of_row[rows])
    quantiles, covered = quantiles_for_runs(
        data=data, correction=correction, spread=spread, run_indices=runs
    )
    keep = rows[covered]
    scores = quantile_score(actual=data.actual[covered][keep], quantiles=quantiles[keep])
    return float(scores.sum()), int(keep.sum())


def correction_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's correction at a site."""
    return WORK_DIR / "corrections" / f"{arm}_{site}.npz"


def spread_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's fitted spread at a site."""
    return WORK_DIR / "spread" / f"{arm}_{site}.npz"


def write_npz_atomic(*, path: Path, **arrays: np.ndarray) -> None:
    """Write arrays through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)  # ty: ignore[invalid-argument-type]
    temporary.rename(path)


def save_correction(*, arm: str, site: str, correction: Correction) -> None:
    """Store an arm's correction."""
    write_npz_atomic(
        path=correction_path(arm=arm, site=site),
        level=correction.level,
        anomaly=correction.anomaly,
        shape=correction.shape,
    )


def load_correction(*, arm: str, site: str) -> Correction:
    """Return a stored correction."""
    with np.load(correction_path(arm=arm, site=site)) as stored:
        return Correction(level=stored["level"], anomaly=stored["anomaly"], shape=stored["shape"])


def save_spread(*, arm: str, site: str, spread: LeadHourSpread) -> None:
    """Store a spread's tables and factors."""
    write_npz_atomic(
        path=spread_path(arm=arm, site=site),
        tables=np.stack([band.table for band in spread.by_lead_band]),
        factors=spread.factors,
    )


def load_spread(*, arm: str, site: str) -> LeadHourSpread:
    """Return a stored spread."""
    from studies.dp_gam import ResidualSpread

    with np.load(spread_path(arm=arm, site=site)) as stored:
        tables, factors = stored["tables"], stored["factors"]
    return LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=table) for table in tables), factors=factors
    )


def frame_for_arm(
    *,
    data: site_module.SiteData,
    arm: str,
    correction: Correction,
    spread: LeadHourSpread,
    run_indices: np.ndarray | None = None,
) -> pl.DataFrame:
    """Return an arm's forecast rows with a reading, in the stored column layout.

    Args:
        data: The site data.
        arm: The arm's name.
        correction: The arm's correction.
        spread: The arm's spread with factors.
        run_indices: The runs to include; every cached run if `None`.

    Returns:
        Columns `time_series_id`, `run`, `power_fcst_init_time`, `valid_time`, `lead_hours`,
        `seed`, `arm`, `q01` to `q99`, `actual`, `mean`, `level`, `anomaly`, `shape`, `hour_band`,
        `quarter`, `switching_mask`, and `period`, in points.
    """
    runs = np.arange(len(data.runs)) if run_indices is None else run_indices
    quantiles, rows = quantiles_for_runs(
        data=data, correction=correction.total, spread=spread, run_indices=runs
    )
    keep = np.isfinite(data.actual[rows])
    rows, quantiles = rows[keep], quantiles[keep]
    n = len(rows)
    valid = data.cache.end_times[data.position[rows]]
    origin = data.run_origins[data.run_of_row[rows]]
    table = data.features.table
    columns: dict[str, object] = {
        "time_series_id": np.full(n, table.time_series_id, dtype=np.int32),
        "run": data.runs[data.run_of_row[rows]].astype(np.int32),
        "power_fcst_init_time": pl.Series(origin).dt.replace_time_zone("UTC"),
        "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
        "lead_hours": data.lead_hours[rows],
        "seed": np.zeros(n, dtype=np.int8),
        "arm": np.full(n, arm),
        **{
            column: quantiles[:, position].astype(np.float32)
            for position, column in enumerate(QUANTILE_COLUMNS)
        },
        "actual": data.actual[rows].astype(np.float32),
        "mean": (data.member_mean + correction.total)[rows].astype(np.float32),
        "level": correction.level[rows].astype(np.float32),
        "anomaly": correction.anomaly[rows].astype(np.float32),
        "shape": correction.shape[rows].astype(np.float32),
        "hour_band": data.hour_band[rows].astype(np.int8),
        "quarter": data.quarter[rows].astype(np.int8),
        "switching_mask": data.features.switching_mask[data.position[rows]],
        "period": np.full(n, ""),
    }
    frame = pl.DataFrame(columns)
    names = {int(i): str(p) for i, p in zip(data.runs, data.run_period, strict=True)}
    periods = np.array([names[int(r)] for r in frame["run"].to_numpy()])
    return frame.with_columns(period=pl.Series(periods))


def replace_period_masks(*, frame: pl.DataFrame, data: site_module.SiteData) -> pl.DataFrame:
    """Add `row_period`: the period whose scoring rule the row meets, or an empty string.

    A tuning or replicate row has its valid time in the period's months too (see
    `SiteData.period_rows`).
    """
    codes = data.month_code[
        np.searchsorted(
            data.cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
    ]
    tuning = np.isin(codes, [100 * y + m for y, m in common.TUNING_MONTHS])
    replicate = np.isin(codes, [100 * y + m for y, m in common.REPLICATE_MONTHS])
    period = frame["period"].to_numpy()
    row_period = np.where(
        period == "scored",
        "scored",
        np.where(
            (period == "tuning") & tuning,
            "tuning",
            np.where((period == "replicate") & replicate, "replicate", ""),
        ),
    )
    return frame.with_columns(row_period=pl.Series(row_period))


def frame_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's frame at a site."""
    return FRAMES_DIR / f"{arm}_{site}.parquet"
