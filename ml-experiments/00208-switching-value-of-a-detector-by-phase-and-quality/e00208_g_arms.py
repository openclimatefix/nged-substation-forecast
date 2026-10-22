"""The arms of experiment 115: how each builds its correction, spread, and quantiles.

**Every arm is GAM-K's cached member forecasts plus a correction, then a rebuilt spread.** An arm's
correction is a flat array with one value per row of `e00208_g_site.SiteData`. The spread is
rebuilt for every arm by experiment 111's routine: the centred post-correction residual quantiles
of the tuning rows, per hour band, quarter, and lead band, scaled by a factor per lead band. The 51
members are pooled by `studies.ens_members.mixture_quantiles`.

| Family | Correction |
|---|---|
| `K0` | GAM-K's per-band level and transient (the shape state off) |
| `KW` | K0 plus a shape with zero mean inside each hour band |
| `KS` | K0's bands plus a shape with zero mean over the whole day |
| `KS1` | one level and transient plus a shape with zero mean over the whole day |
| `KWnomid` | KW with the midday shape frozen at zero |
| `R` | K0 minus kappa times the 14-day D+1 error profile |
| `pl52` | K0's level and transient plus KW's shape from 52 weeks earlier |

A family with a multiplier `m` scales the level step, transient variance, and shape step against
the observation noise (the lead-band tuning).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import e00208_g_common as common
import e00208_g_shape as shape
import e00208_g_site as site_module
import numpy as np
import polars as pl
from studies.dp_assimilation import usable_cutoff_positions
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.dp_gam import QUANTILE_COLUMNS
from studies.ens_members import mixture_quantiles

FamilyType = Literal["K0", "KW", "KS", "KS1", "KWnomid", "KWfast", "R", "pl52"]
FACTOR_GRID: Final[tuple[float, ...]] = (0.8, 0.85, 0.9, 1.0, 1.15, 1.3)
MULTIPLIER_GRID: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0)
KAPPA_GRID: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 1.0)
FAST_SHAPE_FACTOR: Final[float] = 16.0
"""The post hoc faster shape: `q_s` times this factor.

16 is the smallest factor on the grid 2, 4, 8, 16, 32, 64 at which the tuned filter's analytic
recovery of both injections is at least 0.6 (0.62 and 0.72). It was chosen from that gate quantity
before any score of the arm was read.
"""
PROFILE_DAYS: Final[int] = 14
MIN_PROFILE_DAYS: Final[int] = 3
MIN_DAY_READINGS: Final[int] = 40
"""A day enters the 14-day profile when at least this many of its 48 half-hours have a reading."""
MIDDAY_BAND: Final[int] = 2
WORK_DIR: Final[Path] = common.WORK_DIR
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)


@dataclass(frozen=True)
class ArmSpec:
    """One arm.

    Attributes:
        name: The arm's name.
        family: The correction's family.
        multiplier: The state step multiplier for every lead band, for a candidate arm.
        multipliers: Per lead band multipliers, for a lead-band arm (overrides `multiplier`).
        delay_hours: The telemetry delay.
        day_only: True to read only readings up to local midnight before the origin.
        kappa: The recipe's shrink, for family `R`.
        candidate: True for an arm that only feeds a choice (a multiplier or kappa) and is not
            reported.
    """

    name: str
    family: FamilyType
    multiplier: float = 1.0
    multipliers: tuple[float, ...] | None = None
    delay_hours: float = 3.5
    day_only: bool = False
    kappa: float = 0.0
    candidate: bool = False


def candidate_name(*, family: str, multiplier: float) -> str:
    """Return the name of a multiplier candidate."""
    return f"{family}_m{multiplier:g}"


def arm_table() -> dict[str, ArmSpec]:
    """Return every arm by name, candidates included."""
    arms: dict[str, ArmSpec] = {}

    def add(spec: ArmSpec) -> None:
        arms[spec.name] = spec

    add(ArmSpec("K0", "K0"))
    add(ArmSpec("KW", "KW"))
    add(ArmSpec("KW-day", "KW", day_only=True))
    for family in ("K0", "KW"):
        for multiplier in MULTIPLIER_GRID:
            if multiplier != 1.0:
                add(
                    ArmSpec(
                        candidate_name(family=family, multiplier=multiplier),
                        family,
                        multiplier=multiplier,
                        candidate=True,
                    )
                )
    for kappa in KAPPA_GRID:
        add(ArmSpec(f"R_k{kappa:g}", "R", kappa=kappa, candidate=True))
    add(ArmSpec("R75", "R", kappa=0.75))
    add(ArmSpec("KS", "KS"))
    add(ArmSpec("KS1", "KS1"))
    add(ArmSpec("KW-nomid", "KWnomid"))
    add(ArmSpec("KW-fast", "KWfast"))
    add(ArmSpec("KW-pl52", "pl52"))
    for delay in (0.0, 24.0, 33.0):
        add(ArmSpec(f"KW-d{delay:g}", "KW", delay_hours=delay))
        add(ArmSpec(f"K0-d{delay:g}", "K0", delay_hours=delay))
    return arms


@dataclass(frozen=True)
class Correction:
    """An arm's correction at every row of a site.

    Attributes:
        level: The persistent level part.
        anomaly: The decaying transient part.
        shape: The shape part (the recipe's profile shift, for `R`).
    """

    level: np.ndarray
    anomaly: np.ndarray
    shape: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """The correction to add to the member means."""
        return self.level + self.anomaly + self.shape


def structure_of(*, family: str, band_of_half_hour: np.ndarray) -> shape.ShapeStructure:
    """Return the shape structure of a family."""
    if family in {"K0", "KW", "KWfast", "pl52", "R"}:
        return shape.build_structure(variant="KW", band_of_half_hour=band_of_half_hour)
    if family == "KWnomid":
        base = shape.build_structure(variant="KW", band_of_half_hour=band_of_half_hour)
        frozen = np.flatnonzero(band_of_half_hour == MIDDAY_BAND)
        projection = base.projection.copy()
        projection[np.ix_(frozen, frozen)] = 0.0
        return shape.ShapeStructure(group=base.group, projection=projection)
    return shape.build_structure(variant=family, band_of_half_hour=band_of_half_hour)  # ty: ignore[invalid-argument-type]


def hypers_path() -> Path:
    """Return the file that holds the fitted shape hyperparameters."""
    return WORK_DIR / "hypers.json"


def read_hypers() -> dict:
    """Return the stored shape hyperparameters, keyed by family then site."""
    return json.loads(hypers_path().read_text())


def family_hyper(*, data: site_module.SiteData, family: str, hypers: dict) -> shape.ShapeHyper:
    """Return the filter hyperparameters a family uses at a site.

    Args:
        data: The site data.
        family: The family.
        hypers: `read_hypers()`'s content.

    Returns:
        K0's settings with the shape off for `K0`, `R`, and the shapeless part of `pl52`; the fitted
        shape settings for the shape families.
    """
    base = data.kalman_hyper
    if family in {"K0", "R"}:
        return base
    key = "KW" if family in {"KWnomid", "KWfast", "pl52"} else family
    if key == "KS1":
        return shape.ShapeHyper(*hypers[key][data.site])
    log_q_s, log_sigma_eps = hypers[key][data.site]
    if family == "KWfast":
        log_q_s += float(np.log(FAST_SHAPE_FACTOR))
    return base.with_shape(log_q_s=log_q_s, log_sigma_eps=log_sigma_eps)


class TraceCache:
    """Filter traces of one site, kept so an arm family runs each filter once."""

    def __init__(self, *, data: site_module.SiteData, hypers: dict) -> None:
        """Keep the site data and hyperparameters; traces are built on demand.

        Args:
            data: The site data.
            hypers: The stored shape hyperparameters.
        """
        self.data = data
        self.hypers = hypers
        self._traces: dict[tuple[str, float], shape.ShapeTrace] = {}

    def trace(self, *, family: str, multiplier: float) -> shape.ShapeTrace:
        """Return the filter trace of a family at a multiplier."""
        key = (family, multiplier)
        if key not in self._traces:
            data = self.data
            structure = structure_of(family=family, band_of_half_hour=data.band_of_half_hour)
            hyper = family_hyper(data=data, family=family, hypers=self.hypers)
            self._traces[key] = shape.run_shape_filter(
                residual=data.stream,
                half_hour=data.half_hour,
                structure=structure,
                hyper=hyper.scaled(multiplier=multiplier) if multiplier != 1.0 else hyper,
            )
        return self._traces[key]


def kalman_correction(
    *, traces: TraceCache, family: str, multiplier: float, delay_hours: float, day_only: bool
) -> Correction:
    """Return a filter family's correction at every row."""
    data = traces.data
    structure = structure_of(family=family, band_of_half_hour=data.band_of_half_hour)
    cutoff = data.cutoffs(delay_hours=delay_hours, day_only=day_only)[data.run_of_row]
    found = shape.lookup(
        trace=traces.trace(family=family, multiplier=multiplier),
        structure=structure,
        cutoff=cutoff,
        target=data.position,
        half_hour=data.row_half_hour,
    )
    return Correction(level=found.level, anomaly=found.anomaly, shape=found.shape)


def profile_correction(
    *, data: site_module.SiteData, k0_q50: np.ndarray, kappa: float, delay_hours: float
) -> Correction:
    """Return the recipe's correction: minus kappa times the 14-day D+1 error profile.

    For each run, take the last `PROFILE_DAYS` UTC days that were fully observed by the origin
    minus the delay. Each day's error is GAM-K's q50 of the run that forecast it at D+1 minus the
    reading (raw, switching rows included), with the day's mean removed. The profile is the mean
    of those errors by local half-hour. It is subtracted at every lead.

    Args:
        data: The site data.
        k0_q50: K0's median forecast at every row.
        kappa: The shrink.
        delay_hours: The telemetry delay.

    Returns:
        The profile shift alone, in `shape`; the recipe adds it to K0's level and transient.
    """
    d1 = data.d1_rows()
    error = k0_q50 - data.actual
    blocks: dict[np.datetime64, tuple[np.ndarray, np.ndarray]] = {}
    origin_days = data.run_origins.astype("datetime64[D]")
    for run in range(len(data.runs)):
        rows = np.arange(data.offsets[run], data.offsets[run + 1])
        rows = rows[d1[rows]]
        if len(rows) != 48:
            continue
        block = error[rows]
        if int(np.isfinite(block).sum()) < MIN_DAY_READINGS:
            continue
        centred = block - np.nanmean(block)
        blocks[origin_days[run]] = (centred, data.row_half_hour[rows])
    shift = np.zeros(data.n_rows)
    delay_days = int(np.ceil((delay_hours - 9.0) / 24.0)) if delay_hours > 9.0 else 0
    for run in range(len(data.runs)):
        origin_day = origin_days[run]
        total = np.zeros(48)
        count = np.zeros(48)
        used = 0
        for back in range(1 + delay_days, PROFILE_DAYS + 1 + delay_days):
            # The block of D+1 targets for UTC day (origin_day - back) came from the run of the
            # day before it, so it was fully observed before this origin.
            key = origin_day - np.timedelta64(back + 1, "D")
            if key not in blocks:
                continue
            centred, half_hours = blocks[key]
            finite = np.isfinite(centred)
            np.add.at(total, half_hours[finite], centred[finite])
            np.add.at(count, half_hours[finite], 1)
            used += 1
        if used < MIN_PROFILE_DAYS:
            continue
        profile = np.where(count > 0, total / np.maximum(count, 1), 0.0)
        rows = np.arange(data.offsets[run], data.offsets[run + 1])
        shift[rows] = -kappa * profile[data.row_half_hour[rows]]
    zeros = np.zeros(data.n_rows)
    return Correction(level=zeros, anomaly=zeros, shape=shift)


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


def lead_band_correction(
    *, data: site_module.SiteData, family: str, multipliers: tuple[float, ...]
) -> Correction:
    """Return a family's correction with each lead band taking its own multiplier's state."""
    lead = data.lead_band
    parts = [
        load_correction(
            arm=family
            if multiplier == 1.0
            else candidate_name(family=family, multiplier=multiplier),
            site=data.site,
        )
        for multiplier in multipliers
    ]
    out = {}
    for field in ("level", "anomaly", "shape"):
        stacked = np.stack([getattr(part, field) for part in parts])
        out[field] = stacked[lead, np.arange(data.n_rows)]
    return Correction(**out)


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


def year_earlier_shape(*, traces: TraceCache, delay_hours: float) -> np.ndarray:
    """Return KW's shape from 52 weeks before each run, at every row.

    The state is the one KW had at the cutoff a run's origin minus 364 days implies, so each
    scored origin reads the shape that the same weekday of the previous year's stream held.
    """
    data = traces.data
    structure = structure_of(family="KW", band_of_half_hour=data.band_of_half_hour)
    earlier = usable_cutoff_positions(
        end_times=data.cache.end_times,
        initialisation=data.run_origins - np.timedelta64(364, "D"),
        delay_hours=delay_hours,
    )
    found = shape.lookup(
        trace=traces.trace(family="KW", multiplier=1.0),
        structure=structure,
        cutoff=earlier[data.run_of_row],
        target=data.position,
        half_hour=data.row_half_hour,
    )
    return found.shape


def compute_correction(
    *, spec: ArmSpec, traces: TraceCache, k0_q50: np.ndarray | None = None
) -> Correction:
    """Return an arm's correction at every row of a site.

    Args:
        spec: The arm.
        traces: The site's trace cache.
        k0_q50: K0's median forecast at every row, needed by the recipe.

    Returns:
        The correction.

    Raises:
        ValueError: If the recipe is asked for without K0's median.
    """
    data = traces.data
    if spec.family == "R":
        if k0_q50 is None:
            msg = "the recipe needs K0's median forecast"
            raise ValueError(msg)
        profile = profile_correction(
            data=data, k0_q50=k0_q50, kappa=spec.kappa, delay_hours=spec.delay_hours
        )
        k0 = kalman_correction(
            traces=traces,
            family="K0",
            multiplier=1.0,
            delay_hours=spec.delay_hours,
            day_only=False,
        )
        return Correction(level=k0.level, anomaly=k0.anomaly, shape=profile.shape)
    if spec.family == "pl52":
        k0 = kalman_correction(
            traces=traces,
            family="K0",
            multiplier=1.0,
            delay_hours=spec.delay_hours,
            day_only=False,
        )
        return Correction(
            level=k0.level,
            anomaly=k0.anomaly,
            shape=year_earlier_shape(traces=traces, delay_hours=spec.delay_hours),
        )
    return kalman_correction(
        traces=traces,
        family=spec.family,
        multiplier=spec.multiplier,
        delay_hours=spec.delay_hours,
        day_only=spec.day_only,
    )
