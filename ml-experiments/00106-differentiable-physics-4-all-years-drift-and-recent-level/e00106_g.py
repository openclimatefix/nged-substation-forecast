"""The convex model's arms of experiment 106, fitted per substation and corrected by the filters.

One frozen fit per arm, substation, and cut, on data before the cut, with the capacity pinned to
the real-time vintage (`studies.dp_pinned_capacity`). The mean forecast of each scored row is
written once; the quantiles are the mean plus the centred residual quantiles of a replica of the
arm (a fit 12 months before the cut, predicting the following 12 months). A filter arm adds the
recent-level correction to the mean and takes its spread from the replica run through the same
correction (`studies.dp_assimilation.post_filter_lead_spread`).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, Literal

import e00106_common as common
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    Correction,
    CorrectionTrace,
    FilterHyperparameters,
    LeadSpread,
    correction_at,
    day_ahead_initialisation,
    lead_band_of_days_ahead,
    post_filter_lead_spread,
    run_ewma,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    GamFit,
    GamSpec,
    ReplicaResiduals,
    ResidualSpread,
    TrainingWindow,
    fit_gam,
    predict_gam,
    replica_residuals,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity

DEFAULT_RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty; its grid was flat to within 0.003 points."""
DEFAULT_DRIFT_STEP: Final[float] = 3000.0
"""Experiment 104's tuned drift step, used for the replica spread that scores a tuning grid."""

SMOOTH_DRIFT_STEP: Final[float] = 3000.0
"""Experiment 104's drift step, for the smoother descriptive arm `a1_smooth`."""
ACTIVE_STEP_PENALTY: Final[float] = 100.0
"""The step penalty of the exploratory arm `a1s_smooth`.

At the tuned drift step the tuned penalty of 1,000 switches every step off, and a penalty of 300
shrinks a true 8-point step to about 70% of its size (known-answer gate), so the exploratory arm
uses 100 with the smoother drift of experiment 104.
"""

TunedPath = common.WORK_DIR / "tuned.json"

FilterKindType = Literal["kalman", "ewma"]
"""Which recent-level correction a filter arm applies."""

FilterPartType = Literal["both", "level", "anomaly"]
"""Which hidden states of the Kalman filter enter the correction."""


@dataclass(frozen=True)
class ArmDefinition:
    """One convex-model arm.

    Attributes:
        name: The arm's name, used in prediction sets.
        spec: The penalties, drift form, capacity mode, COVID flag, and final-knot treatment.
        window: The training-window rule.
    """

    name: str
    spec: GamSpec
    window: TrainingWindow


def read_tuned() -> dict:
    """Return the tuned settings written by `e00106_tune.py`."""
    return json.loads(TunedPath.read_text())


def arm_definitions(*, tuned: dict) -> dict[str, ArmDefinition]:
    """Build every convex-model arm from the tuned settings.

    Args:
        tuned: The contents of `work/tuned.json`.

    Returns:
        The arms keyed by name: `a0`, `a1`, `a1s`, `w`, `a1_fc`, `a1_cov`, `a1_smooth`,
        `a1s_smooth`, and one `a1_knot_*` arm per final-knot variant.
    """
    base = GamSpec(
        ridge=DEFAULT_RIDGE,
        drift_step=tuned["drift_step"],
        step_penalty=tuned["step_penalty"],
        capacity="pinned",
        final_knot=tuned["final_knot"],
        final_knot_shrink=tuned["final_knot_shrink"],
    )
    all_years = TrainingWindow()
    a1 = replace(base, drift="d1")
    window = tuned["window"]
    window_rule = TrainingWindow(months=window["months"]) if window["months"] else all_years
    arms = {
        "a0": ArmDefinition("a0", replace(base, drift="none", final_knot="free"), all_years),
        "a1": ArmDefinition("a1", a1, all_years),
        "a1s": ArmDefinition("a1s", replace(a1, drift="d1_steps"), all_years),
        "w": ArmDefinition(
            "w",
            replace(base, drift="d1" if window["drift"] else "none", final_knot="free"),
            window_rule,
        ),
        "a1_fc": ArmDefinition("a1_fc", replace(a1, capacity="fitted"), all_years),
        "a1_cov": ArmDefinition("a1_cov", replace(a1, covid_flag=True), all_years),
        "a1_smooth": ArmDefinition(
            "a1_smooth", replace(a1, drift_step=SMOOTH_DRIFT_STEP), all_years
        ),
        "a1s_smooth": ArmDefinition(
            "a1s_smooth",
            replace(
                a1,
                drift="d1_steps",
                drift_step=SMOOTH_DRIFT_STEP,
                step_penalty=ACTIVE_STEP_PENALTY,
            ),
            all_years,
        ),
    }
    for knot, shrink in (("free", 0.5), ("shrunk", 0.25), ("shrunk", 0.5), ("shrunk", 0.75)):
        label = knot if knot == "free" else f"shrunk{round(100 * shrink)}"
        arms[f"a1_knot_{label}"] = ArmDefinition(
            f"a1_knot_{label}",
            replace(a1, final_knot=knot, final_knot_shrink=shrink),
            all_years,
        )
    arms["a1_knot_pinned"] = ArmDefinition(
        "a1_knot_pinned", replace(a1, final_knot="pinned"), all_years
    )
    return arms


@dataclass(frozen=True)
class SiteRun:
    """One arm's frozen fit at one cut for one substation, with its mean over the scored year.

    Attributes:
        site: The substation's label.
        cut: The fit's cut.
        definition: The arm.
        fit: The fitted model.
        pinned: The pinned capacity, or `None` for a fitted-capacity arm.
        index: The scored grid rows: those with a reading in the year after the cut.
        mean: The mean forecast at `index`, in points.
        replica: The replica residuals with the switching rows kept and flagged.
    """

    site: str
    cut: datetime
    definition: ArmDefinition
    fit: GamFit
    pinned: PinnedCapacity | None
    index: np.ndarray
    mean: np.ndarray
    replica: ReplicaResiduals

    def residual_stream(self, *, cache: FeatureCache) -> np.ndarray:
        """Return observed minus forecast on the grid over the scored year, `NaN` elsewhere."""
        stream = np.full(len(cache.end_times), np.nan)
        stream[self.index] = cache.sites[self.site].y[self.index] - self.mean
        return stream

    def uncorrected_spread(self) -> ResidualSpread:
        """Return the replica's centred residual quantiles outside the switching masks."""
        keep = ~self.replica.switching
        return ResidualSpread.fit(
            residuals=self.replica.residuals[keep],
            band=self.replica.band[keep],
            quarter=self.replica.quarter[keep],
            centre=True,
        )


def run_site(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    site: str,
    cut: datetime,
    definition: ArmDefinition,
) -> SiteRun:
    """Fit one arm for one substation at one cut and predict the year after.

    Args:
        cache: The feature cache.
        vintages: Experiment 45's real-time capacity vintages.
        site: The substation's label.
        cut: The fit's cut.
        definition: The arm.

    Returns:
        The run.
    """
    pinned = None
    if definition.spec.capacity == "pinned":
        pinned = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[definition.spec],
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    replica = replica_residuals(
        cache=cache,
        site=site,
        spec=definition.spec,
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
        capacity_vintages=vintages,
        drop_switching=False,
    )
    return SiteRun(
        site=site,
        cut=cut,
        definition=definition,
        fit=fit,
        pinned=pinned,
        index=index,
        mean=predict_gam(cache=cache, fit=fit, index=index),
        replica=replica,
    )


def trace_builder(
    *, kind: FilterKindType, hyper: FilterHyperparameters | None, part: FilterPartType = "both"
) -> Callable[[np.ndarray, np.ndarray], CorrectionTrace]:
    """Return the function that builds a correction trace from a residual stream.

    Args:
        kind: Which correction.
        hyper: The Kalman filter's hyperparameters; unused by the EWMA.
        part: Which hidden states the correction keeps: the other state's trace is set to zero.
            The EWMA has only a level, so `anomaly` gives a zero correction.

    Returns:
        A function taking the grid residuals and bands and returning a `CorrectionTrace`.

    Raises:
        ValueError: If the Kalman filter has no hyperparameters.
    """

    def build(stream: np.ndarray, band: np.ndarray) -> CorrectionTrace:
        if kind == "ewma":
            trace = run_ewma(residual=stream, band=band)
        elif hyper is None:
            msg = "the Kalman correction needs hyperparameters"
            raise ValueError(msg)
        else:
            trace = run_kalman(residual=stream, band=band, hyper=hyper)
        if part == "level":
            return replace(trace, anomaly=tuple(np.zeros_like(a) for a in trace.anomaly))
        if part == "anomaly":
            return replace(trace, level=tuple(np.zeros_like(a) for a in trace.level))
        return trace

    return build


@dataclass(frozen=True)
class FilteredForecast:
    """A corrected arm's forecast, quantiles, and correction at one number of days ahead.

    Attributes:
        corrected_mean: The frozen mean plus the correction.
        quantiles: The 11 quantiles, shape (rows, 11).
        correction: The correction and its two parts.
    """

    corrected_mean: np.ndarray
    quantiles: np.ndarray
    correction: Correction


def filtered_forecast(
    *,
    cache: FeatureCache,
    run: SiteRun,
    kind: FilterKindType,
    hyper: FilterHyperparameters | None,
    days_ahead: int = 1,
    delay_hours: float = DEFAULT_DELAY_HOURS,
    lead_spread: LeadSpread | None = None,
    part: FilterPartType = "both",
) -> FilteredForecast:
    """Correct a run's mean from the readings usable at its day-ahead initialisation.

    Args:
        cache: The feature cache.
        run: The arm's run.
        kind: Which correction.
        hyper: The Kalman filter's hyperparameters.
        days_ahead: How many days before the valid day the forecast is made.
        delay_hours: The telemetry delay.
        lead_spread: The post-filter spread, built from the run's replica if not given.
        part: Which hidden states of the Kalman filter enter the correction.

    Returns:
        The corrected mean, its quantiles, and the correction.
    """
    build = trace_builder(kind=kind, hyper=hyper, part=part)
    band = cache.calendar.band
    trace: CorrectionTrace = build(run.residual_stream(cache=cache), band)
    valid = cache.end_times[run.index]
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=day_ahead_initialisation(end_times=valid, days_ahead=days_ahead),
        delay_hours=delay_hours,
    )
    correction = correction_at(trace=trace, cutoff=cutoff, target=run.index, band=band[run.index])
    spread = lead_spread or post_filter_lead_spread(
        end_times=cache.end_times, replica=run.replica, run=build, delay_hours=delay_hours
    )
    corrected_mean = run.mean + correction.total
    quantiles = spread.quantiles(
        mean=corrected_mean,
        band=band[run.index],
        quarter=cache.calendar.quarter[run.index],
        lead_band=lead_band_of_days_ahead(days_ahead=days_ahead),
    )
    return FilteredForecast(
        corrected_mean=corrected_mean, quantiles=quantiles, correction=correction
    )


def uncorrected_quantiles(*, cache: FeatureCache, run: SiteRun) -> np.ndarray:
    """Return the run's quantiles: the frozen mean plus the replica's centred residual quantiles."""
    return run.uncorrected_spread().quantiles(
        mean=run.mean,
        band=cache.calendar.band[run.index],
        quarter=cache.calendar.quarter[run.index],
    )
