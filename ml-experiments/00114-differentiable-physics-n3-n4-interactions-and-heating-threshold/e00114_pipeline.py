"""Frozen yearly fits, replicas, the Kalman filter, and day-ahead quantiles for experiment 114.

**One fit at cut `c` serves two purposes.** It forecasts the year after `c`, and the same fit is
the replica of the cut 12 months later, because `studies.dp_gam.replica_residuals` refits the same
arm at that earlier cut. A `YearRun` therefore holds one arm's frozen forecast over the year after
its cut, and the residuals of that forecast are both the filter's input stream and the spread's
replica.

The quantiles of an unfiltered arm are the frozen mean plus the replica's centred residual
quantiles per hour band and calendar quarter. The quantiles of a filtered arm (suffix `-K`) are the
corrected mean plus the replica's post-filter residual quantiles
(`studies.dp_assimilation.post_filter_lead_spread`).
"""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final

import e00114_common as common
import e00114_design as design
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    Correction,
    CorrectionTrace,
    FilterHyperparameters,
    LeadSpread,
    TraceBuilder,
    correction_at,
    day_ahead_initialisation,
    post_filter_lead_spread,
    run_kalman,
    stitch_out_of_sample_residuals,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    ReplicaResiduals,
    ResidualSpread,
    assert_before_holdout,
    replica_rows,
)

MIN_TRAINING_ROWS: Final[int] = 90 * 48
"""A site-cut with fewer training rows (90 days) has no fit."""
INTERACTION_ARMS: Final[tuple[str, ...]] = ("t1", "t2", "t3", "t_all")
ARM_NAMES: Final[tuple[str, ...]] = ("b0", *INTERACTION_ARMS, "h_site")
PREVIOUS_LABEL: Final[dict[str, str]] = {
    "tuning": "replica_tuning",
    "replicate": "tuning",
    "scored": "replicate",
}
"""The fit whose year is each scored year's replica."""


@dataclass(frozen=True)
class YearRun:
    """One arm's frozen fit at one site and cut, with its forecast over the year after the cut.

    Attributes:
        site: The substation's label.
        fit_label: The cut's label, a key of `common.FIT_CUTS`.
        arm_name: `b0`, `t1`, `t2`, `t3`, `t_all`, or `h_site`.
        penalty_weight: The difference penalty's weight, 0 for B0 and the threshold arm.
        threshold_c: The heating threshold the fit used.
        index: The grid rows with a reading in the year after the cut.
        mean: The frozen mean forecast at `index`, in points.
        n_training_rows: The number of rows the fit trained on.
        fit: The fitted arm.
    """

    site: str
    fit_label: str
    arm_name: str
    penalty_weight: float
    threshold_c: float
    index: np.ndarray
    mean: np.ndarray
    n_training_rows: int
    fit: design.ArmFit

    @property
    def cut(self) -> datetime:
        """Return the fit's cut."""
        return common.FIT_CUTS[self.fit_label]

    def residual_stream(self, *, cache: FeatureCache) -> np.ndarray:
        """Return observed minus forecast on the grid over the forecast year, `NaN` elsewhere."""
        stream = np.full(len(cache.end_times), np.nan)
        stream[self.index] = cache.sites[self.site].y[self.index] - self.mean
        return stream


def arm_spec(*, arm_name: str, penalty_weight: float, threshold_c: float | None) -> design.ArmSpec:
    """Return the arm for a name, a penalty weight, and (for `h_site`) a chosen threshold.

    Args:
        arm_name: A member of `ARM_NAMES`.
        penalty_weight: The difference penalty's weight, used by the interaction arms.
        threshold_c: The heating threshold of `h_site`.

    Returns:
        The arm.

    Raises:
        ValueError: If `h_site` is asked for without a threshold.
    """
    if arm_name == "b0":
        return design.B0
    if arm_name == "h_site":
        if threshold_c is None:
            msg = "h_site needs the threshold its profile chose"
            raise ValueError(msg)
        return replace(design.threshold_arm(threshold_c=threshold_c), name="h_site")
    return design.interaction_arm(name=arm_name, penalty_weight=penalty_weight)


def profile_thresholds(*, cache: FeatureCache, base: design.BaseFit) -> dict[float, design.ArmFit]:
    """Fit B0 with the hinge at each threshold of the grid, on the full training sample.

    Args:
        cache: The feature cache.
        base: B0's inputs at the cut.

    Returns:
        The fit at each threshold.
    """
    return {
        threshold: design.fit_arm(
            cache=cache, base=base, arm=design.threshold_arm(threshold_c=threshold)
        )
        for threshold in common.HINGE_GRID_C
    }


def choose_threshold(*, fits: dict[float, design.ArmFit]) -> float:
    """Return the threshold with the lowest unpenalised training loss (the earliest on a tie)."""
    return min(fits, key=lambda threshold: (fits[threshold].unpenalised_loss, threshold))


def fit_year(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    site: str,
    fit_label: str,
    arm_name: str,
    penalty_weight: float = 0.0,
) -> YearRun | None:
    """Fit one arm at one site and cut and forecast the year after the cut.

    Args:
        cache: The feature cache.
        vintages: Experiment 45's real-time capacity vintages.
        site: The substation's label.
        fit_label: A key of `common.FIT_CUTS`.
        arm_name: A member of `ARM_NAMES`.
        penalty_weight: The difference penalty's weight, for an interaction arm.

    Returns:
        The run, or `None` if the site has fewer than `MIN_TRAINING_ROWS` training rows at the cut.
    """
    cut = common.FIT_CUTS[fit_label]
    try:
        base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut)
    except ValueError:
        return None
    if base.n < MIN_TRAINING_ROWS:
        return None
    threshold = None
    if arm_name == "h_site":
        threshold = choose_threshold(fits=profile_thresholds(cache=cache, base=base))
    arm = arm_spec(arm_name=arm_name, penalty_weight=penalty_weight, threshold_c=threshold)
    fit = design.fit_arm(cache=cache, base=base, arm=arm)
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    assert_before_holdout(cache=cache, index=index)
    mean = design.predict_arm(cache=cache, base=base, fit=fit, index=index)
    return YearRun(
        site=site,
        fit_label=fit_label,
        arm_name=arm_name,
        penalty_weight=penalty_weight,
        threshold_c=fit.threshold_c,
        index=index,
        mean=mean,
        n_training_rows=base.n,
        fit=fit,
    )


def replica_of(*, cache: FeatureCache, run: YearRun, final_cut: datetime) -> ReplicaResiduals:
    """Return the replica residuals of the cut `final_cut`, taken from the fit 12 months earlier.

    The rows are `studies.dp_gam.replica_rows`' rows with the switching rows kept and flagged, as
    the filter needs, and the residuals are the run's.

    Args:
        cache: The feature cache.
        run: The run whose cut is 12 months before `final_cut`.
        final_cut: The final fit's cut.

    Returns:
        The replica residuals.

    Raises:
        ValueError: If a replica row is not among the run's forecast rows.
    """
    features = cache.sites[run.site]
    rows = replica_rows(
        cache=cache, site=run.site, fit_cut=run.cut, cut=final_cut, drop_switching=False
    )
    positions = np.searchsorted(run.index, rows)
    if len(rows) and (
        positions.max() >= len(run.index) or not np.array_equal(run.index[positions], rows)
    ):
        msg = f"{run.site}: the replica rows of {final_cut:%Y-%m-%d} are not all forecast rows"
        raise ValueError(msg)
    return ReplicaResiduals(
        residuals=features.y[rows] - run.mean[positions],
        band=cache.calendar.band[rows],
        quarter=cache.calendar.quarter[rows],
        index=rows,
        fit_cut=run.cut,
        switching=features.switching_mask[rows],
    )


def trace_builder(*, hyper: FilterHyperparameters) -> TraceBuilder:
    """Return the function that builds the Kalman trace from a residual stream.

    Args:
        hyper: The site's filter hyperparameters.

    Returns:
        A function taking the grid residuals and bands (by keyword) and returning a trace.
    """

    def build(*, residual: np.ndarray, band: np.ndarray) -> CorrectionTrace:
        return run_kalman(residual=residual, band=band, hyper=hyper)

    return build


def stitched_stream(*, cache: FeatureCache, runs: dict[str, YearRun], label: str) -> np.ndarray:
    """Join the out-of-sample residuals of the fits in a scored year's chain into one stream.

    The filter carries its state across the refits at the earlier cuts and switches to each new
    fit's residuals at its cut. A site with no fit at an earlier cut starts the chain at its first.

    Args:
        cache: The feature cache.
        runs: The site's runs of one arm by fit label.
        label: The scored year, a key of `common.CHAIN`.

    Returns:
        The stitched stream on the grid.
    """
    labels = [name for name in common.CHAIN[label] if name in runs]
    cut_positions = [
        int(
            np.searchsorted(
                cache.end_times, common.to_64(moment=common.FIT_CUTS[name]), side="right"
            )
        )
        - 1
        for name in labels
    ]
    return stitch_out_of_sample_residuals(
        cut_positions=cut_positions,
        residuals=[runs[name].residual_stream(cache=cache) for name in labels],
    )


@dataclass(frozen=True)
class ScoredYear:
    """One arm's day-ahead forecast at one site over one scored year.

    Attributes:
        index: The scored grid rows.
        mean: The frozen mean.
        filtered_mean: The mean plus the Kalman correction.
        quantiles: The unfiltered 11 quantiles, shape (rows, 11).
        filtered_quantiles: The filtered 11 quantiles.
        correction: The Kalman correction.
    """

    index: np.ndarray
    mean: np.ndarray
    filtered_mean: np.ndarray
    quantiles: np.ndarray
    filtered_quantiles: np.ndarray
    correction: Correction


def score_year(
    *,
    cache: FeatureCache,
    runs: dict[str, YearRun],
    label: str,
    hyper: FilterHyperparameters,
    delay_hours: float = DEFAULT_DELAY_HOURS,
) -> ScoredYear:
    """Forecast a scored year day ahead, unfiltered and filtered.

    Args:
        cache: The feature cache.
        runs: The site's runs of one arm by fit label; it must hold `label` and its replica's.
        label: The scored year: `tuning`, `replicate`, or `scored`.
        hyper: The site's filter hyperparameters.
        delay_hours: The telemetry delay.

    Returns:
        The forecast.
    """
    run = runs[label]
    replica = replica_of(cache=cache, run=runs[PREVIOUS_LABEL[label]], final_cut=run.cut)
    keep = ~replica.switching
    spread = ResidualSpread.fit(
        residuals=replica.residuals[keep],
        band=replica.band[keep],
        quarter=replica.quarter[keep],
        centre=True,
    )
    band = cache.calendar.band
    quarter = cache.calendar.quarter
    quantiles = spread.quantiles(mean=run.mean, band=band[run.index], quarter=quarter[run.index])
    build = trace_builder(hyper=hyper)
    lead_spread: LeadSpread = post_filter_lead_spread(
        end_times=cache.end_times, replica=replica, run=build, delay_hours=delay_hours
    )
    trace = build(residual=stitched_stream(cache=cache, runs=runs, label=label), band=band)
    valid = cache.end_times[run.index]
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=day_ahead_initialisation(end_times=valid, days_ahead=1),
        delay_hours=delay_hours,
    )
    correction = correction_at(trace=trace, cutoff=cutoff, target=run.index, band=band[run.index])
    filtered_mean = run.mean + correction.total
    filtered_quantiles = lead_spread.quantiles(
        mean=filtered_mean, band=band[run.index], quarter=quarter[run.index], lead_band=0
    )
    return ScoredYear(
        index=run.index,
        mean=run.mean,
        filtered_mean=filtered_mean,
        quantiles=quantiles,
        filtered_quantiles=filtered_quantiles,
        correction=correction,
    )


def scored_frames(
    *, cache: FeatureCache, site: str, arm_name: str, scored: ScoredYear
) -> dict[str, pl.DataFrame]:
    """Return the prediction frames of one arm at one site: unfiltered and filtered.

    Args:
        cache: The feature cache.
        site: The substation's label.
        arm_name: The arm's name; the filtered frame's arm is `<name>_k`.
        scored: The forecast.

    Returns:
        The frames keyed by arm name.
    """
    return {
        arm_name: common.prediction_frame(
            cache=cache,
            site=site,
            index=scored.index,
            quantiles=scored.quantiles,
            mean=scored.mean,
            arm=arm_name,
        ),
        f"{arm_name}_k": common.prediction_frame(
            cache=cache,
            site=site,
            index=scored.index,
            quantiles=scored.filtered_quantiles,
            mean=scored.filtered_mean,
            arm=f"{arm_name}_k",
            level=scored.correction.level,
            anomaly=scored.correction.anomaly,
        ),
    }
