"""The arms of experiment 110, fitted per substation, with the recent-level filter on top.

One frozen fit per arm, substation, and cut, on data before the cut. The mean forecast of each
scored row is written once; the quantiles are the mean plus the centred residual quantiles of a
replica of the arm (a fit 12 months before the cut, predicting the following 12 months). A filtered
forecast adds the Kalman filter's correction to the mean and takes its spread from the replica run
through the same filter (`studies.dp_assimilation.post_filter_lead_spread`). Every arm is written
twice, filtered and unfiltered.

**Three kinds of fit.**

- `gam`: the convex model of `studies.dp_gam` on a signed series (the MW substations), or on `|S|`
  read as signed power (arm N at an MVA substation).
- `meter`: the same model fitted through the MVA meter (`studies.dp_meter`), whose mean forecast is
  the predicted `|S|`.
- A `gam` arm with `zero_solar` is fitted to the dark rows with a pinned capacity of zero, which is
  the demand-alone rule for an MVA substation (arm G-dark).

**The clip is profiled at every cut on the training rows,** and the replica profiles its own clip
at its own cut, so neither sees the year it predicts.
"""

import json
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Final, Literal

import e00110_common as common
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
    lead_band_of_days_ahead,
    post_filter_lead_spread,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    ClipSpec,
    GamFit,
    GamSpec,
    NodeSpec,
    ReplicaResiduals,
    ResidualSpread,
    TrainingWindow,
    fit_gam,
    predict_gam,
    prepare_fit,
    profile_clip,
    replica_cut,
    replica_residuals,
)
from studies.dp_meter import (
    MeterFit,
    QModeType,
    fit_gam_through_meter,
    meter_replica_residuals,
    predict_meter,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity

TUNED_PATH: Final = common.WORK_DIR / "tuned.json"
PART_4_TUNED_PATH: Final = common.PART_4_DIR / "work" / "tuned.json"

ArmKindType = Literal["gam", "meter"]
"""Which fit an arm uses."""


@dataclass(frozen=True)
class ArmDefinition:
    """One arm.

    Attributes:
        name: The arm's name, used in prediction sets.
        spec: The penalties, drift form, capacity mode, and generation-side options. A clip is
            set per cut by profiling, so the spec's own `clip` is `None`.
        window: The training-window rule.
        sites: The substations the arm is fitted at.
        kind: `gam` or `meter`.
        qmode: For a `meter` arm, whether the reactive power is zero or learned.
        clip_sites: The substations at which the clip is profiled and applied.
        zero_solar: True to fit with a pinned capacity of zero (arm G-dark).
    """

    name: str
    spec: GamSpec
    window: TrainingWindow
    sites: tuple[str, ...]
    kind: ArmKindType = "gam"
    qmode: QModeType = "zero"
    clip_sites: tuple[str, ...] = ()
    zero_solar: bool = False


def read_part_4_tuned() -> dict:
    """Return Part 4's tuned settings (drift step, final knot, and filter hyperparameters)."""
    return json.loads(PART_4_TUNED_PATH.read_text())


def read_tuned() -> dict:
    """Return the settings tuned by `e00110_tune.py`."""
    return json.loads(TUNED_PATH.read_text())


def base_specs(*, part_4: dict) -> dict[str, GamSpec]:
    """Return the base arms B (Part 4's A2) and B-D1 (Part 4's A3) as specs.

    Args:
        part_4: Part 4's tuned settings.

    Returns:
        `b`: all years, no drift, pinned capacity. `b_d1`: the same with D1 at Part 4's tuned
        drift step and final-knot rule.
    """
    b = GamSpec(ridge=common.RIDGE, capacity="pinned")
    b_d1 = replace(
        b,
        drift="d1",
        drift_step=part_4["drift_step"],
        final_knot=part_4["final_knot"],
        final_knot_shrink=part_4["final_knot_shrink"],
    )
    return {"b": b, "b_d1": b_d1}


def arm_definitions(*, part_4: dict, tuned: dict | None) -> dict[str, ArmDefinition]:
    """Build every arm.

    Args:
        part_4: Part 4's tuned settings.
        tuned: This experiment's tuned settings (the covariance penalty's weight and the node's
            penalties), or `None` before tuning, when the arms that need them are left out.

    Returns:
        The arms keyed by name.
    """
    base = base_specs(part_4=part_4)
    all_years = TrainingWindow()
    last_two_years = TrainingWindow(months=24)
    mw = common.MW_SCORED_SITES
    fitted = replace(base["b"], capacity="fitted")
    arms: dict[str, ArmDefinition] = {
        "b": ArmDefinition("b", base["b"], all_years, mw),
        "b_d1": ArmDefinition("b_d1", base["b_d1"], all_years, mw),
        "b_clip": ArmDefinition("b_clip", base["b"], all_years, mw, clip_sites=mw),
        "k_base": ArmDefinition("k_base", fitted, all_years, mw),
        "k_ii": ArmDefinition(
            "k_ii",
            replace(fitted, shared_harmonics=True, slow_heating_only=True),
            all_years,
            mw,
        ),
        "k_iii": ArmDefinition("k_iii", replace(fitted, solar="anomaly"), last_two_years, mw),
        "k_iiiii": ArmDefinition(
            "k_iiiii",
            replace(fitted, solar="anomaly", shared_harmonics=True, slow_heating_only=True),
            last_two_years,
            mw,
        ),
        "n": ArmDefinition("n", fitted, all_years, common.MVA_SITES),
        "l0": ArmDefinition("l0", fitted, all_years, common.MVA_SITES, "meter", "zero"),
        "lq": ArmDefinition("lq", fitted, all_years, common.MVA_SITES, "meter", "learned"),
        "g_dark": ArmDefinition(
            "g_dark",
            replace(base["b"], capacity="pinned"),
            TrainingWindow(dark_only=True),
            common.MVA_SITES,
            zero_solar=True,
        ),
    }
    if tuned is not None:
        node = NodeSpec(**tuned["node"])
        arms["k_i"] = ArmDefinition(
            "k_i", replace(fitted, orthogonal_penalty=tuned["orthogonal_lambda"]), all_years, mw
        )
        arms["b_node"] = ArmDefinition("b_node", replace(base["b_d1"], node=node), all_years, mw)
        arms["b_gen"] = ArmDefinition(
            "b_gen",
            replace(base["b_d1"], node=node),
            all_years,
            mw,
            clip_sites=(common.FLAGGED_CLIP_SITE,),
        )
    return arms


def zero_vintages(*, cache: FeatureCache, site: str) -> pl.DataFrame:
    """Return a capacity-vintage frame whose capacity is zero in every week.

    A fit with this frame as its vintages has a pinned capacity of zero, so the solar term
    vanishes. The weeks cover the whole grid and the origin is before every cut.
    """
    first = date(2019, 8, 26)
    weeks = [first + timedelta(days=7 * i) for i in range(380)]
    return pl.DataFrame(
        {
            "time_series_id": cache.sites[site].table.time_series_id,
            "variant": "frozen",
            "origin_monday": date(2019, 9, 2),
            "week_start": weeks,
            "capacity_mw": 0.0,
            "unobserved": False,
        }
    )


@dataclass(frozen=True)
class SiteRun:
    """One arm's frozen fit at one cut for one substation, with its mean over the scored year.

    Attributes:
        site: The substation's label.
        cut: The fit's cut.
        definition: The arm.
        spec: The spec actually fitted: the definition's, with the profiled clip if any.
        fit: The fitted model; for a meter arm, the meter fit's coefficients.
        meter: The meter fit, for a `meter` arm.
        pinned: The pinned capacity, or `None` for a fitted-capacity arm.
        index: The scored grid rows: those with a reading in the year after the cut.
        mean: The mean forecast at `index`, in points (`|S|` for a meter arm).
        real_power: A meter arm's predicted signed real power at `index`.
        reactive_power: A meter arm's predicted reactive power at `index`.
        replica: The replica residuals with the switching rows kept and flagged.
    """

    site: str
    cut: datetime
    definition: ArmDefinition
    spec: GamSpec
    fit: GamFit
    meter: MeterFit | None
    pinned: PinnedCapacity | None
    index: np.ndarray
    mean: np.ndarray
    real_power: np.ndarray | None
    reactive_power: np.ndarray | None
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

    @property
    def extra_columns(self) -> dict[str, np.ndarray]:
        """Return the columns a meter arm adds to its predictions."""
        if self.real_power is None or self.reactive_power is None:
            return {}
        return {"p_mw": self.real_power, "q_mvar": self.reactive_power}


def _vintages_for(
    *, cache: FeatureCache, vintages: pl.DataFrame, definition: ArmDefinition, site: str
) -> pl.DataFrame:
    return zero_vintages(cache=cache, site=site) if definition.zero_solar else vintages


def profiled_clip(
    *,
    cache: FeatureCache,
    site: str,
    spec: GamSpec,
    cut: datetime,
    window: TrainingWindow,
    pinned: PinnedCapacity,
) -> ClipSpec:
    """Return the clip with the lowest penalised objective on the training rows before `cut`.

    The profile is fitted on the spec without its node: it solves 170 systems at once, which with a
    node's thousands of columns would need gigabytes and more pivoting than its solver allows. The
    node is added to the fit that uses the clip.
    """
    return profile_clip(
        cache=cache,
        site=site,
        spec=replace(spec, node=None),
        cut=cut,
        window=window,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    ).best


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
        The run, with its replica.
    """
    table = cache.sites[site].table
    arm_vintages = _vintages_for(cache=cache, vintages=vintages, definition=definition, site=site)
    spec = definition.spec
    replica_spec = spec
    pinned = None
    if spec.capacity == "pinned":
        pinned = pinned_capacity(vintages=arm_vintages, table=table, cut=cut)
        if site in definition.clip_sites:
            spec = replace(
                spec,
                clip=profiled_clip(
                    cache=cache,
                    site=site,
                    spec=spec,
                    cut=cut,
                    window=definition.window,
                    pinned=pinned,
                ),
            )
            replica_pinned = pinned_capacity(
                vintages=arm_vintages, table=table, cut=replica_cut(cut=cut)
            )
            replica_spec = replace(
                definition.spec,
                clip=profiled_clip(
                    cache=cache,
                    site=site,
                    spec=definition.spec,
                    cut=replica_cut(cut=cut),
                    window=definition.window,
                    pinned=replica_pinned,
                ),
            )
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    if definition.kind == "meter":
        meter = fit_gam_through_meter(
            cache=cache,
            site=site,
            spec=spec,
            cut=cut,
            window=definition.window,
            season_years=common.SEASON_YEARS,
            qmode=definition.qmode,
        )
        predicted = predict_meter(cache=cache, meter=meter, index=index)
        replica = meter_replica_residuals(
            cache=cache,
            site=site,
            spec=spec,
            cut=cut,
            window=definition.window,
            season_years=common.SEASON_YEARS,
            qmode=definition.qmode,
            drop_switching=False,
        )
        return SiteRun(
            site=site,
            cut=cut,
            definition=definition,
            spec=spec,
            fit=meter.fit,
            meter=meter,
            pinned=None,
            index=index,
            mean=predicted.apparent_power,
            real_power=predicted.real_power,
            reactive_power=predicted.reactive_power,
            replica=replica,
        )
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    replica = replica_residuals(
        cache=cache,
        site=site,
        spec=replica_spec,
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
        capacity_vintages=arm_vintages,
        drop_switching=False,
    )
    return SiteRun(
        site=site,
        cut=cut,
        definition=definition,
        spec=spec,
        fit=fit,
        meter=None,
        pinned=pinned,
        index=index,
        mean=predict_gam(cache=cache, fit=fit, index=index),
        real_power=None,
        reactive_power=None,
        replica=replica,
    )


def kalman_builder(*, hyper: FilterHyperparameters) -> TraceBuilder:
    """Return the function that builds the Kalman correction trace from a residual stream."""

    def build(*, residual: np.ndarray, band: np.ndarray) -> CorrectionTrace:
        return run_kalman(residual=residual, band=band, hyper=hyper)

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
    hyper: FilterHyperparameters,
    days_ahead: int = 1,
    delay_hours: float = DEFAULT_DELAY_HOURS,
    lead_spread: LeadSpread | None = None,
) -> FilteredForecast:
    """Correct a run's mean from the readings usable at its day-ahead initialisation.

    Args:
        cache: The feature cache.
        run: The arm's run.
        hyper: The Kalman filter's hyperparameters.
        days_ahead: How many days before the valid day the forecast is made.
        delay_hours: The telemetry delay.
        lead_spread: The post-filter spread, built from the run's replica if not given.

    Returns:
        The corrected mean, its quantiles, and the correction.
    """
    build = kalman_builder(hyper=hyper)
    band = cache.calendar.band
    trace = build(residual=run.residual_stream(cache=cache), band=band)
    valid = cache.end_times[run.index]
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=day_ahead_initialisation(end_times=valid, days_ahead=days_ahead),
        delay_hours=delay_hours,
    )
    correction = correction_at(trace=trace, cutoff=cutoff, target=run.index, band=band[run.index])
    if lead_spread is None:
        lead_spread = post_filter_lead_spread(
            end_times=cache.end_times, replica=run.replica, run=build, delay_hours=delay_hours
        )
    corrected_mean = run.mean + correction.total
    quantiles = lead_spread.quantiles(
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


@dataclass(frozen=True)
class CapacityResult:
    """The fitted capacity of the last trained season-year, with its bootstrap range.

    Attributes:
        site: The substation's label.
        cut: The cut.
        arm: The arm.
        season_year: The last trained season-year.
        capacity: The capacity of the unweighted fit, in points.
        bootstrap: The capacity of each month-block weighting, in points.
        n_months: The number of calendar months in the training window.
        vintage_mean: The vintage's mean capacity over that season-year, in points; for D14 the
            capacity held after the cut, which is the plan's reference there.
        vintage_source: Where the vintage came from.
        paired_base_bootstrap: For a narrow-window arm, the base arm's capacity under the same
            month draws, so that a ratio of the two keeps the shared noise; `None` otherwise.
    """

    site: str
    cut: datetime
    arm: str
    season_year: int
    capacity: float
    bootstrap: np.ndarray
    n_months: int
    vintage_mean: float
    vintage_source: str
    paired_base_bootstrap: np.ndarray | None = None

    @property
    def range_90(self) -> tuple[float, float]:
        """The 2nd to the 19th of the 20 ordered values, widened by sqrt(m / (m - 1)).

        The widening is experiment 113's: the resample's spread understates the sampling noise of a
        window of `m` months, and it scales the distance of each end from the median.
        """
        ordered = np.sort(self.bootstrap)
        median = float(np.median(ordered))
        widen = float(np.sqrt(self.n_months / max(self.n_months - 1, 1)))
        return (
            median + widen * (float(ordered[1]) - median),
            median + widen * (float(ordered[-2]) - median),
        )


def stratified_month_weights(
    *, cache: FeatureCache, index: np.ndarray, n_resamples: int, seed: int
) -> np.ndarray:
    """Return month-block bootstrap weights that keep each calendar quarter's months.

    Each resample draws, within each quarter of the year, as many months as that quarter has in the
    window, with replacement (experiment 113's stratified bootstrap), so that every resample holds
    the same number of summers and winters.

    Args:
        cache: The cache.
        index: The training rows.
        n_resamples: The number of resamples.
        seed: Seeds the draws.

    Returns:
        Weights of shape (n_resamples, len(index)): how many times each row's month was drawn.
    """
    month = cache.end_times[index].astype("datetime64[M]")
    months, inverse = np.unique(month, return_inverse=True)
    quarter = (months.astype("datetime64[M]").astype(np.int64) % 12) // 3
    generator = np.random.default_rng(seed)
    counts = np.zeros((n_resamples, len(months)))
    for resample in range(n_resamples):
        for part in range(4):
            members = np.flatnonzero(quarter == part)
            if len(members):
                drawn = generator.choice(members, size=len(members), replace=True)
                counts[resample] += np.bincount(drawn, minlength=len(months))
    return counts[:, inverse]


def paired_weights(
    *, cache: FeatureCache, narrow: np.ndarray, wide: np.ndarray, n_resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return weights for two windows from the same month draws.

    The narrow window's months are drawn as in `stratified_month_weights`. The wide window uses
    those counts for the months it shares with the narrow window and a count of 1 for its earlier
    months, so the two fits share the noise of the shared months.

    Args:
        cache: The cache.
        narrow: The narrow window's training rows.
        wide: The wide window's training rows, which include the narrow window's months.
        n_resamples: The number of resamples.
        seed: Seeds the draws.

    Returns:
        The narrow window's weights and the wide window's weights.
    """
    narrow_weights = stratified_month_weights(
        cache=cache, index=narrow, n_resamples=n_resamples, seed=seed
    )
    narrow_month = cache.end_times[narrow].astype("datetime64[M]")
    wide_month = cache.end_times[wide].astype("datetime64[M]")
    wide_weights = np.ones((n_resamples, len(wide)))
    for month in np.unique(narrow_month):
        counts = narrow_weights[:, np.flatnonzero(narrow_month == month)[0]]
        wide_weights[:, wide_month == month] = counts[:, None]
    return narrow_weights, wide_weights


def vintage_season_mean(*, pinned: PinnedCapacity, season_year: int) -> float:
    """Return the mean weekly capacity of the vintage over a season-year, up to the cut.

    Args:
        pinned: The pinned capacity built for the cut.
        season_year: The year in which the July-to-June season-year begins.

    Returns:
        The mean of the weeks that start in the season-year and on or before the cut.
    """
    start = np.datetime64(f"{season_year}-07-01", "D")
    end = min(np.datetime64(f"{season_year + 1}-07-01", "D"), np.datetime64(pinned.cut.date(), "D"))
    inside = (pinned.weeks >= start) & (pinned.weeks < end)
    if not inside.any():
        return pinned.held_points
    return float(pinned.points[inside].mean())


def capacity_with_bootstrap(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    site: str,
    cut: datetime,
    definition: ArmDefinition,
    paired_base: ArmDefinition | None = None,
) -> CapacityResult:
    """Fit a fitted-capacity arm once and under 20 stratified month-block weightings.

    Args:
        cache: The feature cache.
        vintages: Experiment 45's real-time capacity vintages.
        site: The substation's label.
        cut: The cut.
        definition: A fitted-capacity arm.
        paired_base: A wider-window arm to fit under the same month draws, for a ratio.

    Returns:
        The capacity of the last trained season-year, with its bootstrap values and the vintage's
        mean over that season-year.
    """
    spec = definition.spec
    _, index = prepare_fit(
        cache=cache,
        site=site,
        spec=spec,
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
    )
    single = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
    )
    year = single.last_season_year
    paired_boot = None
    if paired_base is None:
        weights = stratified_month_weights(
            cache=cache,
            index=index,
            n_resamples=common.N_BOOTSTRAP_WEIGHTINGS,
            seed=common.BOOTSTRAP_SEED,
        )
    else:
        _, wide_index = prepare_fit(
            cache=cache,
            site=site,
            spec=paired_base.spec,
            cut=cut,
            window=paired_base.window,
            season_years=common.SEASON_YEARS,
        )
        weights, wide_weights = paired_weights(
            cache=cache,
            narrow=index,
            wide=wide_index,
            n_resamples=common.N_BOOTSTRAP_WEIGHTINGS,
            seed=common.BOOTSTRAP_SEED,
        )
        wide_batch = fit_gam(
            cache=cache,
            site=site,
            specs=[paired_base.spec],
            cut=cut,
            window=paired_base.window,
            season_years=common.SEASON_YEARS,
            weights=wide_weights,
        )
        paired_boot = np.array(
            [
                wide_batch.capacity_points(member=m)[wide_batch.last_season_year]
                for m in range(common.N_BOOTSTRAP_WEIGHTINGS)
            ]
        )
    batch = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
        weights=weights,
    )
    pinned = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)
    reference = (
        pinned.held_points
        if site == common.HELD_VINTAGE_SITE
        else vintage_season_mean(pinned=pinned, season_year=year)
    )
    return CapacityResult(
        site=site,
        cut=cut,
        arm=definition.name,
        season_year=year,
        capacity=single.capacity_points()[year],
        bootstrap=np.array(
            [batch.capacity_points(member=m)[year] for m in range(common.N_BOOTSTRAP_WEIGHTINGS)]
        ),
        n_months=len(np.unique(cache.end_times[index].astype("datetime64[M]"))),
        vintage_mean=reference,
        vintage_source=pinned.source,
        paired_base_bootstrap=paired_boot,
    )


def tuning_window() -> tuple[np.datetime64, np.datetime64]:
    """Return the half-open interval (start, end] of the tuning year's valid times."""
    return (
        common.to_64(moment=common.TUNING_CUT),
        common.to_64(moment=common.year_end(cut=common.TUNING_CUT)),
    )
