"""The arms of experiment 108: A0, A2, F, KF, the hindsight reference, and their variants.

Every arm corrects the same frozen A0 fits (`e00108_fits.py`), so every arm runs in seconds. For a
scored site `s` and a scored year with the fit made at its start:

- **A0** is the fit's mean with the replica's centred residual quantiles around it;
- **A2** adds the per-site two-state Kalman filter of experiment 106 on `r_s`, with the `a0`
  hyperparameters of experiment 106;
- **KF** adds a local filter on `e_s = r_s - beta_s sigma_{s,b} F_{-s}` and `beta_s sigma_{s,b}`
  times a shared filter on the leave-one-out factor `F_{-s}`;
- **F** keeps only the shared term; **KF-hind** replaces the shared filter's state with the factor
  at the target time, a hindsight reference that is never a forecast.

Each forecast sums its terms with `studies.dp_assimilation.summed_correction_at`, and its spread
is the replica run through the same sum (`post_filter_lead_spread`), so the two cannot drift apart.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial

import e00108_common as common
import e00108_factor as factor_lib
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    LEAD_BAND_UPPER_DAYS,
    N_LEAD_BANDS,
    Correction,
    CorrectionTrace,
    FilterHyperparameters,
    LeadSpread,
    SharedTerm,
    TraceBuilder,
    WeightedTrace,
    correction_at,
    day_ahead_initialisation,
    lead_band_of_days_ahead,
    post_filter_lead_spread,
    replica_corrections,
    run_kalman,
    summed_correction_at,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS, ReplicaResiduals, ResidualSpread

MAX_DAYS_AHEAD = LEAD_BAND_UPPER_DAYS[-1]
SAME_DAY_HOUR = 9


def kalman_builder(*, hyper: FilterHyperparameters) -> TraceBuilder:
    """Return the function that builds a Kalman trace with the given hyperparameters."""
    return partial(run_kalman, hyper=hyper)


def null_builder(*, residual: np.ndarray, band: np.ndarray) -> CorrectionTrace:
    """Return a trace with no readings, whose correction is zero everywhere."""
    empty = tuple(np.zeros(0, dtype=np.int64) for _ in range(N_BANDS))
    zeros = tuple(np.zeros(0) for _ in range(N_BANDS))
    return CorrectionTrace(
        kind="kalman", positions=empty, level=zeros, anomaly=zeros, phi=0.5, log_likelihood=0.0
    )


@dataclass(frozen=True)
class FactorSetup:
    """What turns the panel into a corrected forecast: loadings, hyperparameters, and factor.

    Attributes:
        name: A label for the setup (`real`, or a placebo draw).
        factor: For each scored site, the factor on the grid, after any shift or staleness.
        beta: Each site's shrunk loading on the factor, in units of the site's residual scale.
        local_hyper: Each site's hyperparameters for the local filter.
        shared_hyper: The one set of hyperparameters of the shared filter.
        shared_delay_hours: The telemetry delay of the contributors, when it is not the common
            delay; `None` for the common delay.
    """

    name: str
    factor: dict[str, np.ndarray]
    beta: dict[str, float]
    local_hyper: dict[str, FilterHyperparameters]
    shared_hyper: FilterHyperparameters
    shared_delay_hours: float | None = None


@dataclass(frozen=True)
class YearInputs:
    """A scored site's inputs for one scored year.

    Attributes:
        site: The substation's label.
        label: `tuning`, `replicate`, or `scored`.
        year: The year of the cut that starts the scored year.
        index: The grid rows of the scored year with a reading.
        mean: The A0 mean at `index`, in points.
        band: The hour band of each scored row.
        quarter: The calendar quarter of each scored row.
        replica: The year before's out-of-sample residuals, or `None` without a fit there.
        chain: The grid positions the filters run over for the scored year.
        replica_window: The grid positions of the replica year.
    """

    site: str
    label: str
    year: int
    index: np.ndarray
    mean: np.ndarray
    band: np.ndarray
    quarter: np.ndarray
    replica: ReplicaResiduals | None
    chain: slice
    replica_window: slice


def year_inputs(
    *,
    cache: FeatureCache,
    all_fits: dict[str, dict[int, factor_lib.SiteFit]],
    site: str,
    label: str,
) -> YearInputs | None:
    """Collect a site's inputs for a scored year, or `None` if it has no fit for the year."""
    year = common.SCORED_YEARS[label]
    site_fits = all_fits[site]
    if year not in site_fits:
        return None
    fit = site_fits[year]
    features = cache.sites[site]
    cut = common.cut_of(year=year)
    first_year = max(common.TUNING_YEAR, min(site_fits))
    chain = factor_lib.window_positions(
        cache=cache, after=common.cut_of(year=first_year), through=common.year_end(cut=cut)
    )
    replica_window = factor_lib.window_positions(
        cache=cache, after=common.cut_of(year=year - 1), through=cut
    )
    replica = None
    if year - 1 in site_fits:
        previous = site_fits[year - 1]
        index = previous.index
        replica = ReplicaResiduals(
            residuals=features.y[index] - previous.mean,
            band=cache.calendar.band[index],
            quarter=cache.calendar.quarter[index],
            index=index,
            fit_cut=common.cut_of(year=year - 1),
            switching=features.switching_mask[index],
        )
    return YearInputs(
        site=site,
        label=label,
        year=year,
        index=fit.index,
        mean=fit.mean,
        band=cache.calendar.band[fit.index],
        quarter=cache.calendar.quarter[fit.index],
        replica=replica,
        chain=chain,
        replica_window=replica_window,
    )


def uncorrected_quantiles(*, inputs: YearInputs) -> np.ndarray:
    """Return A0's quantiles: the mean plus the replica's centred residual quantiles."""
    assert inputs.replica is not None
    keep = ~inputs.replica.switching
    spread = ResidualSpread.fit(
        residuals=inputs.replica.residuals[keep],
        band=inputs.replica.band[keep],
        quarter=inputs.replica.quarter[keep],
        centre=True,
    )
    return spread.quantiles(mean=inputs.mean, band=inputs.band, quarter=inputs.quarter)


def split_stream(
    *,
    panel: factor_lib.Panel,
    site: str,
    beta: float,
    factor: np.ndarray,
) -> np.ndarray:
    """Return `e_s = r_s - beta sigma_{s,b} F_{-s}`, with the factor taken as zero where missing.

    Where the factor is missing the reading stays in the local stream, so the local filter loses
    no reading.
    """
    return panel.residual[site] - beta * panel.sigma[site] * np.nan_to_num(factor, nan=0.0)


def weights_by_band(
    *, all_fits: dict[str, dict[int, factor_lib.SiteFit]], site: str, year: int, beta: float
) -> np.ndarray:
    """Return `beta sigma_{s,b}` for the fit of `year`, in points per unit of the factor."""
    return beta * all_fits[site][year].sigma


@dataclass(frozen=True)
class ArmSpec:
    """How one filter arm builds its corrections.

    Attributes:
        name: The arm's name.
        local: `a0` for the experiment 106 `a0` filter on `r_s`, `split` for a filter on
            `e_s`, `split_frozen` for a filter on `e_s` with the `a0` hyperparameters, or `none`.
        shared: True if the arm has the shared term.
        hindsight: True for the reference that adds the factor at the target time instead.
        days: The numbers of days ahead written.
        same_day: True to also write the same-day row.
    """

    name: str
    local: str
    shared: bool
    hindsight: bool = False
    days: tuple[int, ...] = (1,)
    same_day: bool = False


ALL_DAYS = tuple(range(1, MAX_DAYS_AHEAD + 1))
A2_SPEC = ArmSpec("a2", local="a0", shared=False, days=ALL_DAYS, same_day=True)
F_SPEC = ArmSpec("f", local="none", shared=True, days=ALL_DAYS, same_day=True)
KF_SPEC = ArmSpec("kf", local="split", shared=True, days=ALL_DAYS, same_day=True)
KF_FROZEN_SPEC = ArmSpec("kf_frz", local="split_frozen", shared=True)
KF_HINDSIGHT_SPEC = ArmSpec("kf_hind", local="split", shared=False, hindsight=True)


def _sameday_rows(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Return a mask over `index` for targets from 09:30 UTC onward on the initialisation day."""
    start = cache.end_times[index] - np.timedelta64(30, "m")
    hour = (start - start.astype("datetime64[D]").astype("datetime64[us]")) / np.timedelta64(1, "h")
    return hour >= SAME_DAY_HOUR


def spread_from_corrections(
    *,
    replica: ReplicaResiduals,
    corrections: tuple[np.ndarray, ...],
    burn_in_days: int,
    end_times: np.ndarray,
) -> LeadSpread:
    """Fit the centred post-correction spread from one correction per number of days ahead."""
    valid_times = end_times[replica.index]
    kept = (
        valid_times > valid_times.min() + np.timedelta64(burn_in_days, "D")
    ) & ~replica.switching
    residuals: list[list[np.ndarray]] = [[] for _ in range(N_LEAD_BANDS)]
    bands: list[list[np.ndarray]] = [[] for _ in range(N_LEAD_BANDS)]
    quarters: list[list[np.ndarray]] = [[] for _ in range(N_LEAD_BANDS)]
    for days_ahead, correction in enumerate(corrections, start=1):
        lead_band = lead_band_of_days_ahead(days_ahead=days_ahead)
        residuals[lead_band].append((replica.residuals - correction)[kept])
        bands[lead_band].append(replica.band[kept])
        quarters[lead_band].append(replica.quarter[kept])
    return LeadSpread.fit(
        residuals=[np.concatenate(r) for r in residuals],
        band=[np.concatenate(b) for b in bands],
        quarter=[np.concatenate(q) for q in quarters],
    )


@dataclass(frozen=True)
class ArmTerms:
    """What an arm's corrections read, for one site and scored year.

    Attributes:
        run_local: Builds the local trace.
        chain_local: The local stream the filter runs over for the scored year.
        replica_local: The local stream over the replica year.
        run_shared: Builds the shared trace, or `None` for an arm without one.
        chain_shared: The factor over the scored year's chain.
        replica_shared: The factor over the replica year.
        weight_year: `beta sigma_{s,b}` for the scored year's fit.
        weight_replica: `beta sigma_{s,b}` for the replica's fit.
        shared_delay_hours: The shared term's own telemetry delay, or `None` for the common one.
    """

    run_local: TraceBuilder
    chain_local: np.ndarray
    replica_local: np.ndarray
    run_shared: TraceBuilder | None
    chain_shared: np.ndarray | None
    replica_shared: np.ndarray | None
    weight_year: np.ndarray | None
    weight_replica: np.ndarray | None
    shared_delay_hours: float | None


def arm_terms(
    *,
    cache: FeatureCache,
    all_fits: dict[str, dict[int, factor_lib.SiteFit]],
    panel: factor_lib.Panel,
    setup: FactorSetup | None,
    a0_hyper: dict[str, FilterHyperparameters],
    inputs: YearInputs,
    spec: ArmSpec,
) -> ArmTerms:
    """Collect the streams, filters, and weights an arm needs for one site and scored year."""
    site = inputs.site
    if spec.local == "a0":
        local_all = panel.residual[site]
        run_local = kalman_builder(hyper=a0_hyper[site])
    elif spec.local == "none":
        local_all = np.full(len(cache.end_times), np.nan)
        run_local = null_builder
    else:
        assert setup is not None
        local_all = split_stream(
            panel=panel, site=site, beta=setup.beta[site], factor=setup.factor[site]
        )
        hyper = a0_hyper[site] if spec.local == "split_frozen" else setup.local_hyper[site]
        run_local = kalman_builder(hyper=hyper)
    terms = ArmTerms(
        run_local=run_local,
        chain_local=factor_lib.keep_window(stream=local_all, window=inputs.chain),
        replica_local=factor_lib.keep_window(stream=local_all, window=inputs.replica_window),
        run_shared=None,
        chain_shared=None,
        replica_shared=None,
        weight_year=None,
        weight_replica=None,
        shared_delay_hours=None,
    )
    if setup is None:
        return terms
    factor = setup.factor[site]
    return replace(
        terms,
        run_shared=kalman_builder(hyper=setup.shared_hyper),
        chain_shared=factor_lib.keep_window(stream=factor, window=inputs.chain),
        replica_shared=factor_lib.keep_window(stream=factor, window=inputs.replica_window),
        weight_year=weights_by_band(
            all_fits=all_fits, site=site, year=inputs.year, beta=setup.beta[site]
        ),
        weight_replica=weights_by_band(
            all_fits=all_fits, site=site, year=inputs.year - 1, beta=setup.beta[site]
        ),
        shared_delay_hours=setup.shared_delay_hours,
    )


def arm_spread(
    *,
    cache: FeatureCache,
    inputs: YearInputs,
    terms: ArmTerms,
    spec: ArmSpec,
    hindsight_factor: np.ndarray | None,
) -> LeadSpread:
    """Fit an arm's spread: its replica run through the same sum of corrections."""
    assert inputs.replica is not None
    end_times = cache.end_times
    grid_band = cache.calendar.band
    if spec.local == "a0":
        return post_filter_lead_spread(
            end_times=end_times, replica=inputs.replica, run=terms.run_local
        )
    if spec.hindsight:
        assert terms.weight_replica is not None
        assert hindsight_factor is not None
        local_only = replica_corrections(
            end_times=end_times,
            replica=inputs.replica,
            run=terms.run_local,
            grid_band=grid_band,
            local_residual=terms.replica_local,
        )
        hind = terms.weight_replica[inputs.replica.band] * np.nan_to_num(
            hindsight_factor[inputs.replica.index], nan=0.0
        )
        return spread_from_corrections(
            replica=inputs.replica,
            corrections=tuple(c + hind for c in local_only),
            burn_in_days=14,
            end_times=end_times,
        )
    shared = []
    if spec.shared:
        assert terms.run_shared is not None
        assert terms.replica_shared is not None
        assert terms.weight_replica is not None
        shared = [
            SharedTerm(
                residual=terms.replica_shared,
                run=terms.run_shared,
                weight_by_band=terms.weight_replica,
                delay_hours=terms.shared_delay_hours,
            )
        ]
    return post_filter_lead_spread(
        end_times=end_times,
        replica=inputs.replica,
        run=terms.run_local,
        grid_band=grid_band,
        local_residual=terms.replica_local,
        shared=shared,
    )


@dataclass(frozen=True)
class ForecastCorrections:
    """The corrections of one set of forecast cases, split into local and shared parts."""

    local: Correction
    shared: Correction
    total: Correction


def forecast_corrections(
    *,
    cache: FeatureCache,
    inputs: YearInputs,
    terms: ArmTerms,
    spec: ArmSpec,
    hindsight_factor: np.ndarray | None,
    initialisation: np.ndarray,
) -> ForecastCorrections:
    """Return the corrections of the scored year's rows for the given initialisations."""
    grid_band = cache.calendar.band
    index, band = inputs.index, inputs.band
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times, initialisation=initialisation, delay_hours=common.DELAY_HOURS
    )
    local_trace = terms.run_local(residual=terms.chain_local, band=grid_band)
    weighted = []
    if spec.shared:
        assert terms.run_shared is not None
        assert terms.chain_shared is not None
        assert terms.weight_year is not None
        shared_cutoff = (
            None
            if terms.shared_delay_hours is None
            else usable_cutoff_positions(
                end_times=cache.end_times,
                initialisation=initialisation,
                delay_hours=terms.shared_delay_hours,
            )
        )
        weighted = [
            WeightedTrace(
                trace=terms.run_shared(residual=terms.chain_shared, band=grid_band),
                weight_by_band=terms.weight_year,
                cutoff=shared_cutoff,
            )
        ]
    total = summed_correction_at(
        local_trace=local_trace, shared_traces=weighted, cutoff=cutoff, target=index, band=band
    )
    local = correction_at(trace=local_trace, cutoff=cutoff, target=index, band=band)
    shared_level = np.zeros(len(index))
    shared_anomaly = np.zeros(len(index))
    for item in weighted:
        part = correction_at(
            trace=item.trace,
            cutoff=cutoff if item.cutoff is None else item.cutoff,
            target=index,
            band=band,
        )
        shared_level += item.weight_by_band[band] * part.level
        shared_anomaly += item.weight_by_band[band] * part.anomaly
    shared = Correction(
        total=shared_level + shared_anomaly, level=shared_level, anomaly=shared_anomaly
    )
    assert np.allclose(total.total, local.total + shared.total, atol=1e-9)
    if spec.hindsight:
        assert terms.weight_year is not None
        assert hindsight_factor is not None
        hind = terms.weight_year[band] * np.nan_to_num(hindsight_factor[index], nan=0.0)
        shared = Correction(total=hind, level=hind, anomaly=np.zeros_like(hind))
        total = Correction(
            total=local.total + hind, level=local.level + hind, anomaly=local.anomaly
        )
    return ForecastCorrections(local=local, shared=shared, total=total)


def filter_arm_frames(
    *,
    cache: FeatureCache,
    all_fits: dict[str, dict[int, factor_lib.SiteFit]],
    panel: factor_lib.Panel,
    setup: FactorSetup | None,
    a0_hyper: dict[str, FilterHyperparameters],
    inputs: YearInputs,
    spec: ArmSpec,
) -> dict[str, pl.DataFrame]:
    """Return a filter arm's prediction frames for one site and scored year.

    Args:
        cache: The feature cache.
        all_fits: The stored fits.
        panel: The residual panel.
        setup: The loadings, hyperparameters, and factor; `None` for the A2 arm, which has none.
        a0_hyper: Experiment 106's `a0` filter hyperparameters per site.
        inputs: The site's inputs for the year.
        spec: The arm.

    Returns:
        The frames by set suffix: `` for day 1, `_d<k>` for k = 2 to 7, `_sameday`.
    """
    terms = arm_terms(
        cache=cache,
        all_fits=all_fits,
        panel=panel,
        setup=setup,
        a0_hyper=a0_hyper,
        inputs=inputs,
        spec=spec,
    )
    hindsight_factor = None if setup is None else setup.factor[inputs.site]
    spread = arm_spread(
        cache=cache, inputs=inputs, terms=terms, spec=spec, hindsight_factor=hindsight_factor
    )
    end_times = cache.end_times
    index = inputs.index
    plans: list[tuple[str, int, np.ndarray, np.ndarray]] = [
        (
            "" if days_ahead == 1 else f"_d{days_ahead}",
            days_ahead,
            np.ones(len(index), dtype=bool),
            day_ahead_initialisation(end_times=end_times[index], days_ahead=days_ahead),
        )
        for days_ahead in spec.days
    ]
    if spec.same_day:
        plans.append(("_sameday", 0, *_sameday_plan(cache=cache, index=index)))
    frames: dict[str, pl.DataFrame] = {}
    for suffix, days_ahead, rows, init in plans:
        parts = forecast_corrections(
            cache=cache,
            inputs=inputs,
            terms=terms,
            spec=spec,
            hindsight_factor=hindsight_factor,
            initialisation=init,
        )
        corrected = inputs.mean + parts.total.total
        quantiles = spread.quantiles(
            mean=corrected,
            band=inputs.band,
            quarter=inputs.quarter,
            lead_band=lead_band_of_days_ahead(days_ahead=max(days_ahead, 1)),
        )
        frames[suffix] = common.prediction_frame(
            cache=cache,
            site=inputs.site,
            index=index[rows],
            quantiles=quantiles[rows],
            mean=corrected[rows],
            arm=spec.name,
            days_ahead=days_ahead,
            init=init[rows],
            columns={
                "local_level": parts.local.level[rows],
                "local_anomaly": parts.local.anomaly[rows],
                "shared_level": parts.shared.level[rows],
                "shared_anomaly": parts.shared.anomaly[rows],
            },
        )
    return frames


def _sameday_plan(*, cache: FeatureCache, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the same-day rows' mask over `index` and their 09:00 UTC initialisations."""
    start_day = (cache.end_times[index] - np.timedelta64(30, "m")).astype("datetime64[D]")
    init = start_day.astype("datetime64[us]") + np.timedelta64(SAME_DAY_HOUR, "h")
    return _sameday_rows(cache=cache, index=index), init


def a0_frame(*, cache: FeatureCache, inputs: YearInputs, same_day: bool) -> dict[str, pl.DataFrame]:
    """Return A0's frames: day 1, and optionally the same-day row."""
    quantiles = uncorrected_quantiles(inputs=inputs)
    frames = {
        "": common.prediction_frame(
            cache=cache,
            site=inputs.site,
            index=inputs.index,
            quantiles=quantiles,
            mean=inputs.mean,
            arm="a0",
        )
    }
    if same_day:
        rows, init = _sameday_plan(cache=cache, index=inputs.index)
        frames["_sameday"] = common.prediction_frame(
            cache=cache,
            site=inputs.site,
            index=inputs.index[rows],
            quantiles=quantiles[rows],
            mean=inputs.mean[rows],
            arm="a0",
            days_ahead=0,
            init=init[rows],
        )
    return frames


def a0_hyper_from_part4() -> dict[str, FilterHyperparameters]:
    """Return experiment 106's `a0` filter hyperparameters per site (D9 is the mean of five)."""
    import json

    tuned = json.loads((common.PART4_DIR / "work" / "tuned.json").read_text())
    return {
        site: FilterHyperparameters.from_array(values=np.array(values))
        for site, values in tuned["filter_hyper"]["a0"].items()
    }


def start_of(*, moment: datetime) -> np.datetime64:
    """Return a datetime as `datetime64[us]`."""
    return common.to_64(moment=moment)


__all__ = [
    "A2_SPEC",
    "F_SPEC",
    "KF_FROZEN_SPEC",
    "KF_HINDSIGHT_SPEC",
    "KF_SPEC",
    "ArmSpec",
    "FactorSetup",
    "YearInputs",
    "a0_frame",
    "a0_hyper_from_part4",
    "filter_arm_frames",
    "replace",
    "year_inputs",
]
