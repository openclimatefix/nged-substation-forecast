"""A simple curtailment-cost metric for ranking quantile forecasts of net export.

NGED's main use of these forecasts is reducing the curtailment of generators. The metric is a
simplified version of Metric 2, Tiers 1 and 2, of the cost-savings metrics roadmap, restricted to
one series at a time. The metric is meant to fall as a forecast improves in the tails. For one
half-hour with export limit `L`:

- The volume curtailed in advance is `V = max(0, q_tau - L) x 0.5 h`, where `q_tau` is the
  forecast's `tau` quantile of net export. Over-forecasting export wastes energy here.
- The volume that needed curtailing is `N = max(0, actual - L) x 0.5 h`, from the measured net
  export. Under-forecasting export leaves `max(0, N - V)` unmet, a risk of breaching the limit.
- The cost is `p x V + p_breach x max(0, N - V)`. Charging the unmet energy is what keeps a forecast
  from winning by procuring little. With `p_breach = p / (1 - tau)`, the total equals `p x N` plus
  `p / (1 - tau)` times the pinball loss of the threshold-censored forecast, so the score is
  minimised in expectation by the true quantile and the first term is the same for every arm.
  `CurtailmentAssumptions.breach_to_curtail_price_ratio` overrides that ratio.
- The unmet share, `sum max(0, N - V) / sum N`, is reported beside the cost.

**Every number is a ranking instrument, never a cost analysis.** The module holds no real
curtailment price and no real export limit. Each default below is an assumption, because no real
value is available, and all of them sit in `CurtailmentAssumptions`:

- The price is GBP 100 per MWh, the whole-system figure the roadmap uses.
- The limit is the `limit_percentile` (99) of each series' own net export in the calibration
  window, times `limit_scale`. A series whose p99 limit is not above zero never exports at that
  percentile, so it is not export-constrained and is left out of the scoring. The limit is a crude
  stand-in. NGED's limits sit at the next substation up for about half of generators, and are often
  set on circuits by power-flow and N-1 contingency analysis (simulate a fault, run a power flow,
  simulate the next credible fault) and bind only after a fault. `limit_scale` below 1 mimics the
  smaller headroom after a fault, and is an assumption to sweep, not a result.
- Each series stands alone. There is no netting across parallel Primary substations (the caveat of
  Tier 1) and no substation hierarchy (Tier 2). The functions work on any series, so a BSP or GSP
  series is scored in the same way as a Primary substation or a generator.
- Only rows whose forecast lead is at least 12 and under 36 hours (`decision_lead_hours`, a
  day-ahead decision) are used, so that one valid time is counted about once, not once for each
  of the many overlapping runs. Rows need a `lead_hours` column; pass `None` to use every row.
- Tau is fixed per arm (for example 0.95), or calibrated pooled over the export-constrained series
  so that the unmet share hits `unmet_target`, on rows whose outcome falls before
  `calibration_end`. The arms are scored on rows whose forecast was made at or after
  `calibration_end`. The stored prediction sets hold no training-window forecasts. They also stop
  at q95, so a 5% unmet target is usually out of reach and tau is clamped, which
  `CalibratedTau.clamped` reports.
- A quantile between two delivered levels is interpolated linearly.

The input frames are one arm's forecast rows, as `studies.experiment_store.read_predictions`
returns them for the demand experiments: `time_series_id`, `power_fcst_init_time`, `valid_time`,
`seed`, `actual`, and a `q<percent>` column per delivered level, all in one power unit. When that
unit is a fraction of capacity, the costs are GBP per MW of capacity.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, Literal

import numpy as np
import polars as pl

from studies.bootstrap import AbsoluteInterval, BootstrapInterval, bootstrap_month_totals
from studies.experiment_store import read_predictions

HOURS_PER_ROW: Final[float] = 0.5
"""The length of one forecast row in hours; a power in MW over a row is 0.5 x power in MWh."""

ROWS_PER_DAY: Final[int] = 48
"""Half-hourly rows in a day."""

DAYS_PER_YEAR: Final[float] = 365.25
"""Days that annualise a per-day cost."""

FULL_YEAR_DAYS: Final[float] = 350.0
"""A scored span of at least this many days counts as a full year, so a per-year cost is shown."""

QUANTILE_COLUMN_PATTERN: Final[re.Pattern[str]] = re.compile(r"^q(\d{2})$")
"""Matches the name of a quantile column, such as `q05`."""

BISECTION_STEPS: Final[int] = 50
"""How many halvings narrow tau, far below the precision of a quantile at a few hundred rows."""

PERFECT_FORECAST: Final[str] = "perfect_forecast"
"""The name of the reference row that procures exactly the unmet-adjusted needed volume."""

ROW_KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time", "seed")
"""The columns that identify one forecast row of one fitting seed."""

PowerUnitType = Literal["MW", "capacity_fraction"]
"""The unit of the power columns: megawatts, or a fraction of each series' capacity."""


@dataclass(frozen=True)
class CurtailmentAssumptions:
    """Every assumption behind the metric, in one place so that a sweep can vary them.

    Attributes:
        calibration_end: Rows whose outcome (`valid_time`) is before this instant calibrate tau and
            the limit. Rows whose forecast (`power_fcst_init_time`) is at or after it are scored.
        price_gbp_per_mwh: The cost of a MWh curtailed in advance. An assumption, not a real price.
        limit_percentile: The percentile of each series' calibration-window net export that is its
            export limit. An assumption, not a real limit.
        unmet_target: The share of needed energy left unmet that tau is calibrated to.
        limit_scale: Multiplies every series' limit, after the p99 decides which series are
            export-constrained. 1.0 is the percentile itself; below 1.0 mimics the smaller headroom
            after a fault (an assumption, not an N-1 study).
        breach_to_curtail_price_ratio: The price of a MWh left unmet divided by the price of a MWh
            curtailed in advance. None sets it to `1 / (1 - tau)` for the arm's tau.
        decision_lead_hours: The lowest forecast lead, in hours, of the rows used, and the lead they
            stay below, or None to use every row. The upper end is excluded, so that with daily runs
            a valid time counts once. The rows then need a `lead_hours` column.
        export_is_negative: True when the power columns are demand (positive towards end-users, as
            for Primary, BSP, and GSP substations), so net export is the negated value. False when
            the columns are generation or a customer meter, where positive is export.
        power_unit: The unit of the power columns, which sets the unit of the costs.
    """

    calibration_end: datetime
    price_gbp_per_mwh: float = 100.0
    limit_percentile: float = 99.0
    unmet_target: float = 0.05
    limit_scale: float = 1.0
    breach_to_curtail_price_ratio: float | None = None
    decision_lead_hours: tuple[float, float] | None = (12.0, 36.0)
    export_is_negative: bool = True
    power_unit: PowerUnitType = "capacity_fraction"


@dataclass(frozen=True)
class CalibratedTau:
    """The procurement quantile chosen for one arm.

    Attributes:
        tau: The quantile of net export that the arm procures against.
        calibration_unmet: The unmet share of needed energy at that tau on the calibration rows.
        clamped: True when the target unmet share lay beyond the delivered quantile levels, so tau
            stopped at the extreme level and the unmet share is not the target.
    """

    tau: float
    calibration_unmet: float
    clamped: bool


@dataclass(frozen=True)
class CurtailmentResult:
    """The metric for several arms on the same rows.

    Attributes:
        taus: One row per arm: `arm`, `tau`, `calibration_unmet`, `clamped`.
        per_series: One row per arm and series: `arm`, `time_series_id`, `limit`, `gbp_per_day`,
            `gbp_per_year` (null unless the scored span is a full year), `unmet_fraction`
            (null when no energy needed curtailing), and `tau_coverage`.
        per_arm: One row per arm, and the perfect-forecast reference (which procures exactly the
            needed energy): `arm`, `tau`, `gbp_per_day` (the total cost, the mean over rows, so over
            series), `procure_gbp_per_day` (the part spent on curtailing in advance),
            `gbp_per_year`, `unmet_fraction`, `tau_coverage`, and `n_series`. The reference has a
            null `tau` and `tau_coverage`.
        month_totals: One row per arm, fitting seed, and month of the forecast's initialisation:
            `arm`, `seed`, `month`, `rows`, and `cost_sum` (the sum of the per-row cost per day).
            `cost_intervals` resamples these totals.
        excluded_series: The series whose limit is not above zero, so they were not scored.
    """

    taus: pl.DataFrame
    per_series: pl.DataFrame
    per_arm: pl.DataFrame
    month_totals: pl.DataFrame
    excluded_series: tuple[int, ...]


def quantile_levels(*, frame: pl.DataFrame) -> list[tuple[float, str]]:
    """Return the delivered quantile levels of a frame with their column names, lowest first.

    Args:
        frame: A forecast frame with `q<percent>` columns.

    Returns:
        Each level as a fraction with its column name.

    Raises:
        ValueError: If fewer than two quantile columns are present, because a quantile between
            levels cannot be interpolated from one.
    """
    found = sorted(
        (int(match[1]) / 100.0, name)
        for name in frame.columns
        if (match := QUANTILE_COLUMN_PATTERN.match(name))
    )
    if len(found) < 2:
        msg = f"need at least two quantile columns named like q05, found {[n for _, n in found]}"
        raise ValueError(msg)
    return found


def align_arms(*, arms: Mapping[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Keep only the rows that every arm holds and that have a finite measured value.

    Two arms are comparable only on the same rows, and a NaN measurement would turn a limit or a
    sum of needed energy into NaN without an error. The caller can compare the heights before and
    after to report how many rows were dropped.

    Args:
        arms: Each arm's forecast rows, with `ROW_KEYS` and `actual`.

    Returns:
        Each arm's rows restricted to the keys all arms share with a finite `actual` in every arm,
        each arm in its own row order.
    """
    names = list(arms)
    shared = arms[names[0]].filter(pl.col("actual").is_finite()).select(ROW_KEYS).unique()
    for name in names[1:]:
        finite_keys = arms[name].filter(pl.col("actual").is_finite()).select(ROW_KEYS).unique()
        shared = shared.join(finite_keys, on=list(ROW_KEYS), how="inner")
    return {
        name: frame.join(shared, on=list(ROW_KEYS), how="inner", maintain_order="left")
        for name, frame in arms.items()
    }


def _export_sign(*, assumptions: CurtailmentAssumptions) -> float:
    """Return -1 when net export is the negated value, and +1 when it is the value itself."""
    return -1.0 if assumptions.export_is_negative else 1.0


def export_quantile(
    *, frame: pl.DataFrame, tau: float, assumptions: CurtailmentAssumptions
) -> np.ndarray:
    """Return each row's forecast `tau` quantile of net export, interpolating between levels.

    Args:
        frame: Forecast rows with `q<percent>` columns in the power unit.
        tau: The quantile of net export, between 0 and 1. A tau beyond the delivered levels is
            clamped to the extreme delivered level.
        assumptions: Says whether net export is the negated value.

    Returns:
        The quantile of net export per row. When net export is the negated value, its `tau`
        quantile is the negated `1 - tau` quantile of the value.
    """
    levels = quantile_levels(frame=frame)
    values = np.sort(frame.select(name for _, name in levels).to_numpy().astype(np.float64), axis=1)
    sign = _export_sign(assumptions=assumptions)
    target = tau if sign > 0 else 1.0 - tau
    grid = np.array([level for level, _ in levels])
    target = float(np.clip(target, grid[0], grid[-1]))
    upper = int(np.clip(np.searchsorted(grid, target), 1, len(grid) - 1))
    weight = (target - grid[upper - 1]) / (grid[upper] - grid[upper - 1])
    value = (1.0 - weight) * values[:, upper - 1] + weight * values[:, upper]
    return sign * value


def tau_bounds(*, frame: pl.DataFrame, assumptions: CurtailmentAssumptions) -> tuple[float, float]:
    """Return the lowest and highest tau that the delivered quantile levels reach.

    Args:
        frame: Forecast rows with `q<percent>` columns.
        assumptions: Says whether net export is the negated value, which flips the levels.

    Returns:
        The lowest and highest tau, as fractions.
    """
    levels = [level for level, _ in quantile_levels(frame=frame)]
    if assumptions.export_is_negative:
        return 1.0 - levels[-1], 1.0 - levels[0]
    return levels[0], levels[-1]


def export_limits(*, actuals: pl.DataFrame, assumptions: CurtailmentAssumptions) -> pl.DataFrame:
    """Return each series' export limit: a percentile of its net export before `calibration_end`.

    Args:
        actuals: Rows with `time_series_id`, `valid_time`, and `actual`. A valid time that several
            forecast runs cover counts once, so overlapping runs do not weight the percentile.
        assumptions: Gives the percentile, the calibration end, and the sign of net export.

    Returns:
        One row per series: `time_series_id`, `limit` (the percentile of net export, in the power
        unit, times `limit_scale`), and `export_constrained` (true when the unscaled limit is
        above zero).
    """
    sign = _export_sign(assumptions=assumptions)
    return (
        actuals.filter(pl.col("valid_time") < assumptions.calibration_end)
        .unique(subset=["time_series_id", "valid_time"])
        .group_by("time_series_id")
        .agg(limit=(sign * pl.col("actual")).quantile(assumptions.limit_percentile / 100.0))
        .with_columns(export_constrained=pl.col("limit") > 0.0)
        .with_columns(limit=pl.col("limit") * assumptions.limit_scale)
        .sort("time_series_id")
    )


def _volumes(
    *, frame: pl.DataFrame, limits: pl.DataFrame, tau: float, assumptions: CurtailmentAssumptions
) -> pl.DataFrame:
    """Add the limit, procured and needed energy, and the tau coverage to each row.

    The returned frame keeps the input rows in order, and adds `limit`, `procured`, `needed`,
    `unmet` (the energy needed but not procured), and `covered` (1.0 where the actual net export
    was at or below the forecast tau quantile). Energies are in the power unit times hours.
    """
    sign = _export_sign(assumptions=assumptions)
    joined = frame.join(limits.select("time_series_id", "limit"), on="time_series_id", how="left")
    forecast_export = export_quantile(frame=joined, tau=tau, assumptions=assumptions)
    actual_export = sign * joined["actual"].to_numpy().astype(np.float64)
    limit = joined["limit"].to_numpy()
    procured = np.maximum(0.0, forecast_export - limit) * HOURS_PER_ROW
    needed = np.maximum(0.0, actual_export - limit) * HOURS_PER_ROW
    return joined.with_columns(
        procured=pl.Series(procured),
        needed=pl.Series(needed),
        unmet=pl.Series(np.maximum(0.0, needed - procured)),
        covered=pl.Series((actual_export <= forecast_export).astype(np.float64)),
    )


def unmet_fraction(*, volumes: pl.DataFrame) -> float | None:
    """Return `sum unmet / sum needed` over the rows, or None when nothing needed curtailing.

    Args:
        volumes: A frame from `_volumes`, holding `needed` and `unmet`.

    Returns:
        The unmet share of needed energy, or None because the share is undefined at zero need.
    """
    needed = float(volumes["needed"].sum())
    return float(volumes["unmet"].sum()) / needed if needed > 0.0 else None


def calibrate_tau(
    *, calibration: pl.DataFrame, limits: pl.DataFrame, assumptions: CurtailmentAssumptions
) -> CalibratedTau:
    """Find the tau whose unmet share on the calibration rows is the target.

    The unmet share falls as tau rises, so bisection finds the lowest tau that reaches the target.

    Args:
        calibration: One arm's rows on the export-constrained series, with outcomes before the
            calibration end.
        limits: The export limits of those series, from `export_limits`.
        assumptions: Gives the target unmet share.

    Returns:
        The calibrated tau, its unmet share, and whether tau stopped at an extreme level.

    Raises:
        ValueError: If no calibration row needed any curtailment, so no tau can be calibrated.
    """
    low, high = tau_bounds(frame=calibration, assumptions=assumptions)

    def unmet_at(*, tau: float) -> float | None:
        return unmet_fraction(
            volumes=_volumes(frame=calibration, limits=limits, tau=tau, assumptions=assumptions)
        )

    unmet_high = unmet_at(tau=high)
    if unmet_high is None:
        msg = "no calibration row exceeds the limit, so tau cannot be calibrated"
        raise ValueError(msg)
    if unmet_high > assumptions.unmet_target:
        return CalibratedTau(tau=high, calibration_unmet=unmet_high, clamped=True)
    unmet_low = unmet_at(tau=low)
    if unmet_low is not None and unmet_low <= assumptions.unmet_target:
        return CalibratedTau(tau=low, calibration_unmet=unmet_low, clamped=True)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        unmet_middle = unmet_at(tau=middle)
        if unmet_middle is not None and unmet_middle > assumptions.unmet_target:
            low = middle
        else:
            high = middle
    return CalibratedTau(tau=high, calibration_unmet=unmet_at(tau=high) or 0.0, clamped=False)


def _breach_ratio(*, tau: float, assumptions: CurtailmentAssumptions) -> float:
    """Return the price of a MWh left unmet over the price of a MWh curtailed in advance."""
    if assumptions.breach_to_curtail_price_ratio is not None:
        return assumptions.breach_to_curtail_price_ratio
    return 1.0 / (1.0 - tau)


def curtailment_cost_for_arms(
    *,
    arms: Mapping[str, pl.DataFrame],
    assumptions: CurtailmentAssumptions,
    fixed_tau: Mapping[str, float] | None = None,
) -> CurtailmentResult:
    """Score several arms' quantile forecasts with the curtailment-cost metric.

    Args:
        arms: Each arm's name mapped to its forecast rows, with `ROW_KEYS`, `actual`, and
            `q<percent>` columns, as `align_arms` returns them: the same rows in every arm and a
            finite `actual`.
        assumptions: The price, the limit, the target, and the calibration split.
        fixed_tau: Arms to score at a stated tau instead of a calibrated one, such as the manual
            heuristic at its operating point of 0.95. Defaults to none.

    Returns:
        The taus, the per-series and per-arm tables, and the per-row costs.

    Raises:
        ValueError: If `actual` is not finite, `decision_lead_hours` is set but `lead_hours` is
            missing, no series is export-constrained, an arm has no rows
            on one side of the split, or the arms do not hold the same rows.
    """
    fixed = dict(fixed_tau or {})
    first = next(iter(arms.values()))
    if not all(frame["actual"].is_finite().all() for frame in arms.values()):
        msg = "the measured value has NaN or null rows; call align_arms first"
        raise ValueError(msg)
    if len({frame.height for frame in arms.values()}) > 1:
        msg = "the arms hold different numbers of rows; call align_arms first"
        raise ValueError(msg)
    limits = export_limits(actuals=first, assumptions=assumptions)
    constrained = limits.filter(pl.col("export_constrained"))
    if constrained.is_empty():
        msg = "no series is export-constrained: every limit is at or below zero"
        raise ValueError(msg)
    excluded = tuple(
        limits.filter(~pl.col("export_constrained"))["time_series_id"].sort().to_list()
    )
    constrained_ids = constrained["time_series_id"].to_list()
    if assumptions.decision_lead_hours is not None:
        low_lead, high_lead = assumptions.decision_lead_hours
        if "lead_hours" not in first.columns:
            msg = "decision_lead_hours needs a lead_hours column; pass None to use every row"
            raise ValueError(msg)
        arms = {
            arm: frame.filter(
                (pl.col("lead_hours") >= low_lead) & (pl.col("lead_hours") < high_lead)
            )
            for arm, frame in arms.items()
        }
    taus: list[dict[str, object]] = []
    scored: dict[str, pl.DataFrame] = {}
    for arm, frame in arms.items():
        kept = frame.filter(pl.col("time_series_id").is_in(constrained_ids))
        calibration = kept.filter(pl.col("valid_time") < assumptions.calibration_end)
        scoring = kept.filter(pl.col("power_fcst_init_time") >= assumptions.calibration_end)
        if calibration.is_empty() or scoring.is_empty():
            msg = f"arm {arm!r} has no rows on one side of the calibration end"
            raise ValueError(msg)
        if arm in fixed:
            chosen = CalibratedTau(tau=fixed[arm], calibration_unmet=float("nan"), clamped=False)
        else:
            chosen = calibrate_tau(
                calibration=calibration, limits=constrained, assumptions=assumptions
            )
        taus.append(
            {
                "arm": arm,
                "tau": chosen.tau,
                "calibration_unmet": chosen.calibration_unmet,
                "clamped": chosen.clamped,
            }
        )
        scored[arm] = (
            _volumes(frame=scoring, limits=constrained, tau=chosen.tau, assumptions=assumptions)
            .with_columns(
                arm=pl.lit(arm),
                procure_gbp_per_day=pl.col("procured")
                * assumptions.price_gbp_per_mwh
                * ROWS_PER_DAY,
                cost_gbp_per_day=(
                    pl.col("procured")
                    + pl.col("unmet") * _breach_ratio(tau=chosen.tau, assumptions=assumptions)
                )
                * assumptions.price_gbp_per_mwh
                * ROWS_PER_DAY,
            )
            .select(
                *ROW_KEYS,
                "arm",
                "limit",
                "procured",
                "needed",
                "unmet",
                "covered",
                "procure_gbp_per_day",
                "cost_gbp_per_day",
            )
        )
    return _tables(
        scored=scored,
        taus=pl.DataFrame(
            taus,
            schema={
                "arm": pl.String,
                "tau": pl.Float64,
                "calibration_unmet": pl.Float64,
                "clamped": pl.Boolean,
            },
        ),
        assumptions=assumptions,
        excluded=excluded,
    )


def _tables(
    *,
    scored: Mapping[str, pl.DataFrame],
    taus: pl.DataFrame,
    assumptions: CurtailmentAssumptions,
    excluded: tuple[int, ...],
) -> CurtailmentResult:
    """Build the per-series, per-arm, and per-row tables from the scored rows."""
    everything = pl.concat(scored.values())
    span_seconds = everything.select(
        (pl.col("valid_time").max() - pl.col("valid_time").min()).dt.total_seconds()
    ).item()
    full_year = span_seconds / 86400.0 >= FULL_YEAR_DAYS
    per_series = (
        everything.group_by("arm", "time_series_id", maintain_order=True)
        .agg(
            limit=pl.col("limit").first(),
            gbp_per_day=pl.col("cost_gbp_per_day").mean(),
            needed=pl.col("needed").sum(),
            unmet=pl.col("unmet").sum(),
            tau_coverage=pl.col("covered").mean(),
        )
        .with_columns(
            gbp_per_year=pl.when(full_year).then(pl.col("gbp_per_day") * DAYS_PER_YEAR),
            unmet_fraction=pl.when(pl.col("needed") > 0.0).then(pl.col("unmet") / pl.col("needed")),
        )
        .select(
            "arm",
            "time_series_id",
            "limit",
            "gbp_per_day",
            "gbp_per_year",
            "unmet_fraction",
            "tau_coverage",
        )
    )
    per_arm = (
        everything.group_by("arm", maintain_order=True)
        .agg(
            gbp_per_day=pl.col("cost_gbp_per_day").mean(),
            procure_gbp_per_day=pl.col("procure_gbp_per_day").mean(),
            needed=pl.col("needed").sum(),
            unmet=pl.col("unmet").sum(),
            tau_coverage=pl.col("covered").mean(),
            n_series=pl.col("time_series_id").n_unique(),
        )
        .join(taus.select("arm", "tau"), on="arm")
        .with_columns(
            gbp_per_year=pl.when(full_year).then(pl.col("gbp_per_day") * DAYS_PER_YEAR),
            unmet_fraction=pl.when(pl.col("needed") > 0.0).then(pl.col("unmet") / pl.col("needed")),
        )
    )
    first = next(iter(scored.values()))
    perfect_per_day = (
        float(first["needed"].sum()) * assumptions.price_gbp_per_mwh * ROWS_PER_DAY / first.height
    )
    perfect = pl.DataFrame(
        {
            "arm": [PERFECT_FORECAST],
            "gbp_per_day": [perfect_per_day],
            "procure_gbp_per_day": [perfect_per_day],
            "unmet_fraction": [0.0],
            "n_series": [first["time_series_id"].n_unique()],
        },
        schema_overrides={"n_series": pl.UInt32},
    ).with_columns(
        gbp_per_year=pl.when(full_year).then(pl.col("gbp_per_day") * DAYS_PER_YEAR),
    )
    per_arm = pl.concat(
        [
            per_arm.select(
                "arm",
                "tau",
                "gbp_per_day",
                "procure_gbp_per_day",
                "gbp_per_year",
                "unmet_fraction",
                "tau_coverage",
                "n_series",
            ),
            perfect.select(
                "arm",
                tau=pl.lit(None, dtype=pl.Float64),
                gbp_per_day="gbp_per_day",
                procure_gbp_per_day="procure_gbp_per_day",
                gbp_per_year="gbp_per_year",
                unmet_fraction="unmet_fraction",
                tau_coverage=pl.lit(None, dtype=pl.Float64),
                n_series="n_series",
            ),
        ]
    )
    month_totals = (
        everything.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        .group_by("arm", "seed", "month")
        .agg(rows=pl.len(), cost_sum=pl.col("cost_gbp_per_day").sum())
        .sort("arm", "seed", "month")
    )
    return CurtailmentResult(
        taus=taus,
        per_series=per_series,
        per_arm=per_arm,
        month_totals=month_totals,
        excluded_series=excluded,
    )


def _arm_month_arrays(*, totals: pl.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return one arm's cost sums, shape (n_seeds, n_months), and its row counts per month."""
    rows = totals.filter(pl.col("arm") == arm)
    sums = rows.pivot(on="month", index="seed", values="cost_sum").sort("seed").drop("seed")
    count_by_seed = rows.pivot(on="month", index="seed", values="rows").sort("seed").drop("seed")
    counts = count_by_seed.select(sorted(count_by_seed.columns)).to_numpy().astype(np.float64)
    if np.isnan(counts).any() or not (counts == counts[0]).all():
        msg = f"arm {arm!r} does not hold the same rows in every month for every seed"
        raise ValueError(msg)
    return sums.select(sorted(sums.columns)).to_numpy(), counts[0]


def cost_intervals(
    *, result: CurtailmentResult, reference: str
) -> tuple[dict[str, AbsoluteInterval], dict[str, BootstrapInterval]]:
    """Interval each arm's cost, and its difference from a reference arm, by resampling months.

    Args:
        result: From `curtailment_cost_for_arms`.
        reference: The arm that the differences are taken from.

    Returns:
        Each arm's absolute interval, and each other arm's cost minus the reference's with its
        paired interval. A negative difference means the arm costs less than the reference. The
        resampling draws whole months of forecast initialisation and a fitting seed, as
        `studies.bootstrap` does.
    """
    arm_names = result.month_totals["arm"].unique(maintain_order=True).to_list()
    arrays = {arm: _arm_month_arrays(totals=result.month_totals, arm=arm) for arm in arm_names}
    reference_sums, reference_counts = arrays[reference]
    absolute: dict[str, AbsoluteInterval] = {}
    paired: dict[str, BootstrapInterval] = {}
    for arm, (sums, counts) in arrays.items():
        lower, upper = bootstrap_month_totals(month_sums=sums, month_counts=counts)
        per_seed = sums.sum(axis=1) / counts.sum()
        absolute[arm] = {
            "value": float(per_seed.mean()),
            "lower_95": lower,
            "upper_95": upper,
            "seed_spread": float(per_seed.std()),
            "n_rows": int(counts.sum()),
            "n_months": len(counts),
        }
        if arm == reference:
            continue
        if not np.array_equal(counts, reference_counts):
            msg = f"arm {arm!r} and the reference hold different rows in some month"
            raise ValueError(msg)
        difference = sums - reference_sums
        lower, upper = bootstrap_month_totals(month_sums=difference, month_counts=counts)
        per_seed_difference = difference.sum(axis=1) / counts.sum()
        paired[arm] = {
            "difference": float(per_seed_difference.mean()),
            "lower_95": lower,
            "upper_95": upper,
            "seed_spread": float(per_seed_difference.std()),
            "n_rows": int(counts.sum()),
            "n_months": len(counts),
        }
    return absolute, paired


def sweep_assumptions(
    *,
    arms: Mapping[str, pl.DataFrame],
    base: CurtailmentAssumptions,
    limit_scales: Sequence[float] = (1.0, 0.75, 0.5),
    breach_ratios: Sequence[float] = (10.0, 20.0),
    fixed_tau: Mapping[str, float] | None = None,
) -> pl.DataFrame:
    """Score the arms under every limit scale and breach ratio, to show how stable the ranks are.

    The set of scored series is decided once by the base assumptions' percentile, so a lower limit
    scale tightens the headroom of the same series instead of changing which series are scored. The
    price is not swept, because it multiplies every arm's cost by the same factor.

    Args:
        arms: As for `curtailment_cost_for_arms`.
        base: The assumptions every other field is taken from.
        limit_scales: The headroom factors to try; below 1.0 mimics the smaller headroom after a
            fault, which is an assumption and not an N-1 study.
        breach_ratios: The prices of a MWh left unmet over a MWh curtailed to try.
        fixed_tau: As for `curtailment_cost_for_arms`.

    Returns:
        One row per combination and arm: `limit_scale`, `breach_ratio`, `arm`, `gbp_per_day`,
        `unmet_fraction`, `rank` (1 is cheapest, ties broken by arm order), and `n_series`.
    """
    frames = []
    for scale in limit_scales:
        for ratio in breach_ratios:
            assumptions = replace(base, limit_scale=scale, breach_to_curtail_price_ratio=ratio)
            result = curtailment_cost_for_arms(
                arms=arms, assumptions=assumptions, fixed_tau=fixed_tau
            )
            frames.append(
                result.per_arm.filter(pl.col("arm") != PERFECT_FORECAST)
                .select("arm", "gbp_per_day", "unmet_fraction", "n_series")
                .with_columns(
                    limit_scale=pl.lit(scale),
                    breach_ratio=pl.lit(ratio),
                    rank=pl.col("gbp_per_day").rank(method="ordinal"),
                )
            )
    return pl.concat(frames).select(
        "limit_scale", "breach_ratio", "arm", "gbp_per_day", "unmet_fraction", "rank", "n_series"
    )


def read_arms(
    *, experiment_id: int, names: Sequence[str], row_set_id: str
) -> dict[str, pl.DataFrame]:
    """Read named prediction sets of one experiment, ready for `curtailment_cost_for_arms`.

    Args:
        experiment_id: The experiment that wrote the sets.
        names: The sets' names, which become the arm names.
        row_set_id: The row set the sets must cover.

    Returns:
        Each arm's forecast rows.
    """
    return {
        name: read_predictions(experiment_id=experiment_id, name=name, row_set_id=row_set_id)
        for name in names
    }


def cost_markdown(
    *,
    result: CurtailmentResult,
    intervals: tuple[dict[str, AbsoluteInterval], dict[str, BootstrapInterval]] | None = None,
) -> str:
    """Render the per-arm table for a `result.md`.

    Args:
        result: From `curtailment_cost_for_arms`.
        intervals: The output of `cost_intervals`; when given, each arm shows its 95% interval.

    Returns:
        A markdown table with one row per arm: tau, total cost per day, the part spent on curtailing
        in advance, the unmet share, and the realised coverage of the tau quantile. Costs per day
        are in GBP, or GBP per MW of capacity when the power columns are a fraction of capacity.
    """
    lines = [
        (
            "| Arm | Tau | Total cost per day (95% interval) | Curtailing in advance "
            "| Unmet share | Coverage of tau |"
        ),
        "|---|---|---|---|---|---|",
    ]
    absolute = intervals[0] if intervals else {}
    for row in result.per_arm.iter_rows(named=True):
        interval = absolute.get(row["arm"])
        cost = f"{row['gbp_per_day']:.3f}"
        if interval:
            cost += f" ({interval['lower_95']:.3f} to {interval['upper_95']:.3f})"
        tau = "" if row["tau"] is None else f"{row['tau']:.3f}"
        unmet = "" if row["unmet_fraction"] is None else f"{row['unmet_fraction']:.3f}"
        coverage = "" if row["tau_coverage"] is None else f"{row['tau_coverage']:.3f}"
        procure = f"{row['procure_gbp_per_day']:.3f}"
        lines.append(f"| {row['arm']} | {tau} | {cost} | {procure} | {unmet} | {coverage} |")
    return "\n".join(lines) + "\n"
