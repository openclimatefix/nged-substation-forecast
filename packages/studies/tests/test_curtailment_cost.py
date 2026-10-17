from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.curtailment_cost import (
    PERFECT_FORECAST,
    CurtailmentAssumptions,
    align_arms,
    calibrate_tau,
    cost_intervals,
    cost_markdown,
    curtailment_cost_for_arms,
    export_limits,
    export_quantile,
    quantile_levels,
    sweep_assumptions,
    tau_bounds,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 3, 1, tzinfo=UTC)


def _assumptions(
    *, limit_percentile: float = 100.0, unmet_target: float = 0.05, export_is_negative: bool = False
) -> CurtailmentAssumptions:
    return CurtailmentAssumptions(
        calibration_end=END,
        decision_lead_hours=None,
        limit_percentile=limit_percentile,
        unmet_target=unmet_target,
        export_is_negative=export_is_negative,
    )


def _rows(
    *, series: int, days: int, scale: float, offset: float = 0.0, seed: int = 0
) -> pl.DataFrame:
    """Four rows a day, each its own run, with a noisy actual and quantiles around a mean."""
    generator = np.random.default_rng(seed + series)
    times = [START + timedelta(days=day, hours=hour) for day in range(days) for hour in range(4)]
    mean = scale * (1.0 + np.sin(np.arange(len(times)) / 7.0))
    actual = mean + generator.normal(0.0, 0.3 * scale, len(times))
    offsets = {"q05": -0.5, "q25": -0.2, "q50": 0.0, "q75": 0.2, "q95": 0.5}
    return pl.DataFrame(
        {
            "time_series_id": series,
            "power_fcst_init_time": times,
            "valid_time": times,
            "seed": 0,
            "actual": actual,
        }
        | {name: mean + offset + scale * shift for name, shift in offsets.items()}
    ).cast({"time_series_id": pl.Int32, "seed": pl.Int8})


def _two_series(*, offset: float = 0.0) -> pl.DataFrame:
    return pl.concat(
        [
            _rows(series=1, days=120, scale=1.0, offset=offset),
            _rows(series=2, days=120, scale=3.0, offset=offset),
        ]
    )


def test_quantile_levels_sorts_and_requires_two_columns() -> None:
    frame = pl.DataFrame({"q95": [1.0], "q05": [0.0], "other": [1]})
    assert quantile_levels(frame=frame) == [(0.05, "q05"), (0.95, "q95")]
    with pytest.raises(ValueError, match="at least two"):
        quantile_levels(frame=pl.DataFrame({"q05": [0.0]}))


def test_export_quantile_interpolates_between_levels() -> None:
    frame = pl.DataFrame({"q05": [0.0], "q95": [10.0]})
    export = export_quantile(frame=frame, tau=0.5, assumptions=_assumptions())
    assert export[0] == pytest.approx(5.0)


def test_export_quantile_of_demand_is_the_negated_opposite_level() -> None:
    frame = pl.DataFrame({"q05": [-4.0], "q50": [0.0], "q95": [6.0]})
    export = export_quantile(
        frame=frame, tau=0.95, assumptions=_assumptions(export_is_negative=True)
    )
    assert export[0] == pytest.approx(4.0)


def test_export_quantile_clamps_beyond_the_delivered_levels() -> None:
    frame = pl.DataFrame({"q05": [0.0], "q95": [10.0]})
    assert export_quantile(frame=frame, tau=0.99, assumptions=_assumptions())[0] == 10.0
    assert tau_bounds(frame=frame, assumptions=_assumptions()) == (0.05, 0.95)
    assert tau_bounds(frame=frame, assumptions=_assumptions(export_is_negative=True)) == (
        pytest.approx(0.05),
        pytest.approx(0.95),
    )


def test_limits_use_only_calibration_rows_and_mark_non_exporters() -> None:
    frame = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1, 2, 2],
            "valid_time": [START, START + timedelta(days=1), END, START, START],
            "actual": [1.0, 3.0, 99.0, -5.0, -5.0],
        }
    )
    limits = export_limits(actuals=frame, assumptions=_assumptions())
    assert limits["limit"].to_list() == [3.0, -5.0]
    assert limits["export_constrained"].to_list() == [True, False]


def test_limits_count_a_valid_time_once_however_many_runs_cover_it() -> None:
    frame = pl.DataFrame(
        {
            "time_series_id": [1] * 5,
            "valid_time": [START] * 4 + [START + timedelta(days=1)],
            "actual": [0.0] * 4 + [10.0],
        }
    )
    # Counted once, the valid times hold {0, 10}; counted per run they would hold mostly zeros.
    limits = export_limits(actuals=frame, assumptions=_assumptions(limit_percentile=50.0))
    assert limits["limit"].item() == 10.0


def test_cost_matches_a_hand_computation_at_a_fixed_tau() -> None:
    calibration = pl.DataFrame(
        {
            "time_series_id": [1] * 4,
            "power_fcst_init_time": [START + timedelta(days=i) for i in range(4)],
            "valid_time": [START + timedelta(days=i) for i in range(4)],
            "seed": [0] * 4,
            "actual": [1.0, 2.0, 3.0, 4.0],
            "q05": [0.0] * 4,
            "q95": [8.0] * 4,
        }
    )
    scoring = pl.DataFrame(
        {
            "time_series_id": [1],
            "power_fcst_init_time": [END],
            "valid_time": [END],
            "seed": [0],
            "actual": [6.0],
            "q05": [4.0],
            "q95": [8.0],
        }
    )
    frame = pl.concat([calibration, scoring]).cast({"time_series_id": pl.Int32, "seed": pl.Int8})
    result = curtailment_cost_for_arms(
        arms={"arm": frame}, assumptions=_assumptions(), fixed_tau={"arm": 0.5}
    )
    # The limit is 4. The forecast export at tau 0.5 is 6, so 1 MWh is procured at GBP 100;
    # the actual of 6 needs 1 MWh, so nothing is unmet. One row stands for 1/48 of a day.
    assert result.per_arm.filter(pl.col("arm") == "arm")["gbp_per_day"].item() == pytest.approx(
        100.0 * 1.0 * 48
    )
    assert result.per_arm.filter(pl.col("arm") == "arm")["unmet_fraction"].item() == 0.0
    perfect = result.per_arm.filter(pl.col("arm") == PERFECT_FORECAST)
    assert perfect["gbp_per_day"].item() == pytest.approx(100.0 * 1.0 * 48)


def test_a_higher_tau_never_lowers_cost_and_calibration_hits_the_target() -> None:
    frame = _two_series()
    assumptions = _assumptions(limit_percentile=90.0, unmet_target=0.2)
    limits = export_limits(actuals=frame, assumptions=assumptions)
    calibration = frame.filter(pl.col("valid_time") < END)
    chosen = calibrate_tau(calibration=calibration, limits=limits, assumptions=assumptions)
    assert not chosen.clamped
    assert chosen.calibration_unmet == pytest.approx(0.2, abs=0.02)
    costs = [
        curtailment_cost_for_arms(
            arms={"arm": frame}, assumptions=assumptions, fixed_tau={"arm": tau}
        )
        .per_arm.filter(pl.col("arm") == "arm")["gbp_per_day"]
        .item()
        for tau in (0.2, 0.5, 0.9)
    ]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]


def test_calibration_clamps_when_the_target_is_out_of_reach() -> None:
    frame = _two_series(offset=-50.0)
    assumptions = _assumptions(limit_percentile=90.0, unmet_target=0.01)
    limits = export_limits(actuals=frame, assumptions=assumptions)
    chosen = calibrate_tau(
        calibration=frame.filter(pl.col("valid_time") < END), limits=limits, assumptions=assumptions
    )
    assert chosen.clamped
    assert chosen.tau == pytest.approx(0.95)
    assert chosen.calibration_unmet > 0.01


def test_calibration_refuses_when_nothing_exceeds_the_limit() -> None:
    frame = _two_series()
    flat = frame.with_columns(actual=pl.lit(1.0))
    assumptions = _assumptions(limit_percentile=100.0)
    limits = export_limits(actuals=flat, assumptions=assumptions)
    with pytest.raises(ValueError, match="cannot be calibrated"):
        calibrate_tau(
            calibration=flat.filter(pl.col("valid_time") < END),
            limits=limits,
            assumptions=assumptions,
        )


def test_series_that_never_export_are_excluded_and_all_excluded_raises() -> None:
    frame = pl.concat([_rows(series=1, days=120, scale=1.0), _rows(series=2, days=120, scale=1.0)])
    frame = frame.with_columns(
        actual=pl.when(pl.col("time_series_id") == 2)
        .then(-pl.col("actual").abs() - 1)
        .otherwise(pl.col("actual"))
    )
    result = curtailment_cost_for_arms(
        arms={"arm": frame},
        assumptions=_assumptions(limit_percentile=90.0, unmet_target=0.3),
    )
    assert result.excluded_series == (2,)
    assert result.per_series["time_series_id"].unique().to_list() == [1]
    with pytest.raises(ValueError, match="no series is export-constrained"):
        curtailment_cost_for_arms(
            arms={"arm": frame.with_columns(actual=-pl.col("actual").abs() - 1)},
            assumptions=_assumptions(),
        )


def test_scoring_uses_only_runs_made_at_or_after_the_calibration_end() -> None:
    frame = _two_series()
    result = curtailment_cost_for_arms(
        arms={"arm": frame},
        assumptions=_assumptions(limit_percentile=90.0, unmet_target=0.3),
    )
    scored_rows = result.month_totals["rows"].sum()
    expected = frame.filter(pl.col("power_fcst_init_time") >= END).height
    assert scored_rows == expected


def test_a_wider_forecast_costs_more_in_total_and_in_advance_and_intervals_say_so() -> None:
    narrow = _two_series()
    wide = narrow.with_columns(q95=pl.col("q95") + 1.0, q75=pl.col("q75") + 0.5)
    result = curtailment_cost_for_arms(
        arms={"narrow": narrow, "wide": wide},
        assumptions=_assumptions(limit_percentile=90.0),
        fixed_tau={"narrow": 0.9, "wide": 0.9},
    )
    per_arm = result.per_arm.filter(pl.col("arm") != PERFECT_FORECAST)
    cost = dict(per_arm.select("arm", "gbp_per_day").iter_rows())
    advance = dict(per_arm.select("arm", "procure_gbp_per_day").iter_rows())
    assert cost["wide"] > cost["narrow"]
    assert advance["wide"] > advance["narrow"]
    absolute, paired = cost_intervals(result=result, reference="narrow")
    assert paired["wide"]["difference"] == pytest.approx(cost["wide"] - cost["narrow"])
    assert paired["wide"]["lower_95"] > 0.0
    assert absolute["narrow"]["value"] == pytest.approx(cost["narrow"])
    markdown = cost_markdown(result=result, intervals=(absolute, paired))
    assert "| wide | 0.900 |" in markdown
    assert f"| {PERFECT_FORECAST} |" in markdown


def test_sweep_returns_every_combination_with_ranks_and_the_same_series() -> None:
    frame = _two_series()
    swept = sweep_assumptions(
        arms={"narrow": frame, "wide": frame.with_columns(q95=pl.col("q95") + 1.0)},
        base=_assumptions(limit_percentile=90.0, unmet_target=0.3),
        limit_scales=(1.0, 0.5),
        breach_ratios=(10.0, 20.0),
        fixed_tau={"narrow": 0.9, "wide": 0.9},
    )
    assert swept.height == 2 * 2 * 2
    assert set(swept["rank"].to_list()) == {1, 2}
    assert swept["n_series"].n_unique() == 1
    one = swept.filter(pl.col("limit_scale") == 1.0, pl.col("arm") == "narrow")
    assert one.sort("breach_ratio")["gbp_per_day"][1] >= one.sort("breach_ratio")["gbp_per_day"][0]


def test_decision_lead_band_keeps_only_rows_inside_it() -> None:
    frame = _two_series().with_columns(lead_hours=pl.int_range(pl.len()).mod(48).cast(pl.Float64))
    everything = curtailment_cost_for_arms(
        arms={"arm": frame}, assumptions=_assumptions(limit_percentile=90.0), fixed_tau={"arm": 0.9}
    )
    banded_assumptions = CurtailmentAssumptions(
        calibration_end=END,
        limit_percentile=90.0,
        export_is_negative=False,
        decision_lead_hours=(10.0, 19.0),
    )
    banded = curtailment_cost_for_arms(
        arms={"arm": frame}, assumptions=banded_assumptions, fixed_tau={"arm": 0.9}
    )
    assert banded.month_totals["rows"].sum() < everything.month_totals["rows"].sum()
    with pytest.raises(ValueError, match="lead_hours"):
        curtailment_cost_for_arms(
            arms={"arm": _two_series()}, assumptions=banded_assumptions, fixed_tau={"arm": 0.9}
        )


def test_a_smaller_limit_scale_raises_cost_without_changing_the_series_scored() -> None:
    frame = _two_series()
    results = [
        curtailment_cost_for_arms(
            arms={"arm": frame},
            assumptions=CurtailmentAssumptions(
                calibration_end=END,
                limit_percentile=90.0,
                limit_scale=scale,
                export_is_negative=False,
                decision_lead_hours=None,
            ),
            fixed_tau={"arm": 0.9},
        )
        for scale in (1.0, 0.5)
    ]
    full, tight = (result.per_arm.filter(pl.col("arm") == "arm") for result in results)
    assert tight["gbp_per_day"].item() > full["gbp_per_day"].item()
    assert tight["n_series"].item() == full["n_series"].item()


def _gaussian_truth(*, rows: int = 20_000) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return rows of a diurnal Gaussian truth, its mean, and its spread, with no forecast yet."""
    generator = np.random.default_rng(7)
    times = [START + timedelta(hours=hour) for hour in range(rows)]
    mean = 5.0 + 2.0 * np.sin(np.arange(rows) * 2 * np.pi / 24)
    spread = np.full(rows, 1.0)
    actual = mean + spread * generator.normal(size=rows)
    frame = pl.DataFrame(
        {
            "time_series_id": pl.Series([1] * rows, dtype=pl.Int32),
            "power_fcst_init_time": times,
            "valid_time": times,
            "seed": pl.Series([0] * rows, dtype=pl.Int8),
            "actual": actual,
        }
    )
    return frame, mean, spread


def _forecast(
    *, truth: pl.DataFrame, mean: np.ndarray, spread: np.ndarray, bias: float, width: float
) -> pl.DataFrame:
    from scipy import stats

    levels = {"q05": 0.05, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q95": 0.95}
    return truth.with_columns(
        **{
            name: pl.Series(mean + bias * spread + width * spread * stats.norm.ppf(level))
            for name, level in levels.items()
        }
    )


def _total_cost(*, forecast: pl.DataFrame) -> float:
    result = curtailment_cost_for_arms(
        arms={"arm": forecast},
        assumptions=CurtailmentAssumptions(
            calibration_end=START + timedelta(days=30),
            limit_percentile=95.0,
            export_is_negative=False,
            decision_lead_hours=None,
        ),
        fixed_tau={"arm": 0.95},
    )
    return result.per_arm.filter(pl.col("arm") == "arm")["gbp_per_day"].item()


def test_the_true_quantiles_score_lower_than_biased_wide_or_narrow_forecasts() -> None:
    truth, mean, spread = _gaussian_truth()
    best = _total_cost(forecast=_forecast(truth=truth, mean=mean, spread=spread, bias=0, width=1))
    for bias, width in [(0.5, 1.0), (-0.5, 1.0), (0.0, 1.5), (0.0, 0.67)]:
        worse = _total_cost(
            forecast=_forecast(truth=truth, mean=mean, spread=spread, bias=bias, width=width)
        )
        assert worse > best, (bias, width)


def test_cost_rises_with_the_size_of_the_bias_in_either_direction() -> None:
    truth, mean, spread = _gaussian_truth()
    for sign in (1.0, -1.0):
        costs = [
            _total_cost(
                forecast=_forecast(truth=truth, mean=mean, spread=spread, bias=sign * size, width=1)
            )
            for size in (0.25, 0.5, 1.0)
        ]
        assert costs == sorted(costs)


def test_a_forecast_biased_low_spends_less_in_advance_but_leaves_more_unmet() -> None:
    truth, mean, spread = _gaussian_truth()
    rows = {}
    for name, bias in [("low", -1.0), ("true", 0.0)]:
        result = curtailment_cost_for_arms(
            arms={"arm": _forecast(truth=truth, mean=mean, spread=spread, bias=bias, width=1)},
            assumptions=CurtailmentAssumptions(
                calibration_end=START + timedelta(days=30),
                limit_percentile=95.0,
                export_is_negative=False,
                decision_lead_hours=None,
            ),
            fixed_tau={"arm": 0.95},
        )
        rows[name] = result.per_arm.filter(pl.col("arm") == "arm").row(0, named=True)
    assert rows["low"]["procure_gbp_per_day"] < rows["true"]["procure_gbp_per_day"]
    assert rows["low"]["unmet_fraction"] > rows["true"]["unmet_fraction"]


def test_align_arms_keeps_shared_rows_and_drops_a_nan_actual() -> None:
    frame = _rows(series=1, days=10, scale=1.0)
    first = frame.with_columns(
        actual=pl.when(pl.col("valid_time") == START).then(float("nan")).otherwise(pl.col("actual"))
    )
    second = frame.slice(5)
    aligned = align_arms(arms={"first": first, "second": second})
    assert aligned["first"].height == aligned["second"].height == frame.height - 5
    assert aligned["first"]["valid_time"].to_list() == aligned["second"]["valid_time"].to_list()
    assert aligned["first"]["actual"].is_finite().all()


def test_a_nan_actual_is_refused_rather_than_poisoning_the_limit() -> None:
    frame = _two_series().with_columns(
        actual=pl.when(pl.col("valid_time") == START).then(float("nan")).otherwise(pl.col("actual"))
    )
    with pytest.raises(ValueError, match="align_arms"):
        curtailment_cost_for_arms(arms={"arm": frame}, assumptions=_assumptions())


def test_the_demand_path_gives_the_same_costs_as_the_mirrored_generation_path() -> None:
    generation = _two_series()
    mirrored = generation.with_columns(
        actual=-pl.col("actual"),
        q05=-pl.col("q95"),
        q25=-pl.col("q75"),
        q75=-pl.col("q25"),
        q95=-pl.col("q05"),
        q50=-pl.col("q50"),
    )
    positive = curtailment_cost_for_arms(
        arms={"arm": generation},
        assumptions=_assumptions(limit_percentile=90.0),
        fixed_tau={"arm": 0.9},
    )
    negative = curtailment_cost_for_arms(
        arms={"arm": mirrored},
        assumptions=_assumptions(limit_percentile=90.0, export_is_negative=True),
        fixed_tau={"arm": 0.9},
    )
    for column in ("gbp_per_day", "procure_gbp_per_day", "unmet_fraction", "tau_coverage"):
        assert negative.per_arm[column].to_list() == pytest.approx(
            positive.per_arm[column].to_list()
        )
    assert negative.per_arm["gbp_per_day"][0] > 0.0


def test_scored_rows_need_a_forecast_made_after_the_end_and_calibration_an_outcome_before_it() -> (
    None
):
    frame = _two_series()
    straddling = frame.head(50).with_columns(
        power_fcst_init_time=pl.lit(END - timedelta(days=2)),
        valid_time=pl.lit(END + timedelta(days=1)),
        actual=pl.lit(1e6),
    )
    both = pl.concat([frame, straddling])
    assumptions = _assumptions(limit_percentile=90.0, unmet_target=0.3)
    plain = curtailment_cost_for_arms(arms={"arm": frame}, assumptions=assumptions)
    with_straddlers = curtailment_cost_for_arms(arms={"arm": both}, assumptions=assumptions)
    # Rows initialised before the end are not scored, and an outcome after it does not calibrate.
    assert with_straddlers.month_totals["rows"].sum() == plain.month_totals["rows"].sum()
    assert with_straddlers.taus["tau"].to_list() == plain.taus["tau"].to_list()


def test_export_quantile_interpolates_inside_a_three_level_grid() -> None:
    frame = pl.DataFrame({"q05": [0.0], "q50": [10.0], "q95": [30.0]})
    upper = export_quantile(frame=frame, tau=0.75, assumptions=_assumptions())
    assert upper[0] == pytest.approx(10.0 + 20.0 * (0.25 / 0.45))
    lower = export_quantile(frame=frame, tau=0.275, assumptions=_assumptions())
    assert lower[0] == pytest.approx(5.0)
    # Demand: the 0.25 quantile of the value is the 0.75 quantile of export, negated.
    demand = export_quantile(
        frame=frame, tau=0.75, assumptions=_assumptions(export_is_negative=True)
    )
    assert demand[0] == pytest.approx(-(10.0 * (0.20 / 0.45)))


def test_the_true_quantiles_cover_the_tau_quantile_about_tau_of_the_time() -> None:
    truth, mean, spread = _gaussian_truth()
    result = curtailment_cost_for_arms(
        arms={"arm": _forecast(truth=truth, mean=mean, spread=spread, bias=0, width=1)},
        assumptions=CurtailmentAssumptions(
            calibration_end=START + timedelta(days=30),
            limit_percentile=95.0,
            export_is_negative=False,
            decision_lead_hours=None,
        ),
        fixed_tau={"arm": 0.95},
    )
    assert result.per_arm.filter(pl.col("arm") == "arm")["tau_coverage"].item() == pytest.approx(
        0.95, abs=0.01
    )


def test_a_nan_actual_in_a_later_arm_is_dropped_by_align_and_refused_without_it() -> None:
    frame = _rows(series=1, days=10, scale=1.0)
    other = frame.with_columns(
        actual=pl.when(pl.col("valid_time") == START).then(float("nan")).otherwise(pl.col("actual"))
    )
    aligned = align_arms(arms={"first": frame, "other": other})
    assert aligned["first"].height == frame.height - 1
    with pytest.raises(ValueError, match="align_arms"):
        curtailment_cost_for_arms(arms={"first": frame, "other": other}, assumptions=_assumptions())


def test_the_lead_band_excludes_its_upper_end() -> None:
    frame = _two_series().with_columns(lead_hours=pl.lit(36.0))
    assumptions = CurtailmentAssumptions(
        calibration_end=END, export_is_negative=False, limit_percentile=90.0
    )
    with pytest.raises(ValueError, match="no rows on one side"):
        curtailment_cost_for_arms(
            arms={"arm": frame}, assumptions=assumptions, fixed_tau={"arm": 0.9}
        )
