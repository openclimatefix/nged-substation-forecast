"""Tests of the scaled scoring, the controls' helpers, and the reproduction checks."""

from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import pytest
from studies import capacity_scaling
from studies.capacity_scaling import (
    SCORED_COLUMN,
    assert_constant_scale_reproduces_baseline,
    assert_refit_reproduces,
    derange_months,
    deranged_scale,
    drop_scaled_loss_columns,
    multiplied_forecast_scores,
    oracle_scale,
    scaled_frame,
)
from studies.cross_validation import (
    QUANTILE_LEVELS,
    HyperParameters,
    clamp_to_cap,
    crps,
    fit_one_fold,
    out_of_fold_losses,
)

FAST: Final[HyperParameters] = {
    "max_depth": 3,
    "learning_rate": 0.2,
    "subsample": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "num_boost_round": 15,
}
FEATURES: Final[tuple[str, ...]] = ("hour_of_day", "signal")
STATIC_P99_MW: Final[float] = 5.0


def _frame(*, rows: int = 400, seed: int = 0) -> pl.DataFrame:
    """One plant's rows with an export cap on some of them and a power that the cap holds down."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    times = [start + timedelta(hours=i) for i in range(rows)]
    signal = rng.uniform(0, 1, rows)
    power = np.clip(STATIC_P99_MW * signal + rng.normal(0, 0.2, rows), 0, None)
    constrained = rng.uniform(size=rows) < 0.15
    cap = np.where(constrained, 2.0, np.nan)
    power = np.where(constrained, np.minimum(power, 2.0), power)
    return pl.DataFrame(
        {
            "site": ["E"] * rows,
            "time": times,
            "month": [t.strftime("%Y-%m") for t in times],
            "fold": (np.arange(rows) // (rows // 5)).clip(max=4).astype(np.int32),
            "power_mw": power.astype(np.float32),
            "cap_mw": [None if np.isnan(c) else float(c) for c in cap],
            "constrained": constrained,
            "effective_capacity_mw": np.full(rows, STATIC_P99_MW, dtype=np.float32),
            "hour_of_day": [t.hour for t in times],
            "signal": signal,
        }
    ).with_columns(pl.col("time").dt.replace_time_zone("UTC"), pl.col("cap_mw").cast(pl.Float64))


def test_a_scaled_frame_divides_the_three_megawatt_columns_by_the_rows_own_scale() -> None:
    frame = _frame(rows=50)
    scale = np.linspace(2.0, 4.0, 50)
    scaled = scaled_frame(frame=frame, scale_mw=scale)
    assert scaled["capacity_scale_mw"].to_numpy() == pytest.approx(scale)
    for column in ("power_mw", "effective_capacity_mw"):
        assert scaled[column].to_numpy() == pytest.approx(frame[column].to_numpy() / scale, rel=1e-6)
    cap = frame["cap_mw"].fill_null(np.nan).to_numpy()
    assert np.allclose(scaled["cap_mw"].fill_null(np.nan).to_numpy(), cap / scale, equal_nan=True)
    assert scaled["cap_mw"].null_count() == frame["cap_mw"].null_count() > 0
    assert scaled["power_mw"].dtype == pl.Float64
    assert scaled["signal"].equals(frame["signal"])


def test_a_scale_that_is_not_positive_or_has_the_wrong_length_is_refused() -> None:
    frame = _frame(rows=20)
    with pytest.raises(ValueError, match="positive and finite"):
        scaled_frame(frame=frame, scale_mw=np.zeros(20))
    with pytest.raises(ValueError, match="positive and finite"):
        scaled_frame(frame=frame, scale_mw=np.full(20, np.nan))
    with pytest.raises(ValueError, match="21 scales"):
        scaled_frame(frame=frame, scale_mw=np.ones(21))


def test_scoring_a_scaled_arm_equals_the_multiplied_back_megawatt_error_over_the_static_power() -> None:
    frame = _frame()
    rng = np.random.default_rng(5)
    scale = rng.uniform(2.5, 7.0, frame.height)
    scaled = scaled_frame(frame=frame, scale_mw=scale)
    losses = out_of_fold_losses(
        site_rows=scaled,
        features=FEATURES,
        target="power_mw",
        hyper_parameters=FAST,
        with_quantiles=False,
    ).join(
        scaled.select("site", "time", "capacity_scale_mw", "cap_mw", "power_mw"),
        on=["site", "time"],
    )
    scale_row = losses["capacity_scale_mw"].to_numpy()
    actual_mw = losses["power_mw"].to_numpy() * scale_row
    prediction_mw = (losses["power_mw"].to_numpy() + losses["signed_error_mw"].to_numpy()) * scale_row
    cap_mw = losses["cap_mw"].fill_null(np.nan).to_numpy() * scale_row
    capped_mw = np.minimum(prediction_mw, np.where(np.isnan(cap_mw), np.inf, cap_mw))
    expected = np.abs(actual_mw - capped_mw) / STATIC_P99_MW
    assert (np.isfinite(cap_mw)).any()
    assert losses[SCORED_COLUMN].to_numpy() == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert (capped_mw < prediction_mw - 1e-9).any(), "no prediction was clamped, so the cap is untested"


def test_the_ranked_probability_score_scales_with_the_row_scale_and_the_clamp() -> None:
    frame = _frame()
    scale = np.random.default_rng(7).uniform(2.5, 7.0, frame.height)
    scaled = scaled_frame(frame=frame, scale_mw=scale)
    train = scaled.filter(pl.col("fold") != 0)
    test = scaled.filter(pl.col("fold") == 0)
    _, quantiles = fit_one_fold(
        train=train,
        test=test,
        features=list(FEATURES),
        target="power_mw",
        hyper_parameters=FAST,
        seed=0,
        with_quantiles=True,
    )
    assert quantiles is not None
    assert quantiles.shape[1] == len(QUANTILE_LEVELS)
    row_scale = test["capacity_scale_mw"].to_numpy()
    actual = test["power_mw"].to_numpy()
    in_scaled_units = crps(
        actual=actual, quantiles=clamp_to_cap(prediction=quantiles, cap_mw=test["cap_mw"])
    )
    cap_mw = test["cap_mw"] * pl.Series(row_scale)
    in_megawatts = crps(
        actual=actual * row_scale,
        quantiles=clamp_to_cap(prediction=quantiles * row_scale[:, None], cap_mw=cap_mw),
    )
    assert (test["cap_mw"].is_not_null()).any()
    assert in_megawatts == pytest.approx(in_scaled_units * row_scale, rel=1e-9)


def test_the_scaled_loss_columns_are_dropped_and_the_fraction_columns_kept() -> None:
    frame = _frame(rows=200)
    losses = out_of_fold_losses(
        site_rows=scaled_frame(frame=frame, scale_mw=np.full(200, 3.0)),
        features=FEATURES,
        target="power_mw",
        hyper_parameters=FAST,
        with_quantiles=True,
    )
    kept = drop_scaled_loss_columns(losses=losses)
    assert SCORED_COLUMN in kept.columns
    assert "absolute_error_mw" not in kept.columns and "crps_capped_mw" not in kept.columns


def test_the_constant_scale_check_passes_when_scoring_is_right_and_raises_when_it_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    worst = assert_constant_scale_reproduces_baseline(
        site_rows=frame,
        features=FEATURES,
        hyper_parameters=FAST,
        scale=0.37,
        tolerance_points=1e-4,
        device="cpu",
    )
    assert worst < 1e-4
    with pytest.raises(ValueError, match="needs a plant with constrained rows"):
        assert_constant_scale_reproduces_baseline(
            site_rows=frame.with_columns(constrained=pl.lit(value=False)),
            features=FEATURES,
            hyper_parameters=FAST,
            scale=0.37,
            tolerance_points=1e-4,
            device="cpu",
        )

    def forgets_the_cap(*, frame: pl.DataFrame, scale_mw: np.ndarray) -> pl.DataFrame:
        scale = pl.Series("capacity_scale_mw", scale_mw, dtype=pl.Float64)
        return frame.with_columns(
            capacity_scale_mw=scale,
            power_mw=pl.col("power_mw").cast(pl.Float64) / scale,
            effective_capacity_mw=pl.col("effective_capacity_mw").cast(pl.Float64) / scale,
        )

    monkeypatch.setattr(capacity_scaling, "scaled_frame", forgets_the_cap)
    with pytest.raises(ValueError, match="moved the mean"):
        assert_constant_scale_reproduces_baseline(
            site_rows=frame,
            features=FEATURES,
            hyper_parameters=FAST,
            scale=0.37,
            tolerance_points=1e-4,
            device="cpu",
        )


def test_a_refit_must_equal_the_published_scores_bit_for_bit() -> None:
    keys = pl.DataFrame(
        {
            "site": ["A", "A", "B"],
            "time": [datetime(2025, 1, 1, h, tzinfo=UTC) for h in (1, 2, 1)],
            "seed": [0, 0, 1],
            SCORED_COLUMN: [0.1, 0.2, 0.3],
        }
    )
    assert_refit_reproduces(refit=keys.reverse(), published=keys)
    nudged = keys.with_columns(pl.col(SCORED_COLUMN) + pl.Series([0.0, 1e-12, 0.0]))
    with pytest.raises(ValueError, match="differs from the published"):
        assert_refit_reproduces(refit=nudged, published=keys)
    with pytest.raises(ValueError, match="rows"):
        assert_refit_reproduces(refit=keys.head(2), published=keys)


def test_a_derangement_leaves_no_month_with_its_own_scale_and_no_low_month_on_a_low_month() -> None:
    months = {0: [f"2025-{m:02d}" for m in range(1, 10)], 1: [f"2025-{m:02d}" for m in range(1, 10)]}
    low = {0: ["2025-01", "2025-03", "2025-04"]}
    for seed in range(20):
        donors = derange_months(months_by_plant=months, low_months=low, seed=seed)
        for plant, mapping in donors.items():
            assert sorted(mapping.values()) == sorted(months[plant])
            assert all(month != donor for month, donor in mapping.items())
        assert all(not (m in low[0] and d in low[0]) for m, d in donors[0].items())
    with pytest.raises(ValueError, match="fewer than two"):
        derange_months(months_by_plant={0: ["2025-01"]}, low_months={}, seed=0)


def test_a_deranged_scale_takes_the_same_day_of_the_donor_month() -> None:
    dates = np.datetime64("2025-01-01") + np.arange(59)
    scale = np.arange(59, dtype=float)[None, :] * np.array([[1.0], [10.0]])
    out = deranged_scale(
        scale_mw=scale, dates=dates, donors={0: {"2025-01": "2025-02", "2025-02": "2025-01"}}
    )
    assert out[0, 0] == scale[0, 31]
    assert out[0, 27] == scale[0, 58]
    # February has 28 days, so January's 29th to 31st repeat its last day.
    assert out[0, 28] == scale[0, 58]
    assert out[0, 30] == scale[0, 58]
    assert out[0, 31] == scale[0, 0]
    assert np.array_equal(out[1], scale[1])


def test_a_forecast_multiplied_by_a_factor_is_scored_against_the_capped_actual() -> None:
    frame = _frame(rows=100)
    baseline = pl.concat(
        [
            frame.select("site", "time", "month", "fold").with_columns(
                seed=pl.lit(seed, dtype=pl.Int32),
                actual=frame["power_mw"].cast(pl.Float64),
                prediction=pl.lit(3.0),
            )
            for seed in (0, 1)
        ]
    )
    flat = multiplied_forecast_scores(
        baseline_losses=baseline, frame=frame, factor=np.full(frame.height, 7.0)
    )
    capped = np.minimum(3.0, frame["cap_mw"].fill_null(np.inf).to_numpy())
    expected = np.abs(frame["power_mw"].to_numpy().astype(float) - capped) / STATIC_P99_MW
    assert flat.filter(pl.col("seed") == 0)[SCORED_COLUMN].to_numpy() == pytest.approx(expected)
    doubled_in_fold_zero = np.where(frame["fold"].to_numpy() == 0, 2.0, 1.0)
    moved = multiplied_forecast_scores(
        baseline_losses=baseline, frame=frame, factor=doubled_in_fold_zero
    ).filter(pl.col("seed") == 0)
    train_mean = doubled_in_fold_zero[
        (frame["fold"].to_numpy() != 1) & ~frame["constrained"].to_numpy()
    ].mean()
    row = int(np.flatnonzero((frame["fold"].to_numpy() == 1) & ~frame["constrained"].to_numpy())[0])
    forecast = 3.0 * 1.0 / train_mean
    assert moved[SCORED_COLUMN][row] == pytest.approx(
        abs(float(frame["power_mw"][row]) - forecast) / STATIC_P99_MW
    )


def test_the_oracle_scale_is_the_true_factor_times_each_plants_static_power() -> None:
    factor = np.array([[1.0, 0.7], [0.9, 1.0]])
    scale = oracle_scale(static_p99_mw=np.array([5.0, 8.0]), factor=factor)
    assert scale.tolist() == [[5.0, 3.5], [7.2, 8.0]]
