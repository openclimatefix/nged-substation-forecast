from datetime import UTC, datetime, timedelta

import forecast_fit as ff
import forecast_synthetic as fs
import numpy as np
import polars as pl
import pytest
from forecast_inputs import ArmSpec

START = datetime(2025, 10, 1, tzinfo=UTC)
N_DAYS = 8
ROWS = N_DAYS * 48
RANK_SPEC = ArmSpec(price_source="actual", own_slot="filler", neighbour_slot="filler")


def _base(*, output: np.ndarray, prices: np.ndarray | None = None) -> pl.DataFrame:
    """Return a wide frame of 8 days in 4 folds of 2 days each."""
    times = [START + timedelta(minutes=30 * i) for i in range(ROWS)]
    rng = np.random.default_rng(1)
    persistence = np.concatenate([np.full(48, np.nan), output[:-48]])
    frame = pl.DataFrame(
        {
            "time": times,
            "month": ["2025-10"] * ROWS,
            "fold": [i // 96 for i in range(ROWS)],
            "tod": [i % 48 for i in range(ROWS)],
            "output_mw": output,
            "persistence_mw": persistence,
            "price_actual": prices if prices is not None else rng.normal(size=ROWS),
            **{name: np.full(ROWS, 0.0) + i for i, name in enumerate(ff.Q_COLUMNS)},
        }
    )
    return frame.with_columns(pl.col("time").dt.cast_time_unit("us"))


def _arm(method: str, spec: ArmSpec | None = None) -> ff.ArmDefinition:
    return ff.ArmDefinition(name=method, method=method, spec=spec)  # ty: ignore[invalid-argument-type]


def _run(base: pl.DataFrame, arm: ff.ArmDefinition) -> pl.DataFrame:
    return ff.run_arm(
        base=base,
        arm=arm,
        scored=np.ones(base.height, dtype=bool),
        hyper_parameters=ff.SETTINGS["primary"],
        seeds=(0, 1, 2),
    )


def test_the_climatology_arm_returns_the_frames_own_quantiles_under_every_seed() -> None:
    base = _base(output=np.linspace(-100.0, 100.0, ROWS))

    result = _run(base, _arm("clim"))

    assert result["seed"].unique().sort().to_list() == [0, 1, 2]
    first = result.filter(pl.col("seed") == 0).select(ff.Q_COLUMNS)
    expected = tuple(float(i) for i in range(len(ff.Q_COLUMNS)))
    assert first.row(0) == pytest.approx(expected, abs=1e-6)


def test_a_forecast_equal_to_the_truth_everywhere_scores_zero_crps() -> None:
    output = np.full(ROWS, 3.0)
    base = _base(output=output).with_columns(
        **{name: pl.lit(3.0) for name in ff.Q_COLUMNS}, persistence_mw=pl.lit(3.0)
    )

    result = _run(base, _arm("clim"))

    assert result["crps_pct"].max() == pytest.approx(0.0, abs=1e-9)


def test_persistence_conformal_offsets_come_from_the_training_folds_only() -> None:
    rng = np.random.default_rng(2)
    output = rng.normal(size=ROWS)
    changed = output.copy()
    changed[-48:] += 5.0  # the last day, in fold 3, which no persistence value reads
    scored = np.arange(ROWS) >= 48

    def run(values: np.ndarray) -> pl.DataFrame:
        return ff.run_arm(
            base=_base(output=values),
            arm=_arm("persistence_conformal"),
            scored=scored,
            hyper_parameters=ff.SETTINGS["primary"],
            seeds=(0,),
        )

    first = run(output)
    second = run(changed)

    fold_three = pl.col("fold") == 3
    assert (
        first.filter(fold_three)
        .select(ff.Q_COLUMNS)
        .equals(second.filter(fold_three).select(ff.Q_COLUMNS))
    )
    assert (
        not first.filter(~fold_three)
        .select(ff.Q_COLUMNS)
        .equals(second.filter(~fold_three).select(ff.Q_COLUMNS))
    )


def test_the_rank_rule_arm_is_exact_on_a_battery_that_follows_the_rule() -> None:
    day = np.concatenate([np.full(8, 10.0), np.full(32, 50.0), np.full(8, 200.0)])
    prices = np.tile(day, N_DAYS)
    prices = prices + np.tile(np.arange(48) * 0.01, N_DAYS)
    from studies.battery_dispatch import rank_rule_schedule

    output = 5.0 * rank_rule_schedule(prices=prices, duration_half_hours=4)
    base = _base(output=output, prices=prices)

    result = _run(base, _arm("rank_conformal", RANK_SPEC))

    assert result["crps_pct"].max() == pytest.approx(0.0, abs=1e-6)


def test_the_model_price_columns_replace_any_earlier_model_columns() -> None:
    base = _base(output=np.zeros(ROWS))
    first = ff.with_model_price(base=base, price_model=base.select("time", price_model=pl.lit(1.0)))
    second = ff.with_model_price(
        base=first, price_model=base.select("time", price_model=pl.col("price_actual"))
    )

    assert second["price_model"].to_list() == base["price_actual"].to_list()
    assert second.columns.count("price_model") == 1
    assert {"price_rank_model", "rank_rule_model"} <= set(second.columns)


def test_the_trailing_profile_never_includes_the_targets_own_day() -> None:
    days = 40
    output = np.repeat(np.arange(days, dtype=float), 48)

    profile = fs.trailing_profile(output=output, days=28)

    matrix = profile.reshape(days, 48)
    assert np.isnan(matrix[0]).all()
    assert matrix[1, 0] == 0.0  # only day 0 precedes day 1
    assert matrix[30, 0] == pytest.approx(np.mean(np.arange(2, 30)))  # days 2 to 29
