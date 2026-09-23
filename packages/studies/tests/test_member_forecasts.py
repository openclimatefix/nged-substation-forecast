from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.cross_validation import (
    N_FOLDS,
    SEEDS,
    HyperParameters,
    out_of_fold_forecasts_for_members,
    out_of_fold_losses,
    out_of_fold_member_forecasts,
    score_prediction,
    summarise_member_forecasts,
)

QUICK: HyperParameters = {
    "max_depth": 2,
    "learning_rate": 0.3,
    "subsample": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "num_boost_round": 20,
}
MEMBERS = 3


def _stacked(*, scored_fold_target: float) -> pl.DataFrame:
    """Return one site's rows over five months, three members each, one month per fold."""
    rows = []
    generator = np.random.default_rng(0)
    for month in range(N_FOLDS):
        for hour in range(40):
            time = datetime(2025, month + 1, 1, tzinfo=UTC) + timedelta(hours=hour)
            target = scored_fold_target if month == N_FOLDS - 1 else 0.0
            rows += [
                {
                    "site": "A",
                    "time": time,
                    "member": member,
                    "month": f"2025-{month + 1:02d}",
                    "fold": month,
                    "cap_mw": None,
                    "constrained": False,
                    "effective_capacity_mw": 10.0,
                    "weather": float(generator.normal()),
                    "power_mw": target,
                }
                for member in range(MEMBERS)
            ]
    return pl.DataFrame(rows)


def test_every_row_gets_one_forecast_per_member_and_seed():
    forecasts = out_of_fold_member_forecasts(
        site_rows=_stacked(scored_fold_target=0.0),
        features=["weather"],
        target="power_mw",
        hyper_parameters=QUICK,
    )

    assert forecasts.height == N_FOLDS * 40 * len(SEEDS)
    assert set(forecasts["forecasts"].list.len().to_list()) == {MEMBERS}


def test_no_member_of_a_scored_month_is_trained_on():
    # The last month's power is 100 and every other month's is 0. A model that had seen any member
    # of the last month would forecast well above 0 there.
    forecasts = out_of_fold_member_forecasts(
        site_rows=_stacked(scored_fold_target=100.0),
        features=["weather"],
        target="power_mw",
        hyper_parameters=QUICK,
    )

    last = forecasts.filter(pl.col("time") >= datetime(2025, N_FOLDS, 1, tzinfo=UTC))
    assert last.select(pl.col("forecasts").list.max().max()).item() < 1.0


def test_the_members_of_one_row_get_their_own_forecasts():
    # Every member of a row shares the measured power, and the weather differs between members, so
    # a member's forecast follows its own weather.
    rows = _stacked(scored_fold_target=0.0).with_columns(
        power_mw=pl.col("time").dt.hour().cast(pl.Float64)
    )
    rows = rows.with_columns(weather=pl.col("power_mw") + pl.col("member") * 5.0)

    forecasts = out_of_fold_member_forecasts(
        site_rows=rows, features=["weather"], target="power_mw", hyper_parameters=QUICK
    )

    assert forecasts.select(pl.col("forecasts").list.n_unique().min()).item() == MEMBERS


def _scored_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": ["A", "B"],
            "time": [datetime(2025, 6, 1, tzinfo=UTC)] * 2,
            "month": ["2025-06"] * 2,
            "fold": [0, 0],
            "constrained": [False, False],
            "effective_capacity_mw": [10.0, 40.0],
            "cap_mw": [3.0, None],
            "power_mw": [2.0, 20.0],
        }
    )


def test_a_prediction_is_held_to_the_cap_and_divided_by_its_own_capacity():
    prediction = pl.DataFrame(
        {
            "site": ["A", "B"],
            "time": [datetime(2025, 6, 1, tzinfo=UTC)] * 2,
            "seed": [0, 0],
            "prediction": [5.0, 30.0],
        }
    )

    scored = score_prediction(rows=_scored_rows(), prediction=prediction, target="power_mw").sort(
        "site"
    )

    assert scored["signed_error_capped_mw"].to_list() == [1.0, 10.0]
    assert scored["absolute_error_capped_fraction_of_capacity"].to_list() == [0.1, 0.25]


def test_identical_members_give_the_forecast_one_row_per_hour_gives():
    # Weighting each member by one over their number makes 51 copies of a row count as one row, so
    # with row subsampling off the stacked model is the one-row model, split for split.
    settings: HyperParameters = {**QUICK, "subsample": 1.0, "min_child_weight": 5.0}
    single = (
        _stacked(scored_fold_target=0.0)
        .filter(pl.col("member") == 0)
        .with_columns(power_mw=pl.col("weather") * 3.0 + 1.0)
    )
    copies = single.drop("member").join(pl.DataFrame({"member": list(range(MEMBERS))}), how="cross")

    stacked = out_of_fold_member_forecasts(
        site_rows=copies, features=["weather"], target="power_mw", hyper_parameters=settings
    )
    joined = stacked.join(
        out_of_fold_losses(
            site_rows=single,
            features=["weather"],
            target="power_mw",
            hyper_parameters=settings,
            with_quantiles=False,
        )
        .join(single.select("time", "power_mw"), on="time")
        .select("time", "seed", alone=pl.col("signed_error_mw") + pl.col("power_mw")),
        on=["time", "seed"],
    )
    assert joined.height == stacked.height
    np.testing.assert_allclose(
        joined["forecasts"].list.first().to_numpy(), joined["alone"].to_numpy(), atol=1e-4
    )


def test_a_time_with_a_missing_member_is_refused():
    rows = _stacked(scored_fold_target=0.0)
    short = rows.filter(~((pl.col("time") == rows["time"][0]) & (pl.col("member") == 0)))

    with pytest.raises(ValueError, match="same number of members"):
        out_of_fold_member_forecasts(
            site_rows=short, features=["weather"], target="power_mw", hyper_parameters=QUICK
        )


def test_the_summary_takes_the_mean_and_the_median_of_the_members():
    forecasts = pl.DataFrame(
        {
            "site": ["A"],
            "time": [datetime(2025, 6, 1, tzinfo=UTC)],
            "seed": [0],
            "forecasts": [[0.0, 1.0, 2.0, 3.0, 14.0]],
        }
    )

    summary = summarise_member_forecasts(forecasts=forecasts).row(0, named=True)

    assert summary["mean"] == 4.0
    assert summary["median"] == 2.0
    assert summary["members"] == 5
    assert summary["p10"] == pytest.approx(0.0)
    assert summary["p90"] == pytest.approx(14.0)


def test_a_model_trained_on_one_input_forecasts_every_member_and_never_its_scored_months():
    rows = _stacked(scored_fold_target=100.0)
    single = rows.filter(pl.col("member") == 0)

    forecasts = out_of_fold_forecasts_for_members(
        site_rows=single,
        member_rows=rows,
        features=["weather"],
        target="power_mw",
        hyper_parameters=QUICK,
    )

    assert set(forecasts["forecasts"].list.len().to_list()) == {MEMBERS}
    assert forecasts.height == N_FOLDS * 40 * len(SEEDS)
    last = forecasts.filter(pl.col("time") >= datetime(2025, N_FOLDS, 1, tzinfo=UTC))
    assert last.select(pl.col("forecasts").list.max().max()).item() < 1.0


def test_a_model_trained_on_one_input_leaves_out_curtailed_hours():
    rows = _stacked(scored_fold_target=0.0).with_columns(
        constrained=pl.col("time").dt.hour() == 0,
        power_mw=pl.when(pl.col("time").dt.hour() == 0).then(1000.0).otherwise(0.0),
    )
    single = rows.filter(pl.col("member") == 0)

    forecasts = out_of_fold_forecasts_for_members(
        site_rows=single,
        member_rows=rows,
        features=["weather"],
        target="power_mw",
        hyper_parameters=QUICK,
    )

    assert forecasts.select(pl.col("forecasts").list.max().max()).item() < 1.0
