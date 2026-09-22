from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import numpy as np
import polars as pl
import pytest
from studies.cross_validation import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    HyperParameters,
    assign_folds,
    booster_parameters,
    clamp_to_cap,
    crps,
    fit_one_fold,
    out_of_fold_losses,
)

from studies import cross_validation


def _months(site: str, n_months: int) -> list[dict[str, object]]:
    return [{"site": site, "month": f"2025-{month + 1:02d}"} for month in range(n_months)]


def test_folds_match_the_published_scheme():
    # Pinned from the implementation the published results were cut with: a dense month rank per
    # site, scaled to five folds. A site with ten months takes two per fold; one with seven
    # still fills every fold.
    folds = assign_folds(dataset=pl.DataFrame(_months("A", 10) + _months("B", 7)))

    by_site = folds.group_by("site", maintain_order=True).agg(pl.col("fold"))

    assert by_site["fold"].to_list() == [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 0, 1, 2, 2, 3, 4]]


def test_every_row_of_a_month_shares_one_fold():
    rows = pl.DataFrame(_months("A", 10) * 3)

    per_month = assign_folds(dataset=rows).group_by("month").agg(pl.col("fold").n_unique())

    assert per_month["fold"].to_list() == [1] * 10


def test_grouping_by_era_puts_every_era_in_every_fold():
    rows = pl.DataFrame(_months("A", 12)).with_columns(
        era=pl.when(pl.col("month") < "2025-08").then(pl.lit("before")).otherwise(pl.lit("after"))
    )

    folds = assign_folds(dataset=rows, by=("site", "era"))

    eras_per_fold = folds.group_by("fold").agg(pl.col("era").n_unique()).sort("fold")
    assert eras_per_fold["fold"].to_list() == list(range(N_FOLDS))
    assert eras_per_fold["era"].to_list() == [2] * N_FOLDS


def test_crps_of_a_forecast_every_level_puts_one_above_the_outcome():
    # Every level misses by 1 on the high side, so each contributes (1 - level), and twice the
    # level spacing times their sum is 2 * 0.1 * 4.5.
    quantiles = np.ones((1, 9))

    assert crps(actual=np.zeros(1), quantiles=quantiles) == pytest.approx([0.9])


def test_crps_sorts_crossing_quantiles_before_scoring():
    ordered = np.linspace(0.0, 2.0, 9)[None, :]

    assert crps(actual=np.ones(1), quantiles=ordered[:, ::-1]) == pytest.approx(
        crps(actual=np.ones(1), quantiles=ordered)
    )


def test_the_booster_never_subsamples_columns():
    parameters = booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=0)

    assert not any(name.startswith("colsample") for name in parameters)


def test_the_clamp_holds_a_prediction_to_its_cap_and_leaves_uncapped_rows_alone():
    clamped = clamp_to_cap(prediction=np.array([5.0, 5.0, 1.0]), cap_mw=pl.Series([3.0, None, 3.0]))

    assert clamped.tolist() == [3.0, 5.0, 1.0]


def _site_rows() -> pl.DataFrame:
    """Ten months of one site, four rows a month, with a fold-specific feature per fold."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    times = [start + timedelta(days=31 * month + day) for month in range(10) for day in range(4)]
    frame = pl.DataFrame(
        {
            "site": "A",
            "time": times,
            "month": [time.strftime("%Y-%m") for time in times],
            "power_mw": np.linspace(0.0, 3.0, len(times)).astype(np.float32),
            "x": np.arange(len(times), dtype=np.float64),
            # Unequal capacities, so dividing by a pooled capacity cannot pass for dividing each row
            # by its own.
            "effective_capacity_mw": np.where(np.arange(len(times)) % 2 == 0, 4.0, 5.0).astype(
                np.float32
            ),
            "cap_mw": [2.0 if index % 5 == 0 else None for index in range(len(times))],
            "constrained": [index % 7 == 0 for index in range(len(times))],
        }
    )
    return assign_folds(dataset=frame).with_columns(
        pl.col("x").alias(f"x_fold{fold}") for fold in range(N_FOLDS)
    )


class _Call(NamedTuple):
    train: pl.DataFrame
    test: pl.DataFrame
    features: list[str]


class _RecordingFit:
    """Stands in for the booster, recording what each fit was shown."""

    def __init__(self, *, offset_mw: float) -> None:
        self.offset_mw = offset_mw
        self.calls: list[_Call] = []

    def __call__(
        self,
        *,
        train: pl.DataFrame,
        test: pl.DataFrame,
        features: list[str],
        target: str,
        hyper_parameters: object,
        seed: int,
        with_quantiles: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self.calls.append(_Call(train=train, test=test, features=features))
        point = test[target].to_numpy() + self.offset_mw
        return point, (np.repeat(point[:, None], 9, axis=1) if with_quantiles else None)


def _run(
    monkeypatch: pytest.MonkeyPatch, *, site_rows: pl.DataFrame, offset_mw: float = 1.0
) -> tuple[pl.DataFrame, _RecordingFit]:
    fit = _RecordingFit(offset_mw=offset_mw)
    monkeypatch.setattr(cross_validation, "fit_one_fold", fit)
    losses = out_of_fold_losses(
        site_rows=site_rows,
        features=["x", "x_fold{fold}"],
        target="power_mw",
        hyper_parameters=PRIMARY_HYPER_PARAMETERS,
        with_quantiles=True,
    )
    return losses, fit


def test_no_fit_trains_on_the_fold_it_scores(monkeypatch: pytest.MonkeyPatch):
    _, fit = _run(monkeypatch, site_rows=_site_rows())

    for call in fit.calls:
        assert set(call.train["time"]).isdisjoint(set(call.test["time"]))


def test_constrained_rows_are_scored_but_never_trained_on(monkeypatch: pytest.MonkeyPatch):
    site_rows = _site_rows()
    losses, fit = _run(monkeypatch, site_rows=site_rows)

    for call in fit.calls:
        assert not call.train["constrained"].any()
    assert losses.filter(pl.col("constrained")).height == (
        site_rows["constrained"].sum() * len(SEEDS)
    )


def test_every_row_is_scored_once_per_seed(monkeypatch: pytest.MonkeyPatch):
    site_rows = _site_rows()
    losses, _ = _run(monkeypatch, site_rows=site_rows)

    assert losses.height == site_rows.height * len(SEEDS)
    assert losses.select("time", "seed").n_unique() == losses.height


def test_a_fold_placeholder_names_the_fold_being_scored(monkeypatch: pytest.MonkeyPatch):
    _, fit = _run(monkeypatch, site_rows=_site_rows())

    for call in fit.calls:
        assert call.features == ["x", f"x_fold{call.test['fold'][0]}"]


def test_a_fold_with_nothing_to_train_on_is_skipped(monkeypatch: pytest.MonkeyPatch):
    # Every row outside fold 0 is constrained, so fold 0 has no training rows at all.
    site_rows = _site_rows().with_columns(constrained=pl.col("fold") != 0)

    losses, _ = _run(monkeypatch, site_rows=site_rows)

    assert 0 not in losses["fold"].to_list()


def test_the_capped_losses_are_clamped_and_the_uncapped_are_not(monkeypatch: pytest.MonkeyPatch):
    losses, _ = _run(monkeypatch, site_rows=_site_rows(), offset_mw=10.0)
    joined = losses.join(_site_rows().select("time", "cap_mw", "power_mw"), on="time")

    assert joined["absolute_error_mw"].to_numpy() == pytest.approx(10.0)
    capped = joined.filter(pl.col("cap_mw").is_not_null())
    assert capped["absolute_error_capped_mw"].to_numpy() == pytest.approx(
        np.abs(capped["cap_mw"].to_numpy() - capped["power_mw"].to_numpy())
    )


def test_each_row_is_divided_by_its_own_capacity(monkeypatch: pytest.MonkeyPatch):
    losses, _ = _run(monkeypatch, site_rows=_site_rows())

    assert (
        losses["absolute_error_fraction_of_capacity"].to_list()
        == (losses["absolute_error_mw"] / losses["effective_capacity_mw"]).to_list()
    )
    assert (
        losses["absolute_error_capped_fraction_of_capacity"].to_list()
        == (losses["absolute_error_capped_mw"] / losses["effective_capacity_mw"]).to_list()
    )


def test_a_real_fit_is_reproducible_for_a_seed():
    site_rows = _site_rows()
    train = site_rows.filter(pl.col("fold") != 0)
    test = site_rows.filter(pl.col("fold") == 0)
    few_rounds = HyperParameters(**{**PRIMARY_HYPER_PARAMETERS, "num_boost_round": 5})

    first, _ = fit_one_fold(
        train=train,
        test=test,
        features=["x"],
        target="power_mw",
        hyper_parameters=few_rounds,
        seed=0,
        with_quantiles=False,
    )
    second, _ = fit_one_fold(
        train=train,
        test=test,
        features=["x"],
        target="power_mw",
        hyper_parameters=few_rounds,
        seed=0,
        with_quantiles=False,
    )

    assert first.tolist() == second.tolist()
