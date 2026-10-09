import numpy as np
import polars as pl
import pytest
from studies.party_bootstrap import (
    PartyMonthCells,
    arm_cells,
    party_month_difference,
    party_month_share,
    resample_totals,
)

PARTY_OF = {"a1": "A", "a2": "A", "b1": "B", "c1": "C", "d1": "D"}
MONTHS = ["2025-10", "2025-11", "2025-12", "2026-01"]


def _losses(*, loss_of, arms=("treatment", "reference")) -> pl.DataFrame:  # noqa: ANN001
    """Two seeds and two half-hours per site and month; `loss_of(site, month)` sets the value."""
    rows = [
        {
            "site": site,
            "month": month,
            "time": i,
            "seed": seed,
            "arm": arm,
            "crps_pct": loss_of(site, month, arm),
        }
        for site in PARTY_OF
        for month in MONTHS
        for i in range(2)
        for seed in (0, 1)
        for arm in arms
    ]
    return pl.DataFrame(rows)


def _cells(*, loss_of, arm: str) -> PartyMonthCells:  # noqa: ANN001
    return arm_cells(losses=_losses(loss_of=loss_of), arm=arm, party_of=PARTY_OF)


def test_cells_sum_the_seed_averaged_loss_within_each_party_and_month() -> None:
    cells = _cells(loss_of=lambda site, month, arm: 2.0, arm="reference")

    assert cells.parties == ["A", "B", "C", "D"]
    assert cells.months == MONTHS
    # Party A has 2 batteries x 2 half-hours = 4 per month, at a seed-averaged loss of 2.
    assert cells.counts[0].tolist() == [4.0] * 4
    assert cells.sums[0].tolist() == [8.0] * 4
    assert cells.counts[1].tolist() == [2.0] * 4


def test_a_constant_gain_gives_a_degenerate_interval() -> None:
    def loss(site: str, month: str, arm: str) -> float:
        return 3.0 if arm == "reference" else 2.5

    result = party_month_difference(
        treatment=_cells(loss_of=loss, arm="treatment"),
        reference=_cells(loss_of=loss, arm="reference"),
    )

    assert result["difference"] == pytest.approx(-0.5)
    assert result["lower_95"] == pytest.approx(-0.5)
    assert result["upper_95"] == pytest.approx(-0.5)
    assert result["party_weighted"] == pytest.approx(-0.5)


def test_an_effect_found_in_one_party_only_has_an_interval_that_reaches_zero() -> None:
    def loss(site: str, month: str, arm: str) -> float:
        gain = 1.0 if PARTY_OF[site] == "A" else 0.0
        return 3.0 if arm == "reference" else 3.0 - gain

    result = party_month_difference(
        treatment=_cells(loss_of=loss, arm="treatment"),
        reference=_cells(loss_of=loss, arm="reference"),
    )

    # A resample that omits party A has no effect at all, so the upper bound is zero.
    assert result["difference"] < 0.0
    assert result["upper_95"] == pytest.approx(0.0)
    assert result["party_weighted"] == pytest.approx(-0.25)  # one of four parties, equal weight


def test_the_party_weighted_mean_counts_each_party_once_however_many_batteries_it_runs() -> None:
    def loss(site: str, month: str, arm: str) -> float:
        gain = 1.0 if PARTY_OF[site] == "A" else 0.0  # party A runs 2 of the 5 batteries
        return 3.0 if arm == "reference" else 3.0 - gain

    result = party_month_difference(
        treatment=_cells(loss_of=loss, arm="treatment"),
        reference=_cells(loss_of=loss, arm="reference"),
    )

    assert result["difference"] == pytest.approx(-2 / 5)
    assert result["party_weighted"] == pytest.approx(-1 / 4)


def test_arms_over_different_half_hours_are_rejected() -> None:
    cells = _cells(loss_of=lambda site, month, arm: 1.0, arm="reference")
    fewer = PartyMonthCells(
        sums=cells.sums, counts=cells.counts - 1.0, parties=cells.parties, months=cells.months
    )

    with pytest.raises(ValueError, match="same parties"):
        party_month_difference(treatment=fewer, reference=cells)


def test_resampled_totals_use_the_drawn_party_and_month_counts() -> None:
    stacked = np.ones((1, 3, 2))

    totals = resample_totals(stacked=stacked, n_resamples=50, seed=1)

    # Each resample draws 3 parties and 2 months, so a grid of ones totals 3 x 2 = 6.
    assert totals[:, 0].tolist() == [6.0] * 50


def test_the_share_recovered_is_the_ratio_of_the_two_gains() -> None:
    def loss(site: str, month: str, arm: str) -> float:
        return {"baseline": 10.0, "full": 6.0, "partial": 9.0}[arm]

    def cells(arm: str) -> PartyMonthCells:
        rows = _losses(loss_of=loss, arms=("baseline", "full", "partial"))
        return arm_cells(losses=rows, arm=arm, party_of=PARTY_OF)

    result = party_month_share(
        baseline=cells("baseline"), full=cells("full"), partial=cells("partial")
    )

    assert result["share"] == pytest.approx(0.25)
    assert result["lower_95"] == pytest.approx(0.25)
    assert result["upper_95"] == pytest.approx(0.25)


def test_the_share_rejects_arms_over_different_half_hours() -> None:
    cells = _cells(loss_of=lambda site, month, arm: 1.0, arm="reference")
    fewer = PartyMonthCells(
        sums=cells.sums, counts=cells.counts - 1.0, parties=cells.parties, months=cells.months
    )

    with pytest.raises(ValueError, match="same parties"):
        party_month_share(baseline=cells, full=cells, partial=fewer)


def test_an_effect_found_in_one_month_only_has_an_interval_that_reaches_zero() -> None:
    def loss(site: str, month: str, arm: str) -> float:
        gain = 1.0 if month == MONTHS[0] else 0.0
        return 3.0 if arm == "reference" else 3.0 - gain

    result = party_month_difference(
        treatment=_cells(loss_of=loss, arm="treatment"),
        reference=_cells(loss_of=loss, arm="reference"),
    )

    # A resample that omits the one month with an effect has no effect at all.
    assert result["difference"] == pytest.approx(-0.25)
    assert result["upper_95"] == pytest.approx(0.0)
    assert result["lower_95"] < -0.25


def test_resampled_totals_equal_the_weighted_sum_of_the_cells() -> None:
    stacked = np.arange(1.0, 13.0).reshape(1, 3, 4)

    totals = resample_totals(stacked=stacked, n_resamples=5, seed=7)

    generator = np.random.default_rng(7)
    party_draws = generator.integers(0, 3, size=(5, 3))
    month_draws = generator.integers(0, 4, size=(5, 4))
    for r in range(5):
        expected = sum(stacked[0, p, m] for p in party_draws[r] for m in month_draws[r])
        assert totals[r, 0] == pytest.approx(expected)
