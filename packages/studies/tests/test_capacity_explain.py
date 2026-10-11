"""Tests of the explainability rubric and the explanation sentence."""

import pytest
from studies.capacity_estimators import ESTIMATOR_FAMILY
from studies.capacity_explain import (
    CRITERIA,
    RUBRIC,
    explanation_sentence,
    fitted_physical_quantities,
    rubric_table,
)


def test_every_estimator_has_a_grade_for_every_criterion_and_a_reason() -> None:
    table = rubric_table()
    assert set(table["estimator"]) == set(ESTIMATOR_FAMILY) == set(RUBRIC)
    for key, _ in CRITERIA:
        assert set(table[key]) <= {"yes", "partly", "no"}
    assert (table["reason"].str.len_chars() > 10).all()
    assert table.filter(table["estimator"] == "DP1")["score"][0] == 4.5
    assert table.filter(table["estimator"] == "R30")["score"][0] == 1.5


def test_the_sentence_names_the_change_the_window_and_the_spread() -> None:
    sentence = explanation_sentence(
        estimator="KF", when="2025-03-14", scale_mw=4.0, earlier_scale_mw=5.0, earlier_days=28,
        relative_sd=0.05, where="the low-output plant",
    )
    assert "fell 20.0% in 28 days to 4.00 MW" in sentence
    assert "standard deviation 5.0%" in sentence
    assert "2025-03-14" in sentence
    rose = explanation_sentence(
        estimator="R30", when="d", scale_mw=6.0, earlier_scale_mw=5.0, earlier_days=7, relative_sd=None
    )
    assert "rose 20.0%" in rose and "standard deviation" not in rose


def test_fitted_quantities_carry_their_units_and_the_availability_factor() -> None:
    quantities = fitted_physical_quantities(
        orientation={"tilt_deg": 25.0, "azimuth_deg": 179.0, "dc_capacity_mw": 7.0, "ac_capacity_mw": 5.0},
        scale_mw=4.5,
        static_p99_mw=6.0,
        relative_sd=0.1,
    )
    assert quantities["tilt"] == "25.0 degrees"
    assert quantities["dc_to_ac_ratio"].startswith("1.40")
    assert quantities["availability_factor"].startswith("75.0%")
    assert "+/- 7.5 points" in quantities["availability_factor"]
    with pytest.raises(KeyError):
        fitted_physical_quantities(orientation={}, scale_mw=1, static_p99_mw=1, relative_sd=None)
