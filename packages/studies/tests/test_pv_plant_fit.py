import numpy as np
import pytest
from studies.pv_plant_fit import (
    ResponseGrid,
    block_folds,
    fit_response_table,
    fit_scale_and_clip,
    fit_scale_only,
    fraction_of_output_fitted,
    plant_output,
    soft_minimum,
    unclipped_unit_power,
)


def test_unclipped_unit_power_derates_with_heat() -> None:
    poa = np.array([0.0, 1000.0])
    cool = unclipped_unit_power(
        poa_w_m2=poa, temperature_c=np.array([0.0, 0.0]), heat_per_kelvin=0.004
    )
    hot = unclipped_unit_power(
        poa_w_m2=poa, temperature_c=np.array([30.0, 30.0]), heat_per_kelvin=0.004
    )
    assert cool[0] == 0.0
    assert cool[1] == pytest.approx(1.0)  # cell = 25 C at 1000 W/m² from air at 0 C
    assert hot[1] < cool[1]


def test_soft_minimum_stays_below_both_arguments_and_infinite_limit_is_identity() -> None:
    power = np.array([0.1, 0.5, 1.0, 2.0])
    clipped = soft_minimum(power=power, limit=1.0)
    assert np.all(clipped <= power + 1e-12)
    assert np.all(clipped <= 1.0)
    np.testing.assert_array_equal(soft_minimum(power=power, limit=float("inf")), power)


def test_fit_scale_only_recovers_a_known_scale() -> None:
    unit = np.linspace(0.0, 1.0, 100)
    fitted = fit_scale_only(unclipped=unit, output=1.3 * unit)
    assert fitted.scale == pytest.approx(1.3)
    assert fitted.squared_error == pytest.approx(0.0, abs=1e-12)


def test_fit_scale_and_clip_recovers_known_scale_and_limit() -> None:
    unit = np.linspace(0.0, 1.2, 400)
    output = plant_output(unclipped=unit, scale=1.4, ac_limit=0.9)
    fitted = fit_scale_and_clip(unclipped=unit, output=output)
    assert fitted.scale == pytest.approx(1.4, rel=0.02)
    assert fitted.ac_limit == pytest.approx(0.9, rel=0.02)
    assert fitted.squared_error < fit_scale_only(unclipped=unit, output=output).squared_error


def test_fit_scale_and_clip_never_worse_than_no_clip() -> None:
    unit = np.linspace(0.0, 1.0, 100)
    fitted = fit_scale_and_clip(unclipped=unit, output=0.8 * unit)
    assert fitted.squared_error <= fit_scale_only(unclipped=unit, output=0.8 * unit).squared_error


def test_fraction_of_output_fitted_is_one_for_a_perfect_fit_and_zero_for_zero() -> None:
    output = np.array([1.0, 2.0, 3.0])
    assert fraction_of_output_fitted(output=output, predicted=output) == 1.0
    assert fraction_of_output_fitted(output=output, predicted=np.zeros(3)) == 0.0


def test_response_table_is_monotone_in_irradiance_and_follows_the_data() -> None:
    rng = np.random.default_rng(0)
    grid = ResponseGrid(
        irradiance_w_m2=np.linspace(0.0, 1000.0, 11), elevation_deg=np.linspace(5.0, 65.0, 7)
    )
    irradiance = rng.uniform(0.0, 1000.0, 3000)
    elevation = rng.uniform(5.0, 65.0, 3000)
    truth = np.minimum(irradiance / 1000.0 * 1.2, 0.9)
    table = fit_response_table(
        grid=grid,
        irradiance_w_m2=irradiance,
        elevation_deg=elevation,
        output=truth + rng.normal(0.0, 0.02, 3000),
        prior_output=np.zeros(3000),
        penalty=0.0,
    )
    assert np.all(np.diff(table.values, axis=0) >= -1e-9)
    predicted = table.predict(irradiance_w_m2=irradiance, elevation_deg=elevation)
    assert np.mean(np.abs(predicted - truth)) < 0.03


def test_response_table_with_a_huge_penalty_reproduces_the_prior() -> None:
    rng = np.random.default_rng(1)
    grid = ResponseGrid(
        irradiance_w_m2=np.linspace(0.0, 1000.0, 11), elevation_deg=np.linspace(5.0, 65.0, 7)
    )
    irradiance = rng.uniform(0.0, 1000.0, 2000)
    elevation = rng.uniform(5.0, 65.0, 2000)
    prior = irradiance / 1000.0 * 0.7
    table = fit_response_table(
        grid=grid,
        irradiance_w_m2=irradiance,
        elevation_deg=elevation,
        output=np.ones(2000),
        prior_output=prior,
        penalty=1e6,
    )
    predicted = table.predict(irradiance_w_m2=irradiance, elevation_deg=elevation)
    assert np.mean(np.abs(predicted - prior)) < 0.02


def test_block_folds_partition_the_rows_and_keep_blocks_whole() -> None:
    blocks = np.repeat(np.arange(10), 5)
    folds = block_folds(block_of_row=blocks, n_folds=5)
    assert sum(int(mask.sum()) for mask in folds) == len(blocks)
    for mask in folds:
        assert set(blocks[mask]).isdisjoint(set(blocks[~mask]))
