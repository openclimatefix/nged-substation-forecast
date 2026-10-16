from pathlib import Path

import numpy as np
import polars as pl
import pytest

from studies import pv_capacity_misfit as m

RATIOS = np.array([0.7, 0.5, 0.5, 0.5, 0.7, 0.55, 0.85])
LEAKAGE = np.array([3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 3.5])


def _unit(*, farm: str, jitter: float) -> m.Unit:
    return m.Unit(
        site="P",
        farm=farm,
        year=2025,
        pv=RATIOS * (1.0 + jitter),
        shapes={"dayL": LEAKAGE},
    )


def test_capacity_recovers_scaled_ratio_vector_with_zero_misfit() -> None:
    capacity, misfit = m.capacity_and_misfit(vectors=3.0 * RATIOS[None, :], ratios=RATIOS)
    assert capacity[0] == pytest.approx(3.0)
    assert misfit[0] == pytest.approx(0.0, abs=1e-12)


def test_misfit_is_positive_when_one_arm_leaks_and_infinite_for_negative_capacity() -> None:
    leaky = 2.0 * RATIOS + LEAKAGE
    _, misfit = m.capacity_and_misfit(vectors=np.vstack([leaky, -RATIOS]), ratios=RATIOS)
    assert misfit[0] > 0.1
    assert np.isinf(misfit[1])


def test_undetected_bias_of_cross_arm_capacity_differs_from_daily_fit() -> None:
    cross_arm, daily = m.undetected_bias(ratios=RATIOS, pv=RATIOS, leakage=LEAKAGE, amplitude=0.25)
    # The daily fit leaks 3.5 / 3.0 as much as M1 against its own ratio, so it is biased more.
    assert daily > cross_arm > 0.0
    assert daily == pytest.approx(0.25 * (3.5 / 3.0) / 0.85)


def test_undetected_bias_is_negative_for_a_load_that_rises_with_sunshine() -> None:
    cross_arm, daily = m.undetected_bias(ratios=RATIOS, pv=RATIOS, leakage=-LEAKAGE, amplitude=0.25)
    assert cross_arm < 0.0
    assert daily == pytest.approx(-0.25 * (3.5 / 3.0) / 0.85)


def test_scenario_with_no_confound_flags_at_the_threshold_rate() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.05, size=(4000, 7))
    _, null_misfit = m.capacity_and_misfit(vectors=2.0 * RATIOS[None, :] + noise, ratios=RATIOS)
    tau = float(np.quantile(null_misfit, m.FLAG_QUANTILE))
    flags, ratio = m.scenario(
        scale=2.0, pv=RATIOS, leakage=None, amplitude=0.0, noise=noise, ratios=RATIOS, tau=tau
    )
    assert flags.mean() == pytest.approx(0.05, abs=1e-9)
    assert np.median(ratio) == pytest.approx(1.0, abs=0.01)


def test_scenario_rejects_a_shape_with_almost_no_m1_leakage() -> None:
    with pytest.raises(AssertionError):
        m.scenario(
            scale=1.0,
            pv=RATIOS,
            leakage=np.zeros(7),
            amplitude=0.25,
            noise=np.zeros((2, 7)),
            ratios=RATIOS,
            tau=1.0,
        )


def test_shape_bound_reads_the_largest_weak_amplitude() -> None:
    rng = np.random.default_rng(1)
    units = [_unit(farm=f, jitter=j) for f, j in (("A", -0.02), ("B", 0.0), ("C", 0.02))]
    noise = {("P", 2025): rng.normal(0.0, 0.15, size=(200, 7))}
    real = {("P", 2025): 2.0 * RATIOS}
    detector = m.Detector(units=units, real=real, noise=noise)
    bound = m.shape_bound(detector=detector, site="P", year=2025, leakage=LEAKAGE)
    assert bound.power_by_amplitude[-1] == pytest.approx(1.0)
    assert bound.power_by_amplitude[0] < m.MIN_POWER
    weakest = max(i for i, p in enumerate(bound.power_by_amplitude) if p < m.MIN_POWER)
    assert bound.bound_amplitude == m.AMPLITUDES[weakest]
    assert bound.bound == pytest.approx(bound.median_error_by_amplitude[weakest])


def test_block_bootstrap_capacities_divides_the_daily_fit_by_its_own_ratio() -> None:
    draws = np.vstack([2.0 * RATIOS, 3.0 * RATIOS])
    capacity, daily = m.block_bootstrap_capacities(draws=draws, ratios=RATIOS)
    assert capacity == pytest.approx([2.0, 3.0])
    assert daily == pytest.approx([2.0, 3.0])


def test_season_year_runs_september_to_august_on_london_time() -> None:
    times = pl.Series(
        "t",
        ["2025-08-31 22:30", "2025-08-31 23:30", "2025-09-01 08:00", "2026-01-15 12:00"],
    ).str.to_datetime(time_zone="UTC")
    years = pl.DataFrame({"t": times}).select(year=m.season_year_of(timestamps=pl.col("t")))
    # 22:30 UTC on 31 August is 23:30 London time, still August; 23:30 UTC is 00:30 on 1 Sept.
    assert years["year"].to_list() == [2024, 2025, 2025, 2025]


def test_shape_bound_is_zero_with_no_amplitude_when_the_test_always_flags() -> None:
    units = [_unit(farm=f, jitter=j) for f, j in (("A", -0.02), ("B", 0.0), ("C", 0.02))]
    noise = {("P", 2025): np.zeros((50, 7))}
    detector = m.Detector(units=units, real={("P", 2025): 2.0 * RATIOS}, noise=noise)
    bound = m.shape_bound(detector=detector, site="P", year=2025, leakage=LEAKAGE)
    assert bound.bound == 0.0
    assert bound.bound_amplitude is None


def test_tau_is_built_from_the_other_farms_with_each_unit_leaving_out_its_own_farm() -> None:
    units = [_unit(farm=f, jitter=j) for f, j in (("A", -0.1), ("B", 0.0), ("C", 0.2))]
    rng = np.random.default_rng(3)
    noise = {("P", 2025): rng.normal(0.0, 0.05, size=(300, 7))}
    detector = m.Detector(units=units, real={("P", 2025): 2.0 * RATIOS}, noise=noise)
    scale = detector.capacity(site="P", year=2025, excluded=frozenset({"A"}))
    values = []
    for unit in units:
        if unit.farm == "A":
            continue
        ratios = np.median(
            np.array([u.pv for u in units if u.farm not in {"A", unit.farm}]), axis=0
        )
        vectors = scale * unit.pv[None, :7] + noise["P", 2025][:, :7]
        values.append(m.capacity_and_misfit(vectors=vectors, ratios=ratios)[1])
    expected = float(np.quantile(np.concatenate(values), m.FLAG_QUANTILE))
    assert detector.tau(site="P", year=2025, excluded=frozenset({"A"})) == pytest.approx(expected)


def test_load_detector_reads_unit_real_and_bootstrap_tables(tmp_path: Path) -> None:
    columns = list(m.ALL_FITS)

    def row(values: np.ndarray) -> dict[str, float]:
        return dict(zip(columns, values, strict=True))

    unit = {"site": "P", "farm": "A", "year": 2025, "kind": "pv", **row(RATIOS)}
    shape = {**unit, "kind": "dayL", **row(LEAKAGE)}
    pl.DataFrame([unit, shape]).write_parquet(tmp_path / "units.parquet")
    real = {"site": "P", "year": 2025, "kind": "real", **row(2 * RATIOS)}
    pl.DataFrame([real]).write_parquet(tmp_path / "real.parquet")
    draws = [{"site": "P", "year": 2025, "draw": i, **row((2 + i) * RATIOS)} for i in range(3)]
    pl.DataFrame(draws).write_parquet(tmp_path / "boot.parquet")
    detector = m.load_detector(
        unit_vectors_path=tmp_path / "units.parquet",
        real_vectors_path=tmp_path / "real.parquet",
        bootstrap_path=tmp_path / "boot.parquet",
    )
    assert [u.farm for u in detector.units] == ["A"]
    assert set(detector.units[0].shapes) == {"dayL"}
    assert detector.noise["P", 2025].mean(axis=0) == pytest.approx(np.zeros(7))
    assert detector.real["P", 2025] == pytest.approx(2 * RATIOS)


def test_misfit_is_finite_for_a_small_positive_capacity() -> None:
    _, misfit = m.capacity_and_misfit(vectors=0.5 * RATIOS[None, :], ratios=RATIOS)
    assert np.isfinite(misfit[0])


def test_ratios_are_the_median_over_farms_not_the_mean() -> None:
    units = [_unit(farm=f, jitter=j) for f, j in (("A", -0.1), ("B", 0.0), ("C", 0.5))]
    detector = m.Detector(units=units, real={}, noise={})
    ratios = detector.ratios(site="P", excluded=frozenset())
    assert ratios is not None
    assert ratios == pytest.approx(RATIOS)


def test_power_of_exactly_the_minimum_does_not_count_as_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exactly_at_the_minimum(**_: object) -> tuple[np.ndarray, np.ndarray]:
        return np.array([True, True, True, True, False]), np.ones(5)

    monkeypatch.setattr(m, "scenario", exactly_at_the_minimum)
    units = [_unit(farm=f, jitter=j) for f, j in (("A", -0.02), ("B", 0.0), ("C", 0.02))]
    noise = {("P", 2025): np.zeros((5, 7))}
    detector = m.Detector(units=units, real={("P", 2025): 2.0 * RATIOS}, noise=noise)
    bound = m.shape_bound(detector=detector, site="P", year=2025, leakage=LEAKAGE)
    assert bound.power_by_amplitude == pytest.approx((0.8,) * len(m.AMPLITUDES))
    assert bound.bound_amplitude is None
