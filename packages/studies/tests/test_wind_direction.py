import numpy as np
import polars as pl
import pytest
from studies.wind_direction import shuffled_by_month, sine_cosine, veer_degrees


def _encode(*, degrees: list[float]) -> pl.DataFrame:
    sin, cos = sine_cosine(direction_deg=pl.col("d"))
    return pl.DataFrame({"d": degrees}).select(sin=sin, cos=cos)


def test_sine_cosine_puts_north_either_side_of_zero_next_to_each_other() -> None:
    encoded = _encode(degrees=[359.0, 1.0, 180.0])
    gap_across_north = np.hypot(
        encoded["sin"][0] - encoded["sin"][1], encoded["cos"][0] - encoded["cos"][1]
    )
    gap_to_south = np.hypot(
        encoded["sin"][0] - encoded["sin"][2], encoded["cos"][0] - encoded["cos"][2]
    )
    assert gap_across_north < 0.05
    assert gap_to_south > 1.9


def test_sine_cosine_treats_zero_and_360_alike_and_east_is_sine_one() -> None:
    encoded = _encode(degrees=[0.0, 360.0, 90.0])
    assert encoded["sin"][0] == pytest.approx(encoded["sin"][1], abs=1e-9)
    assert encoded["cos"][0] == pytest.approx(encoded["cos"][1], abs=1e-9)
    assert (encoded["sin"][0], encoded["cos"][0]) == pytest.approx((0.0, 1.0), abs=1e-9)
    assert (encoded["sin"][2], encoded["cos"][2]) == pytest.approx((1.0, 0.0), abs=1e-9)


@pytest.mark.parametrize(
    ("upper", "lower", "expected"),
    [(10.0, 350.0, 20.0), (350.0, 10.0, -20.0), (200.0, 100.0, 100.0), (0.0, 180.0, -180.0)],
)
def test_veer_degrees_wraps_across_north(upper: float, lower: float, expected: float) -> None:
    frame = pl.DataFrame({"u": [upper], "l": [lower]}).select(
        v=veer_degrees(upper_deg=pl.col("u"), lower_deg=pl.col("l"))
    )
    assert frame["v"][0] == pytest.approx(expected)


def _donor_map(*, seed: int, month_count: int = 6) -> dict[int, int]:
    months = np.repeat(np.arange(month_count), 3)
    values = months * 100.0 + np.tile(np.arange(3.0), month_count)
    shuffled = shuffled_by_month(
        values=values, months=months, generator=np.random.default_rng(seed)
    )
    donors = {}
    for month in range(month_count):
        rows = shuffled[months == month]
        assert len({int(value // 100) for value in rows}) == 1
        donors[month] = int(rows[0] // 100)
    return donors


def test_shuffled_by_month_gives_each_month_one_other_months_values() -> None:
    donors = _donor_map(seed=1)
    assert all(donor != month for month, donor in donors.items())
    assert sorted(donors.values()) == list(range(6))


def test_shuffled_by_month_donor_map_depends_on_the_random_source() -> None:
    maps = [_donor_map(seed=seed) for seed in range(8)]
    assert len({tuple(sorted(donors.items())) for donors in maps}) > 1
    assert any(donor != (month + 1) % 6 for donors in maps for month, donor in donors.items())


def test_shuffled_by_month_repeats_a_short_donor_to_fill_a_long_month() -> None:
    months = np.array(["a"] * 5 + ["b"] * 2)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0])
    shuffled = shuffled_by_month(values=values, months=months, generator=np.random.default_rng(0))
    assert list(shuffled[:5]) == [10.0, 20.0, 10.0, 20.0, 10.0]
    assert list(shuffled[5:]) == [1.0, 2.0]


def test_shuffled_by_month_needs_two_months() -> None:
    with pytest.raises(ValueError, match="at least two months"):
        shuffled_by_month(
            values=np.ones(3), months=np.array(["a"] * 3), generator=np.random.default_rng(0)
        )
