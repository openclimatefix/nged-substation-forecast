"""Unit tests for ``round_to_significand_bits`` (Veltkamp splitting).

These pin down the numeric contract the docstring promises: bounded round-to-nearest relative
error, all-zero low fraction bits (the property compression relies on), passthrough of
non-finite and product-overflowing inputs, and the load-bearing ``Float32`` dtype.
"""

import math

import numpy as np
import polars as pl
import pytest
from delta_store.precision import round_to_significand_bits


def _round(values: list[float], keep_bits: int) -> pl.Series:
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Float32)})
    return df.select(y=round_to_significand_bits(pl.col("x"), keep_bits=keep_bits))["y"]


@pytest.mark.parametrize("keep_bits", [10, 13, 16])
def test_relative_error_bounded_by_unit_roundoff(keep_bits: int) -> None:
    values = [math.pi, -math.e, 123.456, -0.0871, 1e-6, 987654.0, -4321.99, 300.15]
    original = np.array(values, dtype=np.float32)
    rounded = _round(values, keep_bits).to_numpy()
    rel_err = np.abs(rounded - original) / np.abs(original)
    assert (rel_err <= 2.0**-keep_bits).all()


@pytest.mark.parametrize("keep_bits", [10, 13, 16])
def test_low_fraction_bits_are_zero(keep_bits: int) -> None:
    values = [math.pi, -math.e, 0.123456789, 3.0000002, 65503.9]
    bits = _round(values, keep_bits).to_numpy().view(np.uint32)
    # keep_bits significand bits = (keep_bits - 1) explicit fraction bits kept of 23.
    discarded = np.uint32((1 << (23 - (keep_bits - 1))) - 1)
    assert (bits & discarded == 0).all()


def test_rounds_to_nearest_not_toward_zero() -> None:
    # 1 + 2^-12 + 2^-13 lies strictly above the midpoint between the two neighbouring
    # 13-significand-bit values 1 + 2^-12 and 1 + 2^-11, once the +2^-23-ish representation
    # error is included — round-to-nearest must go *up* in magnitude, truncation would go down.
    x = 1.0 + 2.0**-12 + 2.0**-13 + 2.0**-14
    result = _round([x], keep_bits=13)[0]
    assert result == pytest.approx(1.0 + 2.0**-11, abs=0.0)
    # And the mirror case for negatives: rounding away from zero, not toward it.
    assert _round([-x], keep_bits=13)[0] == pytest.approx(-(1.0 + 2.0**-11), abs=0.0)


def test_exactly_representable_values_unchanged() -> None:
    values = [0.0, -0.0, 1.0, -2.0, 1.5, 0.25, 1024.0, 1.0 + 2.0**-12]  # all fit in 13 bits
    result = _round(values, keep_bits=13).to_numpy()
    assert np.array_equal(result, np.array(values, dtype=np.float32))


def test_non_finite_values_pass_through() -> None:
    result = _round([math.nan, math.inf, -math.inf], keep_bits=13)
    assert math.isnan(result[0])
    assert result[1] == math.inf
    assert result[2] == -math.inf


def test_product_overflow_passes_through_unrounded() -> None:
    # |x| * (2^11 + 1) overflows Float32 -> the guard must return x verbatim, not NaN.
    huge = 3.0e38
    result = _round([huge, -huge], keep_bits=13)
    assert result[0] == np.float32(huge)
    assert result[1] == np.float32(-huge)


def test_output_dtype_is_float32() -> None:
    assert _round([1.0], keep_bits=13).dtype == pl.Float32


@pytest.mark.parametrize("keep_bits", [1, 0, 23, 24, -3])
def test_keep_bits_out_of_theorem_range_rejected(keep_bits: int) -> None:
    with pytest.raises(ValueError, match="keep_bits"):
        round_to_significand_bits(pl.col("x"), keep_bits=keep_bits)
