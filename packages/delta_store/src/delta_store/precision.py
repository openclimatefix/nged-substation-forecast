"""Reduce the significand precision of ``Float32`` columns so parquet compression can work.

Nearly every full-precision ``Float32`` value in a large forecast or weather table is distinct,
so the low significand bits are incompressible noise — they defeat every general-purpose codec.
Rounding each value to a small number of significand bits zeroes those low bits, after which
``BYTE_STREAM_SPLIT`` + zstd compress the column dramatically (see
``delta_store.power_forecasts`` for measured numbers), at the cost of a strictly bounded
*relative* error.

The rounding here is pure Polars arithmetic (no bit-twiddling, no numpy round-trip), which works
because of a classical floating-point identity — Veltkamp splitting — documented in detail on
:func:`round_to_significand_bits`.
"""

from typing import Final

import polars as pl

FLOAT32_SIGNIFICAND_BITS: Final[int] = 24
"""IEEE 754 binary32 significand precision ``p``: 23 explicit fraction bits + 1 implicit bit."""


def round_to_significand_bits(expr: pl.Expr, *, keep_bits: int) -> pl.Expr:
    """Round a ``Float32`` expression to ``keep_bits`` significand bits (round-to-nearest).

    The result is exactly representable with a ``keep_bits``-bit significand, so the low
    ``24 - keep_bits`` explicit fraction bits of every finite output are zero — which is what
    lets ``BYTE_STREAM_SPLIT`` + zstd compress the column — and the relative error is bounded by
    the unit roundoff of a ``keep_bits``-bit format:

        |result - x| <= 2**-keep_bits * |x|

    (e.g. ``keep_bits=13`` -> max relative error 2^-13 ~= 1.2e-4). Rounding is to nearest, so
    unlike truncation it introduces no systematic bias toward zero.

    **How it works — Veltkamp splitting.** With ``s = 24 - keep_bits`` and the constant
    ``C = 2**s + 1``, the expression computes, entirely in ``Float32`` round-to-nearest
    arithmetic (``RN``)::

        c = RN(x * C)  # = RN(x*2^s + x)
        result = RN(c - RN(c - x))

    Veltkamp's theorem (Veltkamp 1968; Dekker 1971, "A floating-point technique for extending
    the available precision", *Numerische Mathematik* 18; Muller et al., *Handbook of
    Floating-Point Arithmetic*, 2nd ed., §4.4, Algorithm 4.9 "Split") states that for
    ``2 <= s <= p - 2`` and no overflow, both subtractions are **exact** and ``result`` is ``x``
    rounded to nearest onto ``p - s = keep_bits`` significand bits. The intuition: ``x*C``
    stacks a copy of ``x`` shifted ``s`` exponent positions above itself; rounding that sum to
    ``p`` bits (forming ``c``) is precisely what discards — with correct rounding — the low
    ``s`` bits of the original ``x``; the two exact subtractions then peel the shifted copy back
    off, leaving ``x`` with its low ``s`` significand bits rounded away.

    **Preconditions — each one is load-bearing:**

    - ``expr`` must be ``Float32``. The splitting constant is pinned to ``Float32``, but Polars
      promotes mixed arithmetic: if a ``Float64`` expression is passed, the whole computation
      runs at ``p = 53`` and silently keeps ``53 - s`` bits instead of ``keep_bits``. (A
      ``.cast(pl.Float32)`` is deliberately *not* applied here — silently changing the dtype of
      a ``Float64`` column would be its own trap; callers own their dtypes.)
    - Arithmetic must be evaluated operation-by-operation in IEEE round-to-nearest. Polars
      (Rust) guarantees this: floats are never reassociated and ``a*b + c`` is never contracted
      into an FMA, either of which would break the exactness of the subtractions.
    - Non-finite and overflow cases are guarded: for ``|x| > f32_max / C`` the product ``c``
      overflows to ``inf`` and the naive result would be ``inf - inf = NaN``; for ``x`` NaN or
      ``±inf``, ``c`` is likewise non-finite. Wherever ``c`` is non-finite the input value is
      passed through unchanged (a huge-but-finite value is stored at full precision rather than
      corrupted; ``NaN``/``±inf`` survive verbatim).
    - Subnormal ``x`` (``|x| < 2**-126``) degrades gracefully: the relative-error bound loosens
      because precision is already at the absolute floor of the format, but the result is still
      a faithful nearby value.

    Args:
        expr: A ``Float32`` expression (see the dtype precondition above).
        keep_bits: Significand bits to keep, in ``[2, 22]`` (the theorem's ``2 <= s <= p - 2``).
            Note this counts *significand* bits — the implicit leading 1 plus explicit fraction
            bits — so ``keep_bits=13`` keeps 12 explicit fraction bits.

    Returns:
        A ``Float32`` expression: ``expr`` rounded to ``keep_bits`` significand bits, with
        non-finite and overflowing inputs passed through unchanged.
    """
    shift = FLOAT32_SIGNIFICAND_BITS - keep_bits
    if not 2 <= shift <= FLOAT32_SIGNIFICAND_BITS - 2:
        raise ValueError(
            f"keep_bits must be in [2, {FLOAT32_SIGNIFICAND_BITS - 2}], got {keep_bits}"
        )
    splitter = pl.lit(float(2**shift + 1), dtype=pl.Float32)
    c = expr * splitter
    rounded = c - (c - expr)
    return pl.when(c.is_finite()).then(rounded).otherwise(expr)
