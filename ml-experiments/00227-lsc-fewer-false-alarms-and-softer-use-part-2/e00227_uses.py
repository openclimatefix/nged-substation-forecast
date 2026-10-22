# ruff: noqa: ANN001
"""Uses of an alarm: mask, soft down-weighting, pair-sum correction, hedge, and subtraction.

Every use takes the same alarm spans (`e00227_nra.Span`) and returns a correction at the scored
rows, which `e00227_fed.quantiles_at` turns into quantiles with experiment 115's K0 spread, so a
contrast between uses is a difference in location only.

- MASK: readings inside a known span set missing (experiments 163 and 171).
- SOFT: observation-noise variance multiplied by a factor inside a known span, so the filter
  down-weights the readings and keeps them.
- SUB: the span's detected megawatts subtracted from the readings (experiment 88's correction).
- HALF: the average of the K0 correction and the MASK correction, a hedge that costs about half as
  much per false alarm.
- PAIRSUM: while a pair-balanced alarm is open, each Primary's readings lose the share of its
  residual that the pair's transfer explains. The pair total (the sum of the residuals of the
  metered Primaries of the pair, which a transfer between them leaves unchanged) is split by the
  ratio of the Primaries' expected power. A span whose Primaries are not all metered, or that no
  pair alarm backs, falls back to MASK.
"""

from __future__ import annotations

from typing import Any, Final

import e00227_alarms as al
import e00227_fed as fed
import e00227_nra as nra
import numpy as np
import polars as pl
from e00227_common import load_panel

SOFT_FACTORS: Final[tuple[float, ...]] = (2.0, 4.0, 16.0)
MATCH: Final[int] = al.PAIR_MATCH
"""A source's span is backed by a pair alarm when a pair-gated interval on the same series has its
step within 24 hours of the span's step."""
MIN_EXPECTED_MW: Final[float] = 0.05


def mask(*, data: Any, stream: np.ndarray, spans: list[nra.Span], cutoffs, rows) -> np.ndarray:
    """The MASK correction at `rows`."""
    return nra.subtraction_correction(
        data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=True
    )


def soft(
    *, data: Any, stream: np.ndarray, spans: list[nra.Span], cutoffs, rows, factor: float
) -> np.ndarray:
    """The SOFT correction at `rows` with the noise variance multiplied by `factor` in a span."""
    return nra.subtraction_correction(
        data=data,
        stream=stream,
        spans=spans,
        cutoffs=cutoffs,
        rows=rows,
        soft_factor=factor,
    )


def sub(*, data: Any, stream: np.ndarray, spans: list[nra.Span], cutoffs, rows) -> np.ndarray:
    """The SUB correction at `rows`."""
    return nra.subtraction_correction(
        data=data, stream=stream, spans=spans, cutoffs=cutoffs, rows=rows, mask=False
    )


def half(*, k0: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """The HALF hedge: the average of the K0 and MASK corrections."""
    return 0.5 * (k0 + masked)


def _members(*, pair: pl.DataFrame) -> dict[tuple[int, int], set[int]]:
    """For each pair interval (alarm series, onset): the Primaries involved."""
    out: dict[tuple[int, int], set[int]] = {}
    for r in pair.iter_rows(named=True):
        out.setdefault((r["alarm_series"], r["onset"]), set()).add(r["series"])
    return out


def pairsum_split(
    *, data: Any, site: str, source_table: pl.DataFrame, pair_gated: pl.DataFrame
) -> tuple[list[nra.Span], list[nra.Span], dict[str, int]]:
    """Split a source's spans into PAIRSUM spans (backed by a metered pair) and MASK fallbacks.

    Returns the PAIRSUM spans (subtracted), the fallback spans (masked), and a count of each.
    """
    panel = load_panel()
    series = fed.SERIES_OF_SITE[site]
    members = _members(pair=pair_gated)
    cap = fed.CAPACITY_MW[site]
    spans = nra.detected_spans(data=data, site=site, intervals=source_table)
    own = source_table.filter((pl.col("setting") == "R0_M48") & (pl.col("series") == series))
    # `detected_spans` drops rows whose megawatts are not finite, keeping the table's order
    rows = [r for r in own.iter_rows(named=True) if np.isfinite(100.0 * r["step_mw"] / cap)]
    sub_spans: list[nra.Span] = []
    mask_spans: list[nra.Span] = []
    for span, row in zip(spans, rows, strict=True):
        group = next(
            (
                m
                for (_, onset), m in members.items()
                if series in m and abs(onset - row["onset"]) <= MATCH
            ),
            None,
        )
        usable = group is not None and all(
            s in panel["residual"] and s in panel["expected"] for s in group
        )
        if not usable:
            mask_spans.append(span)
            continue
        residual = {s: np.nan_to_num(panel["residual"][s]) for s in group}
        expected = {s: np.nan_to_num(panel["expected"][s]) for s in group}
        total_res = sum(residual.values())
        total_exp = sum(expected.values())
        share = np.where(
            total_exp > MIN_EXPECTED_MW, expected[series] / np.maximum(total_exp, 1e-9), 0.0
        )
        moved_mw = residual[series] - share * total_res
        points = fed.to_positions(array=100.0 * moved_mw / cap, data=data)
        window = np.zeros_like(points)
        window[span.onset : span.end] = points[span.onset : span.end]
        sub_spans.append(nra.Span(span.onset, span.end, span.start, span.known, window, window))
    return sub_spans, mask_spans, {"pairsum": len(sub_spans), "fallback": len(mask_spans)}


def mixed_correction(
    *,
    data: Any,
    stream: np.ndarray,
    sub_spans: list[nra.Span],
    mask_spans: list[nra.Span],
    cutoffs: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """The correction when some spans are subtracted and the rest masked.

    Rows share a filter run when no span changes status between their cutoffs, as in
    `nra.subtraction_correction`.
    """
    spans = [*sub_spans, *mask_spans]
    out = np.full(len(data.run_of_row), np.nan)
    breaks = np.sort(np.array([v for s in spans for v in (s.start, s.known)] or [0]))
    state = np.searchsorted(breaks, cutoffs[rows], side="right")
    for s in np.unique(state):
        chosen = rows[state == s]
        cutoff = int(breaks[s - 1]) if s > 0 else None
        cleaned = nra.cleaned_stream(stream=stream, spans=sub_spans, cutoff=cutoff, mask=False)
        cleaned = nra.cleaned_stream(stream=cleaned, spans=mask_spans, cutoff=cutoff, mask=True)
        correction = fed.correction_for(
            data=data, stream=cleaned, prior=np.zeros(len(stream)), multiplier=1.0
        )
        out[chosen] = correction[chosen]
    return out
