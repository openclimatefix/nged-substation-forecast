# ruff: noqa: E501
"""The label-free consistency rule that picks pseudo-positives, the hard negatives, and the hold-out.

Nothing here reads an NGED label to choose a pseudo-positive. The labelled transfers appear only in
`holdout_mask`, which removes pseudo-positives that sit on a held-out Primary pair's own transfers.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00249_common import GRID_LENGTH, real_transfers
from e00249_data import block_day_mask

STRICT: Final[dict[str, float]] = {
    "own_z6": 3.0,
    "opp_z6": 2.0,
    "ratio_lo": 0.3,
    "ratio_hi": 2.0,
    "offset_h": 4.0,
}
"""The strict tier: balanced opposite steps at a metered neighbour (the brief's rule, relaxed because
the first draft, which required a ratio of 0.5 to 1.5 and an opposite step of 3.0, passed 2 candidates)."""
OWN_Z24_MIN: Final[float] = 2.5
"""The loose tier: a 24-hour step of RO-NC's candidate size and an opposite step at any neighbour."""
GATE_QUANTILE: Final[float] = 0.75
"""The round's score must exceed this quantile of that fold's scores over every real candidate."""
END_RANGE: Final[tuple[int, int]] = (48, 176)
"""Half-hours from the step to the estimated end; 176 keeps the 12 half-hours after the end in the window."""
RETURN_SHARE: Final[float] = 0.5
RUN: Final[int] = 12
ISOLATION: Final[int] = 7 * 48
"""A hard negative has no other candidate on any series within 7 days."""
ONSET_INDEX: Final[int] = 96
"""Position of the step in the 6-day raw window (2 days before, 4 days after)."""


def blocked(*, table: pl.DataFrame) -> np.ndarray:
    """True where a candidate sits on one of the 17 block days."""
    masks = block_day_mask()
    return np.array(
        [
            bool(masks[s][t]) if s in masks else False
            for s, t in table.select("series", "stamp").iter_rows()
        ]
    )


def _run_mean(*, values: np.ndarray, valid: np.ndarray, lo: int, hi: int) -> float:
    """Mean of `values[lo:hi]` over valid points; NaN when under 8 of the 12 are valid."""
    v = valid[lo:hi] > 0.5
    return float(values[lo:hi][v].mean()) if v.sum() >= 8 else float("nan")


def clean_return(*, table: pl.DataFrame, windows: np.ndarray) -> np.ndarray:
    """True when the own residual falls across the estimated end by at least `RETURN_SHARE` of the step."""
    out = np.zeros(table.height, dtype=bool)
    lag = (table["end_est"] - table["stamp"]).to_numpy()
    for i in np.flatnonzero((lag >= END_RANGE[0]) & (lag <= END_RANGE[1])):
        own, ok = windows[i, 0].astype(np.float32), windows[i, 1]
        step = _run_mean(
            values=own, valid=ok, lo=ONSET_INDEX + 2, hi=ONSET_INDEX + 2 + RUN
        ) - _run_mean(values=own, valid=ok, lo=ONSET_INDEX - 2 - RUN, hi=ONSET_INDEX - 2)
        e = ONSET_INDEX + int(lag[i])
        fall = _run_mean(values=own, valid=ok, lo=e - 2 - RUN, hi=e - 2) - _run_mean(
            values=own, valid=ok, lo=e + 2, hi=e + 2 + RUN
        )
        out[i] = bool(
            np.isfinite(step) and np.isfinite(fall) and step > 0 and fall >= RETURN_SHARE * step
        )
    return out


def consistent(
    *, table: pl.DataFrame, windows: np.ndarray, gate: np.ndarray, tier: str = "loose"
) -> np.ndarray:
    """The pseudo-positive rule. `gate` is the round's score per candidate (finite for every source).

    Both tiers need a score above the gate quantile, a clean return, and no D8 drop or block day.
    `strict` also needs a balanced opposite step at a metered neighbour; `loose` needs a 24-hour step
    of at least `OWN_Z24_MIN` noise units and an opposite step at any neighbour.
    """
    if tier == "strict":
        balanced = (
            (table["own_z6"] >= STRICT["own_z6"])
            & (table["opp_z6"] >= STRICT["opp_z6"])
            & (table["n_metered_opposite"] >= 1)
            & table["opp_ratio"].is_between(STRICT["ratio_lo"], STRICT["ratio_hi"])
            & (table["opp_offset_h"] <= STRICT["offset_h"])
        ).to_numpy()
    else:
        balanced = ((table["own_z24"] >= OWN_Z24_MIN) & (table["n_opposite"] >= 1)).to_numpy()
    usable = ~table["d8_drop"].to_numpy() & ~blocked(table=table)
    cut = np.quantile(gate[np.isfinite(gate)], GATE_QUANTILE)
    passes = balanced & usable & np.isfinite(gate) & (gate > cut)
    out = np.zeros(table.height, dtype=bool)
    out[passes] = clean_return(table=table.filter(pl.Series(passes)), windows=windows[passes])
    return out


def hard_negatives(*, table: pl.DataFrame, rule: np.ndarray) -> np.ndarray:
    """Real candidates that fail the rule and are on a bank holiday or isolated, outside D8 drops, block days, and logged or labelled events."""
    isolated = np.zeros(table.height, dtype=bool)
    stamps = np.sort(table["stamp"].to_numpy())
    for i, t in enumerate(table["stamp"].to_numpy()):
        near = np.searchsorted(stamps, t + ISOLATION, side="right") - np.searchsorted(
            stamps, t - ISOLATION, side="left"
        )
        isolated[i] = near <= 1
    ok = (
        ~rule
        & ~table["d8_drop"].to_numpy()
        & ~table["explained"].to_numpy()
        & ~blocked(table=table)
    )
    return ok & (isolated | (table["holiday"].to_numpy() > 0.5))


def holdout_mask(
    *, table: pl.DataFrame, fold: int, folds: dict[str, int], level: str
) -> np.ndarray:
    """True for candidates that must not train fold `fold`'s round 2.

    `event` removes candidates within 48 half-hours before the onset to 48 after the end of a labelled
    transfer of the fold's pairs, on a series of that transfer. `series` removes every candidate on a
    series of any transfer of the fold's pairs.
    """
    transfers = real_transfers().filter(
        pl.col("pair").is_in([p for p, f in folds.items() if f == fold])
    )
    series, stamp = table["series"].to_numpy(), table["stamp"].to_numpy()
    out = np.zeros(table.height, dtype=bool)
    for tr in transfers.iter_rows(named=True):
        on = np.isin(series, tr["series"])
        if level == "series":
            out |= on
        else:
            out |= on & (stamp >= tr["onset"] - 48) & (stamp <= tr["end"] + 48)
    return out


assert GRID_LENGTH > 0
