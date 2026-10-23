# ruff: noqa: E501
"""As-of resolved residuals and the recalibration of the lower and upper quantiles (arms 2 and 2b).

A residual is `actual - q50` of the reference forecast with the shortest lead at that valid time, so
each valid time counts once. A fit at origin T uses only valid times at or before T within the
trailing 365 days. Fits are repeated every 7 origins and applied to the next 7 daily origins.
"""

from dataclasses import dataclass

import e00264_common as common
import numpy as np
import polars as pl

NS_PER_DAY: int = 86_400 * 10**9
LATENCY_NS: int = common.LATENCY_HOURS * 3600 * 10**9


@dataclass(frozen=True)
class Resolved:
    """The shortest-lead forecast row of every valid time, sorted by valid time.

    Attributes:
        valid_ns: Valid time as UTC nanoseconds, sorted ascending.
        origin_ns: Forecast origin of the row, as UTC nanoseconds.
        quantiles: Quantile matrix of shape (n, 11), columns in `common.QUANTILES` order.
        actual: Observed reading in points.
        cumulative_drops: Cumulative count of drops, with a leading zero, shape (n + 1,).
    """

    valid_ns: np.ndarray
    origin_ns: np.ndarray
    quantiles: np.ndarray
    actual: np.ndarray
    cumulative_drops: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        """Return `actual - q50`."""
        return self.actual - self.quantiles[:, common.QUANTILES.index("q50")]

    def rows_between(self, *, first_ns: int, last_ns: int) -> np.ndarray:
        """Return the indices of rows with `first_ns < valid_ns <= last_ns`."""
        lo = int(np.searchsorted(self.valid_ns, first_ns, side="right"))
        hi = int(np.searchsorted(self.valid_ns, last_ns, side="right"))
        return np.arange(lo, hi)

    def drop_share(self, *, at_ns: np.ndarray, window_days: int) -> np.ndarray:
        """Return the share of half-hours known at `at_ns` in the trailing window that dropped.

        NaN where fewer than 48 half-hours are resolved in the window.
        """
        known_ns = at_ns - LATENCY_NS
        hi = np.searchsorted(self.valid_ns, known_ns, side="right")
        lo = np.searchsorted(self.valid_ns, known_ns - window_days * NS_PER_DAY, side="right")
        count = hi - lo
        drops = self.cumulative_drops[hi] - self.cumulative_drops[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(count >= 48, drops / np.maximum(count, 1), np.nan)
        return share


def resolved_from_reference(*, reference: pl.DataFrame) -> Resolved:
    """Return the shortest-lead row per valid time of a reference forecast frame."""
    shortest = reference.sort("lead_hours").group_by("valid_time").first().sort("valid_time")
    quantiles = shortest.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    actual = shortest["actual"].to_numpy().astype(np.float64)
    drops = (actual - quantiles[:, common.QUANTILES.index("q50")]) < common.DROP_POINTS
    return Resolved(
        valid_ns=shortest["valid_time"].dt.epoch("ns").to_numpy(),
        origin_ns=shortest["power_fcst_init_time"].dt.epoch("ns").to_numpy(),
        quantiles=quantiles,
        actual=actual,
        cumulative_drops=np.concatenate([[0], np.cumsum(drops.astype(np.int64))]),
    )


def hour_block(*, valid_ns: np.ndarray) -> np.ndarray:
    """Return the UTC six-hour block (0 to 3) of each valid time."""
    return ((valid_ns // (3600 * 10**9)) % 24 // 6).astype(np.int64)


def season(*, valid_ns: np.ndarray) -> np.ndarray:
    """Return the season (0 December-February, 1 March-May, 2 June-August, 3 September-November)."""
    months = valid_ns.astype("datetime64[ns]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    return ((months % 12) // 3).astype(np.int64)


def weekend(*, valid_ns: np.ndarray) -> np.ndarray:
    """Return 1 for Saturday and Sunday (UTC), else 0."""
    days = valid_ns.astype("datetime64[ns]").astype("datetime64[D]").astype(np.int64)
    return ((days + 3) % 7 >= 5).astype(np.int64)  # 1970-01-01 was a Thursday


def blocks(*, origins_ns: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Split the sorted unique origins into refit blocks; return (fit time, origins) per block."""
    return [
        (int(origins_ns[start]), origins_ns[start : start + common.REFIT_EVERY_ORIGINS])
        for start in range(0, len(origins_ns), common.REFIT_EVERY_ORIGINS)
    ]


def fit_rows(*, resolved: Resolved, fit_ns: int) -> np.ndarray:
    """Return the indices of rows resolved at `fit_ns` within the trailing fit window."""
    known_ns = fit_ns - LATENCY_NS
    return resolved.rows_between(first_ns=known_ns - common.FIT_WINDOW_DAYS * NS_PER_DAY, last_ns=known_ns)


def _cell_quantiles(
    *, ratio: np.ndarray, hours: np.ndarray, seasons: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Return an array (4, 4, len(probabilities)) of the quantiles of `ratio` in each cell.

    Cells are hour block by season. A cell with under `MIN_CELL_ROWS` rows uses its hour block
    alone, and an hour block with under `MIN_CELL_ROWS` rows uses every row.
    """
    table = np.empty((4, 4, len(probabilities)))
    overall = np.quantile(ratio, probabilities)
    for h in range(4):
        in_hour = hours == h
        hour_values = ratio[in_hour]
        hour_q = np.quantile(hour_values, probabilities) if len(hour_values) >= common.MIN_CELL_ROWS else overall
        for s in range(4):
            cell = ratio[in_hour & (seasons == s)]
            table[h, s] = np.quantile(cell, probabilities) if len(cell) >= common.MIN_CELL_ROWS else hour_q
    return table


def recalibrate(*, reference: pl.DataFrame, sides: tuple[str, ...]) -> pl.DataFrame:
    """Return the reference with the named sides' quantiles widened as-of.

    Args:
        reference: The reference forecast frame from `common.load_reference`.
        sides: `("lower",)` for arm 2 and `("lower", "upper")` for arm 2b.

    Returns:
        The same rows, with `q01` to `q25` (lower) and `q75` to `q99` (upper) replaced where a
        fit existed at the row's origin. The widening never narrows a band.
    """
    resolved = resolved_from_reference(reference=reference)
    quantiles = reference.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    origin_ns = reference["power_fcst_init_time"].dt.epoch("ns").to_numpy()
    valid_ns = reference["valid_time"].dt.epoch("ns").to_numpy()
    hours, seasons = hour_block(valid_ns=valid_ns), season(valid_ns=valid_ns)
    out = quantiles.copy()
    column = {name: i for i, name in enumerate(common.QUANTILES)}
    for fit_ns, origins in blocks(origins_ns=np.unique(origin_ns)):
        rows = fit_rows(resolved=resolved, fit_ns=fit_ns)
        if len(rows) < common.MIN_FIT_ROWS:
            continue
        q = resolved.quantiles[rows]
        half_lower = np.maximum(q[:, column["q50"]] - q[:, column["q05"]], common.HALF_WIDTH_FLOOR)
        half_upper = np.maximum(q[:, column["q95"]] - q[:, column["q50"]], common.HALF_WIDTH_FLOOR)
        ok = np.ones(len(rows), dtype=bool)
        res = resolved.residual[rows]
        r_hours = hour_block(valid_ns=resolved.valid_ns[rows])
        r_seasons = season(valid_ns=resolved.valid_ns[rows])
        applies = np.isin(origin_ns, origins)
        a_hours, a_seasons = hours[applies], seasons[applies]
        target = quantiles[applies]
        updated = target.copy()
        if "lower" in sides:
            shortfall = (-res / np.where(ok, half_lower, np.nan))[ok]
            probs = np.array([1 - common.LEVELS[column[n]] for n in common.LOWER_TAILS])
            table = _cell_quantiles(ratio=shortfall, hours=r_hours[ok], seasons=r_seasons[ok], probabilities=probs)
            base = np.maximum(target[:, column["q50"]] - target[:, column["q05"]], common.HALF_WIDTH_FLOOR)
            for j, name in enumerate(common.LOWER_TAILS):
                own = (target[:, column["q50"]] - target[:, column[name]]) / base
                needed = table[a_hours, a_seasons, j]
                updated[:, column[name]] = target[:, column["q50"]] - np.maximum(own, needed) * base
            for j in range(1, len(common.LOWER_TAILS)):
                lo, hi = column[common.LOWER_TAILS[j - 1]], column[common.LOWER_TAILS[j]]
                updated[:, lo] = np.minimum(updated[:, lo], updated[:, hi])
        if "upper" in sides:
            overshoot = (res / np.where(ok, half_upper, np.nan))[ok]
            probs = np.array([common.LEVELS[column[n]] for n in common.UPPER_TAILS])
            table = _cell_quantiles(ratio=overshoot, hours=r_hours[ok], seasons=r_seasons[ok], probabilities=probs)
            base = np.maximum(target[:, column["q95"]] - target[:, column["q50"]], common.HALF_WIDTH_FLOOR)
            for j, name in enumerate(common.UPPER_TAILS):
                own = (target[:, column[name]] - target[:, column["q50"]]) / base
                needed = table[a_hours, a_seasons, j]
                updated[:, column[name]] = target[:, column["q50"]] + np.maximum(own, needed) * base
            for j in range(len(common.UPPER_TAILS) - 2, -1, -1):
                lo, hi = column[common.UPPER_TAILS[j]], column[common.UPPER_TAILS[j + 1]]
                updated[:, hi] = np.maximum(updated[:, hi], updated[:, lo])
        out[applies] = updated
    return reference.with_columns(
        [pl.Series(name, out[:, i].astype(np.float32)) for i, name in enumerate(common.QUANTILES)]
    )


def switch_states(*, resolved: Resolved, origins_ns: np.ndarray) -> dict[int, bool]:
    """Return the engine-visibility switch at each origin, with hysteresis.

    The switch turns on when the 30-day drop share known at the origin reaches
    `SWITCH_ON_SHARE`, turns off when it falls below `SWITCH_OFF_SHARE`, and otherwise keeps its
    state. It starts off, and an undefined share (under 48 known half-hours) keeps the state.
    """
    shares = resolved.drop_share(at_ns=origins_ns, window_days=common.SWITCH_WINDOW_DAYS)
    state, out = False, {}
    for origin, share in zip(origins_ns.tolist(), shares, strict=True):
        if not np.isnan(share):
            if share >= common.SWITCH_ON_SHARE:
                state = True
            elif share < common.SWITCH_OFF_SHARE:
                state = False
        out[origin] = state
    return out
