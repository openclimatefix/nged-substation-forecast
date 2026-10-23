# ruff: noqa: E501
"""As-of widening of the quantile band, with an optional outside prior and visibility switch.

A residual is `actual - q50` of the reference forecast with the shortest lead at that valid time, so
each valid time counts once. A fit at origin T uses only valid times at or before T minus the
delivery latency, within the trailing 365 days. Fits repeat every 7 origins. The widening of each
tail quantile is the quantile of the ratio of the residual to the band's half-width, taken in cells
of six-hour block by season, and the band is never narrowed.
"""

from dataclasses import dataclass

import e00265_common as common
import numpy as np
import polars as pl

NS_PER_DAY: int = 86_400 * 10**9
LATENCY_NS: int = common.LATENCY_HOURS * 3600 * 10**9
COLUMN = {name: i for i, name in enumerate(common.QUANTILES)}
LOWER_PROBS = np.array([1 - common.LEVELS[COLUMN[n]] for n in common.LOWER_TAILS])
UPPER_PROBS = np.array([common.LEVELS[COLUMN[n]] for n in common.UPPER_TAILS])


@dataclass(frozen=True)
class Resolved:
    """The shortest-lead forecast row of every valid time, sorted by valid time."""

    valid_ns: np.ndarray
    quantiles: np.ndarray
    actual: np.ndarray
    cumulative_drops: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        """Return `actual - q50`."""
        return self.actual - self.quantiles[:, COLUMN["q50"]]

    def rows_between(self, *, first_ns: int, last_ns: int) -> np.ndarray:
        """Return the indices of rows with `first_ns < valid_ns <= last_ns`."""
        lo = int(np.searchsorted(self.valid_ns, first_ns, side="right"))
        hi = int(np.searchsorted(self.valid_ns, last_ns, side="right"))
        return np.arange(lo, hi)

    def drop_share(self, *, at_ns: np.ndarray, window_days: int) -> np.ndarray:
        """Return the share of half-hours known at `at_ns` in the trailing window that dropped."""
        known_ns = at_ns - LATENCY_NS
        hi = np.searchsorted(self.valid_ns, known_ns, side="right")
        lo = np.searchsorted(self.valid_ns, known_ns - window_days * NS_PER_DAY, side="right")
        count = hi - lo
        drops = self.cumulative_drops[hi] - self.cumulative_drops[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count >= 48, drops / np.maximum(count, 1), np.nan)


def resolved_from_reference(*, reference: pl.DataFrame) -> Resolved:
    """Return the shortest-lead row per valid time of a reference forecast frame."""
    shortest = reference.sort("lead_hours").group_by("valid_time").first().sort("valid_time")
    quantiles = shortest.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    actual = shortest["actual"].to_numpy().astype(np.float64)
    drops = (actual - quantiles[:, COLUMN["q50"]]) < common.DROP_POINTS
    return Resolved(
        valid_ns=shortest["valid_time"].dt.epoch("ns").to_numpy(),
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


def blocks(*, origins_ns: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Split the sorted unique origins into refit blocks; return (fit time, origins) per block."""
    return [
        (int(origins_ns[start]), origins_ns[start : start + common.REFIT_EVERY_ORIGINS])
        for start in range(0, len(origins_ns), common.REFIT_EVERY_ORIGINS)
    ]


def fit_rows(*, resolved: Resolved, fit_ns: int) -> np.ndarray:
    """Return the indices of rows resolved at `fit_ns` within the trailing fit window."""
    known_ns = fit_ns - LATENCY_NS
    return resolved.rows_between(
        first_ns=known_ns - common.FIT_WINDOW_DAYS * NS_PER_DAY, last_ns=known_ns
    )


@dataclass(frozen=True)
class Ratios:
    """Residual-to-half-width ratios of a set of resolved rows, with their cell labels.

    Attributes:
        shortfall: `-residual / (q50 - q05)`, the lower-tail ratio.
        overshoot: `residual / (q95 - q50)`, the upper-tail ratio.
        hours: Six-hour block (0 to 3) of each row.
        seasons: Season (0 to 3) of each row.
    """

    shortfall: np.ndarray
    overshoot: np.ndarray
    hours: np.ndarray
    seasons: np.ndarray

    @property
    def n(self) -> int:
        """Return the number of rows."""
        return len(self.shortfall)

    def subset(self, mask: np.ndarray) -> Ratios:
        """Return the rows where `mask` is true."""
        return Ratios(
            self.shortfall[mask], self.overshoot[mask], self.hours[mask], self.seasons[mask]
        )


def ratios_of(*, resolved: Resolved, rows: np.ndarray) -> Ratios:
    """Return the ratios of the given rows of a resolved set."""
    q = resolved.quantiles[rows]
    half_lower = np.maximum(q[:, COLUMN["q50"]] - q[:, COLUMN["q05"]], common.HALF_WIDTH_FLOOR)
    half_upper = np.maximum(q[:, COLUMN["q95"]] - q[:, COLUMN["q50"]], common.HALF_WIDTH_FLOOR)
    residual = resolved.residual[rows]
    valid = resolved.valid_ns[rows]
    return Ratios(
        -residual / half_lower,
        residual / half_upper,
        hour_block(valid_ns=valid),
        season(valid_ns=valid),
    )


def concatenate(*, parts: list[Ratios]) -> Ratios:
    """Return the rows of several ratio sets in one set."""
    return Ratios(
        np.concatenate([p.shortfall for p in parts]),
        np.concatenate([p.overshoot for p in parts]),
        np.concatenate([p.hours for p in parts]),
        np.concatenate([p.seasons for p in parts]),
    )


def _cell_tables(
    *, ratio: np.ndarray, hours: np.ndarray, seasons: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the quantile table (4, 4, P) and row count (4, 4) of each cell of `ratio`.

    A cell with under `MIN_CELL_ROWS` rows uses its hour block, and an hour block with under
    `MIN_CELL_ROWS` rows uses every row. With no rows at all the table is NaN.
    """
    table = np.full((4, 4, len(probabilities)), np.nan)
    counts = np.zeros((4, 4))
    if len(ratio) == 0:
        return table, counts
    overall = np.quantile(ratio, probabilities)
    for h in range(4):
        in_hour = hours == h
        hour_values = ratio[in_hour]
        hour_q = (
            np.quantile(hour_values, probabilities)
            if len(hour_values) >= common.MIN_CELL_ROWS
            else overall
        )
        for s in range(4):
            cell = ratio[in_hour & (seasons == s)]
            counts[h, s] = len(cell)
            table[h, s] = (
                np.quantile(cell, probabilities) if len(cell) >= common.MIN_CELL_ROWS else hour_q
            )
    return table, counts


def shrunk_tables(
    *, own: Ratios, prior: Ratios | None, shrinkage_rows: float = common.SHRINKAGE_ROWS
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower and upper tables, each (4, 4, 4), after shrinking own towards the prior.

    With no prior the tables are the site's own (NaN with no rows). With a prior, each cell is
    `w * own + (1 - w) * prior` with `w = n / (n + K)` and `n` the cell's own row count.
    """
    out = []
    for own_ratio, probs, pool_ratio in (
        (own.shortfall, LOWER_PROBS, None if prior is None else prior.shortfall),
        (own.overshoot, UPPER_PROBS, None if prior is None else prior.overshoot),
    ):
        own_table, counts = _cell_tables(
            ratio=own_ratio, hours=own.hours, seasons=own.seasons, probabilities=probs
        )
        if prior is None or pool_ratio is None:
            out.append(own_table)
            continue
        prior_table, _ = _cell_tables(
            ratio=pool_ratio, hours=prior.hours, seasons=prior.seasons, probabilities=probs
        )
        weight = (counts / (counts + shrinkage_rows))[:, :, None]
        out.append(
            np.where(
                counts[:, :, None] > 0,
                weight * np.nan_to_num(own_table) + (1 - weight) * prior_table,
                prior_table,
            )
        )
    return out[0], out[1]


def switch_states(
    *, resolved: Resolved, origins_ns: np.ndarray, on: float, off: float
) -> dict[int, bool]:
    """Return the engine-visibility switch at each origin, with hysteresis.

    The switch turns on when the 30-day drop share known at the origin reaches `on`, turns off
    when it falls below `off`, and otherwise keeps its state. It starts off, and an undefined share
    (under 48 known half-hours) keeps the state.
    """
    shares = resolved.drop_share(at_ns=origins_ns, window_days=common.SWITCH_WINDOW_DAYS)
    state, out = False, {}
    for origin, share in zip(origins_ns.tolist(), shares, strict=True):
        if not np.isnan(share):
            if share >= on:
                state = True
            elif share < off:
                state = False
        out[origin] = state
    return out


def state_at(
    *, states: dict[int, bool], origins_ns: np.ndarray, times_ns: np.ndarray
) -> np.ndarray:
    """Return the switch state at the latest origin at or before each time (off before the first)."""
    ordered = np.sort(origins_ns)
    values = np.array([states[int(o)] for o in ordered])
    idx = np.searchsorted(ordered, times_ns, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], False)


def _log_frame(*, log: list[tuple[int, int, bool]]) -> pl.DataFrame:
    """Return the per-block log as a frame with a UTC `fit_time`."""
    return pl.DataFrame(
        {
            "fit_time_ns": [r[0] for r in log],
            "n_own": [r[1] for r in log],
            "switch_on": [r[2] for r in log],
        },
        schema={"fit_time_ns": pl.Int64, "n_own": pl.Int64, "switch_on": pl.Boolean},
    ).with_columns(fit_time=pl.col("fit_time_ns").cast(pl.Datetime("ns", "UTC")))


def widen(
    *,
    reference: pl.DataFrame,
    pool: dict[str, Resolved] | None = None,
    pool_mode: str = "none",
    switch: tuple[float, float] | None = None,
    shrinkage_rows: float = common.SHRINKAGE_ROWS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the reference with both tails widened as-of, and a per-origin log.

    Args:
        reference: The site's reference forecast frame from `common.load_reference`.
        pool: The other Primaries' resolved rows, keyed by name; used when `pool_mode` is not
            `"none"`.
        pool_mode: `"none"` for the control (own cells only, a fit needs `MIN_FIT_ROWS`),
            `"fixed"` for a prior from the whole scored year of the other Primaries (a lab
            shortcut), and `"asof"` for a prior from the same as-of window, falling back to the
            fixed prior until it holds `MIN_FIT_ROWS` rows.
        switch: `(on, off)` thresholds of the visibility switch, or None for no switch. With a
            switch, only own rows taken in the same switch state as the origin are used.
        shrinkage_rows: `K` in the weight `n / (n + K)` given to a cell's own rows.

    Returns:
        The widened frame, and a log with one row per refit block (`fit_time`, `n_own`, `state`).
    """
    resolved = resolved_from_reference(reference=reference)
    quantiles = reference.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    origin_ns = reference["power_fcst_init_time"].dt.epoch("ns").to_numpy()
    valid_ns = reference["valid_time"].dt.epoch("ns").to_numpy()
    hours, seasons = hour_block(valid_ns=valid_ns), season(valid_ns=valid_ns)
    unique_origins = np.unique(origin_ns)
    states = (
        switch_states(resolved=resolved, origins_ns=unique_origins, on=switch[0], off=switch[1])
        if switch
        else None
    )
    fixed_prior: Ratios | None = None
    if pool_mode != "none":
        assert pool is not None
        fixed_prior = concatenate(
            parts=[ratios_of(resolved=r, rows=np.arange(len(r.valid_ns))) for r in pool.values()]
        )
    out = quantiles.copy()
    log = []
    for fit_ns, origins in blocks(origins_ns=unique_origins):
        rows = fit_rows(resolved=resolved, fit_ns=fit_ns)
        state = bool(states[fit_ns]) if states is not None else False
        if states is not None and len(rows):
            row_state = state_at(
                states=states, origins_ns=unique_origins, times_ns=resolved.valid_ns[rows]
            )
            rows = rows[row_state == state]
        if pool_mode == "none" and len(rows) < common.MIN_FIT_ROWS:
            log.append((fit_ns, len(rows), state))
            continue
        own = ratios_of(resolved=resolved, rows=rows)
        prior = fixed_prior
        if pool_mode == "asof":
            assert pool is not None
            parts = [
                ratios_of(resolved=r, rows=fit_rows(resolved=r, fit_ns=fit_ns))
                for r in pool.values()
            ]
            asof_prior = concatenate(parts=parts)
            if asof_prior.n >= common.MIN_FIT_ROWS:
                prior = asof_prior
        lower, upper = shrunk_tables(own=own, prior=prior, shrinkage_rows=shrinkage_rows)
        log.append((fit_ns, own.n, state))
        applies = np.isin(origin_ns, origins)
        a_hours, a_seasons = hours[applies], seasons[applies]
        target = quantiles[applies]
        updated = target.copy()
        base = np.maximum(
            target[:, COLUMN["q50"]] - target[:, COLUMN["q05"]], common.HALF_WIDTH_FLOOR
        )
        for j, name in enumerate(common.LOWER_TAILS):
            own_ratio = (target[:, COLUMN["q50"]] - target[:, COLUMN[name]]) / base
            needed = lower[a_hours, a_seasons, j]
            updated[:, COLUMN[name]] = (
                target[:, COLUMN["q50"]] - np.maximum(own_ratio, needed) * base
            )
        for j in range(1, len(common.LOWER_TAILS)):
            lo, hi = COLUMN[common.LOWER_TAILS[j - 1]], COLUMN[common.LOWER_TAILS[j]]
            updated[:, lo] = np.minimum(updated[:, lo], updated[:, hi])
        base = np.maximum(
            target[:, COLUMN["q95"]] - target[:, COLUMN["q50"]], common.HALF_WIDTH_FLOOR
        )
        for j, name in enumerate(common.UPPER_TAILS):
            own_ratio = (target[:, COLUMN[name]] - target[:, COLUMN["q50"]]) / base
            needed = upper[a_hours, a_seasons, j]
            updated[:, COLUMN[name]] = (
                target[:, COLUMN["q50"]] + np.maximum(own_ratio, needed) * base
            )
        for j in range(len(common.UPPER_TAILS) - 2, -1, -1):
            lo, hi = COLUMN[common.UPPER_TAILS[j]], COLUMN[common.UPPER_TAILS[j + 1]]
            updated[:, hi] = np.maximum(updated[:, hi], updated[:, lo])
        out[applies] = updated
    widened = reference.with_columns(
        [pl.Series(name, out[:, i].astype(np.float32)) for i, name in enumerate(common.QUANTILES)]
    )
    return widened, _log_frame(log=log)


def flip_dates(*, reference: pl.DataFrame, on: float, off: float) -> list[str]:
    """Return the dates on which the visibility switch changes state at the given thresholds."""
    resolved = resolved_from_reference(reference=reference)
    origins = np.unique(reference["power_fcst_init_time"].dt.epoch("ns").to_numpy())
    states = switch_states(resolved=resolved, origins_ns=origins, on=on, off=off)
    flips, previous = [], False
    for origin in origins.tolist():
        if states[origin] != previous:
            flips.append(
                f"{np.datetime64(origin, 'ns').astype('datetime64[D]')} -> {'on' if states[origin] else 'off'}"
            )
            previous = states[origin]
    return flips


QUARTER_MONTHS: tuple[tuple[int, int], ...] = ((7, 9), (10, 12), (1, 3), (4, 6))
HEAVY_SHARE: float = 0.15


def has_heavy_lower_tail(*, reference: pl.DataFrame) -> bool:
    """Return True when the share of readings below q05 is 0.15 or more in at least two quarters."""
    months = reference["valid_time"].dt.month().to_numpy()
    below = (reference["actual"].to_numpy() < reference["q05"].to_numpy()).astype(np.float64)
    heavy = [
        below[(months >= m0) & (months <= m1)].mean() >= HEAVY_SHARE for m0, m1 in QUARTER_MONTHS
    ]
    return int(sum(bool(h) for h in heavy)) >= 2
