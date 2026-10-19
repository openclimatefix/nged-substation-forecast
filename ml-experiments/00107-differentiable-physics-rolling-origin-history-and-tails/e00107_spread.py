"""Build the rolling-pool spreads (S0, S1, S2) of experiment 107 and the arms' scored predictions.

**The pool for origin `c` is the arm's own out-of-sample residuals over the 12 months before `c`**:
the scored windows of the four cuts before it. No residual comes from a fit that saw its target, and
none has a valid time after `c`. Rows inside the hand-filtered switching masks are left out.

- **S0** is Part 2's centred spread, `ResidualSpread.fit(centre=True)`, per hour band and calendar
  quarter, fitted on the pool.
- **S1** is the pre-registered alternative: residual quantiles per hour band, calendar quarter, and
  tercile of the arm's predicted mean, centred on each group's median. The tercile cut points come
  from the pool's predicted means alone, per (band, quarter), and each scored row is assigned by
  them. A group under `MIN_SPREAD_GROUP_ROWS` falls back to its (band, quarter) group of S0.
- **S2** (exploratory) is S0 fitted on the pool without its newest window, plus a split-conformal
  shift per level (the median excluded) and hour band, calibrated on that newest window so that its
  coverage is nominal. **Known flaw: S2's results are not interpreted.** The older windows hold no
  rows of the newest window's calendar quarter, so that quarter falls back to a band-level table
  about 3 points wider than S0's quarter table. The shift calibrated on it then narrows every
  quarter's quantiles, which is what produces S2's low coverage. The code review of experiment 107
  found this after the run.

Every function takes only residuals and predicted means that end at or before the origin, which
`e00107_checks.py` tests by perturbing the data at and after the origin.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from studies.dp_gam import (
    MIN_SPREAD_GROUP_ROWS,
    N_BANDS,
    QUANTILE_LEVELS,
    ResidualSpread,
)

N_TERCILES: Final[int] = 3
MEDIAN_POSITION: Final[int] = QUANTILE_LEVELS.index(0.5)


@dataclass(frozen=True)
class LevelTercileSpread:
    """S1: residual quantiles per hour band, calendar quarter, and tercile of the predicted mean.

    Attributes:
        cut_points: Shape (4 bands, 4 quarters, 2): the tercile edges of the pool's predicted
            means in each (band, quarter) group.
        table: Shape (4 bands, 4 quarters, 3 terciles, 11 levels): each group's centred quantiles.
        group_rows: Shape (4, 4, 3): the pool rows in each group.
        group_exceedances_q99: Shape (4, 4, 3): the pool rows above the group's own q99 residual
            quantile.
        fallback: Shape (4, 4, 3), True where the group had fewer than `MIN_SPREAD_GROUP_ROWS` rows
            and took its (band, quarter) quantiles.
    """

    cut_points: np.ndarray
    table: np.ndarray
    group_rows: np.ndarray
    group_exceedances_q99: np.ndarray
    fallback: np.ndarray

    @staticmethod
    def fit(
        *, residuals: np.ndarray, predicted: np.ndarray, band: np.ndarray, quarter: np.ndarray
    ) -> LevelTercileSpread:
        """Fit S1 on a pool.

        Args:
            residuals: Pool residuals (actual minus predicted mean), in points.
            predicted: The pool's predicted means.
            band: The hour band of each row.
            quarter: The calendar quarter of each row.

        Returns:
            The fitted spread.

        Raises:
            ValueError: If the pool is empty.
        """
        if len(residuals) == 0:
            msg = "no residuals to fit the spread on"
            raise ValueError(msg)
        base = ResidualSpread.fit(residuals=residuals, band=band, quarter=quarter, centre=True)
        levels = np.array(QUANTILE_LEVELS)
        cut_points = np.empty((N_BANDS, 4, N_TERCILES - 1))
        table = np.empty((N_BANDS, 4, N_TERCILES, len(levels)))
        rows = np.zeros((N_BANDS, 4, N_TERCILES), dtype=np.int64)
        exceed = np.zeros((N_BANDS, 4, N_TERCILES), dtype=np.int64)
        fallback = np.zeros((N_BANDS, 4, N_TERCILES), dtype=bool)
        for b in range(N_BANDS):
            for q in range(4):
                in_group = (band == b) & (quarter == q)
                group_means = predicted[in_group]
                edges = (
                    np.quantile(group_means, [1 / 3, 2 / 3])
                    if len(group_means)
                    else np.array([np.inf, np.inf])
                )
                cut_points[b, q] = edges
                tercile = np.searchsorted(edges, group_means, side="right")
                for t in range(N_TERCILES):
                    values = residuals[in_group][tercile == t]
                    rows[b, q, t] = len(values)
                    if len(values) >= MIN_SPREAD_GROUP_ROWS:
                        quantiles = np.quantile(values, levels)
                        table[b, q, t] = quantiles - quantiles[MEDIAN_POSITION]
                        exceed[b, q, t] = int((values > np.quantile(values, 0.99)).sum())
                    else:
                        table[b, q, t] = base.table[b, q]
                        fallback[b, q, t] = True
        return LevelTercileSpread(
            cut_points=cut_points,
            table=table,
            group_rows=rows,
            group_exceedances_q99=exceed,
            fallback=fallback,
        )

    def quantiles(self, *, mean: np.ndarray, band: np.ndarray, quarter: np.ndarray) -> np.ndarray:
        """Return the 11 quantiles around each predicted mean, shape (len(mean), 11)."""
        edges = self.cut_points[band, quarter]
        tercile = (mean[:, None] >= edges).sum(axis=1)
        return mean[:, None] + self.table[band, quarter, tercile]


@dataclass(frozen=True)
class ConformalSpread:
    """S2: S0 fitted on older windows plus a per-band, per-level shift from the newest window.

    Attributes:
        base: The centred S0 fitted without the newest window.
        shift: Shape (4 bands, 11 levels), zero at the median.
    """

    base: ResidualSpread
    shift: np.ndarray

    @staticmethod
    def fit(
        *,
        residuals: np.ndarray,
        band: np.ndarray,
        quarter: np.ndarray,
        newest: np.ndarray,
    ) -> ConformalSpread:
        """Fit S2 on a pool whose newest window is flagged.

        Args:
            residuals: Pool residuals in points.
            band: The hour band of each row.
            quarter: The calendar quarter of each row.
            newest: True for the rows of the newest window, which calibrates the shift.

        Returns:
            The fitted spread.

        Raises:
            ValueError: If the older windows or the newest window have no rows.
        """
        if not newest.any() or newest.all():
            msg = "S2 needs older windows to fit on and a newest window to calibrate on"
            raise ValueError(msg)
        base = ResidualSpread.fit(
            residuals=residuals[~newest], band=band[~newest], quarter=quarter[~newest], centre=True
        )
        levels = np.array(QUANTILE_LEVELS)
        shift = np.zeros((N_BANDS, len(levels)))
        for b in range(N_BANDS):
            rows = newest & (band == b)
            if rows.sum() < MIN_SPREAD_GROUP_ROWS:
                continue
            reference = base.table[b, quarter[rows]]
            excess = residuals[rows][:, None] - reference
            for position, level in enumerate(levels):
                if position != MEDIAN_POSITION:
                    shift[b, position] = np.quantile(excess[:, position], level)
        return ConformalSpread(base=base, shift=shift)

    def quantiles(self, *, mean: np.ndarray, band: np.ndarray, quarter: np.ndarray) -> np.ndarray:
        """Return the 11 quantiles around each predicted mean, shifted and kept non-decreasing."""
        raw = self.base.quantiles(mean=mean, band=band, quarter=quarter) + self.shift[band]
        return np.sort(raw, axis=1)
