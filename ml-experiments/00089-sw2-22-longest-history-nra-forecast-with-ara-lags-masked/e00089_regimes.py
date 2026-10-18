"""Retraining regimes of experiment 00089: which refit time serves which daily forecast run.

A regime is a list of refit times. A forecast run is served by the latest refit at or before its
origin. Every refit forecasts all of the daily runs assigned to it, from 2025-07-01 to 2026-06-30.
The refit times are the P-roll schedule of experiment 00049: the 1st of each month.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

MONTHLY: Final[tuple[datetime, ...]] = tuple(
    datetime(2025 + (6 + i) // 12, (6 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(12)
)
"""The twelve monthly refit times, 2025-07-01 to 2026-06-01."""
TRAIN_WINDOW: Final[timedelta] = timedelta(days=365)
"""A refit trains on rows whose valid time lies in the 365 days before the refit."""
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
"""A refit sees only rows whose valid time is at least this long before the refit."""

REGIMES: Final[dict[str, tuple[datetime, ...]]] = {
    "monthly": MONTHLY,
}


def chosen_refit(*, refits: tuple[datetime, ...], origin: datetime) -> datetime:
    """Return the latest refit at or before an origin.

    Args:
        refits: A regime's refit times.
        origin: A forecast run's `power_fcst_init_time`.

    Returns:
        The latest refit time that is not after the origin.

    Raises:
        ValueError: If every refit is later than the origin.
    """
    earlier = [r for r in refits if r <= origin]
    if not earlier:
        msg = f"no refit at or before {origin}"
        raise ValueError(msg)
    return max(earlier)


def served(*, name: str, origins: list[datetime]) -> dict[datetime, list[datetime]]:
    """Return, for a regime, each refit time and the origins it forecasts.

    Args:
        name: A key of `REGIMES`.
        origins: The forecast runs' `power_fcst_init_time`, every run the regime has to cover.

    Returns:
        Each refit time mapped to the origins whose latest refit it is, in ascending order.
    """
    out: dict[datetime, list[datetime]] = {}
    for origin in sorted(origins):
        out.setdefault(chosen_refit(refits=REGIMES[name], origin=origin), []).append(origin)
    return out
