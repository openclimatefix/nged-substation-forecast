"""Retraining regimes of experiment 00075: which refit time serves which daily forecast run.

A regime is a list of refit times. A forecast run is served by the latest refit at or before its
origin. Every refit forecasts all of the daily runs that any regime of its configuration assigns
to it, from 2025-07-01 to 2026-06-30. The refit times are experiment 00048's weekly regime.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

WEEKLY: Final[tuple[datetime, ...]] = tuple(
    datetime(2025, 7, 1, tzinfo=UTC) + timedelta(days=7 * i) for i in range(52)
)
"""Every 7 days from 2025-07-01 to 2026-06-23."""
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
"""A refit sees only rows whose valid time is at least this long before the refit."""

REGIMES: Final[dict[str, tuple[datetime, ...]]] = {
    "weekly": WEEKLY,
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
