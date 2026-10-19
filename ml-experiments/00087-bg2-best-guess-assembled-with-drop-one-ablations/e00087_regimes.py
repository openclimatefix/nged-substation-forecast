"""Retraining regimes of experiment 00087: which refit time serves which daily forecast run.

Copied from experiment 00048's `e00048_regimes.py`, which took the refit times from experiment
00034. A regime is a list of refit times. A forecast run is served by the latest refit at or before
its origin. Every refit forecasts all of the daily runs that any regime of its configuration
assigns to it, from 2025-07-01 to 2026-06-30, so the 73 runs of `fold_a_every5d_v1` are forecast
too.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

MONTHLY: Final[tuple[datetime, ...]] = tuple(
    datetime(2025 + (6 + i) // 12, (6 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(12)
)
"""The twelve monthly refit times, 2025-07-01 to 2026-06-01."""
WEEKLY: Final[tuple[datetime, ...]] = tuple(MONTHLY[0] + timedelta(days=7 * i) for i in range(52))
"""Every 7 days from 2025-07-01 to 2026-06-23."""
FROZEN_REFIT: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
"""The frozen model's anchor: the last valid time it trains on (experiment 00033's recipe)."""
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
"""A refit sees only rows whose valid time is at least this long before the refit."""

REGIMES: Final[dict[str, tuple[datetime, ...]]] = {
    "monthly": MONTHLY,
    "weekly": WEEKLY,
    "frozen": (FROZEN_REFIT,),
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


def jobs_for(
    *, regime_names: tuple[str, ...], origins: list[datetime]
) -> dict[datetime, list[datetime]]:
    """Return each refit time and the union of origins that any of the named regimes assigns it.

    The weekly and monthly regimes share the refit 2025-07-01 and assign it different origins, so
    one fit at that time forecasts the union and a file holds every origin a regime may ask for.

    Args:
        regime_names: Keys of `REGIMES`.
        origins: The forecast runs' `power_fcst_init_time`.

    Returns:
        Each refit time, in ascending order, mapped to its origins in ascending order.
    """
    jobs: dict[datetime, set[datetime]] = {}
    for name in regime_names:
        for refit, assigned in served(name=name, origins=origins).items():
            jobs.setdefault(refit, set()).update(assigned)
    return {refit: sorted(jobs[refit]) for refit in sorted(jobs)}
