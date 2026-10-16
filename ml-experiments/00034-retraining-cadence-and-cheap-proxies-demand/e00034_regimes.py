"""Retraining regimes of experiment 00034: which refit time serves which validation run.

A regime is a list of refit times. A validation run is forecast by the latest refit at or before
its origin. The monthly refit times double as the refit times of the half-yearly, three-a-year, and
quarterly regimes, so one set of boosters serves them all.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import e00034_common as common
import polars as pl

MONTHLY: Final[tuple[datetime, ...]] = tuple(
    datetime(2025 + (6 + i) // 12, (6 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(12)
)
"""The twelve monthly refit times, 2025-07-01 to 2026-06-01."""
HALF_YEARLY: Final[tuple[datetime, ...]] = (MONTHLY[0], MONTHLY[6])
THREE_A_YEAR: Final[tuple[datetime, ...]] = (MONTHLY[0], MONTHLY[4], MONTHLY[8])
QUARTERLY: Final[tuple[datetime, ...]] = (MONTHLY[0], MONTHLY[3], MONTHLY[6], MONTHLY[9])
YEARLY: Final[tuple[datetime, ...]] = (MONTHLY[0],)
WEEKLY: Final[tuple[datetime, ...]] = tuple(MONTHLY[0] + timedelta(days=7 * i) for i in range(52))
"""Every 7 days from 2025-07-01 to 2026-06-23."""
DAILY_MONTHS: Final[tuple[tuple[int, int], ...]] = ((2025, 10), (2026, 1), (2026, 4))
"""The months in which the daily regime refits at every validation origin."""
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
"""A refit sees only rows whose valid time is at least this long before the refit."""

FIXED_REGIMES: Final[dict[str, tuple[datetime, ...]]] = {
    "monthly": MONTHLY,
    "half_yearly": HALF_YEARLY,
    "three_a_year": THREE_A_YEAR,
    "quarterly": QUARTERLY,
    "yearly": YEARLY,
    "weekly": WEEKLY,
}


def run_origins() -> list[datetime]:
    """Return the forecast origins of the validation runs, sorted."""
    cache = common.WORK_DIR / "run_origins.parquet"
    if not cache.exists():
        origins = (
            pl.concat(
                [
                    pl.read_parquet(path, columns=["power_fcst_init_time"]).unique()
                    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
                ]
            )
            .unique()
            .sort("power_fcst_init_time")
        )
        common.WORK_DIR.mkdir(parents=True, exist_ok=True)
        origins.write_parquet(cache)
    return pl.read_parquet(cache)["power_fcst_init_time"].to_list()


def daily_refits(*, origins: list[datetime]) -> tuple[datetime, ...]:
    """Return the daily regime's refit times: the origins that fall in the daily months."""
    return tuple(o for o in origins if (o.year, o.month) in DAILY_MONTHS)


def regime_refits(*, name: str, origins: list[datetime]) -> tuple[datetime, ...]:
    """Return a regime's refit times by name (the fixed regimes and `daily`)."""
    if name == "daily":
        return daily_refits(origins=origins)
    return FIXED_REGIMES[name]


def chosen_refit(*, refits: tuple[datetime, ...], origin: datetime) -> datetime:
    """Return the latest refit at or before an origin.

    Args:
        refits: A regime's refit times.
        origin: A validation run's forecast origin.

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
    """Return, for a regime, each refit time and the validation origins it forecasts.

    Daily covers only the origins in its months; the other regimes cover every origin.
    """
    refits = regime_refits(name=name, origins=origins)
    targets = daily_refits(origins=origins) if name == "daily" else tuple(origins)
    out: dict[datetime, list[datetime]] = {}
    for origin in targets:
        out.setdefault(chosen_refit(refits=refits, origin=origin), []).append(origin)
    return out


def interpolation_pairs(*, origins: list[datetime]) -> dict[datetime, list[datetime]]:
    """Return the extra (anchor, origin) pairs P-interp needs: the next three-a-year anchor.

    The next anchor trained on data after the origin, so these predictions are only for the
    screening proxy and are never scored as a forecast.
    """
    out: dict[datetime, list[datetime]] = {}
    for origin in origins:
        later = [a for a in THREE_A_YEAR if a > origin]
        if later:
            out.setdefault(min(later), []).append(origin)
    return out


def monthly_family_jobs(*, origins: list[datetime]) -> dict[datetime, list[datetime]]:
    """Return each monthly refit and the union of origins that any monthly-family regime assigns it.

    The monthly, half-yearly, three-a-year, quarterly, and yearly regimes, and P-interp's later
    anchor, all draw on the monthly refits, so each refit predicts the union of their origins.
    """
    jobs: dict[datetime, set[datetime]] = {m: set() for m in MONTHLY}
    for name in ("monthly", "half_yearly", "three_a_year", "quarterly", "yearly"):
        for refit, served_origins in served(name=name, origins=origins).items():
            jobs[refit] |= set(served_origins)
    for refit, served_origins in interpolation_pairs(origins=origins).items():
        jobs[refit] |= set(served_origins)
    return {refit: sorted(o) for refit, o in jobs.items()}
