"""Checks on an ensemble forecast's members before they are reduced to one value per hour."""

import polars as pl


def check_one_run_per_hour(*, hourly: pl.DataFrame, members: int) -> None:
    """Stop unless every (site, time) holds one run and each of its members exactly once.

    Averaging members is only an ensemble mean when every member comes from one run. Where a band's
    margin lets two runs reach the same hour, a mean over (site, time) silently averages both runs'
    members instead.

    Args:
        hourly: One row per (site, time, member), with `init_time` and `member`.
        members: How many members a run holds.

    Raises:
        ValueError: If an hour holds more than one run, or not every member exactly once.
    """
    counts = hourly.group_by("site", "time").agg(
        runs=pl.col("init_time").n_unique(), distinct=pl.col("member").n_unique(), rows=pl.len()
    )
    if (counts["runs"] != 1).any():
        msg = "an hour holds members of more than one run"
        raise ValueError(msg)
    if ((counts["distinct"] != members) | (counts["rows"] != members)).any():
        msg = f"an hour does not hold each of its run's {members} members exactly once"
        raise ValueError(msg)
