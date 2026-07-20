"""The NWP analysis-proxy query, shared by the dashboard and the ML feature pipeline.

We hold no weather observations, so the closest available proxy for the *true* weather over
historical times is the control ensemble member at the shortest available lead: for each location
and valid time, the freshest NWP run that had been produced. ``select_analysis_proxy`` centralises
that selection so the dashboard (which exists to show what a model sees) and the feature pipeline
(which builds weather-lag features) compute it identically.
"""

from datetime import datetime, timedelta
from typing import Final

import polars as pl

NWP_PUBLICATION_DELAY_HOURS: Final[int] = 6
"""Default delay between an NWP run's ``init_time`` and when it becomes publicly available.

A run initialised at ``init_time`` cannot be used to reconstruct the analysis at an as-of time
earlier than ``init_time + NWP_PUBLICATION_DELAY_HOURS`` — this is the delay ``select_analysis_proxy``
applies for its optional ``available_at`` leakage cut, and the same delay the feature pipeline uses
to derive ``power_fcst_init_time`` from ``nwp_init_time``.
"""

NWP_ANALYSIS_MEMBER: Final[int] = 0
"""The ensemble member the analysis proxy uses.

Member 0 of ECMWF ENS is the control run — the unperturbed forecast started from the analysis
itself — so its shortest leads are the closest thing to the analysis that the ensemble holds.
"""


def select_analysis_proxy(
    nwp: pl.LazyFrame,
    *,
    group_key: str,
    init_time_col: str = "init_time",
    member: int = NWP_ANALYSIS_MEMBER,
    max_lead: timedelta | None = None,
    available_at: datetime | None = None,
    publication_delay: timedelta = timedelta(hours=NWP_PUBLICATION_DELAY_HOURS),
) -> pl.LazyFrame:
    """Select the freshest-run analysis proxy: one row per ``(group_key, valid_time)``.

    Keeps only the control ``member``, then — for each ``(group_key, valid_time)`` — the row from
    the freshest NWP run (the latest ``init_time_col``, equivalently the shortest lead). The
    ``ensemble_member`` column is dropped from the result.

    ``pl.LazyFrame`` in / ``pl.LazyFrame`` out, so it composes with both ``pl.scan_delta`` (the
    dashboard, keyed by ``h3_index``) and in-memory post-spatial-join frames (the pipeline, keyed
    by ``time_series_id``). The pushdownable filters (``member``, ``max_lead``, ``available_at``)
    are applied *before* the freshest-run window reduction so a Delta scan's partition pruning and
    row-group skipping survive — confirm with ``.explain()`` when wiring a new scan through it.

    The one-row-per-group guarantee relies on ``(group_key, init_time_col, valid_time)`` being
    unique once ``member`` is fixed (true for the raw NWP table and for the upsampled pipeline
    frame). If a caller passes a frame where it is not, the window reduction fans out to every row
    tied at the maximum ``init_time`` rather than collapsing to one.

    Args:
        nwp: NWP rows carrying at least ``ensemble_member``, ``valid_time``, ``init_time_col`` and
            ``group_key``.
        group_key: The location key to group by — ``"h3_index"`` (dashboard) or ``"time_series_id"``
            (pipeline).
        init_time_col: Name of the NWP run's init-time column (``"init_time"`` on the raw table,
            ``"nwp_init_time"`` in the pipeline after its rename).
        member: The ensemble member to keep (the control run by default).
        max_lead: If given, keep only rows with ``valid_time < init_time + max_lead`` — the
            dashboard passes its per-run stitching window; the pipeline leaves it unbounded.
        available_at: If given, keep only runs available by this as-of time, i.e.
            ``init_time + publication_delay <= available_at`` (replay-mode availability — it models
            what a hindcast could have seen, not the live path, which reads only already-published
            runs). Guards against lookahead bias in historical hindcasts.
        publication_delay: The publication delay used by the ``available_at`` cut.

    Returns:
        The analysis-proxy rows, one per ``(group_key, valid_time)``, without ``ensemble_member``.
    """
    lf = nwp.filter(pl.col("ensemble_member") == member)
    if max_lead is not None:
        lf = lf.filter(pl.col("valid_time") < pl.col(init_time_col) + max_lead)
    if available_at is not None:
        lf = lf.filter(pl.col(init_time_col) + publication_delay <= available_at)
    # Freshest run wins per (location, valid_time): the latest init_time, equivalently the shortest
    # lead. Kept as a window filter (not a group_by/agg) so every non-key column rides through
    # untouched; see the docstring for the uniqueness invariant this relies on.
    freshest = lf.filter(
        pl.col(init_time_col) == pl.col(init_time_col).max().over([group_key, "valid_time"])
    )
    return freshest.drop("ensemble_member")
