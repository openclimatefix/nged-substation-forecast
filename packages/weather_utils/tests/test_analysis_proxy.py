"""Tests for ``weather_utils.select_analysis_proxy``.

Mirrors the leakage-test style of ``ml_core``'s ``_nullify_leaky_lags`` tests: small hand-built
frames whose expected freshest-run / availability behaviour is obvious by inspection.
"""

from datetime import UTC, datetime, timedelta

import polars as pl

from weather_utils import NWP_PUBLICATION_DELAY_HOURS, select_analysis_proxy


def _nwp(rows: list[dict]) -> pl.LazyFrame:
    """Build an NWP-like frame from ``(group, init, valid, member, temp)`` dicts."""
    return pl.DataFrame(rows).lazy()


def _row(group: int, init: datetime, valid: datetime, member: int, temp: float) -> dict:
    return {
        "group": group,
        "init_time": init,
        "valid_time": valid,
        "ensemble_member": member,
        "temperature_2m": temp,
    }


BASE = datetime(2026, 1, 1, 0, 0)


def test_freshest_run_wins_per_group_and_valid_time() -> None:
    valid = BASE + timedelta(hours=30)
    lf = _nwp(
        [
            _row(1, BASE, valid, 0, 10.0),  # 30 h lead — older run
            _row(1, BASE + timedelta(hours=24), valid, 0, 11.0),  # 6 h lead — freshest, wins
        ]
    )
    result = select_analysis_proxy(lf, group_key="group").collect()
    assert result.height == 1
    assert result["temperature_2m"].to_list() == [11.0]
    # ensemble_member is dropped from the proxy result.
    assert "ensemble_member" not in result.columns


def test_only_the_control_member_is_kept() -> None:
    valid = BASE + timedelta(hours=6)
    lf = _nwp(
        [
            _row(1, BASE, valid, 0, 10.0),  # control
            _row(1, BASE, valid, 1, 99.0),  # perturbed member — must be excluded
        ]
    )
    result = select_analysis_proxy(lf, group_key="group").collect()
    assert result["temperature_2m"].to_list() == [10.0]


def test_groups_are_independent() -> None:
    valid = BASE + timedelta(hours=6)
    fresher = BASE + timedelta(hours=3)
    lf = _nwp(
        [
            _row(1, BASE, valid, 0, 10.0),
            _row(1, fresher, valid, 0, 11.0),  # group 1 freshest
            _row(2, BASE, valid, 0, 20.0),  # group 2 has only the older run
        ]
    )
    result = select_analysis_proxy(lf, group_key="group").sort("group").collect()
    assert dict(zip(result["group"], result["temperature_2m"], strict=True)) == {1: 11.0, 2: 20.0}


def test_max_lead_bounds_each_run_strictly() -> None:
    # A run at BASE with valid times 0..30 h; max_lead=24 h keeps lead < 24 h (strict).
    valids = [BASE + timedelta(hours=h) for h in (0, 12, 24, 30)]
    lf = _nwp([_row(1, BASE, v, 0, float(i)) for i, v in enumerate(valids)])
    result = select_analysis_proxy(lf, group_key="group", max_lead=timedelta(hours=24)).collect()
    # 24 h and 30 h leads are excluded (>= 24 h); 0 h and 12 h survive.
    assert sorted(result["valid_time"].to_list()) == valids[:2]


def test_available_at_excludes_runs_not_yet_published() -> None:
    # Two runs straddle the publication cut for one valid time. The fresher run is NOT yet
    # available at the as-of time, so the older-but-published run must win — even though it is
    # not the freshest by lead. This is the leakage guard.
    valid = BASE + timedelta(hours=48)
    early = BASE  # published at BASE + delay
    late = BASE + timedelta(hours=24)  # published at BASE + 24 h + delay
    available_at = BASE + timedelta(hours=24)  # < late's publication time
    lf = _nwp(
        [
            _row(1, early, valid, 0, 10.0),
            _row(1, late, valid, 0, 11.0),
        ]
    )
    result = select_analysis_proxy(lf, group_key="group", available_at=available_at).collect()
    assert result["temperature_2m"].to_list() == [10.0]
    # Sanity: with no availability cut the freshest (late) run would have won instead.
    unbounded = select_analysis_proxy(lf, group_key="group").collect()
    assert unbounded["temperature_2m"].to_list() == [11.0]


def test_available_at_boundary_is_inclusive() -> None:
    valid = BASE + timedelta(hours=48)
    init = BASE
    available_at = init + timedelta(hours=NWP_PUBLICATION_DELAY_HOURS)  # exactly published
    lf = _nwp([_row(1, init, valid, 0, 10.0)])
    result = select_analysis_proxy(lf, group_key="group", available_at=available_at).collect()
    assert result["temperature_2m"].to_list() == [10.0]


def test_init_time_col_is_configurable() -> None:
    # The pipeline calls the function with its renamed init-time column.
    valid = BASE + timedelta(hours=6)
    lf = _nwp(
        [
            _row(1, BASE, valid, 0, 10.0),
            _row(1, BASE + timedelta(hours=3), valid, 0, 11.0),
        ]
    ).rename({"init_time": "nwp_init_time"})
    result = select_analysis_proxy(lf, group_key="group", init_time_col="nwp_init_time").collect()
    assert result["temperature_2m"].to_list() == [11.0]


def test_matches_the_group_by_agg_freshest_selection() -> None:
    # Equivalence with the group_by(...).agg(sort_by(lead).first()) form the pipeline used before
    # centralisation, on a frame with a precomputed lead column.
    valids = [BASE + timedelta(hours=h) for h in (6, 12, 18)]
    rows = []
    for i, v in enumerate(valids):
        rows.append(_row(1, BASE, v, 0, float(i)))
        rows.append(_row(1, BASE + timedelta(hours=6), v, 0, float(i) + 100))  # fresher run
    lf = _nwp(rows).with_columns(
        nwp_lead_time_hours=((pl.col("valid_time") - pl.col("init_time")).dt.total_seconds() / 3600)
    )
    old = (
        lf.filter(pl.col("ensemble_member") == 0)
        .drop("ensemble_member")
        .group_by(["group", "valid_time"])
        .agg(pl.all().sort_by("nwp_lead_time_hours").first())
        .sort(["group", "valid_time"])
        .collect()
    )
    new = select_analysis_proxy(lf, group_key="group").sort(["group", "valid_time"]).collect()
    assert new.select(sorted(new.columns)).equals(old.select(sorted(old.columns)))


def test_collapses_to_one_row_when_two_runs_tie_at_the_freshest_init_time() -> None:
    # Two rows share the same (group, valid_time) AND the same (freshest) init_time — as would
    # happen if a second nwp_model_id ever covered the same cell. The reduction must still yield
    # exactly one row rather than fanning out (which would silently double downstream feature rows).
    valid = BASE + timedelta(hours=6)
    lf = _nwp(
        [
            {**_row(1, BASE, valid, 0, 10.0), "nwp_model_id": "model_a"},
            {**_row(1, BASE, valid, 0, 20.0), "nwp_model_id": "model_b"},
        ]
    )
    result = select_analysis_proxy(lf, group_key="group").collect()
    assert result.height == 1


def test_available_at_works_with_tz_aware_datetimes() -> None:
    # The real NWP columns are tz-aware UTC; the leakage cut must work against that dtype, not just
    # the tz-naive datetimes the other tests use.
    base = datetime(2026, 1, 1, tzinfo=UTC)
    valid = base + timedelta(hours=48)
    lf = _nwp(
        [
            _row(1, base, valid, 0, 10.0),  # published at base + delay
            _row(1, base + timedelta(hours=24), valid, 0, 11.0),  # not yet available
        ]
    )
    available_at = base + timedelta(hours=24)
    result = select_analysis_proxy(lf, group_key="group", available_at=available_at).collect()
    assert result["temperature_2m"].to_list() == [10.0]
