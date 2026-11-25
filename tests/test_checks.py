"""Tests for the two data-health asset checks and their pure evaluation cores.

The pure ``evaluate_power_freshness`` cases need neither Dagster nor Delta. Each check also has
end-to-end tests that drive the real ``@asset_check`` — writing temp Delta tables and a metadata
table — so the Settings plumbing, the Delta scans and the ``AssetCheckResult`` mapping are all
exercised together.

One case goes further and drives ``power_data_is_fresh`` through Dagster's executor rather than
calling it: asserting that the *run* still succeeds when the check's internals blow up is the
only way to pin the property the catch-all exists for, and a returned ``AssetCheckResult`` cannot
show it.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from _cleaned_power_test_data import write_metadata
from contracts.common import UTC_DATETIME_DTYPE
from contracts.settings import Settings
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetSelection,
    DagsterExecutionInterruptedError,
    DagsterInstance,
    TableMetadataValue,
    build_asset_check_context,
    materialize,
)
from delta_store.cleaned_power_time_series import CleaningProvenance
from deltalake import CommitProperties, DeltaTable, write_deltalake

from nged_substation_forecast import _sentry
from nged_substation_forecast.defs import checks
from nged_substation_forecast.defs.checks import (
    PowerFreshnessResult,
    evaluate_power_freshness,
)

_NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
_THRESHOLD = timedelta(hours=24)


def _coverage(rows: dict[int, datetime]) -> pl.DataFrame:
    """Build a coverage frame (``time_series_id``, ``first_time``, ``last_time``) like the Delta
    scan returns. ``rows`` maps id -> last_time; first_time is irrelevant to freshness, so it
    is set a day earlier just to populate the column realistically."""
    last = pl.Series("last_time", list(rows.values())).cast(UTC_DATETIME_DTYPE)
    return pl.DataFrame(
        {
            "time_series_id": pl.Series(list(rows.keys()), dtype=pl.Int32),
            "first_time": last.dt.offset_by("-1d"),
            "last_time": last,
        }
    )


def _expected_id_series(ids: list[int]) -> pl.Series:
    return pl.Series("time_series_id", ids, dtype=pl.Int32)


def test_all_fresh_is_healthy() -> None:
    coverage = _coverage({1: _NOW - timedelta(hours=1), 2: _NOW - timedelta(hours=23)})
    result = evaluate_power_freshness(
        coverage=coverage, expected_ids=_expected_id_series([1, 2]), now=_NOW, threshold=_THRESHOLD
    )
    assert result.is_healthy
    assert result.n_late == 0
    assert result.n_series_total == 2
    assert result.late.is_empty()


def test_stale_series_flagged_most_stale_first() -> None:
    coverage = _coverage(
        {
            1: _NOW - timedelta(hours=1),  # fresh
            2: _NOW - timedelta(hours=30),  # stale
            3: _NOW - timedelta(hours=72),  # more stale
        }
    )
    result = evaluate_power_freshness(
        coverage=coverage,
        expected_ids=_expected_id_series([1, 2, 3]),
        now=_NOW,
        threshold=_THRESHOLD,
    )
    assert not result.is_healthy
    assert result.n_stale == 2
    assert result.n_never == 0
    assert result.n_late == 2
    # Most-stale first: id 3 (72h) before id 2 (30h).
    assert result.late["time_series_id"].to_list() == [3, 2]
    assert result.late["status"].to_list() == ["stale", "stale"]
    assert result.late["hours_late"].to_list() == pytest.approx([72.0, 30.0])


def test_the_production_staleness_threshold_is_twenty_four_hours() -> None:
    """Series 8 is a minute past the threshold and series 7 a minute short of it, so retuning
    ``_POWER_DATA_STALENESS_THRESHOLD`` has to change this test too. Every other case passes
    its own ``_THRESHOLD``, which pins the shape of the boundary but not the production
    number."""
    result = evaluate_power_freshness(
        coverage=_coverage(
            {7: _NOW - timedelta(hours=23, minutes=59), 8: _NOW - timedelta(hours=24, minutes=1)}
        ),
        expected_ids=_expected_id_series([7, 8]),
        now=_NOW,
        threshold=checks._POWER_DATA_STALENESS_THRESHOLD,
    )
    assert result.late["time_series_id"].to_list() == [8]


def test_never_reported_ids_flagged_from_expected_ids() -> None:
    """An expected id with no rows in the Delta table counts as late with a null ``last_seen``."""
    coverage = _coverage({1: _NOW - timedelta(hours=1)})
    result = evaluate_power_freshness(
        coverage=coverage,
        expected_ids=_expected_id_series([1, 2, 3]),
        now=_NOW,
        threshold=_THRESHOLD,
    )
    assert not result.is_healthy
    assert result.n_stale == 0
    assert result.n_never == 2
    assert result.n_series_total == 3
    never_rows = result.late.filter(pl.col("status") == "never")
    assert set(never_rows["time_series_id"].to_list()) == {2, 3}
    assert never_rows["last_seen"].null_count() == 2
    assert never_rows["hours_late"].null_count() == 2


def test_never_reported_sorts_before_stale() -> None:
    coverage = _coverage({1: _NOW - timedelta(hours=48)})  # stale
    result = evaluate_power_freshness(
        coverage=coverage, expected_ids=_expected_id_series([1, 2]), now=_NOW, threshold=_THRESHOLD
    )
    assert result.late["status"].to_list() == ["never", "stale"]
    assert result.late["time_series_id"].to_list() == [2, 1]


def test_no_expected_ids_cannot_detect_never_reported() -> None:
    """With no metadata table, only stale series are detectable; total is the on-disk id count."""
    coverage = _coverage({1: _NOW - timedelta(hours=48)})
    result = evaluate_power_freshness(
        coverage=coverage, expected_ids=None, now=_NOW, threshold=_THRESHOLD
    )
    assert result.n_never == 0
    assert result.n_stale == 1
    assert result.n_series_total == 1


def test_empty_table_with_expected_ids_is_all_never() -> None:
    coverage = _coverage({})
    result = evaluate_power_freshness(
        coverage=coverage, expected_ids=_expected_id_series([1, 2]), now=_NOW, threshold=_THRESHOLD
    )
    assert result.n_never == 2
    assert result.n_series_total == 2
    assert not result.is_healthy


def test_result_threshold_hours_reflects_threshold() -> None:
    result = evaluate_power_freshness(
        coverage=_coverage({}), expected_ids=None, now=_NOW, threshold=timedelta(hours=8)
    )
    assert isinstance(result, PowerFreshnessResult)
    assert result.threshold_hours == 8.0


def test_late_series_table_is_capped_but_the_counts_are_not() -> None:
    """A whole-feed stall lists only ``_MAX_LATE_SERIES_IN_TABLE`` series, most-stale first,
    while the counts beside the table stay uncapped."""
    cap = checks._MAX_LATE_SERIES_IN_TABLE
    n_late = cap + 10
    # Series `i` is `i` hours staler than series `i - 1`, so worst-first order is descending id.
    coverage = _coverage({i: _NOW - timedelta(hours=48 + i) for i in range(1, n_late + 1)})
    result = checks._to_asset_check_result(
        evaluate_power_freshness(
            coverage=coverage,
            expected_ids=_expected_id_series(list(range(1, n_late + 1))),
            now=_NOW,
            threshold=_THRESHOLD,
        )
    )

    assert result.metadata["n_late"].value == n_late  # the true count survives truncation
    assert result.metadata["n_late_listed"].value == cap
    late_table = result.metadata["late_time_series"]
    assert isinstance(late_table, TableMetadataValue)  # narrows before reading `.records`
    assert len(late_table.records) == cap
    # The `cap` most stale, not an arbitrary `cap` of them.
    listed_ids = {r.data["time_series_id"] for r in late_table.records}
    assert listed_ids == set(range(n_late - cap + 1, n_late + 1))


def test_the_table_never_holds_more_detail_than_the_sentry_event_context() -> None:
    """The durable listing may not be more detailed than the transient one.

    Both cap the same worst-first ordering, and the table is the one written to the event log
    every hour a stall lasts, so it is the table that must not outgrow the Sentry context.
    Pinning the relationship rather than the number leaves the cap free to be retuned downwards.
    """
    assert checks._MAX_LATE_SERIES_IN_TABLE <= _sentry.MAX_LATE_SERIES_IN_CONTEXT


def test_never_reported_series_crowd_stale_ones_out_of_a_truncated_table() -> None:
    """Every never-reported series outranks every stale one, so at V2 cutover — a populated
    metadata table before data flows — the table can hold no stale series at all, while the counts
    stay exact."""
    cap = checks._MAX_LATE_SERIES_IN_TABLE
    coverage = _coverage({i: _NOW - timedelta(hours=1000) for i in range(1, 11)})  # 10 very stale
    expected_ids = _expected_id_series(
        list(range(1, 11)) + list(range(100, 100 + cap + 5))
    )  # + cap+5 never-reported
    result = checks._to_asset_check_result(
        evaluate_power_freshness(
            coverage=coverage, expected_ids=expected_ids, now=_NOW, threshold=_THRESHOLD
        )
    )

    late_table = result.metadata["late_time_series"]
    assert isinstance(late_table, TableMetadataValue)
    assert {r.data["status"] for r in late_table.records} == {"never"}
    assert result.metadata["n_stale"].value == 10  # exact, though none of them is listed
    assert result.metadata["n_late"].value == cap + 15


# ---------------------------------------------------------------------------
# Silencing series that have already stopped reporting.
# ---------------------------------------------------------------------------


def test_silenced_series_are_withheld_and_others_still_warn() -> None:
    """One fixture covering every way a silenced id can relate to the data.

    7 is stale and watched, 1 is fresh and watched, 23 is stale and silenced, 33 is in the metadata
    table with no data and silenced, and 404 is silenced but exists nowhere. ``n_series_total == 2``
    is the assertion that pins *where* the filtering happens: filtering the late frame instead of
    the inputs would leave the silenced ids in the population and answer 4.
    """
    coverage = _coverage(
        {7: _NOW - timedelta(hours=48), 1: _NOW - timedelta(hours=1), 23: _NOW - timedelta(days=30)}
    )
    result = evaluate_power_freshness(
        coverage=coverage,
        expected_ids=_expected_id_series([7, 1, 23, 33]),
        now=_NOW,
        threshold=_THRESHOLD,
        silenced_ids=(23, 33, 404),
    )

    assert result.n_stale == 1
    assert result.n_never == 0
    assert result.late["time_series_id"].to_list() == [7]
    assert result.n_series_total == 2
    assert result.silenced_ids == (23, 33, 404)
    assert result.resurrected_ids == ()  # a silenced series that is *still* stale has not revived


@pytest.mark.parametrize(
    ("last_seen_offset", "expected_n_stale"),
    [
        (timedelta(0), 0),  # exactly at the cutoff is fresh: `stale` uses `<`, not `<=`
        (timedelta(seconds=-1), 1),
    ],
)
def test_the_staleness_cutoff_is_exclusive(
    last_seen_offset: timedelta, expected_n_stale: int
) -> None:
    """The half of the cutoff that resurrection is the complement of.

    Nothing about silencing here, but the test below relies on this boundary sitting exactly
    where it does: were ``stale`` to use ``<=``, a series last seen on the cutoff would be stale
    *and* resurrected at once.
    """
    result = evaluate_power_freshness(
        coverage=_coverage({7: _NOW - _THRESHOLD + last_seen_offset}),
        expected_ids=_expected_id_series([7]),
        now=_NOW,
        threshold=_THRESHOLD,
    )
    assert result.n_stale == expected_n_stale


@pytest.mark.parametrize(
    ("last_seen_offset", "expected"),
    [
        (timedelta(0), (23, 33)),  # exactly at the cutoff: not stale, so both have revived
        (timedelta(seconds=-1), ()),  # a second staler: still dead
    ],
)
def test_a_silenced_series_that_reports_again_is_resurrected(
    last_seen_offset: timedelta, expected: tuple[int, ...]
) -> None:
    """Resurrection is the exact complement of staleness at the cutoff, and is reported in id order
    rather than in whatever order the coverage frame happens to hold — 33 is the fresher row
    here, so an unsorted answer would come back as ``(33, 23)``.

    ``is_healthy`` stays true either way — a revived series is not a stale one — which is why a
    Sentry "it's back" event needs its own sender rather than this gate.
    """
    last_seen = _NOW - _THRESHOLD + last_seen_offset
    coverage = _coverage({1: _NOW, 33: last_seen, 23: last_seen})
    result = evaluate_power_freshness(
        coverage=coverage,
        expected_ids=_expected_id_series([1, 23, 33]),
        now=_NOW,
        threshold=_THRESHOLD,
        silenced_ids=(33, 23),
    )
    assert result.resurrected_ids == expected
    assert result.is_healthy


def test_silencing_works_without_expected_ids() -> None:
    """No metadata table means never-reported ids cannot be detected, but silencing still applies
    to the coverage frame — and must not trip over the absent ``expected_ids``."""
    coverage = _coverage({1: _NOW - timedelta(hours=1), 33: _NOW - timedelta(days=200)})
    result = evaluate_power_freshness(
        coverage=coverage, expected_ids=None, now=_NOW, threshold=_THRESHOLD, silenced_ids=(33,)
    )
    assert result.is_healthy
    assert result.n_series_total == 1


@pytest.mark.parametrize("stale_id", [None, 7])
def test_the_check_result_reports_silencing(stale_id: int | None) -> None:
    """The ignored ids are named on both description branches. The green one matters most: it is
    the state this feature creates, where the silencing is easiest to forget, and — since
    ``report_power_freshness`` returns early on a healthy result — where the hourly Sentry
    warning stops."""
    rows = {1: _NOW - timedelta(hours=1), 33: _NOW - timedelta(days=200)}
    if stale_id is not None:
        rows[stale_id] = _NOW - timedelta(hours=48)
    result = checks._to_asset_check_result(
        evaluate_power_freshness(
            coverage=_coverage(rows),
            expected_ids=_expected_id_series(list(rows)),
            now=_NOW,
            threshold=_THRESHOLD,
            silenced_ids=(33,),
        )
    )

    assert result.description is not None
    assert "Ignoring 1 silenced time series: 33." in result.description
    assert result.metadata["n_silenced"].value == 1
    assert result.metadata["silenced_time_series_ids"].value == "[33]"
    assert result.passed is (stale_id is None)


def test_a_resurrection_fails_the_check_and_says_where_to_edit() -> None:
    coverage = _coverage({1: _NOW - timedelta(hours=1), 33: _NOW - timedelta(hours=2)})
    result = checks._to_asset_check_result(
        evaluate_power_freshness(
            coverage=coverage,
            expected_ids=_expected_id_series([1, 33]),
            now=_NOW,
            threshold=_THRESHOLD,
            silenced_ids=(33,),
        )
    )
    assert not result.passed
    assert result.description is not None
    assert "Reporting again, so remove 33 from _SILENCED_TIME_SERIES_IDS" in result.description
    assert "_SILENCED_TIME_SERIES_IDS" in result.description


def test_a_deployment_with_no_data_yet_still_says_so_while_silencing() -> None:
    """The silenced list is never empty in production, so "watching nothing" must keep meaning
    "no data yet" — the state every fresh deployment starts in — rather than being read as the
    list having swallowed the metadata table."""
    result = checks._to_asset_check_result(
        evaluate_power_freshness(
            coverage=_coverage({}),
            expected_ids=None,
            now=_NOW,
            threshold=_THRESHOLD,
            silenced_ids=(33,),
        )
    )
    assert result.description == ("No power data on disk yet. Ignoring 1 silenced time series: 33.")
    assert not result.passed


# ---------------------------------------------------------------------------
# End-to-end: the real asset check against a temp Delta table + metadata table.
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``Settings`` at a temp data root (mirrors ``tests/test_assets.py``), and empty the
    silenced list so silencing is something a test opts into rather than inherits from
    whichever ids have really stopped reporting today."""
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    monkeypatch.setenv("DATA_PATH_DELIVERY", str(tmp_path))
    monkeypatch.setenv("LOCAL_ARTIFACTS_PATH", str(tmp_path))
    monkeypatch.setattr(checks, "_SILENCED_TIME_SERIES_IDS", ())
    return tmp_path


def _run_freshness_check() -> AssetCheckResult:
    """Invoke ``power_data_is_fresh`` directly, on an instance closed before we return.

    Dagster builds a context even though this check declares no context parameter, and never
    disposes the instance inside it:
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#fixtures-and-mocking>.
    """
    with DagsterInstance.ephemeral() as instance:
        result = checks.power_data_is_fresh(build_asset_check_context(instance=instance))
    assert isinstance(result, AssetCheckResult)  # narrows the directly-invoked check's return
    return result


def test_power_data_is_fresh_end_to_end(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One fresh series, one stale series, one never-reported id → a WARN naming all late."""
    # Freeze "now" so the fresh/stale split is deterministic regardless of wall-clock.
    now = datetime.now(UTC)
    fresh_time = now - timedelta(hours=1)
    stale_time = now - timedelta(hours=48)

    settings = Settings()
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 2], dtype=pl.Int32),
            "time": pl.Series([fresh_time, stale_time]).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0, 2.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)
    write_metadata(settings.metadata_path, dict.fromkeys([1, 2, 99], "Primary"))

    result = _run_freshness_check()

    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert result.metadata["n_stale"].value == 1  # id 2
    assert result.metadata["n_never_reported"].value == 1  # id 99
    assert result.metadata["n_late"].value == 2
    assert result.metadata["n_series_total"].value == 3
    late_table = result.metadata["late_time_series"]
    assert isinstance(late_table, TableMetadataValue)
    late_ids = {r.data["time_series_id"] for r in late_table.records}
    assert late_ids == {2, 99}
    # Under the cap nothing is truncated, so the listed count is the real one — not the cap.
    assert result.metadata["n_late_listed"].value == 2


def test_power_data_is_fresh_all_current_passes(env: Path) -> None:
    now = datetime.now(UTC)
    settings = Settings()
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 2], dtype=pl.Int32),
            "time": pl.Series([now - timedelta(hours=1), now - timedelta(hours=2)]).cast(
                UTC_DATETIME_DTYPE
            ),
            "power": pl.Series([1.0, 2.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)
    write_metadata(settings.metadata_path, dict.fromkeys([1, 2], "Primary"))

    result = _run_freshness_check()
    assert result.passed is True
    assert result.metadata["n_late"].value == 0
    # Emitted even when nothing is late, so the key keeps one type across runs and stays plottable.
    assert result.metadata["n_late_listed"].value == 0
    assert result.metadata["n_silenced"].value == 0
    assert result.metadata["silenced_time_series_ids"].value == "[]"
    # Silencing nothing says nothing: no "Ignoring 0 silenced time series: ." on every green run.
    assert result.description is not None
    assert "Ignoring" not in result.description


def test_power_data_is_fresh_silences_the_configured_series(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path, through ``Settings`` and a real Delta table: a series that would otherwise
    hold the check yellow for ever is ignored, and said to be ignored."""
    monkeypatch.setattr(checks, "_SILENCED_TIME_SERIES_IDS", (99,))
    now = datetime.now(UTC)
    settings = Settings()
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 99], dtype=pl.Int32),
            "time": pl.Series([now - timedelta(hours=1), now - timedelta(days=200)]).cast(
                UTC_DATETIME_DTYPE
            ),
            "power": pl.Series([1.0, 2.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)
    write_metadata(settings.metadata_path, dict.fromkeys([1, 99], "Primary"))

    result = checks.power_data_is_fresh()
    assert isinstance(result, AssetCheckResult)
    assert result.passed is True
    assert result.metadata["n_late"].value == 0
    assert result.metadata["n_series_total"].value == 1
    assert result.description is not None
    assert "Ignoring 1 silenced time series: 99." in result.description


def test_power_data_is_fresh_names_the_fault_notes_in_the_metadata_table(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path: a note in the metadata parquet reaches the check's description, for a
    silenced series and for a reporting one."""
    monkeypatch.setattr(checks, "_SILENCED_TIME_SERIES_IDS", (99,))
    now = datetime.now(UTC)
    settings = Settings()
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 99], dtype=pl.Int32),
            "time": pl.Series([now - timedelta(hours=1), now - timedelta(days=200)]).cast(
                UTC_DATETIME_DTYPE
            ),
            "power": pl.Series([1.0, 2.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)
    write_metadata(settings.metadata_path, dict.fromkeys([1, 99], "Primary"))
    metadata = pl.read_parquet(settings.metadata_path).with_columns(
        information=pl.Series(["Invented note one.", "Invented note two."], dtype=pl.String)
    )
    metadata.write_parquet(settings.metadata_path)

    result = checks.power_data_is_fresh()

    assert isinstance(result, AssetCheckResult)
    assert result.description is not None
    assert (
        'NGED fault notes: 1: "Invented note one."; 99: "Invented note two.".' in result.description
    )


def test_power_data_is_fresh_uses_the_production_threshold(env: Path) -> None:
    """The check's own boundary, not just the constant's value: series 7, a minute short of
    ``_POWER_DATA_STALENESS_THRESHOLD``, is fresh, and series 8, a minute past it, is stale.

    Both times are derived from the constant, so a deliberate retune leaves this test passing and
    only the wiring going astray — a literal threshold, or a shifted ``now`` — fails it.
    """
    now = datetime.now(UTC)
    threshold = checks._POWER_DATA_STALENESS_THRESHOLD
    settings = Settings()
    pl.DataFrame(
        {
            "time_series_id": pl.Series([7, 8], dtype=pl.Int32),
            "time": pl.Series(
                [now - threshold + timedelta(minutes=1), now - threshold - timedelta(minutes=1)]
            ).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0, 2.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)
    write_metadata(settings.metadata_path, dict.fromkeys([7, 8], "Primary"))

    result = checks.power_data_is_fresh()
    assert isinstance(result, AssetCheckResult)
    assert result.metadata["n_stale"].value == 1
    late_table = result.metadata["late_time_series"]
    assert isinstance(late_table, TableMetadataValue)
    assert [r.data["time_series_id"] for r in late_table.records] == [8]


def test_power_data_is_fresh_no_data_yet_warns(env: Path) -> None:
    """No Delta table and no metadata table → not healthy, WARN, 'no data yet'."""
    result = _run_freshness_check()
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert result.metadata["n_series_total"].value == 0


def test_power_data_is_fresh_hands_evaluated_result_to_sentry(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check forwards the *exact* ``PowerFreshnessResult`` it evaluated to
    ``report_power_freshness`` — reused, not recomputed. We patch ``evaluate_power_freshness``
    to return a sentinel and assert *object identity* on the reporter's argument: asserting
    counts alone would not prove reuse, since the pure function is deterministic and a
    recompute would yield equal counts and pass a weaker test. The reporter self-gates on
    health (tested in ``test_sentry.py``), so the check calls it unconditionally."""
    sentinel = PowerFreshnessResult(
        n_series_total=8,
        n_stale=7,
        n_never=0,
        threshold_hours=24.0,
        late=pl.DataFrame(
            {
                "time_series_id": pl.Series(range(7), dtype=pl.Int32),
                "last_seen": pl.Series([datetime(2020, 1, 1, tzinfo=UTC)] * 7).cast(
                    UTC_DATETIME_DTYPE
                ),
                "hours_late": pl.Series([9999.0] * 7, dtype=pl.Float64),
                "status": pl.Series(["stale"] * 7),
            }
        ),
    )
    monkeypatch.setattr(checks, "evaluate_power_freshness", lambda **_: sentinel)
    captured: list[PowerFreshnessResult] = []
    monkeypatch.setattr(
        checks, "report_power_freshness", lambda settings, result: captured.append(result)
    )

    # The check still reads coverage + metadata table before calling the (patched) evaluator, so a
    # minimal Delta table and metadata table must exist for those reads to succeed.
    now = datetime.now(UTC)
    settings = Settings()
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1], dtype=pl.Int32),
            "time": pl.Series([now - timedelta(hours=1)]).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)
    write_metadata(settings.metadata_path, dict.fromkeys([1], "Primary"))

    check_result = _run_freshness_check()
    assert len(captured) == 1
    assert captured[0] is sentinel  # the exact evaluated object, not a recomputation
    # ...and that same object drove the returned check result (n_late == n_stale + n_never == 7).
    assert check_result.metadata["n_late"].value == 7


def _write_one_fresh_series(settings: Settings) -> None:
    """Write a minimal ``power_time_series`` Delta table holding one up-to-date series."""
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1], dtype=pl.Int32),
            "time": pl.Series([datetime.now(UTC) - timedelta(hours=1)]).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)


class _FakePanic(BaseException):
    """Stands in for pyo3's ``PanicException``, which also derives from ``BaseException``.

    The real class cannot be imported: each compiled extension defines its own, and there is no
    importable ``pyo3_runtime`` module to reach them through. What matters to the checks' guard
    is only that a panic is *not* an ``Exception``, which this reproduces exactly.
    """


def _raise_inside_the_check(**_kwargs: object) -> PowerFreshnessResult:
    """Stand in for ``evaluate_power_freshness`` to simulate a bug inside the check."""
    raise RuntimeError("simulated bug inside the check")


def _panic_inside_the_check(**_kwargs: object) -> PowerFreshnessResult:
    """Stand in for ``evaluate_power_freshness`` to simulate a Rust panic inside the check."""
    raise _FakePanic("simulated rust panic inside the check")


def _never_called(check_name: str, exc: BaseException) -> None:
    """Stand in for ``report_check_degradation`` on a path that must not report to Sentry.

    Parameter names match the real function's, because the caller passes them by keyword.
    """
    raise AssertionError(
        f"report_check_degradation({check_name!r}, {exc!r}) should not have been called"
    )


def test_power_data_is_fresh_degrades_on_a_corrupt_metadata_parquet(env: Path) -> None:
    """A half-written metadata table is a realistic on-disk raiser: ``metadata.parquet`` is
    written in place, so a process killed mid-write leaves a file that exists and will not parse.
    The check must warn rather than raise — it is one step of the hooked
    ``power_time_series_and_metadata_job``.

    ``upsert_metadata`` reads the same file first and raises, but the asset swallows that so the
    power write survives. This pins the check's half: it has no metadata table to read while the
    fault lasts.

    The assertion is on our own description prefix rather than on Polars' message, which is not
    ours to pin.
    """
    settings = Settings()
    _write_one_fresh_series(settings)
    Path(settings.metadata_path).write_bytes(b"not a parquet file")

    result = _run_freshness_check()
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert "Could not evaluate power-data freshness" in str(result.description)


def test_power_data_is_fresh_never_fails_the_run(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The whole point of the catch-all, and the one property the return-value tests cannot pin.

    A raise inside this check fails the *step*, and so the hourly run, and so pages via
    ``power_time_series_and_metadata_job``'s ``sentry_capture_failure`` hook — even though the
    check is ``blocking=False``, which governs only whether a *failed* check blocks downstream
    assets, not whether an *erroring* one fails the run. Running the check through Dagster's
    executor is the only way to assert that; ``AssetSelection.checks`` runs the check step alone,
    so no asset materialises and nothing touches S3. No fixture data is needed either: with no
    tables on disk ``time_series_coverage`` returns an empty frame and
    ``_read_expected_ids_and_fault_notes`` returns ``(None, {})``, so the patched evaluator is still
    reached.
    """
    monkeypatch.setattr(checks, "evaluate_power_freshness", _raise_inside_the_check)
    reported: list[tuple[str, BaseException]] = []
    monkeypatch.setattr(
        checks,
        "report_check_degradation",
        lambda check_name, exc: reported.append((check_name, exc)),
    )

    result = materialize(
        [checks.power_data_is_fresh],
        selection=AssetSelection.checks(checks.power_data_is_fresh),
        instance=dagster_instance,
        raise_on_error=False,
    )

    assert result.success
    (evaluation,) = result.get_asset_check_evaluations()
    assert evaluation.passed is False
    assert evaluation.severity == AssetCheckSeverity.WARN
    assert "simulated bug inside the check" in str(evaluation.description)
    # Not failing the run means the Sentry failure hook no longer fires, so the handler must send
    # the exception itself or the fault reaches nobody outside Dagster's Checks view.
    assert [name for name, _ in reported] == ["power_data_is_fresh"]
    assert isinstance(reported[0][1], RuntimeError)


def test_power_data_is_fresh_degrades_on_a_rust_panic(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the guard catches ``BaseException`` rather than ``Exception``.

    A panic in any of the pyo3 extensions this check reads through — Polars, delta-rs (via
    ``delta_table_exists``) or obstore (via ``object_exists``) — is not an ``Exception``, so a
    narrower guard would let it through and fail the hourly run. This is the test that fails if
    someone tidies the guard back down to ``except Exception``; the cancellation test below
    cannot catch that, because ``DagsterExecutionInterruptedError`` is not an ``Exception``
    either and so propagates out of a narrow guard on its own.
    """
    monkeypatch.setattr(checks, "evaluate_power_freshness", _panic_inside_the_check)
    reported: list[tuple[str, BaseException]] = []
    monkeypatch.setattr(
        checks,
        "report_check_degradation",
        lambda check_name, exc: reported.append((check_name, exc)),
    )

    result = _run_freshness_check()
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert "simulated rust panic inside the check" in str(result.description)
    assert [name for name, _ in reported] == ["power_data_is_fresh"]


def test_power_data_is_fresh_re_raises_a_cancelled_run(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one thing the catch-all must *not* swallow.

    It catches ``BaseException``, because a pyo3 panic derives from that rather than from
    ``Exception``. Cancellation lands in the same net, so the handler re-raises it explicitly: a
    run the operator cancelled has to stop, not report itself as merely unhealthy.
    """

    def _cancel(**_kwargs: object) -> PowerFreshnessResult:
        raise DagsterExecutionInterruptedError

    monkeypatch.setattr(checks, "evaluate_power_freshness", _cancel)
    monkeypatch.setattr(checks, "report_check_degradation", _never_called)

    with pytest.raises(DagsterExecutionInterruptedError):
        _run_freshness_check()


# --- cleaned_power_keeps_up_with_raw ---------------------------------------------------------


def _write_raw_power_commits(settings: Settings, n_commits: int) -> DeltaTable:
    """Write ``n_commits`` appends to the raw power table, so its Delta version is n_commits - 1."""
    for index in range(n_commits):
        pl.DataFrame(
            {
                "time_series_id": pl.Series([1], dtype=pl.Int32),
                "time": pl.Series([datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)]).cast(
                    UTC_DATETIME_DTYPE
                ),
                "power": pl.Series([1.0], dtype=pl.Float32),
            }
        ).write_delta(settings.power_time_series_data_path, mode="append")
    return DeltaTable(settings.power_time_series_data_path)


def _write_cleaned_built_from(settings: Settings, raw_table_id: str, raw_version: int) -> None:
    provenance = CleaningProvenance(
        raw_table_id=raw_table_id,
        raw_version=raw_version,
        code_hash="hash",
        metadata_hash="metadata",
        git_sha="sha",
    )
    write_deltalake(
        settings.cleaned_power_time_series_data_path,
        pl.DataFrame({"a": [1]}).to_arrow(),
        mode="overwrite",
        commit_properties=CommitProperties(custom_metadata=provenance.to_commit_metadata()),
    )


def _run_keeps_up_check() -> AssetCheckResult:
    with DagsterInstance.ephemeral() as instance:
        result = checks.cleaned_power_keeps_up_with_raw(
            build_asset_check_context(instance=instance)
        )
    assert isinstance(result, AssetCheckResult)
    return result


@pytest.mark.parametrize("lag", [0, 1])
def test_cleaned_power_keeps_up_passes_when_at_most_one_commit_behind(env: Path, lag: int) -> None:
    settings = Settings()
    raw = _write_raw_power_commits(settings, 4)
    _write_cleaned_built_from(settings, str(raw.metadata().id), raw.version() - lag)

    result = _run_keeps_up_check()

    assert result.passed
    assert result.metadata["n_raw_commits_behind"].value == lag


def test_cleaned_power_keeps_up_warns_at_two_commits_behind(env: Path) -> None:
    settings = Settings()
    raw = _write_raw_power_commits(settings, 4)
    _write_cleaned_built_from(settings, str(raw.metadata().id), raw.version() - 2)

    result = _run_keeps_up_check()

    assert not result.passed
    assert result.severity == AssetCheckSeverity.WARN


def test_cleaned_power_keeps_up_warns_when_the_raw_table_id_differs(env: Path) -> None:
    settings = Settings()
    raw = _write_raw_power_commits(settings, 2)
    _write_cleaned_built_from(settings, "some-other-table", raw.version())

    result = _run_keeps_up_check()

    assert not result.passed
    assert result.severity == AssetCheckSeverity.WARN


def test_cleaned_power_keeps_up_passes_when_the_raw_table_is_absent(env: Path) -> None:
    result = _run_keeps_up_check()

    assert result.passed


def test_cleaned_power_keeps_up_warns_when_the_cleaned_table_is_absent(env: Path) -> None:
    _write_raw_power_commits(Settings(), 2)

    result = _run_keeps_up_check()

    assert not result.passed
    assert result.severity == AssetCheckSeverity.WARN


def test_cleaned_power_keeps_up_warns_when_the_cleaned_table_has_no_provenance(env: Path) -> None:
    settings = Settings()
    _write_raw_power_commits(settings, 2)
    write_deltalake(
        settings.cleaned_power_time_series_data_path,
        pl.DataFrame({"a": [1]}).to_arrow(),
        mode="overwrite",
    )

    result = _run_keeps_up_check()

    assert not result.passed
    assert result.severity == AssetCheckSeverity.WARN


def test_cleaned_power_keeps_up_degrades_rather_than_raising(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reported: list[str] = []
    monkeypatch.setattr(
        checks, "report_check_degradation", lambda check_name, exc: reported.append(check_name)
    )
    _write_raw_power_commits(Settings(), 2)

    def unreadable_raw_table(*args: object, **kwargs: object) -> DeltaTable:
        raise OSError("the raw table is unreadable")

    monkeypatch.setattr(checks, "DeltaTable", unreadable_raw_table)

    result = _run_keeps_up_check()

    assert not result.passed
    assert result.severity == AssetCheckSeverity.WARN
    assert reported == ["cleaned_power_keeps_up_with_raw"]


# --- NGED fault notes -----------------------------------------------------------------------


def test_the_description_names_each_fault_note_whether_or_not_the_series_is_silenced() -> None:
    result = checks.evaluate_power_freshness(
        coverage=_coverage({1: _NOW}),
        expected_ids=pl.Series([1, 33], dtype=pl.Int32),
        now=_NOW,
        threshold=_THRESHOLD,
        silenced_ids=(33,),
    )

    description = checks._describe_power_freshness(
        result, {1: "Analogues suspect.", 33: "Analogues not working."}
    )

    assert description.endswith(
        'NGED fault notes: 1: "Analogues suspect."; 33: "Analogues not working.".'
    )
    assert "Ignoring 1 silenced time series: 33." in description


def test_the_description_is_unchanged_when_there_are_no_fault_notes() -> None:
    result = checks.evaluate_power_freshness(
        coverage=_coverage({1: _NOW}), expected_ids=None, now=_NOW, threshold=_THRESHOLD
    )

    assert checks._describe_power_freshness(result, {}) == checks._describe_power_freshness(result)


def test_read_expected_ids_and_fault_notes_returns_the_non_null_notes(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    pl.DataFrame(
        {"time_series_id": [1, 2, 3], "information": ["Invented note.", None, "Another note."]}
    ).write_parquet(metadata_path)

    ids, notes = checks._read_expected_ids_and_fault_notes(str(metadata_path), None)

    assert ids is not None
    assert ids.to_list() == [1, 2, 3]
    assert notes == {1: "Invented note.", 3: "Another note."}


def test_read_expected_ids_and_fault_notes_copes_with_a_table_without_the_column(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    pl.DataFrame({"time_series_id": [1, 2]}).write_parquet(metadata_path)

    ids, notes = checks._read_expected_ids_and_fault_notes(str(metadata_path), None)

    assert ids is not None
    assert ids.to_list() == [1, 2]
    assert notes == {}


def test_read_expected_ids_and_fault_notes_copes_with_no_metadata_table(tmp_path: Path) -> None:
    assert checks._read_expected_ids_and_fault_notes(str(tmp_path / "absent.parquet"), None) == (
        None,
        {},
    )
