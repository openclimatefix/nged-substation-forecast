"""Tests for the two profiles of `studies/weather_downloads/fetch_ukv_ceda.py`.

No test touches the network or `data/`. Each test is built to fail on the bug it exists for: the
default product drifting from the store that running processes already write, a `T120` slot grid
that accepts a 00 UTC run, a `T120` file set missing a file, and a commit that lands in the wrong
slot of a store opened under the other profile.
"""

import dataclasses
import importlib.util
import sys
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import pytest

pytest.importorskip("icechunk")
pytest.importorskip("zarr")
pytest.importorskip("pyproj")

SCRIPT_DIR: Final[Path] = Path(__file__).parent.parent / "studies" / "weather_downloads"
T120_TAGS: Final[tuple[str, ...]] = (
    "Wholesale1T120",
    "Wholesale2T120",
    "Wholesale3T120",
)


def _load() -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "fetch_ukv_ceda", SCRIPT_DIR / "fetch_ukv_ceda.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fetch = _load()


@pytest.fixture(autouse=True)
def _restore_default_profile() -> Iterator[None]:
    fetch.set_profile(fetch.DEFAULT_PROFILE)
    yield
    fetch.set_profile(fetch.DEFAULT_PROFILE)


def test_default_profile_is_unchanged() -> None:
    assert fetch.active_profile() is fetch.DEFAULT_PROFILE
    assert fetch.slot_for(datetime(2019, 9, 2, 6, tzinfo=UTC)) == 5
    assert fetch.FILE_TAGS == (
        "Wholesale1",
        "Wholesale2",
        "Wholesale3",
        "Wholesale4",
        "Wholesale1T54",
        "Wholesale2T54",
        "Wholesale3T54",
    )
    assert fetch.active_profile().file_tags == fetch.FILE_TAGS
    assert fetch.N_STEPS == 55
    assert fetch.active_profile().n_steps == 55


def test_t120_profile_slot_grid() -> None:
    fetch.set_profile(fetch.T120_PROFILE)
    assert fetch.slot_for(datetime(2019, 9, 1, 3, tzinfo=UTC)) == 0
    assert fetch.slot_for(datetime(2019, 9, 1, 15, tzinfo=UTC)) == 1
    with pytest.raises(ValueError, match="slot grid"):
        fetch.slot_for(datetime(2019, 9, 2, 0, tzinfo=UTC))


def test_t120_profile_files_and_steps() -> None:
    fetch.set_profile(fetch.T120_PROFILE)
    tags = fetch.active_profile().file_tags
    assert len(tags) == 10
    assert tags[-3:] == T120_TAGS
    assert tags[:7] == fetch.FILE_TAGS
    assert fetch.active_profile().n_steps == 121
    spec = next(spec for spec in fetch.FIELDS if spec.variable == "precipitation_amount")
    assert spec.expected_steps(tag="Wholesale1T120") == tuple(range(57, 121, 3))
    gust = next(spec for spec in fetch.FIELDS if spec.variable == "gust_10m")
    assert gust.tags == ("Wholesale4",)


def test_run_times_follow_the_profile() -> None:
    days = {"start": date(2024, 11, 1), "end": date(2024, 11, 2)}
    default = list(fetch.run_times(**days))
    assert [time.hour for time in default] == [0, 6, 12, 18] * 2
    fetch.set_profile(fetch.T120_PROFILE)
    t120 = list(fetch.run_times(**days))
    assert [time.hour for time in t120] == [3, 15, 3, 15]
    assert list(fetch.run_times(**days, newest_first=True)) == t120[::-1]


def _tiny_grid() -> object:
    return fetch.CellGrid(
        row_start=0,
        col_start=0,
        n_rows=1,
        n_cols=2,
        latitude=np.array([52.0, 52.0]),
        longitude=np.array([-1.0, -0.98]),
    )


def _fake_run(init_time: datetime, *, n_files: int) -> Any:
    n_steps = fetch.active_profile().n_steps
    return fetch.RunResult(
        init_time=init_time,
        status=fetch.STATUS_COMPLETE,
        files_expected=n_files,
        files_received=n_files,
        blocks={"temperature_1p5m": np.full((n_steps, 2), 280.0, dtype=np.float32)},
    )


@pytest.mark.parametrize(
    ("profile", "init_time", "slot", "n_steps"),
    [
        (fetch.DEFAULT_PROFILE, datetime(2019, 9, 2, 6, tzinfo=UTC), 5, 55),
        (fetch.T120_PROFILE, datetime(2019, 9, 2, 15, tzinfo=UTC), 3, 121),
    ],
)
def test_commit_run_newest_first_grows_the_axis_and_leaves_gaps(
    tmp_path: Path, profile: Any, init_time: datetime, slot: int, n_steps: int
) -> None:
    fetch.set_profile(profile)
    store = fetch.UkvStore.open(store_path=tmp_path / "store")
    store.initialise(_tiny_grid())
    assert len(store.statuses()) == 0
    assert fetch._status_at(store.statuses(), init_time) == 0
    store.commit_run(_fake_run(init_time, n_files=len(profile.file_tags)))
    statuses = store.statuses()
    assert len(statuses) == slot + 1
    assert statuses[slot] == fetch.STATUS_COMPLETE
    assert not statuses[:slot].any()
    group = fetch.zarr.open_group(store.repository.readonly_session(branch="main").store, mode="r")
    assert group["temperature_1p5m"].shape == (slot + 1, n_steps, 2)
    assert group.attrs["product"] == profile.product_name
    assert int(group["init_time"][slot]) == int(init_time.timestamp())
    assert fetch._status_at(statuses, init_time) == fetch.STATUS_COMPLETE


def test_merge_run_expects_ten_files_under_the_t120_profile(tmp_path: Path) -> None:
    fetch.set_profile(fetch.T120_PROFILE)
    init_time = datetime(2024, 11, 1, 3, tzinfo=UTC)
    run_dir = tmp_path / "product" / "_scratch" / "20241101T03"
    report = {"found": {}, "problems": []}
    tags = fetch.active_profile().file_tags
    for tag in tags[:-1]:
        fetch.cache_file(run_dir, tag=tag, arrays={}, report=report)
    partial = fetch.merge_run(run_dir, init_time=init_time, files_expected=len(tags))
    assert (partial.status, partial.files_received) == (fetch.STATUS_PARTIAL, 9)
    fetch.cache_file(run_dir, tag=tags[-1], arrays={}, report=report)
    complete = fetch.merge_run(run_dir, init_time=init_time, files_expected=len(tags))
    assert (complete.status, complete.files_received) == (fetch.STATUS_COMPLETE, 10)


def test_step_constants() -> None:
    assert (*range(37, 49), 51, 54) == fetch.T54_STEPS
    assert tuple(range(57, 121, 3)) == fetch.T120_STEPS


@pytest.mark.parametrize(
    ("opened_by", "written_by"),
    [
        (fetch.T120_PROFILE, fetch.DEFAULT_PROFILE),
        (fetch.DEFAULT_PROFILE, fetch.T120_PROFILE),
    ],
)
def test_store_refuses_the_other_profile(tmp_path: Path, opened_by: Any, written_by: Any) -> None:
    fetch.set_profile(written_by)
    store = fetch.UkvStore.open(store_path=tmp_path / "store")
    store.initialise(_tiny_grid())
    fetch.set_profile(opened_by)
    with pytest.raises(RuntimeError, match="the store's product is"):
        store.initialise(_tiny_grid())


@pytest.mark.parametrize("profile", [fetch.DEFAULT_PROFILE, fetch.T120_PROFILE])
def test_store_reopens_under_its_own_profile(tmp_path: Path, profile: Any) -> None:
    fetch.set_profile(profile)
    fetch.UkvStore.open(store_path=tmp_path / "store").initialise(_tiny_grid())
    fetch.UkvStore.open(store_path=tmp_path / "store").initialise(_tiny_grid())


def test_store_refuses_a_different_slot_spacing(tmp_path: Path) -> None:
    store = fetch.UkvStore.open(store_path=tmp_path / "store")
    store.initialise(_tiny_grid())
    fetch.set_profile(dataclasses.replace(fetch.DEFAULT_PROFILE, cycle_hours=12))
    with pytest.raises(RuntimeError, match="cycle_hours"):
        store.initialise(_tiny_grid())


def test_older_slot_committed_after_a_newer_slot_keeps_both(tmp_path: Path) -> None:
    store = fetch.UkvStore.open(store_path=tmp_path / "store")
    store.initialise(_tiny_grid())
    n_files = len(fetch.active_profile().file_tags)
    newer, older = datetime(2019, 9, 2, 6, tzinfo=UTC), datetime(2019, 9, 1, 12, tzinfo=UTC)
    for init_time, value in ((newer, 281.0), (older, 279.0)):
        run = _fake_run(init_time, n_files=n_files)
        run.blocks["temperature_1p5m"][:] = value
        store.commit_run(run)
    assert len(store.statuses()) == 6
    group = fetch.zarr.open_group(store.repository.readonly_session(branch="main").store, mode="r")
    assert group["init_time"].shape == (6,)
    assert float(group["temperature_1p5m"][5, 0, 0]) == 281.0
    assert float(group["temperature_1p5m"][2, 0, 0]) == 279.0


class _FakeMessage(dict):
    """A GRIB message reduced to the keys `_semantic_fault` reads."""


def _accumulation(monkeypatch: pytest.MonkeyPatch, *, start_step: int) -> _FakeMessage:
    monkeypatch.setattr(fetch, "_scaled_key", lambda handle, key: handle[key])
    return _FakeMessage(typeOfStatisticalProcessing=1, startStep=start_step)


def test_semantic_fault_accepts_a_three_hour_interval_at_lead_57(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = next(spec for spec in fetch.FIELDS if spec.variable == "precipitation_amount")
    assert fetch._semantic_fault(_accumulation(monkeypatch, start_step=54), spec, step=57) is None


def test_semantic_fault_rejects_a_one_hour_interval_at_lead_57(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = next(spec for spec in fetch.FIELDS if spec.variable == "precipitation_amount")
    fault = fetch._semantic_fault(_accumulation(monkeypatch, start_step=56), spec, step=57)
    assert fault == "precipitation_amount covers 1 h at lead 57, expected 3"


def test_problems_at_t120_leads() -> None:
    fetch.set_profile(fetch.T120_PROFILE)
    specs = fetch._specs_for_tag("Wholesale1T120")
    found = {spec.variable: list(spec.expected_steps(tag="Wholesale1T120")) for spec in specs}
    assert fetch._problems(found, tag="Wholesale1T120") == []
    found[specs[0].variable] = found[specs[0].variable][1:]
    assert fetch._problems(found, tag="Wholesale1T120") == [
        f"Wholesale1T120: {specs[0].variable} has 1 leads missing"
    ]


def _run_with(status: int, problems: tuple[str, ...]) -> object:
    return fetch.RunResult(
        init_time=datetime(2024, 11, 1, tzinfo=UTC),
        status=status,
        files_expected=7,
        files_received=6,
        blocks={},
        problems=problems,
    )


def test_partial_streak_aborts_after_five_identical_partial_runs() -> None:
    streak = fetch.PartialStreak()
    partial = _run_with(fetch.STATUS_PARTIAL, ("a: x absent",))
    assert [streak.record(partial) for _ in range(5)] == [False] * 4 + [True]


def test_partial_streak_resets_on_complete_or_different_problems() -> None:
    streak = fetch.PartialStreak()
    partial = _run_with(fetch.STATUS_PARTIAL, ("a: x absent",))
    other = _run_with(fetch.STATUS_PARTIAL, ("b: y absent",))
    complete = _run_with(fetch.STATUS_COMPLETE, ())
    for _ in range(4):
        streak.record(partial)
    assert not streak.record(complete)
    for _ in range(4):
        assert not streak.record(partial)
    assert not streak.record(other)
    assert streak.length == 1


def test_partial_streak_ignores_files_not_received() -> None:
    streak = fetch.PartialStreak()
    missing = _run_with(
        fetch.STATUS_PARTIAL,
        (f"Wholesale1T120{fetch.FILE_NOT_RECEIVED}", f"Wholesale2T120{fetch.FILE_NOT_RECEIVED}"),
    )
    assert not any(streak.record(missing) for _ in range(20))
    real = _run_with(fetch.STATUS_PARTIAL, ("a: x absent", f"b{fetch.FILE_NOT_RECEIVED}"))
    assert [streak.record(real) for _ in range(5)] == [False] * 4 + [True]


def test_partial_streak_is_neutral_on_missing_only_runs() -> None:
    streak = fetch.PartialStreak()
    missing = _run_with(fetch.STATUS_PARTIAL, (f"Wholesale1T120{fetch.FILE_NOT_RECEIVED}",))
    real = _run_with(fetch.STATUS_PARTIAL, ("a: x absent",))
    results = []
    for _ in range(5):
        results.append(streak.record(real))
        results.append(streak.record(missing))
    assert results == [False] * 8 + [True, False]
    assert streak.length == 5


def test_partial_streak_survives_missing_only_runs_between_identical_problems() -> None:
    streak = fetch.PartialStreak()
    missing = _run_with(fetch.STATUS_PARTIAL, (f"Wholesale1T120{fetch.FILE_NOT_RECEIVED}",))
    real = _run_with(fetch.STATUS_PARTIAL, ("a: x absent",))
    streak.record(real)
    streak.record(real)
    streak.record(missing)
    assert streak.length == 2
    assert streak.problems == frozenset({"a: x absent"})
