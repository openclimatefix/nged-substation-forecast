"""Tests for the pure helpers of `studies/weather_downloads/fetch_ukv_aws_pilot.py`.

No test touches the network or `data/`. Each is built to fail on the bug it exists for: a packed
or filled variable stored undecoded, an object key built with the wrong valid time, a crop
rectangle with rows and columns swapped, a listing that loops forever, a retry rule that retries a
fault or gives up on a transient error, a valid-time mapping that is not checked, an absent object
stored as data, and a retried day recorded complete although its files hold gaps.
"""

import datetime as dt
import importlib
import json
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree

import h5py
import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("fsspec")
pytest.importorskip("aiohttp")
pyproj = pytest.importorskip("pyproj")


@pytest.fixture(scope="module")
def pilot() -> ModuleType:
    """Import the script as a module."""
    return importlib.import_module("fetch_ukv_aws_pilot")


def test_decode_values_applies_packing_and_fill(pilot: ModuleType) -> None:
    raw = np.array([0, 10, -999], dtype=np.int16)
    decoded = pilot.decode_values(
        raw=raw, attributes={"scale_factor": "0.5", "add_offset": "100", "_FillValue": "-999"}
    )
    assert decoded[:2].tolist() == [100.0, 105.0]
    assert np.isnan(decoded[2])
    assert decoded.dtype == np.float32


def test_decode_values_leaves_unpacked_floats_alone(pilot: ModuleType) -> None:
    raw = np.array([1.5, 2.5], dtype=np.float32)
    assert pilot.decode_values(raw=raw, attributes={}).tolist() == [1.5, 2.5]


def test_decode_values_rejects_strings(pilot: ModuleType) -> None:
    with pytest.raises(TypeError):
        pilot.decode_values(raw=np.array(["a"]), attributes={})


def test_wanted_keys_builds_the_valid_time_and_lead_into_each_name(pilot: ModuleType) -> None:
    run = "20261002T2300Z"
    expected = (
        "uk-deterministic-2km/20261002T2300Z/"
        "20261003T0200Z-PT0003H00M-temperature_at_screen_level.nc"
    )
    available = {expected: 1}
    found, missing = pilot.wanted_keys(run=run, available=available)
    assert found == [(expected, 3, "temperature_at_screen_level")]
    assert len(missing) == len(pilot.LEADS_HOURS) * len(pilot.ALL_FILES) - 1


def test_crop_rectangle_rows_follow_y_and_columns_follow_x(pilot: ModuleType) -> None:
    crs = pyproj.CRS.from_proj4("+proj=laea +lat_0=54.9 +lon_0=-2.5 +x_0=0 +y_0=0 +R=6371229")
    x = np.arange(-100_000.0, 100_000.0, 2000.0)
    y = np.arange(-50_000.0, 50_000.0, 2000.0)
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x[60], y[20])
    margin = 0.01
    rectangle = pilot.compute_crop_rectangle(
        x=x, y=y, crs=crs, box_bounds=(lat - margin, lat + margin, lon - margin, lon + margin)
    )
    assert rectangle.row_start <= 20 < rectangle.row_stop
    assert rectangle.col_start <= 60 < rectangle.col_stop


def test_crop_rectangle_raises_when_the_box_is_off_the_grid(pilot: ModuleType) -> None:
    crs = pyproj.CRS.from_proj4("+proj=laea +lat_0=54.9 +lon_0=-2.5 +R=6371229")
    axis = np.arange(-1000.0, 1000.0, 2000.0)
    with pytest.raises(RuntimeError):
        pilot.compute_crop_rectangle(x=axis, y=axis, crs=crs, box_bounds=(0.0, 1.0, 0.0, 1.0))


def test_truncated_listing_without_a_token_raises(pilot: ModuleType) -> None:
    namespace = pilot._S3_NAMESPACE
    page = ElementTree.fromstring(
        f'<R xmlns="{namespace[1:-1]}"><IsTruncated>true</IsTruncated></R>'
    )
    with pytest.raises(RuntimeError):
        pilot._next_token(page=page)


def _runs_for(days: list[str], *, hours: int = 24) -> list[str]:
    return [f"{day}T{hour:02d}00Z" for day in days for hour in range(hours)]


def test_default_range_starts_at_the_first_full_day_and_ends_yesterday(pilot: ModuleType) -> None:
    runs = [
        "20241004T2100Z",
        *_runs_for(["20241005"], hours=7),
        *_runs_for(["20241006", "20241007"]),
    ]
    first, last = pilot.default_range(runs=runs, today=dt.date(2026, 10, 7))
    assert (first, last) == (dt.date(2024, 10, 6), dt.date(2026, 10, 6))


def test_default_range_raises_without_a_full_day(pilot: ModuleType) -> None:
    with pytest.raises(RuntimeError):
        pilot.default_range(runs=_runs_for(["20241005"], hours=7), today=dt.date(2026, 10, 7))


def test_day_range_is_inclusive_and_bounded(pilot: ModuleType) -> None:
    days = pilot.day_range(start=dt.date(2024, 10, 6), end=dt.date(2024, 10, 8))
    assert days == [dt.date(2024, 10, 6), dt.date(2024, 10, 7), dt.date(2024, 10, 8)]
    with pytest.raises(ValueError, match="limit"):
        pilot.day_range(start=dt.date(2024, 10, 8), end=dt.date(2024, 10, 6))
    with pytest.raises(ValueError, match="limit"):
        pilot.day_range(start=dt.date(2020, 1, 1), end=dt.date(2026, 1, 1))


def test_commit_day_writes_the_ledger_atomically(pilot: ModuleType, tmp_path: Path) -> None:
    day = dt.date(2024, 10, 6)
    path = pilot.ledger_path(product_dir=tmp_path, day=day)
    assert not path.exists()
    pilot.commit_day(product_dir=tmp_path, day=day, record={"runs": 24})
    assert json.loads(path.read_text()) == {"runs": 24}
    assert not list(path.parent.glob("*.partial"))


def test_listing_bound_covers_the_last_lead_but_not_the_next_hour(pilot: ModuleType) -> None:
    bound = pilot.listing_bound(run="20261002T2100Z")
    prefix = "uk-deterministic-2km/20261002T2100Z/"
    assert f"{prefix}20261003T0200Z-PT0005H00M-wind_speed_at_10m.nc" <= bound
    assert f"{prefix}20261003T0300Z-PT0006H00M-wind_speed_at_10m.nc" > bound


def test_list_keys_stops_paging_once_past_the_bound(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = pilot._S3_NAMESPACE[1:-1]
    calls: list[str | None] = []

    def fake_page(*, prefix: str, delimiter: str | None, token: str | None) -> ElementTree.Element:
        calls.append(token)
        key = "a/z" if token is None else "b/y"
        return ElementTree.fromstring(
            f'<R xmlns="{namespace}"><IsTruncated>{str(token is None).lower()}</IsTruncated>'
            f"<NextContinuationToken>t</NextContinuationToken>"
            f"<Contents><Key>{key}</Key><Size>1</Size></Contents></R>"
        )

    monkeypatch.setattr(pilot, "_list_page", fake_page)
    assert pilot.list_keys(prefix="a/", until="a/m") == {"a/z": 1}
    assert len(calls) == 1
    assert set(pilot.list_keys(prefix="a/", until="c")) == {"a/z", "b/y"}


def test_day_is_committed_needs_a_complete_record_covering_the_hours(
    pilot: ModuleType, tmp_path: Path
) -> None:
    day = dt.date(2024, 10, 6)
    everything = list(range(24))
    assert not pilot.day_is_committed(product_dir=tmp_path, day=day, run_hours=everything)
    pilot.commit_day(
        product_dir=tmp_path, day=day, record={"complete": False, "run_hours": everything}
    )
    assert not pilot.day_is_committed(product_dir=tmp_path, day=day, run_hours=everything)
    pilot.commit_day(product_dir=tmp_path, day=day, record={"complete": True, "run_hours": [0, 12]})
    assert pilot.day_is_committed(product_dir=tmp_path, day=day, run_hours=[0])
    assert not pilot.day_is_committed(product_dir=tmp_path, day=day, run_hours=everything)


def test_day_is_committed_treats_a_record_without_the_new_keys_as_complete(
    pilot: ModuleType, tmp_path: Path
) -> None:
    day = dt.date(2024, 10, 6)
    pilot.commit_day(product_dir=tmp_path, day=day, record={"runs": 24})
    assert pilot.day_is_committed(product_dir=tmp_path, day=day, run_hours=list(range(24)))


def test_write_durably_leaves_only_the_final_file(pilot: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "sub" / "file.bin"
    pilot.write_durably(path=path, write=lambda handle: handle.write(b"abc"))
    assert path.read_bytes() == b"abc"
    assert [item.name for item in path.parent.iterdir()] == ["file.bin"]


def test_listing_retries_a_503_then_succeeds(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = pytest.importorskip("requests")
    answers = [503, 503, 200]
    sleeps: list[float] = []

    def fake_get(*_: object, **__: object) -> object:
        response = requests.Response()
        response.status_code = answers.pop(0)
        response._content = b"<R/>"
        return response

    monkeypatch.setattr(pilot.requests, "get", fake_get)
    monkeypatch.setattr(pilot.time, "sleep", sleeps.append)
    pilot._list_page(prefix="p", delimiter=None, token=None)
    assert sleeps == [5, 10]


def test_listing_does_not_retry_a_403(pilot: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = pytest.importorskip("requests")

    def fake_get(*_: object, **__: object) -> object:
        response = requests.Response()
        response.status_code = 403
        return response

    monkeypatch.setattr(pilot.requests, "get", fake_get)
    with pytest.raises(requests.HTTPError):
        pilot._list_page(prefix="p", delimiter=None, token=None)


def _read_args(pilot: ModuleType) -> dict[str, object]:
    axes = (np.arange(2.0), np.arange(2.0))
    return {
        "key": "k",
        "run": "20261002T0300Z",
        "lead": 1,
        "rectangle": pilot.CropRectangle(0, 2, 0, 2),
        "expected_axes": axes,
    }


@pytest.mark.parametrize("fault", [FileNotFoundError, ValueError])
def test_read_object_does_not_retry_a_vanished_object_or_a_wrong_file(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch, fault: type[Exception]
) -> None:
    calls: list[int] = []

    def fake_read(**_: object) -> object:
        calls.append(1)
        raise fault

    monkeypatch.setattr(pilot, "_read_once", fake_read)
    monkeypatch.setattr(pilot.time, "sleep", lambda _: None)
    with pytest.raises(fault):
        pilot.read_object(**_read_args(pilot))
    assert len(calls) == 1


def test_read_object_retries_a_transient_error_then_succeeds(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes: list[object] = [OSError("reset"), TimeoutError(), "field"]
    sleeps: list[float] = []

    def fake_read(**_: object) -> object:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(pilot, "_read_once", fake_read)
    monkeypatch.setattr(pilot.time, "sleep", sleeps.append)
    assert pilot.read_object(**_read_args(pilot)) == "field"
    assert sleeps == [5, 10]


def test_read_object_gives_up_after_the_last_attempt(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_read(**_: object) -> object:
        calls.append(1)
        raise OSError

    monkeypatch.setattr(pilot, "_read_once", fake_read)
    monkeypatch.setattr(pilot.time, "sleep", lambda _: None)
    with pytest.raises(OSError):  # noqa: PT011
        pilot.read_object(**_read_args(pilot))
    assert len(calls) == pilot.MAX_ATTEMPTS


RUN_NAME = "20261002T0300Z"
RUN_SECONDS = int(dt.datetime(2026, 10, 2, 3, tzinfo=dt.UTC).timestamp())


def _time_file(
    *, reference: int, period: int, valid: int, units: str = "seconds since 1970-01-01 00:00:00"
) -> h5py.File:
    dataset = h5py.File("memory", "w", driver="core", backing_store=False)
    for name, value, unit in (
        ("forecast_reference_time", reference, units),
        ("forecast_period", period, "seconds"),
        ("time", valid, units),
    ):
        variable = dataset.create_dataset(name, data=value)
        variable.attrs["units"] = unit
    return dataset


def test_check_times_accepts_a_file_whose_times_match_its_name(pilot: ModuleType) -> None:
    with _time_file(
        reference=RUN_SECONDS, period=2 * 3600, valid=RUN_SECONDS + 2 * 3600
    ) as dataset:
        assert pilot._check_times(dataset=dataset, run=RUN_NAME, lead=2) == RUN_SECONDS + 7200


@pytest.mark.parametrize(
    ("reference", "period", "valid"),
    [
        (RUN_SECONDS + 3600, 7200, RUN_SECONDS + 7200),
        (RUN_SECONDS, 3600, RUN_SECONDS + 7200),
        (RUN_SECONDS, 7200, RUN_SECONDS + 3600),
    ],
)
def test_check_times_rejects_each_time_that_differs_from_the_name(
    pilot: ModuleType, reference: int, period: int, valid: int
) -> None:
    with (
        _time_file(reference=reference, period=period, valid=valid) as dataset,
        pytest.raises(ValueError, match="file says"),
    ):
        pilot._check_times(dataset=dataset, run=RUN_NAME, lead=2)


def test_check_times_rejects_a_time_unit_that_is_not_seconds(pilot: ModuleType) -> None:
    with (
        _time_file(
            reference=RUN_SECONDS,
            period=7200,
            valid=RUN_SECONDS + 7200,
            units="hours since 1970-01-01",
        ) as dataset,
        pytest.raises(ValueError, match="is not in"),
    ):
        pilot._check_times(dataset=dataset, run=RUN_NAME, lead=2)


def _field(pilot: ModuleType, *, value: float, lead: int) -> object:
    return pilot.FieldRead(
        values=np.full((2, 3), value, dtype=np.float32),
        heights_m=None,
        attributes={"units": "K"},
        time_s=RUN_SECONDS + 3600 * lead,
        time_bounds_s=None,
        wire_bytes=10,
    )


def test_assemble_run_arrays_fills_an_absent_lead_with_nan_and_the_absent_marker(
    pilot: ModuleType,
) -> None:
    name = pilot.SURFACE_FILES[0]
    results = {
        (name, lead): _field(pilot, value=280.0, lead=lead)
        for lead in pilot.LEADS_HOURS
        if lead != 2
    }
    arrays = pilot.assemble_run_arrays(
        results=results, missing=[f"2:{name}"], axes=(np.arange(3.0), np.arange(2.0))
    )
    assert np.isnan(arrays[name][2]).all()
    assert not np.isnan(arrays[name][[0, 1, 3, 4, 5]]).any()
    assert arrays[f"{name}__time_s"].tolist() == [
        RUN_SECONDS + 3600 * lead if lead != 2 else pilot.NO_BOUNDS for lead in pilot.LEADS_HOURS
    ]
    assert json.loads(str(arrays["missing"])) == [f"2:{name}"]
    assert int(arrays["wire_bytes"]) == 50
    assert pilot.SURFACE_FILES[1] not in arrays


def _stub_fetch_run(
    pilot: ModuleType, *, product_dir: Path, written: list[str], missing_now: list[str]
) -> object:
    def fake(*, run: str, missing: list[str], **_: object) -> tuple[Path, int]:
        written.append(run)
        runs_dir = product_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        np.savez(runs_dir / f"{run}.npz", missing=np.array(json.dumps(missing_now or missing)))
        return runs_dir / f"{run}.npz", 5

    return fake


def _day_arguments(pilot: ModuleType, *, runs: list[str]) -> dict[str, object]:
    return {
        "day": dt.date(2026, 10, 2),
        "plan": {run: ([("k", 0, "n")], [], 1) for run in runs},
        "run_hours": [3],
        "rectangle": pilot.CropRectangle(0, 1, 0, 1),
        "axes": (np.arange(1.0), np.arange(1.0)),
        "pool": None,
        "max_wire_bytes": 1e12,
        "wire_total": 0,
    }


def test_a_retried_day_stays_incomplete_while_a_run_file_records_a_gap(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pilot, "PRODUCT_DIR", tmp_path)
    written: list[str] = []
    monkeypatch.setattr(
        pilot,
        "fetch_run",
        _stub_fetch_run(pilot, product_dir=tmp_path, written=written, missing_now=[]),
    )
    (tmp_path / "runs").mkdir()
    # A first pass left a run file with an absent object. The fresh listing now has every object.
    np.savez(tmp_path / "runs" / f"{RUN_NAME}.npz", missing=np.array(json.dumps(["0:n"])))
    pilot._fetch_day(**_day_arguments(pilot, runs=[RUN_NAME]))
    assert written == []
    record = json.loads(
        pilot.ledger_path(product_dir=tmp_path, day=dt.date(2026, 10, 2)).read_text()
    )
    assert record["absent_objects"] == 1
    assert record["complete"] is False
    assert not pilot.day_is_committed(product_dir=tmp_path, day=dt.date(2026, 10, 2), run_hours=[3])


def test_a_day_whose_run_files_hold_no_gap_is_committed_complete(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pilot, "PRODUCT_DIR", tmp_path)
    written: list[str] = []
    monkeypatch.setattr(
        pilot,
        "fetch_run",
        _stub_fetch_run(pilot, product_dir=tmp_path, written=written, missing_now=[]),
    )
    pilot._fetch_day(**_day_arguments(pilot, runs=[RUN_NAME]))
    assert written == [RUN_NAME]
    record = json.loads(
        pilot.ledger_path(product_dir=tmp_path, day=dt.date(2026, 10, 2)).read_text()
    )
    assert record["complete"] is True
    assert record["absent_objects"] == 0
    assert record["code_version"] == pilot.CODE_VERSION
