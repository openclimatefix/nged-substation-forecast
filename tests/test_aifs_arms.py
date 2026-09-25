"""Tests for the AIFS build, verify and fit scripts under `studies/nwp_forecast_comparison/`.

Each test is written to fail on the defect it names. The scripts are imported by path because
`studies/` is not an importable package.
"""

import hashlib
import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Final

import h3.api.basic_int as h3
import numpy as np
import polars as pl
import pytest

STUDIES_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "studies"
"""The `studies/` directory, whose scripts do bare imports of their siblings."""


def _load(*, name: str) -> ModuleType:
    """Import a study script by name, with both study directories on `sys.path` while it loads."""
    paths = [str(STUDIES_DIR / "nwp_forecast_comparison"), str(STUDIES_DIR / "beam_diffuse_split")]
    sys.path[:0] = paths
    try:
        return importlib.import_module(name)
    finally:
        for path in paths:
            sys.path.remove(path)


b = _load(name="build_forecast_inputs")
efh = _load(name="ens_forecast_horizons")
fa = _load(name="fit_aifs")
v = _load(name="verify_aifs_steps")


def _store(
    path: Path, *, cells: list[tuple[int, int, float]], nan_at: tuple[int, int] | None = None
) -> Path:
    """Write a 2-run, 3x3-cell AIFS-shaped store whose 100 m u equals the cell's given value."""
    records = []
    for run in range(2):
        for lead in range(0, 79, 6):
            for i, j, u in cells:
                records.append(
                    {
                        "init_time": datetime(2025, 3, 1) + timedelta(days=run),
                        "lead_time": timedelta(hours=lead),
                        "lat_index": i,
                        "lon_index": j,
                        "downward_short_wave_radiation_flux_surface": np.nan
                        if lead == 0
                        else 200.0,
                        "temperature_2m": 10.0,
                        "wind_u_10m": 1.0,
                        "wind_v_10m": 0.0,
                        "wind_u_100m": np.nan if (i, j) == nan_at and lead == 24 else u,
                        "wind_v_100m": 0.0,
                    }
                )
    pl.DataFrame(records).with_columns(
        pl.col("init_time").cast(pl.Datetime("ns")),
        pl.col("lat_index", "lon_index").cast(pl.Int16),
        pl.selectors.float().cast(pl.Float32),
    ).write_parquet(path)
    return path


def _weights(rows: list[tuple[int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": ["W1"] * len(rows),
            "lat_index": [r[0] for r in rows],
            "lon_index": [r[1] for r in rows],
            "weight": [r[2] for r in rows],
        },
        schema={
            "site": pl.String,
            "lat_index": pl.Int16,
            "lon_index": pl.Int16,
            "weight": pl.Float64,
        },
    )


def test_aifs_members_frame_is_the_weighted_mean_of_the_cells(tmp_path: Path) -> None:
    store = _store(tmp_path / "s.parquet", cells=[(1, 1, 2.0), (1, 2, 6.0), (0, 0, 100.0)])
    frame = b.aifs_members_frame(
        store=store,
        weights=_weights([(1, 1, 0.75), (1, 2, 0.25)]),
        ensemble=False,
        first_init=datetime(2025, 3, 1, tzinfo=UTC),
    )
    assert frame["speed_100m"].unique().to_list() == [3.0]  # 0.75*2 + 0.25*6, cell (0, 0) unread
    assert frame.filter(pl.col("lead_hours") == 0)["ghi_w_m2"].null_count() == 2
    assert frame["ensemble_member"].unique().to_list() == [0]


def test_aifs_members_frame_raises_on_nan_after_the_run_filter(tmp_path: Path) -> None:
    store = _store(tmp_path / "s.parquet", cells=[(1, 1, 2.0)], nan_at=(1, 1))
    with pytest.raises(ValueError, match="NaN after the run filter"):
        b.aifs_members_frame(
            store=store,
            weights=_weights([(1, 1, 1.0)]),
            ensemble=False,
            first_init=datetime(2025, 3, 1, tzinfo=UTC),
        )


def test_h3_crop_weights_raise_when_the_hexagon_leaves_the_crop() -> None:
    cell = h3.latlng_to_cell(52.1, 0.12, 5)
    lat0, lon0 = 52.0, 0.0
    full = pl.DataFrame(
        {
            "lat_index": [i for i in range(3) for _ in range(3)],
            "lon_index": [j for _ in range(3) for j in range(3)],
            "latitude": [lat0 + 0.25 * i for i in range(3) for _ in range(3)],
            "longitude": [lon0 + 0.25 * j for _ in range(3) for j in range(3)],
        },
        schema={
            "lat_index": pl.Int16,
            "lon_index": pl.Int16,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
        },
    )
    weights = b._h3_crop_weights(site_cells={"A": cell}, grid_cells=full)
    assert weights["weight"].sum() == pytest.approx(1.0)
    used = weights.select("lon_index").unique()["lon_index"].to_list()
    with pytest.raises(ValueError, match="do not sum to 1"):
        b._h3_crop_weights(
            site_cells={"A": cell}, grid_cells=full.filter(pl.col("lon_index") != max(used))
        )


# `ens_forecast_horizons._long`'s `explode()` raises a Polars 2.0 deprecation warning, which the
# repo's warnings-as-errors setting would turn into a failure; fixing it is out of scope here.
@pytest.mark.filterwarnings("ignore:In Polars 2.0:DeprecationWarning")
def test_ens_member_arms_reads_the_run_d_days_before_at_lead_24d_plus_h(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(efh, "clear_sky_table", lambda **_: pl.DataFrame())
    records = [
        {
            "init_time": datetime(2025, 3, 1) + timedelta(days=k),
            "lead_time": timedelta(hours=lead),
            "lat_index": 1,
            "lon_index": 1,
            "downward_short_wave_radiation_flux_surface": 1.0,
            "temperature_2m": 1.0,
            "wind_u_10m": 1.0,
            "wind_v_10m": 0.0,
            "wind_u_100m": float(1000 * (k + 1) + lead),
            "wind_v_100m": 0.0,
        }
        for k in range(5)
        for lead in range(0, 79, 6)
    ]
    store = tmp_path / "s.parquet"
    pl.DataFrame(records).with_columns(
        pl.col("init_time").cast(pl.Datetime("ns")), pl.col("lat_index", "lon_index").cast(pl.Int16)
    ).write_parquet(store)
    extract = b.aifs_members_frame(
        store=store,
        weights=_weights([(1, 1, 1.0)]),
        ensemble=False,
        first_init=datetime(2025, 3, 1, tzinfo=UTC),
    )
    for frame in b.ens_member_arms(
        extract=extract,
        domain="wind",
        days=(1, 2),
        method="speed_components",
        ensemble_size=1,
        arm_name=lambda way, day: f"aifs_single_day{day}",
        ways=("control",),
        fine_step_last_lead=0,
        keep_init_time=True,
    ):
        day = int(frame.columns[2].split("_day")[1][0])
        on_step = frame.filter(pl.col("time").dt.hour() % 6 == 0)
        checked = on_step.with_columns(
            expected_run=(pl.col("time").dt.truncate("1d") - pl.duration(days=day)).dt.date(),
            run=pl.col(f"aifs_single_day{day}_init_time").dt.date(),
        ).with_columns(
            expected=1000.0 * ((pl.col("run") - pl.date(2025, 3, 1)).dt.total_days() + 1)
            + 24 * day
            + pl.col("time").dt.hour()
        )
        assert (checked["run"] == checked["expected_run"]).all()
        assert (
            checked[f"aifs_single_day{day}_speed_100m"] - checked["expected"]
        ).abs().max() < 1e-6


def test_check_runs_raises_when_an_aifs_arm_lacks_its_init_time_column() -> None:
    frame = pl.DataFrame(
        {"time": [datetime(2025, 3, 2, 12, tzinfo=UTC)], "era_code": [0]},
        schema_overrides={"era_code": pl.Int8},
    )
    with pytest.raises(ValueError, match="init_time is missing"):
        fa.check_runs(frame=frame, domain="wind", row_set="single")


def test_near_line() -> None:
    near = {"difference": 0.1, "lower_95": 0.01, "upper_95": 0.2}
    far = {"difference": 0.1, "lower_95": 0.05, "upper_95": 0.15}
    assert fa.near_line(interval=near)
    assert not fa.near_line(interval=far)


def _radiation_fixture(*, era5_hours: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return AIFS radiation equal to ERA5's 6-hour mean ending at each valid time, and ERA5."""
    hours = pl.datetime_range(
        datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 6, 10, tzinfo=UTC), "1h", eager=True
    )
    full = pl.DataFrame({"site": "A", "time": hours}).with_columns(
        era5=(12 - (pl.col("time").dt.hour() - 12).abs()).clip(0) * 50.0
    )
    valid = hours.filter(hours.dt.hour() % 6 == 0)
    truth = (
        pl.DataFrame({"site": "A", "valid_time": valid})
        .join(
            pl.concat(
                [
                    full.select("site", valid_time=pl.col("time") + pl.duration(hours=k), e="era5")
                    for k in range(6)
                ]
            ),
            on=["site", "valid_time"],
        )
        .group_by("site", "valid_time")
        .agg(ghi_w_m2=pl.col("e").mean())
        .with_columns(lead_hours=pl.lit(24))
    )
    return truth, full.filter(pl.col("time").dt.hour().is_in(era5_hours.to_list()))


def test_radiation_window_check_scores_every_offset_on_the_same_rows() -> None:
    """ERA5 with the night hours missing must leave the same rows at every offset, or none."""
    truth, era5 = _radiation_fixture(era5_hours=pl.Series(range(24)))
    table = v.radiation_window_table(aifs=truth, era5=era5)
    for label in ("all", "06"):
        n = table.filter(pl.col("valid_hours") == label)["n"]
        assert n.n_unique() == 1, f"{label}: rows per offset {n.to_list()}"
        assert n[0] > 0
    assert v.radiation_verdict(table=table) == []
    # ERA5 at daylight hours only: no valid time has all 18 hours, so the check fails, never raises.
    _, daylight = _radiation_fixture(era5_hours=pl.Series(range(5, 21)))
    empty = v.radiation_window_table(aifs=truth, era5=daylight)
    assert empty["n"].max() == 0
    assert v.radiation_verdict(table=empty)


def test_radiation_window_check_finds_a_window_labelled_at_its_start() -> None:
    """AIFS radiation equal to ERA5's mean over the six hours after its valid time must fail."""
    truth, era5 = _radiation_fixture(era5_hours=pl.Series(range(24)))
    shifted = truth.with_columns(valid_time=pl.col("valid_time") - pl.duration(hours=6))
    table = v.radiation_window_table(aifs=shifted, era5=era5)
    best = table.filter(pl.col("valid_hours") == "all").sort("mae").row(0, named=True)
    assert best["offset"] == 6
    assert v.radiation_verdict(table=table)


def test_orientation_lead_filter_builds() -> None:
    """`is_in` over a list of `pl.duration` expressions raises before any data is read."""
    frame = pl.DataFrame({"lead_time": [timedelta(hours=24), timedelta(hours=30)]})
    leads = [timedelta(hours=lead) for lead in v.ORIENTATION_LEADS]
    assert frame.filter(pl.col("lead_time").is_in(leads)).height == 1
    with pytest.raises(TypeError):
        frame.filter(
            pl.col("lead_time").is_in([pl.duration(hours=lead) for lead in v.ORIENTATION_LEADS])
        )


def _orientation_weather_dir(root: Path, *, aifs_lon_cells: int, swap_gefs: bool = False) -> Path:
    """Write synthetic AIFS Single and GEFS downloads whose shared 3x3 block has equal anomalies.

    The AIFS crop has `aifs_lon_cells` columns; its first 3 columns lie at the GEFS cells'
    coordinates. With `swap_gefs`, the GEFS data is mirrored in latitude, as a wrongly oriented
    download would be.
    """
    rng = np.random.default_rng(0)
    inits = [datetime(2026, 3, 1) + timedelta(days=d) for d in range(20)]
    leads = [timedelta(hours=lead) for lead in v.ORIENTATION_LEADS]
    values = {
        (i, j, t, lead): float(rng.normal())
        for i in range(3)
        for j in range(aifs_lon_cells)
        for t in inits
        for lead in leads
    }

    def grid(*, columns: int) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "lat_index": i,
                    "lon_index": j,
                    "latitude": 52.0 + 0.25 * i,
                    "longitude": 0.25 * j,
                }
                for i in range(3)
                for j in range(columns)
            ]
        )

    def store(*, columns: int, mirror: bool) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "init_time": t,
                    "lead_time": lead,
                    "lat_index": i,
                    "lon_index": j,
                    "ensemble_member": 0,
                    "temperature_2m": values[(2 - i if mirror else i, j, t, lead)],
                }
                for i in range(3)
                for j in range(columns)
                for t in inits
                for lead in leads
            ]
        )

    aifs_dir = root / v.AIFS_SINGLE_DIR_NAME
    gefs_dir = root / v.GEFS_WINDOW_DIR_NAME
    (gefs_dir / "_month_cache").mkdir(parents=True)
    aifs_dir.mkdir()
    grid(columns=aifs_lon_cells).write_parquet(aifs_dir / "_grid_cells.parquet")
    store(columns=aifs_lon_cells, mirror=False).write_parquet(
        aifs_dir / f"{v.AIFS_SINGLE_DIR_NAME}.parquet"
    )
    grid(columns=3).write_parquet(gefs_dir / "_grid_cells.parquet")
    store(columns=3, mirror=swap_gefs).write_parquet(
        gefs_dir / "_month_cache" / f"{v.ORIENTATION_MONTH}.parquet"
    )
    return root


def test_orientation_check_passes_on_a_wider_crop(tmp_path: Path) -> None:
    root = _orientation_weather_dir(tmp_path, aifs_lon_cells=4)
    table, failures = v.orientation_table(weather_dir=root)
    assert failures == []
    assert table.height == 8  # the 8 non-central cells of the shared 3 by 3 block


def test_orientation_check_fails_on_a_mirrored_download(tmp_path: Path) -> None:
    root = _orientation_weather_dir(tmp_path, aifs_lon_cells=4, swap_gefs=True)
    _, failures = v.orientation_table(weather_dir=root)
    assert any("mirrored" in failure for failure in failures)


def test_aifs_members_frame_raises_when_a_weighted_cell_is_missing(tmp_path: Path) -> None:
    store = _store(tmp_path / "s.parquet", cells=[(1, 1, 2.0), (1, 2, 6.0)])
    frame = pl.read_parquet(store)
    dropped = frame.filter(
        ~((pl.col("lon_index") == 2) & (pl.col("init_time") == datetime(2025, 3, 1)))
    )
    dropped.write_parquet(store)
    with pytest.raises(ValueError, match="miss a weighted cell"):
        b.aifs_members_frame(
            store=store,
            weights=_weights([(1, 1, 0.75), (1, 2, 0.25)]),
            ensemble=False,
            first_init=datetime(2025, 3, 1, tzinfo=UTC),
        )


def _losses_and_frame(*, row_set: str = "single") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return a frame of two rows and saved primary losses holding every arm of the row set."""
    spec = fa.ROW_SETS[row_set]
    arms = [*spec.arms, *(f"{arm}{fa.NO_DOY_SUFFIX}" for arm in spec.deciding or ())]
    times = [datetime(2025, 3, 1, 12), datetime(2025, 3, 1, 13)]
    frame = pl.DataFrame({"site": ["A", "A"], "time": times, "fold": [0, 0]})
    losses = pl.DataFrame(
        [
            {"setting": fa.PRIMARY, "arm": arm, "site": "A", "time": time, "fold": 0}
            for arm in arms
            for time in times
        ]
    )
    return losses, frame


def _check(
    *, tmp_path: Path, losses: pl.DataFrame, frame: pl.DataFrame, stamp_on_disk: dict | None
) -> None:
    stamp = {"inputs_sha256": "abc", "device": "cuda"}
    stamp_file = tmp_path / "losses.json"
    if stamp_on_disk is not None:
        stamp_file.write_text(json.dumps(stamp_on_disk))
    fa.check_saved_losses(
        losses=losses, frame=frame, row_set="single", stamp_file=stamp_file, stamp=stamp
    )


def test_saved_losses_pass_when_stamp_arms_and_rows_match(tmp_path: Path) -> None:
    losses, frame = _losses_and_frame()
    _check(
        tmp_path=tmp_path,
        losses=losses,
        frame=frame,
        stamp_on_disk={"inputs_sha256": "abc", "device": "cuda"},
    )


@pytest.mark.parametrize(
    "stamp_on_disk",
    [None, {"inputs_sha256": "old", "device": "cuda"}, {"inputs_sha256": "abc", "device": "cpu"}],
)
def test_saved_losses_raise_on_a_missing_or_different_stamp(
    tmp_path: Path, stamp_on_disk: dict | None
) -> None:
    losses, frame = _losses_and_frame()
    with pytest.raises(ValueError, match="another build or device"):
        _check(tmp_path=tmp_path, losses=losses, frame=frame, stamp_on_disk=stamp_on_disk)


def test_saved_losses_raise_on_other_arms(tmp_path: Path) -> None:
    losses, frame = _losses_and_frame()
    losses = losses.filter(pl.col("arm") != "aifs_single_day1")
    with pytest.raises(ValueError, match="other arms"):
        _check(
            tmp_path=tmp_path,
            losses=losses,
            frame=frame,
            stamp_on_disk={"inputs_sha256": "abc", "device": "cuda"},
        )


def test_saved_losses_raise_on_other_rows(tmp_path: Path) -> None:
    losses, frame = _losses_and_frame()
    with pytest.raises(ValueError, match="other rows or folds"):
        _check(
            tmp_path=tmp_path,
            losses=losses,
            frame=frame.head(1),
            stamp_on_disk={"inputs_sha256": "abc", "device": "cuda"},
        )


def test_saved_losses_raise_on_other_folds(tmp_path: Path) -> None:
    losses, frame = _losses_and_frame()
    frame = frame.with_columns(fold=pl.Series([0, 1]))
    with pytest.raises(ValueError, match="other rows or folds"):
        _check(
            tmp_path=tmp_path,
            losses=losses,
            frame=frame,
            stamp_on_disk={"inputs_sha256": "abc", "device": "cuda"},
        )


def test_build_stamp_hashes_both_inputs_and_records_settings_columns_and_device(
    tmp_path: Path,
) -> None:
    (tmp_path / "solar_aifs_inputs.parquet").write_bytes(b"abc")
    (tmp_path / "solar_forecast_inputs.parquet").write_bytes(b"xyz")
    stamp = fa.build_stamp(published_dir=tmp_path, aifs_dir=tmp_path, domain="solar")
    assert stamp["inputs_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert stamp["published_sha256"] == hashlib.sha256(b"xyz").hexdigest()
    assert stamp["device"] == fa.DEVICE
    assert json.loads(stamp["settings"]) == json.loads(json.dumps(fa.SETTINGS))
    columns = json.loads(stamp["columns"])
    assert columns["aifs_single_day1"] == list(
        fa.arm_features(arm="aifs_single_day1", domain="solar")
    )


def test_build_stamp_changes_with_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "solar_aifs_inputs.parquet").write_bytes(b"abc")
    (tmp_path / "solar_forecast_inputs.parquet").write_bytes(b"xyz")
    before = fa.build_stamp(published_dir=tmp_path, aifs_dir=tmp_path, domain="solar")
    changed = {name: {**setting, "n_estimators": 1} for name, setting in fa.SETTINGS.items()}
    monkeypatch.setattr(fa, "SETTINGS", changed)
    assert fa.build_stamp(published_dir=tmp_path, aifs_dir=tmp_path, domain="solar") != before


def test_check_gpu_visible_raises_when_nvidia_smi_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fa, "DEVICE", "cuda")
    monkeypatch.setattr(
        fa.subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], returncode=9)
    )
    with pytest.raises(RuntimeError, match="sees no GPU"):
        fa.check_gpu_visible()
    monkeypatch.setattr(
        fa.subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], returncode=0)
    )
    fa.check_gpu_visible()
