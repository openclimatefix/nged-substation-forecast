import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from studies.bootstrap import bootstrap_difference
from studies.experiment_store import (
    FOLD_A_TUESDAYS_V1,
    losses_for_bootstrap,
    read_predictions,
    reserve_experiment,
    write_predictions,
)

from studies import experiment_store

COMMIT = "0123456789abcdef"


@pytest.fixture(autouse=True)
def _fixed_code_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_store, "_code_state", lambda: (COMMIT, True))


def _reserve(*, tmp_path: Path, slug: str = "demo") -> int:
    return reserve_experiment(slug=slug, data_dir=tmp_path / "data", docs_dir=tmp_path / "docs")


def _frame(*, valid_time: datetime = datetime(2026, 1, 1, tzinfo=UTC)) -> pl.DataFrame:
    return pl.DataFrame({"valid_time": [valid_time], "power_fcst": [1.0]})


def _write(*, tmp_path: Path, experiment_id: int, **overrides: object) -> Path:
    arguments: dict[str, object] = {
        "experiment_id": experiment_id,
        "name": "arm-a",
        "frame": _frame(),
        "kind": "members",
        "row_set_id": FOLD_A_TUESDAYS_V1,
        "data_dir": tmp_path / "data",
    }
    return write_predictions(**(arguments | overrides))  # ty: ignore[invalid-argument-type]


def test_reservations_count_up_and_each_creates_a_folder_and_a_marker(tmp_path: Path):
    assert [_reserve(tmp_path=tmp_path, slug="a"), _reserve(tmp_path=tmp_path, slug="b")] == [1, 2]
    assert (tmp_path / "data" / "00001-a").is_dir()
    assert (tmp_path / "data" / ".ids" / "00002").is_dir()


def test_an_id_present_only_in_the_docs_folder_is_never_reused(tmp_path: Path):
    (tmp_path / "docs" / "00007-merged-long-ago").mkdir(parents=True)

    assert _reserve(tmp_path=tmp_path) == 8


def test_a_marker_taken_between_listing_and_creation_moves_the_id_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Another session took id 1 under a different slug after this session listed the folders, so
    # the stale listing proposes id 1 again and only the marker's creation can refuse it.
    (tmp_path / "data" / ".ids" / "00001").mkdir(parents=True)
    (tmp_path / "data" / "00001-someone-elses-slug").mkdir()
    real_listing = experiment_store._experiment_ids_in
    stale_listings = iter([[], []])
    monkeypatch.setattr(
        experiment_store,
        "_experiment_ids_in",
        lambda *, folder: next(stale_listings, None) or real_listing(folder=folder),
    )

    assert _reserve(tmp_path=tmp_path) == 2


@pytest.mark.parametrize("slug", ["", "a/b", "..", "Upper", "two--hyphens", "-lead", "has space"])
def test_a_bad_slug_is_rejected(tmp_path: Path, slug: str):
    with pytest.raises(ValueError, match="slug"):
        _reserve(tmp_path=tmp_path, slug=slug)


def test_a_written_set_reads_back_with_its_manifest(tmp_path: Path):
    experiment_id = _reserve(tmp_path=tmp_path)
    just_before_the_holdout = datetime(2026, 6, 30, 23, 30, tzinfo=UTC)
    frame = _frame(valid_time=just_before_the_holdout)

    destination = _write(
        tmp_path=tmp_path, experiment_id=experiment_id, frame=frame, kind="quantiles"
    )

    assert json.loads((destination / "manifest.json").read_text()) == {
        "kind": "quantiles",
        "row_set_id": FOLD_A_TUESDAYS_V1,
        "commit": COMMIT,
        "dirty": True,
    }
    assert read_predictions(
        experiment_id=experiment_id,
        name="arm-a",
        row_set_id=FOLD_A_TUESDAYS_V1,
        data_dir=tmp_path / "data",
    ).equals(frame)


def test_a_set_is_never_overwritten(tmp_path: Path):
    experiment_id = _reserve(tmp_path=tmp_path)
    _write(tmp_path=tmp_path, experiment_id=experiment_id)

    with pytest.raises(FileExistsError, match="arm-a-v2"):
        _write(tmp_path=tmp_path, experiment_id=experiment_id)


def test_an_interrupted_write_leaves_no_set_and_the_name_can_be_written_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    experiment_id = _reserve(tmp_path=tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(experiment_store.json, "dumps", _raise_interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            _write(tmp_path=tmp_path, experiment_id=experiment_id)

    assert not (tmp_path / "data" / "00001-demo" / "predictions" / "arm-a").exists()
    _write(tmp_path=tmp_path, experiment_id=experiment_id)


def _raise_interrupted(*_arguments: object, **_keywords: object) -> str:
    raise RuntimeError("interrupted")


def test_reading_a_set_on_a_different_row_set_raises(tmp_path: Path):
    experiment_id = _reserve(tmp_path=tmp_path)
    _write(tmp_path=tmp_path, experiment_id=experiment_id)

    with pytest.raises(ValueError, match="not comparable"):
        read_predictions(
            experiment_id=experiment_id,
            name="arm-a",
            row_set_id="another_row_set",
            data_dir=tmp_path / "data",
        )


@pytest.mark.parametrize(
    "valid_time", [datetime(2026, 7, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)]
)
def test_a_valid_time_on_or_after_the_holdout_start_is_refused_on_write_and_on_read(
    tmp_path: Path, valid_time: datetime
):
    experiment_id = _reserve(tmp_path=tmp_path)
    frame = _frame(valid_time=valid_time)
    with pytest.raises(ValueError, match="holdout"):
        _write(tmp_path=tmp_path, experiment_id=experiment_id, frame=frame)

    # A set written by other code, around `write_predictions`, is refused on read too.
    folder = tmp_path / "data" / "00001-demo" / "predictions" / "arm-a"
    folder.mkdir(parents=True)
    frame.write_parquet(folder / "predictions.parquet")
    manifest = {"kind": "members", "row_set_id": FOLD_A_TUESDAYS_V1, "commit": "x", "dirty": False}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="holdout"):
        read_predictions(
            experiment_id=experiment_id,
            name="arm-a",
            row_set_id=FOLD_A_TUESDAYS_V1,
            data_dir=tmp_path / "data",
        )


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"power_fcst": [1.0]}),
        pl.DataFrame({"valid_time": [datetime(2026, 1, 1)]}),
    ],
    ids=["no valid_time", "naive valid_time"],
)
def test_a_frame_without_a_utc_valid_time_is_refused(tmp_path: Path, frame: pl.DataFrame):
    experiment_id = _reserve(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="valid_time"):
        _write(tmp_path=tmp_path, experiment_id=experiment_id, frame=frame)


def test_a_bad_set_name_is_refused(tmp_path: Path):
    experiment_id = _reserve(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="set name"):
        _write(tmp_path=tmp_path, experiment_id=experiment_id, name="../escape")


def _arm(*, loss: float, seeds: tuple[int, ...] = (0,)) -> pl.DataFrame:
    # Two runs a day apart, so every valid time after the first run sits in both of them.
    runs = [datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 7, 2, tzinfo=UTC)]
    valid_times = [datetime(2025, 7, 3, tzinfo=UTC) + timedelta(hours=hour) for hour in range(3)]
    rows = [
        {
            "time_series_id": series,
            "power_fcst_init_time": run,
            "valid_time": valid_time,
            "seed": seed,
            "crps": loss,
        }
        for series in (1, 2)
        for run in runs
        for valid_time in valid_times
        for seed in seeds
    ]
    return pl.DataFrame(rows)


def test_a_valid_time_in_two_runs_gives_two_rows_not_a_many_to_many_blow_up():
    losses = losses_for_bootstrap(
        arm_losses={"treatment": _arm(loss=1.0), "control": _arm(loss=2.0)}
    )

    interval = bootstrap_difference(
        losses=losses, treatment="treatment", reference="control", metric="crps"
    )

    assert interval["n_rows"] == 12
    assert interval["difference"] == pytest.approx(-1.0)
    assert losses.filter(pl.col("arm") == "treatment")["time"].n_unique() == 6


def test_the_month_label_comes_from_the_run_and_the_time_from_run_and_valid_time():
    # The run starts on 30 June and the valid time falls in July, so the run stays in June.
    frame = pl.DataFrame(
        {
            "time_series_id": [1],
            "power_fcst_init_time": [datetime(2025, 6, 30, tzinfo=UTC)],
            "valid_time": [datetime(2025, 7, 2, tzinfo=UTC)],
            "seed": [0],
            "crps": [1.0],
        }
    )

    losses = losses_for_bootstrap(arm_losses={"a": frame})

    assert losses["month"].to_list() == ["2025-06"]
    assert losses["time"].to_list() == ["2025-06-30T00:00:00|2025-07-02T00:00:00"]


def test_arms_with_different_seeds_are_refused():
    with pytest.raises(ValueError, match="same seeds"):
        losses_for_bootstrap(
            arm_losses={"a": _arm(loss=1.0, seeds=(0, 1)), "b": _arm(loss=1.0, seeds=(0,))}
        )


def test_an_arm_with_two_rows_for_one_key_is_refused():
    duplicated = pl.concat([_arm(loss=1.0), _arm(loss=1.0).head(1)])

    with pytest.raises(ValueError, match="same run"):
        losses_for_bootstrap(arm_losses={"a": duplicated})
