import json
import subprocess
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
from studies.sources import ML_EXPERIMENTS_DATA_DIR, ML_EXPERIMENTS_DOCS_DIR, REPO_DATA_DIR

from studies import experiment_store

COMMIT = "0123456789abcdef"
_real_code_state = experiment_store._code_state


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
    stale_listings = iter([[], [], []])

    def listing(*, folder: Path) -> list[int]:
        stale = next(stale_listings, None)
        return real_listing(folder=folder) if stale is None else stale

    monkeypatch.setattr(experiment_store, "_experiment_ids_in", listing)

    assert _reserve(tmp_path=tmp_path) == 2


@pytest.mark.parametrize(
    "slug", ["", "a/b", "..", "Upper", "two--hyphens", "-lead", "trail-", "has space", "a\n"]
)
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
        pl.DataFrame({"valid_time": [datetime(2026, 1, 1).date()]}),
        pl.DataFrame({"valid_time": ["2026-01-01"]}),
    ],
    ids=["no valid_time", "naive valid_time", "date valid_time", "string valid_time"],
)
def test_a_frame_without_a_utc_valid_time_is_refused(tmp_path: Path, frame: pl.DataFrame):
    experiment_id = _reserve(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="valid_time"):
        _write(tmp_path=tmp_path, experiment_id=experiment_id, frame=frame)


@pytest.mark.parametrize(
    "name", ["../escape", "Upper", "a--b", "a__b", "a-", "_a", "", "a\n", ".hidden"]
)
def test_a_bad_set_name_is_refused(tmp_path: Path, name: str):
    experiment_id = _reserve(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="set name"):
        _write(tmp_path=tmp_path, experiment_id=experiment_id, name=name)


def test_a_set_name_with_underscores_and_hyphens_is_accepted(tmp_path: Path):
    experiment_id = _reserve(tmp_path=tmp_path)

    assert _write(
        tmp_path=tmp_path, experiment_id=experiment_id, name="era5_pre-trained-v2"
    ).is_dir()


def test_the_id_search_reads_every_source_and_takes_the_largest(tmp_path: Path):
    (tmp_path / "docs" / "00003-a").mkdir(parents=True)
    (tmp_path / "data" / "00007-b").mkdir(parents=True)
    (tmp_path / "data" / ".ids" / "00005").mkdir(parents=True)

    assert _reserve(tmp_path=tmp_path) == 8


@pytest.mark.parametrize(
    ("where", "name"),
    [("data", "00009-x"), ("data/.ids", "00009"), ("docs", "00009-x")],
)
def test_each_source_of_ids_counts_alone(tmp_path: Path, where: str, name: str):
    (tmp_path / where / name).mkdir(parents=True)

    assert _reserve(tmp_path=tmp_path) == 10


@pytest.mark.parametrize("junk", ["000012-x", "00012x", "a00012-x", "abc", "0012-x"])
def test_a_folder_that_is_not_five_digits_then_a_hyphen_is_not_an_id(tmp_path: Path, junk: str):
    (tmp_path / "docs" / junk).mkdir(parents=True)

    assert _reserve(tmp_path=tmp_path) == 1


def test_two_folders_with_one_id_or_none_make_the_experiment_unreadable(tmp_path: Path):
    data_dir = tmp_path / "data"
    (data_dir / "00004-a").mkdir(parents=True)
    (data_dir / "00004-b").mkdir()

    for experiment_id in (4, 5):
        with pytest.raises(ValueError, match="expected one folder"):
            _write(tmp_path=tmp_path, experiment_id=experiment_id)


def test_the_manifest_records_the_row_set_it_was_given(tmp_path: Path):
    experiment_id = _reserve(tmp_path=tmp_path)
    _write(tmp_path=tmp_path, experiment_id=experiment_id, row_set_id="another_row_set")

    read_predictions(
        experiment_id=experiment_id,
        name="arm-a",
        row_set_id="another_row_set",
        data_dir=tmp_path / "data",
    )


def test_one_valid_time_before_the_holdout_does_not_excuse_another_after_it(tmp_path: Path):
    experiment_id = _reserve(tmp_path=tmp_path)
    frame = pl.DataFrame(
        {"valid_time": [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 7, 2, tzinfo=UTC)]}
    )

    with pytest.raises(ValueError, match="holdout"):
        _write(tmp_path=tmp_path, experiment_id=experiment_id, frame=frame)


def test_the_stores_folders_are_named_as_the_skill_says():
    assert ML_EXPERIMENTS_DATA_DIR.parent == REPO_DATA_DIR
    assert ML_EXPERIMENTS_DATA_DIR.name == "ml-experiments"
    assert ML_EXPERIMENTS_DOCS_DIR.parts[-2:] == ("docs", "ml-experiments")


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


def test_arms_that_do_not_hold_the_same_rows_are_refused():
    short = _arm(loss=1.0).head(5)

    with pytest.raises(ValueError, match="same rows"):
        losses_for_bootstrap(arm_losses={"a": _arm(loss=1.0), "b": short})


@pytest.mark.parametrize("column", ["arm", "site", "time", "month"])
def test_a_frame_that_already_has_a_built_column_is_refused(column: str):
    with pytest.raises(ValueError, match=column):
        losses_for_bootstrap(arm_losses={"a": _arm(loss=1.0).with_columns(**{column: pl.lit("x")})})


def test_arms_whose_seeds_are_listed_in_different_orders_are_compared():
    first = _arm(loss=1.0, seeds=(0, 1))
    second = _arm(loss=2.0, seeds=(1, 0)).sort("seed", descending=True)

    assert losses_for_bootstrap(arm_losses={"a": first, "b": second}).height == 48


def test_two_rows_that_differ_only_in_seed_are_not_duplicates():
    assert losses_for_bootstrap(arm_losses={"a": _arm(loss=1.0, seeds=(0, 1))}).height == 24


def test_a_first_arm_missing_rows_that_a_later_arm_has_is_refused():
    with pytest.raises(ValueError, match="same rows"):
        losses_for_bootstrap(arm_losses={"a": _arm(loss=1.0).head(5), "b": _arm(loss=1.0)})


def _git(*, repo: Path, arguments: list[str]) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def test_the_code_state_is_the_commit_of_the_checkout_and_whether_it_is_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _git(repo=tmp_path, arguments=["init", "-q"])
    (tmp_path / "file.txt").write_text("one")
    _git(repo=tmp_path, arguments=["add", "file.txt"])
    _git(
        repo=tmp_path,
        arguments=["-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "one"],
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setattr(experiment_store, "PROJECT_ROOT", tmp_path)

    assert _real_code_state() == (head, False)

    (tmp_path / "file.txt").write_text("two")

    assert _real_code_state() == (head, True)

    (tmp_path / "untracked.txt").write_text("new")
    _git(repo=tmp_path, arguments=["checkout", "--", "file.txt"])

    assert _real_code_state() == (head, True)
