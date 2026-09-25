import sys
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from nwp_forecast_charts import (  # noqa: E402
    DIAMOND_DAYS,
    KEY_COLUMNS,
    KEY_LABELS,
    KEY_ROW_PX,
    LEAD_COLOURS,
    MAX_LINE_DAY,
    PRODUCT_COLOURS,
    PRODUCT_NAMES,
    check_single_device,
    combine_losses,
    extra_arm_devices,
    line_key,
)


def _losses(arms: list[str], device: str | None = None) -> pl.DataFrame:
    frame = pl.DataFrame({"arm": arms, "value": list(range(len(arms)))})
    return frame if device is None else frame.with_columns(device=pl.lit(device))


def test_an_arm_in_two_extra_folders_raises_naming_both_devices():
    first = _losses(["ens_mean_day5"], device="cuda")
    second = _losses(["ens_mean_day5"], device="cpu")

    with pytest.raises(ValueError, match=r"ens_mean_day5.*cuda.*cpu"):
        extra_arm_devices(extras=[first, second])


def test_arms_in_different_extra_folders_keep_their_own_device():
    devices = extra_arm_devices(extras=[_losses(["a_day5"], "cuda"), _losses(["b_day7"], "cuda")])

    assert devices == {"a_day5": "cuda", "b_day7": "cuda"}


def test_an_extra_folder_without_a_device_column_counts_as_the_gpu():
    assert extra_arm_devices(extras=[_losses(["a_day5"])]) == {"a_day5": "cuda"}


def test_one_arm_on_two_devices_inside_one_folder_raises():
    mixed = pl.concat([_losses(["a_day5"], "cuda"), _losses(["a_day5"], "cpu")])

    with pytest.raises(ValueError, match="a_day5"):
        extra_arm_devices(extras=[mixed])


def test_marks_from_the_gpu_and_the_cpu_raise():
    with pytest.raises(ValueError, match="mix devices"):
        check_single_device(
            arms=["ens_mean_day1", "ens_mean_day5"],
            extra_devices={"ens_mean_day5": "cuda"},
            published_arms={"ens_mean_day1"},
        )


def test_marks_that_all_come_from_the_gpu_pass():
    check_single_device(
        arms=["ens_mean_day1", "ens_mean_day5"],
        extra_devices={"ens_mean_day1": "cuda", "ens_mean_day5": "cuda"},
        published_arms={"ens_mean_day1"},
    )


def test_a_mark_with_no_fit_at_all_raises():
    with pytest.raises(ValueError, match="unknown"):
        check_single_device(arms=["ens_mean_day1"], extra_devices={}, published_arms=set())


def test_the_leaderboard_takes_an_arm_both_hold_from_the_extra_folder():
    published = _losses(["ens_mean_day1", "ukv_day1"])
    extra = _losses(["ens_mean_day1"]).with_columns(value=pl.lit(99))

    board = combine_losses(published=published, extras=[extra], prefer_extras=True)

    assert board.filter(pl.col("arm") == "ens_mean_day1")["value"].to_list() == [99]
    assert board.filter(pl.col("arm") == "ukv_day1")["value"].to_list() == [1]


def test_the_other_figures_keep_the_published_arm_and_append_the_new_ones():
    published = _losses(["ens_mean_day1"])
    extra = _losses(["ens_mean_day1", "ens_mean_day7"], device="cuda").with_columns(
        value=pl.lit(99)
    )

    losses = combine_losses(published=published, extras=[extra], prefer_extras=False)

    assert losses.sort("arm")["value"].to_list() == [0, 99]
    assert "device" not in losses.columns


def test_day_seven_has_a_grey_colour_and_a_diamond_and_the_lines_stop_at_day_three():
    assert 7 in LEAD_COLOURS
    assert 7 in DIAMOND_DAYS
    assert len(set(LEAD_COLOURS.values())) == len(LEAD_COLOURS)
    assert MAX_LINE_DAY == 3


def test_the_ifs_hres_row_reuses_the_ifs_025_colour_and_has_a_short_key_label() -> None:
    name = PRODUCT_NAMES["ifs_single"]

    assert name == "IFS HRES (9 km, Open-Meteo)"
    assert PRODUCT_COLOURS[name] == PRODUCT_COLOURS["IFS 0.25°"]
    assert len(KEY_LABELS[name]) < len(name)


def test_a_key_of_more_entries_than_a_row_holds_wraps_and_grows_taller() -> None:
    labels = [f"product {index}" for index in range(KEY_COLUMNS + 4)]

    wrapped = line_key(labels=labels, colours=["#000000"] * len(labels), columns=KEY_COLUMNS)
    single = line_key(labels=labels[:KEY_COLUMNS], colours=["#000000"] * KEY_COLUMNS)

    assert wrapped.height == single.height + KEY_ROW_PX
