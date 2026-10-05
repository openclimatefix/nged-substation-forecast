from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from studies.cross_validation import N_FOLDS, UKV_UPGRADE_MONTH

from studies import solar_product_frames


def test_a_products_column_takes_its_name_without_the_unit_suffix():
    assert solar_product_frames.named("bhi_w_m2", "icon_eu") == "bhi_icon_eu"


def test_eras_split_at_the_ukv_upgrade_month_and_folds_are_cut_within_each_era():
    pre_months = [*(f"2025-{month:02d}" for month in range(3, 13)), "2026-01"]
    post_months = [f"2026-{month:02d}" for month in range(2, 8)]
    assert post_months[0] == UKV_UPGRADE_MONTH
    months = [*pre_months, *post_months]
    frame = pl.DataFrame({"site": ["A"] * len(months), "month": months})

    result = solar_product_frames.with_eras(frame=frame)

    assert result["era"].to_list() == ["pre"] * 11 + ["post"] * 6
    assert result["era_code"].to_list() == [0] * 11 + [1] * 6
    for era in ("pre", "post"):
        folds = result.filter(era=era)["fold"].to_list()
        assert folds == sorted(folds)
        assert sorted(set(folds)) == list(range(N_FOLDS))


def test_common_rows_drops_zero_half_hour_hours_the_corrupt_block_and_the_upgrade_tail(
    monkeypatch: pytest.MonkeyPatch,
):
    start, end = solar_product_frames.ICON_EU_CORRUPT_BLOCK
    times = {
        "kept": datetime(2025, 5, 1, 12, tzinfo=UTC),
        "zero_hour": datetime(2025, 5, 1, 13, tzinfo=UTC),
        "corrupt_block": start + timedelta(hours=1),
        "after_corrupt_block": end + timedelta(hours=1),
        "upgrade_tail": solar_product_frames.UPGRADE_DAY + timedelta(days=3),
        "february": datetime(2026, 2, 1, tzinfo=UTC),
    }
    frame = pl.DataFrame({"site": ["A"] * len(times), "time": list(times.values())})
    hourly = pl.DataFrame(
        {"site": ["A"], "time": [times["zero_hour"]], "has_zero_half_hour": [True]}
    )
    monkeypatch.setattr(solar_product_frames, "pv_sites", pl.DataFrame)
    monkeypatch.setattr(solar_product_frames, "solar_hourly_power", lambda *, sites: hourly)

    result = solar_product_frames.common_rows(frame=frame)

    assert result["time"].to_list() == sorted(
        [times["kept"], times["after_corrupt_block"], times["february"]]
    )
