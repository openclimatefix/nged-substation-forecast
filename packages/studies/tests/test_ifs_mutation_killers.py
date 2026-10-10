import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from studies import checkpointed_fits, ifs_lead_days
from studies.cross_validation import search_fold_offsets
from studies.ifs_decisions import stack_lead_days
from studies.ifs_ladder import (
    GROUP_CONTROL_RUNGS,
    PRODUCTION_FEED_VARIABLES,
    cloud_cover_control_features,
    cloud_layers_control_features,
    partner_control_features,
    rung_features,
    with_partner_features,
)
from studies.ifs_lead_days import (
    INSTANTANEOUS_VARIABLES,
    PARTNER_COLUMNS,
    blend_month_folds,
    find_fold_offsets,
    join_aifs_partner,
    join_forecast_at_lead_day,
    raise_unless_same_folds,
)


# --- ifs_decisions.stack_lead_days ---------------------------------------------------------------
def _losses(sites: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": sites,
            "time": [datetime(2025, 3, 1, 10)] * len(sites),
            "error": [1.0] * len(sites),
        }
    )


def test_k_d28_one_label_holding_the_separator_among_clean_ones_is_refused():
    with pytest.raises(ValueError, match="already holds"):
        stack_lead_days(losses_by_lead_day={1: _losses(["A", "B-L1"])})


def test_k_d29_the_stacked_lead_day_is_int8():
    stacked = stack_lead_days(losses_by_lead_day={1: _losses(["A"]), 2: _losses(["A"])})
    assert stacked["lead_day"].dtype == pl.Int8


# --- ifs_ladder ------------------------------------------------------------------------------------
def test_k_l4_l5_the_partner_arm_adds_the_partner_columns_and_its_control_permutes_them():
    for rung in ("f0", "f6"):
        assert with_partner_features(rung=rung) == (*rung_features(rung=rung), *PARTNER_COLUMNS)
        assert partner_control_features(rung=rung) == (
            *rung_features(rung=rung),
            *(f"{name}_shuffled" for name in PARTNER_COLUMNS),
        )


def test_k_l12_l13_the_cloud_controls_permute_their_own_groups():
    assert cloud_cover_control_features() == (*rung_features(rung="f0"), "cloud_cover_shuffled")
    assert cloud_layers_control_features() == (
        *rung_features(rung="f1"),
        "cloud_cover_low_shuffled",
        "cloud_cover_mid_shuffled",
        "cloud_cover_high_shuffled",
    )


def test_k_l17_l18_l19_the_plan_constants():
    from studies.ifs_ladder import OPEN_DATA_AVAILABILITY

    assert {n for n, s in OPEN_DATA_AVAILABILITY.items() if s == "not carried"} >= {
        "convective_inhibition",
        "visibility",
        "cape",
    }
    assert {n for n, s in OPEN_DATA_AVAILABILITY.items() if s == "carried"} >= {
        "snow_depth",
        "snowfall",
        "wind_gusts_10m",
    }
    assert PRODUCTION_FEED_VARIABLES == {
        "shortwave_radiation",
        "temperature_2m",
        "dew_point_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_10m",
    }
    assert GROUP_CONTROL_RUNGS == ("f3", "f4", "f5")


# --- ifs_lead_days.join_forecast_at_lead_day ------------------------------------------------------
def _rows(sites: list[str], time: datetime) -> pl.DataFrame:
    return pl.DataFrame({"site": sites, "time": [time] * len(sites)}).with_columns(
        pl.col("time").dt.cast_time_unit("us")
    )


def _forecasts(sites: list[str], inits: list[datetime], columns: list[str]) -> pl.DataFrame:
    rows = [
        {
            "site": site,
            "init_time": init,
            "valid_time": init,
            "lead_hours": lead,
            **{c: float((index + 1) * 100_000 + init.hour * 1000 + lead) for c in columns},
        }
        for index, site in enumerate(sites)
        for init in inits
        for lead in range(80)
    ]
    return pl.DataFrame(rows, schema_overrides={"lead_hours": pl.Int32}).with_columns(
        pl.col("init_time").dt.cast_time_unit("us"), pl.col("valid_time").dt.cast_time_unit("us")
    )


def test_k_j4_j5_the_instantaneous_set_is_the_plans():
    assert {"temperature_2m", "dew_point_2m"} <= INSTANTANEOUS_VARIABLES
    assert (
        not {
            "shortwave_radiation",
            "direct_radiation",
            "precipitation",
            "snowfall",
            "wind_gusts_10m",
        }
        & INSTANTANEOUS_VARIABLES
    )


def test_k_j4_temperature_is_averaged_and_precipitation_is_not():
    joined = join_forecast_at_lead_day(
        rows=_rows(["A"], datetime(2025, 3, 10, 13, tzinfo=UTC)),
        forecasts=_forecasts(["A"], [datetime(2025, 3, 9)], ["temperature_2m", "precipitation"]),
        lead_day=1,
    )
    assert joined["temperature_2m"].to_list() == [100_000 + 36.5]
    assert joined["precipitation"].to_list() == [100_000 + 37]


def test_k_j7_a_mix_of_00_and_12_utc_runs_is_refused():
    forecasts = _forecasts(["A"], [datetime(2025, 3, 9), datetime(2025, 3, 9, 12)], ["cloud_cover"])
    with pytest.raises(ValueError, match="00 UTC"):
        join_forecast_at_lead_day(
            rows=_rows(["A"], datetime(2025, 3, 10, 13, tzinfo=UTC)),
            forecasts=forecasts,
            lead_day=1,
        )


def test_k_j9_j10_j11_two_farms_each_get_their_own_forecast_and_the_lead_day():
    joined = join_forecast_at_lead_day(
        rows=_rows(["A", "B"], datetime(2025, 3, 10, 13, tzinfo=UTC)),
        forecasts=_forecasts(["A", "B"], [datetime(2025, 3, 9)], ["cloud_cover"]),
        lead_day=1,
    ).sort("site")
    assert joined["site"].to_list() == ["A", "B"]
    assert joined["cloud_cover"].to_list() == [100_000 + 36.5, 200_000 + 36.5]
    assert joined["lead_day"].to_list() == [1, 1]


# --- eras and folds --------------------------------------------------------------------------------
def _monthly_rows(*, first: str, last: str, sites: list[str]) -> pl.DataFrame:
    months = []
    year, month = int(first[:4]), int(first[5:])
    while f"{year}-{month:02d}" <= last:
        months.append(f"{year}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return pl.DataFrame(
        {
            "site": [s for s in sites for _ in months],
            "month": [m for _ in sites for m in months],
            "time": [datetime(int(m[:4]), int(m[5:]), 15, 12) for _ in sites for m in months],
        }
    )


def test_k_f1_find_fold_offsets_returns_the_first_simplest_design():
    rows = _monthly_rows(first="2024-03", last="2026-04", sites=["A", "B"])
    kept = rows.filter(~pl.col("month").is_in(["2024-11"]))
    assert (
        find_fold_offsets(rows=rows)
        == search_fold_offsets(frame=kept, first_months=("2024-12",))[0]
    )


def test_k_f2_find_fold_offsets_searches_without_the_dropped_months(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[pl.DataFrame] = []

    def fake(*, frame: pl.DataFrame, first_months: tuple[str, ...]) -> list[dict[int, int]]:
        seen.append(frame)
        return [{0: 0, 1: 0}]

    monkeypatch.setattr("studies.ifs_lead_days.search_fold_offsets", fake)
    find_fold_offsets(rows=_monthly_rows(first="2024-03", last="2026-06", sites=["A"]))
    assert not set(seen[0]["month"]) & {"2024-11", "2026-05", "2026-06"}


def test_k_r1_r3_consistent_folds_over_several_hours_and_farms_pass():
    t1 = datetime(2025, 3, 10, 13, tzinfo=UTC)
    t2 = t1 + timedelta(days=60)
    frame = pl.DataFrame({"site": ["A", "A", "B"], "time": [t1, t2, t1], "fold": [1, 3, 4]})
    raise_unless_same_folds(frames=[frame, frame.slice(0, 2)])


# --- AIFS partner and blend folds ----------------------------------------------------------------
def test_k_a9_two_farms_each_get_their_own_partner_value():
    time = datetime(2025, 4, 10, 12, tzinfo=UTC)
    init = datetime(2025, 4, 9, tzinfo=UTC)
    partner = pl.DataFrame(
        {
            "site": ["A", "B"],
            "time": [time, time],
            "aifs_single_day1_ghi": [100.0, 200.0],
            "aifs_single_day1_temp": [5.0, 6.0],
            "aifs_single_day1_init_time": [init, init],
        }
    ).with_columns(pl.col("time").dt.cast_time_unit("us"))
    rows = pl.DataFrame({"site": ["A", "B"], "time": [time, time]}).with_columns(
        pl.col("time").dt.cast_time_unit("us")
    )
    joined = join_aifs_partner(rows=rows, partner=partner, lead_day=1).sort("site")
    assert joined[PARTNER_COLUMNS[0]].to_list() == [100.0, 200.0]


def test_k_b3_exactly_as_many_months_as_folds_gives_one_month_per_fold():
    months = [f"2025-{m:02d}" for m in range(3, 8)]
    assert blend_month_folds(months=months) == {m: i for i, m in enumerate(months)}


def test_k_b4_thirteen_months_fall_in_blocks_of_3_3_2_3_2():
    months = [f"2025-{m:02d}" for m in range(3, 13)] + ["2026-01", "2026-02", "2026-03"]
    folds = blend_month_folds(months=months)
    assert [list(folds.values()).count(f) for f in range(5)] == [3, 3, 2, 3, 2]


# --- checkpointed_fits stamps -----------------------------------------------------------------------
def _stamp(path: Path) -> str:
    return checkpointed_fits.run_stamp(dataset_path=path, arms={"a": ("x",)}, device="cpu")


def test_k_c1_a_rewrite_within_the_same_second_changes_the_stamp(tmp_path: Path):
    dataset = tmp_path / "data.parquet"
    dataset.write_bytes(b"1234")
    second = 1_700_000_000 * 10**9
    os.utime(dataset, ns=(second, second))
    base = _stamp(dataset)
    os.utime(dataset, ns=(second + 1000, second + 1000))
    assert _stamp(dataset) != base


def test_k_c6_a_rewrite_of_another_size_with_the_same_mtime_changes_the_stamp(tmp_path: Path):
    dataset = tmp_path / "data.parquet"
    dataset.write_bytes(b"1234")
    os.utime(dataset, ns=(10**18, 10**18))
    base = _stamp(dataset)
    dataset.write_bytes(b"12345")
    os.utime(dataset, ns=(10**18, 10**18))
    assert _stamp(dataset) != base


_ = ifs_lead_days
