import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from build_forecast_inputs import EXTRA_LEAD_BUILDS, ExtraBatchType, extra_ens_ways  # noqa: E402
from fit_extra_leads import (  # noqa: E402
    BATCHES,
    NEW_PREFIXES,
    REFERENCE_PREFIXES,
    ROW_SET_REFERENCE_ARM,
    SECOND_NEW_PREFIXES,
    SECOND_REFERENCE_PREFIXES,
    arm_rows,
    batch_prefixes,
    check_context_arms,
    check_saved_losses_hold_arms,
    contrast_arms,
    domain_prefixes,
    fit_arms,
    intersection_contrast_line,
    noise_floor_lines,
    row_set_diagnostic,
    shared_rows,
)
from nwp_forecast_comparison import METRIC, DomainType, arm_columns  # noqa: E402


def test_wind_prefixes_exclude_the_solar_only_products():
    wind = domain_prefixes(domain="wind", prefixes=(*NEW_PREFIXES, *REFERENCE_PREFIXES))

    assert not [p for p in wind if p.startswith(("arpege", "arome"))]
    assert "ens_mean_day5" in wind


def test_solar_prefixes_keep_every_arm():
    prefixes = (*NEW_PREFIXES, *REFERENCE_PREFIXES)

    assert domain_prefixes(domain="solar", prefixes=prefixes) == prefixes


def test_saved_losses_missing_a_new_arm_raise():
    arms = domain_prefixes(domain="wind", prefixes=(*NEW_PREFIXES, *REFERENCE_PREFIXES))
    complete = pl.DataFrame({"arm": list(arms)})
    check_saved_losses_hold_arms(
        losses=complete, domain="wind", path=Path("wind_losses.parquet"), batch=BATCHES["first"]
    )

    incomplete = complete.filter(pl.col("arm") != NEW_PREFIXES[0])
    with pytest.raises(ValueError, match=NEW_PREFIXES[0]):
        check_saved_losses_hold_arms(
            losses=incomplete,
            domain="wind",
            path=Path("wind_losses.parquet"),
            batch=BATCHES["first"],
        )


def test_saved_losses_of_the_first_batch_lack_the_second_batchs_arms():
    first = pl.DataFrame({"arm": list(batch_prefixes(batch=BATCHES["first"], domain="solar"))})

    with pytest.raises(ValueError, match="ens_mean_day7"):
        check_saved_losses_hold_arms(
            losses=first,
            domain="solar",
            path=Path("solar_losses.parquet"),
            batch=BATCHES["second"],
        )


def test_the_two_batches_fit_disjoint_arms():
    first = set(NEW_PREFIXES) | set(REFERENCE_PREFIXES)
    second = set(SECOND_NEW_PREFIXES) | set(SECOND_REFERENCE_PREFIXES)

    assert not first & second


def test_the_second_batch_leaves_the_control_members_first_batch_arms_alone():
    second = set(SECOND_NEW_PREFIXES)

    assert {"ens_control_day0", "ens_control_day1"}.isdisjoint(second)
    assert {"ens_control_day2", "ens_control_day3", "ens_control_day14"} <= second


def test_the_second_batch_leaves_the_days_the_plan_omits_absent():
    arms = set(batch_prefixes(batch=BATCHES["second"], domain="solar"))

    assert not {arm for arm in arms if arm.startswith(("ifs025", "gfs"))} - {
        "ifs025_day2",
        "gfs_day2",
    }
    assert not {arm for arm in arms if arm.endswith(("_day10", "_day14"))} - {
        "ens_control_day10",
        "ens_control_day14",
    }


def test_the_second_batch_drops_the_solar_only_products_for_wind():
    wind = batch_prefixes(batch=BATCHES["second"], domain="wind")

    assert not [arm for arm in wind if arm.startswith(("arpege", "arome"))]
    assert "icon_eu_day3" in wind


def test_the_second_build_adds_the_control_member_and_reads_no_previous_runs():
    build = EXTRA_LEAD_BUILDS["second"]

    assert build.ens_control_days == (5, 7, 10, 14)
    assert build.ens_mean_days == (7,)
    assert build.gefs_days == (7,)
    assert not build.product_day_offsets


def test_ens_ways_follow_the_days_each_reduction_is_wanted_at():
    kwargs = {"mean_days": (7,), "control_days": (5, 7)}

    assert extra_ens_ways(day=7, **kwargs) == ("mean", "control")
    assert extra_ens_ways(day=5, **kwargs) == ("control",)
    assert extra_ens_ways(day=10, **kwargs) == ()


def _all_contrast_arms(*, batch: ExtraBatchType) -> set[str]:
    selected = BATCHES[batch]
    pairs = (
        *selected.same_product_contrasts,
        *selected.ensemble_contrasts,
        *selected.open_meteo_gfs_contrasts,
        *selected.ifs_025_contrasts,
        *selected.icon_eu_contrasts,
        *selected.elsewhere_contrasts,
    )
    return {arm for pair in pairs for arm in pair}


def test_the_third_batch_fits_eight_native_gfs_arms_per_technology():
    for domain in ("solar", "wind"):
        arms = batch_prefixes(batch=BATCHES["third"], domain=domain)

        assert arms == tuple(f"gfs_native_day{day}" for day in (0, 1, 2, 3, 5, 7, 10, 14))


def test_the_third_batch_refits_nothing_the_earlier_batches_fitted():
    third = set(BATCHES["third"].new_prefixes)
    earlier = (
        set(NEW_PREFIXES)
        | set(REFERENCE_PREFIXES)
        | set(SECOND_NEW_PREFIXES)
        | set(SECOND_REFERENCE_PREFIXES)
    )

    assert not third & earlier
    assert not BATCHES["third"].reference_prefixes


def test_every_arm_a_third_batch_contrast_names_is_fitted_in_some_batch():
    fitted = (
        set(BATCHES["third"].new_prefixes)
        | set(NEW_PREFIXES)
        | set(REFERENCE_PREFIXES)
        | set(SECOND_NEW_PREFIXES)
        | set(SECOND_REFERENCE_PREFIXES)
    )

    assert _all_contrast_arms(batch="third") <= fitted


def test_the_third_batch_contrasts_open_meteo_gfs_at_days_one_to_three_five_and_seven():
    pairs = BATCHES["third"].open_meteo_gfs_contrasts

    assert pairs == tuple((f"gfs_native_day{day}", f"gfs_day{day}") for day in (1, 2, 3, 5, 7))


def test_the_third_batch_contrasts_each_day_with_the_ens_mean_of_the_same_day():
    assert all(
        treatment.removeprefix("gfs_native_") == reference.removeprefix("ens_mean_")
        for treatment, reference in BATCHES["third"].ensemble_contrasts
    )


def test_a_batch_with_no_reference_arms_writes_no_noise_floor_section():
    empty = pl.DataFrame({"arm": []}, schema={"arm": pl.String})

    lines = noise_floor_lines(
        domain="solar", losses=empty, published=empty, batch=BATCHES["third"], header=[]
    )

    assert lines == []


def _earlier_arms(*, batch: ExtraBatchType, domain: DomainType) -> set[str]:
    selected = BATCHES[batch]
    return contrast_arms(batch=selected, domain=domain) - set(
        batch_prefixes(batch=selected, domain=domain)
    )


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_the_first_two_batches_complete_the_third_batch_contrasts(domain: DomainType) -> None:
    check_context_arms(
        domain=domain,
        context_arms=[
            set(batch_prefixes(batch=BATCHES["first"], domain=domain)),
            set(batch_prefixes(batch=BATCHES["second"], domain=domain)),
        ],
        batch=BATCHES["third"],
    )


@pytest.mark.parametrize("batch", ["second", "third"])
def test_a_contrast_arm_fitted_by_no_batch_raises(batch: ExtraBatchType) -> None:
    arms = _earlier_arms(batch=batch, domain="solar")
    with pytest.raises(ValueError, match="fitted by no batch"):
        check_context_arms(domain="solar", context_arms=[arms - {min(arms)}], batch=BATCHES[batch])


def test_an_arm_in_two_batches_raises() -> None:
    arms = _earlier_arms(batch="third", domain="solar")
    with pytest.raises(ValueError, match="fitted twice"):
        check_context_arms(
            domain="solar", context_arms=[arms, {"ens_mean_day7"}], batch=BATCHES["third"]
        )


def test_the_fourth_batch_fits_six_ifs_single_runs_arms_and_no_day_ten():
    for domain in ("solar", "wind"):
        arms = batch_prefixes(batch=BATCHES["fourth"], domain=domain)

        assert arms == tuple(f"ifs_single_day{day}" for day in (0, 1, 2, 3, 5, 7))
    assert not BATCHES["fourth"].reference_prefixes


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_the_first_two_batches_complete_the_fourth_batch_contrasts(domain: DomainType) -> None:
    check_context_arms(
        domain=domain,
        context_arms=[
            set(batch_prefixes(batch=BATCHES["first"], domain=domain)),
            set(batch_prefixes(batch=BATCHES["second"], domain=domain)),
        ],
        batch=BATCHES["fourth"],
    )


def test_the_first_batch_alone_leaves_a_fourth_batch_contrast_arm_unfitted() -> None:
    with pytest.raises(ValueError, match="fitted by no batch"):
        check_context_arms(
            domain="solar",
            context_arms=[set(batch_prefixes(batch=BATCHES["first"], domain="solar"))],
            batch=BATCHES["fourth"],
        )


def test_the_fourth_batch_contrasts_ifs_025_at_the_days_it_is_fitted_and_icon_eu_at_one_to_three():
    batch = BATCHES["fourth"]

    assert batch.ifs_025_contrasts == tuple(
        (f"ifs_single_day{day}", f"ifs025_day{day}") for day in (1, 2, 3, 5, 7)
    )
    assert batch.icon_eu_contrasts == tuple(
        (f"ifs_single_day{day}", f"icon_eu_day{day}") for day in (1, 2, 3)
    )
    assert batch.drop_gap_rows
    assert not any(BATCHES[name].drop_gap_rows for name in ("first", "second", "third"))


def test_a_batch_that_drops_gap_rows_drops_only_the_rows_with_a_null_in_the_arms_columns():
    frame = pl.DataFrame(
        {"a": [1.0, None, 3.0, 4.0], "b": [1.0, 2.0, None, 4.0], "other": [None, 1, 2, 3]}
    )

    assert arm_rows(frame=frame, columns=("a", "b"), drop_gap_rows=True)["a"].to_list() == [
        1.0,
        4.0,
    ]
    assert arm_rows(frame=frame, columns=("a", "b"), drop_gap_rows=False).height == 4


def _losses(*, rows: dict[str, list[int]], errors: dict[str, float]) -> pl.DataFrame:
    """Two seeds of per-row losses: each arm holds the given days (one site) at its own error."""
    records = []
    for arm, days in rows.items():
        for day in days:
            time = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=day)
            records.extend(
                {
                    "arm": arm,
                    "site": "A",
                    "time": time,
                    "seed": seed,
                    "month": time.strftime("%Y-%m"),
                    METRIC: errors[arm] + 0.001 * day,
                }
                for seed in (0, 1)
            )
    return pl.DataFrame(records)


def test_shared_rows_keeps_only_the_rows_both_arms_hold():
    losses = _losses(
        rows={"first": [0, 1, 2, 3], "second": [2, 3, 4]}, errors={"first": 0.1, "second": 0.2}
    )

    shared = shared_rows(losses=losses, treatment="first", reference="second")

    assert sorted(shared["time"].dt.day().unique().to_list()) == [3, 4]
    assert shared.filter(pl.col("arm") == "first").height == 4
    assert shared.filter(pl.col("arm") == "second").height == 4


def test_an_intersection_contrast_prints_the_shared_row_count_and_both_absolute_errors():
    rows_first = list(range(48))
    rows_second = list(range(24, 72))
    losses = _losses(
        rows={"first": rows_first, "second": rows_second}, errors={"first": 0.10, "second": 0.20}
    )

    line = intersection_contrast_line(losses=losses, treatment="first", reference="second")

    assert line is not None
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    # Days 24 to 47 are shared: 24 rows a seed, in January and February.
    assert cells[0] == "first − second"
    assert cells[4] == "24"
    assert cells[5] == "2"
    assert abs(float(cells[2]) - (10.0 + 0.1 * 35.5)) < 0.01
    assert abs(float(cells[3]) - (20.0 + 0.1 * 35.5)) < 0.01


def test_an_intersection_contrast_with_an_absent_arm_is_none():
    losses = _losses(rows={"first": [0, 1]}, errors={"first": 0.1})

    assert intersection_contrast_line(losses=losses, treatment="first", reference="gone") is None


class _RecordedFit:
    """A stand-in for `out_of_fold_losses` that records the rows each fit is given."""

    def __init__(self) -> None:
        self.rows: list[pl.DataFrame] = []

    def __call__(self, *, site_rows: pl.DataFrame, **_: object) -> pl.DataFrame:
        self.rows.append(site_rows)
        return pl.DataFrame({"loss": [0.0]})


def _rows_with_one_gap() -> pl.DataFrame:
    """One site's rows, one of which has a null in the arm's columns."""
    columns = arm_columns(domain="wind", prefixes=("ifs_single_day1",))
    return pl.DataFrame(
        {
            "site": ["A"] * 3,
            **{
                column: [1.0, None if column.endswith("speed_10m") else 1.0, 1.0]
                for column in columns
            },
        }
    )


def test_a_gap_dropping_fit_hands_the_model_no_null_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _RecordedFit()
    monkeypatch.setattr("fit_extra_leads.out_of_fold_losses", recorded)

    fit_arms(
        frame=_rows_with_one_gap(),
        domain="wind",
        prefixes=("ifs_single_day1",),
        workers=1,
        drop_gap_rows=True,
    )

    assert [rows.height for rows in recorded.rows] == [2]
    assert not any(recorded.rows[0].null_count().row(0))


def test_a_fit_that_keeps_gap_rows_hands_the_model_the_null_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _RecordedFit()
    monkeypatch.setattr("fit_extra_leads.out_of_fold_losses", recorded)

    fit_arms(
        frame=_rows_with_one_gap(),
        domain="wind",
        prefixes=("ifs_single_day1",),
        workers=1,
    )

    assert [rows.height for rows in recorded.rows] == [3]


def test_only_the_gap_dropping_batch_asks_for_dropped_rows() -> None:
    assert [name for name, batch in BATCHES.items() if batch.drop_gap_rows] == ["fourth"]


def test_the_row_set_diagnostic_measures_the_reference_arm_on_both_row_sets() -> None:
    losses = _losses(
        rows={ROW_SET_REFERENCE_ARM: list(range(48)), "ifs_single_day1": list(range(24, 72))},
        errors={ROW_SET_REFERENCE_ARM: 0.10, "ifs_single_day1": 0.20},
    )

    line = row_set_diagnostic(losses=losses, gap_arm="ifs_single_day1")

    assert line is not None
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    # All 48 days average 0.1 + 0.001 * 23.5, and the 24 shared days 0.1 + 0.001 * 35.5.
    assert cells[0] == ROW_SET_REFERENCE_ARM
    assert cells[3] == "+1.200"
    assert abs(float(cells[1]) - 12.35) < 0.01
    assert abs(float(cells[2]) - 13.55) < 0.01
    assert cells[4:] == ["48", "24"]


def test_the_row_set_diagnostic_is_none_without_the_gap_arm() -> None:
    losses = _losses(rows={ROW_SET_REFERENCE_ARM: [0, 1]}, errors={ROW_SET_REFERENCE_ARM: 0.1})

    assert row_set_diagnostic(losses=losses, gap_arm="ifs_single_day1") is None
