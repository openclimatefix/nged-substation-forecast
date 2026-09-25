import sys
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
    SECOND_NEW_PREFIXES,
    SECOND_REFERENCE_PREFIXES,
    batch_prefixes,
    check_context_arms,
    check_saved_losses_hold_arms,
    contrast_arms,
    domain_prefixes,
    noise_floor_lines,
)
from nwp_forecast_comparison import DomainType  # noqa: E402


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
