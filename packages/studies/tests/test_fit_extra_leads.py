import sys
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from build_forecast_inputs import EXTRA_LEAD_BUILDS, extra_ens_ways  # noqa: E402
from fit_extra_leads import (  # noqa: E402
    BATCHES,
    NEW_PREFIXES,
    REFERENCE_PREFIXES,
    SECOND_NEW_PREFIXES,
    SECOND_REFERENCE_PREFIXES,
    batch_prefixes,
    check_first_batch_arms,
    check_saved_losses_hold_arms,
    domain_prefixes,
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


def _all_contrast_arms(domain: DomainType) -> set[str]:
    batch = BATCHES["second"]
    named = {
        arm
        for pairs in (batch.same_product_contrasts, batch.ensemble_contrasts)
        for pair in pairs
        for arm in pair
    } | set(batch.climatology_contrasts)
    own = set(batch_prefixes(batch=batch, domain=domain))
    return set(domain_prefixes(domain=domain, prefixes=tuple(sorted(named)))) - own


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_first_batch_arms_that_complete_the_contrasts_pass(domain: DomainType) -> None:
    check_first_batch_arms(
        domain=domain, first_batch_arms=_all_contrast_arms(domain), batch=BATCHES["second"]
    )


def test_a_contrast_arm_fitted_by_neither_batch_raises() -> None:
    arms = _all_contrast_arms("solar")
    with pytest.raises(ValueError, match="fitted by neither batch"):
        check_first_batch_arms(
            domain="solar", first_batch_arms=arms - {min(arms)}, batch=BATCHES["second"]
        )


def test_an_arm_in_both_batches_raises() -> None:
    batch = BATCHES["second"]
    own = batch_prefixes(batch=batch, domain="solar")[0]
    with pytest.raises(ValueError, match="both batches"):
        check_first_batch_arms(
            domain="solar", first_batch_arms=_all_contrast_arms("solar") | {own}, batch=batch
        )
