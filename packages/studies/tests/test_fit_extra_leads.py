import sys
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from fit_extra_leads import (  # noqa: E402
    NEW_PREFIXES,
    REFERENCE_PREFIXES,
    check_saved_losses_hold_arms,
    domain_prefixes,
)


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
    check_saved_losses_hold_arms(losses=complete, domain="wind", path=Path("wind_losses.parquet"))

    incomplete = complete.filter(pl.col("arm") != NEW_PREFIXES[0])
    with pytest.raises(ValueError, match=NEW_PREFIXES[0]):
        check_saved_losses_hold_arms(
            losses=incomplete, domain="wind", path=Path("wind_losses.parquet")
        )
