import polars as pl
import pytest
from studies.product_frames import (
    METRIC,
    SOLAR,
    WIND,
    IntervalRecord,
    contrast_interval,
    contrast_line,
    site_range,
)

from studies import product_frames


def _losses(*, differences: dict[str, float]) -> pl.DataFrame:
    """Build per-row losses for arms `a` and `b`, where `a` exceeds `b` by a site's difference.

    Args:
        differences: Each site's difference, in fractions of capacity, added to arm `a`.

    Returns:
        Three seeds, six months, one row per month, per site and per arm.
    """
    rows = [
        {
            "arm": arm,
            "site": site,
            "time": month,
            "seed": seed,
            "fold": month % 2,
            "month": f"2025-{month + 1:02d}",
            METRIC: 0.10 + (difference if arm == "a" else 0.0),
        }
        for site, difference in differences.items()
        for seed in range(3)
        for month in range(6)
        for arm in ("a", "b")
    ]
    return pl.DataFrame(rows)


def test_the_site_range_is_the_lowest_and_highest_generators_mean_difference():
    losses = _losses(differences={"A": 0.01, "B": 0.03, "C": 0.02})

    lowest, highest = site_range(pair=losses, treatment="a", reference="b")

    assert (lowest, highest) == pytest.approx((0.01, 0.03))


def test_an_interval_reports_percentage_points_and_which_folds_agree():
    losses = _losses(differences={"A": 0.02, "B": 0.02})

    record = contrast_interval(
        losses=losses,
        contrast=("a", "b"),
        domain=SOLAR,
        setting="pooled",
        section="planned",
    )

    assert record["difference_pp"] == pytest.approx(2.0)
    assert record["site_lowest_pp"] == pytest.approx(2.0)
    assert record["site_highest_pp"] == pytest.approx(2.0)
    assert (record["domain"], record["treatment"], record["reference"]) == ("solar", "a", "b")
    assert (record["folds_agreeing"], record["n_folds"]) == (2, 2)
    assert record["n_months"] == 6


def test_a_contrast_line_matches_the_header_columns():
    record: IntervalRecord = {
        "domain": "solar",
        "setting": "pooled",
        "section": "planned",
        "scope": "all",
        "treatment": "a",
        "reference": "b",
        "difference_pp": 1.5,
        "lower_95_pp": 1.0,
        "upper_95_pp": 2.0,
        "fold_lower_95_pp": 0.5,
        "fold_upper_95_pp": 2.5,
        "site_lowest_pp": 1.0,
        "site_highest_pp": 2.0,
        "seed_spread_pp": 0.1,
        "excludes_zero": True,
        "folds_agreeing": 4,
        "n_folds": 5,
        "n_rows": 12345,
        "n_months": 10,
    }

    line = contrast_line(record)

    assert line == (
        "| solar | pooled | all | a − b | +1.500 | [+1.000, +2.000] | **yes** | 4 of 5 | 12,345 |"
    )
    assert line.count("|") == product_frames.CONTRAST_HEADER[1].count("|")


def test_the_two_domains_name_the_same_products_the_studies_fit():
    assert SOLAR.name == "solar"
    assert WIND.name == "wind"
    assert SOLAR.single("cams").startswith("cams")
    assert SOLAR.product_of(SOLAR.single("icon_eu")) == "icon_eu"
    assert len(WIND.columns("ukv")) == 4
