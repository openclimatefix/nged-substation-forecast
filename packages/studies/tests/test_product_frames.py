import polars as pl
import pytest
from studies.bootstrap import bootstrap_difference, fold_t_interval, per_fold_differences
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


def _varying_losses() -> pl.DataFrame:
    """Build losses whose difference varies by fold, month and seed, with one fold disagreeing."""
    fold_difference = {0: 0.020, 1: 0.012, 2: -0.020}
    rows = [
        {
            "arm": arm,
            "site": site,
            "time": month,
            "seed": seed,
            "fold": month // 2,
            "month": f"2025-{month + 1:02d}",
            METRIC: 0.10
            + (
                fold_difference[month // 2] + 0.001 * seed + 0.0005 * month + site_offset
                if arm == "a"
                else 0.0
            ),
        }
        for site, site_offset in (("A", 0.0), ("B", 0.004))
        for seed in range(3)
        for month in range(6)
        for arm in ("a", "b")
    ]
    return pl.DataFrame(rows)


def test_the_site_range_is_the_lowest_and_highest_generators_mean_difference():
    losses = _losses(differences={"A": 0.01, "B": 0.03, "C": 0.02})

    lowest, highest = site_range(pair=losses, treatment="a", reference="b")

    assert (lowest, highest) == pytest.approx((0.01, 0.03))


def test_every_field_of_an_interval_is_the_underlying_statistic_in_percentage_points():
    losses = _varying_losses()
    folds = per_fold_differences(losses=losses, treatment="a", reference="b", metric=METRIC)
    interval = bootstrap_difference(losses=losses, treatment="a", reference="b", metric=METRIC)
    fold_lower, fold_upper = fold_t_interval(fold_differences=folds)
    site_lowest, site_highest = site_range(pair=losses, treatment="a", reference="b")

    record = contrast_interval(
        losses=losses,
        contrast=("a", "b"),
        domain=WIND,
        setting="sensitivity",
        section="exploratory",
        scope="ukv era",
    )

    assert record == {
        "domain": "wind",
        "setting": "sensitivity",
        "section": "exploratory",
        "scope": "ukv era",
        "treatment": "a",
        "reference": "b",
        "difference_pp": pytest.approx(interval["difference"] * 100.0),
        "lower_95_pp": pytest.approx(interval["lower_95"] * 100.0),
        "upper_95_pp": pytest.approx(interval["upper_95"] * 100.0),
        "fold_lower_95_pp": pytest.approx(fold_lower * 100.0),
        "fold_upper_95_pp": pytest.approx(fold_upper * 100.0),
        "site_lowest_pp": pytest.approx(site_lowest * 100.0),
        "site_highest_pp": pytest.approx(site_highest * 100.0),
        "seed_spread_pp": pytest.approx(interval["seed_spread"] * 100.0),
        "excludes_zero": record["lower_95_pp"] > 0.0 or record["upper_95_pp"] < 0.0,
        "folds_agreeing": 2,
        "n_folds": 3,
        "n_rows": interval["n_rows"],
        "n_months": 6,
    }
    assert record["lower_95_pp"] < record["difference_pp"] < record["upper_95_pp"]
    assert record["n_rows"] == 12


def test_an_interval_excludes_zero_only_when_both_bounds_sit_on_one_side():
    strong = contrast_interval(
        losses=_losses(differences={"A": 0.02, "B": 0.02}),
        contrast=("a", "b"),
        domain=SOLAR,
        setting="pooled",
        section="planned",
    )
    none = contrast_interval(
        losses=_losses(differences={"A": 0.0, "B": 0.0}).with_columns(
            pl.when(pl.col("arm") == "a")
            .then(pl.col(METRIC) + 0.01 * (1 - 2 * (pl.col("time") % 2)))
            .otherwise(pl.col(METRIC))
            .alias(METRIC)
        ),
        contrast=("a", "b"),
        domain=SOLAR,
        setting="pooled",
        section="planned",
    )

    assert strong["excludes_zero"] is True
    assert strong["lower_95_pp"] > 0.0
    assert none["lower_95_pp"] < 0.0 < none["upper_95_pp"]
    assert none["excludes_zero"] is False


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
