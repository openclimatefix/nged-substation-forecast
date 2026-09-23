from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from studies.bootstrap import (
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_by_year,
    fold_t_interval,
    per_fold_differences,
)

# Months hold unequal numbers of rows, so a resample that lost the pairing on `seed`, or drew rows
# rather than whole months, would land on a different interval.
ROWS_PER_MONTH = ((1, 3), (2, 5), (3, 2), (4, 4))


def _losses(*, seeds: tuple[int, ...], difference: float | None = None) -> pl.DataFrame:
    generator = np.random.default_rng(7)
    records = []
    for seed in seeds:
        for month, n_rows in ROWS_PER_MONTH:
            for day in range(n_rows):
                for arm in ("T", "R"):
                    noise = float(generator.normal())
                    loss = noise if difference is None else (difference if arm == "T" else 0.0)
                    records.append(
                        {
                            "arm": arm,
                            "site": "A",
                            "time": datetime(2025, month, 1 + day, tzinfo=UTC),
                            "seed": seed,
                            "month": f"2025-{month:02d}",
                            "fold": month % 2,
                            "loss": loss,
                        }
                    )
    return pl.DataFrame(records)


def test_the_interval_reproduces_the_published_random_stream():
    # Pinned from the implementation every published interval was computed with. A change to the
    # order of the seed and month draws, to the pairing, or to the resampling unit moves these.
    interval = bootstrap_difference(
        losses=_losses(seeds=(0, 1, 2)), treatment="T", reference="R", metric="loss"
    )

    assert interval == {
        "difference": -0.07228614354707967,
        "lower_95": -0.89888694820251,
        "upper_95": 0.6052753886101083,
        "seed_spread": 0.28637193863860544,
        "n_rows": 14,
        "n_months": 4,
    }


def test_the_interval_does_not_depend_on_the_order_of_the_rows():
    # The runners stack their losses in whatever order the fits finish, so the pairing has to
    # sort before it assigns months, or the published digits would change from run to run.
    shuffled = _losses(seeds=(0, 1, 2)).sample(fraction=1.0, shuffle=True, seed=3)

    assert bootstrap_difference(
        losses=shuffled, treatment="T", reference="R", metric="loss"
    ) == bootstrap_difference(
        losses=_losses(seeds=(0, 1, 2)), treatment="T", reference="R", metric="loss"
    )


def test_rows_are_paired_within_a_site():
    # Every site shares the same timestamps, so pairing on time alone would cross sites.
    site_a = _losses(seeds=(0, 1, 2), difference=0.25)
    site_b = site_a.with_columns(site=pl.lit("B"), loss=pl.col("loss") * 2)

    interval = bootstrap_difference(
        losses=pl.concat([site_a, site_b]), treatment="T", reference="R", metric="loss"
    )

    assert interval["difference"] == pytest.approx(0.375)
    assert interval["n_rows"] == 28


def test_each_interval_starts_its_own_random_stream():
    losses = _losses(seeds=(0, 1, 2))

    first = bootstrap_difference(losses=losses, treatment="T", reference="R", metric="loss")
    second = bootstrap_difference(losses=losses, treatment="T", reference="R", metric="loss")

    assert first == second


def test_a_constant_difference_gives_a_zero_width_interval_at_that_value():
    losses = _losses(seeds=(0, 1, 2), difference=0.25)

    interval = bootstrap_difference(losses=losses, treatment="T", reference="R", metric="loss")

    assert interval["difference"] == pytest.approx(0.25)
    assert interval["lower_95"] == pytest.approx(0.25)
    assert interval["upper_95"] == pytest.approx(0.25)


def test_one_seed_is_enough():
    interval = bootstrap_difference(
        losses=_losses(seeds=(0,), difference=0.25), treatment="T", reference="R", metric="loss"
    )

    assert interval["difference"] == pytest.approx(0.25)
    assert interval["seed_spread"] == 0.0


def test_bootstrap_absolute_reproduces_the_published_random_stream():
    # Pinned from the implementation every published leaderboard interval was computed with.
    interval = bootstrap_absolute(losses=_losses(seeds=(0, 1, 2)), arm="T", metric="loss")

    assert interval == {
        "value": -0.21010131421631167,
        "lower_95": -1.1626544831593595,
        "upper_95": 0.4557367201731675,
        "seed_spread": 0.32966017229876704,
        "n_rows": 14,
        "n_months": 4,
    }


def test_bootstrap_absolute_does_not_depend_on_the_order_of_the_rows():
    shuffled = _losses(seeds=(0, 1, 2)).sample(fraction=1.0, shuffle=True, seed=3)

    assert bootstrap_absolute(losses=shuffled, arm="T", metric="loss") == bootstrap_absolute(
        losses=_losses(seeds=(0, 1, 2)), arm="T", metric="loss"
    )


def test_bootstrap_absolute_ignores_the_other_arm():
    losses = _losses(seeds=(0, 1, 2), difference=0.25)

    interval = bootstrap_absolute(losses=losses, arm="T", metric="loss")

    assert interval["value"] == pytest.approx(0.25)
    assert interval["lower_95"] == pytest.approx(0.25)
    assert interval["upper_95"] == pytest.approx(0.25)
    assert interval["n_rows"] == 14


def test_bootstrap_absolute_raises_when_the_seeds_hold_different_rows():
    # Same row count in every seed, so only a check on the rows themselves catches the mismatch.
    losses = _losses(seeds=(0, 1)).with_columns(
        time=pl.when((pl.col("seed") == 1) & (pl.col("time").dt.day() == 1))
        .then(pl.col("time") + pl.duration(days=20))
        .otherwise(pl.col("time"))
    )

    with pytest.raises(ValueError, match="same \\(site, time\\) rows"):
        bootstrap_absolute(losses=losses, arm="T", metric="loss")


def test_per_fold_differences_give_one_value_per_fold_in_fold_order():
    losses = _losses(seeds=(0, 1), difference=0.25).with_columns(
        loss=pl.when(pl.col("fold") == 1).then(pl.col("loss") * 2).otherwise(pl.col("loss"))
    )

    assert per_fold_differences(
        losses=losses, treatment="T", reference="R", metric="loss"
    ) == pytest.approx([0.25, 0.5])


def test_the_fold_t_interval_matches_the_textbook_formula():
    # Five folds: mean 2, sample standard deviation sqrt(2.5), t(0.975, 4) = 2.7764451.
    lower, upper = fold_t_interval(fold_differences=[0.0, 1.0, 2.0, 3.0, 4.0])

    half_width = 2.7764451052 * np.sqrt(2.5) / np.sqrt(5.0)
    assert (lower, upper) == pytest.approx((2.0 - half_width, 2.0 + half_width))


def test_a_single_fold_has_no_t_interval():
    with pytest.raises(ValueError, match="at least two folds"):
        fold_t_interval(fold_differences=[1.0])


def _yearly_losses(*, months_2024: int) -> pl.DataFrame:
    # ERA5 is worse by 0.5 in every row, over `months_2024` months of 2024 and all 12 of 2025.
    records = []
    for year, months in ((2024, months_2024), (2025, 12)):
        for month in range(1, months + 1):
            for seed in (0, 1):
                for arm, loss in (("era5", 1.0), ("other", 0.5)):
                    records.append(
                        {
                            "arm": arm,
                            "site": "A",
                            "time": datetime(year, month, 1, tzinfo=UTC),
                            "seed": seed,
                            "month": f"{year}-{month:02d}",
                            "loss": loss,
                            "setting": "pooled",
                        }
                    )
    return pl.DataFrame(records)


def test_the_yearly_difference_is_treatment_minus_reference():
    intervals = bootstrap_difference_by_year(
        losses=_yearly_losses(months_2024=12),
        treatment="era5",
        references=("other",),
        metric="loss",
    )

    assert [interval["year"] for interval in intervals] == [2024, 2025]
    assert [interval["difference"] for interval in intervals] == pytest.approx([0.5, 0.5])


def test_a_year_of_fewer_than_six_months_is_flagged():
    intervals = bootstrap_difference_by_year(
        losses=_yearly_losses(months_2024=5), treatment="era5", references=("other",), metric="loss"
    )

    assert [(i["n_months"], i["enough_months"]) for i in intervals] == [(5, False), (12, True)]


def test_losses_at_two_settings_raise():
    losses = _yearly_losses(months_2024=12)
    both = pl.concat([losses, losses.with_columns(setting=pl.lit("sensitivity"))])

    with pytest.raises(ValueError, match="settings"):
        bootstrap_difference_by_year(
            losses=both, treatment="era5", references=("other",), metric="loss"
        )
