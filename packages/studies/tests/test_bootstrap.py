from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from studies.bootstrap import (
    BootstrapInterval,
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_by_year,
    bootstrap_row_difference,
    bootstrap_year_change,
    bracket_verdict,
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


def test_a_year_of_exactly_six_months_is_enough():
    intervals = bootstrap_difference_by_year(
        losses=_yearly_losses(months_2024=6), treatment="era5", references=("other",), metric="loss"
    )

    assert (intervals[0]["n_months"], intervals[0]["enough_months"]) == (6, True)


def _yearly_losses_with_a_late_year_jump() -> pl.DataFrame:
    # ERA5's deficit is 0.5 from January to September, and jumps to 5.0 from October to December,
    # in both years. Restricting to months 1-9 must drop the October-December rows entirely.
    records = []
    for year in (2024, 2025):
        for month in range(1, 13):
            difference = 0.5 if month <= 9 else 5.0
            for seed in (0, 1):
                for arm, loss in (("era5", difference), ("other", 0.0)):
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


def test_months_restricts_every_year_to_the_given_calendar_months():
    intervals = bootstrap_difference_by_year(
        losses=_yearly_losses_with_a_late_year_jump(),
        treatment="era5",
        references=("other",),
        metric="loss",
        months=(1, 2, 3, 4, 5, 6, 7, 8, 9),
    )

    assert [interval["n_months"] for interval in intervals] == [9, 9]
    assert [interval["difference"] for interval in intervals] == pytest.approx([0.5, 0.5])


def test_no_months_filter_keeps_the_full_year():
    intervals = bootstrap_difference_by_year(
        losses=_yearly_losses_with_a_late_year_jump(),
        treatment="era5",
        references=("other",),
        metric="loss",
    )

    assert [interval["n_months"] for interval in intervals] == [12, 12]


def _year_records(*, year: int, month_values: dict[int, float]) -> list[dict]:
    records = []
    for month, value in month_values.items():
        for seed in (0, 1):
            for arm, loss in (("treatment", value), ("reference", 0.0)):
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
    return records


def test_change_is_year1_difference_minus_year0_difference():
    # 2024's difference is a constant 1.0 across its 4 months; 2025's alternates 0.0/2.0 across 12
    # months, which also averages to 1.0, so the point change is exactly 0 despite the two years
    # having very different within-year spread.
    losses = pl.DataFrame(
        [
            *_year_records(year=2024, month_values={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}),
            *_year_records(
                year=2025,
                month_values={m: (0.0 if m % 2 else 2.0) for m in range(1, 13)},
            ),
        ]
    )

    result = bootstrap_year_change(
        losses=losses,
        treatment="treatment",
        reference="reference",
        metric="loss",
        year0=2024,
        year1=2025,
    )

    assert result["change"] == pytest.approx(0.0)
    assert result["n_months_year0"] == 4
    assert result["n_months_year1"] == 12


def test_change_direction_is_year1_minus_year0():
    # 2024 is a constant 1.0, 2025 a constant 3.0, so the true change is unambiguous: +2.0 if
    # computed year1 minus year0, and -2.0 the other way around. A bug that swapped the order, or
    # that attributed 2025's rows to year0's slot, would flip this value's sign.
    losses = pl.DataFrame(
        [
            *_year_records(year=2024, month_values={1: 1.0, 2: 1.0}),
            *_year_records(year=2025, month_values={1: 3.0, 2: 3.0}),
        ]
    )

    result = bootstrap_year_change(
        losses=losses,
        treatment="treatment",
        reference="reference",
        metric="loss",
        year0=2024,
        year1=2025,
    )

    assert result["change"] == pytest.approx(2.0)


def test_months_filter_applies_to_both_years():
    # 2024 holds 1.0 in January and 5.0 in February; 2025 holds 1.0 in January and 9.0 in
    # February. Restricting to January alone must drop both years' February rows, leaving the
    # change at 1.0 - 1.0 = 0.0 rather than 1.0 - 5.0 (unfiltered, treating only 2024) or some
    # other mix.
    losses = pl.DataFrame(
        [
            *_year_records(year=2024, month_values={1: 1.0, 2: 5.0}),
            *_year_records(year=2025, month_values={1: 1.0, 2: 9.0}),
        ]
    )

    result = bootstrap_year_change(
        losses=losses,
        treatment="treatment",
        reference="reference",
        metric="loss",
        year0=2024,
        year1=2025,
        months=(1,),
    )

    assert result["n_months_year0"] == 1
    assert result["n_months_year1"] == 1
    assert result["change"] == pytest.approx(0.0)


def test_a_year_with_no_rows_after_the_months_filter_raises():
    losses = pl.DataFrame(
        [
            *_year_records(year=2024, month_values={1: 1.0}),
            *_year_records(year=2025, month_values={1: 1.0}),
        ]
    )

    with pytest.raises(ValueError, match="no rows"):
        bootstrap_year_change(
            losses=losses,
            treatment="treatment",
            reference="reference",
            metric="loss",
            year0=2024,
            year1=2025,
            months=(12,),
        )


def test_year_change_losses_at_two_settings_raise():
    losses = pl.DataFrame(
        [
            *_year_records(year=2024, month_values={1: 1.0}),
            *_year_records(year=2025, month_values={1: 1.0}),
        ]
    )
    both = pl.concat([losses, losses.with_columns(setting=pl.lit("sensitivity"))])

    with pytest.raises(ValueError, match="settings"):
        bootstrap_year_change(
            losses=both,
            treatment="treatment",
            reference="reference",
            metric="loss",
            year0=2024,
            year1=2025,
        )


def test_year_change_interval_resamples_each_year_independently_of_the_other():
    # Both years hold the same two month values (0.0 and 10.0), with one seed, so a correct
    # resample draws year0's months and year1's months independently: the change can range from
    # -10.0 (year0 draws only the 10.0 month, year1 only the 0.0 month) to +10.0 (the reverse).
    # A bug that dropped year0 from the resampled change (using year1's resampled mean alone)
    # gives [0.0, 10.0] on this fixture, and a bug that drew the same months for both years ties
    # each resample's year0 and year1 means together and collapses the interval to [0.0, 0.0].
    # Both mutants therefore disagree with the pinned bounds below.
    losses = pl.DataFrame(
        [
            *_year_records(year=2024, month_values={1: 0.0, 2: 10.0}),
            *_year_records(year=2025, month_values={1: 0.0, 2: 10.0}),
        ]
    )

    result = bootstrap_year_change(
        losses=losses,
        treatment="treatment",
        reference="reference",
        metric="loss",
        year0=2024,
        year1=2025,
    )

    assert result["change"] == pytest.approx(0.0)
    assert result["lower_95"] == pytest.approx(-10.0)
    assert result["upper_95"] == pytest.approx(10.0)


def test_bootstrap_row_difference_is_the_mean_of_the_values():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    months = np.array(["2024-01", "2024-01", "2024-02", "2024-02"])

    result = bootstrap_row_difference(values=values, months=months)

    assert result["difference"] == pytest.approx(2.5)
    assert result["n_months"] == 2
    assert result["n_rows"] == 4
    assert result["seed_spread"] == 0.0


def test_bootstrap_row_difference_resamples_whole_months_not_individual_rows():
    # Two months, twenty rows each, at two very different levels (0.0 and 10.0). A whole-month
    # resample can only draw each month whole, so the resampled mean can only be 0.0 (both draws
    # land on the 0.0 month), 10.0 (both land on the 10.0 month), or 5.0 (one of each) — the 2.5th
    # and 97.5th percentiles land inside the 0.0 and 10.0 clusters (each a quarter of the draws),
    # so the interval is exactly [0.0, 10.0]. A bug that resampled individual rows instead would
    # draw from the pooled 0.0/10.0 values row by row, which the central limit theorem pulls
    # toward the 5.0 mean, giving a much narrower interval that does not reach 0.0 or 10.0.
    values = np.array([0.0] * 20 + [10.0] * 20)
    months = np.array(["2024-01"] * 20 + ["2024-02"] * 20)

    result = bootstrap_row_difference(values=values, months=months)

    assert result["lower_95"] == pytest.approx(0.0)
    assert result["upper_95"] == pytest.approx(10.0)


def test_bootstrap_row_difference_interval_collapses_with_no_month_spread():
    # Every month holds the same value, so a whole-month resample can never draw anything but 5.0.
    values = np.array([5.0] * 20)
    months = np.array([f"2024-{m:02d}" for m in range(1, 11) for _ in range(2)])

    result = bootstrap_row_difference(values=values, months=months)

    assert result["lower_95"] == pytest.approx(5.0)
    assert result["upper_95"] == pytest.approx(5.0)


def _interval(*, difference: float, lower_95: float, upper_95: float) -> BootstrapInterval:
    return {
        "difference": difference,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "seed_spread": 0.0,
        "n_rows": 100,
        "n_months": 10,
    }


def test_bracket_verdict_beats_when_the_lower_side_is_negative_and_significant():
    lower_side = _interval(difference=-0.5, lower_95=-0.8, upper_95=-0.2)
    upper_side = _interval(difference=0.1, lower_95=-0.1, upper_95=0.3)

    assert bracket_verdict(lower_side=lower_side, upper_side=upper_side) == "beats"


def test_bracket_verdict_loses_when_the_upper_side_is_positive_and_significant():
    lower_side = _interval(difference=0.3, lower_95=-0.1, upper_95=0.7)
    upper_side = _interval(difference=0.5, lower_95=0.2, upper_95=0.8)

    assert bracket_verdict(lower_side=lower_side, upper_side=upper_side) == "loses"


def test_bracket_verdict_unresolved_when_neither_side_is_significant():
    lower_side = _interval(difference=-0.1, lower_95=-0.4, upper_95=0.2)
    upper_side = _interval(difference=0.1, lower_95=-0.2, upper_95=0.4)

    assert bracket_verdict(lower_side=lower_side, upper_side=upper_side) == "unresolved"


def test_bracket_verdict_unresolved_when_an_interval_touches_zero():
    # The lower side's upper bound sits exactly at zero, so it does not clear zero and is not
    # significant: a `<=` in place of `<` in the function's key line would wrongly say "beats".
    lower_side = _interval(difference=-0.4, lower_95=-0.8, upper_95=0.0)
    upper_side = _interval(difference=0.4, lower_95=0.0, upper_95=0.8)

    assert bracket_verdict(lower_side=lower_side, upper_side=upper_side) == "unresolved"
