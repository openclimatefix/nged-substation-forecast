from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from studies.bootstrap import bootstrap_difference, per_fold_differences

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


def test_per_fold_differences_give_one_value_per_fold_in_fold_order():
    losses = _losses(seeds=(0, 1), difference=0.25).with_columns(
        loss=pl.when(pl.col("fold") == 1).then(pl.col("loss") * 2).otherwise(pl.col("loss"))
    )

    assert per_fold_differences(
        losses=losses, treatment="T", reference="R", metric="loss"
    ) == pytest.approx([0.25, 0.5])
