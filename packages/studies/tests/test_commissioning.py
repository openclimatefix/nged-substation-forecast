from datetime import UTC, datetime

import polars as pl
from studies.commissioning import SETTLED_OUTPUT_FROM, drop_commissioning_ramp


def test_rows_before_the_settled_date_are_dropped_for_the_listed_site_only():
    site, settled_from = next(iter(SETTLED_OUTPUT_FROM.items()))
    before = settled_from.replace(year=settled_from.year - 1)
    dataset = pl.DataFrame(
        {
            "site": [site, site, "other", "other"],
            "time": [before, settled_from, before, settled_from],
        },
        schema={"site": pl.String, "time": pl.Datetime("us", "UTC")},
    )

    kept = drop_commissioning_ramp(dataset=dataset)

    assert kept.to_dicts() == [
        {"site": site, "time": settled_from},
        {"site": "other", "time": before},
        {"site": "other", "time": settled_from},
    ]


def test_a_frame_with_no_listed_site_keeps_every_row():
    dataset = pl.DataFrame(
        {"site": ["other"], "time": [datetime(2020, 1, 1, tzinfo=UTC)]},
        schema={"site": pl.String, "time": pl.Datetime("us", "UTC")},
    )

    assert drop_commissioning_ramp(dataset=dataset).equals(dataset)
