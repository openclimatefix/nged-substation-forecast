"""Drop the half-hours in which a generator was still being built.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**A solar farm's first months of telemetry measure a smaller plant than the one the site's capacity
describes, so those rows are removed from the experiment entirely.** Site E's output relative to
the other five photovoltaic sites climbs through flat multi-day plateaus — 0%, 12%, 29%, 56%, 76%
and 88% of its settled level through April 2024, then the same sequence again from 12% in July
after a 24-day outage — and only reaches that settled level in October 2024. The deficit is flat at
79% to 85% of the settled level across every decile of fleet output, which is what a fraction of an
array being energised looks like and is not what either inverter clipping or an export cap looks
like.

**Removal is the only honest treatment, which is why these rows are dropped rather than masked out
of training alone.** A curtailed hour can be kept for scoring, because the export cap says what the
generator was allowed to produce and a prediction can be held down to it. A half-built plant
publishes no such number: nothing in the record says which fraction of the array was connected on a
given day, so there is no ceiling to clamp a prediction to and no way to score the hour fairly. A
model fitted on the settled plant overshoots every one of these rows by construction.

**The cut-off is the fitted changepoint at which the site reaches its settled output, not the date
the network scheme went live.** The two are 61 days apart and the scheme went live first, so the
[export cap](export_cap.py) does not mark these hours. Winter has a fifth of summer's bright
half-hours, so the changepoint is bracketed only to somewhere between October 2024 and March 2025;
taking the earliest date in that bracket keeps the most data, and every later date would remove
more rows rather than fewer.

**A uniform metering correction would leave the same trace as a final block of array, and nothing
in the telemetry separates them.** Either way the rows measure a different quantity from the rest
of the site's record, which is the reason to drop them.
"""

import logging
from datetime import UTC, datetime
from typing import Final

import polars as pl

_LOG: Final[logging.Logger] = logging.getLogger("commissioning")

SETTLED_OUTPUT_FROM: Final[dict[str, datetime]] = {
    "E": datetime(2024, 10, 6, tzinfo=UTC),
}
"""The first day each generator reached its settled output, for the ones still being built.

Keyed by the anonymous site label `build_dataset` assigns. A site absent from this mapping has no
detected commissioning ramp and keeps every row.
"""


def drop_commissioning_ramp(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Remove the rows a generator produced before it reached its settled output.

    Args:
        dataset: The modelling frame, carrying `site` and `time`.

    Returns:
        The same frame without those rows.
    """
    keep = pl.lit(value=True)
    for site, settled_from in SETTLED_OUTPUT_FROM.items():
        keep = keep & ~((pl.col("site") == site) & (pl.col("time") < settled_from))
    kept = dataset.filter(keep)
    _LOG.info(
        "commissioning ramp: dropping %d of %d rows",
        dataset.height - kept.height,
        dataset.height,
    )
    return kept
