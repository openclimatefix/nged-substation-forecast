"""Sum each arm's scores per shift and phase, and per valid month for ordinary rows.

Reads the per-site, per-world score tables of `e00296_score.py` and writes two small tables the
report reads: `event_sums.parquet` (one row per site, world, shift, phase and arm) and
`ordinary_sums.parquet` (one row per valid month and arm). The node-only curtailment cost per row
follows experiment 283: the 0.95 quantile above the limit is the planned cut, and a breach
that the cut does not cover costs 20 times as much, at GBP 2,000 per MWh, for 12 to 36 hour leads.

    uv run --no-sync python e00296_sums.py
"""

from __future__ import annotations

import json
from typing import Final

import e00296_fed as fed
import numpy as np
import polars as pl
from e00296_common import WORK_DIR

WORLDS: Final = WORK_DIR / "worlds"
MARGIN_STAMPS: Final[int] = 336
"""Seven days either side of a shift, as in `e00296_score.py`."""

SCORES: Final = WORK_DIR / "scores"
RESULTS: Final = WORK_DIR / "results"
FRACTIONS: Final[tuple[float, ...]] = (0.5, 0.75, 0.9)
PRICE_GBP_PER_MWH: Final[float] = 2000.0
BREACH_RATIO: Final[int] = 20
LEAD_MIN_HOURS: Final[float] = 12.0
LEAD_MAX_HOURS: Final[float] = 36.0
METRICS: Final[tuple[str, ...]] = (
    "qs_nra",
    "ae_median_nra",
    "ae_mean_nra",
    "tw95_nra",
    "tw99_nra",
    "qs_obs",
    "ae_median_obs",
    "ae_mean_obs",
    "tw95_obs",
    "tw99_obs",
    *(f"cost{f}" for f in FRACTIONS),
    "low_nra",
    "high_nra",
    "pin05_nra",
    "pin95_nra",
    "low_obs",
    "high_obs",
    "pin05_obs",
    "pin95_obs",
)
"""`low` and `high` are 1 where the target lies below the 0.05 quantile or above the 0.95 quantile;
`pin05` and `pin95` are the pinball losses at those two levels. The score tables hold the 0.05,
0.5, and 0.95 quantiles only, so coverage at 0.01 and 0.99 is not available."""


def with_cost(*, frame: pl.DataFrame, constants: dict) -> pl.DataFrame:
    """Add the per-row cost columns (GBP per half-hour) for each scaled limit."""
    site = frame["site"][0]
    cap = constants[site]["capacity_mw"] / 100.0
    peak = constants[site]["max_before_points"] * cap
    lead = frame["lead_hours"].to_numpy()
    inside = (lead >= LEAD_MIN_HOURS) & (lead < LEAD_MAX_HOURS)
    q95 = frame["q95"].to_numpy().astype(np.float64) * cap
    actual = frame["actual_obs"].to_numpy().astype(np.float64) * cap
    columns = []
    for fraction in FRACTIONS:
        limit = fraction * peak
        planned = np.maximum(q95 - limit, 0.0) * 0.5
        breach = np.maximum(actual - planned / 0.5 - limit, 0.0) * 0.5
        cost = PRICE_GBP_PER_MWH * (planned + BREACH_RATIO * breach)
        columns.append(pl.Series(f"cost{fraction}", np.where(inside, cost, np.nan)))
    tails = []
    for target in ("nra", "obs"):
        actual_t = pl.col(f"actual_{target}").cast(pl.Float64)
        q05, q95_ = pl.col("q05").cast(pl.Float64), pl.col("q95").cast(pl.Float64)
        tails += [
            (actual_t < q05).cast(pl.Float64).alias(f"low_{target}"),
            (actual_t > q95_).cast(pl.Float64).alias(f"high_{target}"),
            pl.max_horizontal(0.05 * (actual_t - q05), -0.95 * (actual_t - q05)).alias(
                f"pin05_{target}"
            ),
            pl.max_horizontal(0.95 * (actual_t - q95_), -0.05 * (actual_t - q95_)).alias(
                f"pin95_{target}"
            ),
        ]
    return frame.with_columns(columns).with_columns(tails)


def ambiguous_rows(*, frame: pl.DataFrame, events: pl.DataFrame, offset: int) -> np.ndarray:
    """True for rows whose target time lies within 7 days of two different shifts of the world.

    The score table gives such a row to the later shift, so its score would credit that shift with
    the effect of the earlier one. Those rows are left out of the per-shift sums.
    """
    n = int(frame["position"].to_numpy().max()) + 2
    count = np.zeros(n + 1, dtype=np.int64)
    for onset, end in events.select("onset", "end").iter_rows():
        lo = max(onset + offset - MARGIN_STAMPS, 0)
        hi = min(end + offset + MARGIN_STAMPS, n)
        count[lo:hi] += 1
    return count[frame["position"].to_numpy()] >= 2


def main() -> None:
    """Aggregate every score table."""
    constants = json.loads((SCORES / "site_constants.json").read_text())
    event_parts, ordinary_parts = [], []
    all_events = pl.read_parquet(WORLDS / "events.parquet")
    offset = fed.stamp_offset(fed.get_site("D5"))
    dropped = 0
    total = 0
    for path in sorted(SCORES.glob("D*_*.parquet")):
        frame = with_cost(frame=pl.read_parquet(path), constants=constants)
        site, world = path.stem.split("_")
        mine = all_events.filter((pl.col("site") == site) & (pl.col("world") == int(world)))
        flag = ambiguous_rows(frame=frame, events=mine, offset=offset)
        keep_arm = frame["arm"].to_numpy() == "K0"
        dropped += int((flag & keep_arm & (frame["event"].to_numpy() >= 0)).sum())
        total += int((keep_arm & (frame["event"].to_numpy() >= 0)).sum())
        frame = frame.filter(~pl.Series(flag & (frame["event"].to_numpy() >= 0)))
        sums = [pl.col(m).fill_nan(None).sum().alias(f"sum_{m}") for m in METRICS]
        counts = [
            pl.col(m).fill_nan(None).is_not_null().sum().alias(f"n_{m}")
            for m in ("qs_nra", "cost0.5")
        ]
        near = frame.filter(pl.col("event") >= 0)
        event_parts.append(
            near.group_by("site", "world", "event", "phase", "arm").agg(
                *sums,
                *counts,
                pl.col("event_hours").first(),
                pl.col("fraction").first(),
                pl.col("sign").first(),
                pl.col("shape").first(),
                pl.col("duration_bin").first(),
            )
        )
        valid = pl.col("power_fcst_init_time") + pl.duration(
            hours=pl.col("lead_hours").cast(pl.Int64)
        )
        ordinary_parts.append(
            frame.filter((pl.col("event") < 0) & pl.col("arm").is_in(["K0", "PERFECT"]))
            .with_columns(month=valid.dt.strftime("%Y%m").cast(pl.Int64))
            .group_by("site", "month", "arm")
            .agg(*sums, *counts)
        )
        print(path.name, flush=True)
    print(f"Rows within 7 days of two shifts, left out: {dropped} of {total} K0 rows near a shift")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ambiguous_rows.json").write_text(
        json.dumps({"dropped": dropped, "near_rows": total})
    )
    pl.concat(event_parts).write_parquet(RESULTS / "event_sums.parquet")
    pl.concat(ordinary_parts).write_parquet(RESULTS / "ordinary_sums.parquet")


if __name__ == "__main__":
    main()
