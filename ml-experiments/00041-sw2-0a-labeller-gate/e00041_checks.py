"""The two hindsight-model checks: mean z around long closures, and logged against random exclusion.

uv run python ml-experiments/00041-sw2-0a-labeller-gate/e00041_checks.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from e00041_common import SERIES_IDS, WORK_DIR, logged_intervals
from e00041_panel import INDEX, N_STAMPS, first_affected, last_affected, load_panel

CHECK_SERIES = (3, 8, 10, 14, 17, 25)
WEEK_STAMPS = 336
LONG_CLOSURE_DAYS = 3


def mean_z_around_long_closures() -> None:
    """Check 2: the mean hindsight z in the 7 days either side of closures over 3 days."""
    panel = load_panel()
    z = (panel.obs - panel.q50) / ((panel.q90 - panel.q10) / 2.56)
    values = []
    for row in logged_intervals().iter_rows(named=True):
        if (row["end_utc"] - row["start_utc"]).total_seconds() < LONG_CLOSURE_DAYS * 86400:
            continue
        first, last = (
            first_affected(close_utc=row["start_utc"]),
            last_affected(open_utc=row["end_utc"]),
        )
        for side in (row["a"], row["b"]):
            if not (side and side.isdigit() and int(side) in INDEX):
                continue
            for lo, hi in (
                (first - 48 - WEEK_STAMPS, first - 48),
                (last + 48, last + 48 + WEEK_STAMPS),
            ):
                if lo >= 0 and hi < N_STAMPS and not np.all(np.isnan(z[INDEX[int(side)], lo:hi])):
                    values.append(np.nanmean(z[INDEX[int(side)], lo:hi]))
    arr = np.array(values)
    print(
        f"{len(arr)} windows: mean z {arr.mean():.3f}, "
        f"standard error {arr.std() / np.sqrt(len(arr)):.3f}"
    )
    outside = np.nanmean((panel.obs > panel.q90) | (panel.obs < panel.q10))
    print(f"share outside q10 to q90: {outside:.3f}")


def logged_against_random_exclusion() -> None:
    """Check 1: out-of-event MAE with logged closures excluded against random exclusion."""
    assert set(CHECK_SERIES) <= set(SERIES_IDS)
    for sid in CHECK_SERIES:
        logged = pl.read_parquet(WORK_DIR / "hindsight" / f"{sid}.parquet")
        random = pl.read_parquet(WORK_DIR / "hindsight_random" / f"{sid}.parquet")
        joined = logged.join(random, on="valid_time", suffix="_r").filter(
            ~pl.col("in_logged"), pl.col("obs_mw").is_not_null()
        )
        err_logged = (joined["obs_mw"] - joined["q50_mw"]).abs()
        err_random = (joined["obs_mw"] - joined["q50_mw_r"]).abs()
        diff = err_random - err_logged
        by_month = (
            pl.DataFrame({"m": joined["valid_time"].dt.strftime("%Y-%m"), "d": diff})
            .group_by("m")
            .agg(pl.col("d").sum().alias("sum"), pl.len().alias("n"))
        )
        sums, counts = by_month["sum"].to_numpy(), by_month["n"].to_numpy()
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(2000):
            pick = rng.choice(len(sums), len(sums))
            draws.append(sums[pick].sum() / counts[pick].sum())
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(
            f"series {sid}: MAE logged {err_logged.mean():.4f}, random {err_random.mean():.4f}, "
            f"random minus logged {diff.mean():.4f} ({lo:.4f} to {hi:.4f})"
        )


if __name__ == "__main__":
    mean_z_around_long_closures()
    logged_against_random_exclusion()
