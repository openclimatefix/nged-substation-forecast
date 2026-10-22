"""Score every arm of one series on the ordinary rows and on the rows after each load shift.

Every arm shares one offset table per series, hour band, and lead band, so a score difference is a
difference in the location of the forecast only. Sums are stored per group, not per row:

- `ordinary_<s>.parquet`: for every arm, calendar month of the target, and ordinary-row variant,
  the row count and the sums of the quantile score, the mean-forecast and median-forecast errors,
  the pinball loss at four tail levels, and the number of readings below each tail quantile. A
  variant `all_rows` (every row with a reading) feeds the net annual figure.
- `after_<s>.parquet`: the same sums for the 72 hours of target time after each labelled load
  shift ends, per load shift, lead band, and arm.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00258_score.py 10 3
"""

from __future__ import annotations

import sys
from typing import Final

import e00258_data as dat
import e00258_harness as h
import numpy as np
import polars as pl

AFTER_HALF_HOURS: Final[int] = 144
TAIL_LEVELS: Final[tuple[int, ...]] = (0, 1, 9, 10)
"""Indices into `h.LEVELS` of the 1%, 5%, 95%, and 99% quantiles."""
METRICS: Final[tuple[str, ...]] = (
    "qs",
    "err_mean",
    "err_median",
    *[f"below_{i}" for i in TAIL_LEVELS],
    *[f"pin_{i}" for i in TAIL_LEVELS],
)
VARIANTS: Final[tuple[str, ...]] = (
    "all_rows",
    "ordinary",
    "ordinary_clean",
    "ordinary_no_d8_d9",
    "ordinary_wide",
)
"""Row sets. `ordinary` is outside every logged closure that touches the series and the 72 hours
after it. `ordinary_clean` also drops D8's multi-day drops and the 17 block days.
`ordinary_no_d8_d9` drops series 26 and 7 entirely. `ordinary_wide` also drops closures of a
connected neighbour."""
D8_SERIES: Final[int] = 26
D9_SERIES: Final[int] = 7


def closure_blocks(*, series: int, neighbours: bool) -> np.ndarray:
    """True on every grid stamp inside a logged closure of the series, or 72 hours after it."""
    wanted = {str(series)}
    if neighbours:
        wanted |= {str(s) for s in dat.ADJACENCY[series]}
    out = np.zeros(h.N_GRID + AFTER_HALF_HOURS, dtype=bool)
    for r in dat.logged_closures().iter_rows(named=True):
        if r["side_a"] in wanted or r["side_b"] in wanted:
            lo = max(int(r["first_stamp"]), 0)
            hi = min(int(r["last_stamp_exclusive"]) + AFTER_HALF_HOURS, h.N_GRID)
            out[lo:hi] = True
    return out[: h.N_GRID]


def extra_blocks(*, series: int) -> np.ndarray:
    """True on D8's multi-day drops and the unlabelled block days of the series."""
    out = np.zeros(h.N_GRID, dtype=bool)
    if series == D8_SERIES:
        for r in pl.read_parquet(dat.D8_DROPS).iter_rows(named=True):
            out[int(r["first_stamp"]) : int(r["last_stamp"]) + 1] = True
    for lo, hi in dat.block_day_ranges().get(series, []):
        out[max(lo, 0) : hi + 1] = True
    return out


def row_metrics(
    *, data: h.SeriesData, correction: np.ndarray, offsets: np.ndarray, band: np.ndarray
) -> dict[str, np.ndarray]:
    """Per (origin, lead) metrics of one arm, NaN where the target has no reading."""
    _, _, targets = h.origin_grid()
    clipped = np.minimum(targets, h.N_GRID - 1)
    actual = data.actual[clipped].astype(np.float64)
    location = data.expected[clipped] + correction
    cell = offsets[band[clipped], np.broadcast_to(h.lead_band_of_lead()[None, :], targets.shape)]
    bad = (targets >= h.N_GRID) | ~np.isfinite(actual)
    error = actual[..., None] - (location[..., None] + cell)
    pinball = np.maximum(h.LEVELS * error, (h.LEVELS - 1.0) * error)
    out = {
        "qs": 2.0 * pinball.mean(axis=-1),
        "err_mean": location - actual,
        "err_median": location + cell[..., 5] - actual,
    }
    for i in TAIL_LEVELS:
        out[f"below_{i}"] = (error[..., i] < 0).astype(np.float64)
        out[f"pin_{i}"] = pinball[..., i]
    return {name: np.where(bad, np.nan, value) for name, value in out.items()}


def _sums(*, metrics: dict[str, np.ndarray], mask: np.ndarray, code: np.ndarray, size: int) -> dict:
    """Row counts and metric sums per group code over the masked rows."""
    keep = mask & np.isfinite(metrics["qs"])
    group = code[keep]
    out = {"n": np.bincount(group, minlength=size)}
    for name in METRICS:
        out[name] = np.bincount(group, weights=metrics[name][keep], minlength=size)
    return out


def main(series: int) -> None:
    """Score one series and write its two tables to the harness folder."""
    band = h.band_grid()
    data = h.load_series(series=series)
    stored = np.load(h.HARNESS_DIR / f"series_{series}.npz")
    offsets = stored["offsets"]
    origins, _, targets = h.origin_grid()
    clipped = np.minimum(targets, h.N_GRID - 1)
    year_month = dat.stamp_year_month()[clipped]
    codes_ym = np.unique(year_month)
    ym_index = np.searchsorted(codes_ym, year_month)
    lead_band = np.broadcast_to(h.lead_band_of_lead()[None, :], targets.shape)
    own = closure_blocks(series=series, neighbours=False)[clipped]
    wide = closure_blocks(series=series, neighbours=True)[clipped]
    extra = extra_blocks(series=series)[clipped]
    masks = {
        "all_rows": np.ones(targets.shape, dtype=bool),
        "ordinary": ~own,
        "ordinary_clean": ~own & ~extra,
        "ordinary_no_d8_d9": ~own & (series not in (D8_SERIES, D9_SERIES)),
        "ordinary_wide": ~wide,
    }
    events = h.labelled_events().filter(pl.col("series").list.contains(series))
    after_masks = {}
    for r in events.iter_rows(named=True):
        in_window = (
            (origins[:, None] >= r["onset"])
            & (targets >= r["end"])
            & (targets < r["end"] + AFTER_HALF_HOURS)
        )
        after_masks[r["card_id"]] = in_window
    arm_names = [k for k in stored.files if k not in ("origins", "offsets", "folds")]
    ordinary_rows, after_rows = [], []
    for arm in arm_names:
        metrics = row_metrics(
            data=data, correction=stored[arm].astype(np.float64), offsets=offsets, band=band
        )
        for variant in VARIANTS:
            sums = _sums(metrics=metrics, mask=masks[variant], code=ym_index, size=len(codes_ym))
            frame = pl.DataFrame({"year_month": codes_ym, **sums}).filter(pl.col("n") > 0)
            ordinary_rows.append(
                frame.with_columns(
                    series=pl.lit(series),
                    arm=pl.lit(arm),
                    variant=pl.lit(variant),
                    odd=pl.col("year_month").mod(100).is_in(list(h.TUNING_MONTHS)),
                )
            )
        for card, mask in after_masks.items():
            sums = _sums(metrics=metrics, mask=mask, code=lead_band, size=len(h.LEAD_BAND_UPPER))
            frame = pl.DataFrame({"lead_band": range(len(h.LEAD_BAND_UPPER)), **sums})
            after_rows.append(
                frame.filter(pl.col("n") > 0).with_columns(
                    series=pl.lit(series), arm=pl.lit(arm), card=pl.lit(card)
                )
            )
    pl.concat(ordinary_rows).write_parquet(h.HARNESS_DIR / f"ordinary_{series}.parquet")
    pl.concat(after_rows).write_parquet(h.HARNESS_DIR / f"after_{series}.parquet")
    print(f"series {series}: scored {len(arm_names)} arms", flush=True)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(int(arg))
