"""Build the three band-width variants from experiment 155's stored forecasts.

| Arm | Spread offsets along the lead |
|---|---|
| `ref` | the stored forecasts, each row reading its own lead band |
| `i` | linear interpolation between the lead-band centres (12, 36, 96, and 156 hours) |
| `m` | the running maximum (upper offsets) and minimum (lower offsets) over lead bands |
| `im` | the `m` table, interpolated |

The mean is the reference's. The stored spread table is recovered from the tuning rows as the
quantile minus the mean, and the same table is checked against the replicate and scored rows.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00160-*/e00160_run.py [--workers 5]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00160_common as common
import numpy as np
import polars as pl

TABLE_TOLERANCE: Final[float] = 1e-3
"""Offsets of one cell may differ by this many points (float32 rounding of the stored columns)."""
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (*common.BASE_COLUMNS, "arm")


def lead_band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the stored lead band (0 to 3) of leads in hours."""
    edges = np.array(common.LEAD_BAND_UPPER_HOURS)
    return np.minimum(np.searchsorted(edges, lead_hours, side="left"), common.N_LEAD_BANDS - 1)


def recover_table(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the stored spread table, shape (hour band, quarter, lead band, 11).

    Args:
        frame: One period's reference forecasts, with `mean`, the quantile columns, `hour_band`,
            `quarter`, and `lead_hours`.

    Returns:
        The offsets from the mean, `NaN` for a cell the period holds no row of.

    Raises:
        ValueError: If a cell is empty or holds two different offsets.
    """
    band = frame["hour_band"].to_numpy().astype(np.int64)
    quarter = frame["quarter"].to_numpy().astype(np.int64)
    lead = lead_band(lead_hours=frame["lead_hours"].to_numpy())
    mean = frame["mean"].to_numpy().astype(np.float64)
    offsets = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64) - mean[:, None]
    table = np.full(
        (common.N_HOUR_BANDS, common.N_QUARTERS, common.N_LEAD_BANDS, len(common.QUANTILES)),
        np.nan,
    )
    for b in range(common.N_HOUR_BANDS):
        for q in range(common.N_QUARTERS):
            for k in range(common.N_LEAD_BANDS):
                chosen = (band == b) & (quarter == q) & (lead == k)
                if not chosen.any():
                    continue
                cell = offsets[chosen]
                if np.ptp(cell, axis=0).max() > TABLE_TOLERANCE:
                    msg = f"cell ({b}, {q}, {k}) holds more than one spread"
                    raise ValueError(msg)
                table[b, q, k] = cell.mean(axis=0)
    return table


def fill_empty_cells(*, table: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    """Fill cells the tuning rows leave empty from the other periods' tables.

    Args:
        table: The tuning rows' table.
        others: The replicate and scored periods' tables.

    Returns:
        The table with no empty cell.

    Raises:
        ValueError: If a cell is empty in every period, or two periods disagree on a cell.
    """
    out = table.copy()
    for other in others:
        both = np.isfinite(out[..., 0]) & np.isfinite(other[..., 0])
        if both.any() and np.abs(out[both] - other[both]).max() > TABLE_TOLERANCE:
            msg = "two periods hold different spread tables"
            raise ValueError(msg)
        fill = ~np.isfinite(out[..., 0]) & np.isfinite(other[..., 0])
        out[fill] = other[fill]
    if not np.isfinite(out).all():
        msg = "a spread cell is empty in every period"
        raise ValueError(msg)
    return out


def monotone_table(*, table: np.ndarray) -> np.ndarray:
    """Return the table with upper offsets rising and lower offsets falling along the lead."""
    out = table.copy()
    upper = common.LEVELS > 0.5
    lower = common.LEVELS < 0.5
    out[..., upper] = np.maximum.accumulate(table[..., upper], axis=2)
    out[..., lower] = np.minimum.accumulate(table[..., lower], axis=2)
    return out


def offsets_stepped(
    *, table: np.ndarray, band: np.ndarray, quarter: np.ndarray, lead_hours: np.ndarray
) -> np.ndarray:
    """Return each row's offsets read from its own lead band."""
    return table[band, quarter, lead_band(lead_hours=lead_hours)]


def offsets_interpolated(
    *, table: np.ndarray, band: np.ndarray, quarter: np.ndarray, lead_hours: np.ndarray
) -> np.ndarray:
    """Return each row's offsets, linearly interpolated between the lead-band centres.

    A lead below the first centre reads the first band's offsets and a lead above the last centre
    reads the last band's.
    """
    centres = np.array(common.LEAD_BAND_CENTRES_HOURS)
    low = np.clip(np.searchsorted(centres, lead_hours, side="right") - 1, 0, len(centres) - 2)
    weight = np.clip((lead_hours - centres[low]) / (centres[low + 1] - centres[low]), 0.0, 1.0)
    return (1.0 - weight)[:, None] * table[band, quarter, low] + weight[:, None] * table[
        band, quarter, low + 1
    ]


def build_arms(*, frame: pl.DataFrame, table: np.ndarray) -> dict[str, pl.DataFrame]:
    """Return the four arms' frames, each with the reference's rows and mean."""
    band = frame["hour_band"].to_numpy().astype(np.int64)
    quarter = frame["quarter"].to_numpy().astype(np.int64)
    lead = frame["lead_hours"].to_numpy().astype(np.float64)
    mean = frame["mean"].to_numpy().astype(np.float64)
    monotone = monotone_table(table=table)
    offsets = {
        "ref": offsets_stepped(table=table, band=band, quarter=quarter, lead_hours=lead),
        "i": offsets_interpolated(table=table, band=band, quarter=quarter, lead_hours=lead),
        "m": offsets_stepped(table=monotone, band=band, quarter=quarter, lead_hours=lead),
        "im": offsets_interpolated(table=monotone, band=band, quarter=quarter, lead_hours=lead),
    }
    arms = {}
    for name, offset in offsets.items():
        quantiles = mean[:, None] + offset
        arms[name] = frame.with_columns(
            *[
                pl.Series(q, quantiles[:, position].astype(np.float32))
                for position, q in enumerate(common.QUANTILES)
            ],
            arm=pl.lit(name),
        )
    return arms


def _job(site: str) -> str:
    frames = {
        period: pl.read_parquet(common.reference_path(period=period, site=site)).select(
            list(common.BASE_COLUMNS)
        )
        for period in common.PERIODS
    }
    tables = {period: recover_table(frame=frame) for period, frame in frames.items()}
    table = fill_empty_cells(table=tables["tuning"], others=[tables["replicate"], tables["scored"]])
    rows = 0
    reproduction = 0.0
    for period, frame in frames.items():
        arms = build_arms(frame=frame, table=table)
        reproduction = max(
            reproduction,
            float(
                np.abs(
                    arms["ref"].select(list(common.QUANTILES)).to_numpy()
                    - frame.select(list(common.QUANTILES)).to_numpy()
                ).max()
            ),
        )
        for name, arm_frame in arms.items():
            target = common.out_path(arm=name, period=period, site=site)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            arm_frame.select(list(OUTPUT_COLUMNS)).write_parquet(temporary)
            temporary.rename(target)
            rows += arm_frame.height
    np.save(common.WORK_DIR / f"table_{site}.npy", table)
    return f"{site}: {rows:,} rows, largest reproduction gap {reproduction:.2e} points"


def main() -> None:
    """Build every arm at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, common.SCORED_SITES):
            print(line, flush=True)


if __name__ == "__main__":
    main()
