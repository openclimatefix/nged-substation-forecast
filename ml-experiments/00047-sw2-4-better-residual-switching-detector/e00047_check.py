"""Check that the fast residual agrees with a refit on injected power, for the new arm.

The arms are fitted on real power, and `fast_residual` adds the injected MW to the real residual.
This script takes the batch with the most headline-band transfers, refits the new arm's seed-0
models on injected power for every series the batch uses and every month that holds one of its
transfers, and compares the outcomes of the two residuals for each transfer-and-setting pair. The
check passes when the found and start-found outcomes agree for at least 95% of the pairs.

    uv run python ml-experiments/00047-*/e00047_check.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00047_common import GRID_LENGTH, GRID_START, WORK_DIR
from e00047_detect import (
    SEED,
    ArmInputs,
    ArmPanel,
    capacities,
    detect_all,
    drawn_events,
    load_arm_panel,
    rows_for_events,
)
from e00047_fit import FITS_DIR, fit_months, load_frame
from e00047_inject import HEADLINE_BAND, Event, batches, inject_power

ARM: Final[str] = "HOL_s0"
MIN_AGREEMENT: Final[float] = 0.95
N_WORKERS: Final[int] = 4


def _next_month(*, month: datetime) -> datetime:
    return datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=UTC)


def event_months(*, events: list[Event]) -> dict[int, set[datetime]]:
    """Return, for each series a batch uses, the months that hold a stamp of one of its events."""
    out: dict[int, set[datetime]] = {}
    for event in events:
        first = GRID_START + timedelta(minutes=30 * event.onset)
        last = GRID_START + timedelta(minutes=30 * (event.end - 1))
        month = datetime(first.year, first.month, 1, tzinfo=UTC)
        months: set[datetime] = set()
        while month <= last:
            months.add(month)
            month = _next_month(month=month)
        for series in event.series:
            out.setdefault(series, set()).update(months)
    return out


def _refit(args: tuple[int, set[datetime], np.ndarray, float]) -> tuple[int, np.ndarray]:
    """Refit the new arm for one series on injected target, returning expected MW per month."""
    series, months, delta, capacity = args
    frame = load_frame(time_series_id=series, arm="HOL")
    stamps = ((frame["valid_time"].dt.epoch("s") - int(GRID_START.timestamp())) // 1800).to_numpy()
    inside = (stamps >= 0) & (stamps < GRID_LENGTH)
    shift = np.zeros(frame.height)
    shift[inside] = delta[stamps[inside]] / capacity
    frame = frame.with_columns(target=pl.col("target") + pl.Series(shift).cast(pl.Float32))
    refit = fit_months(frame=frame, arm="HOL", seed=0, only_months=months)
    out = np.full(GRID_LENGTH, np.nan)
    refit_stamps = (
        (refit["valid_time"].dt.epoch("s") - int(GRID_START.timestamp())) // 1800
    ).to_numpy()
    out[refit_stamps] = refit["prediction"].to_numpy().astype(np.float64) * capacity
    return series, out


def main() -> None:
    """Run the check and print the agreement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    parser.add_argument("--fits", type=Path, default=FITS_DIR)
    parser.add_argument("--fallback", action="store_true", help="smoke test: fill missing months")
    parser.add_argument("--max-series", type=int, default=None, help="smoke test: refit fewer")
    args = parser.parse_args()
    panel = load_arm_panel(name=ARM, fits_dir=args.fits, fallback=args.fallback)
    thresholds = json.loads((args.out / f"thresholds_{ARM}.json").read_text())
    scales = ArmInputs.of(residual=panel.residual).scales()
    events = drawn_events(power=panel.power)
    groups = batches(events=events)
    headline = [
        sum((e.size_band, e.duration_band) == HEADLINE_BAND for e in group) for group in groups
    ]
    index = int(np.argmax(headline))
    batch = groups[index]
    print(f"batch {index}: {len(batch)} transfers, {headline[index]} in the headline band")
    injection = inject_power(power=panel.power, events=batch, seed=SEED + index)
    capacity = capacities()
    months = event_months(events=batch)
    series_list = sorted(months)[: args.max_series]
    jobs = [(s, months[s], injection.delta[s], capacity[s]) for s in series_list]
    print(f"refitting {sum(len(months[s]) for s in series_list)} series-months")
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        refits = dict(pool.map(_refit, jobs))
    expected = {s: v.copy() for s, v in panel.expected.items()}
    for series, array in refits.items():
        mask = np.isfinite(array)
        expected[series][mask] = array[mask]
    refit_panel = ArmPanel(
        "refit",
        injection.power,
        expected,
        {s: injection.power[s] - expected[s] for s in expected},
    )
    runs = {
        "fast": detect_all(
            residual=injection.fast_residual(panel.residual),
            expected=panel.expected,
            scales=scales,
            thresholds=thresholds,
        ),
        "refit": detect_all(
            residual=refit_panel.residual,
            expected=refit_panel.expected,
            scales=scales,
            thresholds=thresholds,
        ),
    }
    rows = {k: pl.DataFrame(rows_for_events(batch=batch, results=v)) for k, v in runs.items()}
    keys = ["event_id", "setting", "variant"]
    joined = (
        rows["fast"]
        .join(rows["refit"], on=keys, suffix="_refit", validate="1:1")
        .filter(pl.col("variant") == "plain")
    )
    agree = (pl.col("found") == pl.col("found_refit")) & (
        pl.col("onset_found") == pl.col("onset_found_refit")
    )
    share = float(joined.select(agree.mean()).item())
    print(f"{joined.height} transfer-setting pairs; found and start-found agree for {share:.3f}")
    for setting in sorted(joined["setting"].unique()):
        part = joined.filter(pl.col("setting") == setting)
        print(
            f"  {setting}: agree {float(part.select(agree.mean()).item()):.3f}; found fast "
            f"{int(part['found'].sum())}, refit {int(part['found_refit'].sum())} of {part.height}"
        )
    joined.write_parquet(args.out / "injection_check.parquet")
    print("PASS" if share >= MIN_AGREEMENT else "FAIL: report as a limit shared by both arms")


if __name__ == "__main__":
    main()
