"""The redesigned known-answer gate of experiment 197.

Experiment 190 injected a one-signed offset into the observed power of ordinary days and scored
every
pair of (origin, target half-hour), including the many pairs for which no reading since the offset
began could reach the target's hour band. A filter with one state per band cannot recover those
pairs. This gate keeps experiment 190's injection (same seed, so the same days and offsets) and its
origins, and labels every pair by what the filter could know:

- `band_read`: the target's own band has a reading since the offset began.
- `other_read`: the target's band has none, but another band has one.
- `unread`: no band has a reading since the offset began, so no filter can know about the offset.

Each pair also carries the hours since the offset began. The script writes, for every variant, the
summed loss and the pair count per case, size, duration, hours-since-onset bin, and read class, so
that `e00197_report.py` can resample whole cases for the intervals. A matched 8 days per substation
with no injection show the harm.

The variants are A0 (no boost), A1 (gated boost, per band) and A3 (gated boost, level shared across
bands) on the grid m in {3, 5, 10, 20, 50}, k in {2, 4, 6}, z = 0.5, plus experiment 190's frozen
setting (m 5, k 8, z 0.5) for A1 and A3.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00197-*/e00197_gate.py [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Final

import e00197_base as base
import e00197_common as common
import e00197_gated as gated
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import CorrectionTrace, correction_at

SITES: Final[tuple[str, ...]] = common.DECISION_SITES
SIZES: Final[tuple[float, ...]] = (0.05, 0.10, 0.20)
DURATIONS_H: Final[tuple[int, ...]] = (8, 16, 36)
DAYS_PER_CELL_PER_SITE: Final[int] = 8
ORIGIN_STEP: Final[int] = 12
"""Half-hours between origins (6 hours)."""
DELAY: Final[int] = 7
"""The telemetry delay in half-hours (3.5 hours)."""
MAX_LEAD: Final[int] = 72
"""Half-hours after the origin that an origin scores (36 hours)."""
HOURS_BINS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-3.5 h", 0.0, 3.5),
    ("3.5-6 h", 3.5, 6.0),
    ("6-12 h", 6.0, 12.0),
    ("12-24 h", 12.0, 24.0),
    ("24-36 h", 24.0, 36.0),
)
READ_CLASSES: Final[tuple[str, ...]] = ("band_read", "other_read", "unread")
SEED: Final[int] = 190
"""Experiment 190's seed, so the injected days and offsets are the same."""
GRID_M: Final[tuple[float, ...]] = (3.0, 5.0, 10.0, 20.0, 50.0)
GRID_K: Final[tuple[int, ...]] = (2, 4, 6)
GRID_Z: Final[float] = 0.5
LEVELS: Final = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
OUT_PATH: Final = common.WORK_DIR / "gate" / "cells.parquet"


def variant_name(*, mode: str, m: float, k: int) -> str:
    """Return the variant's label, such as `a1_m10_k4` (z is fixed at `GRID_Z`)."""
    return f"{'a1' if mode == 'gated' else 'a3'}_m{m:g}_k{k}"


def variants() -> dict[str, gated.Gate]:
    """Return A0, the grid for A1 and A3, and experiment 190's frozen setting for both."""
    out: dict[str, gated.Gate] = {"a0": gated.Gate(mode="none")}
    for mode, m, k in product(("gated", "shared"), GRID_M, GRID_K):
        out[variant_name(mode=mode, m=m, k=k)] = gated.Gate(mode=mode, m=m, k=k, z=GRID_Z)
    out["a1_e190"] = gated.Gate(mode="gated", m=5.0, k=8, z=0.5)
    out["a3_e190"] = gated.Gate(mode="shared", m=5.0, k=8, z=0.5)
    return out


def pinball(*, error: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss of quantiles `prediction + table`, per error."""
    diff = error[:, None] - table
    return 2.0 * np.mean(np.maximum(LEVELS * diff, (LEVELS - 1.0) * diff), axis=1)


def pairs_for(*, starts: np.ndarray, ends: np.ndarray, n_grid: int) -> pl.DataFrame:
    """Return the (case, cutoff, target) pairs of every origin of every case.

    Origins come every 6 hours from 6 hours before the case's first position. Each scores the
    case's half-hours within 36 hours after it.
    """
    rows = []
    for case, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        for origin in range(start - ORIGIN_STEP, end, ORIGIN_STEP):
            cutoff = origin - DELAY
            low = max(origin, start)
            high = min(origin + MAX_LEAD, end, n_grid)
            if cutoff < 0 or high <= low:
                continue
            target = np.arange(low, high)
            rows.append(
                pl.DataFrame(
                    {
                        "case": np.full(len(target), case, dtype=np.int32),
                        "cutoff": np.full(len(target), cutoff, dtype=np.int64),
                        "target": target.astype(np.int64),
                        "start": np.full(len(target), start, dtype=np.int64),
                        "since_start_h": (target - start) / 2.0,
                    }
                )
            )
    return pl.concat(rows)


def ordinary_days(*, context: base.SiteContext) -> np.ndarray:
    """Return the first grid position of every replicate-month UTC day that is ordinary.

    A day is ordinary when it has 46 or more readings, none in a hand-filtered switching event.
    """
    cache = context.cache
    features = cache.sites[context.site]
    days = cache.end_times.astype("datetime64[D]")
    months = days.astype("datetime64[M]")
    out = []
    for day in np.unique(days):
        month = months[days == day][0]
        if (int(str(month)[:4]), int(str(month)[5:7])) not in common.REPLICATE_MONTHS:
            continue
        rows = np.flatnonzero(days == day)
        if (
            len(rows) < 46
            or not np.isfinite(features.y[rows]).all()
            or features.switching_mask[rows].any()
        ):
            continue
        out.append(int(rows[0]))
    return np.asarray(out)


def spread_table(*, context: base.SiteContext, stream: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Return the clean A0 errors' quantile offsets by hour band, shape (4, 11).

    The errors are those of origins every 6 hours from each tuning-month day's 09:00 UTC over the
    36 hours that follow each origin.
    """
    cache = context.cache
    clean = gated.gated_stream(
        residual=stream, band=band, settings=base.pool_settings(), gate=gated.Gate(mode="none")
    )
    hours = cache.end_times.astype("datetime64[h]").astype(np.int64) % 24
    minutes = (cache.end_times.astype("datetime64[m]").astype(np.int64)) % 60
    origins = np.flatnonzero((hours == 9) & (minutes == 0))
    months = cache.end_times.astype("datetime64[M]")
    tuning = np.array(
        [(int(str(m)[:4]), int(str(m)[5:7])) in common.TUNING_MONTHS for m in months[origins]]
    )
    origins = origins[tuning]
    pairs = pairs_for(starts=origins, ends=origins + MAX_LEAD, n_grid=len(stream)).filter(
        pl.col("target") >= 0
    )
    correction = correction_at(
        trace=clean.stream.trace,
        cutoff=pairs["cutoff"].to_numpy(),
        target=pairs["target"].to_numpy(),
        band=band[pairs["target"].to_numpy()].astype(np.int64),
    )
    error = stream[pairs["target"].to_numpy()] - correction.total
    pair_band = band[pairs["target"].to_numpy()]
    table = np.zeros((4, 11))
    for index in range(4):
        values = error[(pair_band == index) & np.isfinite(error)]
        table[index] = np.quantile(values, LEVELS)
    return table


def read_class(
    *, pairs: pl.DataFrame, band: np.ndarray, clean_trace: CorrectionTrace
) -> np.ndarray:
    """Return each pair's read class: what readings since the offset began could reach the target.

    Args:
        pairs: The pairs, with `cutoff`, `target`, and `start`.
        band: The hour band of each grid row.
        clean_trace: Any trace of the stream (it lists each band's reading positions).

    Returns:
        An array of `READ_CLASSES` labels.
    """
    cutoff = pairs["cutoff"].to_numpy()
    start = pairs["start"].to_numpy()
    target_band = band[pairs["target"].to_numpy()].astype(np.int64)
    band_read = np.zeros(len(cutoff), dtype=bool)
    any_read = np.zeros(len(cutoff), dtype=bool)
    for index in range(4):
        positions = clean_trace.positions[index]
        # The newest reading of this band at or before the cutoff.
        found = np.searchsorted(positions, cutoff, side="right") - 1
        newest = np.where(found >= 0, positions[np.maximum(found, 0)], -1)
        read_here = newest >= start
        any_read |= read_here
        band_read |= read_here & (target_band == index)
    return np.where(band_read, "band_read", np.where(any_read, "other_read", "unread"))


def losses_of(
    *,
    target_stream: np.ndarray,
    traces: dict,
    pairs: pl.DataFrame,
    band: np.ndarray,
    table: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return each variant's loss at every pair: `target_stream` minus the variant's correction."""
    target = pairs["target"].to_numpy()
    pair_band = band[target].astype(np.int64)
    out = {}
    for name, trace in traces.items():
        correction = correction_at(
            trace=trace, cutoff=pairs["cutoff"].to_numpy(), target=target, band=pair_band
        )
        error = target_stream[target] - correction.total
        out[name] = np.where(
            np.isfinite(error), pinball(error=np.nan_to_num(error), table=table[pair_band]), np.nan
        )
    return out


def aggregate(
    *, losses: dict[str, np.ndarray], pairs: pl.DataFrame, classes: np.ndarray, meta: dict
) -> list[dict]:
    """Return summed loss and pair count per case, bin, read class, and variant."""
    rows: list[dict] = []
    since = pairs["since_start_h"].to_numpy()
    case = pairs["case"].to_numpy()
    for name, loss in losses.items():
        frame = (
            pl.DataFrame(
                {
                    "case": case,
                    "since": since,
                    "read_class": classes,
                    "loss": loss,
                }
            )
            .with_columns(pl.col("loss").fill_nan(None))
            .drop_nulls()
        )
        for label, low, high in HOURS_BINS:
            part = frame.filter((pl.col("since") >= low) & (pl.col("since") < high))
            summary = part.group_by("case", "read_class").agg(
                pl.col("loss").sum().alias("loss_sum"), pl.len().alias("n")
            )
            rows.extend(
                {**meta, "variant": name, "bin": label, **row}
                for row in summary.iter_rows(named=True)
            )
    return rows


def _site_job(site: str) -> list[dict]:
    """Return the aggregated pair losses of every variant, case, size, and duration at a site."""
    torch.set_num_threads(1)
    started = time.monotonic()
    context = base.load_site(site=site)
    cache = context.cache
    features = cache.sites[site]
    band = cache.calendar.band
    settings = base.pool_settings()
    clean = base.residual_stream(context=context)
    table = spread_table(context=context, stream=clean, band=band)
    arms = variants()
    clean_run = gated.gated_stream(residual=clean, band=band, settings=settings, gate=arms["a0"])
    clean_traces = {
        name: gated.gated_stream(
            residual=clean, band=band, settings=settings, gate=gate
        ).stream.trace
        for name, gate in arms.items()
    }
    days = ordinary_days(context=context)
    rng = np.random.default_rng([SEED, sum(site.encode())])
    rows: list[dict] = []
    alone = rng.choice(days, size=DAYS_PER_CELL_PER_SITE, replace=False)
    pairs = pairs_for(starts=alone, ends=alone + 48, n_grid=len(clean))
    classes = read_class(pairs=pairs, band=band, clean_trace=clean_run.stream.trace)
    losses = losses_of(
        target_stream=clean, traces=clean_traces, pairs=pairs, band=band, table=table
    )
    rows.extend(
        aggregate(
            losses=losses,
            pairs=pairs,
            classes=np.full(len(classes), "alone"),
            meta={"site": site, "kind": "alone", "size": 0.0, "hours": 0},
        )
    )
    for size in SIZES:
        for hours in DURATIONS_H:
            chosen = rng.choice(days, size=DAYS_PER_CELL_PER_SITE * 3, replace=False)
            picked: list[int] = []
            for day in chosen.tolist():
                if all(abs(day - other) >= 144 for other in picked):
                    picked.append(day)
                if len(picked) == DAYS_PER_CELL_PER_SITE:
                    break
            starts = np.asarray(picked) + rng.integers(0, 48, size=len(picked))
            ends = starts + 2 * hours
            injected = clean.copy()
            for start, end, day in zip(starts.tolist(), ends.tolist(), picked, strict=True):
                injected[start:end] += size * float(np.nanmedian(features.y[day : day + 48]))
            traces = {
                name: gated.gated_stream(
                    residual=injected, band=band, settings=settings, gate=gate
                ).stream.trace
                for name, gate in arms.items()
            }
            pairs = pairs_for(starts=starts, ends=ends, n_grid=len(clean))
            classes = read_class(pairs=pairs, band=band, clean_trace=traces["a0"])
            hurt = losses_of(
                target_stream=injected, traces=traces, pairs=pairs, band=band, table=table
            )
            hurt["clean_a0"] = losses_of(
                target_stream=clean, traces={"a0": clean_traces["a0"]}, pairs=pairs, band=band,
                table=table,
            )["a0"]  # fmt: skip
            rows.extend(
                aggregate(
                    losses=hurt,
                    pairs=pairs,
                    classes=classes,
                    meta={"site": site, "kind": "injected", "size": size, "hours": hours},
                )
            )
    print(f"{site}: gate done, {time.monotonic() - started:.0f} s", flush=True)
    return rows


def main() -> None:
    """Run the gate at the five decision Primaries and write `work/gate/cells.parquet`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="*", default=list(SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_site_job, args.sites))
    frame = pl.DataFrame([row for part in results for row in part], infer_schema_length=None)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT_PATH)
    print(frame.shape)


if __name__ == "__main__":
    main()
