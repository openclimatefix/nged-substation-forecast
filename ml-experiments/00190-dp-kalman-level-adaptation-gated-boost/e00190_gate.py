"""The known-answer gate of experiment 190: inject a one-signed offset and measure the recovery.

A one-signed offset of 0.05, 0.10, or 0.20 of the day's median load, lasting 8, 16, or 36 hours, is
added to the observed power on random ordinary days of the replicate months (the even months of
2024-07 to 2025-06), 8 days per substation per size and duration (40 injected days). The filter
reads the injected power; the forecast of each target half-hour is the proxy-weather prediction
plus the filter's correction, so its error is the injected residual minus the correction. Origins
come every 6 hours from 6 hours before the offset starts, with the 3.5-hour telemetry delay of
experiment 168, and each origin scores the offset's half-hours within 36 hours after it.

The loss is twice the mean pinball loss over the 11 quantiles, with the quantile offsets taken from
the clean A0 errors of the tuning months by hour band. A matched set of 40 ordinary days with no
injection (8 per substation) shows the harm of the boost where nothing happened.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00190-*/e00190_gate.py [--workers 4]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00190_base as base
import e00190_common as common
import e00190_gated as gated
import e00190_run as run
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import correction_at

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
SEED: Final[int] = 190
VARIANTS: Final[dict[str, gated.Gate]] = {
    "a0": gated.Gate(mode="none"),
    "a1": gated.Gate(mode="gated"),  # replaced by the frozen setting in `variants`
    "a2": gated.Gate(mode="always"),
    "a3": gated.Gate(mode="shared"),
    "a1_preregistered": run.PREREGISTERED,
    "a1_fast": gated.Gate(mode="gated", m=10.0, k=4, z=0.5),
    "a1_x100": gated.Gate(mode="gated", m=100.0, k=4, z=0.5),
    "a2_x100": gated.Gate(mode="always", m=100.0),
    "a3_x100": gated.Gate(mode="shared", m=100.0, k=4, z=0.5),
}
LEVELS: Final = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])


def variants() -> dict[str, gated.Gate]:
    """Return the arms of the gate, with A1, A2, and A3 at the frozen setting."""
    frozen = json.loads(run.FROZEN_PATH.read_text())
    out = dict(VARIANTS)
    for name, mode in (("a1", "gated"), ("a2", "always"), ("a3", "shared")):
        out[name] = gated.Gate(mode=mode, m=frozen["m"], k=frozen["k"], z=frozen["z"])
    return out


def pinball(*, error: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss of quantiles `prediction + table`, per error.

    Args:
        error: Observed minus predicted mean, shape (n,).
        table: The quantile offsets, shape (n, 11).

    Returns:
        The loss of each error, in points.
    """
    diff = error[:, None] - table
    return 2.0 * np.mean(np.maximum(LEVELS * diff, (LEVELS - 1.0) * diff), axis=1)


def pairs_for(*, starts: np.ndarray, ends: np.ndarray, n_grid: int) -> pl.DataFrame:
    """Return the (case, cutoff, target) pairs of every origin of every case.

    Args:
        starts: The first grid position of each case's window.
        ends: One past the last grid position of the window.
        n_grid: The grid length.

    Returns:
        One row per pair, with `case`, `cutoff`, `target`, and `since_start_h`.
    """
    rows = []
    for case, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        first_origin = start - ORIGIN_STEP
        for origin in range(first_origin, end, ORIGIN_STEP):
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
    out = []
    months = days.astype("datetime64[M]")
    for day in np.unique(days):
        month = months[days == day][0]
        stamp = (int(str(month)[:4]), int(str(month)[5:7]))
        if stamp not in common.REPLICATE_MONTHS:
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

    The errors are those of origins every 6 hours from 6 hours before each tuning-month day's
    09:00 UTC to 36 hours after it, over the 36 hours that follow each origin.
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


def losses_of(
    *,
    stream: np.ndarray,
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
    del stream
    return out


def _site_job(site: str) -> list[dict]:
    """Return the pair losses of every variant, case, size, and duration at a substation."""
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
    clean_traces = {
        name: gated.gated_stream(
            residual=clean, band=band, settings=settings, gate=gate
        ).stream.trace
        for name, gate in arms.items()
    }
    days = ordinary_days(context=context)
    rng = np.random.default_rng([SEED, sum(site.encode())])
    rows: list[dict] = []
    # Left-alone days: the same window length as the 36-hour cases, no injection.
    alone = rng.choice(days, size=DAYS_PER_CELL_PER_SITE, replace=False)
    pairs = pairs_for(starts=alone, ends=alone + 48, n_grid=len(clean))
    losses = losses_of(
        stream=clean, target_stream=clean, traces=clean_traces, pairs=pairs, band=band, table=table
    )
    for name, loss in losses.items():
        frame = pl.DataFrame({"case": pairs["case"], "loss": loss}).drop_nulls()
        for case, value, n in (
            frame.group_by("case").agg(pl.col("loss").mean(), pl.len()).iter_rows()
        ):
            rows.append(
                {"site": site, "kind": "alone", "size": 0.0, "hours": 0, "variant": name,
                 "case": case, "loss": value, "n": n, "since_start_h": -1.0}
            )  # fmt: skip
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
                median = float(np.nanmedian(features.y[day : day + 48]))
                injected[start:end] += size * median
            traces = {
                name: gated.gated_stream(
                    residual=injected, band=band, settings=settings, gate=gate
                ).stream.trace
                for name, gate in arms.items()
            }
            pairs = pairs_for(starts=starts, ends=ends, n_grid=len(clean))
            hurt = losses_of(
                stream=clean, target_stream=injected, traces=traces, pairs=pairs, band=band,
                table=table,
            )  # fmt: skip
            clean_a0 = losses_of(
                stream=clean, target_stream=clean, traces={"a0": clean_traces["a0"]}, pairs=pairs,
                band=band, table=table,
            )["a0"]  # fmt: skip
            hurt["clean_a0"] = clean_a0
            for name, loss in hurt.items():
                frame = pl.DataFrame(
                    {
                        "case": pairs["case"],
                        "since_start_h": pairs["since_start_h"],
                        "loss": loss,
                    }
                ).drop_nulls()
                for label, low, high in (("all", 0.0, 1e9), *HOURS_BINS):
                    part = frame.filter(
                        (pl.col("since_start_h") >= low) & (pl.col("since_start_h") < high)
                    )
                    for case, value, n in (
                        part.group_by("case").agg(pl.col("loss").mean(), pl.len()).iter_rows()
                    ):
                        rows.append(
                            {"site": site, "kind": "injected", "size": size, "hours": hours,
                             "variant": name, "case": case, "loss": value, "n": n,
                             "bin": label, "since_start_h": 0.0}
                        )  # fmt: skip
    print(f"{site}: gate done, {time.monotonic() - started:.0f} s", flush=True)
    return rows


def main() -> None:
    """Run the gate at the five decision Primaries and write `work/gate/cases.parquet`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_site_job, run.SITES))
    frame = pl.DataFrame([row for part in results for row in part], infer_schema_length=None)
    target = common.WORK_DIR / "gate" / "cases.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)
    print(frame.shape)


if __name__ == "__main__":
    main()
