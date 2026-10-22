"""Inject gaps into the readings of the event runs, forecast under every rule, and score them.

For each decision substation the script chooses event runs, then for each gap length L in hours and
each distance k in hours between the gap's end and the run's data cutoff, hides the readings of
the gap from the filter and rebuilds the forecast of every run under these rules:

| Rule | Forecast |
|---|---|
| `n` | no gap: experiment 155's arm D rebuilt (filter, winter night offset, recipe R) |
| `a` | coast: arm D on the gapped stream |
| `b` | coast with widened bands: `a` with its quantile offsets from the mean widened |
| `c2` | the frozen model with its own offset and recipe R, no filter |
| `c` | the frozen model alone, no filter, offset, or recipe R |
| `a0`, `b0` | `a` and `b` with the filter alone (no offset, no recipe R) |
| `n0` | no gap, filter alone |

The output is one parquet file per substation under `work/rows/`, long by (L, k, rule), holding
the per-row losses of the event runs' rows at leads of 0 to 168 hours outside the switching events.
The no-gap run (`n`, `n0`, `c`) is written with a gap length of 0. The controls are written to
`work/controls/<site>.json`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00205-*/e00205_run.py [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00205_base as base
import e00205_common as common
import e00205_corrections as corr
import e00205_filters as filters
import e00205_prep as prep
import numpy as np
import polars as pl
from studies.dp_assimilation import CorrectionTrace
from studies.dp_gam import QUANTILE_COLUMNS

ROWS_DIR: Final[Path] = common.WORK_DIR / "rows"
GAP_HOURS: Final[tuple[int, ...]] = (3, 6, 24, 72)
DISTANCE_HOURS: Final[tuple[int, ...]] = (0, 3, 6, 12, 24, 48)
SEED: Final[int] = 205
BLOCK_DAYS: Final[int] = 8
EVENT_WINDOW_DAYS: Final[int] = 3
FIRST_EVENT_ORIGIN: Final[datetime] = datetime(2024, 7, 16, 9, tzinfo=UTC)
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
HORIZON: Final[float] = 168.0
GAUSSIAN_90_WIDTH: Final[float] = 3.29
"""The width of a Gaussian's central 90% interval in standard deviations."""
FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    "actual", "hour_band", "quarter", "switching_mask", "period", "target", "cutoff",
)  # fmt: skip


def choose_event_origins(
    *, origins: list[datetime], periods: dict[datetime, str]
) -> list[datetime]:
    """Return one origin per 8-day block, drawn from the block's first 3 days.

    Args:
        origins: Every run's origin at a substation.
        periods: Each origin's period.

    Returns:
        The event origins, ascending: replicate or scored runs on or after 2024-07-16.
    """
    rng = np.random.default_rng(SEED)
    known = set(origins)
    last = max(origins)
    chosen = []
    block_start = FIRST_EVENT_ORIGIN
    while block_start <= last:
        day = int(rng.integers(0, EVENT_WINDOW_DAYS))
        origin = block_start + timedelta(days=day)
        if origin in known and periods[origin] in ("replicate", "scored"):
            chosen.append(origin)
        block_start += timedelta(days=BLOCK_DAYS)
    return chosen


def gap_positions(*, cutoffs: np.ndarray, hours: int, distance: int) -> np.ndarray:
    """Return the grid positions hidden by the gaps that end `distance` hours before each cutoff."""
    if hours == 0:
        return np.empty(0, dtype=np.int64)
    ends = cutoffs - 2 * distance
    return np.unique(np.concatenate([np.arange(end - 2 * hours + 1, end + 1) for end in ends]))


def last_reading_gap(*, trace: CorrectionTrace, frame: pl.DataFrame) -> np.ndarray:
    """Return the half-hours from the target band's newest reading to the cutoff at each row."""
    out = np.zeros(frame.height)
    band = frame["hour_band"].to_numpy().astype(np.int64)
    cutoff = frame["cutoff"].to_numpy()
    for index in range(len(trace.positions)):
        rows = np.flatnonzero(band == index)
        positions = trace.positions[index]
        found = np.searchsorted(positions, cutoff[rows], side="right") - 1
        known = found >= 0
        newest = positions[np.maximum(found, 0)]
        out[rows] = np.where(known, cutoff[rows] - newest, np.nan)
    return out


def variance_growth(*, half_hours: np.ndarray, settings: filters.FilterSettings) -> np.ndarray:
    """Return the level and transient variance accrued over `half_hours`, in points squared."""
    phi = settings.phi
    level = settings.q * half_hours / 48.0
    transient = settings.sigma_z**2 * (1.0 - phi ** (2.0 * half_hours)) / (1.0 - phi**2)
    return level + transient


def widen(*, frame: pl.DataFrame, extra_variance: np.ndarray) -> pl.DataFrame:
    """Multiply each quantile's offset from the mean by sqrt(1 + extra variance / s^2)."""
    mean = frame["mean"].to_numpy().astype(np.float64)
    width = (frame["q95"].to_numpy() - frame["q05"].to_numpy()).astype(np.float64)
    scale = np.sqrt(1.0 + extra_variance / np.maximum((width / GAUSSIAN_90_WIDTH) ** 2, 1e-12))
    widened = extra_variance > 0.0
    return frame.with_columns(
        pl.Series(
            c,
            np.where(
                widened,
                mean + (frame[c].to_numpy().astype(np.float64) - mean) * scale,
                frame[c].to_numpy().astype(np.float64),
            ),
        )
        .cast(pl.Float32)
        .alias(c)
        for c in QUANTILE_COLUMNS
    )


def d_pipeline(*, frame: pl.DataFrame, hidden_actual: np.ndarray) -> pl.DataFrame:
    """Return arm D's frame: the winter night offset, then recipe R on the offset forecast.

    Args:
        frame: A forecast frame with `mean`, `period`, `actual`, and the columns of `add_calendar`.
        hidden_actual: The actual with the readings inside gaps set to `NaN`, for the offset and R.

    Returns:
        The frame with the corrected mean and a spread refitted on the tuning months.
    """
    mean = frame["mean"].to_numpy().astype(np.float64)
    shifted = frame.with_columns(pl.Series("actual", hidden_actual.astype(np.float32)))
    offset, _ = corr.winter_offset_shift(frame=shifted, variant="winter")
    shift = offset + corr.recipe_r_shift(frame=shifted, forecast=mean + offset)
    return corr.apply_shift_and_refit_spread(frame=frame, shift=shift)


def losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the per-row losses of a frame: quantile score, mean error, and tail terms."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    score = 2.0 * np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)
    out = {
        "qs": score,
        "abs_mean": np.abs(actual - frame["mean"].to_numpy()),
        "abs_median": np.abs(actual - frame["q50"].to_numpy()),
        "median_minus_mean": frame["q50"].to_numpy().astype(np.float64)
        - frame["mean"].to_numpy().astype(np.float64),
    }
    for name, level in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99)):
        diff = actual - frame[name].to_numpy().astype(np.float64)
        out[f"pin_{name}"] = np.maximum(level * diff, (level - 1.0) * diff)
        out[f"cov_{name}"] = (diff <= 0.0).astype(np.float64)
    return pl.DataFrame(out)


def scored_rows(*, frame: pl.DataFrame, events: list[int], rule: str, hours: int, distance: int):  # noqa: ANN201
    """Return the event runs' losses at leads of 0 to 168 hours outside switching events."""
    keep = (
        frame["run"].is_in(events)
        & (frame["lead_hours"] > 0.0)
        & (frame["lead_hours"] <= HORIZON)
        & ~frame["switching_mask"]
    )
    kept = frame.filter(keep)
    return pl.concat(
        [
            kept.select("run", "power_fcst_init_time", "valid_time", "lead_hours", "hour_band"),
            losses(frame=kept),
        ],
        how="horizontal",
    ).with_columns(rule=pl.lit(rule), gap_hours=pl.lit(hours), distance_hours=pl.lit(distance))


def read_stored_d(*, site: str) -> pl.DataFrame:
    """Return experiment 155's stored arm D at a substation, all three periods."""
    return pl.concat(
        [
            pl.read_parquet(
                common.WORK_DIR.parent.parent
                / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
                / "work"
                / "dp_preds"
                / f"d_{period}_{site}.parquet"
            ).select("run", "valid_time", "mean")
            for period in base.PERIODS
        ]
    )


def _job(site: str) -> str:  # noqa: PLR0915
    started = time.monotonic()
    frame0 = pl.read_parquet(prep.base_path(site=site))
    stream, band = prep.read_stream(site=site)
    settings = base.pool_settings()
    cutoff_of_run = frame0.group_by("run").agg(pl.col("cutoff").first()).sort("run")
    origin_of_run = frame0.group_by("run").agg(pl.col("power_fcst_init_time").first())
    period_of_run = frame0.group_by("run").agg(pl.col("period").first())
    runs = cutoff_of_run.join(origin_of_run, on="run").join(period_of_run, on="run")
    origins = [
        o.replace(tzinfo=UTC) if o.tzinfo is None else o for o in runs["power_fcst_init_time"]
    ]
    periods = dict(zip(origins, runs["period"].to_list(), strict=True))
    event_origins = choose_event_origins(origins=origins, periods=periods)
    events = runs.filter(pl.col("power_fcst_init_time").is_in(event_origins))
    event_runs = events["run"].to_list()
    event_cutoffs = events["cutoff"].to_numpy()
    true_actual = frame0["actual"].to_numpy().astype(np.float64)
    target = frame0["target"].to_numpy()

    def mean_frame_of(*, trace: CorrectionTrace | None) -> pl.DataFrame:
        if trace is None:
            return frame0.with_columns(
                mean=pl.col("base_mean").cast(pl.Float32), level=pl.lit(0.0), anomaly=pl.lit(0.0)
            )
        return base.arm_mean_frame(base=frame0, trace=trace)

    pinned: dict[str, pl.DataFrame] = {}

    def build(
        *, trace: CorrectionTrace | None, hidden: np.ndarray
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        mean_frame = mean_frame_of(trace=trace)
        if trace is not None and "no_gap" in pinned:
            # Tuning-month rows keep the no-gap filter's forecast, so every rule's spread table is
            # the no-gap one.
            is_tuning = (mean_frame["period"] == "tuning").to_numpy()
            reference = pinned["no_gap"]
            mean_frame = mean_frame.with_columns(
                pl.Series(c, np.where(is_tuning, reference[c].to_numpy(), mean_frame[c].to_numpy()))
                .cast(pl.Float32)
                .alias(c)
                for c in ("mean", "level", "anomaly")
            )
        a0 = corr.add_calendar(frame=base.arm_frame(mean_frame=mean_frame))
        return a0, d_pipeline(frame=a0, hidden_actual=hidden)

    trace0 = filters.filter_stream(residual=stream, band=band, settings=settings).trace
    pinned["no_gap"] = mean_frame_of(trace=trace0)
    filter_only_0, d_0 = build(trace=trace0, hidden=true_actual)
    gap0 = last_reading_gap(trace=trace0, frame=frame0)
    frozen_a0, _ = build(trace=None, hidden=true_actual)
    # The frozen model alone has a spread refitted on its tuning months, with no offset or R.
    out: list[pl.DataFrame] = [
        scored_rows(frame=d_0, events=event_runs, rule="n", hours=0, distance=0),
        scored_rows(frame=filter_only_0, events=event_runs, rule="n0", hours=0, distance=0),
        scored_rows(frame=frozen_a0, events=event_runs, rule="c", hours=0, distance=0),
    ]
    # Control 1: the rebuilt no-gap arm D reproduces the stored arm D of experiment 155.
    stored = read_stored_d(site=site)
    joined = d_0.select("run", "valid_time", "mean").join(
        stored.rename({"mean": "stored"}), on=["run", "valid_time"], how="inner"
    )
    reproduce = float(np.max(np.abs((joined["mean"] - joined["stored"]).to_numpy())))
    # Control 2: widening by zero extra variance leaves the quantiles unchanged.
    same = widen(frame=d_0, extra_variance=np.zeros(d_0.height))
    widen_zero = float(
        max(
            float(np.max(np.abs((same[c].cast(pl.Float64) - d_0[c].cast(pl.Float64)).to_numpy())))
            for c in QUANTILE_COLUMNS
        )
    )
    for hours in GAP_HOURS:
        for distance in DISTANCE_HOURS:
            masked = gap_positions(cutoffs=event_cutoffs, hours=hours, distance=distance)
            gapped = stream.copy()
            gapped[masked] = np.nan
            trace = filters.filter_stream(residual=gapped, band=band, settings=settings).trace
            hidden = np.where(np.isin(target, masked), np.nan, true_actual)
            filter_only, d_a = build(trace=trace, hidden=hidden)
            extra = np.maximum(
                variance_growth(
                    half_hours=last_reading_gap(trace=trace, frame=frame0), settings=settings
                )
                - variance_growth(half_hours=gap0, settings=settings),
                0.0,
            )
            extra = np.nan_to_num(extra)
            _, d_c2 = build(trace=None, hidden=hidden)
            for rule, frame in (
                ("a", d_a),
                ("b", widen(frame=d_a, extra_variance=extra)),
                ("c2", d_c2),
                ("a0", filter_only),
                ("b0", widen(frame=filter_only, extra_variance=extra)),
            ):
                out.append(
                    scored_rows(
                        frame=frame, events=event_runs, rule=rule, hours=hours, distance=distance
                    )
                )
    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    result = pl.concat(out).with_columns(site=pl.lit(site))
    target_path = ROWS_DIR / f"{site}.parquet"
    temporary = target_path.with_name(f"{target_path.name}.tmp")
    result.write_parquet(temporary)
    temporary.rename(target_path)
    controls = {
        "site": site,
        "n_events": len(event_runs),
        "reproduce_stored_arm_d_max_abs_mean_difference": reproduce,
        "widen_by_zero_max_abs_quantile_difference": widen_zero,
    }
    common.write_json_atomic(path=common.WORK_DIR / "controls" / f"{site}.json", content=controls)
    seconds = time.monotonic() - started
    return (
        f"{site}: {len(event_runs)} events, {result.height:,} rows, "
        f"reproduce {reproduce:.2e}, {seconds:.0f} s"
    )


def main() -> None:
    """Run every decision substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=list(common.DECISION_SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
