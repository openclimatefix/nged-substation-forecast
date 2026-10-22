"""Build arm D (R0) and the network-before-filter arm (PRE) at every substation.

Every arm starts from experiment 136's arm A2 rows, whose `mean` is the convex forecast plus the
Kalman filter's level and anomaly. The convex forecast alone is `raw = mean - level - anomaly`.

- `r0`: the pipeline of experiment 155's arm D with no network. It must reproduce the stored arm D
  (a control).
- `pre_s<k>`: the network's correction `delta` is subtracted from the residual stream the filter
  reads (the stream is `observed - convex forecast`, so the filter then tracks the error of the
  network-corrected convex forecast); the mean of each row is `raw + delta + level' + anomaly'`
  with the new filter's state at the row's cut-off. The winter night offset, recipe R and the
  spread refit follow exactly as in arm D. `k` scales the filter's observation noise (`1.00` is
  POOL's setting).

    uv run --no-sync python ml-experiments/00222-*/e00222_build.py [--workers 4]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00222_common as common
import e00222_corrections as corr
import e00222_fit as fit
import e00222_stream as stream_module
import numpy as np
import polars as pl

ARMS_DIR: Final[Path] = common.WORK_DIR / f"arms{common.VARIANT}"
NOISE_SCALES: Final[tuple[float, ...]] = (0.75, 1.0, 1.33)
"""The pre-registered grid for PRE-T: factors on the filter's observation standard deviation."""
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")


def read_a2(*, site: str) -> pl.DataFrame:
    """Return experiment 136's arm A2 rows at a site with their filter level and anomaly."""
    pieces = [
        pl.read_parquet(
            common.E136_WORK_DIR / "dp_preds" / f"a2_{period}_{site}.parquet"
        ).with_columns(period=pl.lit(period))
        for period in PERIODS
    ]
    return pl.concat(pieces).sort(["power_fcst_init_time", "valid_time"])


def arm_d(*, base: pl.DataFrame) -> pl.DataFrame:
    """Apply the winter night offset, recipe R, and the spread refit, as experiment 155's arm D."""
    framed = corr.add_calendar(frame=base)
    mean_a = framed["mean"].to_numpy().astype(np.float64)
    offset, _ = corr.winter_offset_shift(frame=framed, variant="winter")
    shift = offset + corr.recipe_r_shift(frame=framed, forecast=mean_a + offset)
    return corr.apply_shift_and_refit_spread(frame=framed, shift=shift)


def corrected_stream(*, site: str, stream: np.ndarray) -> np.ndarray:
    """Return the residual stream with the network's correction subtracted."""
    stored = np.load(fit.NN_DIR / f"{site}.npz")
    return np.where(stored["has_delta"], stream - stored["delta"], stream)


def pre_frame(
    *,
    site: str,
    base: pl.DataFrame,
    site_stream: stream_module.SiteStream,
    noise_scale: float,
    stream: np.ndarray | None = None,
    delta_scale: float = 1.0,
) -> pl.DataFrame:
    """Return the network-before-filter frame at a site, before arm D's corrections.

    Args:
        site: The site's label.
        base: Arm A2's rows with the filter's level and anomaly.
        site_stream: The site's cache and arm A2 residual stream.
        noise_scale: The factor on the filter's observation standard deviation.
        stream: The residual stream to filter, or `None` for the network-corrected stream.
        delta_scale: The factor on the network's correction (0 gives the uncorrected arm).

    Returns:
        `base` with `mean`, `level`, and `anomaly` replaced, plus `raw` and `delta`.
    """
    settings = stream_module.pool_settings()
    settings = replace(settings, sigma_eps=settings.sigma_eps * noise_scale)
    delta_stream = np.load(fit.NN_DIR / f"{site}.npz")
    if stream is None:
        stream = np.where(
            delta_stream["has_delta"],
            site_stream.stream - delta_scale * delta_stream["delta"],
            site_stream.stream,
        )
    trace = stream_module.run_filter(cache=site_stream.cache, stream=stream, settings=settings)
    level, anomaly = stream_module.corrections_for_rows(
        cache=site_stream.cache, trace=trace, frame=base
    )
    rows = pl.read_parquet(fit.NN_DIR / f"{site}_rows.parquet")
    joined = base.join(rows, on=["run", "valid_time"], how="left")
    if joined.height != base.height or joined["delta"].null_count() > 0:
        msg = f"{site}: network corrections do not cover every forecast row"
        raise ValueError(msg)
    raw = (base["mean"] - base["level"] - base["anomaly"]).to_numpy().astype(np.float64)
    delta = joined["delta"].to_numpy().astype(np.float64) * delta_scale
    mean = raw + delta + level + anomaly
    return base.with_columns(
        pl.Series("mean", mean.astype(np.float32)),
        pl.Series("level", level.astype(np.float32)),
        pl.Series("anomaly", anomaly.astype(np.float32)),
        pl.Series("raw", raw.astype(np.float32)),
        pl.Series("delta", delta.astype(np.float32)),
    )


def write_arm(*, name: str, site: str, frame: pl.DataFrame) -> None:
    """Write one arm's frame at one site, atomically."""
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    target = ARMS_DIR / f"{name}_{site}.parquet"
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)


def _job(site: str) -> str:
    base = read_a2(site=site)
    site_stream = stream_module.load_site_stream(site=site)
    # R0: arm D from arm A2's stored rows, then the filter re-run on the unedited stream.
    write_arm(name="r0", site=site, frame=arm_d(base=base))
    control = pre_frame(
        site=site, base=base, site_stream=site_stream, noise_scale=1.0, delta_scale=0.0
    )
    write_arm(name="r0_rerun", site=site, frame=arm_d(base=control))
    for scale in NOISE_SCALES:
        frame = pre_frame(site=site, base=base, site_stream=site_stream, noise_scale=scale)
        write_arm(name=f"pre_s{scale:.2f}", site=site, frame=arm_d(base=frame))
    return f"{site}: {base.height:,} rows"


def main() -> None:
    """Build every arm at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, common.SCORED_SITES):
            print(line, flush=True)


if __name__ == "__main__":
    main()
