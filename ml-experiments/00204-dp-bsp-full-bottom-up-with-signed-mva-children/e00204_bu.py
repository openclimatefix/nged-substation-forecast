"""Build the full bottom-up forecasts of each BSP and write them in the `dp_preds` layout.

For a BSP `b` with children `i` and weights `w_i`, the observed reading decomposes exactly into the
remainder series and the weighted children (readings in points of each series' own effective
capacity):

    y_b = r + sum_i w_i * (cap_i / cap_b) * y_i

The bottom-up mean is the same sum of means: arm D's mean for the remainder, plus each child's mean.
A child that reports real power contributes experiment 155's arm D mean. A child that reports
apparent power contributes experiment 152's arm N mean in arm (c) (the naive reading), and its
latent
signed flow times the power factor in arms (d) and (d-p50). Every arm takes the quantiles of the
top-down forecast and shifts them by the difference of the means, so the spread table is the
top-down one (experiment 161's `r_tdspread` construction). The oracle replaces the children's means
with their observed readings as arm (d) sees them.

    uv run --no-sync python ml-experiments/00204-*/e00204_bu.py
"""

import json
import os
from pathlib import Path
from typing import Final

import e00204_common as common
import e00204_sign as sign
import numpy as np
import polars as pl

KEYS: Final[list[str]] = ["power_fcst_init_time", "valid_time"]
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *common.QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip
SCORED_PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
ARMS: Final[dict[str, str]] = {"c": "naive", "d": "fold", "dp50": "fold50", "oracle": "fold"}
"""Each arm's name and the remainder variant it uses."""


def td_path(*, period: str, bsp: str) -> Path:
    """Return the file of the top-down arm (arm D at the BSP), written by experiment 161."""
    return common.E161_OUT_DIR / f"d_{period}_{bsp}.parquet"


def remainder_path(*, period: str, bsp: str, variant: str) -> Path:
    """Return the file of arm D at a remainder series."""
    return common.OUT_DIR / f"d_{period}_{bsp}{common.VARIANT_SUFFIX[variant]}.parquet"


def out_path(*, arm: str, period: str, bsp: str) -> Path:
    """Return the file of a bottom-up arm for one period at a BSP."""
    return common.OUT_DIR / f"bu_{arm}_{period}_{bsp}.parquet"


def mva_path(*, period: str, child: str) -> Path:
    """Return experiment 152's stored forecasts of an apparent-power child."""
    return common.E152_PREDS_DIR / f"{period}_{child}.parquet"


def read_child(*, child: str, period: str, mode: str) -> pl.DataFrame | None:
    """Return a child's `mean`, `actual`, and switching mask for a period, or None if absent.

    `mode` is `naive` (arm N's mean) or `fold` (arm F's latent signed flow times the power factor)
    for an apparent-power child. A real-power child ignores `mode` and returns experiment 155's
    arm D mean.
    """
    if child in common.ALL_MVA:
        path = mva_path(period=period, child=child)
        if not path.exists():
            return None
        frame = pl.read_parquet(path)
        if mode == "naive":
            frame = frame.filter(pl.col("arm") == "N").with_columns(m=pl.col("mean"))
        else:
            frame = frame.filter(pl.col("arm") == "F_fold").with_columns(
                m=pl.col("flow") * common.POWER_FACTOR
            )
    else:
        path = common.E155_PREDS_DIR / f"d_spread_a_{period}_{child}.parquet"
        frame = pl.read_parquet(path).with_columns(m=pl.col("mean"))
    return frame.select(
        *KEYS, m=pl.col("m"), a=pl.col("actual"), k=pl.col("switching_mask")
    ).rename({"m": f"mean_{child}", "a": f"actual_{child}", "k": f"mask_{child}"})


def signed_actual(*, cache, child: str, valid_time: pl.Series, variant: str) -> np.ndarray:  # noqa: ANN001
    """Return a child's observed reading as arm (d) sees it, at the given valid times."""
    series = sign.child_series(cache=cache, child=child, variant=variant)
    stamps = valid_time.dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    rows = np.searchsorted(cache.end_times, stamps)
    ok = cache.end_times[np.clip(rows, 0, len(cache.end_times) - 1)] == stamps
    return np.where(ok, series[np.clip(rows, 0, len(series) - 1)], np.nan)


def build(*, bsp: str, period: str, arm: str, cache) -> pl.DataFrame | None:  # noqa: ANN001
    """Return one bottom-up arm at a BSP for a period, or None if an input file is missing."""
    variant = ARMS[arm]
    mode = "naive" if variant == "naive" else "fold"
    td = pl.read_parquet(td_path(period=period, bsp=bsp)).select(list(OUTPUT_COLUMNS))
    remainder = pl.read_parquet(remainder_path(period=period, bsp=bsp, variant=variant)).select(
        *KEYS, mean_r=pl.col("mean")
    )
    frame = td.join(remainder, on=KEYS, how="inner", validate="1:1")
    for child in common.BSPS[bsp]:
        part = read_child(child=child, period=period, mode=mode)
        if part is None:
            return None
        frame = frame.join(part, on=KEYS, how="inner", validate="1:1")
    cap = cache.sites[bsp].table.effective_capacity_mw
    weights = common.weights_for(cache=cache, bsp=bsp, variant=variant)
    total = frame["mean_r"].to_numpy().astype(np.float64)
    for child in common.BSPS[bsp]:
        scale = cache.sites[child].table.effective_capacity_mw / cap
        if arm == "oracle":
            if child in common.ALL_MVA:
                values = signed_actual(
                    cache=cache, child=child, valid_time=frame["valid_time"], variant=variant
                )
            else:
                values = frame[f"actual_{child}"].to_numpy().astype(np.float64)
        else:
            values = frame[f"mean_{child}"].to_numpy().astype(np.float64)
        total = total + weights[child] * scale * values
    masks = [pl.col(f"mask_{child}") for child in common.BSPS[bsp]]
    shift = total - frame["mean"].to_numpy().astype(np.float64)
    drop = [c for c in frame.columns if c.startswith(("mean_", "actual_", "mask_"))]
    return (
        frame.with_columns(switching_mask=pl.col("switching_mask") | pl.any_horizontal(*masks))
        .drop(drop)
        .with_columns(
            pl.Series("mean", total.astype(np.float32)),
            *[(pl.col(q) + pl.Series(shift)).cast(pl.Float32) for q in common.QUANTILES],
        )
        .select(list(OUTPUT_COLUMNS))
        .filter(pl.col("mean").is_finite())
    )


def main() -> None:
    """Build every arm at every BSP and print the weights used."""
    cache = common.load_cache()
    report = {
        bsp: {v: common.weights_for(cache=cache, bsp=bsp, variant=v) for v in common.VARIANT_SUFFIX}
        for bsp in common.BSPS
    }
    print(json.dumps(report, indent=2))
    for bsp in common.BSPS:
        for arm in ARMS:
            for period in SCORED_PERIODS:
                frame = build(bsp=bsp, period=period, arm=arm, cache=cache)
                if frame is None:
                    print(bsp, arm, period, "skipped: an input file is missing", flush=True)
                    continue
                target = out_path(arm=arm, period=period, bsp=bsp)
                temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
                frame.write_parquet(temporary)
                temporary.rename(target)
                print(bsp, arm, period, frame.height, flush=True)


if __name__ == "__main__":
    main()
