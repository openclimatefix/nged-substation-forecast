"""Build the bottom-up forecasts of each BSP from arm D of the BSP's remainder and its children.

For a BSP `b` with covered children `i` and weights `w_i`, the observed reading decomposes exactly
into the remainder series and the weighted children:

    y_b = r + sum_i w_i * (cap_i / cap_b) * y_i

(readings in points of each series' own effective capacity). The bottom-up mean forecast is the
same sum of arm D means. Quantiles are the mean plus a spread table fitted on the bottom-up mean's
own tuning-month residuals. The oracle replaces the children's forecasts by their observed
readings. `r_tdspread` gives the bottom-up mean the top-down spread table's offsets row by row.

    uv run --no-sync python ml-experiments/00161-*/e00161_bu.py
"""

import json
import os
from pathlib import Path
from typing import Final

import e00161_common as common
import e00161_corr as corr
import numpy as np
import polars as pl

KEYS: Final[list[str]] = ["power_fcst_init_time", "valid_time"]
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *common.QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip
VARIANTS: Final[tuple[str, ...]] = ("r", "s", "f", "r_tdspread", "oracle")


def d_path(*, period: str, site: str) -> str:
    """Return the file of arm D for one period at a chain series."""
    return str(common.OUT_DIR / f"d_{period}_{site}.parquet")


def child_path(*, period: str, site: str) -> str:
    """Return the file of experiment 155's `d_spread_a` for one period at a Primary."""
    return str(common.E155_PREDS_DIR / f"d_spread_a_{period}_{site}.parquet")


def out_path(*, name: str, period: str, bsp: str) -> str:
    """Return the file of a bottom-up variant for one period at a BSP."""
    return str(common.OUT_DIR / f"bu_{name}_{period}_{bsp}.parquet")


def td_path(*, period: str, bsp: str) -> str:
    """Return the file of the top-down arm (arm D at the BSP) for one period."""
    return str(common.OUT_DIR / f"td_{period}_{bsp}.parquet")


def joined_period(*, bsp: str, suffix: str, period: str) -> pl.DataFrame:
    """Join the BSP, one remainder, and the covered children on run and valid time.

    Args:
        bsp: The BSP's label.
        suffix: The remainder's label suffix (`r`, `s`, or `f`).
        period: `tuning`, `replicate`, or `scored`.

    Returns:
        The BSP's arm D frame with `mean_r`, `actual_r`, `mean_<child>`, `actual_<child>`, and a
        switching mask that is true if the BSP or any child is inside a switching event.
    """
    frame = pl.read_parquet(d_path(period=period, site=bsp)).select(list(OUTPUT_COLUMNS))
    remainder = pl.read_parquet(d_path(period=period, site=f"{bsp}{suffix}")).select(
        *KEYS, mean_r=pl.col("mean"), actual_r=pl.col("actual")
    )
    frame = frame.join(remainder, on=KEYS, how="inner", validate="1:1")
    for child in common.BSPS[bsp]:
        part = pl.read_parquet(child_path(period=period, site=child)).select(
            *KEYS,
            **{
                f"mean_{child}": pl.col("mean"),
                f"actual_{child}": pl.col("actual"),
                f"mask_{child}": pl.col("switching_mask"),
            },
        )
        frame = frame.join(part, on=KEYS, how="inner", validate="1:1")
    masks = [pl.col(f"mask_{child}") for child in common.BSPS[bsp]]
    return frame.with_columns(
        switching_mask=pl.col("switching_mask") | pl.any_horizontal(*masks)
    ).drop([f"mask_{child}" for child in common.BSPS[bsp]])


def bottom_up_mean(
    *,
    frame: pl.DataFrame,
    bsp: str,
    weights: dict[str, float],
    cache_scale: dict[str, float],
    oracle: bool,
) -> np.ndarray:
    """Return the remainder's mean plus the weighted children, in the BSP's points."""
    total = frame["mean_r"].to_numpy().astype(np.float64)
    column = "actual" if oracle else "mean"
    for child, weight in weights.items():
        total = total + weight * cache_scale[child] * frame[f"{column}_{child}"].to_numpy().astype(
            np.float64
        )
    return total


def with_period(*, frame: pl.DataFrame, period: str) -> pl.DataFrame:
    """Add the period column that the spread refit reads."""
    return frame.with_columns(period=pl.lit(period))


def build_bsp(*, bsp: str, cache) -> dict[str, int]:  # noqa: ANN001
    """Write the top-down and every bottom-up variant at a BSP; return the row counts."""
    cap = cache.sites[bsp].table.effective_capacity_mw
    scale = {c: cache.sites[c].table.effective_capacity_mw / cap for c in common.BSPS[bsp]}
    counts: dict[str, int] = {}
    built: dict[str, list[pl.DataFrame]] = {name: [] for name in ("td", *VARIANTS)}
    for variant, suffix in common.VARIANT_SUFFIX.items():
        weights = common.weights_for(cache=cache, bsp=bsp, variant=variant)
        frames = {
            period: with_period(
                frame=joined_period(bsp=bsp, suffix=suffix, period=period), period=period
            )
            for period in common.PERIODS
        }
        full = pl.concat(frames.values()).sort(KEYS)
        name = suffix
        for oracle in (False, True) if variant == "learned" else (False,):
            mean = bottom_up_mean(
                frame=full, bsp=bsp, weights=weights, cache_scale=scale, oracle=oracle
            )
            base = full.select(*OUTPUT_COLUMNS, "period")
            arm = corr.apply_shift_and_refit_spread(
                frame=base, shift=mean - base["mean"].to_numpy().astype(np.float64)
            )
            built["oracle" if oracle else name].append(arm)
            if variant == "learned" and not oracle:
                shift = mean - base["mean"].to_numpy().astype(np.float64)
                tdspread = base.with_columns(
                    pl.Series("mean", (base["mean"].to_numpy() + shift).astype(np.float32)),
                    *[(pl.col(q) + pl.Series(shift)).cast(pl.Float32) for q in common.QUANTILES],
                )
                built["r_tdspread"].append(tdspread)
                if not built["td"]:
                    built["td"].append(base)
    for name, pieces in built.items():
        if not pieces:
            continue
        frame = pieces[0]
        for period in common.PERIODS:
            part = frame.filter(pl.col("period") == period).select(list(OUTPUT_COLUMNS))
            target = (
                td_path(period=period, bsp=bsp)
                if name == "td"
                else out_path(name=name, period=period, bsp=bsp)
            )
            temporary = f"{target}.{os.getpid()}.tmp"
            part.write_parquet(temporary)
            Path(temporary).replace(target)
            counts[f"{name}_{period}"] = part.height
    return counts


def main() -> None:
    """Build every bottom-up variant at every BSP and print the weights used."""
    cache = common.load_cache()
    weights = {
        bsp: {v: common.weights_for(cache=cache, bsp=bsp, variant=v) for v in common.VARIANT_SUFFIX}
        for bsp in common.BSPS
    }
    print(json.dumps(weights, indent=2))
    for bsp in common.BSPS:
        print(bsp, build_bsp(bsp=bsp, cache=cache), flush=True)


if __name__ == "__main__":
    main()
