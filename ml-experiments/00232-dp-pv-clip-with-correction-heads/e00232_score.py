"""Rescore experiment 178's sunny-daylight cut with experiment 201's stored correction heads.

**No fitting.** Arm C is experiment 180's control (arm D), arm K is experiment 180's 60% clip, arm H
is arm C plus a correction head of experiment 201, and arm KH is arm K plus the same head. A head
is read from its stored forecasts: its median shift is `delta`, and its spread is the offsets of
each stored quantile from its stored median. The head's seeds are averaged. Arm H reproduces the
stored head forecast, because arm D's median plus `delta` is the stored median. Arm KH puts the
head's shift and spread on arm K's median. Experiment 200's head stored only per-month scores, so
it cannot be rescored here.

Writes `scores.json` under the experiment's private data folder.

    uv run --no-sync python ml-experiments/00232-*/e00232_score.py
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_month_totals
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00232-dp-pv-clip-with-correction-heads"
E180: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00180-dp-sunny-daylight-fix-d5-solar-shape" / "work" / "arms"
)
E201: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00201-dp-gam-residual-mlp" / "work" / "preds"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
BG2_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
SITES: Final[tuple[str, ...]] = ("D5", "D10", "D16")
HEAD_ARMS: Final[tuple[int, ...]] = (1, 2, 3)
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SUN_UP: Final[float] = 0.01
HORIZON_HOURS: Final[float] = 168.0
INTERVAL_LEVEL: Final[float] = 98.75
KEY: Final[list[str]] = ["power_fcst_init_time", "valid_time"]


def pinball(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's pinball loss averaged over the 11 quantiles."""
    error = actual[:, None] - quantiles
    return np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)


def read_arm(*, arm: str, site: str, capacity: float) -> pl.DataFrame:
    """Return experiment 180's scored-year forecasts for an arm at a site, in MW."""
    path = E180 / arm / "dp_preds_s2" / f"d_scored_{site}.parquet"
    frame = pl.read_parquet(path).filter(
        ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= HORIZON_HOURS)
    )
    return frame.select(
        *KEY,
        "actual",
        *QUANTILES,  # points of capacity
    ).with_columns([pl.col(c) * capacity / 100.0 for c in ("actual", *QUANTILES)])


def read_head(*, head_arm: int, site: str, capacity: float) -> pl.DataFrame:
    """Return a head's seed-averaged median shift and quantile offsets from its median, in MW."""
    frames = [
        pl.read_parquet(E201 / f"arm{head_arm}_seed{seed}.parquet").filter(pl.col("site") == site)
        for seed in SEEDS
    ]
    parts = [
        f.select(
            *KEY,
            "delta",
            **{f"o_{q}": pl.col(q) - pl.col("q50") for q in QUANTILES},
        )
        for f in frames
    ]
    stacked = pl.concat(parts)
    mean = stacked.group_by(KEY).agg([pl.col(c).mean() for c in stacked.columns if c not in KEY])
    return mean.with_columns([pl.col(c) * capacity / 100.0 for c in mean.columns if c not in KEY])


def with_head(*, base: pl.DataFrame, head: pl.DataFrame) -> pl.DataFrame:
    """Return `base` with the head's median shift and spread applied."""
    joined = base.join(head, on=KEY, how="inner", validate="1:1")
    median = pl.col("q50") + pl.col("delta")
    return joined.with_columns(**{q: median + pl.col(f"o_{q}") for q in QUANTILES}).select(
        *KEY, "actual", *QUANTILES
    )


def add_losses(*, frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Return the frame reduced to its keys plus the arm's absolute error and pinball loss."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
    return frame.select(
        *KEY,
        **{
            f"abs_{name}": (pl.col("actual") - pl.col("q50")).abs(),
            f"pin_{name}": pl.Series(pinball(actual=actual, quantiles=quantiles)),
        },
    )


def interval(*, frame: pl.DataFrame, column: str) -> tuple[float, float, float]:
    """Return the mean of a column and its month-block interval at `INTERVAL_LEVEL`."""
    monthly = frame.group_by("month").agg(total=pl.col(column).sum(), count=pl.len()).sort("month")
    sums = monthly["total"].to_numpy()[None, :]
    counts = monthly["count"].to_numpy().astype(np.float64)
    low, high = bootstrap_month_totals(month_sums=sums, month_counts=counts, level=INTERVAL_LEVEL)
    return float(sums.sum() / counts.sum()), float(low), float(high)


def sunny_flags(*, cache: FeatureCache, site: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the fleet-shape `unit` and the month, as experiment 180's scorer does."""
    index = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    features = cache.sites[site]
    return frame.with_columns(
        unit=pl.Series(features.pv_unit_shape[index]),
        month=frame["valid_time"].dt.convert_time_zone("Europe/London").dt.strftime("%Y-%m"),
    )


def tail_table(
    *, part: pl.DataFrame, arms: dict[str, pl.DataFrame], names: tuple[str, ...]
) -> dict:
    """Return each named arm's coverage and pinball loss at q01, q05, q95, and q99 on `part`."""
    table: dict = {}
    for name in names:
        rows = part.select(KEY).join(arms[name], on=KEY, validate="1:1")
        actual = rows["actual"].to_numpy().astype(np.float64)
        cell = {}
        for q, level in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99)):
            value = rows[q].to_numpy().astype(np.float64)
            error = actual - value
            cell[q] = {
                "coverage": float(np.mean(actual <= value)),
                "pinball": float(np.mean(np.maximum(level * error, (level - 1.0) * error))),
            }
        table[name] = cell
    return table


def overlap(*, part: pl.DataFrame, arms: dict[str, pl.DataFrame], head_arm: int) -> dict:
    """Return the mean median shifts of the clip and of a head on `part`, and their correlation."""
    keys = part.select(KEY)
    c = keys.join(arms["C"], on=KEY, validate="1:1")["q50"].to_numpy().astype(np.float64)
    k = keys.join(arms["K"], on=KEY, validate="1:1")["q50"].to_numpy().astype(np.float64)
    h = keys.join(arms[f"H{head_arm}"], on=KEY, validate="1:1")["q50"].to_numpy().astype(np.float64)
    clip_shift, head_shift = k - c, h - c
    return {
        "mean_clip_shift": float(clip_shift.mean()),
        "mean_head_shift": float(head_shift.mean()),
        "correlation": float(np.corrcoef(clip_shift, head_shift)[0, 1]),
    }


def main() -> None:
    """Score every arm at D5, D10, and D16 and write the tables."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    bg2_all = pl.read_parquet(BG2_PATH)
    out: dict = {}
    for site in SITES:
        capacity = float(bg2_all.filter(pl.col("site") == site)["capacity_mw"][0])
        control = read_arm(arm="c", site=site, capacity=capacity)
        clip = read_arm(arm="k60", site=site, capacity=capacity)
        bg2 = bg2_all.filter(pl.col("site") == site).select(
            *KEY, **{f"b_{q}": pl.col(q) * capacity for q in QUANTILES}
        )
        arms: dict[str, pl.DataFrame] = {"C": control, "K": clip}
        for head_arm in HEAD_ARMS:
            head = read_head(head_arm=head_arm, site=site, capacity=capacity)
            arms[f"H{head_arm}"] = with_head(base=control, head=head)
            arms[f"KH{head_arm}"] = with_head(base=clip, head=head)
        rows = control.select(KEY).join(bg2.select(KEY), on=KEY, how="inner")
        for frame in arms.values():
            rows = rows.join(frame.select(KEY), on=KEY, how="inner")
        if rows.height != control.height:
            msg = f"{site}: {rows.height} shared rows, the control has {control.height}"
            raise ValueError(msg)
        losses = rows
        for name, frame in arms.items():
            losses = losses.join(add_losses(frame=frame, name=name), on=KEY, validate="1:1")
        bg2_rows = rows.join(bg2, on=KEY, validate="1:1").join(
            control.select(*KEY, "actual"), on=KEY, validate="1:1"
        )
        b = bg2_rows.select([f"b_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
        bg2_losses = bg2_rows.select(
            *KEY,
            abs_BG2=(pl.col("actual") - pl.col("b_q50")).abs(),
            pin_BG2=pl.Series(
                pinball(actual=bg2_rows["actual"].to_numpy().astype(np.float64), quantiles=b)
            ),
        )
        losses = sunny_flags(
            cache=cache, site=site, frame=losses.join(bg2_losses, on=KEY, validate="1:1")
        )
        day = losses.filter(pl.col("unit") > SUN_UP)["unit"].to_numpy()
        edge = float(np.quantile(day, 2 / 3))
        for cut, part in (("sunny", losses.filter(pl.col("unit") >= edge)), ("all", losses)):
            cell: dict = {"rows": part.height}
            for name in (*arms, "BG2"):
                cell[f"mae_{name}"] = float(part[f"abs_{name}"].to_numpy().mean())
                cell[f"pin_{name}"] = float(part[f"pin_{name}"].to_numpy().mean())
            for name in arms:
                diff = part.with_columns(
                    d=pl.col(f"abs_{name}") - pl.col("abs_C"),
                    dk=pl.col(f"abs_{name}") - pl.col("abs_K"),
                    gap=pl.col(f"abs_{name}") - pl.col("abs_BG2"),
                    dp=pl.col(f"pin_{name}") - pl.col("pin_C"),
                )
                cell[f"{name}_minus_C"] = interval(frame=diff, column="d")
                cell[f"{name}_minus_K"] = interval(frame=diff, column="dk")
                cell[f"{name}_gap_to_BG2"] = interval(frame=diff, column="gap")
                cell[f"{name}_pin_minus_C"] = interval(frame=diff, column="dp")
            for head_arm in HEAD_ARMS:
                add = part.with_columns(
                    a=pl.col(f"abs_KH{head_arm}")
                    - pl.col("abs_K")
                    - pl.col(f"abs_H{head_arm}")
                    + pl.col("abs_C")
                )
                cell[f"additivity_H{head_arm}"] = interval(frame=add, column="a")
            cell["tails"] = tail_table(part=part, arms=arms, names=("C", "K", "H2", "KH2"))
            cell["overlap"] = overlap(part=part, arms=arms, head_arm=2)
            out[f"{site}|{cut}"] = cell
            print(site, cut, part.height)
            for name in arms:
                print(
                    f"  {name}: mae {cell[f'mae_{name}']:.3f}  gap to BG2 "
                    f"{cell[f'{name}_gap_to_BG2'][0]:+.3f}  minus C "
                    f"{cell[f'{name}_minus_C'][0]:+.3f} "
                    f"({cell[f'{name}_minus_C'][1]:+.3f} to {cell[f'{name}_minus_C'][2]:+.3f})"
                )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "scores.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
