"""Experiment 247: score the learned-cap arms against the control and the stored clip arms.

Reads each arm's stage 2 forecasts at D5, D9, and the placebos D10 and D16, on experiment 178's
rows, and contrasts them with the unclipped control `c`. The 60% clip and the as-of choice are
experiment 214's stored scores. Writes `work/scores.json`, including the closure at D5 and D9:
the learned arm's sunny-daylight error reduction as a share of the 60% clip's.

    uv run --no-sync python ml-experiments/00247-*/e00247_asof.py
"""

import json
from typing import Final

import e00247_common as common
import e00247_score as score
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_month_totals
from studies.dp_features import FeatureCache

ARMS: Final[tuple[str, ...]] = ("p", "e", "sat")
SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D16")
LEVEL: Final[float] = 95.0
KEY: Final[list[str]] = ["power_fcst_init_time", "valid_time"]
TAILS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
TAIL_LEVEL: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
ASOF_PATH: Final = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00214-dp-pv-clip-ceiling-as-of-and-screen"
    / "work"
    / "asof.json"
)


def load(
    *, cache: FeatureCache, site: str, arm: str, period: str, bg2: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Return an arm's rows in MW at a site and period, with the unit shape and the abs error."""
    frame = score.arm_frame(cache=cache, arm=arm, site=site, period=period, bg2=bg2)
    return frame.with_columns(
        abs_err=(pl.col("actual") - pl.col("q50")).abs(),
        origin_month=pl.col("power_fcst_init_time")
        .dt.convert_time_zone("UTC")
        .dt.truncate("1mo")
        .dt.replace_time_zone("UTC"),
    )


def interval(*, frame: pl.DataFrame, column: str) -> list[float]:
    """Return the mean of a column and its month-block interval at `LEVEL`."""
    monthly = frame.group_by("month").agg(total=pl.col(column).sum(), count=pl.len()).sort("month")
    sums = monthly["total"].to_numpy()[None, :]
    counts = monthly["count"].to_numpy().astype(np.float64)
    low, high = bootstrap_month_totals(month_sums=sums, month_counts=counts, level=LEVEL)
    return [float(sums.sum() / counts.sum()), float(low), float(high)]


def closure(*, learned: pl.DataFrame, clip: pl.DataFrame) -> list[float]:
    """Return the learned arm's error reduction as a share of the clip's, with its interval.

    Both arms are scored against the control on the sunny rows; months are resampled as whole
    blocks, in the same draw for both arms.
    """
    key = [*KEY, "month"]
    both = (
        learned.filter(pl.col("sunny"))
        .select(*key, dl=pl.col("abs_err") - pl.col("c_abs"))
        .join(
            clip.filter(pl.col("sunny")).select(*key, dk=pl.col("abs_err") - pl.col("c_abs")),
            on=key,
            how="inner",
            validate="1:1",
        )
    )
    monthly = (
        both.group_by("month")
        .agg(dl=pl.col("dl").sum(), dk=pl.col("dk").sum(), n=pl.len())
        .sort("month")
    )
    dl, dk, n = (monthly[c].to_numpy().astype(np.float64) for c in ("dl", "dk", "n"))
    point = float(dl.sum() / dk.sum())
    rng = np.random.default_rng(247)
    picks = rng.integers(0, len(n), size=(20000, len(n)))
    ratios = dl[picks].sum(axis=1) / dk[picks].sum(axis=1)
    low, high = np.percentile(ratios, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return [point, float(low), float(high)]


def tail_row(*, frame: pl.DataFrame, suffix: str = "") -> dict:
    """Return coverage and pinball loss at each tail quantile on the rows of a frame."""
    out = {}
    actual = frame["actual"].to_numpy()
    for name in TAILS:
        q = frame[name].to_numpy()
        level = TAIL_LEVEL[name]
        error = actual - q
        out[name] = {
            "coverage": float(np.mean(actual <= q)),
            "pinball": float(np.mean(np.maximum(level * error, (level - 1.0) * error))),
        }
    return out


def contrast(
    *, arm: pl.DataFrame, control: pl.DataFrame, bg2: pl.DataFrame | None
) -> tuple[dict, pl.DataFrame]:
    """Return the sunny and all-rows scores of an arm against the control (same rows)."""
    arm_s = score.add_slices(frame=arm)
    control_s = score.add_slices(frame=control)
    if arm_s.height != control_s.height:
        msg = f"arm has {arm_s.height} rows, control {control_s.height}"
        raise ValueError(msg)
    joined = arm_s.join(
        control_s.select(
            *KEY,
            c_abs=pl.col("abs_err"),
            c_pin=pl.col("pin"),
            c_mean_abs=(pl.col("actual") - pl.col("mean")).abs(),
        ),
        on=KEY,
        how="inner",
        validate="1:1",
    ).with_columns(mean_abs=(pl.col("actual") - pl.col("mean")).abs())
    out: dict = {}
    for name in ("all", "sunny"):
        part = joined.filter(pl.col(name))
        diff = part.with_columns(
            d=pl.col("abs_err") - pl.col("c_abs"),
            dp=pl.col("pin") - pl.col("c_pin"),
            dm=pl.col("mean_abs") - pl.col("c_mean_abs"),
        )
        cell = {
            "rows": part.height,
            "mae": float(part["abs_err"].to_numpy().mean()),
            "control_mae": float(part["c_abs"].to_numpy().mean()),
            "pin": float(part["pin"].to_numpy().mean()),
            "control_pin": float(part["c_pin"].to_numpy().mean()),
            "mae_minus_control": interval(frame=diff, column="d"),
            "pin_minus_control": interval(frame=diff, column="dp"),
            "mean_abs_minus_control": interval(frame=diff, column="dm"),
            "median_minus_mean_mw": float((part["q50"] - part["mean"]).to_numpy().mean()),
            "tails": tail_row(frame=part),
            "control_tails": tail_row(frame=control_s.filter(pl.col(name))),
        }
        if "b_q50" in part.columns:
            gap = part.with_columns(
                g=pl.col("abs_err") - (pl.col("actual") - pl.col("b_q50")).abs()
            )
            cell["gap_to_bg2"] = interval(frame=gap, column="g")
            cell["bg2_mae"] = float((part["actual"] - part["b_q50"]).abs().to_numpy().mean())
        out[name] = cell
    return out, joined


def pooled(*, joined_by_site: dict[str, pl.DataFrame]) -> dict:
    """Return the pooled sunny contrast over sites, months totalled across sites."""
    parts = [
        j.filter(pl.col("sunny")).with_columns(d=pl.col("abs_err") - pl.col("c_abs"))
        for j in joined_by_site.values()
    ]
    stacked = pl.concat([p.select("month", "d") for p in parts])
    return {"sites": list(joined_by_site), "mae_minus_control": interval(frame=stacked, column="d")}


def main() -> None:
    """Write `work/scores.json`."""
    cache = common.load_cache()
    bg2 = pl.read_parquet(score.BG2_PATH)
    result: dict = {}
    for site in SITES:
        control = load(cache=cache, site=site, arm="c", period="scored", bg2=bg2)
        clip = load(cache=cache, site=site, arm="k60", period="scored", bg2=bg2)
        clip_stats, clip_joined = contrast(arm=clip, control=control, bg2=bg2)
        result[f"{site}|k60"] = clip_stats
        for arm in ARMS:
            frame = load(cache=cache, site=site, arm=arm, period="scored", bg2=bg2)
            stats, joined = contrast(arm=frame, control=control, bg2=bg2)
            stats["closure_sunny"] = closure(learned=joined, clip=clip_joined)
            result[f"{site}|{arm}"] = stats
            print(
                site, arm, stats["sunny"]["mae_minus_control"], stats["closure_sunny"], flush=True
            )
    stored = json.loads(ASOF_PATH.read_text())["scored"]
    result["as_of_214"] = {s: stored[s]["as_of"]["sunny"]["mae_minus_control"] for s in stored}
    common.write_json_atomic(path=common.WORK_DIR / "scores.json", content=result)


if __name__ == "__main__":
    main()
