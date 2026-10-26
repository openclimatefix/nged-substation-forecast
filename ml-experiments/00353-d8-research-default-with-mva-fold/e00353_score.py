"""Score experiment 353's arms at D8 and write every number the write-up quotes.

Arms (all at D8, the scored year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours):

| Arm | Built from |
|---|---|
| `a` | experiment 252's stage-2 arm d (the card's forecast), as stored |
| `b` | experiment 152's arm A (meter fit, symmetric quantiles) plus stage-2 arm d |
| `b_fold` | experiment 152's arm F_fold (quantiles folded at zero) plus the stage-2 shift |
| `c` | experiment 152's arm N (reading as signed demand) plus stage-2 arm d |
| `raw_n`, `raw_a`, `raw_f` | experiment 152's arms N, A, and F_fold with no stage 2 |
| `b_off`, `b_rr` | arm B with only the night offset, or only recipe R (post hoc) |
| `b_causal` | arm B with each month's spread fitted on D8's own earlier months only (post hoc) |
| `b_own` | arm B with its spread refitted on its own scored residuals (post hoc, in-sample) |

The stage-2 code is experiment 252's, imported unchanged: it needs experiment 252's source and arm
in the environment, which this script sets before importing. Intervals come from experiment 252's
paired bootstrap that resamples whole valid-time months.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries \
        uv run --no-sync python ml-experiments/00353-*/e00353_score.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00252_common as common  # ty: ignore[unresolved-import]
import e00252_score as score_module  # ty: ignore[unresolved-import]
import e00252s2_common as s2common  # ty: ignore[unresolved-import]
import e00252s2_corrections as corr  # ty: ignore[unresolved-import]
import e00252s2_run as stage2  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_calendar import HOLIDAY_PERIODS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITE: Final[str] = "D8"
QUANTILES: Final[tuple[str, ...]] = score_module.QUANTILES
KEY: Final[list[str]] = score_module.KEY
STORED_152: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries"
    / "work"
    / "preds"
    / "real"
    / f"scored_{SITE}.parquet"
)
OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00353-d8-research-default-with-mva-fold" / "work"
ARMS: Final[tuple[str, ...]] = (
    "a", "b", "b_fold", "c", "raw_n", "raw_a", "raw_f", "b_off", "b_rr", "b_own", "b_causal",
)  # fmt: skip
SLICES: Final[tuple[str, ...]] = (
    *score_module.SLICES, "high_block", "low_block", "daylight", "first_half", "second_half",
)  # fmt: skip
HIGH_BLOCK_MW: Final[float] = 4.5
"""The midpoint of the 1 to 3 MW and 6 to 8 MW levels visible on D8's card."""
SECOND_HALF_START: Final = datetime(2026, 1, 1, tzinfo=UTC)
"""Post hoc split: the stage-2 corrections have a winter of history only in the second half."""


def read_152(*, arm: str) -> pl.DataFrame:
    """Return experiment 152's stored scored-year forecasts of one arm at D8."""
    return (
        pl.read_parquet(STORED_152)
        .filter(pl.col("arm") == arm)
        .select([*s2common.BASE_COLUMNS, "flow"])
        .with_columns(period=pl.lit("scored"))
        .sort(["power_fcst_init_time", "valid_time"])
    )


def causal_own_band(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return `frame` with each month's spread fitted on D8's own earlier months only.

    A forecast run belongs to the month of its origin. The spread of a month comes from the rows
    whose valid time is before the month's first instant (the rows whose outcome was known), so
    nothing in the band looks ahead. The first month keeps the donor spread, because D8 has no
    earlier row.
    """
    month = frame["power_fcst_init_time"].dt.truncate("1mo")
    pieces = []
    for start in month.unique().sort().to_list():
        part = month == start
        prior = (frame["valid_time"] < start).to_numpy()
        if not prior.any():
            pieces.append(frame.filter(part))
            continue
        flagged = frame.with_columns(
            period=pl.when(pl.Series(prior)).then(pl.lit("tuning")).otherwise(pl.lit("scored"))
        )
        refitted = corr.apply_shift_and_refit_spread(frame=flagged, shift=np.zeros(frame.height))
        pieces.append(refitted.filter(part))
    return pl.concat(pieces).sort(["power_fcst_init_time", "valid_time"])


def build_arms() -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return every arm's frame, and arm A's frame with the fitted flow."""
    stored = {name: read_152(arm=arm) for name, arm in (("n", "N"), ("a", "A"), ("f", "F_fold"))}
    built = {
        name: stage2.build_arms(site=SITE, frame=corr.add_calendar(frame=frame.drop("flow")))
        for name, frame in stored.items()
    }
    columns = [*s2common.BASE_COLUMNS]
    frames = {
        "a": pl.read_parquet(s2common.out_path(arm="d", period="scored", site=SITE)),
        "b": built["a"]["d"],
        "b_fold": built["f"]["d_spread_a"],
        "c": built["n"]["d"],
        "b_off": built["a"]["b"],
        "b_rr": built["a"]["c"],
        "b_causal": causal_own_band(frame=built["a"]["d"]),
        "b_own": corr.apply_shift_and_refit_spread(
            frame=built["a"]["d"].with_columns(period=pl.lit("tuning")),
            shift=np.zeros(built["a"]["d"].height),
        ),
        "raw_n": stored["n"],
        "raw_a": stored["a"],
        "raw_f": stored["f"],
    }
    return {k: v.select(columns) for k, v in frames.items()}, stored["a"]


def with_flags(*, frame: pl.DataFrame, cache) -> pl.DataFrame:  # noqa: ANN001
    """Add experiment 252's slice flags and the two block flags, from the series alone."""
    features = cache.sites[SITE]
    index = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    unit = features.pv_unit_shape[index]
    local_month = cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % 12 + 1
    holiday = cache.calendar.holiday_period[index].astype(np.int64)
    daylight = unit > score_module.SUN_UP
    edge = float(np.quantile(unit[daylight], 2 / 3))
    winter = np.isin(local_month, [11, 12, 1, 2, 3])
    night = features.elevation_deg[index] < -5
    christmas = [HOLIDAY_PERIODS.index(n) for n in ("Christmas Day", "Boxing Day")]
    megawatts = features.y * features.table.effective_capacity_mw / 100.0
    rolling = (
        pl.DataFrame({"valid_time": cache.end_times, "mw": megawatts})
        .with_columns(pl.col("mw").rolling_median(window_size=48, center=True, min_samples=24))
        .filter(
            pl.col("valid_time").is_in(frame["valid_time"].dt.replace_time_zone(None).implode())
        )
    )
    state = dict(zip(rolling["valid_time"].to_list(), rolling["mw"].to_list(), strict=True))
    median_mw = np.array(
        [state.get(t, np.nan) for t in frame["valid_time"].dt.replace_time_zone(None).to_list()]
    )
    return frame.with_columns(
        month=frame["valid_time"].dt.convert_time_zone("Europe/London").dt.strftime("%Y-%m"),
        all=pl.lit(True),
        sunny=pl.Series(unit >= edge),
        daylight=pl.Series(daylight),
        winter=pl.Series(winter),
        winter_night=pl.Series(winter & night),
        cold_night=pl.Series(night & (features.temperature_c[index] < 2.0)),
        christmas=pl.Series(np.isin(holiday, christmas)),
        bank_holiday=pl.Series(holiday >= 0),
        first_half=pl.col("valid_time") < pl.lit(SECOND_HALF_START),
        second_half=pl.col("valid_time") >= pl.lit(SECOND_HALF_START),
        high_block=pl.Series(median_mw >= HIGH_BLOCK_MW),
        low_block=pl.Series(median_mw < HIGH_BLOCK_MW),
    )


def wide_frame(*, frames: dict[str, pl.DataFrame], flow: pl.DataFrame, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return the rows every arm and every reference share, with per-arm columns and flags."""
    base = (
        frames["a"]
        .filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= score_module.HORIZON_HOURS)
        )
        .select([*KEY, "lead_hours", "actual", "run"])
    )
    for arm, frame in frames.items():
        renamed = frame.select(
            *KEY,
            pl.col("mean").alias(f"{arm}__mean"),
            *[pl.col(q).alias(f"{arm}__{q}") for q in QUANTILES],
        )
        base = base.join(renamed, on=KEY, how="inner", validate="1:1")
    for name in score_module.REFERENCES:
        base = base.join(
            score_module.reference_frame(name=name, site=SITE), on=KEY, how="inner", validate="1:1"
        )
    base = base.join(flow.select(*KEY, "flow"), on=KEY, how="left", validate="1:1")
    return with_flags(frame=base, cache=cache)


def add_scores(*, wide: pl.DataFrame) -> pl.DataFrame:
    """Add each arm's and reference's per-row quantile score and mean-forecast error."""
    actual = wide["actual"].to_numpy().astype(np.float64)
    columns = []
    for arm in (*ARMS, *score_module.REFERENCES):
        names = (
            [f"{arm}__{q}" for q in QUANTILES] if arm in ARMS else [f"{arm}_{q}" for q in QUANTILES]
        )
        columns.append(
            pl.Series(
                f"s_{arm}",
                score_module.score(
                    actual=actual, quantiles=wide.select(names).to_numpy().astype(np.float64)
                ),
            )
        )
    wide = wide.with_columns(*columns)
    return wide.with_columns(
        **{f"e_{arm}": (pl.col("actual") - pl.col(f"{arm}__mean")).abs() for arm in ARMS},
        c1=pl.col("s_b") - pl.col("s_c"),
        c1_fold=pl.col("s_b_fold") - pl.col("s_c"),
        c2=pl.col("s_b") - pl.col("s_a"),
        c4=pl.col("s_b_own") - pl.col("s_b"),
        c6=pl.col("s_b_causal") - pl.col("s_b"),
        c7=pl.col("s_b_causal") - pl.col("s_bg2"),
        c5=pl.col("s_b_own") - pl.col("s_bg2"),
        c3=pl.col("s_b") - pl.col("s_bg2"),
        a_bg2=pl.col("s_a") - pl.col("s_bg2"),
        c_bg2=pl.col("s_c") - pl.col("s_bg2"),
    )


def tails(*, wide: pl.DataFrame, flag: str) -> dict:
    """Return each arm's share of readings at or below q01, q05, q95, and q99 over a slice."""
    part = wide.filter(pl.col(flag))
    actual = part["actual"].to_numpy()
    return {
        arm: {
            q: float((actual <= part[f"{arm}__{q}"].to_numpy()).mean())
            for q in ("q01", "q05", "q95", "q99")
        }
        for arm in ARMS
    } | {
        "bg2": {
            q: float((actual <= part[f"bg2_{q}"].to_numpy()).mean())
            for q in ("q01", "q05", "q95", "q99")
        }
    }


def main() -> None:
    """Score everything and write `work/scores.json`."""
    cache = common.load_cache()
    frames, flow = build_arms()
    wide = add_scores(wide=wide_frame(frames=frames, flow=flow, cache=cache))
    shared = wide.filter(pl.col("s_climatology").is_finite() & pl.col("s_manual").is_finite())
    out: dict = {
        "effective_capacity_mw": cache.sites[SITE].table.effective_capacity_mw,
        "rows": wide.height,
        "rows_shared_with_references": shared.height,
    }
    out["absolute"] = {
        flag: {
            name: float(shared.filter(pl.col(flag))[f"s_{name}"].to_numpy().mean())
            for name in (*ARMS, *score_module.REFERENCES)
        }
        | {"rows": shared.filter(pl.col(flag)).height}
        for flag in SLICES
    }
    out["contrast"] = {
        column: {
            flag: score_module.contrast(frames={SITE: wide}, column=column, flag=flag)
            for flag in SLICES
        }
        for column in ("c1", "c1_fold", "c2", "c3", "c4", "c5", "c6", "c7", "a_bg2", "c_bg2")
    }
    out["tails"] = {flag: tails(wide=wide, flag=flag) for flag in ("all", "sunny", "cold_night")}
    out["checks"] = {
        "median_minus_mean": {
            arm: float((wide[f"{arm}__q50"] - wide[f"{arm}__mean"]).to_numpy().mean())
            for arm in ARMS
        },
        "mae_mean_forecast": {arm: float(wide[f"e_{arm}"].to_numpy().mean()) for arm in ARMS},
        "mae_bg2_median": float((wide["actual"] - wide["bg2_q50"]).abs().to_numpy().mean()),
    }
    flows = wide["flow"].to_numpy()
    sunny = wide["sunny"].to_numpy()
    out["reverse_flow"] = {
        "share_half_hours": float((flows < 0).mean()),
        "share_sunny_half_hours": float((flows[sunny] < 0).mean()),
        "n_half_hours_valid_times": int(wide.filter(pl.col("flow") < 0)["valid_time"].n_unique()),
        "n_valid_times": int(wide["valid_time"].n_unique()),
        "min_flow_points": float(flows.min()),
        "by_lead_band": {
            band: float(
                (
                    flows[
                        (wide["lead_hours"].to_numpy() > lo) & (wide["lead_hours"].to_numpy() <= hi)
                    ]
                    < 0
                ).mean()
            )
            for band, (lo, hi) in {"0-24": (0, 24), "24-72": (24, 72), "72-168": (72, 168)}.items()
        },
        "negative_by_month": {
            k[0]: int(k[1])
            for k in wide.filter(pl.col("flow") < 0)
            .group_by("month")
            .agg(pl.col("valid_time").n_unique().alias("n"))
            .iter_rows()
        },
        "negative_by_utc_hour": {
            int(k[0]): int(k[1])
            for k in wide.filter(pl.col("flow") < 0)
            .group_by(pl.col("valid_time").dt.hour().alias("hour"))
            .agg(pl.col("valid_time").n_unique().alias("n"))
            .sort("hour")
            .iter_rows()
        },
        "share_actual_below_5_points": float((wide["actual"].to_numpy() < 5).mean()),
    }
    out["block_rows"] = {
        flag: int(wide.filter(pl.col(flag)).height) for flag in ("high_block", "low_block")
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.write_parquet(OUT_DIR / "wide_scored_rows.parquet")
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2, default=float))
    print_report(out=out)


def print_report(*, out: dict) -> None:
    """Print every number the write-up quotes."""
    print("rows", out["rows"], "shared", out["rows_shared_with_references"])
    for flag, values in out["absolute"].items():
        print("absolute", flag, {k: round(v, 3) for k, v in values.items()})
    for column, by_flag in out["contrast"].items():
        for flag, result in by_flag.items():
            point, low, high = result["pooled"]
            print(f"contrast {column} {flag}: {point:+.3f} ({low:+.3f} to {high:+.3f})")
    print("tails", json.dumps(out["tails"], default=float))
    print("checks", json.dumps(out["checks"], default=float))
    print("reverse flow", json.dumps(out["reverse_flow"], default=float))
    print("block rows", out["block_rows"])


if __name__ == "__main__":
    main()
