"""Score arms C, M, S (three penalties) and S+ at D8, and C and S at the placebo Primaries.

Every frame is stage `d` (night offset and recipe R) with experiment 354's arm-3 band (own
residuals by UTC six-hour block and season). Arm M is experiment 361's masking arm, read from its
stored forecasts. Arm S+ widens arm S (middle penalty) in the upper half of the distribution on
rows whose latest stretch at the fit's cut is a detected fall: the quantiles at q50 and above come
from the mixture `(1 - p) F(x) + p F(x - d)`, with p the share of training rows inside detected
falls and d their median depth in points. Prints every number the write-up quotes and writes
`work/scores.json` and `work/scores_wide_D8.parquet`.

    PYTHONPATH=$E252:$E264:$E354:$E361:$E363 \
        OMP_NUM_THREADS=2 uv run --no-sync python -W ignore ml-experiments/00363-*/e00363_score.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import pickle
from datetime import timedelta
from typing import Final

import e00252_common as common252  # ty: ignore[unresolved-import]
import e00252_score as score_module  # ty: ignore[unresolved-import]
import e00354_bands as bands  # ty: ignore[unresolved-import]
import e00354_lib as lib  # ty: ignore[unresolved-import]
import e00354_step as step  # ty: ignore[unresolved-import]
import e00361_detect as detect  # ty: ignore[unresolved-import]
import e00363_fit as fit_module
import e00363_segment as segment
import numpy as np
import polars as pl
from studies.dp_assimilation import DEFAULT_DELAY_HOURS

E361_WORK: Final = fit_module.WORK.parent.parent / (
    "00361-d8-step-catalogue-detector-injection-and-gam-stack/work"
)
S_ARMS: Final[tuple[str, ...]] = ("seg_weak", "seg_mid", "seg_strong")
LEADS: Final[dict[str, tuple[float, float]]] = {
    "lead_le12": (0.0, 12.0),
    "lead_12_36": (12.0, 36.0),
    "lead_36_72": (36.0, 72.0),
    "lead_gt72": (72.0, 1e9),
}


def read_arm(*, site: str, arm: str) -> pl.DataFrame:
    """Return an arm's stage-d scored frame with arm 3's band."""
    frame = pl.read_parquet(fit_module.arm_dir(site=site, arm=arm) / f"d_scored_{site}.parquet")
    return lib.cell_residual_band(frame=lib.clean(frame=frame.filter(pl.col("period") == "scored")))


def widen_upper(*, frame: pl.DataFrame, site: str) -> pl.DataFrame:
    """Return arm S+: the mixture quantiles at q50 and above on rows whose latest stretch is a fall."""
    fits = pickle.loads((fit_module.arm_dir(site=site, arm="seg_mid") / "tall_fits.pkl").read_bytes())
    cuts = np.array([np.datetime64(f.cut.replace(tzinfo=None), "us") for f in fits])
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    delay = np.timedelta64(int(timedelta(hours=DEFAULT_DELAY_HOURS).total_seconds()), "s")
    idx = np.clip(np.searchsorted(cuts + delay, origin, side="right") - 1, 0, len(fits) - 1)
    in_fall = np.array([fits[i].arm.last_in_fall for i in idx])
    share = np.array([fits[i].arm.fall_share for i in idx])
    depth = np.array([fits[i].arm.fall_depth_points for i in idx])
    names = list(lib.QUANTILES)
    levels = np.array([float(n[1:]) / 100.0 for n in names])
    q = frame.select(names).to_numpy().astype(np.float64)
    out = q.copy()
    upper = [i for i, level in enumerate(levels) if level >= 0.5]
    for row in np.flatnonzero(in_fall & (share > 0) & (depth > 0)):
        grid = np.linspace(q[row, 0] - 1.0, q[row, -1] + depth[row] + 1.0, 400)
        cdf = np.interp(grid, q[row], levels, left=0.0, right=1.0)
        shifted = np.interp(grid - depth[row], q[row], levels, left=0.0, right=1.0)
        mixture = (1 - share[row]) * cdf + share[row] * shifted
        out[row, upper] = np.interp(levels[upper], mixture, grid)
    return frame.with_columns(
        [pl.Series(name, out[:, i].astype(np.float32)) for i, name in enumerate(names)]
    ).with_columns(widened=pl.Series(in_fall & (share > 0)))


def day_flags(*, wide: pl.DataFrame) -> pl.DataFrame:
    """Add lead-band flags and a flag for days that hold a hindsight-detected step or drop."""
    cache = common252.load_cache()
    features = cache.sites["D8"]
    power = (features.y * float(features.table.effective_capacity_mw) / 100.0).astype(np.float64)
    flag = np.zeros(len(power), dtype=bool)
    for seg in (detect.detect(power_mw=power), detect.detect_drops(power_mw=power)):
        for s, e in zip(seg.start, seg.end, strict=True):
            flag[s:e] = True
    days = pl.DataFrame(
        {"valid_time": cache.end_times.astype("datetime64[us]"), "flag": flag}
    ).with_columns(day=pl.col("valid_time").dt.date())
    step_days = days.group_by("day").agg(step_day=pl.col("flag").any())
    wide = wide.with_columns(day=pl.col("valid_time").dt.date()).join(
        step_days, on="day", how="left", validate="m:1"
    )
    wide = wide.with_columns(step_day=pl.col("step_day").fill_null(False)).with_columns(
        no_step_day=~pl.col("step_day")
    )
    return wide.with_columns(
        *[
            ((pl.col("lead_hours") > lo) & (pl.col("lead_hours") <= hi)).alias(name)
            for name, (lo, hi) in LEADS.items()
        ]
    )


def main() -> None:
    """Score every frame and write `work/scores.json`."""
    step.install()
    segment.install()
    cache = common252.load_cache()
    frames: dict[str, pl.DataFrame] = {"ctl": read_arm(site="D8", arm="ctl")}
    masked = pl.read_parquet(E361_WORK / "fits_t0" / "d_scored_D8.parquet")
    frames["m"] = lib.cell_residual_band(frame=lib.clean(frame=masked))
    for arm in S_ARMS:
        frames[arm] = read_arm(site="D8", arm=arm)
    frames["seg_plus"] = widen_upper(frame=frames["seg_mid"], site="D8")
    wide = bands.wide_frame(site=bands.SITE_D8, cache=cache, frames=frames)
    wide = day_flags(wide=wide)
    capacity_mw = float(cache.sites[bands.SITE_D8].table.effective_capacity_mw)
    names, refs = list(frames), (*score_module.REFERENCES,)
    out: dict = {"rows": wide.height, "capacity_mw": capacity_mw}
    slices = ("all", "sunny", "cold_night", "winter_night", "y2026", "step_day", "no_step_day", *LEADS)
    out["absolute"] = {
        flag: {n: float(wide.filter(pl.col(flag))[f"s_{n}"].mean()) for n in (*names, *refs)}
        for flag in slices
    }
    out["rows_by_slice"] = {flag: int(wide.filter(pl.col(flag)).height) for flag in slices}
    pairs = [(a, "ctl") for a in (*S_ARMS, "m", "seg_plus")] + [
        ("seg_mid", "m"),
        ("seg_plus", "seg_mid"),
    ]
    wide = wide.with_columns(
        *[(pl.col(f"s_{x}") - pl.col(f"s_{y}")).alias(f"c_{x}__{y}") for x, y in pairs],
        *[(pl.col(f"s_{n}") - pl.col("s_bg2")).alias(f"g_{n}") for n in names],
    )
    out["contrast"] = {
        column: {
            flag: lib.contrast_both(wide=wide, column=column, flag=flag, site=bands.SITE_D8)
            for flag in ("all", "sunny", "cold_night", "y2026", "step_day", "no_step_day", *LEADS)
        }
        for column in [c for c in wide.columns if c.startswith("c_")]
    }
    out["tails_all"] = {n: lib.coverage_and_tails(frame=wide, prefix=f"{n}__") for n in names}
    actual = wide["actual"].to_numpy()
    out["checks"] = {
        "median_minus_mean_points": {
            n: float((wide[f"{n}__q50"] - wide[f"{n}__mean"]).mean()) for n in names
        },
        "mae_median_mw": {
            n: float(np.abs(actual - wide[f"{n}__q50"].to_numpy()).mean()) * capacity_mw / 100.0
            for n in names
        },
        "mae_mean_forecast_mw": {
            n: float(np.abs(actual - wide[f"{n}__mean"].to_numpy()).mean()) * capacity_mw / 100.0
            for n in names
        },
        "bias_mean_forecast_mw": {
            n: float((wide[f"{n}__mean"].to_numpy() - actual).mean()) * capacity_mw / 100.0
            for n in names
        },
        "correlation_mean_forecast": {
            n: float(np.corrcoef(actual, wide[f"{n}__mean"].to_numpy())[0, 1]) for n in names
        },
    }
    out["share_widened_rows"] = float(frames["seg_plus"]["widened"].mean())
    wide.write_parquet(fit_module.WORK / "scores_wide_D8.parquet")
    out["placebo"] = {}
    for site in ("D2", "D4"):
        pframes = {arm: read_arm(site=site, arm=arm) for arm in ("ctl", "seg_mid")}
        pwide = bands.wide_frame(site=site, cache=cache, frames=pframes)
        pwide = pwide.with_columns((pl.col("s_seg_mid") - pl.col("s_ctl")).alias("c_seg_mid__ctl"))
        out["placebo"][site] = {
            "rows": pwide.height,
            "ctl": float(pwide["s_ctl"].mean()),
            "seg_mid": float(pwide["s_seg_mid"].mean()),
            "contrast": lib.contrast_both(wide=pwide, column="c_seg_mid__ctl", flag="all", site=site),
            "tails": {a: lib.coverage_and_tails(frame=pwide, prefix=f"{a}__") for a in pframes},
        }
    (fit_module.WORK / "scores.json").write_text(json.dumps(out, indent=2, default=float))
    print("rows", out["rows"], "widened share", round(out["share_widened_rows"], 3))
    for flag in slices:
        print(flag, out["rows_by_slice"][flag], {n: round(v, 3) for n, v in out["absolute"][flag].items()})
    for column, by_flag in out["contrast"].items():
        for flag, v in by_flag.items():
            print("contrast", column, flag, [round(x, 3) for x in v["ci99"]])
    for key in ("median_minus_mean_points", "mae_median_mw", "mae_mean_forecast_mw", "bias_mean_forecast_mw", "correlation_mean_forecast"):
        print(key, {n: round(v, 3) for n, v in out["checks"][key].items()})
    print("tails", {n: {k: round(v, 3) for k, v in t.items()} for n, t in out["tails_all"].items()})
    for site, v in out["placebo"].items():
        print("placebo", site, v["rows"], round(v["ctl"], 3), round(v["seg_mid"], 3), [round(x, 3) for x in v["contrast"]["ci99"]], v["tails"])


if __name__ == "__main__":
    main()
