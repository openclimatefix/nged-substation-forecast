"""Fix 1: score D8's own-residual bands against the donated spread, with placebo Primaries.

Arms at D8 (the scored year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours, stored forecasts):

| Arm | Built from |
|---|---|
| `arm1` | experiment 252's stage-2 arm d, with the spread donated by six other Primaries |
| `arm2` | arm 1's median plus the as-of 365-day quantiles of D8's own residuals |
| `arm3` | arm 2 by hour block and season cell |
| `arm4` | arm 2 shrunk towards BG2's own as-of residual quantiles |
| `a1`, `a2` | experiment 252's stage-2 arm a (no corrections), with the donated spread, and with arm 2's band |

The placebo Primaries D2, D3, and D4 get arms 1 to 3 from their stored arm-d forecasts.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8 \
        uv run --no-sync python ml-experiments/00354-*/e00354_bands.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from typing import Final

import e00252_common as common252
import e00252_score as score_module
import e00354_lib as lib
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00354-d8-own-residual-bands-and-longer-fit-history" / "work"
STORED: Final = common252.ARM_DIR / "dp_preds_s2"
SITE_D8: Final[str] = "D8"
PLACEBOS: Final[tuple[str, ...]] = ("D2", "D3", "D4")
D8_ARMS: Final[tuple[str, ...]] = ("arm1", "arm2", "arm3", "arm4", "a1", "a2")
PLACEBO_ARMS: Final[tuple[str, ...]] = ("arm1", "arm2", "arm3")
QUARTERS: Final[dict[str, tuple[str, ...]]] = {
    "Q1": ("2025-07", "2025-08", "2025-09"),
    "Q2": ("2025-10", "2025-11", "2025-12"),
    "Q3": ("2026-01", "2026-02", "2026-03"),
    "Q4": ("2026-04", "2026-05", "2026-06"),
}


def read_stored(*, arm: str, site: str) -> pl.DataFrame:
    """Return the stored scored-year forecasts of a stage-2 arm at a site, cleaned."""
    return lib.clean(frame=pl.read_parquet(STORED / f"{arm}_scored_{site}.parquet"))


def build_frames(*, site: str) -> dict[str, pl.DataFrame]:
    """Return every arm's frame at a site."""
    arm1 = read_stored(arm="d", site=site)
    frames = {
        "arm1": arm1,
        "arm2": lib.own_residual_band(frame=arm1),
        "arm3": lib.cell_residual_band(frame=arm1),
    }
    if site == "D8":
        frames["arm4"] = lib.shrunk_band(
            frame=arm1, prior=lib.bg2_prior_frame(frame=arm1, site=site)
        )
        a1 = read_stored(arm="a", site=site)
        frames["a1"] = a1
        frames["a2"] = lib.own_residual_band(frame=a1)
    return frames


def wide_frame(*, site: str, cache, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:  # noqa: ANN001
    """Return the shared rows with per-arm quantile columns, references, scores, and flags."""
    base = next(iter(frames.values())).select(*lib.KEY, "lead_hours", "actual", "run")
    for arm, frame in frames.items():
        renamed = frame.select(
            *lib.KEY,
            pl.col("mean").alias(f"{arm}__mean"),
            *[pl.col(q).alias(f"{arm}__{q}") for q in lib.QUANTILES],
        )
        base = base.join(renamed, on=lib.KEY, how="inner", validate="1:1")
    for name in score_module.REFERENCES:
        base = base.join(
            score_module.reference_frame(name=name, site=site),
            on=lib.KEY,
            how="inner",
            validate="1:1",
        )
    base = lib.with_flags(frame=base, cache=cache, site=site)
    actual = base["actual"].to_numpy().astype(np.float64)
    scores = [
        pl.Series(f"s_{arm}", lib.row_scores(actual=actual, frame=base, prefix=f"{arm}__"))
        for arm in frames
    ] + [
        pl.Series(f"s_{n}", lib.row_scores(actual=actual, frame=base, prefix=f"{n}_"))
        for n in score_module.REFERENCES
    ]
    return base.with_columns(*scores).filter(
        pl.col("s_climatology").is_finite() & pl.col("s_manual").is_finite()
    )


def summarise(*, wide: pl.DataFrame, arms: tuple[str, ...], site: str, capacity_mw: float) -> dict:
    """Return absolute scores, contrasts, monthly closure, coverage, and checks for one site."""
    slices = ("all", "sunny", "daylight", "cold_night", "winter_night", "first3", "y2026")
    out: dict = {"rows": wide.height, "capacity_mw": capacity_mw}
    names = (*arms, *score_module.REFERENCES)
    out["absolute"] = {
        flag: {n: float(wide.filter(pl.col(flag))[f"s_{n}"].mean()) for n in names}
        for flag in slices
    }
    monthly = wide.group_by("month").agg(*[pl.col(f"s_{n}").mean() for n in names]).sort("month")
    out["monthly"] = {r["month"]: {n: r[f"s_{n}"] for n in names} for r in monthly.to_dicts()}
    pairs = [(a, "arm1") for a in arms if a != "arm1"] + [("arm3", "arm2")]
    if "arm4" in arms:
        pairs.append(("arm4", "arm2"))
    if "a2" in arms:
        pairs.append(("a2", "arm2"))
    wide = wide.with_columns(
        *[(pl.col(f"s_{a}") - pl.col(f"s_{b}")).alias(f"c_{a}_{b}") for a, b in pairs],
        *[(pl.col(f"s_{a}") - pl.col("s_bg2")).alias(f"g_{a}") for a in arms],
    )
    out["contrast"] = {
        column: {
            flag: lib.contrast_both(wide=wide, column=column, flag=flag, site=site)
            for flag in ("all", "sunny", "cold_night", "first3", "y2026")
        }
        for column in [c for c in wide.columns if c.startswith(("c_", "g_"))]
    }
    out["coverage_by_quarter"] = {
        arm: {
            qname: lib.coverage_and_tails(
                frame=wide.filter(pl.col("month").is_in(list(months))), prefix=f"{arm}__"
            )
            for qname, months in QUARTERS.items()
        }
        for arm in arms
    }
    out["tails_all"] = {arm: lib.coverage_and_tails(frame=wide, prefix=f"{arm}__") for arm in arms}
    out["tails_all"]["bg2"] = lib.coverage_and_tails(frame=wide, prefix="bg2_")
    out["checks"] = {
        "median_minus_mean_points": {
            arm: float((wide[f"{arm}__q50"] - wide[f"{arm}__mean"]).mean()) for arm in arms
        },
        "mae_median_mw": {
            arm: float((wide["actual"] - wide[f"{arm}__q50"]).abs().mean()) * capacity_mw / 100.0
            for arm in arms
        }
        | {"bg2": float((wide["actual"] - wide["bg2_q50"]).abs().mean()) * capacity_mw / 100.0},
        "mae_mean_forecast_mw": {
            arm: float((wide["actual"] - wide[f"{arm}__mean"]).abs().mean()) * capacity_mw / 100.0
            for arm in arms
        },
    }
    gap = {
        month: values["arm1"] - values["bg2"] for month, values in out["monthly"].items()
    }
    out["closure_by_month"] = {
        month: {
            "gap_arm1_minus_bg2": gap[month],
            "closed_share_arm2": (values["arm1"] - values["arm2"]) / gap[month]
            if abs(gap[month]) > 1e-9
            else None,
            "arm2_minus_bg2": values["arm2"] - values["bg2"],
        }
        for month, values in out["monthly"].items()
    }
    return out


def main() -> None:
    """Score D8 and the placebos and write `work/bands_scores.json`."""
    cache = common252.load_cache()
    results: dict = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for site in ("D8", *PLACEBOS):
        frames = build_frames(site=site)
        wide = wide_frame(site=site, cache=cache, frames=frames)
        wide.write_parquet(OUT_DIR / f"bands_wide_{site}.parquet")
        arms = D8_ARMS if site == "D8" else PLACEBO_ARMS
        results[site] = summarise(
            wide=wide,
            arms=arms,
            site=site,
            capacity_mw=float(cache.sites[site].table.effective_capacity_mw),
        )
        print(site, wide.height, {k: round(v, 3) for k, v in results[site]["absolute"]["all"].items()})
    gate = results["D8"]["absolute"]["all"]
    print("gate arm1", round(gate["arm1"], 3), "bg2", round(gate["bg2"], 3))
    (OUT_DIR / "bands_scores.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
