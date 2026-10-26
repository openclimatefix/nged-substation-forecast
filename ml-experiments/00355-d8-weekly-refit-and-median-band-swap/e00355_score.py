# ruff: noqa: E501
"""Score D8's three refit cadences and the median/band swaps against BG2.

Frames: `M`, `W`, `Q` are the monthly, weekly, and quarterly fits of experiment 354's fit `b`, stage-2
arm D, with arm 3's own-residual band by hour block and season (experiment 354's best stack). `D` is
`M` with the donated spread. The swaps use arm `M`: `S1` is M's median with BG2's spread (BG2
quantile minus BG2 median), `S2` is BG2's median with M's own-residual spread, `S3` is BG2's median
with the donated spread. A swapped frame's `mean` column is its median.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8:ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history:ml-experiments/00355-d8-weekly-refit-and-median-band-swap \
        uv run --no-sync python ml-experiments/00355-*/e00355_score.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from typing import Final

import e00252_common as common252  # ty: ignore[unresolved-import]
import e00252_score as score_module  # ty: ignore[unresolved-import]
import e00354_bands as bands  # ty: ignore[unresolved-import]
import e00354_lib as lib  # ty: ignore[unresolved-import]
import e00354_step as step  # ty: ignore[unresolved-import]
import e00355_fit as fit_module
import numpy as np
import polars as pl

SITE: Final[str] = "D8"
OUT_FILE: Final = fit_module.WORK / "scores.json"
GATE_M: Final[float] = 8.382
GATE_BG2: Final[float] = 7.593
CADENCE_FRAMES: Final[dict[str, str]] = {"M": "monthly", "W": "weekly", "Q": "quarterly"}
SLICES: Final[tuple[str, ...]] = ("all", "sunny", "daylight", "cold_night", "first3", "y2026")


def swapped(
    *, base: pl.DataFrame, median_from: pl.DataFrame, spread_from: pl.DataFrame
) -> pl.DataFrame:
    """Return `base`'s rows with `median_from`'s median and `spread_from`'s offsets from its median."""
    median = median_from["q50"].to_numpy().astype(np.float64)
    out = base.select(*lib.KEY, "run", "lead_hours", "actual")
    columns = [
        pl.Series(
            name,
            (median + spread_from[name].to_numpy() - spread_from["q50"].to_numpy()).astype(
                np.float32
            ),
        )
        for name in lib.QUANTILES
    ]
    return out.with_columns(*columns, pl.Series("mean", median.astype(np.float32)))


def build_frames() -> dict[str, pl.DataFrame]:
    """Return every arm's frame on the shared rows."""
    frames: dict[str, pl.DataFrame] = {}
    for name, cadence in CADENCE_FRAMES.items():
        raw = lib.clean(
            frame=pl.read_parquet(
                fit_module.folder_of(site=SITE, cadence=cadence) / f"d_scored_{SITE}.parquet"
            )
        )
        frames[name] = lib.cell_residual_band(frame=raw)
        if name == "M":
            frames["D"] = raw
    own = frames["M"]
    bg2 = lib.bg2_prior_frame(frame=own, site=SITE).sort(*lib.KEY)
    own_sorted = own.sort(*lib.KEY).join(bg2.select(*lib.KEY), on=lib.KEY, how="inner")
    bg2 = bg2.join(own_sorted.select(*lib.KEY), on=lib.KEY, how="inner").sort(*lib.KEY)
    donated = frames["D"].sort(*lib.KEY).join(bg2.select(*lib.KEY), on=lib.KEY, how="inner")
    frames["S1"] = swapped(base=own_sorted, median_from=own_sorted, spread_from=bg2)
    frames["S2"] = swapped(base=own_sorted, median_from=bg2, spread_from=own_sorted)
    frames["S3"] = swapped(base=own_sorted, median_from=bg2, spread_from=donated)
    return frames


def main() -> None:
    """Score every arm and write `work/scores.json`."""
    step.install()
    cache = common252.load_cache()
    frames = build_frames()
    wide = bands.wide_frame(site=SITE, cache=cache, frames=frames)
    capacity_mw = float(cache.sites[SITE].table.effective_capacity_mw)
    names = list(frames)
    references = list(score_module.REFERENCES)
    out: dict = {"rows": wide.height, "capacity_mw": capacity_mw}
    out["absolute"] = {
        flag: {n: float(wide.filter(pl.col(flag))[f"s_{n}"].mean()) for n in (*names, *references)}
        for flag in SLICES
    }
    out["monthly"] = {
        r["month"]: {n: r[f"s_{n}"] for n in (*names, *references)}
        for r in wide.group_by("month")
        .agg(*[pl.col(f"s_{n}").mean() for n in (*names, *references)])
        .sort("month")
        .to_dicts()
    }
    pairs = [
        ("W", "M"),
        ("Q", "M"),
        ("W", "Q"),
        ("S1", "M"),
        ("S2", "M"),
        ("S3", "M"),
        ("S2", "S3"),
        ("D", "M"),
    ]
    wide = wide.with_columns(
        *[(pl.col(f"s_{x}") - pl.col(f"s_{y}")).alias(f"c_{x}__{y}") for x, y in pairs],
        *[(pl.col(f"s_{n}") - pl.col("s_bg2")).alias(f"g_{n}") for n in names],
    )
    out["contrast"] = {
        column: {
            flag: lib.contrast_both(wide=wide, column=column, flag=flag, site=SITE)
            for flag in SLICES
        }
        for column in [c for c in wide.columns if c.startswith(("c_", "g_"))]
    }
    out["tails_all"] = {n: lib.coverage_and_tails(frame=wide, prefix=f"{n}__") for n in names}
    out["tails_all"]["bg2"] = lib.coverage_and_tails(frame=wide, prefix="bg2_")
    out["coverage_by_quarter"] = {
        n: {
            q: lib.coverage_and_tails(
                frame=wide.filter(pl.col("month").is_in(list(m))), prefix=f"{n}__"
            )["inside_q05_q95"]
            for q, m in bands.QUARTERS.items()
        }
        for n in names
    }
    scale = capacity_mw / 100.0
    out["checks"] = {
        "median_minus_mean_points": {
            n: float((wide[f"{n}__q50"] - wide[f"{n}__mean"]).mean()) for n in names
        },
        "mae_median_mw": {
            n: float((wide["actual"] - wide[f"{n}__q50"]).abs().mean()) * scale for n in names
        }
        | {"bg2": float((wide["actual"] - wide["bg2_q50"]).abs().mean()) * scale},
        "mae_mean_forecast_mw": {
            n: float((wide["actual"] - wide[f"{n}__mean"]).abs().mean()) * scale for n in names
        },
        "mae_median_mw_sunny": {
            n: float(
                (wide.filter(pl.col("sunny"))["actual"] - wide.filter(pl.col("sunny"))[f"{n}__q50"])
                .abs()
                .mean()
            )
            * scale
            for n in names
        },
    }
    out["gate"] = {
        "M": out["absolute"]["all"]["M"],
        "M_expected": GATE_M,
        "bg2": out["absolute"]["all"]["bg2"],
        "bg2_expected": GATE_BG2,
    }
    wide.write_parquet(fit_module.WORK / "scores_wide.parquet")
    OUT_FILE.write_text(json.dumps(out, indent=2, default=float))
    print("gate", out["gate"], "rows", wide.height)
    for flag in SLICES:
        print(flag, {n: round(v, 3) for n, v in out["absolute"][flag].items()})
    for column in ("c_W__M", "c_Q__M", "c_S1__M", "c_S2__M", "c_S3__M"):
        for flag in ("all", "first3", "y2026", "sunny"):
            print(column, flag, [round(v, 3) for v in out["contrast"][column][flag]["ci99"]])
    print("mae", {k: round(v, 3) for k, v in out["checks"]["mae_median_mw"].items()})
    print("tails", {k: round(v["inside_q05_q95"], 3) for k, v in out["tails_all"].items()})


if __name__ == "__main__":
    main()
