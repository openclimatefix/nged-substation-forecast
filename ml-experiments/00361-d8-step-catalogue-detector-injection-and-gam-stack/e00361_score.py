# ruff: noqa: E501
"""Score the three fit arms (control, t1, t0) under three band rules and print every number.

Frames are named `<arm>_<band>`, stage `d` throughout (night offset and recipe R, as experiment
354's primary), band `donated`, `arm2`, or `arm3`. Arm 3 (own residuals by hour block and season)
is the band of the pre-registered contrasts. Gate: `ctl_arm3` must score 8.382 on all rows.

    PYTHONPATH=<the folders of experiments 252, 264, 354, and 361> \
        OMP_NUM_THREADS=2 uv run --no-sync python -W ignore ml-experiments/00361-*/e00361_score.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import e00252_common as common252  # ty: ignore[unresolved-import]
import e00252_score as score_module  # ty: ignore[unresolved-import]
import e00354_bands as bands  # ty: ignore[unresolved-import]
import e00354_lib as lib  # ty: ignore[unresolved-import]
import e00361_fit as fit_module
import polars as pl

BANDS = ("donated", "arm2", "arm3")


def main() -> None:
    """Score every frame and write `work/scores.json`."""
    import e00354_step as step  # ty: ignore[unresolved-import]

    step.install()
    cache = common252.load_cache()
    frames: dict[str, pl.DataFrame] = {}
    for arm in fit_module.ARMS:
        raw = lib.clean(frame=pl.read_parquet(fit_module.arm_dir(arm=arm) / "d_scored_D8.parquet"))
        frames |= {
            f"{arm}_donated": raw,
            f"{arm}_arm2": lib.own_residual_band(frame=raw),
            f"{arm}_arm3": lib.cell_residual_band(frame=raw),
        }
    wide = bands.wide_frame(site=bands.SITE_D8, cache=cache, frames=frames)
    capacity_mw = float(cache.sites[bands.SITE_D8].table.effective_capacity_mw)
    names = list(frames)
    refs = (*score_module.REFERENCES,)
    out: dict = {"rows": wide.height, "capacity_mw": capacity_mw}
    slices = ("all", "sunny", "daylight", "cold_night", "winter_night", "first3", "y2026")
    out["absolute"] = {
        flag: {n: float(wide.filter(pl.col(flag))[f"s_{n}"].mean()) for n in (*names, *refs)}
        for flag in slices
    }
    out["monthly"] = {
        r["month"]: {n: r[f"s_{n}"] for n in (*names, *refs)}
        for r in wide.group_by("month")
        .agg(*[pl.col(f"s_{n}").mean() for n in (*names, *refs)])
        .sort("month")
        .to_dicts()
    }
    pairs = [(f"{a}_{b}", f"ctl_{b}") for a in ("t1", "t0") for b in BANDS] + [
        ("t1_arm3", "t0_arm3")
    ]
    wide = wide.with_columns(
        *[(pl.col(f"s_{x}") - pl.col(f"s_{y}")).alias(f"c_{x}__{y}") for x, y in pairs],
        *[(pl.col(f"s_{n}") - pl.col("s_bg2")).alias(f"g_{n}") for n in names],
    )
    out["contrast"] = {
        column: {
            flag: lib.contrast_both(wide=wide, column=column, flag=flag, site=bands.SITE_D8)
            for flag in ("all", "sunny", "cold_night", "winter_night", "y2026")
        }
        for column in [c for c in wide.columns if c.startswith("c_")]
    }
    out["tails_all"] = {n: lib.coverage_and_tails(frame=wide, prefix=f"{n}__") for n in names}
    out["coverage_by_quarter"] = {
        n: {
            q: lib.coverage_and_tails(
                frame=wide.filter(pl.col("month").is_in(list(months))), prefix=f"{n}__"
            )["inside_q05_q95"]
            for q, months in bands.QUARTERS.items()
        }
        for n in names
    }
    out["checks"] = {
        "median_minus_mean_points": {
            n: float((wide[f"{n}__q50"] - wide[f"{n}__mean"]).mean()) for n in names
        },
        "mae_median_mw": {
            n: float((wide["actual"] - wide[f"{n}__q50"]).abs().mean()) * capacity_mw / 100.0
            for n in names
        },
        "mae_mean_forecast_mw": {
            n: float((wide["actual"] - wide[f"{n}__mean"]).abs().mean()) * capacity_mw / 100.0
            for n in names
        },
    }
    wide.write_parquet(fit_module.WORK / "scores_wide.parquet")
    (fit_module.WORK / "scores.json").write_text(json.dumps(out, indent=2, default=float))
    print("rows", out["rows"])
    for flag in slices:
        print(
            flag,
            {
                n: round(v, 3)
                for n, v in out["absolute"][flag].items()
                if n.endswith("arm3") or n in refs
            },
        )
    for column, by_flag in out["contrast"].items():
        for flag in ("all", "y2026"):
            print("contrast", column, flag, [round(v, 3) for v in by_flag[flag]["ci99"]])
    print(
        "tails",
        {
            n: {k: round(v, 3) for k, v in t.items()}
            for n, t in out["tails_all"].items()
            if n.endswith("arm3")
        },
    )
    print(
        "median-mean",
        {
            n: round(v, 3)
            for n, v in out["checks"]["median_minus_mean_points"].items()
            if n.endswith("arm3")
        },
    )
    print(
        "mae_mean_mw",
        {
            n: round(v, 3)
            for n, v in out["checks"]["mae_mean_forecast_mw"].items()
            if n.endswith("arm3")
        },
    )
    print(
        "mae_median_mw",
        {n: round(v, 3) for n, v in out["checks"]["mae_median_mw"].items() if n.endswith("arm3")},
    )


if __name__ == "__main__":
    main()
