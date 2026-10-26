# ruff: noqa: E501
"""Fix 3: score D8's three fit histories, with and without stage 2, under three band rules.

Frames are named `<fit>_<stage>_<band>`: fit `a` (from 2025-04), `b` (from 2023-09 with a level
step per hour band at 2025-04-01), or `c` (from 2023-09, no step); stage `d` (night offset and
recipe R, the card's stack) or `a` (no corrections); band `donated` (as stored), `arm2` (as-of
residual quantiles), or `arm3` (the same by hour block and season cell). Arm 3 is the band rule
that scored best in fix 1 on all rows, so it carries the pre-registered contrasts.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8:ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history \
        uv run --no-sync python ml-experiments/00354-*/e00354_score_fits.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import pickle
from typing import Final

import e00252_common as common252  # ty: ignore[unresolved-import]
import e00252_score as score_module  # ty: ignore[unresolved-import]
import e00354_bands as bands
import e00354_fit as fit_module
import e00354_lib as lib
import polars as pl

FITS: Final[tuple[str, ...]] = ("a", "b", "c")
STAGES: Final[tuple[str, ...]] = ("d", "a")
BAND_RULES: Final[tuple[str, ...]] = ("donated", "arm2", "arm3")
PRIMARY: Final[tuple[str, str]] = ("d", "arm3")
"""The stage and band rule of the pre-registered contrasts (the card's stack, fix 1's best band)."""
OUT_FILE: Final = fit_module.WORK / "fits_scores.json"


def frames_for(*, fit: str, stage: str) -> dict[str, pl.DataFrame]:
    """Return one fit and stage's frame under every band rule."""
    raw = lib.clean(
        frame=pl.read_parquet(fit_module.arm_dir(arm=fit) / f"{stage}_scored_D8.parquet")
    )
    return {
        f"{fit}_{stage}_donated": raw,
        f"{fit}_{stage}_arm2": lib.own_residual_band(frame=raw),
        f"{fit}_{stage}_arm3": lib.cell_residual_band(frame=raw),
    }


def step_levels(*, capacity_mw: float) -> dict:
    """Return arm b's fitted level step per hour band, in MW, at every cut."""
    fits = pickle.loads((fit_module.arm_dir(arm="b") / "tall_fits.pkl").read_bytes())
    return {
        fit.cut.strftime("%Y-%m-%d"): [float(x) * capacity_mw / 100.0 for x in fit.fit.theta[-4:]]
        for fit in fits
    }


def main() -> None:
    """Score every frame and write `work/fits_scores.json`."""
    import e00354_step as step

    step.install()
    cache = common252.load_cache()
    frames: dict[str, pl.DataFrame] = {}
    for fit in FITS:
        for stage in STAGES:
            frames |= frames_for(fit=fit, stage=stage)
    wide = bands.wide_frame(site=bands.SITE_D8, cache=cache, frames=frames)
    capacity_mw = float(cache.sites[bands.SITE_D8].table.effective_capacity_mw)
    names = list(frames)
    out: dict = {"rows": wide.height, "capacity_mw": capacity_mw}
    slices = ("all", "sunny", "daylight", "cold_night", "winter_night", "first3", "y2026")
    out["absolute"] = {
        flag: {
            n: float(wide.filter(pl.col(flag))[f"s_{n}"].mean())  # ty: ignore[invalid-argument-type]
            for n in (*names, *score_module.REFERENCES)
        }
        for flag in slices
    }
    monthly = (
        wide.group_by("month")
        .agg(*[pl.col(f"s_{n}").mean() for n in (*names, *score_module.REFERENCES)])
        .sort("month")
    )
    out["monthly"] = {
        r["month"]: {n: r[f"s_{n}"] for n in (*names, *score_module.REFERENCES)}
        for r in monthly.to_dicts()
    }
    pairs = []
    for stage in STAGES:
        for band in BAND_RULES:
            pairs += [
                (f"b_{stage}_{band}", f"a_{stage}_{band}"),
                (f"c_{stage}_{band}", f"a_{stage}_{band}"),
                (f"b_{stage}_{band}", f"c_{stage}_{band}"),
            ]
    wide = wide.with_columns(
        *[(pl.col(f"s_{x}") - pl.col(f"s_{y}")).alias(f"c_{x}__{y}") for x, y in pairs],
        *[(pl.col(f"s_{n}") - pl.col("s_bg2")).alias(f"g_{n}") for n in names],
    )
    out["contrast"] = {
        column: {
            flag: lib.contrast_both(wide=wide, column=column, flag=flag, site=bands.SITE_D8)
            for flag in ("all", "sunny", "first3", "y2026")
        }
        for column in [c for c in wide.columns if c.startswith(("c_", "g_"))]
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
            n: float((wide[f"{n}__q50"] - wide[f"{n}__mean"]).mean())  # ty: ignore[invalid-argument-type]
            for n in names
        },
        "mae_median_mw": {
            n: float((wide["actual"] - wide[f"{n}__q50"]).abs().mean()) * capacity_mw / 100.0  # ty: ignore[invalid-argument-type]
            for n in names
        },
        "mae_mean_forecast_mw": {
            n: float((wide["actual"] - wide[f"{n}__mean"]).abs().mean()) * capacity_mw / 100.0  # ty: ignore[invalid-argument-type]
            for n in names
        },
    }
    out["step_levels_mw_by_cut"] = step_levels(capacity_mw=capacity_mw)
    wide.write_parquet(fit_module.WORK / "fits_wide.parquet")
    OUT_FILE.write_text(json.dumps(out, indent=2, default=float))
    for flag in ("all", "first3", "y2026"):
        print(
            flag,
            {
                n: round(v, 3)
                for n, v in out["absolute"][flag].items()
                if n.endswith(("donated", "arm3")) or n in score_module.REFERENCES
            },
        )
    stage, band = PRIMARY
    for x, y in (
        (f"b_{stage}_{band}", f"a_{stage}_{band}"),
        (f"c_{stage}_{band}", f"a_{stage}_{band}"),
    ):
        for flag in ("all", "first3", "y2026"):
            print(
                "contrast",
                x,
                y,
                flag,
                [round(v, 2) for v in out["contrast"][f"c_{x}__{y}"][flag]["ci99"]],
            )


if __name__ == "__main__":
    main()
