# ruff: noqa: E501
"""Score quantile blends of the GAM stack and BG2 at D8, with D2, D3, and D4 as placebos.

The GAM stack at D8 is experiment 355's weekly-refit arm `W` (own-residual band by hour block and
season). At the placebo Primaries it is experiment 354's arm 1 (the stored stack with its donated
spread). A blend with weight `w` has each of the 11 quantiles equal to `w` times the stack's plus
`1 - w` times BG2's. The weight is chosen on the first half of the scored year (valid months
2025-07 to 2025-12) and reported on the second half (2026-01 to 2026-06) and the whole year.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8:ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history \
        uv run --no-sync python ml-experiments/00359-*/e00359_blend.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from typing import Final

import e00354_lib as lib  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00359-d8-gam-bg2-quantile-blend" / "work"
D355: Final = ML_EXPERIMENTS_DATA_DIR / "00355-d8-weekly-refit-and-median-band-swap" / "work"
D354: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00354-d8-own-residual-bands-and-longer-fit-history" / "work"
)
QUANTILES: Final = lib.QUANTILES
WEIGHTS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
FIRST_HALF: Final[tuple[str, ...]] = (
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
)
GATE_GAM: Final[float] = 8.338
GATE_BG2: Final[float] = 7.593
PLACEBOS: Final[tuple[str, ...]] = ("D2", "D3", "D4")
SLICES: Final[tuple[str, ...]] = (
    "all", "h1", "h2", "lead_0_24", "lead_24_72", "lead_72_168", "sunny_amj", "sunny", "cold_night",
)  # fmt: skip


def _mean(series: pl.Series) -> float:
    """Return the mean of a numeric series as a float."""
    return float(series.mean())  # ty: ignore[invalid-argument-type]


def weight_name(weight: float) -> str:
    """Return the arm name of a blend weight, such as `b50`."""
    return f"b{round(weight * 100):02d}"


def build(*, stack: pl.DataFrame, stack_prefix: str) -> pl.DataFrame:
    """Return the rows with the stack, BG2, and blend quantile columns, their scores, and slice flags."""
    actual = stack["actual"].to_numpy().astype(np.float64)
    gam = {q: stack[f"{stack_prefix}{q}"].to_numpy().astype(np.float64) for q in QUANTILES}
    bg2 = {q: stack[f"bg2_{q}"].to_numpy().astype(np.float64) for q in QUANTILES}
    columns = [pl.Series(f"gam__{q}", gam[q]) for q in QUANTILES]
    for weight in WEIGHTS:
        columns += [
            pl.Series(f"{weight_name(weight)}__{q}", weight * gam[q] + (1 - weight) * bg2[q])
            for q in QUANTILES
        ]
    out = stack.with_columns(*columns)
    scored = [
        pl.Series(f"s_{name}", lib.row_scores(actual=actual, frame=out, prefix=f"{name}__"))
        for name in ("gam", *map(weight_name, WEIGHTS))
    ]
    month = out["month"]
    lead = out["lead_hours"]
    return out.with_columns(
        *scored,
        h1=month.is_in(list(FIRST_HALF)),
        h2=~month.is_in(list(FIRST_HALF)),
        lead_0_24=lead <= 24,
        lead_24_72=(lead > 24) & (lead <= 72),
        lead_72_168=lead > 72,
        sunny_amj=out["sunny"] & month.is_in(["2026-04", "2026-05", "2026-06"]),
    )


def summarise(*, wide: pl.DataFrame, site: str) -> dict:
    """Return absolute scores, contrasts, monthly scores, coverage, and checks for one site."""
    arms = ["gam", *map(weight_name, WEIGHTS)]
    names = [*arms, "bg2", "climatology", "manual"]
    out: dict = {"rows": wide.height}
    out["absolute"] = {
        flag: {n: _mean(wide.filter(pl.col(flag))[f"s_{n}"]) for n in names} for flag in SLICES
    }
    first_half = out["absolute"]["h1"]
    chosen = min(map(weight_name, WEIGHTS), key=lambda name: first_half[name])
    out["chosen_on_first_half"] = chosen
    pairs = list(
        dict.fromkeys(
            [(a, "bg2") for a in (*map(weight_name, WEIGHTS), "gam")]
            + [("b50", "gam"), (chosen, "gam")]
        )
    )
    wide = wide.with_columns(
        *[(pl.col(f"s_{x}") - pl.col(f"s_{y}")).alias(f"c_{x}__{y}") for x, y in pairs]
    )
    out["contrast"] = {
        f"c_{x}__{y}": {
            flag: lib.contrast_both(wide=wide, column=f"c_{x}__{y}", flag=flag, site=site)
            for flag in ("all", "h1", "h2", "lead_0_24", "lead_24_72", "lead_72_168", "sunny_amj")
        }
        for x, y in pairs
    }
    out["monthly"] = {
        r["month"]: {n: r[f"s_{n}"] for n in names}
        for r in wide.group_by("month")
        .agg(*[pl.col(f"s_{n}").mean() for n in names])
        .sort("month")
        .to_dicts()
    }
    out["tails_all"] = {n: lib.coverage_and_tails(frame=wide, prefix=f"{n}__") for n in arms}
    out["tails_all"]["bg2"] = lib.coverage_and_tails(frame=wide, prefix="bg2_")
    out["checks"] = {
        "median_minus_mean_forecast_note": "blend rows have no mean column; the median error stands in",
        "mae_median_mw_points": {n: _mean((wide["actual"] - wide[f"{n}__q50"]).abs()) for n in arms}
        | {"bg2": _mean((wide["actual"] - wide["bg2_q50"]).abs())},
        "gam_median_minus_mean_points": _mean(wide["gam__q50"] - wide["W__mean"])
        if "W__mean" in wide.columns
        else _mean(wide["gam__q50"] - wide["arm1__mean"]),
        "bg2_median_minus_gam_median_points": _mean(wide["bg2_q50"] - wide["gam__q50"]),
        "corr_median_errors": float(
            np.corrcoef(
                (wide["actual"] - wide["gam__q50"]).to_numpy(),
                (wide["actual"] - wide["bg2_q50"]).to_numpy(),
            )[0, 1]
        ),
    }
    return out


def main() -> None:
    """Score D8 and the placebos and write `work/blend_scores.json`."""
    WORK.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    d8 = pl.read_parquet(D355 / "scores_wide.parquet")
    wide = build(stack=d8, stack_prefix="W__")
    gate = {
        "gam": _mean(wide["s_gam"]),
        "gam_expected": GATE_GAM,
        "bg2": _mean(wide["s_bg2"]),
        "bg2_expected": GATE_BG2,
        "rows": wide.height,
    }
    print("gate", gate)
    if abs(gate["gam"] - GATE_GAM) > 0.0015 or abs(gate["bg2"] - GATE_BG2) > 0.0015:
        raise SystemExit("reproduction gate failed")
    results["D8"] = summarise(wide=wide, site="D8")
    results["gate"] = gate
    wide.write_parquet(WORK / "blend_wide_D8.parquet")
    for site in PLACEBOS:
        placebo = build(
            stack=pl.read_parquet(D354 / f"bands_wide_{site}.parquet"), stack_prefix="arm1__"
        )
        results[site] = summarise(wide=placebo, site=site)
    (WORK / "blend_scores.json").write_text(json.dumps(results, indent=2, default=float))
    for site in ("D8", *PLACEBOS):
        r = results[site]
        print(site, r["rows"], "chosen", r["chosen_on_first_half"])
        for flag in ("all", "h1", "h2"):
            print("  ", flag, {k: round(v, 3) for k, v in r["absolute"][flag].items()})


if __name__ == "__main__":
    main()
