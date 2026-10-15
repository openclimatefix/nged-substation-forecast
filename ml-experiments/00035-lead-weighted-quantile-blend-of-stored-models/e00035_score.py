"""Score the blend arms: absolute scores, planned contrasts with intervals, stop rule, verdicts.

Reads the arms `e00035_blend.py` wrote and writes tables under
`data/ml-experiments/00035-.../work/results/` that the write-up quotes. Scores are in points of
capacity (the stored quantiles are fractions of capacity, multiplied by 100).

    uv run python ml-experiments/00035-lead-weighted-quantile-blend-of-stored-models/e00035_score.py
"""

from pathlib import Path
from typing import Any, Final

import e00035_blend as blend
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

RESULTS_DIR: Final[Path] = blend.WORK_DIR / "results"
DECISIVE_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3)
N_RESAMPLES_DECISIVE: Final[int] = 20_000
N_RESAMPLES: Final[int] = 5_000
THRESHOLDS: Final[dict[str, float]] = {"Primary": 0.05, "solar": 0.05, "wind": 0.10}
DECISIVE: Final[dict[str, str]] = {"Primary": "C1", "solar": "C2", "wind": "C3"}
STOP_SHARE: Final[float] = 10 / 12
FAMILY_SIZES: Final[dict[str, int]] = {
    "member": 6,
    "simpler": 6,
    "placement": 1,
    "band": 15,
}
CAPACITY_POINTS: Final[float] = 100.0
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _, _ in blend.LEAD_BANDS)


def mean_of(series: pl.Series) -> float:
    """Return the mean of a numeric series as a float."""
    return float(series.to_numpy().mean())


def read_arm(*, folder: Path, name: str) -> pl.DataFrame:
    """Return one arm's per-case losses in points of capacity."""
    frame = pl.read_parquet(folder / f"{name}.parquet").sort(*blend.KEYS)
    quantiles = frame.select(blend.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.select(*blend.KEYS, "band", "lead_hours").with_columns(
        qs=pl.Series(CAPACITY_POINTS * blend.quantile_score(quantiles=quantiles, actual=actual)),
        mae=pl.Series(CAPACITY_POINTS * np.abs(quantiles[:, blend.MEDIAN] - actual)),
        covered=pl.Series(
            ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, -1])).astype(np.float64)
        ),
    )


def arm_names(*, folder: Path) -> list[str]:
    """Return the arm files of a technology, without the weight tables."""
    return sorted(p.stem for p in folder.glob("*.parquet") if not p.stem.startswith("weights_"))


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    level: float,
    n: int,
    band: int | None = None,
) -> dict[str, Any]:
    """Return the quantile-score difference treatment minus reference with its interval."""
    scoped = {
        name: (arms[name] if band is None else arms[name].filter(pl.col("band") == band)).select(
            *blend.KEYS, "qs"
        )
        for name in (treatment, reference)
    }
    losses = experiment_store.losses_for_bootstrap(arm_losses=scoped)
    differences, _ = bootstrap.paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric="qs"
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric="qs",
        level=level,
        n_resamples=n,
    )
    return {
        "treatment": treatment,
        "reference": reference,
        "band": "all" if band is None else BAND_NAMES[band],
        "level": level,
        "difference": float(differences.mean()),
        "lower": lower,
        "upper": upper,
        "cases": int(differences.size),
    }


def stop_rule(*, folder: Path) -> dict[str, Any]:
    """Return the corner-cell share of the rolling weight table (60 cells)."""
    table = pl.read_parquet(folder / "weights_rolling.parquet")
    weight_columns = [c for c in table.columns if c.startswith("w_")]
    corner = (table.select(pl.max_horizontal(weight_columns)).to_series() >= 1.0 - 1e-9) & (
        table["history_cases"] > 0
    )
    per_band = (
        table.with_columns(corner=corner)
        .group_by("band")
        .agg(corner=pl.col("corner").sum(), cells=pl.len())
        .sort("band")
    )
    return {
        "corner_cells": int(corner.sum()),
        "cells": table.height,
        "share": float(corner.sum() / table.height),
        "fires": bool(corner.sum() / table.height >= STOP_SHARE - 1e-12),
        "per_band": per_band.to_dicts(),
    }


def outcome(*, row: dict[str, Any], threshold: float, fires: bool) -> str:
    """Return the decisive contrast's outcome under the plan's rule."""
    if fires:
        return "no combination"
    lower, upper, estimate = float(row["lower"]), float(row["upper"]), float(row["difference"])
    if upper < 0:
        return "port" if estimate <= -threshold else "small gain"
    if lower > 0:
        return "harm"
    return "null"


def overall_verdict(*, outcomes: list[str]) -> str:
    """Return the experiment's verdict from the three technologies' outcomes."""
    if "port" in outcomes:
        return "supported"
    if "small gain" in outcomes:
        return "small gain"
    if all(o in {"no combination", "harm"} for o in outcomes):
        return "refuted"
    return "inconclusive"


def main() -> None:
    """Score every technology and write the tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    absolute, contrasts, stops, verdicts = [], [], [], []
    for folder in sorted(
        p for p in blend.WORK_DIR.iterdir() if (p / "weights_rolling.parquet").exists()
    ):
        label = folder.name
        names = arm_names(folder=folder)
        arms = {name: read_arm(folder=folder, name=name) for name in names}
        members = [n for n in names if n.startswith("member_")]
        for name, frame in arms.items():
            losses = experiment_store.losses_for_bootstrap(
                arm_losses={name: frame.select(*blend.KEYS, "qs", "mae", "covered")}
            )
            interval = bootstrap.bootstrap_absolute(losses=losses, arm=name, metric="qs")
            by_band = {
                band_name: mean_of(frame.filter(pl.col("band") == i)["qs"])
                for i, band_name in enumerate(BAND_NAMES)
            }
            absolute.append(
                {
                    "group": label,
                    "arm": name,
                    "qs": interval["value"],
                    "qs_lower_95": interval["lower_95"],
                    "qs_upper_95": interval["upper_95"],
                    "mae": mean_of(frame["mae"]),
                    "coverage_5_95": mean_of(frame["covered"]),
                    "cases": frame.height,
                    **{f"qs_{k}": v for k, v in by_band.items()},
                }
            )
        best = min(members, key=lambda n: mean_of(arms[n]["qs"]))
        blend_arm = "rolling"
        plan = [
            ("decisive", blend_arm, best, DECISIVE_LEVEL, N_RESAMPLES_DECISIVE, None),
        ]
        fam = lambda f: 100.0 * (1.0 - 0.05 / FAMILY_SIZES[f])  # noqa: E731
        for simple in ("equal", "global"):
            plan.append(("member", simple, best, fam("member"), N_RESAMPLES, None))
            plan.append(("simpler", "rolling", simple, fam("simpler"), N_RESAMPLES, None))
        for b in range(len(blend.LEAD_BANDS)):
            plan.append(("band", "rolling", best, fam("band"), N_RESAMPLES, b))  # noqa: PERF401
        if "d2" in arms:
            plan.append(("placement", "d2", "d3", fam("placement"), N_RESAMPLES, None))
            plan.append(("descriptive", "d3", best, 95.0, N_RESAMPLES, None))
        for other in members:
            if other != best:
                plan.append(("descriptive", blend_arm, other, 95.0, N_RESAMPLES, None))  # noqa: PERF401
        for family, treatment, reference, level, n, band in plan:
            row = contrast(
                arms=arms, treatment=treatment, reference=reference, level=level, n=n, band=band
            )
            contrasts.append({"group": label, "family": family, **row})
        stop = stop_rule(folder=folder)
        stops.append({"group": label, **{k: v for k, v in stop.items() if k != "per_band"}})
        pl.DataFrame(stop["per_band"]).with_columns(group=pl.lit(label)).write_parquet(
            RESULTS_DIR / f"stop_per_band_{label}.parquet"
        )
        if label in THRESHOLDS:
            decisive = next(
                c for c in contrasts if c["group"] == label and c["family"] == "decisive"
            )
            verdicts.append(
                {
                    "group": label,
                    "contrast": DECISIVE[label],
                    "best_member": best,
                    **{k: decisive[k] for k in ("difference", "lower", "upper")},
                    "threshold": THRESHOLDS[label],
                    "stop_rule_fires": stop["fires"],
                    "outcome": outcome(
                        row=decisive, threshold=THRESHOLDS[label], fires=bool(stop["fires"])
                    ),
                }
            )
    for name, rows in (
        ("absolute", absolute),
        ("contrasts", contrasts),
        ("stop_rule", stops),
        ("verdicts", verdicts),
    ):
        pl.DataFrame(rows).write_parquet(RESULTS_DIR / f"{name}.parquet")
    print("overall verdict:", overall_verdict(outcomes=[str(v["outcome"]) for v in verdicts]))
    with pl.Config(tbl_rows=200, tbl_cols=30, fmt_str_lengths=40, tbl_width_chars=250):
        print(
            pl.DataFrame(absolute).select(
                "group", "arm", "qs", "qs_lower_95", "qs_upper_95", "mae", "coverage_5_95", "cases"
            )
        )
        print(
            pl.DataFrame(contrasts).select(
                "group",
                "family",
                "treatment",
                "reference",
                "band",
                "difference",
                "lower",
                "upper",
                "level",
            )
        )
        print(pl.DataFrame(stops))
        print(pl.DataFrame(verdicts))


if __name__ == "__main__":
    main()
