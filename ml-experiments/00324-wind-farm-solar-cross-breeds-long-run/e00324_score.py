# ruff: noqa: E501
"""Score experiment 324's arms and read the planned contrasts.

Reads `final/<arm>.npy` (3 seeds, scored cases, 11), reuses experiment 303's row metrics and its
month-and-seed resampler, and writes `score.json`. Contrasts are first minus second, so a negative
number means the first arm is better. The planned contrasts are read at 99% (Bonferroni across the
five predictions) and at 95%. Farms are not resampled.

    PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting \
        uv run --no-sync python ml-experiments/00324-*/e00324_score.py
"""

import json
from typing import Final

import e00303_score as s303  # ty: ignore[unresolved-import]
import e00324_common as common
import numpy as np
import polars as pl

s303.LEVELS_REPORTED = (0.95, 0.99)  # type: ignore[misc]

ARMS: Final[tuple[str, ...]] = (
    "W0",
    "W1",
    "P2",
    "P3",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "W7",
    "W5_LOFO",
    "W8_W1",
    "W8_W4",
)
SETS: Final[tuple[str, ...]] = (
    "kept",
    "all",
    "kept_days_1_2",
    "kept_days_3_10",
    "days_1_2",
    "days_3_10",
    "lead_12_36",
)
ALIAS: Final[dict[str, str]] = {"W1": "P1"}
PLANNED: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "W4", "W1"),
    ("C2", "W5", "W4"),
    ("C3", "W6", "W7"),
    ("C4", "W3", "W2"),
)
EXTRA: Final[tuple[tuple[str, str, str], ...]] = (
    ("x1", "W5", "W1"),
    ("x2", "W5", "P3"),
    ("x3", "W5", "W3"),
    ("x4", "W5", "W2"),
    ("x5", "W3", "W1"),
    ("x6", "W2", "W1"),
    ("x7", "W6", "W5"),
    ("x8", "W5_LOFO", "W5"),
    ("x9", "W5_LOFO", "W1"),
    ("x10", "W8_W4", "W4"),
    ("x11", "W4", "P3"),
    ("x12", "W1", "W0"),
    ("x13", "W5", "W0"),
    ("x14", "W8_W1", "W1"),
)


def main() -> None:
    """Score every stored arm and the contrasts."""
    cases = s303.ordered_cases()
    cases = cases.with_columns(days_3_10=cases["lead_day"].is_between(3, 10)).with_columns(
        kept_days_1_2=pl.col("kept") & pl.col("days_1_2"),
        kept_days_3_10=pl.col("kept") & pl.col("days_3_10"),
    )
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    flags = {name: cases[name].to_numpy() for name in SETS}
    resampler = s303.Resampler(month=cases["month"].to_numpy(), seeds=len(common.SEEDS))
    arms: dict[str, dict[str, np.ndarray]] = {}
    for name in ARMS:
        path = common.FINAL_DIR / f"{ALIAS.get(name, name)}.npy"
        if path.exists():
            arms[name] = s303.row_metrics(quantiles=np.load(path), actual=actual)
    report: dict = {
        "rows": {k: int(v.sum()) for k, v in flags.items()},
        "scores": {},
        "contrasts": {},
    }
    for name, metrics in arms.items():
        report["scores"][name] = {
            row_set: {m: float(v[:, flags[row_set]].mean()) for m, v in metrics.items()}
            for row_set in SETS
        }
        report["scores"][name]["by_farm_kept"] = {
            f: float(metrics["qs"][:, flags["kept"] & (cases["site"] == f).to_numpy()].mean())
            for f in common.FARMS
        }
        report["scores"][name]["by_lead_day_kept"] = {
            str(d): float(
                metrics["qs"][:, flags["kept"] & (cases["lead_day"] == d).to_numpy()].mean()
            )
            for d in range(11)
            if (flags["kept"] & (cases["lead_day"] == d).to_numpy()).any()
        }
    against = tuple(("vsW1", name, "W1") for name in arms if name != "W1")
    for label, first, second in (*PLANNED, *EXTRA, *against):
        if first not in arms or second not in arms:
            continue
        key = f"{label}: {first} - {second}"
        if key in report["contrasts"]:
            continue
        entry: dict = {}
        for metric in (
            "qs",
            "p01",
            "p99",
            "median_abs_error",
            "mean_abs_error",
            "below_median",
            "covered",
        ):
            difference = arms[first][metric] - arms[second][metric]
            for row_set in SETS:
                entry[f"{row_set} | {metric}"] = resampler.interval(
                    values=difference, keep=flags[row_set]
                )
        for farm in common.FARMS:
            site = (cases["site"] == farm).to_numpy()
            difference = arms[first]["qs"] - arms[second]["qs"]
            entry[f"kept without {farm} | qs"] = resampler.interval(
                values=difference, keep=flags["kept"] & ~site
            )
            entry[f"kept only {farm} | qs"] = resampler.interval(
                values=difference, keep=flags["kept"] & site
            )
        report["contrasts"][key] = entry
    (common.DATA_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for name in arms:
        s = report["scores"][name]
        print(
            f"{name:8s} kept {s['kept']['qs']:.3f} all {s['all']['qs']:.3f} d1-2 {s['days_1_2']['qs']:.3f}"
            f" d3-10 {s['days_3_10']['qs']:.3f} p01 {s['kept']['p01']:.3f} p99 {s['kept']['p99']:.3f}"
            f" cov {s['kept']['covered']:.3f} below {s['kept']['below_median']:.3f}"
        )
    for key, value in report["contrasts"].items():
        k = value["kept | qs"]
        print(key, round(k["mean"], 3), [round(x, 3) for x in k["0.99000"]])


if __name__ == "__main__":
    main()
