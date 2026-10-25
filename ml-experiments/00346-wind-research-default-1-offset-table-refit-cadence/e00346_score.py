"""Score experiment 00346's arms and read the adoption conditions.

Reads `work/final/` (written by `e00346_final.py`). Contrasts are arm minus default, in points of
capacity, so a negative difference is an improvement. Intervals resample whole months of the run
date and the three seeds (50,000 resamples, experiment 00342's resampler), at 95% and at the
99.67% level (1 - 0.01/3) that condition 1 reads for three arms. Kept rows are the half-hours that
experiment 210's idle-band mask does not flag.

Output: `work/score.json`.

    uv run --no-sync python ml-experiments/00346-*/e00346_score.py   (PYTHONPATH: see below)

PYTHONPATH also needs the folders of experiments 00340 and 00342.
"""

import json
import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00342_score as score342  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00346_common as common  # noqa: E402
import e00346_final as final  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

DEFAULT: Final[str] = "default"
ADOPTION_ORDER: Final[tuple[str, ...]] = ("x2", "x1", "x3")
DESCRIPTIVE: Final[tuple[str, ...]] = ("refq", "plc", "defaultG")
ARMS: Final[tuple[str, ...]] = (DEFAULT, *ADOPTION_ORDER, *DESCRIPTIVE)
OOT_ARMS: Final[tuple[str, ...]] = (DEFAULT, "x1", "x2", "x3", "refq")
METRICS: Final[tuple[str, ...]] = (
    "qs",
    "tgt",
    "p01",
    "p05",
    "p95",
    "p99",
    "above_q99",
    "above_q95",
    "below_q01",
    "below_q05",
    "mean_abs_error",
    "median_abs_error",
    "median_minus_mean",
)
HEADLINE_TOLERANCE: Final[float] = 0.02
TAIL_TOLERANCE: Final[float] = 0.05
FIRST_HALF: Final[tuple[str, ...]] = tuple(f"2025-{m:02d}" for m in range(7, 13))
SECOND_HALF: Final[tuple[str, ...]] = tuple(f"2026-{m:02d}" for m in range(1, 7))


def subsets(*, frame: pl.DataFrame, out_of_time: bool) -> dict[str, np.ndarray]:
    """Return the boolean row classes of a case frame."""
    kept = ~frame["masked"].to_numpy()
    if out_of_time:
        kept &= (frame["valid_time"] <= final.MASK_END).to_numpy()
    lead = frame["lead_day"].clip(0, 10)
    month = frame["month"].to_numpy()
    return {
        "all": np.ones(frame.height, dtype=bool),
        "kept": kept,
        "kept_days_1_2": kept & lead.is_between(1, 2).to_numpy(),
        "kept_days_3_10": kept & lead.is_between(3, 10).to_numpy(),
        "kept_first_half": kept & np.isin(month, FIRST_HALF),
        "kept_second_half": kept & np.isin(month, SECOND_HALF),
        "idle210": frame["masked"].to_numpy(),
    }


def score_frame(*, name: str, suffix: str, arms: tuple[str, ...], out_of_time: bool) -> dict:
    """Return the scores, contrasts, and per-month gains of one case frame."""
    frame = pl.read_parquet(common.FINAL_DIR / f"frames{suffix}.parquet")
    actual = frame["actual_norm"].to_numpy().astype(np.float64)
    classes = subsets(frame=frame, out_of_time=out_of_time)
    metrics = {
        arm: score342.all_metrics(
            quantiles=np.load(common.FINAL_DIR / f"{arm}{suffix}.npy"), actual=actual
        )
        for arm in arms
    }
    resampler = score342.Intervals(month=frame["month"].to_numpy(), seeds=len(common.SEEDS))
    site = frame["site"].to_numpy()
    month = frame["month"].to_numpy()
    report: dict = {"frame": name, "cases": frame.height, "scores": {}, "contrasts": {}}
    for arm, values in metrics.items():
        report["scores"][arm] = {
            subset: {m: float(values[m][:, keep].mean()) for m in METRICS if keep.any()}
            for subset, keep in classes.items()
            if keep.any()
        } | {
            "by_farm_kept": {
                farm: {
                    m: float(values[m][:, classes["kept"] & (site == farm)].mean()) for m in METRICS
                }
                for farm in common.FARMS
            }
        }
    for arm in arms[1:]:
        entry: dict = {}
        for metric in METRICS:
            difference = metrics[arm][metric] - metrics[DEFAULT][metric]
            for subset, keep in classes.items():
                if keep.any():
                    entry[f"{subset} | {metric}"] = resampler.read(values=difference, keep=keep)
        difference = metrics[arm]["qs"] - metrics[DEFAULT]["qs"]
        entry["by_month | qs"] = {
            m: float(difference[:, classes["kept"] & (month == m)].mean())
            for m in np.unique(month)
            if (classes["kept"] & (month == m)).any()
        }
        for farm in common.FARMS:
            entry[f"kept only {farm} | qs"] = resampler.read(
                values=difference, keep=classes["kept"] & (site == farm)
            )
            entry[f"kept without {farm} | qs"] = resampler.read(
                values=difference, keep=classes["kept"] & (site != farm)
            )
        report["contrasts"][arm] = entry
    return report


def conditions(*, scored: dict, reproduction: dict, leakage: list[dict], known: dict) -> dict:
    """Read the adoption conditions for x2, x1, x3 in order; at most one arm is adopted."""
    level = f"{score342.CONDITION_1_LEVEL:.4f}"
    placebo_gain = scored["contrasts"]["plc"]["kept | qs"]["mean"]
    leaks = sum(row["latest_valid_used"] > row["month_start"] for row in leakage)
    out: dict = {"adopted": None, "arms": {}}
    for arm in ADOPTION_ORDER:
        c = scored["contrasts"][arm]
        gain = c["kept | qs"]
        checks = {
            "c1_interval_below_zero": gain[level][1] < 0.0,
            "c1_point_estimate_below_zero": gain["mean"] < 0.0,
            "c2_headline_upper_bound": gain["upper_95_one_sided"] <= HEADLINE_TOLERANCE,
            "c2_p01_upper_bound": c["kept | p01"]["upper_95_one_sided"] <= TAIL_TOLERANCE,
            "c2_p99_upper_bound": c["kept | p99"]["upper_95_one_sided"] <= TAIL_TOLERANCE,
            "c4_gate": abs(reproduction["difference_defaultG"]) <= 0.01,
            "c4_placebo_keeps_under_half_of_x2_gain": -placebo_gain
            < 0.5 * -scored["contrasts"]["x2"]["kept | qs"]["mean"],
            "c4_known_answer": known["passed"],
            "c5_no_leak": leaks == 0,
        }
        out["arms"][arm] = {"checks": checks, "all_pass": all(checks.values())}
        if out["adopted"] is None and all(checks.values()):
            out["adopted"] = arm
    return out


def main() -> None:
    """Score the scored year and the out-of-time months, and write the report."""
    reproduction = json.loads((common.FINAL_DIR / "reproduction.json").read_text())
    leakage = json.loads((common.FINAL_DIR / "leakage.json").read_text())
    known = json.loads((common.FINAL_DIR / "known_answer.json").read_text())
    scored = score_frame(name="scored", suffix="", arms=ARMS, out_of_time=False)
    oot = score_frame(name="out_of_time", suffix="_oot", arms=OOT_ARMS, out_of_time=True)
    report = {
        "reproduction": reproduction,
        "known_answer": known,
        "scored": scored,
        "out_of_time": oot,
        "conditions": conditions(
            scored=scored, reproduction=reproduction, leakage=leakage, known=known
        ),
    }
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    level = f"{score342.CONDITION_1_LEVEL:.4f}"
    print("reproduction", reproduction)
    for label, rep in (("scored year", scored), ("out of time", oot)):
        print(f"\n{label}: {rep['cases']:,} cases; kept-row quantile score and gain vs default")
        for arm, values in rep["scores"].items():
            line = f"{arm:9s} qs {values['kept']['qs']:.3f}  p01 {values['kept']['p01']:.3f}"
            line += (
                f"  p99 {values['kept']['p99']:.3f}  above_q99 {values['kept']['above_q99']:.4f}"
            )
            if arm != DEFAULT:
                gain = rep["contrasts"][arm]["kept | qs"]
                line += f"  gain {gain['mean']:+.3f} [{gain[level][0]:+.3f}, {gain[level][1]:+.3f}]"
            print(line)
    print("\nconditions:", json.dumps(report["conditions"], indent=1))


if __name__ == "__main__":
    main()
