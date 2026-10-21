"""Evaluate the known-answer gate's readings from `gate.json`, against thresholds frozen here.

The thresholds are the plan's; they are written as constants before the gate runs, and a reading
that fails is not redefined afterwards. Reading 1 (interval coverage) and the heating-slope width of
reading 2 are not run, as `e00124_gate.py` says.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from e00124_common import WORK_DIR

CAPACITY_BIAS_RATIO: Final[float] = 1.5
PV_RELATIVE_ERROR: Final[float] = 0.30
LEVEL_ERROR_MW: Final[float] = 0.5
SIBLING_RATIO: Final[float] = 1.2
SIBLING_FLOOR_POINTS: Final[float] = 0.01
SABOTAGE_SIGMAS: Final[float] = 3.0
SABOTAGE_SHARE: Final[float] = 0.8
REDUCTION_MINIMUM: Final[float] = 0.30
MIN_DAMAGE_POINTS: Final[float] = 0.02
HALF_WIDTH_LIMIT: Final[float] = 0.03
BOOTSTRAP_DRAWS: Final[int] = 5000
INTERVAL_PERCENTILES: Final[tuple[float, float]] = (0.415, 99.585)
"""The 99.17% interval, the Bonferroni level for three contrasts at 97.5%."""


def main() -> None:
    """Print each reading's number, its threshold, and whether it passes."""
    gate = json.loads((WORK_DIR / "gate.json").read_text())
    rows = gate["rows"]
    w_star = 1
    lines = ["# Experiment 124, known-answer gate", ""]
    closed = [r for r in rows if not r["open"]]
    open_ = [r for r in rows if r["open"]]
    results: dict[str, bool] = {}

    bias = {
        k: np.median([abs(v[k]) for r in rows for v in r["capacity_bias"].values()])
        for k in (0, w_star)
    }
    pv_error = np.median(
        [abs(r["pv"][w_star] - gate["pv_truth"]) / abs(gate["pv_truth"]) for r in rows]
    )
    level_error = np.median([abs(r["level"][w_star] - r["level_truth"]) for r in closed])
    ok2 = (
        bias[w_star] <= CAPACITY_BIAS_RATIO * bias[0]
        and pv_error <= PV_RELATIVE_ERROR
        and level_error <= LEVEL_ERROR_MW
    )
    results["2 no harm"] = ok2
    lines.append(
        "- Reading 2 (no harm, interval width not run): "
        f"median capacity bias {bias[w_star]:.3f} points "
        f"against weight 0's {bias[0]:.3f} (limit {CAPACITY_BIAS_RATIO} times); unmetered PV "
        f"capacity error {100 * pv_error:.0f}% (limit {100 * PV_RELATIVE_ERROR:.0f}%); level error "
        f"{level_error:.2f} MW in closed systems (limit {LEVEL_ERROR_MW} MW): "
        f"{'pass' if ok2 else 'FAIL'}"
    )

    sibling = {
        k: np.median([d[k] for r in rows for d in r["sibling_damage"].values()])
        for k in (0, w_star)
    }
    ok4 = sibling[w_star] <= max(SIBLING_RATIO * sibling[0], SIBLING_FLOOR_POINTS)
    results["4 error leak"] = ok4
    lines.append(
        f"- Reading 4 (error leak): the siblings' median damage is {sibling[w_star]:.3f} points at "
        f"the chosen weight against {sibling[0]:.3f} at weight 0 (limit {SIBLING_RATIO} times, "
        f"floor {SIBLING_FLOOR_POINTS}): {'pass' if ok4 else 'FAIL'}"
    )

    level_sd = float(np.std([r["level"][w_star] - r["level_truth"] for r in closed], ddof=1))
    shifts = np.array([abs(r["level_sabotaged"][w_star] - r["level"][w_star]) for r in rows])
    share = float(np.mean(shifts > SABOTAGE_SIGMAS * level_sd))
    ok5 = share >= SABOTAGE_SHARE
    results["5 sabotage"] = ok5
    lines.append(
        "- Reading 5 (sabotage): dropping one Primary from the BSP row moves the remainder's level "
        f"beyond {SABOTAGE_SIGMAS:g} clean-system standard deviations ({level_sd:.3f} MW) in "
        f"{100 * share:.0f}% of systems (need {100 * SABOTAGE_SHARE:.0f}%): "
        f"{'pass' if ok5 else 'FAIL'}"
    )

    halves = {}
    for name, group in (("closed", closed), ("open", open_)):
        reductions = [
            1.0 - r["damage"][w_star] / r["damage"][0]
            for r in group
            if r["damage"][0] > MIN_DAMAGE_POINTS
        ]
        halves[name] = (
            (float(np.median(reductions)), len(reductions)) if reductions else (float("nan"), 0)
        )
    ok6 = all(v[0] >= REDUCTION_MINIMUM for v in halves.values())
    results["6 positive control"] = ok6
    lines.append(
        "- Reading 6 (positive control, damage on the clean scored year): median reduction "
        + ", ".join(
            f"{k} systems {100 * v[0]:.0f}% ({v[1]} with damage)" for k, v in halves.items()
        )
        + f" (need {100 * REDUCTION_MINIMUM:.0f}% in both): {'pass' if ok6 else 'FAIL'}"
    )

    diffs = np.array([np.mean([m[w_star] - m[0] for m in r["clean_mae"].values()]) for r in rows])
    rng = np.random.default_rng(0)
    means = np.array([rng.choice(diffs, size=len(diffs)).mean() for _ in range(BOOTSTRAP_DRAWS)])
    low, high = np.percentile(means, INTERVAL_PERCENTILES)
    half_width = float((high - low) / 2.0)
    ok7 = half_width <= HALF_WIDTH_LIMIT
    results["7 power"] = ok7
    lines.append(
        f"- Reading 7 (power for C1): mean clean-score difference between the chosen weight and "
        f"weight 0 is {diffs.mean():+.3f} points over {len(diffs)} systems; the 99.17% interval "
        f"half-width is {half_width:.3f} points (limit {HALF_WIDTH_LIMIT}): "
        f"{'pass' if ok7 else 'capped at partly supported'}"
    )
    lines.append("")
    lines.append(
        f"Passed: {', '.join(k for k, v in results.items() if v) or 'none'}. "
        f"Failed: {', '.join(k for k, v in results.items() if not v) or 'none'}."
    )
    text = "\n".join(lines) + "\n"
    (WORK_DIR / "gate_readings.md").write_text(text)
    (
        Path(__file__).parents[2]
        / "docs"
        / "ml-experiments"
        / Path(__file__).parent.name
        / "gate_readings.md"
    ).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
