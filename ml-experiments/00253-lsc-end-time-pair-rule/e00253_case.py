r"""Print the intervals every rule gives for the 14 January 2026 shift between D5 and D16.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00253-*/e00253_case.py
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime, timedelta

import numpy as np
from e00253_common import GRID_START, WORK_DIR
from e00253_endrule import RULES, Candidates, set_scorer
from e00253_run import CAP_DAYS, build_detectors, detect, make_state
from e00253_variants import load_both

SIDES: dict[int, str] = {17: "D5", 18: "D16"}
"""Detector series of the two substations, and their labels."""
WINDOW_DAYS: int = 12


def main() -> None:
    """Print each side's interval around 14 January 2026 under every rule."""
    panel, offset_panel = load_both()
    state = make_state()
    with (WORK_DIR / "run" / "scorer.pkl").open("rb") as handle:
        set_scorer(scorer=pickle.load(handle))
    results, _ = build_detectors(
        panel=panel,
        offset_panel=offset_panel,
        residual=panel["residual"],
        offset=offset_panel["residual"],
        scales=state["scales"],
        thresholds=state["thresholds"],
        models=state["models"],
    )
    centre = int((datetime(2026, 1, 14, tzinfo=UTC) - GRID_START).total_seconds() // 1800)
    for rule in RULES:
        for interval in results[rule][1]:
            if interval.series in SIDES and abs(interval.onset - centre) <= WINDOW_DAYS * 48:
                start = GRID_START + timedelta(minutes=30 * interval.onset)
                print(
                    f"{rule} {SIDES[interval.series]}: starts {start:%Y-%m-%d %H:%M}, "
                    f"lasts {(interval.end - interval.onset) / 48:.1f} days"
                )
    diagnose(panel=panel, offset_panel=offset_panel, state=state, centre=centre)


def diagnose(*, panel: dict, offset_panel: dict, state: dict, centre: int) -> None:
    """Print why the step rules cannot move the ends: how many stamps qualify and balance."""
    _, tracked, fast = detect(
        panel=panel,
        offset_panel=offset_panel,
        residual=panel["residual"],
        offset=offset_panel["residual"],
        scales=state["scales"],
        thresholds=state["thresholds"],
        models=state["models"],
    )
    for interval in tracked:
        if interval.series in SIDES and abs(interval.onset - centre) < 3 * 48:
            cand = Candidates(inputs=fast, iv=interval, cap_days=CAP_DAYS)
            qualifying = cand.qualifies()
            print(
                f"{SIDES[interval.series]}: tracked end {(interval.end - interval.onset) / 48:.1f} "
                f"days, noise units needed {cand.need_z:.1f}, largest return "
                f"{np.nanmax(cand.z_ret):.1f}, qualifying stamps {int(qualifying.sum())}, "
                f"balanced and qualifying {int((qualifying & cand.balanced()).sum())}"
            )


if __name__ == "__main__":
    main()
