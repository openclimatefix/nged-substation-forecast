"""Build the frames of the round-2 post hoc follow-ups to the lagged-power-features study.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. The
second science review of the draft page asked for four small fit blocks. **Every output is post
hoc** and sits under `<output-root>/ens_mean/followups2/`, so no file of the first run or of round
1 is written. The blocks, and where each reads its frame:

1. **PC in the plant-specific step control** at 5% and 10%: this script builds the two frames,
   `control_step_pc_s05` and `control_step_pc_s10`, with the same steps as round 1's
   `control_step_s<percent>` frames (the steps are seeded by `followup_frames.plant_steps`) and the
   PC columns built from the stepped power. The leak probe runs on the 10% frame. Round 1's
   `control_step` frames hold B0 and O on the same rows, so PC pairs with them.
2. **Width-matched nulls** N2-2 and N2-3 at lead-day 1: read from the first run's lead-day 1 frame,
   which holds all 16 random-lag columns. No build.
3. **Day-of-year test**: B0, PC, W7, and TF refitted without `day_of_year` at lead-day 1, read from
   the first run's lead-day 1 frame. No build.
4. **IFS HRES replicate**: `build_lag_frame.py --weather-product ifs_single` builds the frames
   (`<output-root>/ifs_single/`); this script builds nothing for it.

Run it with `uv run python studies/lag_features/followup2_frames.py`.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

from build_lag_frame import (
    LAG_FEATURES_DIR,
    arms_columns_table,
    lag_source_hourly,
    raw_hourly_power,
    shared_rows,
    weather_table,
    write_parquet_atomic,
)
from followup_frames import (
    CONTROL_SHIFTS,
    PRODUCT,
    REQUIRED_ARM,
    build_control,
    plant_steps,
    step_factor,
)
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("followup2_frames")

STEP_PC_ARMS: Final[tuple[str, ...]] = ("PC",)
"""The arm the round-2 step-control frames serve: PC, whose columns round 1's step frames lack."""

LEAK_PROBED_PERCENT: Final[str] = "10"
"""The step frame the leak probe runs on, as in round 1."""


def followup2_dir(*, root: Path) -> Path:
    """Return the folder every round-2 output lives in.

    Args:
        root: The output root.

    Returns:
        `<root>/ens_mean/followups2`.
    """
    return root / PRODUCT / "followups2"


def frame_names() -> tuple[str, ...]:
    """Return the names of the frames this script builds.

    Returns:
        `control_step_pc_s05` and `control_step_pc_s10`.
    """
    return tuple(f"control_step_pc_s{round(shift * 100):02d}" for shift in CONTROL_SHIFTS)


def frame_path(*, root: Path, name: str) -> Path:
    """Return a round-2 frame's file.

    Args:
        root: The output root.
        name: The frame's name from `frame_names`.

    Returns:
        The parquet path, with the product in the file name.
    """
    return followup2_dir(root=root) / "frames" / f"{name}_{PRODUCT}.parquet"


def main() -> int:
    """Build the round-2 frames and write the build report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    arguments = parser.parse_args()
    root: Path = arguments.output_root
    names = frame_names()
    report_path = followup2_dir(root=root) / f"build_report_followups2_{PRODUCT}.md"
    refuse_to_overwrite(paths=[report_path, *(frame_path(root=root, name=n) for n in names)])

    shared = shared_rows()
    hourly = lag_source_hourly()
    raw = raw_hourly_power()
    weather_lead0 = weather_table(product=PRODUCT, lead_day=0)
    steps = plant_steps(hourly=hourly)
    report = [
        "# Round-2 follow-up frames (post hoc)",
        "",
        "All frames are post hoc, built after the second science review.",
        "",
    ]
    for shift in CONTROL_SHIFTS:
        percent = f"{round(shift * 100):02d}"
        name = f"control_step_pc_s{percent}"
        _LOG.info("building %s", name)
        arms = tuple(dict.fromkeys((REQUIRED_ARM, "O", *STEP_PC_ARMS)))
        frame, lines = build_control(
            name=name,
            arms=arms,
            factor=step_factor(steps=steps, shift=shift),
            shared=shared,
            hourly=hourly,
            raw=raw,
            weather_lead0=weather_lead0,
            probe=percent == LEAK_PROBED_PERCENT,
        )
        write_parquet_atomic(frame=frame, path=frame_path(root=root, name=name))
        _LOG.info("%s: %d rows", name, frame.height)
        report += [f"## {name}", "", f"Arms: {', '.join(arms)}.", "", *lines, ""]
        report += [*arms_columns_table(arms=arms), ""]
        report += [f"Rows: {frame.height}.", ""]
        del frame
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n")
    sys.stdout.write("\n".join(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
