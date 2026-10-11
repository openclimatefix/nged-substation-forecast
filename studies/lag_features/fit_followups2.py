"""Fit the round-2 post hoc follow-ups to the lagged-power-features study.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. The
second science review of the draft page asked for these fits. **They add to the first run and to
round 1 and never change either**: every output sits under `<output-root>/ens_mean/followups2/`,
with its own checkpoints and manifest, and the earlier files are only read.

**Fits** (a per-plant arm is 90 fits: 6 plants x 5 folds x 3 seeds; point model, primary setting):

1. **PC in the plant-specific step control** at 5% and 10% (180 fits), on the frames
   `followup2_frames.py` builds. B0 and O of the same control are round 1's fits.
2. **Width-matched nulls** N2-2 and N2-3 at lead-day 1 (180 fits), on the first run's lead-day 1
   frame, which holds all 16 random-lag columns.
3. **Day-of-year test**: B0, PC, W7, and TF without `day_of_year` (named `<arm>-noDOY`) at
   lead-day 1 (360 fits), on the first run's lead-day 1 frame. Each is compared with its own fit
   with `day_of_year`, which is the first run's.
4. **IFS HRES replicate**: B0 and L1 at lead-days 1 and 2 (360 fits), on the frames
   `build_lag_frame.py --weather-product ifs_single` builds under `<output-root>/ifs_single/`.

`--dry-run` prints the number of fits and stops. Checkpoints are one file per (scope, setting,
arm). `checkpoints/run_manifest.json` records the device, the mode and a SHA-256 of every file
read, and a resume that differs raises. `--smoke` runs the real path on 20 seeded random rows of
each plant's month with 20 boosting rounds, writes `_smoke` files, and is not a result.

Run it with `uv run python studies/lag_features/fit_followups2.py`.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Final

import polars as pl
from build_lag_frame import (
    FULL_SWEEP_LEAD_DAY,
    IFS_LEAD_DAYS,
    LAG_FEATURES_DIR,
    output_paths,
    run_suffix,
    smoke_subsample,
    write_parquet_atomic,
)
from fit_lag_arms import (
    NO_DAY_OF_YEAR_SUFFIX,
    Context,
    Fit,
    check_manifest,
    fit_all,
)
from followup2_frames import followup2_dir, frame_names, frame_path
from followup_frames import CONTROL_SHIFTS, PRODUCT
from studies.cross_validation import N_FOLDS, SEEDS
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fit_followups2")

PLANTS: Final[int] = 6
"""How many plants a per-plant fit loops over."""

STEP_PC_ARM: Final[str] = "PC"
"""The arm fitted in the plant-specific step control."""

WIDTH_NULL_ARMS: Final[tuple[str, ...]] = ("N2-2", "N2-3")
"""The nulls with two and three random-day lags, so the page can state the width bias that the
no-padding rule leaves between a one-column null and a multi-column arm."""

DAY_OF_YEAR_ARMS: Final[tuple[str, ...]] = ("B0", "PC", "W7", "TF")
"""The arms refitted without `day_of_year`."""

IFS_ARMS: Final[tuple[str, ...]] = ("B0", "L1")
"""The arms fitted with IFS HRES."""

LEAD1_KEY: Final[str] = "lead1"
"""The key of the first run's lead-day 1 frame in the frames the fits read."""


def ifs_key(*, lead_day: int) -> str:
    """Return the key of an IFS HRES frame.

    Args:
        lead_day: The lead-day.

    Returns:
        `ifs_lead<N>`.
    """
    return f"ifs_lead{lead_day}"


def followup2_fits() -> list[Fit]:
    """Return every round-2 fit, in the order they run.

    Returns:
        The fits: the step-control PC fits, the width-matched nulls, the day-of-year test, and the
        IFS HRES replicate.
    """
    lead = FULL_SWEEP_LEAD_DAY
    fits: list[Fit] = []
    for shift in CONTROL_SHIFTS:
        name = f"control_step_pc_s{round(shift * 100):02d}"
        fits.append(Fit(name, f"fu2_{name}", lead, STEP_PC_ARM, "primary", quantiles=False))
    fits += [
        Fit(LEAD1_KEY, "fu2_lead1", lead, arm, "primary", quantiles=False)
        for arm in WIDTH_NULL_ARMS
    ]
    fits += [
        Fit(LEAD1_KEY, "fu2_lead1", lead, f"{arm}{NO_DAY_OF_YEAR_SUFFIX}", "primary", False)
        for arm in DAY_OF_YEAR_ARMS
    ]
    for lead_day in IFS_LEAD_DAYS:
        fits += [
            Fit(
                ifs_key(lead_day=lead_day),
                f"fu2_ifs_lead{lead_day}",
                lead_day,
                arm,
                "primary",
                quantiles=False,
            )
            for arm in IFS_ARMS
        ]
    return fits


def fits_in(*, fit: Fit) -> int:
    """Return how many XGBoost fits one `Fit` makes.

    Args:
        fit: The fit.

    Returns:
        90 for a per-plant arm: 6 plants x 5 folds x 3 seeds.
    """
    return PLANTS * N_FOLDS * len(SEEDS)


def fit_count_lines(*, fits: list[Fit]) -> list[str]:
    """Describe the number of fits per group and in all.

    Args:
        fits: `followup2_fits`' result.

    Returns:
        Report lines.
    """
    groups = {
        "1. PC in the plant-specific step control": [
            f for f in fits if f.scope.startswith("fu2_control_step_pc")
        ],
        "2. width-matched nulls N2-2 and N2-3": [
            f for f in fits if f.arm in WIDTH_NULL_ARMS and f.scope == "fu2_lead1"
        ],
        "3. day-of-year test": [f for f in fits if f.arm.endswith(NO_DAY_OF_YEAR_SUFFIX)],
        "4. IFS HRES replicate": [f for f in fits if f.scope.startswith("fu2_ifs")],
    }
    lines = []
    total = 0
    for name, group in groups.items():
        count = sum(fits_in(fit=fit) for fit in group)
        total += count
        lines.append(f"- {name}: {len(group)} arm-settings, {count} fits")
    lines.append(f"- total: {len(fits)} arm-settings, {total} fits")
    return lines


def input_files(*, root: Path, smoke: bool) -> list[Path]:
    """Return every file the round-2 fits read.

    Args:
        root: The output root.
        smoke: Whether the run is a smoke run.

    Returns:
        The round-2 frames, the first run's lead-day 1 frame, and the IFS HRES frames. The fits
        read no losses.
    """
    first = output_paths(root=root, product=PRODUCT, lead_days=(FULL_SWEEP_LEAD_DAY,))
    ifs = output_paths(root=root, product="ifs_single", lead_days=IFS_LEAD_DAYS)
    return [
        *(frame_path(root=root, name=name) for name in frame_names()),
        first["day1"],
        *(ifs[f"day{day}"] for day in IFS_LEAD_DAYS),
    ]


def input_digests(*, root: Path, smoke: bool) -> dict[str, str]:
    """Return the SHA-256 of every file the round-2 fits read.

    Args:
        root: The output root.
        smoke: Whether the run is a smoke run.

    Returns:
        The digests by file name.
    """
    return {
        file.parent.parent.name + "/" + file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(input_files(root=root, smoke=smoke))
    }


def verify_followup2_manifest(*, root: Path, smoke: bool) -> None:
    """Raise if the files on disk are not the ones the round-2 fits read.

    Args:
        root: The output root.
        smoke: Whether the fit was a smoke run.

    Raises:
        ValueError: If the manifest's input digests differ from the files' own, so a report would
            describe other frames or losses than the ones fitted.
    """
    path = followup2_dir(root=root) / f"checkpoints{run_suffix(smoke=smoke)}" / "run_manifest.json"
    saved = json.loads(path.read_text())["inputs"]
    if saved != input_digests(root=root, smoke=smoke):
        msg = f"the input files differ from the ones {path} records the fit as reading"
        raise ValueError(msg)


def read_frames(*, root: Path, smoke: bool) -> dict[str, pl.DataFrame]:
    """Read every frame the fits name, subsampling them as a smoke run does.

    Args:
        root: The output root.
        smoke: Whether to subsample each frame.

    Returns:
        The frames by key.
    """
    frames = {name: pl.read_parquet(frame_path(root=root, name=name)) for name in frame_names()}
    first = output_paths(root=root, product=PRODUCT, lead_days=(FULL_SWEEP_LEAD_DAY,))
    frames[LEAD1_KEY] = pl.read_parquet(first["day1"])
    ifs = output_paths(root=root, product="ifs_single", lead_days=IFS_LEAD_DAYS)
    for lead_day in IFS_LEAD_DAYS:
        frames[ifs_key(lead_day=lead_day)] = pl.read_parquet(ifs[f"day{lead_day}"])
    if smoke:
        frames = {name: smoke_subsample(frame=frame) for name, frame in frames.items()}
    return frames


def require_inputs(*, root: Path, smoke: bool) -> None:
    """Fail early, with the fix, if a file the fits read is missing.

    Args:
        root: The output root.
        smoke: Whether the run is a smoke run.

    Raises:
        FileNotFoundError: Naming the missing file and the script that builds it.
    """
    missing = [path for path in input_files(root=root, smoke=smoke) if not path.exists()]
    if missing:
        hint = (
            " (run build_lag_frame.py, followup2_frames.py and build_lag_frame.py "
            "--weather-product ifs_single into this --output-root first)"
        )
        msg = f"missing input {missing[0]}{hint}"
        raise FileNotFoundError(msg)


def main() -> int:
    """Fit the round-2 follow-ups and write their losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the fit count and stop.")
    arguments = parser.parse_args()
    fits = followup2_fits()
    if arguments.dry_run:
        sys.stdout.write("\n".join(fit_count_lines(fits=fits)) + "\n")
        return 0
    root: Path = arguments.output_root
    suffix = run_suffix(smoke=arguments.smoke)
    require_inputs(root=root, smoke=arguments.smoke)
    directory = followup2_dir(root=root)
    final = directory / f"losses_followups2_{PRODUCT}{suffix}.parquet"
    refuse_to_overwrite(paths=[final])
    checkpoints = directory / f"checkpoints{suffix}"
    checkpoints.mkdir(parents=True, exist_ok=True)
    check_manifest(
        directory=checkpoints,
        device=arguments.device,
        smoke=arguments.smoke,
        inputs=input_digests(root=root, smoke=arguments.smoke),
    )
    context = Context(
        checkpoint_dir=checkpoints,
        device=arguments.device,
        max_workers=arguments.max_workers,
        smoke=arguments.smoke,
    )
    frames = read_frames(root=root, smoke=arguments.smoke)
    results = fit_all(context=context, frames=frames, fits=fits)
    losses = pl.concat(results, how="diagonal_relaxed")
    write_parquet_atomic(frame=losses, path=final)
    _LOG.info("wrote %s (%d rows)", final, losses.height)
    sys.stdout.write("\n".join(fit_count_lines(fits=fits)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
