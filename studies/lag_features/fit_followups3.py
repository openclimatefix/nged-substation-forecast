"""Fit the round-3 post hoc follow-ups to the lagged-power-features study.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. The
third science review asked for these fits. **They add to the earlier runs and never change them**:
every output sits under `<output-root>/ens_mean/followups3/`, with its own checkpoints and
manifest, and the earlier files are only read. **No new column is built**: both blocks read the
first run's lead-day 1 frame, whose columns carry the first run's anchor assertions and leak probe.

**Fits** (point model, primary setting, three seeds, five folds):

1. **Derated-months exclusion** (75 fits: 5 arms x 5 folds x 3 seeds, one plant).
   `followup3_plan.py` chooses the plant (the largest B0 error in a single month of the 8
   unselected months) and its derated months (the months in which its output is below 80% of the
   other plants') by a rule that names neither. B0, PC, W7, TF and Q30 are refitted on that
   plant with the derated months removed from training only. The months are removed by marking
   their rows `constrained`, which
   `out_of_fold_losses` already drops from training and keeps for scoring, so every row is scored
   as before and no row's features change. The saved losses carry that marked flag. Each arm is
   compared with B0 fitted the same way.
   A control, B0 on that plant's rows with nothing marked (15 fits), shows whether a refit on one
   plant's frame reproduces the first run's B0 at that plant.
2. **No day of year for more arms** (360 fits): L1, Q30, AN, and N2-2 without `day_of_year` (the
   `-noDOY` suffix), each at all six plants, to be paired with its own fit with `day_of_year` and
   with B0-noDOY (round 2's fit).

`--dry-run` prints the number of fits and stops. Checkpoints are one file per (scope, setting,
arm). `checkpoints/run_manifest.json` records the device, the mode and a SHA-256 of every file
read, and a resume that differs raises. `--smoke` runs the real path on 20 seeded random rows of
each plant's month with 20 boosting rounds, writes `_smoke` files, and is not a result.

Run it with `uv run python studies/lag_features/fit_followups3.py`.
"""

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import polars as pl
from build_lag_frame import (
    FULL_SWEEP_LEAD_DAY,
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
from followup3_plan import followup3_dir, plan_path
from followup_frames import PRODUCT
from studies.cross_validation import N_FOLDS, SEEDS
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fit_followups3")

PLANTS: Final[int] = 6
"""How many plants a per-plant fit loops over."""

DERATED_ARMS: Final[tuple[str, ...]] = ("B0", "PC", "W7", "TF", "Q30")
"""The arms refitted with the derated months removed from training."""

NO_DAY_OF_YEAR_ARMS: Final[tuple[str, ...]] = ("L1", "Q30", "AN", "N2-2")
"""The arms refitted without `day_of_year`, besides round 2's B0, PC, W7 and TF."""

LEAD1_KEY: Final[str] = "lead1"
"""The key of the first run's lead-day 1 frame."""

DERATED_KEY: Final[str] = "derated"
"""The key of the one-plant frame with the derated months marked."""

DERATED_SCOPE: Final[str] = "fu3_derated"
"""The scope the derated-months fits are saved under."""

PLANT_KEY: Final[str] = "plant_control"
"""The key of the one-plant frame with nothing marked."""

PLANT_CONTROL_SCOPE: Final[str] = "fu3_plant_control"
"""The scope of the control: B0 on the chosen plant's rows with no month removed, which should
reproduce the first run's B0 at that plant and so shows whether a refit on this machine, as one
plant's frame, matches the first run's fit."""


def followup3_fits() -> list[Fit]:
    """Return every round-3 fit, in the order they run.

    Returns:
        The derated-months fits, then the no-day-of-year fits.
    """
    lead = FULL_SWEEP_LEAD_DAY
    fits = [Fit(DERATED_KEY, DERATED_SCOPE, lead, arm, "primary", False) for arm in DERATED_ARMS]
    fits.append(Fit(PLANT_KEY, PLANT_CONTROL_SCOPE, lead, "B0", "primary", False))
    fits += [
        Fit(LEAD1_KEY, "fu3_lead1", lead, f"{arm}{NO_DAY_OF_YEAR_SUFFIX}", "primary", False)
        for arm in NO_DAY_OF_YEAR_ARMS
    ]
    return fits


def fits_in(*, fit: Fit) -> int:
    """Return how many XGBoost fits one `Fit` makes.

    Args:
        fit: The fit.

    Returns:
        15 for a derated-months arm (one plant x 5 folds x 3 seeds), otherwise 90 (six plants).
    """
    plants = 1 if fit.scope in {DERATED_SCOPE, PLANT_CONTROL_SCOPE} else PLANTS
    return plants * N_FOLDS * len(SEEDS)


def fit_count_lines(*, fits: list[Fit]) -> list[str]:
    """Describe the number of fits per group and in all.

    Args:
        fits: `followup3_fits`' result.

    Returns:
        Report lines.
    """
    groups = {
        "1. derated-months exclusion": [f for f in fits if f.scope == DERATED_SCOPE],
        "1b. control: B0 on the plant with nothing removed": [
            f for f in fits if f.scope == PLANT_CONTROL_SCOPE
        ],
        "2. no day of year for more arms": [f for f in fits if f.scope == "fu3_lead1"],
    }
    lines = []
    total = 0
    for name, group in groups.items():
        count = sum(fits_in(fit=fit) for fit in group)
        total += count
        lines.append(f"- {name}: {len(group)} arm-settings, {count} fits")
    lines.append(f"- total: {len(fits)} arm-settings, {total} fits")
    return lines


def input_files(*, root: Path) -> list[Path]:
    """Return every file the round-3 fits read.

    Args:
        root: The output root.

    Returns:
        The plan and the first run's lead-day 1 frame.
    """
    first = output_paths(root=root, product=PRODUCT, lead_days=(FULL_SWEEP_LEAD_DAY,))
    return [plan_path(root=root), first["day1"]]


def input_digests(*, root: Path) -> dict[str, str]:
    """Return the SHA-256 of every file the round-3 fits read.

    Args:
        root: The output root.

    Returns:
        The digests by `<folder>/<file>`.
    """
    return {
        file.parent.name + "/" + file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(input_files(root=root))
    }


def verify_followup3_manifest(*, root: Path, smoke: bool) -> None:
    """Raise if the files on disk are not the ones the round-3 fits read.

    Args:
        root: The output root.
        smoke: Whether the fit was a smoke run.

    Raises:
        ValueError: If the manifest's input digests differ from the files' own.
    """
    path = followup3_dir(root=root) / f"checkpoints{run_suffix(smoke=smoke)}" / "run_manifest.json"
    saved = json.loads(path.read_text())["inputs"]
    if saved != input_digests(root=root):
        msg = f"the input files differ from the ones {path} records the fit as reading"
        raise ValueError(msg)


def derated_frame(*, frame: pl.DataFrame, plan: Mapping[str, object]) -> pl.DataFrame:
    """Return the chosen plant's rows, with its derated months marked as not for training.

    Args:
        frame: The lead-day 1 frame.
        plan: `followup3_plan.choose_plan`'s result, read back from `derated_plan.json`.

    Returns:
        The plant's rows, with `constrained` true in the derated months. `out_of_fold_losses` drops
        a `constrained` row from training and still scores it, so only training changes.
    """
    months = list(plan["months"])  # ty: ignore[invalid-argument-type]
    return frame.filter(pl.col("site") == plan["site"]).with_columns(
        constrained=pl.col("constrained") | pl.col("month").is_in(months)
    )


def read_frames(*, root: Path, smoke: bool) -> dict[str, pl.DataFrame]:
    """Read every frame the fits name, subsampling them as a smoke run does.

    Args:
        root: The output root.
        smoke: Whether to subsample the frame.

    Returns:
        The lead-day 1 frame and the one-plant frame with the derated months marked.
    """
    first = output_paths(root=root, product=PRODUCT, lead_days=(FULL_SWEEP_LEAD_DAY,))
    lead1 = pl.read_parquet(first["day1"])
    if smoke:
        lead1 = smoke_subsample(frame=lead1)
    plan = json.loads(plan_path(root=root).read_text())
    return {
        LEAD1_KEY: lead1,
        DERATED_KEY: derated_frame(frame=lead1, plan=plan),
        PLANT_KEY: lead1.filter(pl.col("site") == plan["site"]),
    }


def require_inputs(*, root: Path) -> None:
    """Fail early, with the fix, if a file the fits read is missing.

    Args:
        root: The output root.

    Raises:
        FileNotFoundError: Naming the missing file and the script that builds it.
    """
    missing = [path for path in input_files(root=root) if not path.exists()]
    if missing:
        hint = " (run build_lag_frame.py and followup3_plan.py into this --output-root first)"
        msg = f"missing input {missing[0]}{hint}"
        raise FileNotFoundError(msg)


def main() -> int:
    """Fit the round-3 follow-ups and write their losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the fit count and stop.")
    arguments = parser.parse_args()
    fits = followup3_fits()
    if arguments.dry_run:
        sys.stdout.write("\n".join(fit_count_lines(fits=fits)) + "\n")
        return 0
    root: Path = arguments.output_root
    suffix = run_suffix(smoke=arguments.smoke)
    require_inputs(root=root)
    directory = followup3_dir(root=root)
    final = directory / f"losses_followups3_{PRODUCT}{suffix}.parquet"
    refuse_to_overwrite(paths=[final])
    checkpoints = directory / f"checkpoints{suffix}"
    checkpoints.mkdir(parents=True, exist_ok=True)
    check_manifest(
        directory=checkpoints,
        device=arguments.device,
        smoke=arguments.smoke,
        inputs=input_digests(root=root),
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
