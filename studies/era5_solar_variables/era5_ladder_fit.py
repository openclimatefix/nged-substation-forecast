"""Fit the ERA5 variable ladder, its controls, and the drop-one-group runs, out of fold.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. It reads the frame
`era5_ladder_build_dataset.py` wrote and, for each target, fits one XGBoost model per arm, farm,
fold, and seed with `studies.arm_runner.run_all`. It writes the per-row losses and a JSON naming
every arm's columns, the device, and the hyperparameter settings, under
`data/studies/per_study/era5_solar_variables/results/`.

**Every arm is scored on exactly the same rows.** The script raises if two arms of one fit hold
different (farm, time, seed) rows, and it raises if an arm is shown a column that is missing on any
row, except `cbh` and `cin`, whose missing values are kept as values.

**The two targets share their folds and rows.** The output target is each farm's output in MW,
scored as a fraction of the farm's capacity. The CAMS target is CAMS clearness index, with the
capacity set to 1 so that the error is in clearness-index units; no export cap applies and no hour
is treated as constrained. Hours the network operator constrained are scored for the output target
and left out of its training.

**The hyperparameter settings.** Every arm is fitted at `PRIMARY_HYPER_PARAMETERS`. The arms of the
planned contrasts are also fitted at `SENSITIVITY_HYPER_PARAMETERS`; `--sensitivity-arms` adds more.
Column subsampling is off, so an arm with more columns has no advantage from the count.

**The aerosol view refits `g9` and `g10` on the rows that EAC4 covers**, with the folds cut again on
that shorter span, because the two arms must be scored on the same rows.

Run it with `uv run python studies/era5_solar_variables/era5_ladder_fit.py`. Check that no other
XGBoost run is using the CPU first. The script prints the load average and refuses to start above
`MAX_LOAD_PER_CORE` unless `--ignore-load` is given.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from typing import Final

import polars as pl
from era5_ladder_arms import (
    AEROSOL_REFERENCE,
    AEROSOL_RUNG,
    METRIC,
    NEGATIVE_CONTROL_SEED,
    PERMUTATION_GROUPING,
    PRIMARY_SETTING,
    RESULTS_DIR,
    SENSITIVITY_ARMS,
    SENSITIVITY_SETTING,
    TARGET_COLUMNS,
    TARGETS,
    FitKey,
    TargetType,
    aerosol_arm_features,
    arm_features,
    arms_path,
    dataset_path,
    results_path,
)
from studies.arm_runner import Job, run_all
from studies.blending import PERMUTED_SUFFIX, climatology_permutation
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    DeviceType,
    HyperParameters,
    assign_folds,
)
from studies.era5_ladder import (
    AEROSOL_COLUMNS,
    MISSING_UNDER_CLEAR_SKY_VARIABLES,
    RUNGS,
    RungType,
    negative_control_columns,
    raise_unless_same_rows,
)
from studies.guards import check_no_missing, refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("era5_ladder_fit")

MAX_LOAD_PER_CORE: Final[float] = 0.5
"""The run refuses to start if the one-minute load average per core is above this."""

DEFAULT_WORKERS: Final[int] = 4
"""How many (arm, farm) fits run at once. A GPU fit needs about one CPU core."""

SETTINGS: Final[dict[str, HyperParameters]] = {
    PRIMARY_SETTING: PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_SETTING: SENSITIVITY_HYPER_PARAMETERS,
}
"""The two hyperparameter settings by the name the losses carry."""

SNOW_VARIANT_ARMS: Final[tuple[str, ...]] = ("g7", "g8")
"""The only arms fitted on the snow variant's rows: the snow rung and the rung below it."""


def choose_device(*, requested: str) -> DeviceType:
    """Return the XGBoost device: the GPU if `nvidia-smi` shows one, unless the caller chooses.

    Args:
        requested: `auto`, `cpu`, or `cuda`.

    Returns:
        `"cuda"` or `"cpu"`.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    if shutil.which("nvidia-smi") is None:
        return "cpu"
    probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=False, text=True)
    return "cuda" if probe.returncode == 0 and "GPU" in probe.stdout else "cpu"


def refuse_if_machine_is_busy(*, ignore: bool) -> None:
    """Raise if another job is already using the CPU, because it would slow every fit.

    Args:
        ignore: Skip the check.

    Raises:
        RuntimeError: If the load average per core is above `MAX_LOAD_PER_CORE`.
    """
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    _LOG.info("one-minute load average per core: %.2f", load)
    if load > MAX_LOAD_PER_CORE and not ignore:
        msg = f"load per core is {load:.2f}, above {MAX_LOAD_PER_CORE}; stop the other job first"
        raise RuntimeError(msg)


def with_permuted_columns(*, frame: pl.DataFrame, through_rung: RungType) -> pl.DataFrame:
    """Add the negative control's permuted columns, if the frame holds every rung.

    All the permuted columns are one group, so their joint distribution survives the permutation.

    Args:
        frame: The kept rows, carrying `site`, `month`, and `hour_of_day`.
        through_rung: The highest rung the frame holds.

    Returns:
        `frame`, with `<column>_shuffled` for every column of `negative_control_columns`, or
        `frame` unchanged below `g9`.
    """
    if through_rung != RUNGS[-1]:
        return frame
    return climatology_permutation(
        frame=frame,
        column_groups=[negative_control_columns()],
        by=PERMUTATION_GROUPING,
        seed=NEGATIVE_CONTROL_SEED,
        suffix=PERMUTED_SUFFIX,
    )


def target_view(*, frame: pl.DataFrame, target: TargetType) -> pl.DataFrame:
    """Return the frame as the fit loop wants it for one target.

    The CAMS view has a capacity of 1, no export cap, and no constrained hours.

    Args:
        frame: The kept rows with their folds.
        target: The target.

    Returns:
        The frame with `effective_capacity_mw`, `cap_mw`, and `constrained` set for the target.
    """
    if target == "pv":
        return frame
    return frame.with_columns(
        effective_capacity_mw=pl.lit(1.0),
        cap_mw=pl.lit(None, dtype=pl.Float64),
        constrained=pl.lit(value=False),
    )


def jobs_for(
    *, arms: dict[str, tuple[str, ...]], target: TargetType, sensitivity_arms: Sequence[str]
) -> list[Job]:
    """Return the fits to run: every arm at the primary setting, and some at the second.

    Args:
        arms: Each arm's name and columns.
        target: The target the fits predict.
        sensitivity_arms: The arms also fitted at the sensitivity setting.

    Returns:
        One job per (arm, setting).
    """
    jobs: list[Job] = []
    for name, features in arms.items():
        settings = [PRIMARY_SETTING]
        if name in sensitivity_arms:
            settings.append(SENSITIVITY_SETTING)
        jobs.extend(
            (name, setting, TARGET_COLUMNS[target], features, SETTINGS[setting], False)
            for setting in settings
        )
    return jobs


def raise_if_columns_missing(*, frame: pl.DataFrame, arms: dict[str, tuple[str, ...]]) -> None:
    """Raise if an arm is shown a column the frame lacks, or one that is missing on some row.

    Args:
        frame: The rows every arm is scored on.
        arms: Each arm's name and columns.

    Raises:
        ValueError: If a column is absent from the frame.
    """
    needed = {name for features in arms.values() for name in features}
    absent = sorted(needed - set(frame.columns))
    if absent:
        msg = f"the frame lacks {absent}"
        raise ValueError(msg)
    # A derived index is missing where its denominator is too small, so it is allowed too.
    may_be_missing = {*MISSING_UNDER_CLEAR_SKY_VARIABLES, "clear_sky_index", "clearness_index"}
    allowed_missing = may_be_missing | {f"{name}{PERMUTED_SUFFIX}" for name in may_be_missing}
    check_no_missing(frame=frame, columns=sorted(needed - allowed_missing))


def fit_one_view(
    *,
    frame: pl.DataFrame,
    arms: dict[str, tuple[str, ...]],
    target: TargetType,
    sensitivity_arms: Sequence[str],
    device: DeviceType,
    max_workers: int,
) -> pl.DataFrame:
    """Fit every arm of one view for one target and check that the arms share their rows.

    Args:
        frame: The rows with folds, permuted columns, and the target.
        arms: Each arm's name and columns.
        target: The target.
        sensitivity_arms: The arms also fitted at the sensitivity setting.
        device: XGBoost's device.
        max_workers: How many (arm, farm) fits run at once.

    Returns:
        The stacked per-row losses of every arm.
    """
    raise_if_columns_missing(frame=frame, arms=arms)
    view = target_view(frame=frame, target=target)
    losses = run_all(
        dataset=view,
        jobs=jobs_for(arms=arms, target=target, sensitivity_arms=sensitivity_arms),
        max_workers=max_workers,
        device=device,
    )
    for setting in losses["setting"].unique().to_list():
        in_setting = losses.filter(pl.col("setting") == setting)
        raise_unless_same_rows(losses=in_setting, arms=sorted(in_setting["arm"].unique().to_list()))
    return losses


def write_arms(
    *, arms: dict[str, tuple[str, ...]], key: FitKey, device: DeviceType, rows: int
) -> None:
    """Write each arm's columns, the device, and the settings, for the report to print.

    Args:
        arms: Each arm's name and columns.
        key: Which fit the arms belong to.
        device: XGBoost's device.
        rows: The number of rows scored.
    """
    arms_path(key=key).write_text(
        json.dumps(
            {
                "device": device,
                "rows": rows,
                "metric": METRIC,
                "settings": SETTINGS,
                "arms": {name: list(features) for name, features in arms.items()},
            },
            indent=2,
        )
    )


def run_ladder_view(
    *,
    frame: pl.DataFrame,
    variant: str,
    through_rung: RungType,
    only_arms: Sequence[str] | None,
    sensitivity_arms: Sequence[str],
    device: DeviceType,
    max_workers: int,
) -> None:
    """Fit the ladder, controls, and drop-one-group arms on every kept row, for both targets.

    Args:
        frame: The kept rows with folds and permuted columns.
        variant: `main` or `snow_zero_hours`.
        through_rung: The highest rung the frame holds.
        only_arms: The arms to fit, or `None` for all of them.
        sensitivity_arms: The arms also fitted at the sensitivity setting.
        device: XGBoost's device.
        max_workers: How many (arm, farm) fits run at once.
    """
    for target in TARGETS:
        arms = arm_features(target=target, through_rung=through_rung)
        if only_arms is not None:
            arms = {name: features for name, features in arms.items() if name in only_arms}
        key = FitKey(variant=variant, through_rung=through_rung, target=target, view="ladder")
        refuse_to_overwrite(paths=[results_path(key=key), arms_path(key=key)])
        losses = fit_one_view(
            frame=frame,
            arms=arms,
            target=target,
            sensitivity_arms=sensitivity_arms,
            device=device,
            max_workers=max_workers,
        )
        losses.write_parquet(results_path(key=key))
        write_arms(arms=arms, key=key, device=device, rows=frame.height)
        _LOG.info("%s target: wrote %d loss rows", target, losses.height)


def run_aerosol_view(
    *,
    frame: pl.DataFrame,
    variant: str,
    through_rung: RungType,
    sensitivity_arms: Sequence[str],
    device: DeviceType,
    max_workers: int,
) -> None:
    """Fit `g9` and `g10` on the rows that EAC4 aerosol covers, with folds cut on that span.

    Args:
        frame: The kept rows, carrying the aerosol columns.
        variant: `main` or `snow_zero_hours`.
        through_rung: The highest rung the frame holds, which must be the last.
        sensitivity_arms: The arms also fitted at the sensitivity setting.
        device: XGBoost's device.
        max_workers: How many (arm, farm) fits run at once.
    """
    covered = frame.filter(
        pl.all_horizontal(pl.col(name).is_not_null() for name in AEROSOL_COLUMNS)
    )
    covered = assign_folds(dataset=covered.drop("fold"), by=("site",))
    _LOG.info("aerosol view: %d of %d rows are covered by EAC4", covered.height, frame.height)
    for target in TARGETS:
        key = FitKey(variant=variant, through_rung=through_rung, target=target, view="aerosol_rows")
        refuse_to_overwrite(paths=[results_path(key=key), arms_path(key=key)])
        arms = aerosol_arm_features()
        losses = fit_one_view(
            frame=covered,
            arms=arms,
            target=target,
            sensitivity_arms=[AEROSOL_REFERENCE, AEROSOL_RUNG, *sensitivity_arms],
            device=device,
            max_workers=max_workers,
        )
        losses.write_parquet(results_path(key=key))
        write_arms(arms=arms, key=key, device=device, rows=covered.height)


def main() -> int:
    """Fit the arms the command line asks for and write the losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-rung", choices=RUNGS, default=RUNGS[-1])
    parser.add_argument("--variant", choices=("main", "snow_zero_hours"), default="main")
    parser.add_argument("--arms", nargs="*", default=None, help="Fit only these arms.")
    parser.add_argument("--sensitivity-arms", nargs="*", default=list(SENSITIVITY_ARMS))
    parser.add_argument("--aerosol", action="store_true", help="Also run the aerosol view.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--ignore-load", action="store_true")
    arguments = parser.parse_args()

    refuse_if_machine_is_busy(ignore=arguments.ignore_load)
    device = choose_device(requested=arguments.device)
    _LOG.info("fitting on %s", device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    only_arms = arguments.arms
    if arguments.variant == "snow_zero_hours" and only_arms is None:
        only_arms = list(SNOW_VARIANT_ARMS)
    frame = pl.read_parquet(
        dataset_path(through_rung=arguments.through_rung, variant=arguments.variant)
    )
    frame = with_permuted_columns(frame=frame, through_rung=arguments.through_rung)
    frame = assign_folds(dataset=frame, by=("site",))
    _LOG.info("%d rows, %d farms", frame.height, frame["site"].n_unique())

    run_ladder_view(
        frame=frame,
        variant=arguments.variant,
        through_rung=arguments.through_rung,
        only_arms=only_arms,
        sensitivity_arms=arguments.sensitivity_arms,
        device=device,
        max_workers=arguments.max_workers,
    )
    if arguments.aerosol:
        run_aerosol_view(
            frame=frame,
            variant=arguments.variant,
            through_rung=arguments.through_rung,
            sensitivity_arms=arguments.sensitivity_arms,
            device=device,
            max_workers=arguments.max_workers,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
