"""Fit the IFS variable ladder, its controls, and the blend arms, out of fold.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>. It reads the frames
`ifs_ladder_build_dataset.py` wrote and, for each lead day and target, fits one XGBoost model per
arm, farm, fold, and seed with `studies.arm_runner.run_all`, in checkpointed groups
(`studies.checkpointed_fits`). It writes the per-row losses and a JSON naming every arm's columns,
the device, and the hyperparameter settings, under
`data/studies/per_study/ifs_solar_variables/results/`.

**Every arm of one fit is scored on exactly the same rows.** The script raises if two arms of one
setting hold different (farm, time, seed) rows, and it raises if an arm is shown a column that is
missing on any row, except convective inhibition (undefined, so missing by design) and the two
ERA5 variables `cbh` and `cin`, whose missing values are kept as values.

**One fit is one lead day, one target, and one view.** The `ladder` view fits every kept row of a
lead day. The output target (the default) fits all of that lead day's arms. The CAMS target is
exploratory and fits only `CAMS_ARMS`, with a capacity of 1, no export cap, and no constrained
hour. The `blend` view fits the minimal and full IFS arms with and without AIFS Single's radiation
and temperature, on the blend rows of lead days 1 to 3, and their padding controls.

**The hyperparameter settings.** Every arm is fitted at `PRIMARY_HYPER_PARAMETERS`. The arms of
`SENSITIVITY_ARMS` (the arms of the planned contrasts, the controls, and `f3` to `f5`) are also
fitted at `SENSITIVITY_HYPER_PARAMETERS` at the planned lead days. `--sensitivity-arms` adds more.
Column subsampling is off, so an arm with more columns has no advantage from the count.

**The controls' permuted columns are made once per fit**, among the rows that share a farm, a month,
and an hour of day (`studies.blending.climatology_permutation`), and every arm of the fit reads the
same permuted values.

The script refuses to overwrite a losses file. A fit that is killed resumes from its checkpoint
groups when it is run again with the same arguments. Each `.parts` directory carries a stamp file
holding a hash of the dataset file's size and modification time, the arms' column lists, and the
device, and the script raises if a directory's stamp differs from the current one: move the
directory then. `--arms` raises on a name that is not an arm of the fit.

Run it with `uv run python studies/ifs_solar_variables/ifs_ladder_fit.py --lead-days 1 2 3`.
Check that no other XGBoost run is using the CPU first. The script prints the load average and
refuses to start above `studies.checkpointed_fits.MAX_LOAD_PER_CORE` unless `--ignore-load` is
given.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Final

import polars as pl
from ifs_ladder_arms import (
    METRIC,
    PERMUTATION_GROUPING,
    PERMUTATION_SEED,
    PRIMARY_SETTING,
    RESULTS_DIR,
    SENSITIVITY_SETTING,
    TARGET_COLUMNS,
    FitKey,
    TargetType,
    ViewType,
    arm_features,
    arms_path,
    checkpoint_dir_for,
    dataset_path_for,
    fit_keys,
    permutation_groups,
    results_path,
    settings_for,
)
from studies.arm_runner import Job
from studies.blending import PERMUTED_SUFFIX, climatology_permutation
from studies.checkpointed_fits import (
    choose_device,
    claim_checkpoint_dir,
    fit_in_groups,
    refuse_if_machine_is_busy,
    run_stamp,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    DeviceType,
    HyperParameters,
)
from studies.era5_ladder import raise_unless_same_rows
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.ifs_ladder import MISSING_BY_DESIGN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("ifs_ladder_fit")

DEFAULT_WORKERS: Final[int] = 4
"""How many (arm, farm) fits run at once. A GPU fit needs about one CPU core."""

SETTINGS: Final[dict[str, HyperParameters]] = {
    PRIMARY_SETTING: PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_SETTING: SENSITIVITY_HYPER_PARAMETERS,
}
"""The two hyperparameter settings by the name the losses carry."""

MAY_BE_MISSING: Final[frozenset[str]] = frozenset({*MISSING_BY_DESIGN, "era5_cbh", "era5_cin"})
"""The columns whose missing values XGBoost reads as values of their own."""


def load_frame(*, key: FitKey) -> pl.DataFrame:
    """Read a fit's frame and add the controls' permuted columns.

    Args:
        key: Which fit.

    Returns:
        The fit's rows sorted by farm and time, with a permuted copy of every control column.
    """
    frame = pl.read_parquet(dataset_path_for(key=key))
    if key.view == "blend":
        frame = frame.filter(pl.col("lead_day") == key.lead_day)
    return climatology_permutation(
        frame=frame.sort("site", "time"),
        column_groups=permutation_groups(view=key.view),
        by=PERMUTATION_GROUPING,
        seed=PERMUTATION_SEED,
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
    *,
    key: FitKey,
    arms: dict[str, tuple[str, ...]],
    extra_sensitivity_arms: tuple[str, ...],
) -> list[Job]:
    """Return the fits to run: every arm at the primary setting, and some at the second.

    Args:
        key: Which fit.
        arms: Each arm's name and columns.
        extra_sensitivity_arms: Further arms to fit at both settings.

    Returns:
        One job per (arm, setting), none with quantiles.
    """
    return [
        (name, setting, TARGET_COLUMNS[key.target], features, SETTINGS[setting], False)
        for name, features in arms.items()
        for setting in settings_for(
            view=key.view, arm=name, lead_day=key.lead_day, extra=extra_sensitivity_arms
        )
    ]


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
    allowed = MAY_BE_MISSING | {f"{name}{PERMUTED_SUFFIX}" for name in MAY_BE_MISSING}
    check_no_missing(frame=frame, columns=sorted(needed - allowed))


def write_arms(
    *, arms: dict[str, tuple[str, ...]], key: FitKey, device: DeviceType, rows: int, jobs: list[Job]
) -> None:
    """Write each arm's columns, the device, and the settings, for the report to print.

    Args:
        arms: Each arm's name and columns.
        key: Which fit the arms belong to.
        device: XGBoost's device.
        rows: The number of rows scored.
        jobs: The fits that were run.
    """
    settings_by_arm: dict[str, list[str]] = {}
    for name, setting, *_ in jobs:
        settings_by_arm.setdefault(name, []).append(setting)
    arms_path(key=key).write_text(
        json.dumps(
            {
                "device": device,
                "rows": rows,
                "metric": METRIC,
                "settings": SETTINGS,
                "settings_by_arm": settings_by_arm,
                "arms": {name: list(features) for name, features in arms.items()},
            },
            indent=2,
        )
    )


def run_fit(
    *,
    key: FitKey,
    only_arms: Sequence[str] | None,
    extra_sensitivity_arms: tuple[str, ...],
    device: DeviceType,
    max_workers: int,
) -> None:
    """Fit one lead day, target, and view, and write its losses and arms.

    Args:
        key: Which fit.
        only_arms: The arms to fit, or `None` for all of them.
        extra_sensitivity_arms: Further arms to fit at both settings.
        device: XGBoost's device.
        max_workers: How many (arm, farm) fits run at once.

    Raises:
        ValueError: If `only_arms` names an arm this fit does not hold.
        RuntimeError: If the checkpoint directory was written for another dataset file, arm
            columns, or device.
    """
    arms = arm_features(view=key.view, target=key.target, lead_day=key.lead_day)
    if only_arms is not None:
        unknown = sorted(set(only_arms) - set(arms))
        if unknown:
            msg = f"{unknown} are not arms of {key}; its arms are {sorted(arms)}"
            raise ValueError(msg)
        arms = {name: features for name, features in arms.items() if name in only_arms}
    refuse_to_overwrite(paths=[results_path(key=key), arms_path(key=key)])
    frame = load_frame(key=key)
    raise_if_columns_missing(frame=frame, arms=arms)
    view = target_view(frame=frame, target=key.target)
    jobs = jobs_for(key=key, arms=arms, extra_sensitivity_arms=extra_sensitivity_arms)
    _LOG.info("%s: %d rows, %d arms, %d jobs", key, view.height, len(arms), len(jobs))
    checkpoint_dir = checkpoint_dir_for(key=key)
    claim_checkpoint_dir(
        checkpoint_dir=checkpoint_dir,
        stamp=run_stamp(dataset_path=dataset_path_for(key=key), arms=arms, device=device),
    )
    losses = fit_in_groups(
        dataset=view,
        jobs=jobs,
        checkpoint_dir=checkpoint_dir,
        max_workers=max_workers,
        device=device,
    )
    for setting in losses["setting"].unique().to_list():
        in_setting = losses.filter(pl.col("setting") == setting)
        raise_unless_same_rows(losses=in_setting, arms=sorted(in_setting["arm"].unique().to_list()))
    losses.write_parquet(results_path(key=key))
    write_arms(arms=arms, key=key, device=device, rows=view.height, jobs=jobs)
    _LOG.info("%s: wrote %d loss rows", key, losses.height)


def main() -> int:
    """Fit the arms the command line asks for and write the losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-days", type=int, nargs="*", default=None, help="Fit only these.")
    parser.add_argument("--arms", nargs="*", default=None, help="Fit only these arms.")
    parser.add_argument("--view", choices=("ladder", "blend"), default="ladder")
    parser.add_argument(
        "--target", choices=("pv", "cams", "both"), default="both", help="Ladder view only."
    )
    parser.add_argument("--sensitivity-arms", nargs="*", default=[])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--ignore-load", action="store_true")
    arguments = parser.parse_args()

    refuse_if_machine_is_busy(ignore=arguments.ignore_load)
    device = choose_device(requested=arguments.device)
    _LOG.info("fitting on %s", device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    view: ViewType = arguments.view
    keys = [
        key
        for key in fit_keys()
        if key.view == view
        and (arguments.lead_days is None or key.lead_day in arguments.lead_days)
        and arguments.target in ("both", key.target)
    ]
    if not keys:
        msg = "no fit matches --view, --lead-days, and --target"
        raise ValueError(msg)
    extra = (*arguments.sensitivity_arms,)
    for key in keys:
        run_fit(
            key=key,
            only_arms=arguments.arms,
            extra_sensitivity_arms=extra,
            device=device,
            max_workers=arguments.max_workers,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
