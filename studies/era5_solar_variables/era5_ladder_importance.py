"""Refit four ERA5 ladder arms and save each column's share of XGBoost's total gain.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. `studies.cross_validation`'s
fit loop does not return its boosters, so this script refits `g0`, `g2`, `g9`, and `g9` with
shuffled copies of the `g3` to `g9` columns, with the same frame, folds, seeds, training rows, and
primary hyperparameter setting as `era5_ladder_fit.py`, and reads each booster's gains.

**Importance is descriptive.** Gain is measured on the training data and splits credit between
correlated columns according to which the greedy split search picks first. The planned contrasts
decide whether a variable helps; this script only shows what the fitted models used.

**The shuffled copies set a noise reference.** The fourth arm shows `g9` and a copy of every `g3` to
`g9` column shuffled over all rows of a farm, so each copy keeps its column's distribution and
carries no information about the hour. A real column's share has to stand clear of the largest
shuffled copy's share before the page reads anything into it. This arm is never scored in a planned
contrast, because the extra columns change the trees.

It writes one parquet of shares by (target, arm, farm, fold, seed, column) to
`data/studies/per_study/era5_solar_variables/results/`. A column the booster never split on has a
share of 0, so a mean over folds counts it as 0 rather than leaving it out.

Run it with `uv run python studies/era5_solar_variables/era5_ladder_importance.py --through-rung g9`
after asking the study coordinator for a GPU slot.
"""

import argparse
import logging
import sys
from typing import Final, cast

import polars as pl
import xgboost as xgb
from era5_ladder_arms import (
    NEGATIVE_CONTROL_SEED,
    PRIMARY_SETTING,
    RESULTS_DIR,
    TARGET_COLUMNS,
    TARGETS,
    TargetType,
    dataset_path,
    importance_path,
)
from era5_ladder_fit import (
    SETTINGS,
    choose_device,
    refuse_if_machine_is_busy,
    target_view,
)
from studies.blending import climatology_permutation
from studies.cross_validation import SEEDS, DeviceType, assign_folds, booster_parameters
from studies.era5_ladder import (
    RUNGS,
    RungType,
    gain_shares,
    negative_control_columns,
    rung_features,
)
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("era5_ladder_importance")

SHUFFLED_SUFFIX: Final[str] = "_noise"
"""What the shuffle appends to a column's name. It differs from the negative control's suffix."""

SHUFFLED_ARM: Final[str] = "g9_with_shuffled"
"""The arm of `g9` and a shuffled copy of every `g3` to `g9` column."""

IMPORTANCE_ARMS: Final[tuple[str, ...]] = ("g0", "g2", "g9", SHUFFLED_ARM)
"""The arms refitted for importance."""

SHUFFLE_SEED: Final[int] = NEGATIVE_CONTROL_SEED + 1
"""The seed of the shuffle, different from the negative control's so the two never coincide."""


def arm_columns(*, through_rung: RungType) -> dict[str, tuple[str, ...]]:
    """Return the columns each importance arm is shown.

    Args:
        through_rung: The highest rung the frame holds, which must be `g9`.

    Returns:
        The columns by arm name.

    Raises:
        ValueError: If the frame does not hold every rung.
    """
    if through_rung != RUNGS[-1]:
        msg = f"importance is fitted on the full frame (through {RUNGS[-1]}), not {through_rung}"
        raise ValueError(msg)
    shuffled = tuple(f"{name}{SHUFFLED_SUFFIX}" for name in negative_control_columns())
    return {
        "g0": rung_features(rung="g0"),
        "g2": rung_features(rung="g2"),
        "g9": rung_features(rung="g9"),
        SHUFFLED_ARM: (*rung_features(rung="g9"), *shuffled),
    }


def fit_shares(
    *,
    train: pl.DataFrame,
    features: tuple[str, ...],
    target: str,
    seed: int,
    device: DeviceType,
) -> dict[str, float]:
    """Fit the point model on a fold's training rows and return each column's share of gain.

    Args:
        train: The training rows.
        features: The columns to show the model.
        target: The column to predict.
        seed: The XGBoost seed.
        device: XGBoost's device.

    Returns:
        Each column's share of the total gain, with names as keys.
    """
    hyper_parameters = SETTINGS[PRIMARY_SETTING]
    matrix = xgb.DMatrix(
        train.select(features).to_numpy(),
        label=train[target].to_numpy(),
        feature_names=list(features),
    )
    booster = xgb.train(
        {
            **booster_parameters(hyper_parameters=hyper_parameters, seed=seed, device=device),
            "objective": "reg:absoluteerror",
        },
        matrix,
        num_boost_round=hyper_parameters["num_boost_round"],
    )
    # A single-output model reports one float per column, never a list.
    total_gain = cast("dict[str, float]", booster.get_score(importance_type="total_gain"))
    return gain_shares(total_gain=total_gain, features=features)


def importance_for_target(
    *, frame: pl.DataFrame, target: TargetType, through_rung: RungType, device: DeviceType
) -> pl.DataFrame:
    """Refit every importance arm for one target and return the shares by farm, fold, and seed.

    Args:
        frame: The kept rows with folds and the shuffled columns.
        target: The target.
        through_rung: The highest rung the frame holds.
        device: XGBoost's device.

    Returns:
        One row per (arm, farm, fold, seed, column) with the column's `share`.
    """
    view = target_view(frame=frame, target=target)
    records: list[dict[str, object]] = []
    for arm, features in arm_columns(through_rung=through_rung).items():
        for site in sorted(view["site"].unique().to_list()):
            site_rows = view.filter(pl.col("site") == site)
            for fold in sorted(site_rows["fold"].unique().to_list()):
                # A constrained hour is left out of training, as in the ladder fit.
                train = site_rows.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
                if train.is_empty():
                    continue
                for seed in SEEDS:
                    shares = fit_shares(
                        train=train,
                        features=features,
                        target=TARGET_COLUMNS[target],
                        seed=seed,
                        device=device,
                    )
                    records.extend(
                        {
                            "target": target,
                            "arm": arm,
                            "site": site,
                            "fold": fold,
                            "seed": seed,
                            "column": name,
                            "share": share,
                        }
                        for name, share in shares.items()
                    )
            _LOG.info("%s target, arm %s, farm %s done", target, arm, site)
    return pl.DataFrame(records)


def main() -> int:
    """Refit the importance arms and write the shares."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-rung", choices=RUNGS, default=RUNGS[-1])
    parser.add_argument("--variant", choices=("main", "snow_zero_hours"), default="main")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ignore-load", action="store_true")
    arguments = parser.parse_args()

    refuse_if_machine_is_busy(ignore=arguments.ignore_load)
    device = choose_device(requested=arguments.device)
    output = importance_path(variant=arguments.variant, through_rung=arguments.through_rung)
    refuse_to_overwrite(paths=[output])
    columns_by_arm = arm_columns(through_rung=arguments.through_rung)

    frame = pl.read_parquet(
        dataset_path(through_rung=arguments.through_rung, variant=arguments.variant)
    )
    # The shuffle runs over all rows of a farm, so a shuffled copy keeps its column's distribution.
    frame = climatology_permutation(
        frame=frame,
        column_groups=[negative_control_columns()],
        by=("site",),
        seed=SHUFFLE_SEED,
        suffix=SHUFFLED_SUFFIX,
    )
    frame = assign_folds(dataset=frame, by=("site",))
    missing = {name for features in columns_by_arm.values() for name in features} - set(
        frame.columns
    )
    if missing:
        msg = f"the frame lacks columns the importance arms need: {sorted(missing)}"
        raise ValueError(msg)
    _LOG.info("fitting %d arms on %s, %d rows", len(columns_by_arm), device, frame.height)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shares = pl.concat(
        importance_for_target(
            frame=frame, target=target, through_rung=arguments.through_rung, device=device
        )
        for target in TARGETS
    )
    shares.write_parquet(output)
    _LOG.info("wrote %d rows to %s", shares.height, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
