"""POST HOC: fit Stage B arms for estimators other than the pre-registered selected one.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>.
**These arms are not part of the pre-registered verdict.** Each arm is the same B0 as
`fit_capacity_arms.py` (same seven columns, folds, seeds, and hyperparameters, primary setting)
with the powers divided by the arm's causal issue-time scale on the original record, named
`posthoc_causal__<estimator>`. They run in this order: DSKF, HYST (see `posthoc_capacity.py`), then
HM, CX2, NR, PR, CX1, and DP1. Losses go to `losses_stage_b_posthoc.parquet` and the checkpoints to
`checkpoints_posthoc/`, so the pre-registered files are untouched.

Run it with `uv run python studies/effective_capacity/fit_posthoc_arms.py --device cuda
--max-workers 4`.
"""

import argparse
import logging
import sys
from typing import Final

import numpy as np
import polars as pl
from capacity_paths import (
    ALL_MODULES,
    LAG_FRAME_PATH,
    ORIGINAL,
    claim_root,
    fingerprint,
    load_tuned,
    record_for,
    sha256_of,
)
from capacity_scores import scale_for
from fit_capacity_arms import B0_COLUMNS, arm_frame
from posthoc_capacity import POSTHOC_ESTIMATORS
from studies.capacity_scaling import assert_arm_covers_frame, drop_scaled_loss_columns
from studies.checkpointed_fits import (
    choose_device,
    claim_checkpoint_dir,
    fit_in_groups,
    run_stamp,
)
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SEEDS, DeviceType
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fit_posthoc_arms")

ARM_ORDER: Final[tuple[str, ...]] = (*POSTHOC_ESTIMATORS, "HM", "CX2", "NR", "PR", "CX1", "DP1")
"""The estimators fitted, in order; later ones run only if time remains."""


def main() -> int:
    """Fit the post hoc arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--estimators", nargs="+", default=list(ARM_ORDER))
    arguments = parser.parse_args()
    root = claim_root(smoke=False)
    device: DeviceType = choose_device(requested=arguments.device)
    frame = pl.read_parquet(LAG_FRAME_PATH)
    record = record_for(root=root, record_id=ORIGINAL)
    scales = {}
    for name in arguments.estimators:
        params = {} if name in POSTHOC_ESTIMATORS else load_tuned(root=root, estimator=name)
        scales[f"posthoc_causal__{name}"] = scale_for(
            root=root, estimator=name, params=params, file_id=ORIGINAL
        ).scale_mw
    np.savez_compressed(root / "scales_stage_b_posthoc.npz", **scales)  # ty: ignore[invalid-argument-type]
    stamp = run_stamp(
        dataset_path=LAG_FRAME_PATH,
        arms={
            **{arm: list(B0_COLUMNS) for arm in scales},
            "scales": [sha256_of(path=root / "scales_stage_b_posthoc.npz")],
            "code": [fingerprint(root=root, modules=ALL_MODULES)],
        },
        device=device,
    )
    checkpoint_root = root / "checkpoints_posthoc"
    claim_checkpoint_dir(checkpoint_dir=checkpoint_root, stamp=stamp)
    final = root / "losses_stage_b_posthoc.parquet"
    refuse_to_overwrite(paths=[final])
    results: list[pl.DataFrame] = []
    for arm, scale in scales.items():
        arm_data = arm_frame(frame=frame, record=record, scale=scale, factor=None)
        losses = fit_in_groups(
            dataset=arm_data,
            jobs=[(arm, "primary", "power_mw", B0_COLUMNS, PRIMARY_HYPER_PARAMETERS, False)],
            checkpoint_dir=checkpoint_root / f"{arm}__primary",
            max_workers=arguments.max_workers,
            device=device,
        )
        fractions = losses.with_columns(
            crps_capped_fraction_of_capacity=pl.col("crps_capped_mw")
            / pl.col("effective_capacity_mw")
        )
        annotated = drop_scaled_loss_columns(losses=fractions).join(
            arm_data.select("site", "time", "capacity_scale_mw", actual_scaled=pl.col("power_mw")),
            on=["site", "time"],
        )
        assert_arm_covers_frame(losses=annotated, frame=arm_data, seeds=SEEDS)
        results.append(annotated)
        _LOG.info("fitted %s", arm)
    pl.concat(results, how="diagonal_relaxed").write_parquet(final)
    _LOG.info("wrote %s", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
