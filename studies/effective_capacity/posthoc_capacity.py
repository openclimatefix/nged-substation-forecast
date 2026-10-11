"""POST HOC: estimate DSKF and HYST on every record and score them like the pre-registered ones.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>.
**Nothing here is part of the pre-registered selection or verdict.** `studies.capacity_posthoc`
holds the two estimators, which post-process the Kalman filter's raw level at its tuned setting.

**Phases:**

- `estimates`: for each file of the tuned Kalman filter (the original, the tuning-phase original,
  the 11 draws, the control, and the ANM-blind original), write DSKF and HYST estimates under
  `estimates/<estimator>/default/`, in megawatts. A plant's reference level is the median of its
  deseasonalised level over the reference year of the original record.
- `stage_a`: score both on the 10 selection draws exactly as `score_capacity_recovery.py` does, and
  write `tables/posthoc_stage_a.parquet`, which also holds the false-movement share of every
  estimator on the same real windows.

Run it with `uv run python studies/effective_capacity/posthoc_capacity.py --phase estimates` and
then `--phase stage_a`.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from capacity_paths import (
    ORIGINAL,
    claim_root,
    draw_id,
    estimate_path,
    load_estimate,
    load_events,
    load_tuned,
    record_for,
    save_estimate,
    setting_name,
)
from capacity_scores import scale_for
from score_capacity_recovery import draw_scores
from studies.capacity_estimators import reference_day_indices
from studies.capacity_posthoc import deseasonalise_causally, hysteresis_level
from studies.capacity_recovery import (
    draw_primary_metric,
    fallback_share_on_events,
    false_movement_share,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("posthoc_capacity")

POSTHOC_ESTIMATORS: Final[tuple[str, ...]] = ("DSKF", "HYST")
"""The two post hoc estimators."""

COMPARISON: Final[tuple[str, ...]] = ("S0", "KF", "HM", "CX1", "CX2", "NR", "PR", "DP1")
"""The pre-registered estimators whose false movement the post hoc table repeats."""


def _file_ids(*, root: Path, kf_setting: str) -> list[str]:
    """Return the identifiers of the Kalman filter's files that a post hoc estimate mirrors."""
    folder = estimate_path(root=root, estimator="KF", setting=kf_setting, record_id="x").parent
    names = sorted(path.stem for path in folder.glob("*.npz"))
    return [n for n in names if "reversed" not in n and "hindsight" not in n]


def write_estimates(*, root: Path) -> None:
    """Write DSKF and HYST estimates for every Kalman filter file.

    Args:
        root: The output root.
    """
    kf_params = load_tuned(root=root, estimator="KF")
    kf_setting = setting_name(params=kf_params)
    ids = _file_ids(root=root, kf_setting=kf_setting)
    original = record_for(root=root, record_id=ORIGINAL)
    days = reference_day_indices(record=original)
    dskf_original = deseasonalise_causally(
        raw=load_estimate(
            path=estimate_path(root=root, estimator="KF", setting=kf_setting, record_id=ORIGINAL)
        ),
        dates=original.dates,
    )
    with np.errstate(all="ignore"):
        reference = np.nanmedian(dskf_original.level[:, days], axis=1)
    static = original.static_p99_mw
    for file_id in ids:
        kf = load_estimate(
            path=estimate_path(root=root, estimator="KF", setting=kf_setting, record_id=file_id)
        )
        dskf = deseasonalise_causally(raw=kf, dates=original.dates)
        factor = (static / reference)[:, None]
        save_estimate(
            path=estimate_path(root=root, estimator="DSKF", setting="default", record_id=file_id),
            level=dskf.level * factor,
            sd=None if dskf.sd is None else dskf.sd * factor,
        )
        ratio = dskf.level / reference[:, None]
        held = np.stack([hysteresis_level(ratio=row) for row in ratio]) * static[:, None]
        save_estimate(
            path=estimate_path(root=root, estimator="HYST", setting="default", record_id=file_id),
            level=held,
            sd=None,
        )
        _LOG.info("wrote DSKF and HYST for %s", file_id)


def stage_a(*, root: Path) -> None:
    """Score the post hoc estimators and tabulate false movement beside the pre-registered ones.

    Args:
        root: The output root.
    """
    events = load_events(root=root)
    rows: list[dict[str, object]] = []
    for estimator in (*POSTHOC_ESTIMATORS, *COMPARISON):
        params = (
            {} if estimator in POSTHOC_ESTIMATORS else load_tuned(root=root, estimator=estimator)
        )
        tables = [
            draw_scores(root=root, estimator=estimator, params=params, index=i, tuning=False)
            for i in range(1, len(events.draws))
        ]
        errors = np.array([draw_primary_metric(scores=t) for t in tables])
        original = scale_for(root=root, estimator=estimator, params=params, file_id=ORIGINAL)
        fallback = np.mean(
            [
                fallback_share_on_events(
                    events=events.draws[i],
                    planted=scale_for(
                        root=root, estimator=estimator, params=params, file_id=draw_id(index=i)
                    ),
                )
                for i in range(1, len(events.draws))
            ]
        )
        steps = pl.concat(tables).filter(pl.col("shape") == "step")["delay"].drop_nulls()
        rows.append(
            {
                "estimator": estimator,
                "post_hoc": estimator in POSTHOC_ESTIMATORS,
                "mean_selection_error": float(errors.mean()),
                "false_movement_real_windows": false_movement_share(
                    windows=events.false_movement_windows, original=original
                ),
                "fallback_share_on_event_days": float(fallback),
                "mean_step_detection_delay_days": float(np.mean(steps.to_numpy()))
                if len(steps)
                else None,
            }
        )
    table = pl.DataFrame(rows)
    path = root / "tables" / "posthoc_stage_a.parquet"
    table.write_parquet(path)
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200):
        print(table)
    _LOG.info("wrote %s (windows: %d)", path, len(events.false_movement_windows))


def main() -> int:
    """Run one phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("estimates", "stage_a"), required=True)
    arguments = parser.parse_args()
    root = claim_root(smoke=False)
    if arguments.phase == "estimates":
        write_estimates(root=root)
    else:
        stage_a(root=root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
