"""Score every estimator on the planted losses, and choose the estimator by the fixed rule.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>. Stage
A answers "which technique?" from how well each estimator recovers a planted loss, with no forecast
in the loop. This script fits no XGBoost model and runs before any is fitted.

**Phases:**

- `tuning`: for each estimator, score every setting on the tuning draw (`draw00`) and write the
  lowest-error setting to `tuned/tuned_<estimator>.json`. Run it after `estimate_capacity.py --phase
  tuning` (and after `estimate_capacity_torch.py`, for DP1). Never read by a forecast.
- `selection`: score the tuned setting of every estimator on the 10 selection draws, apply the
  false-movement gate and the selection rule (`studies.capacity_recovery.choose_estimators`), and
  write `selection.json`, `tables/stage_a_events.parquet`, and `tables/stage_a_summary.parquet`.
  Run it after `estimate_capacity.py --phase main`.

Run it with `uv run python studies/effective_capacity/score_capacity_recovery.py --phase selection`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from capacity_paths import (
    ESTIMATOR_MODULES,
    ORIGINAL,
    TUNING_GRIDS,
    check_stamp,
    claim_root,
    draw_id,
    fingerprint,
    inputs_dir,
    load_events,
    load_tuned,
    setting_name,
    tuned_path,
    write_json_atomic,
)
from capacity_scores import cached_record, load_raw, original_file_id, scale_for
from studies.capacity_estimators import ESTIMATOR_FAMILY, ISSUE_LAG_DAYS
from studies.capacity_recovery import (
    SelectionInputs,
    choose_estimators,
    draw_primary_metric,
    fallback_share_on_events,
    false_movement_share,
    score_events,
    write_selection,
)
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("score_capacity_recovery")

SATELLITE_SENSITIVITY: Final[tuple[str, ...]] = ("PR", "KF", "CX2")
"""The estimators that read satellite irradiance, whose metric is recomputed at read lag 3."""

SATELLITE_SENSITIVITY_LAG_DAYS: Final[int] = 3


def draw_scores(
    *,
    root: Path,
    estimator: str,
    params: dict[str, float],
    index: int,
    tuning: bool,
    lag_days: int = ISSUE_LAG_DAYS,
) -> pl.DataFrame:
    """Score one estimator at one setting on one draw.

    Args:
        root: The output root.
        estimator: The estimator.
        params: Its settings.
        index: The draw (0 for the tuning draw).
        tuning: Whether the reference constants come from the tuning phase's original file.
        lag_days: The read lag.

    Returns:
        `score_events`' table with a `draw` column.
    """
    events = load_events(root=root).draws[index]
    planted = scale_for(
        root=root,
        estimator=estimator,
        params=params,
        file_id=draw_id(index=index),
        tuning=tuning,
        lag_days=lag_days,
    )
    original = scale_for(
        root=root,
        estimator=estimator,
        params=params,
        file_id=original_file_id(estimator=estimator, tuning=tuning),
        tuning=tuning,
        lag_days=lag_days,
    )
    days = planted.scale_mw.shape[1]
    return score_events(events=events, planted=planted, original=original, days=days).with_columns(
        draw=pl.lit(index, dtype=pl.Int32)
    )


def tune(*, root: Path, estimators: list[str]) -> None:
    """Choose each estimator's setting on the tuning draw and write it.

    Args:
        root: The output root.
        estimators: The estimators to tune.
    """
    for estimator in estimators:
        path = tuned_path(root=root, estimator=estimator)
        if path.exists():
            _LOG.info("%s is already tuned; leaving %s alone", estimator, path.name)
            continue
        errors = {}
        for params in TUNING_GRIDS[estimator]:
            scores = draw_scores(
                root=root, estimator=estimator, params=params, index=0, tuning=True
            )
            errors[setting_name(params=params)] = draw_primary_metric(scores=scores)
        best = min(TUNING_GRIDS[estimator], key=lambda p: errors[setting_name(params=p)])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            path=path, payload={"params": best, "tuning_draw_error_by_setting": errors}
        )
        _LOG.info("%s: chose %s of %s", estimator, setting_name(params=best), errors)


FITTED_ESTIMATORS: Final[tuple[str, ...]] = ("KF", "DP1")
"""The estimators whose fitted physical quantities the explainability report prints."""


def fitted_quantities(
    *, root: Path, estimator: str, params: dict[str, float]
) -> list[dict[str, object]]:
    """Return one row per plant of an estimator's fitted physical quantities, with units.

    The tilt, azimuth, and alternating-current limit are the plant's orientation fit, which KF and
    DP1 both take as given. The capacity series is on the original record: for DP1 the weekly
    direct-current capacity in megawatts, and for KF the effective capacity (the calibrated scale)
    in megawatts. The summary is over the days on which the estimate exists.

    Args:
        root: The output root.
        estimator: The estimator.
        params: Its tuned settings.

    Returns:
        Nothing for an estimator that does not report fitted quantities.
    """
    if estimator not in FITTED_ESTIMATORS:
        return []
    orientation = json.loads((inputs_dir(root=root) / "orientation.json").read_text())
    record = cached_record(root=root, record_id=ORIGINAL)
    if estimator == "DP1":
        raw = load_raw(root=root, estimator=estimator, params=params, file_id=ORIGINAL)
        series = raw.level * record.static_p99_mw[:, None]
        name = "weekly direct-current capacity (MW)"
    else:
        series = scale_for(root=root, estimator=estimator, params=params, file_id=ORIGINAL).scale_mw
        name = "effective capacity (MW)"
    rows = []
    for plant, site in enumerate(record.sites):
        fitted = orientation[site]
        values = series[plant][np.isfinite(series[plant])]
        rows.append(
            {
                "estimator": estimator,
                "plant": site,
                "tilt_deg": fitted["tilt_deg"],
                "azimuth_deg": fitted["azimuth_deg"],
                "ac_limit_mw": fitted["ac_capacity_mw"],
                "fitted_dc_capacity_mw": fitted["dc_capacity_mw"],
                "series": name,
                "series_min_mw": float(values.min()),
                "series_median_mw": float(np.median(values)),
                "series_max_mw": float(values.max()),
                "series_days": len(values),
            }
        )
    return rows


def select(*, root: Path) -> None:
    """Score the tuned settings on the selection draws and choose the estimator.

    Args:
        root: The output root.
    """
    events = load_events(root=root)
    windows = events.false_movement_windows
    estimators = list(ESTIMATOR_FAMILY)
    outputs = [root / "selection.json"]
    refuse_to_overwrite(paths=outputs)
    tuning_error: dict[str, float] = {}
    selection_errors: dict[str, np.ndarray] = {}
    movement: dict[str, float] = {}
    fallback: dict[str, float] = {}
    event_tables: list[pl.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    fitted_rows: list[dict[str, object]] = []
    for estimator in estimators:
        params = load_tuned(root=root, estimator=estimator)
        tuning_error[estimator] = draw_primary_metric(
            scores=draw_scores(root=root, estimator=estimator, params=params, index=0, tuning=True)
        )
        tables = [
            draw_scores(root=root, estimator=estimator, params=params, index=i, tuning=False)
            for i in range(1, len(events.draws))
        ]
        selection_errors[estimator] = np.array([draw_primary_metric(scores=t) for t in tables])
        original = scale_for(root=root, estimator=estimator, params=params, file_id=ORIGINAL)
        movement[estimator] = false_movement_share(windows=windows, original=original)
        fallback[estimator] = float(
            np.mean(
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
        )
        event_tables += [t.with_columns(estimator=pl.lit(estimator)) for t in tables]
        row: dict[str, object] = {
            "estimator": estimator,
            "family": ESTIMATOR_FAMILY[estimator],
            "setting": setting_name(params=params),
            "tuning_error": tuning_error[estimator],
            "mean_selection_error": float(selection_errors[estimator].mean()),
            "false_movement": movement[estimator],
            "fallback_share_on_event_days": fallback[estimator],
        }
        steps = pl.concat(tables).filter(pl.col("shape") == "step")
        row["steps_detected_share"] = float(np.mean(steps["delay"].is_not_null().to_numpy()))
        row["mean_step_detection_delay_days"] = (
            steps["delay"].drop_nulls().mean()
            if steps["delay"].null_count() < steps.height
            else None
        )
        fitted_rows += fitted_quantities(root=root, estimator=estimator, params=params)
        if estimator in SATELLITE_SENSITIVITY:
            sensitivity = [
                draw_primary_metric(
                    scores=draw_scores(
                        root=root,
                        estimator=estimator,
                        params=params,
                        index=i,
                        tuning=False,
                        lag_days=SATELLITE_SENSITIVITY_LAG_DAYS,
                    )
                )
                for i in range(1, len(events.draws))
            ]
            row["mean_selection_error_irradiance_one_day_older"] = float(np.mean(sensitivity))
        summary_rows.append(row)
        _LOG.info("%s: %s", estimator, row)
    inputs = SelectionInputs(
        tuning_error=tuning_error,
        selection_errors=selection_errors,
        false_movement=movement,
        fallback_share=fallback,
    )
    selection = choose_estimators(inputs=inputs)
    tables_dir = root / "tables"
    tables_dir.mkdir(exist_ok=True)
    pl.concat(event_tables).write_parquet(tables_dir / "stage_a_events.parquet")
    pl.DataFrame(summary_rows).write_parquet(tables_dir / "stage_a_summary.parquet")
    pl.DataFrame(fitted_rows).write_parquet(tables_dir / "fitted_quantities.parquet")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, fmt_str_lengths=80):
        print(pl.DataFrame(summary_rows))
        print(pl.DataFrame(fitted_rows))
    write_selection(path=outputs[0], selection=selection, inputs=inputs)
    _LOG.info(
        "selected %s (status %s); shortlist %s",
        selection.selected,
        selection.status,
        selection.shortlist,
    )


def main() -> int:
    """Run one phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tuning", "selection"), required=True)
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=[e for e in ESTIMATOR_FAMILY if e != "DP1"],
        help="The estimators to tune (tuning phase only).",
    )
    parser.add_argument("--smoke", action="store_true", help="Read and write under _smoke/.")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    check_stamp(directory=root / "tuned", stamp=fingerprint(root=root, modules=ESTIMATOR_MODULES))
    if arguments.phase == "tuning":
        tune(root=root, estimators=arguments.estimators)
    else:
        select(root=root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
