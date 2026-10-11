"""Fit the forecast arms of Stage B: B0 divided by a capacity scale, and the controls.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>. Every
arm is the lag study's baseline B0 (seven columns, the absolute-error objective, no column
subsampling, three seeds, five folds of whole months) with `power_mw`, `cap_mw`, and
`effective_capacity_mw` divided by the arm's scale, so that every arm is scored in megawatts over
the plant's fixed 99th-percentile power (`studies.capacity_scaling`).

**Checks that raise before the first scored fit**, in this order: the constant-scale assertion (B0
divided by one constant reproduces B0, at the ANM plant on a fold with constrained rows); the refit
of B0 reproducing the lag study's B0 losses bit for bit; the forecast-row join probe (the scale of
sampled rows does not change when every input from the issue day onward is noise); and a scored
month with no row.

**Arms** (primary setting, point model, unless stated): `causal__<E>` for each shortlisted
estimator (the selected one also at the sensitivity setting and with quantiles); `hindsight__<E>`
for each; `nc2__<selected>` (the negative control: whole months of the scale deranged within each
plant); `noDOY__<selected>` (without day of year); and the positive control's `control_b0`,
`control_oracle`, and `control_selected`, fitted on a frame whose target carries a 30% planted loss.
B0 itself is reused from the lag study. The scales are saved to `scales_stage_b.npz` and the
columns of every arm to `arm_columns.json`.

`--smoke` fits 2 plants on 20 sampled rows per plant-month with 20 boosting rounds and writes under
`_smoke/`; it is not a result. Run it with
`uv run python studies/effective_capacity/fit_capacity_arms.py --device cuda --max-workers 4`.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from capacity_paths import (
    CONTROL,
    DERANGEMENT_SEED,
    LAG_FRAME_PATH,
    LAG_LOSSES_PATH,
    ORIGINAL,
    claim_root,
    inputs_dir,
    load_events,
    load_tuned,
    record_for,
    write_json_atomic,
)
from capacity_scores import load_raw, reference_for, scale_for
from studies.arm_runner import Job
from studies.capacity_convex import estimate_convex_plant
from studies.capacity_estimators import (
    ISSUE_LAG_DAYS,
    CapacityRecord,
    RawEstimate,
    ScaleFunction,
    assert_scale_ignores_the_issue_day_onward,
    hindsight_reference_estimates,
    issue_time_estimates,
    run_cheap_estimator,
    scale_for_rows,
    target_day_indices,
)
from studies.capacity_recovery import factor_matrix
from studies.capacity_scaling import (
    assert_constant_scale_reproduces_baseline,
    assert_refit_reproduces,
    derange_months,
    deranged_scale,
    drop_scaled_loss_columns,
    oracle_scale,
    scaled_frame,
)
from studies.checkpointed_fits import (
    choose_device,
    claim_checkpoint_dir,
    fit_in_groups,
    run_stamp,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    DeviceType,
    HyperParameters,
)
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fit_capacity_arms")

B0_COLUMNS: Final[tuple[str, ...]] = (
    "hour_of_day",
    "day_of_year",
    "era_code",
    "solar_elevation_deg",
    "solar_azimuth_deg",
    "nwp_ghi",
    "nwp_temp",
)
"""B0's seven columns, as the lag study defines them."""

NO_DAY_OF_YEAR_COLUMNS: Final[tuple[str, ...]] = tuple(c for c in B0_COLUMNS if c != "day_of_year")
"""B0's columns without day of year."""

CONSTANT_SCALE: Final[float] = 0.37
CONSTANT_SCALE_TOLERANCE_POINTS: Final[float] = 1e-4
"""The constant divisor of the constant-scale assertion, and the tolerance on the mean error."""

JOIN_PROBE_ROWS: Final[int] = 6
"""How many target days the forecast-row join probe perturbs."""

SMOKE_ROWS_PER_PLANT_MONTH: Final[int] = 20
SMOKE_ROUNDS: Final[int] = 20
SMOKE_PLANTS: Final[tuple[str, ...]] = ("A", "E")
"""A smoke run's sample per plant-month, boosting rounds, and plants."""


def smoke_frame(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Sample a small frame for a smoke run, keeping every constrained row of the ANM plant.

    Args:
        frame: The lag frame.

    Returns:
        Rows of the smoke plants: 20 seeded rows per plant-month plus the constrained rows.
    """
    mine = frame.filter(pl.col("site").is_in(list(SMOKE_PLANTS)))
    sampled = (
        mine.with_row_index("_row")
        .sort("_row")
        .group_by("site", "month", maintain_order=True)
        .agg(pl.col("_row").shuffle(seed=1138).head(SMOKE_ROWS_PER_PLANT_MONTH))
        .explode("_row")["_row"]
    )
    keep = pl.col("_row").is_in(sampled.to_list()) | pl.col("constrained")
    return mine.with_row_index("_row").filter(keep).drop("_row")


def dp1_raw(*, root: Path, record: CapacityRecord, params: dict[str, float]) -> RawEstimate:
    """Run DP1 on one record, with the orientation fits from the build.

    Args:
        root: The output root.
        record: The record.
        params: DP1's settings.

    Returns:
        The raw estimate.
    """
    from studies.capacity_physics_torch import run_dp1

    orientation = json.loads((inputs_dir(root=root) / "orientation.json").read_text())
    dc = np.array([orientation[s]["dc_capacity_mw"] for s in record.sites]) / record.static_p99_mw
    ac = np.array([orientation[s]["ac_capacity_mw"] for s in record.sites]) / record.static_p99_mw
    return run_dp1(records=[record], dc_prior=dc, ac_limit=ac, params=params)[0]


def scale_function_for(
    *, root: Path, estimator: str, params: dict[str, float], reference: np.ndarray | None
) -> ScaleFunction:
    """Return a function that recomputes an estimator's issue-time scale on any record.

    Args:
        root: The output root.
        estimator: The estimator.
        params: Its settings.
        reference: The reference-year constants, held fixed.

    Returns:
        A function from a record and target-day indices to the scale of every plant on those days.
    """

    def scale_function(record: CapacityRecord, days: np.ndarray) -> np.ndarray:
        if estimator in ("CX1", "CX2"):
            level = np.stack(
                [
                    estimate_convex_plant(
                        name=estimator,  # type: ignore[arg-type]
                        record=record,
                        plant=plant,
                        params=params,
                        days=days - ISSUE_LAG_DAYS,
                    ).level
                    for plant in range(len(record.sites))
                ]
            )
            raw = RawEstimate(level=level, sd=None)
        elif estimator == "DP1":
            raw = dp1_raw(root=root, record=record, params=params)
        else:
            raw = run_cheap_estimator(name=estimator, record=record, params=params)
        return issue_time_estimates(record=record, raw=raw, reference=reference).scale_mw[:, days]

    return scale_function


def build_scales(
    *, root: Path, selection: dict[str, Any], record: CapacityRecord
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return each arm's scale and fallback reason, shape (plants, days).

    Args:
        root: The output root.
        selection: `selection.json`.
        record: The original record.

    Returns:
        A mapping from arm name to (scale in megawatts, fallback reason).
    """
    events = load_events(root=root)
    selected = str(selection["selected"])
    shortlist = [str(name) for name in selection["shortlist"]]  # type: ignore[attr-defined]
    scales: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in shortlist:
        params = load_tuned(root=root, estimator=name)
        causal = scale_for(root=root, estimator=name, params=params, file_id=ORIGINAL)
        scales[f"causal__{name}"] = (causal.scale_mw, causal.fallback_reason)
        forward_id = ORIGINAL if name == "DP1" else "hindsight_forward"
        reference = reference_for(root=root, estimator=name, params=params, tuning=False)
        hindsight = hindsight_reference_estimates(
            record=record,
            forward=load_raw(root=root, estimator=name, params=params, file_id=forward_id),
            backward_on_reversed=load_raw(
                root=root, estimator=name, params=params, file_id=f"{ORIGINAL}__reversed"
            ),
            reference=reference,
        )
        scales[f"hindsight__{name}"] = (hindsight.scale_mw, hindsight.fallback_reason)
    sel_scale, sel_reason = scales[f"causal__{selected}"]
    months = {plant: list(events.scored_months) for plant in range(len(record.sites))}
    donors = derange_months(
        months_by_plant=months,
        low_months={events.low_plant: events.low_months},
        seed=DERANGEMENT_SEED,
    )
    scales[f"nc2__{selected}"] = (
        deranged_scale(scale_mw=sel_scale, dates=record.dates, donors=donors),
        sel_reason,
    )
    scales[f"noDOY__{selected}"] = (sel_scale, sel_reason)
    factor = factor_matrix(events=events.control, plants=len(record.sites), days=len(record.dates))
    zero = np.zeros_like(sel_reason)
    scales["control_b0"] = (
        np.broadcast_to(record.static_p99_mw[:, None], sel_scale.shape).copy(),
        zero,
    )
    scales["control_oracle"] = (
        oracle_scale(static_p99_mw=record.static_p99_mw, factor=factor),
        zero,
    )
    params = load_tuned(root=root, estimator=selected)
    control = scale_for(root=root, estimator=selected, params=params, file_id=CONTROL)
    scales["control_selected"] = (control.scale_mw, control.fallback_reason)
    return scales


def arm_frame(
    *, frame: pl.DataFrame, record: CapacityRecord, scale: np.ndarray, factor: np.ndarray | None
) -> pl.DataFrame:
    """Return the lag frame divided by an arm's scale, with a planted target if given.

    Args:
        frame: The lag frame.
        record: The original record (for site and day indices).
        scale: The arm's scale, shape (plants, days).
        factor: The planted share of power left, shape (plants, days), or `None`.

    Returns:
        The frame, with `power_mw`, `cap_mw`, and `effective_capacity_mw` divided by the row's
        scale.
    """
    base = frame
    if factor is not None:
        plants, days = target_day_indices(record=record, rows=frame)
        multiplier = factor[plants, days]
        base = frame.with_columns(
            power_mw=(pl.col("power_mw").cast(pl.Float64) * pl.Series(multiplier)).cast(pl.Float32)
        )
    return scaled_frame(
        frame=base, scale_mw=scale_for_rows(record=record, scale_mw=scale, rows=frame)
    )


def check_months_covered(*, frame: pl.DataFrame, scored_months: list[str], smoke: bool) -> None:
    """Raise if a scored month has no row at a plant in the frame.

    Args:
        frame: The lag frame.
        scored_months: The 18 scored months.
        smoke: Whether the frame is a smoke sample, where only the month list is checked.

    Raises:
        ValueError: If a scored month is missing from the frame, or from a plant's rows.
    """
    held = set(frame["month"].unique().to_list())
    if held != set(scored_months):
        msg = (
            f"the frame's months {sorted(held ^ set(scored_months))} differ from the scored months"
        )
        raise ValueError(msg)
    if not smoke:
        counts = frame.group_by("site").agg(months=pl.col("month").n_unique())
        if counts["months"].to_numpy().min() < len(scored_months):
            msg = "a plant has no rows in some scored month"
            raise ValueError(msg)


def main() -> int:  # noqa: PLR0915
    """Run Stage B."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Tiny run; not a result.")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    device: DeviceType = "cpu" if arguments.smoke else choose_device(requested=arguments.device)
    selection = json.loads((root / "selection.json").read_text())
    selected = str(selection["selected"])
    events = load_events(root=root)
    record = record_for(root=root, record_id=ORIGINAL)
    frame = pl.read_parquet(LAG_FRAME_PATH)
    if arguments.smoke:
        frame = smoke_frame(frame=frame)
    check_months_covered(frame=frame, scored_months=events.scored_months, smoke=arguments.smoke)
    primary: HyperParameters = (
        {**PRIMARY_HYPER_PARAMETERS, "num_boost_round": SMOKE_ROUNDS}
        if arguments.smoke
        else PRIMARY_HYPER_PARAMETERS
    )
    sensitivity: HyperParameters = (
        {**SENSITIVITY_HYPER_PARAMETERS, "num_boost_round": SMOKE_ROUNDS}
        if arguments.smoke
        else SENSITIVITY_HYPER_PARAMETERS
    )
    anm_site = record.sites[events.anm_plant]
    # 1. The constant-scale assertion, before any other fit.
    worst = assert_constant_scale_reproduces_baseline(
        site_rows=frame.filter(pl.col("site") == anm_site),
        features=B0_COLUMNS,
        hyper_parameters=primary,
        scale=CONSTANT_SCALE,
        tolerance_points=CONSTANT_SCALE_TOLERANCE_POINTS,
        device=device,
    )
    _LOG.info("constant-scale assertion passed (largest difference %.2e points)", worst)
    scales = build_scales(root=root, selection=selection, record=record)
    reference = reference_for(
        root=root,
        estimator=selected,
        params=load_tuned(root=root, estimator=selected),
        tuning=False,
    )
    # 3. The forecast-row join probe on the selected estimator's causal scale.
    probe_days = np.unique(target_day_indices(record=record, rows=frame)[1])[
        :: max(1, len(frame) // 2000)
    ]
    assert_scale_ignores_the_issue_day_onward(
        record=record,
        scale_function=scale_function_for(
            root=root,
            estimator=selected,
            params=load_tuned(root=root, estimator=selected),
            reference=reference,
        ),
        target_days=np.random.default_rng(1143).choice(probe_days, JOIN_PROBE_ROWS, replace=False),
        seed=11,
    )
    _LOG.info("join probe passed")
    columns = {arm: list(B0_COLUMNS) for arm in scales} | {"b0_refit": list(B0_COLUMNS)}
    columns[f"noDOY__{selected}"] = list(NO_DAY_OF_YEAR_COLUMNS)
    write_json_atomic(path=root / "arm_columns.json", payload=columns)
    arrays: dict[str, np.ndarray] = {}
    for arm, (scale_array, reason_array) in scales.items():
        arrays[f"{arm}__scale"] = scale_array
        arrays[f"{arm}__reason"] = reason_array
    np.savez_compressed(root / "scales_stage_b.npz", **arrays)  # ty: ignore[invalid-argument-type]
    checkpoint_root = root / "checkpoints"
    stamp = run_stamp(
        dataset_path=LAG_FRAME_PATH,
        arms={
            **columns,
            "selection": [hashlib.sha256((root / "selection.json").read_bytes()).hexdigest()],
        },
        device=device,
    )
    claim_checkpoint_dir(checkpoint_dir=checkpoint_root, stamp=stamp)
    final = root / "losses_stage_b.parquet"
    refuse_to_overwrite(paths=[final])
    # Arms: (name, features, settings to fit, planted?).
    frames: dict[str, pl.DataFrame] = {}
    factor = factor_matrix(events=events.control, plants=len(record.sites), days=len(record.dates))
    ones = np.ones_like(scales["control_b0"][0])
    frames["b0_refit"] = arm_frame(frame=frame, record=record, scale=ones, factor=None)
    specs: list[tuple[str, tuple[str, ...], list[tuple[str, HyperParameters, bool]], bool]] = []
    shortlist = [str(name) for name in selection["shortlist"]]
    specs.append(("b0_refit", B0_COLUMNS, [("primary", primary, False)], False))
    for name in shortlist:
        settings = [("primary", primary, name == selected)]
        if name == selected:
            settings.append(("sensitivity", sensitivity, False))
        specs.append((f"causal__{name}", B0_COLUMNS, settings, False))
        specs.append((f"hindsight__{name}", B0_COLUMNS, [("primary", primary, False)], False))
    specs.append((f"nc2__{selected}", B0_COLUMNS, [("primary", primary, False)], False))
    specs.append(
        (f"noDOY__{selected}", NO_DAY_OF_YEAR_COLUMNS, [("primary", primary, False)], False)
    )
    specs.extend(
        (arm, B0_COLUMNS, [("primary", primary, False)], True)
        for arm in ("control_b0", "control_oracle", "control_selected")
    )
    results: list[pl.DataFrame] = []
    for arm, features, settings, planted in specs:
        if arm not in frames:
            frames[arm] = arm_frame(
                frame=frame, record=record, scale=scales[arm][0], factor=factor if planted else None
            )
        for setting, hyper_parameters, quantiles in settings:
            job: Job = (arm, setting, "power_mw", features, hyper_parameters, quantiles)
            losses = fit_in_groups(
                dataset=frames[arm],
                jobs=[job],
                checkpoint_dir=checkpoint_root / f"{arm}__{setting}",
                max_workers=arguments.max_workers,
                device=device,
            )
            fractions = losses.with_columns(
                crps_capped_fraction_of_capacity=pl.col("crps_capped_mw")
                / pl.col("effective_capacity_mw")
            )
            annotated = drop_scaled_loss_columns(losses=fractions).join(
                frames[arm].select(
                    "site", "time", "capacity_scale_mw", actual_scaled=pl.col("power_mw")
                ),
                on=["site", "time"],
            )
            results.append(annotated)
            if arm == "b0_refit" and not arguments.smoke:
                published = pl.read_parquet(LAG_LOSSES_PATH).filter(
                    (pl.col("scope") == "lead1")
                    & (pl.col("arm") == "B0")
                    & (pl.col("setting") == "primary")
                )
                assert_refit_reproduces(refit=annotated, published=published)
                _LOG.info("the refit reproduces the lag study's B0 bit for bit")
    pl.concat(results, how="diagonal_relaxed").write_parquet(final)
    _LOG.info("wrote %s", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
