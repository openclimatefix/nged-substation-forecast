"""Exploratory forecast-level check: divide the lag study's B0 by the labellers' capacity.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
**This is exploratory and has no decision rule.** It reuses the first capacity study's Stage B
machinery (`studies.capacity_scaling`): B0's seven columns, the 29,445 lag-study rows, the folds,
the seeds, and the primary XGBoost setting, scored in megawatts over each plant's fixed static
power. Each arm divides the powers by a capacity series before the fit, so a better capacity
estimate should lower the megawatt error.

**Arms:** `b0_refit` (a divisor of 1, which must reproduce the published B0 bit for bit),
`tools_only`, `llm_hindsight_checked` (the model's knots after the hard checks), `llm_causal` (the
causal series, with the static power outside the last 6 months), and `HYST2` (the Part 1 estimator
that passed the false-movement gate). The causal arm is compared with `b0_refit` on those months
only. A scale is floored at 20% of the static power.

**Assertions before any arm:** dividing by a constant reproduces B0 (the first study's check), and
the refit equals the published B0.

Run it with `uv run python studies/llm_capacity_labeller/fit_forecast_arms.py --device cuda
--max-workers 4`. `--smoke` fits two plants on a sample on the CPU and is not a result.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from labeller_paths import (
    FIRST_STUDY_BUILT,
    check_stamp,
    claim_root,
    fingerprint,
    load_first_events,
    load_first_record,
    sha256_of,
    tables_dir,
)
from studies.arm_runner import Job
from studies.capacity_estimators import FLOOR_SHARE_OF_STATIC, CapacityRecord, scale_for_rows
from studies.capacity_scaling import (
    assert_arm_covers_frame,
    assert_constant_scale_reproduces_baseline,
    assert_refit_reproduces,
    drop_scaled_loss_columns,
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
    SEEDS,
    DeviceType,
    HyperParameters,
)
from studies.guards import refuse_to_overwrite
from studies.sources import study_dir_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fit_forecast_arms")

LAG_DIR: Final[Path] = study_dir_for(study="lag_features") / "ens_mean"
"""The lag-features study's outputs, which this script reads and never writes."""

LAG_FRAME_PATH: Final[Path] = LAG_DIR / "lag_frame_ens_mean_day1.parquet"
"""The 29,445 lead-day 1 rows. Path copied from the first capacity study's `capacity_paths.py`."""

LAG_LOSSES_PATH: Final[Path] = LAG_DIR / "losses_ens_mean.parquet"
"""The lag study's per-row losses, from which B0 is reused."""

LAG_FRAME_ROWS: Final[int] = 29445
"""The row count the lag study published."""

B0_COLUMNS: Final[tuple[str, ...]] = (
    "hour_of_day",
    "day_of_year",
    "era_code",
    "solar_elevation_deg",
    "solar_azimuth_deg",
    "nwp_ghi",
    "nwp_temp",
)
"""B0's seven columns, as the lag study defines them (copied from `fit_capacity_arms.py`)."""

CONSTANT_SCALE: Final[float] = 0.37
CONSTANT_SCALE_TOLERANCE_POINTS: Final[float] = 1e-3
"""The constant divisor of the constant-scale assertion, and its tolerance (copied likewise)."""

SMOKE_PLANTS: Final[tuple[str, ...]] = ("A", "E")
SMOKE_ROWS_PER_PLANT_MONTH: Final[int] = 20
SMOKE_ROUNDS: Final[int] = 20
"""A smoke run's plants, rows per plant-month, and boosting rounds."""


def smoke_frame(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Sample a small frame for a smoke run, keeping every constrained row.

    Args:
        frame: The lag frame.

    Returns:
        Rows of the smoke plants: 20 seeded rows per plant-month plus the constrained rows.
    """
    mine = frame.filter(pl.col("site").is_in(list(SMOKE_PLANTS))).with_row_index("_row")
    sampled = (
        mine.group_by("site", "month", maintain_order=True)
        .agg(pl.col("_row").shuffle(seed=1147).head(SMOKE_ROWS_PER_PLANT_MONTH))
        .explode("_row")["_row"]
    )
    keep = pl.col("_row").is_in(sampled.to_list()) | pl.col("constrained")
    return mine.filter(keep).drop("_row")


def arm_frame(*, frame: pl.DataFrame, record: CapacityRecord, scale: np.ndarray) -> pl.DataFrame:
    """Return the lag frame divided by an arm's scale.

    Args:
        frame: The lag frame.
        record: The original record (for site and day indices).
        scale: The arm's scale, shape (plants, days).

    Returns:
        The frame with `power_mw`, `cap_mw`, and `effective_capacity_mw` divided by the row's scale.
    """
    return scaled_frame(
        frame=frame, scale_mw=scale_for_rows(record=record, scale_mw=scale, rows=frame)
    )


def arm_scales(*, root: Path, record: CapacityRecord) -> dict[str, np.ndarray]:
    """Read each arm's scale from the stored capacity series, floored at 20% of the static power.

    Args:
        root: The output root.
        record: The original record.

    Returns:
        The scale by arm name; an arm whose series is missing is left out. The causal series has
        NaN before its first issue date, and the static power stands in there.
    """
    static = np.repeat(record.static_p99_mw[:, None], len(record.dates), axis=1)
    part1 = root / "part1" / "scales_original.npz"
    with np.load(tables_dir(root=root) / "capacity_series.npz") as stored:
        found = {
            "tools_only": stored.get("original/tools_only"),
            "llm_hindsight_checked": stored.get("original/llm_hindsight_checked"),
            "llm_causal": stored.get("original/llm_causal"),
        }
    if part1.exists():
        with np.load(part1) as stored_part1:
            found["HYST2"] = stored_part1["HYST2"]
    return {
        name: np.maximum(
            np.where(np.isfinite(series), series, static), FLOOR_SHARE_OF_STATIC * static
        )
        for name, series in found.items()
        if series is not None
    }


def main() -> int:
    """Run the exploratory forecast arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU run; not a result.")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    device: DeviceType = "cpu" if arguments.smoke else choose_device(requested=arguments.device)
    record = load_first_record()
    frame = pl.read_parquet(LAG_FRAME_PATH)
    if frame.height != LAG_FRAME_ROWS:
        msg = f"the lag frame holds {frame.height} rows, not {LAG_FRAME_ROWS}"
        raise ValueError(msg)
    if arguments.smoke:
        frame = smoke_frame(frame=frame)
    primary: HyperParameters = (
        {**PRIMARY_HYPER_PARAMETERS, "num_boost_round": SMOKE_ROUNDS}
        if arguments.smoke
        else PRIMARY_HYPER_PARAMETERS
    )
    anm_site = record.sites[load_first_events().anm_plant]
    worst = assert_constant_scale_reproduces_baseline(
        site_rows=frame.filter(pl.col("site") == anm_site),
        features=B0_COLUMNS,
        hyper_parameters=primary,
        scale=CONSTANT_SCALE,
        tolerance_points=CONSTANT_SCALE_TOLERANCE_POINTS,
        device=device,
    )
    _LOG.info("constant-scale assertion passed (largest difference %.2e points)", worst)
    scales = {
        "b0_refit": np.ones((len(record.sites), len(record.dates))),
        **arm_scales(root=root, record=record),
    }
    forecast_dir = root / "forecast"
    stamp = run_stamp(
        dataset_path=LAG_FRAME_PATH,
        arms={
            **{arm: list(B0_COLUMNS) for arm in scales},
            "series": [sha256_of(path=tables_dir(root=root) / "capacity_series.npz")],
            "code": [fingerprint(extra_files=(Path(__file__),))],
        },
        device=device,
    )
    check_stamp(
        directory=forecast_dir / "outputs", stamp=fingerprint(extra_files=(Path(__file__),))
    )
    claim_checkpoint_dir(checkpoint_dir=forecast_dir / "checkpoints", stamp=stamp)
    final = forecast_dir / "outputs" / "losses_forecast_arms.parquet"
    refuse_to_overwrite(paths=[final])
    results: list[pl.DataFrame] = []
    for arm, scale in scales.items():
        data = arm_frame(frame=frame, record=record, scale=scale)
        job: Job = (arm, "primary", "power_mw", B0_COLUMNS, primary, False)
        losses = fit_in_groups(
            dataset=data,
            jobs=[job],
            checkpoint_dir=forecast_dir / "checkpoints" / f"{arm}__primary",
            max_workers=arguments.max_workers,
            device=device,
        )
        fractions = losses.with_columns(
            crps_capped_fraction_of_capacity=pl.col("crps_capped_mw")
            / pl.col("effective_capacity_mw")
        )
        annotated = drop_scaled_loss_columns(losses=fractions).join(
            data.select("site", "time", "capacity_scale_mw", actual_scaled=pl.col("power_mw")),
            on=["site", "time"],
        )
        assert_arm_covers_frame(losses=annotated, frame=data, seeds=SEEDS)
        results.append(annotated)
        if arm == "b0_refit" and not arguments.smoke:
            published = pl.read_parquet(LAG_LOSSES_PATH).filter(
                (pl.col("scope") == "lead1")
                & (pl.col("arm") == "B0")
                & (pl.col("setting") == "primary")
            )
            assert_refit_reproduces(refit=annotated, published=published)
            _LOG.info("the refit reproduces the lag study's B0 bit for bit")
        _LOG.info("fitted %s", arm)
    final.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(results, how="diagonal_relaxed").write_parquet(final)
    _LOG.info("wrote %s (inputs read from %s)", final, FIRST_STUDY_BUILT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
