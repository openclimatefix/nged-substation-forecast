"""Fit the post hoc follow-ups to the lagged-power-features study, and save their per-row losses.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. The
first science review of the first run asked for these fits. **They add to the first run and never
change it**: every output sits under `<output-root>/ens_mean/followups/`, with its own checkpoints
and manifest, and the first run's files are only read.

**Fits** (a per-plant arm is 90 fits: 6 plants x 5 folds x 3 seeds; point model, primary setting
unless stated):

1. **Positive controls** on the frames `followup_frames.py` builds: the month-level control at 5%
   and 10% for B0, O (the oracle), W7, Q30, TF, AN and PC, and the plant-specific control at 5% and
   10% for B0, O, L1, W7, Q30, TF and AN.
2. **Long leads** (lead-days 7, 10, 14): CL (B0 plus the out-of-fold climatology), W7+CL and N2-3
   (a null as wide as W7); and the sensitivity setting for B0, N2, W7 and Q30 at lead-days 10 and
   14. With no fit, the 50/50 blends of B0 and of W7 with climatology are scored from the first
   run's saved predictions (`B0xCL` and `W7xCL`).
3. **PC2**, PC with a 2-day CAMS latency, at lead-day 1.
4. **Fingerprint decomposition** in the global scope, with the first run's fleet-wide folds:
   G-ID+TF, G-FPnoCK (G-FP without CK), leave-one-plant-out of G-FPnoCK, and per-plant B0 on the
   fleet-wide folds (so global against per-plant is paired).

`--dry-run` prints the number of fits and stops. Checkpoints are one file per (scope, setting, arm).
`checkpoints/run_manifest.json` records the device, the mode and a SHA-256 of every frame read, and
a resume that differs raises. `--smoke` runs the real path on 20 seeded random rows of each plant's
month with 20 boosting rounds, writes `_smoke` files, and is not a result.

Run it with `uv run python studies/lag_features/fit_followups.py`.
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Final

import polars as pl
from build_lag_frame import (
    FULL_SWEEP_LEAD_DAY,
    LAG_FEATURES_DIR,
    TARGET,
    output_paths,
    run_suffix,
    smoke_subsample,
    write_parquet_atomic,
)
from fit_lag_arms import (
    Context,
    Fit,
    check_manifest,
    fit_all,
    pooled_frame,
)
from followup_frames import (
    CONTROL_SHIFTS,
    LONG_LEADS,
    MONTH_CONTROL_ARMS,
    PRODUCT,
    STEP_CONTROL_ARMS,
    followup_dir,
    frame_names,
    frame_path,
)
from studies.cross_validation import N_FOLDS, SEEDS, score_prediction
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fit_followups")

LONG_LEAD_ARMS: Final[tuple[str, ...]] = ("CL", "W7+CL", "N2-3")
"""The arms fitted at each long lead, point model, primary setting."""

LONG_LEAD_SENSITIVITY_LEADS: Final[tuple[int, ...]] = (10, 14)
"""The long leads that also get the sensitivity setting."""

LONG_LEAD_SENSITIVITY_ARMS: Final[tuple[str, ...]] = ("B0", "N2", "W7", "Q30")
"""The arms fitted at the sensitivity setting at those leads."""

BLEND_SHARE: Final[float] = 0.5
"""The share of the climatology in the no-fit blends."""

BLEND_ARMS: Final[dict[str, str]] = {"B0xCL": "B0", "W7xCL": "W7"}
"""Each blend's name and the first run's arm it blends with climatology."""

GLOBAL_KEY: Final[str] = "global_lead1"
"""The key of the pooled lead-day 1 frame, fleet-wide folds, in the frames the fits read."""

PLANTS: Final[int] = 6
"""How many plants a per-plant fit and a leave-one-plant-out fit loop over."""


def followup_fits() -> list[Fit]:
    """Return every follow-up fit, in the order they run.

    Returns:
        The fits: control frames first, then the long leads, PC2 and the global scope.
    """
    lead = FULL_SWEEP_LEAD_DAY
    fits: list[Fit] = []
    for shift in CONTROL_SHIFTS:
        percent = f"{round(shift * 100):02d}"
        for kind, arms in (("month", MONTH_CONTROL_ARMS), ("step", STEP_CONTROL_ARMS)):
            name = f"control_{kind}_s{percent}"
            fits += [Fit(name, f"fu_{name}", lead, arm, "primary", quantiles=False) for arm in arms]
    for lead_days in LONG_LEADS:
        name = f"lead{lead_days}"
        fits += [
            Fit(name, f"fu_{name}", lead_days, arm, "primary", quantiles=False)
            for arm in LONG_LEAD_ARMS
        ]
        if lead_days in LONG_LEAD_SENSITIVITY_LEADS:
            fits += [
                Fit(name, f"fu_{name}", lead_days, arm, "sensitivity", quantiles=False)
                for arm in LONG_LEAD_SENSITIVITY_ARMS
            ]
    fits.append(Fit("lead1_pc2", "fu_lead1_pc2", lead, "PC2", "primary", quantiles=False))
    fits += [
        Fit(GLOBAL_KEY, "fu_global", lead, arm, "primary", quantiles=False, pooled=True)
        for arm in ("G-ID+TF", "G-FPnoCK")
    ]
    fits.append(Fit(GLOBAL_KEY, "fu_lopo", lead, "G-FPnoCK", "primary", quantiles=False, lopo=True))
    fits.append(Fit(GLOBAL_KEY, "fu_fleetfold", lead, "B0", "primary", quantiles=False))
    return fits


def fits_in(*, fit: Fit) -> int:
    """Return how many XGBoost fits one `Fit` makes.

    Args:
        fit: The fit.

    Returns:
        90 for a per-plant arm, 90 for a leave-one-plant-out arm and 15 for a pooled arm (folds
        times seeds), since a pooled model is one model per fold and seed.
    """
    per_model = N_FOLDS * len(SEEDS)
    return per_model if fit.pooled else PLANTS * per_model


def fit_count_lines(*, fits: list[Fit]) -> list[str]:
    """Describe the number of fits per group and in all.

    Args:
        fits: `followup_fits`' result.

    Returns:
        Report lines.
    """
    groups = {
        "1. positive controls": [f for f in fits if f.scope.startswith("fu_control")],
        "2. long leads": [f for f in fits if f.scope.startswith("fu_lead") and f.lead_day != 1],
        "3. PC2": [f for f in fits if f.scope == "fu_lead1_pc2"],
        "4. fingerprint decomposition": [
            f for f in fits if f.scope in {"fu_global", "fu_lopo", "fu_fleetfold"}
        ],
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
    """Return every file the follow-up fits read.

    Args:
        root: The output root.

    Returns:
        The follow-up frames and the first run's lead-day 1 frame.
    """
    first = output_paths(root=root, product=PRODUCT, lead_days=(FULL_SWEEP_LEAD_DAY,))
    return [*(frame_path(root=root, name=name) for name in frame_names()), first["day1"]]


def input_digests(*, root: Path) -> dict[str, str]:
    """Return the SHA-256 of every file the follow-up fits read.

    Args:
        root: The output root.

    Returns:
        The digests by file name.
    """
    return {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(input_files(root=root))
    }


def read_frames(*, root: Path, smoke: bool) -> dict[str, pl.DataFrame]:
    """Read every frame the fits name, subsampling them as a smoke run does.

    Args:
        root: The output root.
        smoke: Whether to subsample each frame.

    Returns:
        The frames by key; the pooled lead-day 1 frame has the first run's fleet-wide folds.
    """
    frames = {name: pl.read_parquet(frame_path(root=root, name=name)) for name in frame_names()}
    first = pl.read_parquet(
        output_paths(root=root, product=PRODUCT, lead_days=(FULL_SWEEP_LEAD_DAY,))["day1"]
    )
    if smoke:
        frames = {name: smoke_subsample(frame=frame) for name, frame in frames.items()}
        first = smoke_subsample(frame=first)
    frames[GLOBAL_KEY] = pooled_frame(frame=first)
    return frames


def blend_losses(*, root: Path, frames: dict[str, pl.DataFrame], smoke: bool) -> pl.DataFrame:
    """Score the 50/50 blends of B0 and of W7 with the out-of-fold climatology, with no fit.

    The blend of an arm is its saved first-run prediction and the row's climatology, weighted
    `1 - BLEND_SHARE` and `BLEND_SHARE`, held to the export cap and scored as a fitted arm is.

    Args:
        root: The output root.
        frames: The follow-up frames, by key.
        smoke: Whether the first run's smoke losses are the ones to read.

    Returns:
        Per-row losses for each blend at each long lead, with `arm`, `setting`, `scope`,
        `lead_day` and `prediction`.
    """
    first = pl.read_parquet(root / PRODUCT / f"losses_{PRODUCT}{run_suffix(smoke=smoke)}.parquet")
    parts = []
    for lead in LONG_LEADS:
        frame = frames[f"lead{lead}"]
        climate = pl.coalesce(
            pl.when(pl.col("fold") == fold).then(pl.col(f"climatology_fold{fold}"))
            for fold in range(N_FOLDS)
        )
        rows = frame.select(
            "site",
            "time",
            "month",
            "fold",
            "constrained",
            "effective_capacity_mw",
            "cap_mw",
            TARGET,
            climate=climate,
        )
        for blend, arm in BLEND_ARMS.items():
            saved = first.filter(
                (pl.col("scope") == f"lead{lead}")
                & (pl.col("setting") == "primary")
                & (pl.col("arm") == arm)
            ).select("site", "time", "seed", saved=pl.col("prediction"))
            prediction = (
                rows.select("site", "time", "climate")
                .join(saved, on=["site", "time"])
                .select(
                    "site",
                    "time",
                    "seed",
                    prediction=(1 - BLEND_SHARE) * pl.col("saved")
                    + BLEND_SHARE * pl.col("climate"),
                )
            )
            parts.append(
                score_prediction(rows=rows, prediction=prediction, target=TARGET)
                .join(prediction, on=["site", "time", "seed"])
                .with_columns(
                    arm=pl.lit(blend),
                    setting=pl.lit("primary"),
                    scope=pl.lit(f"fu_lead{lead}"),
                    lead_day=pl.lit(lead, dtype=pl.Int32),
                )
            )
    return pl.concat(parts)


def main() -> int:
    """Fit the follow-ups and write their losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the fit count and stop.")
    arguments = parser.parse_args()
    fits = followup_fits()
    if arguments.dry_run:
        sys.stdout.write("\n".join(fit_count_lines(fits=fits)) + "\n")
        return 0
    root: Path = arguments.output_root
    suffix = run_suffix(smoke=arguments.smoke)
    directory = followup_dir(root=root)
    final = directory / f"losses_followups_{PRODUCT}{suffix}.parquet"
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
    results.append(blend_losses(root=root, frames=frames, smoke=arguments.smoke))
    losses = pl.concat(results, how="diagonal_relaxed")
    write_parquet_atomic(frame=losses, path=final)
    _LOG.info("wrote %s (%d rows)", final, losses.height)
    sys.stdout.write("\n".join(fit_count_lines(fits=fits)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
