"""Score the sweep: absolute errors, planned contrasts with intervals, and breakdowns.

Reads the prediction files `e00006_fit.py` wrote, builds the per-row loss frame the repo's month
bootstrap pairs and resamples, and prints the numbers the write-up quotes.

    uv run python ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_report.py planned select_2024
"""

import sys
from typing import Final

import e00006_fit as fit
import e00006_sweep as sweep
import polars as pl
from studies.bootstrap import bootstrap_absolute, bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap

N_PLANNED_CONTRASTS: Final[int] = 9
LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_PLANNED_CONTRASTS)
"""Bonferroni-widened coverage, in percent, across the nine planned contrasts."""
N_RESAMPLES: Final[int] = 20_000


def read_seeds(*, stage: str, fold_name: str, config: str = "series") -> pl.DataFrame:
    """Read every seed of a stage and return long rows `(time_series_id, valid_time, seed, ...)`."""
    folder = fit.FOLDS_DIR / stage / fold_name / config
    pieces = []
    for seed_dir in sorted(folder.glob("seed*")):
        seed = int(seed_dir.name.removeprefix("seed"))
        pieces.append(
            sweep.read_stage(stage=stage, fold_name=fold_name, config=config, seed=seed).with_columns(
                seed=pl.lit(seed)
            )
        )
    return pl.concat(pieces, how="diagonal")


def losses(*, rows: pl.DataFrame, arms: list[str]) -> pl.DataFrame:
    """Return the bootstrap's loss frame, with `abs_error` in points of capacity."""
    keyed = rows.with_columns(
        power_fcst_init_time=pl.col("valid_time").dt.truncate("1mo"),
    )
    return losses_for_bootstrap(
        arm_losses={
            arm: keyed.select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "seed",
                abs_error=(pl.col(arm) - pl.col("target")).abs() * 100.0,
            )
            for arm in arms
        }
    )


def contrast_line(*, frame: pl.DataFrame, treatment: str, reference: str) -> str:
    """Return one planned contrast as text: difference and Bonferroni-widened interval."""
    low, high = bootstrap_difference_at_level(
        losses=frame,
        treatment=treatment,
        reference=reference,
        metric="abs_error",
        level=LEVEL,
        n_resamples=N_RESAMPLES,
    )
    mean = (
        frame.filter(pl.col("arm") == treatment)["abs_error"].mean()
        - frame.filter(pl.col("arm") == reference)["abs_error"].mean()
    )
    return f"{treatment} minus {reference}: {mean:+.3f} ({low:+.3f} to {high:+.3f})"


def absolute_line(*, frame: pl.DataFrame, arm: str) -> str:
    """Return one arm's mean absolute error and its 95% month-resampled interval."""
    result = bootstrap_absolute(losses=frame, arm=arm, metric="abs_error")
    return f"{arm}: {result['value']:.3f} ({result['lower_95']:.3f} to {result['upper_95']:.3f})"


def planned_selection_fold(*, fold_name: str) -> None:
    """Print P1 to P7 and the absolute scores on a selection fold."""
    rows = read_seeds(stage="planned", fold_name=fold_name)
    arms = [c for c in rows.columns if c not in ("time_series_id", "valid_time", "target", "seed")]
    frame = losses(rows=rows, arms=arms)
    pairs = [
        ("P1", "W+clock", "W"),
        ("P2", "W+clock+doy1", "W+clock"),
        ("P3", "W+clock+hol_flags", "W+clock"),
        ("P4", "W+clock+t24", "W+clock"),
        ("P5", "W+windchill", "W"),
        ("P6", "W+clock+covid_flag", "W+clock"),
        ("P7", "W+clock|recency", "W+clock"),
    ]
    print(f"{fold_name}: {rows['time_series_id'].n_unique()} series, level {LEVEL:.2f}%")
    for label, treatment, reference in pairs:
        print(label, contrast_line(frame=frame, treatment=treatment, reference=reference))
    for arm in arms:
        print("abs", absolute_line(frame=frame, arm=arm))


def main() -> None:
    """Dispatch a report named on the command line."""
    kind, fold_name = sys.argv[1], sys.argv[2]
    {"planned": planned_selection_fold}[kind](fold_name=fold_name)


if __name__ == "__main__":
    main()
