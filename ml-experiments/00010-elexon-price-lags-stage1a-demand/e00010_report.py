r"""Score experiment 10: absolute errors, the planned contrasts with intervals, and a month table.

Reads the prediction files `e00010_fit.py` wrote and prints and writes tables under the
experiment's private data folder. Errors are mean absolute error of the median in points of
capacity (1% of the series' effective capacity). The planned contrasts use 98.3% month-resampled
intervals (95% familywise across C1, C2, and C4). The listed series are B2 and P4, chosen from
NGED's Embedded Capacity Register before any fit.

    uv run python ml-experiments/00010-elexon-price-lags-stage1a-demand/e00010_report.py \
        --fold select_2024
"""

import argparse
from pathlib import Path
from typing import Final

import e00010_inputs as inputs
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_absolute, bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap

RESULTS_DIR: Final[Path] = inputs.EXPERIMENT_DIR / "results"
ARMS: Final[tuple[str, ...]] = ("BASE", "PRICE", "CONTROL", "SHUFFLED")
LISTED: Final[tuple[str, ...]] = ("B2", "P4")
LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3)
N_RESAMPLES: Final[int] = 20_000
C4_MARGIN: Final[float] = 0.2
SERIES_PATH: Final[Path] = inputs.EXPERIMENT_00006_DIR / "series.parquet"


def read_fits(*, fold: str) -> pl.DataFrame:
    """Return every unit's predictions for a fold, with the seed and the series label."""
    frames = [
        pl.read_parquet(path).with_columns(seed=pl.lit(int(path.parent.name.removeprefix("seed"))))
        for path in sorted((inputs.EXPERIMENT_DIR / "fits" / fold).glob("seed*/*.parquet"))
    ]
    series = pl.read_parquet(SERIES_PATH).select("time_series_id", "site", "substation_type")
    return pl.concat(frames).join(series, on="time_series_id")


def loss_frame(*, rows: pl.DataFrame, arms: tuple[str, ...]) -> pl.DataFrame:
    """Return the bootstrap loss frame with `abs_error` in points of capacity."""
    keyed = rows.with_columns(power_fcst_init_time=pl.col("valid_time").dt.truncate("1mo"))
    return losses_for_bootstrap(
        arm_losses={
            arm: keyed.select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "seed",
                abs_error=(pl.col(arm) - pl.col("target")).abs() * 100.0,
            ).drop_nulls("abs_error")
            for arm in arms
        }
    )


def contrast(*, rows: pl.DataFrame, treatment: str, reference: str, label: str) -> dict:
    """Return one paired difference (treatment minus reference) with its 98.3% interval."""
    frame = loss_frame(rows=rows, arms=(treatment, reference))
    low, high = bootstrap_difference_at_level(
        losses=frame,
        treatment=treatment,
        reference=reference,
        metric="abs_error",
        level=LEVEL,
        n_resamples=N_RESAMPLES,
    )
    means = frame.group_by("arm").agg(pl.col("abs_error").mean())
    value = dict(zip(means["arm"], means["abs_error"], strict=True))
    return {
        "scope": label,
        "contrast": f"{treatment} - {reference}",
        "difference": value[treatment] - value[reference],
        "lower": low,
        "upper": high,
    }


def difference_of_gains(*, rows: pl.DataFrame, seed: int = 0) -> dict:
    """Return C4: the PRICE - BASE gain at the listed series minus that at the others.

    Each month's gain is the mean over rows and seeds of the paired loss difference, within the
    listed group and within the other series. Months are resampled with replacement.
    """
    monthly = (
        rows.with_columns(
            month=pl.col("valid_time").dt.truncate("1mo"),
            gain=(pl.col("PRICE") - pl.col("target")).abs() * 100.0
            - (pl.col("BASE") - pl.col("target")).abs() * 100.0,
            listed=pl.col("site").is_in(LISTED),
        )
        .group_by("month", "listed")
        .agg(pl.col("gain").mean())
        .pivot(on="listed", index="month", values="gain")
        .sort("month")
    )
    delta = (monthly["true"] - monthly["false"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(N_RESAMPLES, len(delta))).mean(axis=1)
    tail = (100.0 - LEVEL) / 2.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return {
        "scope": "listed minus others",
        "contrast": "C4: (PRICE - BASE) listed - others",
        "difference": float(delta.mean()),
        "lower": float(low),
        "upper": float(high),
    }


def main() -> None:
    """Print and write the levels, the contrasts, and the per-series gains for a fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True)
    fold = parser.parse_args().fold
    rows = read_fits(fold=fold)
    scopes: dict[str, pl.DataFrame] = {
        "listed (B2, P4)": rows.filter(pl.col("site").is_in(LISTED)),
        "B2": rows.filter(pl.col("site") == "B2"),
        "P4": rows.filter(pl.col("site") == "P4"),
        "others (18)": rows.filter(~pl.col("site").is_in(LISTED)),
        "Primary": rows.filter(pl.col("substation_type") == "Primary"),
    }
    levels, contrasts = [], []
    for label, scope in scopes.items():
        if scope.is_empty():
            print(f"{label}: no fitted series in {fold}, skipped")
            continue
        frame = loss_frame(rows=scope, arms=ARMS)
        for arm in ARMS:
            result = bootstrap_absolute(losses=frame, arm=arm, metric="abs_error")
            levels.append(
                {
                    "scope": label,
                    "arm": arm,
                    "value": result["value"],
                    "lower": result["lower_95"],
                    "upper": result["upper_95"],
                }
            )
        for treatment, reference in (
            ("PRICE", "BASE"),
            ("PRICE", "CONTROL"),
            ("PRICE", "SHUFFLED"),
        ):
            contrasts.append(
                contrast(rows=scope, treatment=treatment, reference=reference, label=label)
            )
    contrasts.append(difference_of_gains(rows=rows))
    per_series = (
        rows.group_by("site", "substation_type")
        .agg(
            base=((pl.col("BASE") - pl.col("target")).abs() * 100.0).mean(),
            price_gain=(
                (pl.col("PRICE") - pl.col("target")).abs()
                - (pl.col("BASE") - pl.col("target")).abs()
            ).mean()
            * 100.0,
            control_gain=(
                (pl.col("CONTROL") - pl.col("target")).abs()
                - (pl.col("BASE") - pl.col("target")).abs()
            ).mean()
            * 100.0,
        )
        .sort("site")
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with pl.Config(tbl_rows=100, tbl_width_chars=200, float_precision=3):
        for name, table in (
            ("levels", pl.DataFrame(levels)),
            ("contrasts", pl.DataFrame(contrasts)),
            ("per_series", per_series),
        ):
            table.write_parquet(RESULTS_DIR / f"{fold}_{name}.parquet")
            print(f"\n## {fold} {name}\n{table}")
    seeds = sorted(rows["seed"].unique().to_list())
    print(f"\nC4 margin required: {C4_MARGIN} points; seeds: {seeds}")


if __name__ == "__main__":
    main()
