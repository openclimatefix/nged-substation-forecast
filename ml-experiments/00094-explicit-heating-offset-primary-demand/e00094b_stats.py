"""Score block B of experiment 00094: `amh` (solar plus heating) against experiment 00064's `am`.

Both arms cover the five series with a solar term, the 73 runs of `fold_a_every5d_v1`, and seeds 0
to 2, with experiment 00064's 14-day correction and spread factor 1.15. A case is cold when its
ensemble-mean ENS 2 m temperature is below 3 C. The contrasts are exploratory (95%): each takes the
mean over the three seeds of the paired difference at each case, then resamples whole ISO weeks of
valid time. Prints, and writes to `work/results/`: `b_contrasts`, `b_slices`.

    uv run python ml-experiments/00094-*/e00094b_stats.py
"""

from typing import Final

import e00094_stats as base
import e00094b_assemble as assembling
import e00094b_common as common
import numpy as np
import polars as pl

REFERENCE_PATH: Final = common.WORK_DIR / "arms" / "am.parquet"  # experiment 00064's scored form
CANDIDATE_PATH: Final = assembling.fitting.B_WORK_DIR / "b_arms" / "amh.parquet"


def case_table(*, path: str, name: str) -> pl.DataFrame:
    """Return one arm's per-case, per-seed metrics."""
    frame = (
        pl.read_parquet(path)
        .filter(pl.col("actual").is_finite() & (pl.col("valid_time") < common.HOLDOUT))
        .sort(*common.KEYS, "seed")
    )
    quantiles = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1)
    actual = frame["actual"].to_numpy().astype(np.float64)
    miss = actual[:, None] - quantiles
    pinball = np.maximum(common.LEVELS * miss, (common.LEVELS - 1.0) * miss) * 100.0
    return frame.select(*common.KEYS, "seed").with_columns(
        **{
            f"qs_{name}": pl.Series(
                common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0
            ),
            f"bias_{name}": pl.Series(miss[:, common.MEDIAN] * 100.0),
            f"pin05_{name}": pl.Series(pinball[:, 0]),
            f"pin95_{name}": pl.Series(pinball[:, 8]),
            f"above95_{name}": pl.Series((actual > quantiles[:, 8]).astype(float)),
            f"below05_{name}": pl.Series((actual < quantiles[:, 0]).astype(float)),
        }
    )


def ens_temperature() -> pl.DataFrame:
    """Return the ensemble-mean ENS temperature of each case of the 73 runs."""
    pieces = [
        pl.read_parquet(path, columns=[*common.KEYS, "temperature_2m"])
        .group_by(*common.KEYS)
        .agg(ens_temperature=pl.col("temperature_2m").mean())
        for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
    ]
    return pl.concat(pieces)


def main() -> None:
    """Print the contrasts and slices and write them under `work/results/`."""
    frame = (
        case_table(path=str(REFERENCE_PATH), name="am")
        .join(
            case_table(path=str(CANDIDATE_PATH), name="amh"),
            on=[*common.KEYS, "seed"],
            how="inner",
            validate="1:1",
        )
        .join(ens_temperature(), on=list(common.KEYS), how="left", validate="m:1")
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            local_month=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.month(),
            week=pl.col("valid_time").dt.strftime("%G-W%V"),
        )
    )
    metrics = ("qs", "bias", "pin05", "pin95", "above95", "below05")
    per_case = frame.group_by(
        *common.KEYS, "week", "lead_hours", "local_month", "ens_temperature"
    ).agg([pl.col(f"{m}_{arm}").mean() for m in metrics for arm in ("am", "amh")])
    cold = per_case["ens_temperature"] < base.COLD_C
    short = per_case["lead_hours"] <= base.SHORT_LEAD_HOURS
    winter = per_case["local_month"].is_in([12, 1, 2])
    summer = per_case["local_month"].is_in([4, 5, 6, 7, 8, 9])
    slices = {
        "all": pl.Series(np.ones(per_case.height, dtype=bool)),
        "cold": cold,
        "cold_short": cold & short,
        "cold_long": cold & ~short,
        "mild_winter": winter & ~cold,
        "april_to_september": summer,
    }
    contrasts = []
    summary = []
    for name, mask in slices.items():
        subset = per_case.filter(mask)
        difference = (subset["qs_amh"] - subset["qs_am"]).to_numpy()
        lower, upper, n_weeks = base.week_bounds(
            values=difference,
            weeks=subset["week"].to_numpy(),
            level=base.EXPLORATORY_LEVEL,
            generator=np.random.default_rng(base.SEED),
        )
        contrasts.append(
            {
                "slice": name,
                "difference": float(difference.mean()),
                "lower": lower,
                "upper": upper,
                "cases": subset.height,
                "weeks": n_weeks,
            }
        )
        summary.extend(
            {"slice": name, "arm": arm, "cases": subset.height}
            | {m: subset[f"{m}_{arm}"].mean() for m in metrics}
            for arm in ("am", "amh")
        )
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        for name, table in (
            ("b_contrasts", pl.DataFrame(contrasts)),
            ("b_slices", pl.DataFrame(summary)),
        ):
            print(f"\n== {name}")
            print(table)
            common.write_atomically(frame=table, target=base.common.RESULTS_DIR / f"{name}.parquet")


if __name__ == "__main__":
    main()
