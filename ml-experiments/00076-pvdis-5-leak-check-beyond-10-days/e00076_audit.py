"""Audit the base margin of experiment 00064's arm `am` for leakage, and print each check.

Checks (each prints PASS or FAIL; the script exits non-zero if one fails):

a. The weight, and so the base margin, is zero in every training row and every validation row
   whose lead is 10 days or more, for each of the five series.
b. Every training row is at or before the last training valid time, so the weights (cross-fitted on
   the training rows) cannot hold a validation actual. The capacity and the calibration slopes are
   checked by reading `e00064_capacity.py` and `e00064_inputs.py`; see the result page.
c. Rebuilding `am`'s predictions from its stored booster with the margin that `validation_runs`
   supplies (seed 0, first runs of each series) reproduces experiment 00064's stored raw
   quantiles, and a margin of zero changes them only at leads under 10 days.

    uv run python ml-experiments/00076-*/e00076_audit.py
"""

import sys
from typing import Final

import e00076_common as common
import e00076_model as model
import numpy as np
import polars as pl
import xgboost as xgb

N_RUNS: Final[int] = 8
TOLERANCE: Final[float] = 1e-6
LONG_LEAD_HOURS: Final[float] = 240.0

failures: list[str] = []


def report(*, passed: bool, message: str) -> None:
    """Print a check's outcome and remember a failure."""
    print(("PASS: " if passed else "FAIL: ") + message, flush=True)
    if not passed:
        failures.append(message)


def audit_training(*, ctx: model.SeriesContext, weights: pl.DataFrame) -> None:
    """Checks a and b on one series' training rows."""
    rows = model.training_rows(ctx=ctx)
    latest = common.latest_time(times=rows["valid_time"])
    report(
        passed=latest <= common.LAST_TRAINING_VALID,
        message=f"b {ctx.site}: {rows.height:,} training rows, latest valid time {latest}",
    )
    weight = model.weights_per_row(
        weights=weights,
        site=ctx.site,
        band=rows["band"].to_numpy(),
        month=rows["month"].to_numpy(),
    )
    lead = rows["lead_hours"].to_numpy()
    long = lead >= LONG_LEAD_HOURS
    margin = model.margin_rows(
        weight=weight, anomaly=rows[model.ANOMALY_COLUMN].fill_null(0.0).to_numpy(), scale=ctx.scale
    )
    report(
        passed=bool(long.any())
        and float(np.abs(margin[long]).max()) == 0.0
        and bool((rows["band"].to_numpy()[long] >= model.NO_SKILL_BAND).all()),
        message=f"a {ctx.site} training: {int(long.sum()):,} rows at 10+ days, largest |margin| "
        f"{float(np.abs(margin[long]).max())}; {int((margin[~long] != 0).any(axis=1).sum()):,} "
        f"of {int((~long).sum()):,} shorter-lead rows have a margin",
    )


def audit_prediction(*, ctx: model.SeriesContext, weights: pl.DataFrame, device: str) -> None:
    """Checks a and c on the first validation runs of one series (seed 0)."""
    booster = xgb.Booster(params={"nthread": 1, "device": device})
    booster.load_model(common.E64_BOOSTER_DIR / f"am_seed0_{ctx.site}.ubj")
    stored = pl.read_parquet(common.E64_RAW_ARMS_DIR / f"am_seed0_{ctx.site}.parquet").filter(
        pl.col("variant") == "base"
    )
    largest_long = 0.0
    largest_rebuild = 0.0
    changed_short = 0.0
    long_rows = 0
    for run in model.validation_runs(ctx=ctx, weights=weights, max_runs=N_RUNS):
        lead = (
            (run.keys["valid_time"] - run.keys["power_fcst_init_time"]).dt.total_minutes() / 60.0
        ).to_numpy()
        long = lead >= LONG_LEAD_HOURS
        margin = model.margin_rows(
            weight=run.solar.weight, anomaly=run.solar.anomaly_mean, scale=ctx.scale
        )
        largest_long = max(largest_long, float(np.abs(margin[long]).max()) if long.any() else 0.0)
        long_rows += int(long.sum())
        with_margin = common.mixture_quantiles(
            member_quantiles=model.predict_members(
                booster=booster,
                members=run.members,
                columns=model.d1l_columns(),
                margin=margin,
                n_members=run.n_members,
            )
        )
        without = common.mixture_quantiles(
            member_quantiles=model.predict_members(
                booster=booster,
                members=run.members,
                columns=model.d1l_columns(),
                margin=np.zeros_like(margin),
                n_members=run.n_members,
            )
        )
        frame = model.variant_frame(keys=run.keys, quantiles=with_margin, variant="base", seed=0)
        both = frame.join(stored, on=[*common.KEYS, "variant"], suffix="_s", validate="1:1")
        rebuilt = np.abs(
            both.select(common.LEVEL_COLUMNS).to_numpy()
            - both.select([f"{c}_s" for c in common.LEVEL_COLUMNS]).to_numpy()
        ).max()
        largest_rebuild = max(largest_rebuild, float(rebuilt))
        difference = np.abs(with_margin - without).max(axis=1)
        largest_long = max(largest_long, float(difference[long].max()) if long.any() else 0.0)
        changed_short = max(changed_short, float(difference[~long].max()))
    report(
        passed=largest_long == 0.0 and long_rows > 0,
        message=f"a {ctx.site} validation: {long_rows:,} rows at 10+ days, largest |margin| and "
        f"largest change from zeroing the margin both {largest_long}",
    )
    report(
        passed=largest_rebuild <= TOLERANCE,
        message=f"c {ctx.site}: rebuilt am quantiles match experiment 00064's stored ones "
        f"(largest difference {largest_rebuild:.2e}); zeroing the margin changes shorter leads by "
        f"up to {changed_short:.3f}",
    )


def main() -> None:
    """Run the checks on the five series."""
    weights = model.read_weights()
    for site in common.SITES:
        ctx = model.load_context(site=site)
        audit_training(ctx=ctx, weights=weights)
        audit_prediction(ctx=ctx, weights=weights, device="cpu")
    if failures:
        print(f"\n{len(failures)} check(s) failed")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
