"""Write every arm's quantiles for the replicate months and the scored year, from the cache.

The residual tables, factors, and rescaling come from the tuning months only (`e00125_tune.py`):
K0 keeps experiment 111's stored table and factors, K0-w re-tunes K0's factors on the wide grid,
MP fits its table on the tuning months' proxy-driven residuals, MP-s adds the tuned rescaling of
the members, MM centres the pooled forecast on the ensemble mean, and MO uses no table. The
predictions are written once per (arm, period, site) and are never rewritten.

    uv run python ml-experiments/00125-*/e00125_predict.py --periods replicate scored
"""

import argparse
import json

import e00125_arms as arms
import e00125_cache as cache_module
import e00125_common as common
import e00125_post as post
import numpy as np
import polars as pl
from studies.dp_gam import QUANTILE_COLUMNS

ARM_NAMES: tuple[str, ...] = ("K0", "K0-w", "MP", "MP-s", "MM", "MO", "MP-1")
PERIOD_MONTHS = {"tuning": common.TUNING_MONTHS, "replicate": common.REPLICATE_MONTHS}


def preds_path(*, arm: str, period: str, site: str):  # noqa: ANN201
    """Return the file that holds an arm's predictions at a site and period."""
    return common.WORK_DIR / "preds" / f"{arm}_{period}_{site}.parquet"


def arm_rows(*, site: str, runs: list[cache_module.RunCache], tuned: dict) -> dict[str, post.Rows]:
    """Return every arm's rows for a site's runs.

    Args:
        site: The substation's label.
        runs: The cached runs to forecast.
        tuned: The contents of `work/tuned_n8.json`.

    Returns:
        The rows of each arm in `ARM_NAMES`.
    """
    tuning = cache_module.read_cache(site=site, period="tuning")
    mask = post.stacked_months_mask(runs=tuning, months=common.TUNING_MONTHS)
    mp_table = post.residual_table(runs=tuning, centre="proxy", row_filter=mask)
    k0_table = arms.load_spread(arm="gamk", site=site)
    return {
        "K0": post.build_rows(
            runs=runs,
            spread=k0_table.with_factors(factors=tuned["stored_k0_factors"]),
            centre="members",
        ),
        "K0-w": post.build_rows(
            runs=runs,
            spread=k0_table.with_factors(factors=tuned["k0w_factors"]),
            centre="members",
        ),
        "MP": post.build_rows(
            runs=runs,
            spread=mp_table.with_factors(factors=tuned["mp_factors"]),
            centre="members",
        ),
        "MP-s": post.build_rows(
            runs=runs,
            spread=mp_table.with_factors(factors=tuned["mp_s_factors"]),
            centre="members",
            s=tuned["mp_s"],
        ),
        "MM": post.build_rows(
            runs=runs,
            spread=k0_table.with_factors(factors=tuned["mm_factors"]),
            centre="mean",
        ),
        "MO": post.build_rows(runs=runs, spread=None, centre="members"),
        "MP-1": post.build_rows(runs=runs, spread=mp_table, centre="members"),
    }


def write_site(*, site: str, period: str, tuned: dict) -> str:
    """Write every arm's predictions for one site and period."""
    runs = cache_module.read_cache(site=site, period=period)
    if period in PERIOD_MONTHS:
        valid = post.stacked_months_mask(runs=runs, months=PERIOD_MONTHS[period])
    else:
        valid = None
    for arm, all_rows in arm_rows(site=site, runs=runs, tuned=tuned).items():
        rows = all_rows if valid is None else post.select(rows=all_rows, mask=valid)
        path = preds_path(arm=arm, period=period, site=site)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            msg = f"{path} exists; predictions are written once"
            raise FileExistsError(msg)
        frame = pl.DataFrame(
            {
                "site": np.full(len(rows.actual), site),
                "run": rows.run.astype(np.int32),
                "valid_time": pl.Series(rows.valid_time).dt.replace_time_zone("UTC"),
                "lead_hours": rows.lead_hours,
                "hour_band": rows.hour_band,
                "quarter": rows.quarter,
                "switching_mask": rows.switching_mask,
                "actual": rows.actual,
                **{
                    column: rows.quantiles[:, position]
                    for position, column in enumerate(QUANTILE_COLUMNS)
                },
            }
        )
        frame.write_parquet(path)
    return f"{site} {period}: {len(runs)} runs"


def main() -> None:
    """Write the predictions of the chosen periods."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", nargs="+", default=["replicate", "scored"])
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    tuned = json.loads((common.WORK_DIR / "tuned_n8.json").read_text())
    for period in args.periods:
        for site in args.sites:
            print(write_site(site=site, period=period, tuned=tuned), flush=True)


if __name__ == "__main__":
    main()
