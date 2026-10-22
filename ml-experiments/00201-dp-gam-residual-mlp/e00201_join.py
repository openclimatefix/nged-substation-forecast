"""Join the past-weather rebuild and experiment 155's stored arm D into one frame per substation.

The output `work/gam_d/<site>.parquet` has experiment 155's `dp_preds` columns (`time_series_id`,
`run`, `power_fcst_init_time`, `valid_time`, `lead_hours`, `seed`, `arm`, `q01` to `q99`, `actual`,
`mean`, `hour_band`, `quarter`, `switching_mask`) plus `level` and `anomaly` (the Kalman filter's
persistent and decaying corrections, from experiment 136's arm A2 for the ensemble era), `era`
(`past_weather` or `ens`) and `period` (`early`, or experiment 155's `tuning`, `replicate`,
`scored`). The row key is (`time_series_id`, `power_fcst_init_time`, `valid_time`). The ensemble-era
rows are experiment 155's arm D unchanged, so its scored-year numbers reproduce exactly.

    uv run --no-sync python ml-experiments/00201-*/e00201_join.py
"""

import os
from pathlib import Path
from typing import Final

import e00201_base as base
import e00201_common as common
import polars as pl

OUT_DIR: Final[Path] = common.WORK_DIR / "gam_d"
E136_PREDS: Final[Path] = (
    common.SOURCE_WORK_DIR.parent.parent
    / "00136-dp-bg-stage1-assembly-on-the-ensemble-mean"
    / "work"
    / "dp_preds"
)
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")


def ens_frame(*, site: str) -> pl.DataFrame:
    """Return experiment 155's arm D at a site with the filter's level and anomaly attached."""
    pieces = []
    for period in PERIODS:
        stored = pl.read_parquet(common.E155_DP_PREDS / f"d_{period}_{site}.parquet")
        a2 = pl.read_parquet(
            E136_PREDS / f"a2_{period}_{site}.parquet",
            columns=["run", "valid_time", "level", "anomaly"],
        )
        pieces.append(
            stored.join(a2, on=["run", "valid_time"], how="left").with_columns(
                era=pl.lit("ens"), period=pl.lit(period)
            )
        )
    return pl.concat(pieces)


def join_site(*, site: str) -> str:
    """Write the joined frame of one substation."""
    early = pl.concat(
        [pl.read_parquet(f) for f in sorted(base.BASE_DIR.glob(f"{site}_*.parquet"))]
    ).with_columns(period=pl.lit("early"))
    ens = ens_frame(site=site).select(early.columns)
    joined = pl.concat([early, ens]).sort(["power_fcst_init_time", "valid_time"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / f"{site}.parquet"
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    joined.write_parquet(temporary)
    temporary.rename(target)
    nulls = int(ens["level"].null_count())
    return f"{site}: {early.height:,} early + {ens.height:,} ens rows, {nulls:,} ens null levels"


def main() -> None:
    """Join every substation."""
    for site in common.SCORED_SITES:
        print(join_site(site=site), flush=True)


if __name__ == "__main__":
    main()
